"""Read-only verification of prerequisite primary-run evidence.

The experiment runners use these checks before claiming a new test sentinel.
Every accepted predecessor must be complete, hash-consistent, on the current
frozen protocol/vendor reference, and below the memory limit.  This module
does not create, overwrite, or repair evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .runstore import SCHEMA_VERSION, canonical_sha256
from .runtime import (
    INFRASTRUCTURE_OVERRIDE_PREFIX,
    PEAK_RSS_LIMIT_BYTES,
    primary_test_override_log_path,
    sha256_file,
)


class EvidenceGateError(RuntimeError):
    """Raised when prerequisite evidence is missing, incomplete, or corrupt."""


@dataclass(frozen=True, slots=True)
class CompletedPrimaryEvidence:
    """One fully verified primary run loaded from immutable JSON artefacts."""

    experiment: str
    run_id: str
    run_directory: Path
    completed_at_utc: str
    configuration: Mapping[str, object]
    seeds: tuple[int, ...]
    aggregate: Mapping[str, object]
    start_manifest: Mapping[str, object]
    final_manifest: Mapping[str, object]


def evidence_provenance(record: CompletedPrimaryEvidence) -> dict[str, str]:
    """Return the exact predecessor identity hashed into a dependent claim."""

    values = {
        "aggregate_sha256": record.final_manifest.get("aggregate_sha256"),
        "code_commit": record.start_manifest.get("code_commit"),
        "config_sha256": record.final_manifest.get("config_sha256"),
        "experiment": record.experiment,
        "run_id": record.run_id,
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise EvidenceGateError("predecessor evidence lacks a complete identity")
    return {name: str(value) for name, value in values.items()}


def require_prerequisite_lineage(
    dependent: CompletedPrimaryEvidence,
    prerequisites: Sequence[CompletedPrimaryEvidence],
) -> None:
    """Require a dependent run to bind exactly to the supplied predecessors."""

    config_payload = _as_mapping(
        dependent.start_manifest.get("config_payload"), label="config_payload"
    )
    observed = config_payload.get("prerequisite_evidence")
    expected = [evidence_provenance(record) for record in prerequisites]
    if not _json_equal(observed, expected):
        raise EvidenceGateError(
            f"{dependent.run_id} is not bound to the required prerequisite lineage"
        )


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceGateError(f"cannot read evidence JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceGateError(f"evidence JSON must contain an object: {path}")
    return value


def _clean_experiment(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in cleaned
    ):
        raise ValueError("experiment must use only lowercase letters, digits, _ or -")
    return cleaned


def _as_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise EvidenceGateError(f"{label} must be a JSON object")
    return value


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EvidenceGateError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise EvidenceGateError(f"{label} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceGateError(f"{label} must be in UTC")
    return parsed


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``True == 1`` aliasing."""

    try:
        return canonical_sha256(left) == canonical_sha256(right)
    except (TypeError, ValueError):
        return False


def _verify_memory_gate(final: Mapping[str, object], run_directory: Path) -> None:
    memory = _as_mapping(final.get("memory_gate"), label="memory_gate")
    peak = memory.get("peak_rss_bytes")
    limit = memory.get("limit_bytes")
    headroom = memory.get("headroom_bytes")
    passed = memory.get("passed")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (peak, limit, headroom)
    ) or not isinstance(passed, bool):
        raise EvidenceGateError(f"memory gate has invalid field types: {run_directory}")
    assert isinstance(peak, int) and isinstance(limit, int) and isinstance(headroom, int)
    if limit != PEAK_RSS_LIMIT_BYTES:
        raise EvidenceGateError(f"memory gate does not use the frozen limit: {run_directory}")
    if peak < 0 or headroom != limit - peak or passed != (peak < limit):
        raise EvidenceGateError(f"memory gate arithmetic is inconsistent: {run_directory}")
    if not passed:
        raise EvidenceGateError(f"memory gate did not pass: {run_directory}")

    measurement_value = final.get("memory_measurement")
    if measurement_value is None:
        return
    measurement = _as_mapping(
        measurement_value, label="memory_measurement"
    )
    parent = measurement.get("current_process_peak_rss_bytes")
    child = measurement.get("maximum_observed_direct_child_peak_rss_bytes")
    complete = measurement.get("child_peak_observation_complete")
    equivalent = measurement.get("process_tree_equivalent")
    upper = measurement.get("conservative_process_tree_upper_bound_bytes")
    if (
        isinstance(parent, bool)
        or not isinstance(parent, int)
        or parent < 0
        or isinstance(child, bool)
        or not isinstance(child, int)
        or child < 0
        or not isinstance(complete, bool)
        or not isinstance(equivalent, bool)
    ):
        raise EvidenceGateError(
            f"memory measurement has invalid field types: {run_directory}"
        )
    if equivalent:
        if complete is not True or upper != parent + child or peak != upper:
            raise EvidenceGateError(
                f"process-tree memory bound is inconsistent: {run_directory}"
            )
    elif upper is not None or peak != parent:
        raise EvidenceGateError(
            f"current-process memory measurement is inconsistent: {run_directory}"
        )


def _verify_access_claim(
    root: Path,
    experiment: str,
    run_id: str,
    seeds: tuple[int, ...],
    start: Mapping[str, object],
) -> None:
    sentinel_path = root / "run_state" / f"{experiment}.primary-access.json"
    sentinel = _read_object(sentinel_path)
    if sentinel.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceGateError("primary access sentinel schema version is invalid")
    if sentinel.get("seed") != seeds[0]:
        raise EvidenceGateError("primary access sentinel seed is inconsistent")
    sentinel_time = _parse_utc(
        sentinel.get("claimed_at_utc"), label="sentinel claimed_at_utc"
    )
    start_time = _parse_utc(start.get("started_at_utc"), label="started_at_utc")
    if sentinel_time > start_time:
        raise EvidenceGateError("primary access sentinel post-dates the start manifest")

    was_override = start.get("was_override")
    if not isinstance(was_override, bool):
        raise EvidenceGateError("start manifest was_override must be Boolean")
    if not was_override:
        if sentinel.get("run_id") != run_id:
            raise EvidenceGateError("original primary run does not match its sentinel")
        return

    log_path = primary_test_override_log_path(sentinel_path)
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvidenceGateError("override run has no readable override log") from error
    matches: list[Mapping[str, object]] = []
    predecessor_run_id = sentinel.get("run_id")
    sentinel_sha256 = sha256_file(sentinel_path)
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceGateError("override log contains invalid JSON") from error
        if not isinstance(item, dict):
            raise EvidenceGateError("override log entry must be an object")
        if item.get("predecessor_run_id") != predecessor_run_id:
            raise EvidenceGateError("override log predecessor chain is inconsistent")
        if item.get("predecessor_sentinel_sha256") != sentinel_sha256:
            raise EvidenceGateError("override log sentinel hash is inconsistent")
        next_run_id = item.get("run_id")
        if not isinstance(next_run_id, str) or not next_run_id:
            raise EvidenceGateError("override log entry has no run_id")
        predecessor_run_id = next_run_id
        if item.get("run_id") == run_id:
            matches.append(item)
    if len(matches) != 1:
        raise EvidenceGateError("override run needs exactly one matching log entry")
    match = matches[0]
    reason = match.get("reason")
    if not isinstance(reason, str) or not reason.lower().startswith(
        INFRASTRUCTURE_OVERRIDE_PREFIX
    ):
        raise EvidenceGateError("override log reason is not an infrastructure failure")
    if match.get("schema_version") != SCHEMA_VERSION or match.get("seed") != seeds[0]:
        raise EvidenceGateError("override log schema or seed is inconsistent")
    override_time = _parse_utc(
        match.get("claimed_at_utc"), label="override claimed_at_utc"
    )
    if override_time > start_time:
        raise EvidenceGateError("override claim post-dates the start manifest")


def _verify_run(
    root: Path,
    experiment: str,
    run_directory: Path,
) -> CompletedPrimaryEvidence:
    start_path = run_directory / "manifest.start.json"
    aggregate_path = run_directory / "aggregate.json"
    final_path = run_directory / "manifest.final.json"
    for path in (start_path, aggregate_path, final_path):
        if not path.is_file() or path.is_symlink():
            raise EvidenceGateError(f"evidence path is not a regular file: {path}")
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise EvidenceGateError(f"evidence path escapes the repository: {path}") from error
    start = _read_object(start_path)
    aggregate = _read_object(aggregate_path)
    final = _read_object(final_path)
    run_id = run_directory.name

    if start.get("run_id") != run_id or final.get("run_id") != run_id:
        raise EvidenceGateError(f"run-id mismatch in {run_directory}")
    if start.get("experiment") != experiment or final.get("experiment") != experiment:
        raise EvidenceGateError(f"experiment mismatch in {run_directory}")
    if final.get("status") != "completed":
        raise EvidenceGateError(f"run is not completed: {run_directory}")
    if any(
        record.get("schema_version") != SCHEMA_VERSION
        for record in (start, aggregate, final)
    ):
        raise EvidenceGateError(f"evidence schema version mismatch in {run_directory}")
    _verify_memory_gate(final, run_directory)
    if experiment == "e5":
        measurement = _as_mapping(
            final.get("memory_measurement"), label="E5 memory_measurement"
        )
        if measurement.get("process_tree_equivalent") is not True:
            raise EvidenceGateError(
                f"E5 process-tree memory gate is not verified: {run_directory}"
            )

    config_payload = _as_mapping(start.get("config_payload"), label="config_payload")
    config_hash = canonical_sha256(config_payload)
    if start.get("config_sha256") != config_hash or final.get("config_sha256") != config_hash:
        raise EvidenceGateError(f"configuration hash mismatch in {run_directory}")
    base_run_id = f"{experiment}-{config_hash[:16]}"
    if run_id != base_run_id and re.fullmatch(
        re.escape(base_run_id) + r"-attempt[0-9]{2,}", run_id
    ) is None:
        raise EvidenceGateError(f"hash-stamped run id mismatch in {run_directory}")

    current_protocol_hash = sha256_file(root / "PREREGISTRATION.md")
    current_vendor_hash = sha256_file(root / "vendor" / "manifest.json")
    if (
        start.get("protocol_sha256") != current_protocol_hash
        or final.get("protocol_sha256") != current_protocol_hash
        or config_payload.get("protocol_sha256") != current_protocol_hash
    ):
        raise EvidenceGateError(f"protocol hash mismatch in {run_directory}")
    if (
        start.get("vendor_manifest_sha256") != current_vendor_hash
        or config_payload.get("vendor_manifest_sha256") != current_vendor_hash
    ):
        raise EvidenceGateError(f"vendor-manifest hash mismatch in {run_directory}")
    if final.get("aggregate_sha256") != sha256_file(aggregate_path):
        raise EvidenceGateError(f"aggregate hash mismatch in {run_directory}")

    configuration = _as_mapping(
        config_payload.get("configuration"), label="configuration"
    )
    aggregate_configuration = _as_mapping(
        aggregate.get("configuration"), label="aggregate configuration"
    )
    if not _json_equal(configuration, aggregate_configuration):
        raise EvidenceGateError(f"aggregate/start configuration mismatch in {run_directory}")
    seeds_value = config_payload.get("seeds")
    if (
        not isinstance(seeds_value, list)
        or not seeds_value
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFFFFFF
            for seed in seeds_value
        )
        or len(set(seeds_value)) != len(seeds_value)
    ):
        raise EvidenceGateError(f"missing seed vector in {run_directory}")
    seeds = tuple(seeds_value)
    if not _json_equal(start.get("seeds"), list(seeds)):
        raise EvidenceGateError(f"start-manifest seed mismatch in {run_directory}")
    if config_payload.get("experiment") != experiment:
        raise EvidenceGateError(f"config experiment mismatch in {run_directory}")
    if config_payload.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceGateError(f"config schema version mismatch in {run_directory}")
    prerequisite_records = config_payload.get("prerequisite_evidence", [])
    if not isinstance(prerequisite_records, list):
        raise EvidenceGateError(f"prerequisite evidence must be a list in {run_directory}")
    for predecessor in prerequisite_records:
        if not isinstance(predecessor, dict) or set(predecessor) != {
            "aggregate_sha256",
            "code_commit",
            "config_sha256",
            "experiment",
            "run_id",
        }:
            raise EvidenceGateError(f"invalid prerequisite identity in {run_directory}")
        if any(not isinstance(value, str) or not value for value in predecessor.values()):
            raise EvidenceGateError(f"invalid prerequisite fields in {run_directory}")

    code_commit = start.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in code_commit
    ):
        raise EvidenceGateError(f"invalid code commit in {run_directory}")
    completed_at = final.get("completed_at_utc")
    completed_time = _parse_utc(completed_at, label="completed_at_utc")
    started_time = _parse_utc(start.get("started_at_utc"), label="started_at_utc")
    if completed_time < started_time:
        raise EvidenceGateError(f"completion precedes start in {run_directory}")
    command = start.get("command")
    environment = start.get("environment")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise EvidenceGateError(f"missing command provenance in {run_directory}")
    if not isinstance(environment, dict) or not environment:
        raise EvidenceGateError(f"missing environment provenance in {run_directory}")
    peak_start = start.get("peak_rss_at_start_bytes")
    if isinstance(peak_start, bool) or not isinstance(peak_start, int) or peak_start < 0:
        raise EvidenceGateError(f"invalid start peak RSS in {run_directory}")
    if "start_manifest_sha256" in final and final.get(
        "start_manifest_sha256"
    ) != sha256_file(start_path):
        raise EvidenceGateError(f"start manifest hash mismatch in {run_directory}")
    repository_gate = final.get("repository_integrity_gate")
    if repository_gate is not None:
        repository_mapping = _as_mapping(
            repository_gate, label="repository_integrity_gate"
        )
        if repository_mapping.get("passed") is not True:
            raise EvidenceGateError(f"repository integrity gate failed in {run_directory}")

    _verify_access_claim(root, experiment, run_id, seeds, start)

    return CompletedPrimaryEvidence(
        experiment=experiment,
        run_id=run_id,
        run_directory=run_directory,
        completed_at_utc=completed_at,
        configuration=configuration,
        seeds=seeds,
        aggregate=aggregate,
        start_manifest=start,
        final_manifest=final,
    )


def completed_primary_runs(
    repository: str | Path,
    experiment: str,
) -> tuple[CompletedPrimaryEvidence, ...]:
    """Load and verify every completed run for one experiment."""

    root = Path(repository).resolve()
    clean = _clean_experiment(experiment)
    experiment_root = root / "runs" / clean
    if not experiment_root.is_dir():
        return ()
    records: list[CompletedPrimaryEvidence] = []
    for run_directory in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        final_path = run_directory / "manifest.final.json"
        if not final_path.is_file():
            continue
        final_preview = _read_object(final_path)
        if final_preview.get("status") == "failed":
            # Failed attempts remain durable reportable evidence, but cannot
            # satisfy a prerequisite gate.
            continue
        records.append(_verify_run(root, clean, run_directory))
    records.sort(key=lambda record: (record.completed_at_utc, record.run_id))
    return tuple(records)


def require_completed_primary(
    repository: str | Path,
    experiment: str,
    *,
    required_configuration: Mapping[str, object] | None = None,
    required_seeds: Sequence[int] | None = None,
) -> CompletedPrimaryEvidence:
    """Return the newest verified run matching the exact requested protocol."""

    records = completed_primary_runs(repository, experiment)
    expected_seeds = (
        None if required_seeds is None else tuple(int(seed) for seed in required_seeds)
    )
    matching = [
        record
        for record in records
        if (
            required_configuration is None
            or _json_equal(record.configuration, required_configuration)
        )
        and (expected_seeds is None or record.seeds == expected_seeds)
    ]
    if not matching:
        detail = ""
        if records:
            detail = f"; {len(records)} completed run(s) exist but none match"
        raise EvidenceGateError(
            f"no completed {experiment} prerequisite with the required frozen configuration{detail}"
        )
    return matching[-1]


def require_e1_passed(repository: str | Path) -> CompletedPrimaryEvidence:
    """Require the latest completed E1 evidence to record an overall pass."""

    records = completed_primary_runs(repository, "e1")
    if not records:
        raise EvidenceGateError("E1 must be completed and pass before later experiments")
    record = records[-1]
    expected_seeds = tuple(range(1_000, 1_010))
    expected_configuration = {
        "day": "C",
        "private_field_access_configured": True,
        "private_output_policy": "aggregate metrics only; no stdout or rows retained",
        "seeds": list(expected_seeds),
    }
    if record.seeds != expected_seeds or not _json_equal(
        record.configuration, expected_configuration
    ):
        raise EvidenceGateError("latest E1 run does not use the frozen E1 configuration")
    gate = _as_mapping(record.aggregate.get("gate"), label="E1 aggregate gate")
    if gate.get("overall_status") != "pass":
        raise EvidenceGateError(
            f"latest completed E1 gate status is {gate.get('overall_status')!r}, not 'pass'"
        )
    if (
        _as_mapping(gate.get("synthetic"), label="E1 synthetic gate").get("status")
        != "pass"
        or _as_mapping(gate.get("field"), label="E1 field gate").get("status")
        != "pass"
        or not isinstance(record.aggregate.get("field"), dict)
    ):
        raise EvidenceGateError("latest E1 run lacks coherent synthetic/field pass evidence")
    return record


__all__ = [
    "CompletedPrimaryEvidence",
    "EvidenceGateError",
    "completed_primary_runs",
    "evidence_provenance",
    "require_completed_primary",
    "require_e1_passed",
    "require_prerequisite_lineage",
]
