"""Read-only verification of prerequisite primary-run evidence.

The experiment runners use these checks before claiming a new test sentinel.
Every accepted predecessor must be complete, hash-consistent, on the current
frozen protocol/vendor reference, and below the memory limit.  This module
does not create, overwrite, or repair evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .runstore import canonical_sha256
from .runtime import sha256_file


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


def _verify_run(
    root: Path,
    experiment: str,
    run_directory: Path,
) -> CompletedPrimaryEvidence:
    start_path = run_directory / "manifest.start.json"
    aggregate_path = run_directory / "aggregate.json"
    final_path = run_directory / "manifest.final.json"
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

    memory_gate = _as_mapping(final.get("memory_gate"), label="memory_gate")
    if memory_gate.get("passed") is not True:
        raise EvidenceGateError(f"memory gate did not pass: {run_directory}")

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
    seeds_value = config_payload.get("seeds")
    if not isinstance(seeds_value, list) or not seeds_value:
        raise EvidenceGateError(f"missing seed vector in {run_directory}")
    try:
        seeds = tuple(int(seed) for seed in seeds_value)
    except (TypeError, ValueError) as error:
        raise EvidenceGateError(f"invalid seed vector in {run_directory}") from error
    if list(start.get("seeds", [])) != list(seeds):
        raise EvidenceGateError(f"start-manifest seed mismatch in {run_directory}")
    if config_payload.get("experiment") != experiment:
        raise EvidenceGateError(f"config experiment mismatch in {run_directory}")

    code_commit = start.get("code_commit")
    if not isinstance(code_commit, str) or len(code_commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in code_commit
    ):
        raise EvidenceGateError(f"invalid code commit in {run_directory}")
    completed_at = final.get("completed_at_utc")
    if not isinstance(completed_at, str) or not completed_at:
        raise EvidenceGateError(f"missing completion time in {run_directory}")

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
            or dict(record.configuration) == dict(required_configuration)
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
    gate = _as_mapping(record.aggregate.get("gate"), label="E1 aggregate gate")
    if gate.get("overall_status") != "pass":
        raise EvidenceGateError(
            f"latest completed E1 gate status is {gate.get('overall_status')!r}, not 'pass'"
        )
    return record


__all__ = [
    "CompletedPrimaryEvidence",
    "EvidenceGateError",
    "completed_primary_runs",
    "require_completed_primary",
    "require_e1_passed",
]
