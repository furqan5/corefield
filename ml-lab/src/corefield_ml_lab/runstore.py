"""Write-once manifests for primary experiment access and aggregate results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Mapping, Sequence, TypeVar

from .runtime import (
    INFRASTRUCTURE_OVERRIDE_PREFIX,
    PEAK_RSS_LIMIT_BYTES,
    capture_environment,
    claim_primary_test_access,
    completed_child_peak_rss_bytes,
    evaluate_peak_rss_gate,
    process_peak_rss_bytes,
    primary_test_override_log_path,
    sha256_file,
)


SCHEMA_VERSION = 1
_ACTIVE_PRIMARY_RUN: ContextVar[PrimaryRun | None]  # assigned after class definition
_OBSERVED_CHILD_PEAK_RSS_BYTES: ContextVar[int] = ContextVar(
    "corefield_ml_lab_observed_child_peak_rss_bytes", default=0
)
_CHILD_PEAK_OBSERVATION_COMPLETE: ContextVar[bool] = ContextVar(
    "corefield_ml_lab_child_peak_observation_complete", default=True
)
_ReturnT = TypeVar("_ReturnT")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: object) -> bytes:
    """Canonical UTF-8 representation used for all content hashes."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    """Lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json_exclusive(path: Path, payload: object) -> None:
    """Create one JSON artefact and refuse any overwrite."""

    # Complete serialization before claiming the write-once path.  This
    # prevents a late non-serializable value (or NaN) from leaving behind a
    # truncated file that blocks honest failure finalization.
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _run_git(repository: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one read-only Git command and retain a conservative child-RSS peak."""

    process = subprocess.Popen(
        ["git", *arguments],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    stdout, stderr = process.communicate()
    try:
        child_peak = completed_child_peak_rss_bytes(process)
    except (OSError, RuntimeError, TypeError, ValueError):
        _CHILD_PEAK_OBSERVATION_COMPLETE.set(False)
    else:
        _OBSERVED_CHILD_PEAK_RSS_BYTES.set(
            max(_OBSERVED_CHILD_PEAK_RSS_BYTES.get(), child_peak)
        )
    completed = subprocess.CompletedProcess(
        process.args,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    completed.check_returncode()
    return completed


def git_head(repository: Path) -> str:
    """Resolve the current commit without mutating Git state."""

    completed = _run_git(repository, ["rev-parse", "HEAD"])
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError(f"unexpected Git HEAD value: {value!r}")
    return value


def git_status_porcelain(repository: Path) -> tuple[str, ...]:
    """Return every tracked or untracked worktree change without mutation."""

    completed = _run_git(
        repository,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def require_clean_worktree(repository: Path) -> None:
    """Refuse a primary claim unless its complete code/evidence state is committed."""

    completed = _run_git(repository, ["rev-parse", "--show-toplevel"])
    top_level = Path(completed.stdout.strip()).resolve()
    if os.path.normcase(str(top_level)) != os.path.normcase(str(repository.resolve())):
        raise RuntimeError(
            f"primary repository must be the Git top level: {repository} != {top_level}"
        )
    changes = git_status_porcelain(repository)
    if changes:
        preview = "; ".join(changes[:5])
        if len(changes) > 5:
            preview += f"; ... ({len(changes)} paths total)"
        raise RuntimeError(
            "primary runs require a clean committed Git worktree; unresolved paths: "
            + preview
        )


@dataclass(frozen=True, slots=True)
class PrimaryRun:
    """Open run handle; its start manifest and sentinel are already durable."""

    experiment: str
    repository: Path
    run_id: str
    run_directory: Path
    config_sha256: str
    protocol_sha256: str
    code_commit: str
    seeds: tuple[int, ...]
    start_monotonic_s: float
    peak_rss_at_start_bytes: int
    was_override: bool


_ACTIVE_PRIMARY_RUN = ContextVar("corefield_ml_lab_active_primary_run", default=None)


def require_active_primary_run(run: PrimaryRun, *, experiment: str) -> None:
    """Require the exact post-sentinel run handle in the current execution context."""

    if _ACTIVE_PRIMARY_RUN.get() is not run or run.experiment != experiment:
        raise RuntimeError(
            f"{experiment} primary data require the active post-sentinel run handle"
        )


def record_primary_failures(
    function: Callable[..., _ReturnT],
) -> Callable[..., _ReturnT]:
    """Finalize an already-claimed run as failed before re-raising its error.

    Durable failure evidence records the exception type and a hash of the
    console-visible message, but not the message itself; this prevents a
    private path or row value from entering Git history.
    """

    @wraps(function)
    def wrapped(*args: object, **kwargs: object) -> _ReturnT:
        _ACTIVE_PRIMARY_RUN.set(None)
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            run = _ACTIVE_PRIMARY_RUN.get()
            if run is not None and not (
                run.run_directory / "manifest.final.json"
            ).exists():
                message_hash = hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest()
                failure_payload = {
                    "failure": {
                        "exception_message_persisted": False,
                        "exception_message_sha256": message_hash,
                        "exception_type": type(error).__name__,
                    },
                    "result_status": "infrastructure_failure",
                    "schema_version": SCHEMA_VERSION,
                }
                try:
                    finish_primary_run(run, failure_payload, status="failed")
                except BaseException as finalization_error:
                    # Never replace the primary exception with a secondary
                    # evidence-write error.  Preserve a separate terminal
                    # marker if aggregate/final publication was interrupted.
                    emergency_payload = {
                        "completed_at_utc": _utc_now(),
                        "experiment": run.experiment,
                        "failure": failure_payload["failure"],
                        "finalization_exception_type": type(
                            finalization_error
                        ).__name__,
                        "run_id": run.run_id,
                        "schema_version": SCHEMA_VERSION,
                        "status": "failed",
                    }
                    try:
                        write_json_exclusive(
                            run.run_directory / "manifest.failure.json",
                            emergency_payload,
                        )
                    except BaseException:
                        pass
                    _ACTIVE_PRIMARY_RUN.set(None)
            raise

    return wrapped


def begin_primary_run(
    repository: str | os.PathLike[str],
    *,
    experiment: str,
    configuration: Mapping[str, object],
    seeds: Sequence[int],
    command: Sequence[str],
    prerequisites: Sequence[Mapping[str, object]] = (),
    override: bool = False,
    override_reason: str | None = None,
) -> PrimaryRun:
    """Claim test access and create a hash-stamped start manifest.

    Call this immediately before any primary test truth is generated or read.
    The sentinel is one per experiment.  Its ``run_id`` contains the complete
    canonical configuration hash, so the access claim itself is hash-stamped.
    """

    root = Path(repository).resolve()
    clean_experiment = experiment.strip().lower()
    if not clean_experiment or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in clean_experiment
    ):
        raise ValueError("experiment must use only lowercase letters, digits, _ or -")
    raw_seeds = tuple(seeds)
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) <= 0xFFFFFFFF
        for seed in raw_seeds
    ):
        raise ValueError("every seed must be an integer in the range 0..2**32-1")
    seed_tuple = tuple(int(seed) for seed in raw_seeds)
    if not seed_tuple or len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be non-empty and unique")
    if not command or any(not str(part).strip() for part in command):
        raise ValueError("command must contain non-empty arguments")

    preregistration = root / "PREREGISTRATION.md"
    vendor_manifest = root / "vendor" / "manifest.json"
    protocol_hash = sha256_file(preregistration)
    vendor_hash = sha256_file(vendor_manifest)
    config_payload: dict[str, object] = {
        "configuration": dict(configuration),
        "experiment": clean_experiment,
        "protocol_sha256": protocol_hash,
        "schema_version": SCHEMA_VERSION,
        "seeds": seed_tuple,
        "vendor_manifest_sha256": vendor_hash,
    }
    if prerequisites:
        config_payload["prerequisite_evidence"] = [
            dict(record) for record in prerequisites
        ]
    config_hash = canonical_sha256(config_payload)
    base_run_id = f"{clean_experiment}-{config_hash[:16]}"
    sentinel_path = root / "run_state" / f"{clean_experiment}.primary-access.json"

    # Preserve the original write-once error when a non-override access was
    # already claimed.  For a fresh claim (or an audited override), require
    # that the exact code and prior evidence state are committed first.
    if sentinel_path.exists() and not override:
        claim_primary_test_access(
            sentinel_path,
            run_id=base_run_id,
            seed=seed_tuple[0],
            override=False,
            override_reason=override_reason,
        )
        raise AssertionError("existing primary sentinel unexpectedly allowed access")
    if override and not sentinel_path.exists():
        raise RuntimeError(
            "primary-test override requires an existing access sentinel; "
            "refusing an unlogged first access"
        )
    if override:
        reason = "" if override_reason is None else override_reason.strip()
        if not reason.lower().startswith(INFRASTRUCTURE_OVERRIDE_PREFIX):
            raise ValueError(
                "override_reason must begin with "
                f"{INFRASTRUCTURE_OVERRIDE_PREFIX!r}"
            )
    _OBSERVED_CHILD_PEAK_RSS_BYTES.set(0)
    _CHILD_PEAK_OBSERVATION_COMPLETE.set(True)
    require_clean_worktree(root)
    # Resolve every subprocess-backed provenance field before the access
    # claim.  E2--E5 then run in one Python process, so their process tree is
    # exactly the measured process (native numerical threads share its RSS).
    environment = capture_environment()
    code_commit = git_head(root)
    peak_start = process_peak_rss_bytes()

    run_id = base_run_id
    override_predecessor_run_id: str | None = None
    experiment_directory = root / "runs" / clean_experiment
    base_run_directory = experiment_directory / base_run_id
    if not override and base_run_directory.exists():
        raise FileExistsError(
            f"primary run directory already exists without an override: {base_run_directory}"
        )
    if override:
        try:
            sentinel_payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("cannot validate the existing access sentinel") from error
        original_run_id = sentinel_payload.get("run_id")
        if original_run_id != base_run_id:
            raise RuntimeError(
                "primary-test override must use the exact configuration, seeds, "
                "protocol, vendor evidence, and prerequisite lineage of the original claim"
            )

        claimed_run_ids = [base_run_id]
        override_log = primary_test_override_log_path(sentinel_path)
        if override_log.exists():
            try:
                log_lines = override_log.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                raise RuntimeError("cannot validate the existing override log") from error
            predecessor = base_run_id
            for attempt_number, line in enumerate(log_lines, start=2):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError("existing override log is not valid JSONL") from error
                expected_run_id = f"{base_run_id}-attempt{attempt_number:02d}"
                if not isinstance(entry, dict) or (
                    entry.get("predecessor_run_id") != predecessor
                    or entry.get("run_id") != expected_run_id
                ):
                    raise RuntimeError(
                        "existing override log is not a sequential same-configuration chain"
                    )
                predecessor = expected_run_id
                claimed_run_ids.append(expected_run_id)

        if experiment_directory.is_dir():
            exact_pattern = re.compile(
                rf"^{re.escape(base_run_id)}(?:-attempt[0-9]{{2,}})?$"
            )
            unclaimed_directories = sorted(
                path.name
                for path in experiment_directory.iterdir()
                if path.is_dir()
                and exact_pattern.fullmatch(path.name)
                and path.name not in claimed_run_ids
            )
            if unclaimed_directories:
                raise RuntimeError(
                    "same-configuration run directories lack matching access claims: "
                    + ", ".join(unclaimed_directories)
                )

        for claimed_run_id in claimed_run_ids:
            final_path = experiment_directory / claimed_run_id / "manifest.final.json"
            if not final_path.exists():
                continue
            try:
                prior_final = json.loads(final_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "cannot validate a prior same-configuration final manifest"
                ) from error
            prior_memory = prior_final.get("memory_gate")
            prior_passed = (
                isinstance(prior_memory, dict)
                and prior_memory.get("passed") is True
            )
            if prior_final.get("status") == "completed" and prior_passed:
                raise RuntimeError(
                    "a completed, memory-passing run already exists for this exact "
                    "configuration; an override may not repeat it"
                )

        override_predecessor_run_id = claimed_run_ids[-1]
        run_id = f"{base_run_id}-attempt{len(claimed_run_ids) + 1:02d}"
    claim = claim_primary_test_access(
        sentinel_path,
        run_id=run_id,
        seed=seed_tuple[0],
        override=override,
        override_reason=override_reason,
        override_predecessor_run_id=override_predecessor_run_id,
    )
    run_directory = root / "runs" / clean_experiment / run_id
    start_payload = {
        "code_commit": code_commit,
        "command": [str(part) for part in command],
        "config_payload": config_payload,
        "config_sha256": config_hash,
        "environment": environment.to_dict(),
        "experiment": clean_experiment,
        "peak_rss_at_start_bytes": peak_start,
        "protocol_sha256": protocol_hash,
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "seeds": list(seed_tuple),
        "started_at_utc": _utc_now(),
        "vendor_manifest_sha256": vendor_hash,
        "was_override": claim.was_override,
    }
    primary_run = PrimaryRun(
        experiment=clean_experiment,
        repository=root,
        run_id=run_id,
        run_directory=run_directory,
        config_sha256=config_hash,
        protocol_sha256=protocol_hash,
        code_commit=code_commit,
        seeds=seed_tuple,
        start_monotonic_s=time.perf_counter(),
        peak_rss_at_start_bytes=peak_start,
        was_override=claim.was_override,
    )
    # Set the active handle before publishing the start manifest.  If that
    # final pre-run write fails, the decorator can still attribute and close
    # the already-claimed access rather than leaving an anonymous sentinel.
    _ACTIVE_PRIMARY_RUN.set(primary_run)
    write_json_exclusive(run_directory / "manifest.start.json", start_payload)
    return primary_run


def _repository_integrity_at_finish(run: PrimaryRun) -> dict[str, object]:
    """Check that only this run's write-once evidence changed after its claim."""

    issues: list[str] = []
    try:
        observed_head = git_head(run.repository)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        observed_head = None
        issues.append(f"cannot resolve final Git HEAD: {type(error).__name__}")
    if observed_head is not None and observed_head != run.code_commit:
        issues.append("Git HEAD changed during the primary run")

    try:
        changes = git_status_porcelain(run.repository)
    except (OSError, subprocess.SubprocessError) as error:
        changes = ()
        issues.append(f"cannot resolve final Git status: {type(error).__name__}")

    allowed_run_prefix = f"runs/{run.experiment}/{run.run_id}/"
    allowed_state = {
        f"run_state/{run.experiment}.primary-access.json",
        f"run_state/{run.experiment}.primary-access.json.overrides.jsonl",
    }
    unexpected: list[str] = []
    for entry in changes:
        raw_path = entry[3:].replace("\\", "/")
        paths = raw_path.split(" -> ") if " -> " in raw_path else [raw_path]
        if any(
            not (path.startswith(allowed_run_prefix) or path in allowed_state)
            for path in paths
        ):
            unexpected.append(entry)
    if unexpected:
        issues.append(
            "unexpected worktree changes during primary run: "
            + "; ".join(unexpected[:5])
        )
    return {
        "code_commit_unchanged": observed_head == run.code_commit,
        "passed": not issues,
        "unexpected_change_count": len(unexpected),
        "issues": issues,
    }


def finish_primary_run(
    run: PrimaryRun,
    aggregate_payload: Mapping[str, object],
    *,
    status: str = "completed",
) -> Mapping[str, object]:
    """Write aggregate and final manifests, each exactly once."""

    requested_status = status.strip().lower()
    if requested_status not in {"completed", "failed"}:
        raise ValueError("status must be 'completed' or 'failed'")
    aggregate_path = run.run_directory / "aggregate.json"
    write_json_exclusive(aggregate_path, dict(aggregate_payload))
    repository_integrity = _repository_integrity_at_finish(run)
    parent_peak = process_peak_rss_bytes()
    child_peak = _OBSERVED_CHILD_PEAK_RSS_BYTES.get()
    child_observation_complete = _CHILD_PEAK_OBSERVATION_COMPLETE.get()
    process_tree_upper_bound_available = bool(
        run.experiment != "e1" and child_observation_complete
    )
    measured_peak = (
        parent_peak + child_peak
        if process_tree_upper_bound_available
        else parent_peak
    )
    memory = evaluate_peak_rss_gate(
        measured_peak, limit_bytes=PEAK_RSS_LIMIT_BYTES
    )
    effective_status = (
        "completed"
        if requested_status == "completed"
        and memory.passed
        and repository_integrity["passed"] is True
        and (run.experiment == "e1" or process_tree_upper_bound_available)
        else "failed"
    )
    final_payload: dict[str, object] = {
        "aggregate_sha256": sha256_file(aggregate_path),
        "completed_at_utc": _utc_now(),
        "config_sha256": run.config_sha256,
        "experiment": run.experiment,
        "memory_gate": asdict(memory),
        "memory_measurement": {
            "child_peak_observation_complete": child_observation_complete,
            "conservative_process_tree_upper_bound_bytes": (
                measured_peak if process_tree_upper_bound_available else None
            ),
            "current_process_peak_rss_bytes": parent_peak,
            "maximum_observed_direct_child_peak_rss_bytes": child_peak,
            "metric": "peak resident working set [bytes]",
            "process_tree_equivalent": process_tree_upper_bound_available,
            "scope": (
                "conservative current-process peak plus maximum direct-child peak"
                if process_tree_upper_bound_available
                else "current Python process only"
            ),
            "scope_note": (
                "E2--E5 and confirmation runners spawn no experiment workers; native numerical threads share this process. "
                "Read-only Git provenance children are measured from retained OS peak counters and added conservatively. "
                "E1 can invoke a private child adapter, so its child-process peak is not captured by this field."
            ),
        },
        "protocol_sha256": run.protocol_sha256,
        "run_id": run.run_id,
        "schema_version": SCHEMA_VERSION,
        "start_manifest_sha256": sha256_file(
            run.run_directory / "manifest.start.json"
        ),
        "repository_integrity_gate": repository_integrity,
        "status": effective_status,
        "wall_time_s": time.perf_counter() - run.start_monotonic_s,
    }
    write_json_exclusive(run.run_directory / "manifest.final.json", final_payload)
    if _ACTIVE_PRIMARY_RUN.get() is run:
        _ACTIVE_PRIMARY_RUN.set(None)
    if not memory.passed:
        raise MemoryError(
            f"peak RSS {memory.peak_rss_bytes} bytes is not below "
            f"{memory.limit_bytes} bytes"
        )
    if repository_integrity["passed"] is not True:
        raise RuntimeError("repository integrity gate failed during the primary run")
    if run.experiment != "e1" and not process_tree_upper_bound_available:
        raise RuntimeError("process-tree peak RSS could not be verified")
    return final_payload


__all__ = [
    "PrimaryRun",
    "begin_primary_run",
    "canonical_json_bytes",
    "canonical_sha256",
    "finish_primary_run",
    "git_head",
    "git_status_porcelain",
    "record_primary_failures",
    "require_active_primary_run",
    "require_clean_worktree",
    "write_json_exclusive",
]
