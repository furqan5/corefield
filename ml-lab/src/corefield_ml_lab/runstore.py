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
import subprocess
import time
from typing import Callable, Mapping, Sequence, TypeVar

from .runtime import (
    INFRASTRUCTURE_OVERRIDE_PREFIX,
    PEAK_RSS_LIMIT_BYTES,
    capture_environment,
    claim_primary_test_access,
    evaluate_peak_rss_gate,
    process_peak_rss_bytes,
    sha256_file,
)


SCHEMA_VERSION = 1
_ACTIVE_PRIMARY_RUN: ContextVar[PrimaryRun | None]  # assigned after class definition
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


def git_head(repository: Path) -> str:
    """Resolve the current commit without mutating Git state."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError(f"unexpected Git HEAD value: {value!r}")
    return value


def git_status_porcelain(repository: Path) -> tuple[str, ...]:
    """Return every tracked or untracked worktree change without mutation."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def require_clean_worktree(repository: Path) -> None:
    """Refuse a primary claim unless its complete code/evidence state is committed."""

    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
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
    require_clean_worktree(root)
    # Resolve every subprocess-backed provenance field before the access
    # claim.  E2--E5 then run in one Python process, so their process tree is
    # exactly the measured process (native numerical threads share its RSS).
    environment = capture_environment()
    code_commit = git_head(root)
    peak_start = process_peak_rss_bytes()

    run_id = base_run_id
    base_run_directory = root / "runs" / clean_experiment / base_run_id
    if base_run_directory.exists():
        if not override:
            raise FileExistsError(
                f"primary run directory already exists without an override: {base_run_directory}"
            )
        final_path = base_run_directory / "manifest.final.json"
        if final_path.is_file():
            try:
                prior_final = json.loads(final_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "cannot validate the prior same-configuration final manifest"
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
        attempt = 2
        while True:
            candidate = f"{base_run_id}-attempt{attempt:02d}"
            if not (root / "runs" / clean_experiment / candidate).exists():
                run_id = candidate
                break
            attempt += 1
    claim = claim_primary_test_access(
        sentinel_path,
        run_id=run_id,
        seed=seed_tuple[0],
        override=override,
        override_reason=override_reason,
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
    peak = process_peak_rss_bytes()
    memory = evaluate_peak_rss_gate(peak, limit_bytes=PEAK_RSS_LIMIT_BYTES)
    repository_integrity = _repository_integrity_at_finish(run)
    effective_status = (
        "completed"
        if requested_status == "completed"
        and memory.passed
        and repository_integrity["passed"] is True
        else "failed"
    )
    final_payload: dict[str, object] = {
        "aggregate_sha256": sha256_file(aggregate_path),
        "completed_at_utc": _utc_now(),
        "config_sha256": run.config_sha256,
        "experiment": run.experiment,
        "memory_gate": asdict(memory),
        "memory_measurement": {
            "metric": "peak resident working set [bytes]",
            "process_tree_equivalent": run.experiment != "e1",
            "scope": "current Python process",
            "scope_note": (
                "E2--E5 and confirmation runners spawn no child workers; native numerical threads share this process. "
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
    "require_clean_worktree",
    "write_json_exclusive",
]
