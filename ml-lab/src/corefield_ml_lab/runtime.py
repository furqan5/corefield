"""Runtime safeguards for reproducible, CPU-only experiments.

This module deliberately has no third-party dependencies.  Optional numerical
backends are imported only when they are already installed.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


LOGGER = logging.getLogger(__name__)

# The project limit is decimal bytes, not 2 GiB.
PEAK_RSS_LIMIT_BYTES = 2_000_000_000
DEFAULT_HASH_CHUNK_SIZE_BYTES = 1_048_576
INFRASTRUCTURE_OVERRIDE_PREFIX = "infrastructure failure:"

CPU_ONLY_ENVIRONMENT: Mapping[str, str] = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "JAX_PLATFORMS": "cpu",
    "JAX_PLATFORM_NAME": "cpu",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

DEFAULT_VERSION_PACKAGES: tuple[str, ...] = (
    "corefield",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "pytest",
    "torch",
    "jax",
    "jaxlib",
    "scikit-learn",
)


class CpuOnlyViolationError(RuntimeError):
    """Raised when an installed Torch runtime can still see a CUDA device."""


class PrimaryTestAlreadyClaimedError(RuntimeError):
    """Raised when the primary test is claimed more than once without override."""


class MemoryLimitExceededError(RuntimeError):
    """Raised when process peak resident memory does not satisfy the strict gate."""


@dataclass(frozen=True, slots=True)
class TorchDeviceStatus:
    """Observed Torch/CUDA state after CPU-only environment enforcement."""

    installed: bool
    version: str | None
    cuda_available: bool | None
    visible_cuda_device_count: int | None
    cpu_only: bool


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """Machine-readable record of deterministic seeds applied to one run."""

    run_id: str
    seed: int
    recorded_at_utc: str
    python_random_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    torch_deterministic_algorithms: bool
    python_hash_seed: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PackageVersion:
    """Installed distribution version, or ``None`` when absent."""

    name: str
    version: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    """Interpreter, platform, CPU guard, and package-version snapshot."""

    captured_at_utc: str
    python_version: str
    python_executable: str
    python_implementation: str
    platform: str
    machine: str
    architecture_bits: int
    logical_cpu_count: int | None
    cpu_environment: tuple[tuple[str, str | None], ...]
    packages: tuple[PackageVersion, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""

        payload = asdict(self)
        payload["cpu_environment"] = dict(self.cpu_environment)
        return payload


@dataclass(frozen=True, slots=True)
class PrimaryTestClaim:
    """Result of claiming first or explicitly overridden primary-test access."""

    sentinel_path: Path
    run_id: str
    seed: int
    claimed_at_utc: str
    was_override: bool
    override_log_path: Path | None


@dataclass(frozen=True, slots=True)
class MemoryGateResult:
    """Strict process peak-RSS gate result; all quantities are bytes."""

    peak_rss_bytes: int
    limit_bytes: int
    passed: bool
    headroom_bytes: int


def enforce_cpu_only_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Disable CUDA and request CPU execution from JAX-compatible backends.

    The variables must be set before importing Torch or JAX.  Existing values
    are overwritten intentionally.  The returned mapping is the exact state
    applied; no hardware availability is inferred from it.
    """

    target = os.environ if environment is None else environment
    applied = dict(CPU_ONLY_ENVIRONMENT)
    target.update(applied)
    return applied


def _optional_import(module_name: str) -> Any | None:
    """Import an optional backend only when it can be found."""

    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


def inspect_torch_device(torch_module: object | None = None) -> TorchDeviceStatus:
    """Inspect CUDA visibility when Torch is installed, without requiring it.

    ``torch_module`` is accepted for testability.  When omitted, Torch is
    discovered and imported lazily.  ``cpu_only`` is true only when Torch is
    absent or reports both no available CUDA runtime and zero visible devices.
    """

    module = _optional_import("torch") if torch_module is None else torch_module
    if module is None:
        return TorchDeviceStatus(
            installed=False,
            version=None,
            cuda_available=None,
            visible_cuda_device_count=None,
            cpu_only=True,
        )

    cuda = getattr(module, "cuda", None)
    if cuda is None:
        cuda_available = False
        device_count = 0
    else:
        cuda_available = bool(cuda.is_available())
        device_count = int(cuda.device_count())

    return TorchDeviceStatus(
        installed=True,
        version=str(getattr(module, "__version__", "unknown")),
        cuda_available=cuda_available,
        visible_cuda_device_count=device_count,
        cpu_only=not cuda_available and device_count == 0,
    )


def require_torch_cpu_only(torch_module: object | None = None) -> TorchDeviceStatus:
    """Return Torch status or raise if any CUDA device remains visible."""

    status = inspect_torch_device(torch_module)
    if not status.cpu_only:
        raise CpuOnlyViolationError(
            "CPU-only execution required, but Torch reports "
            f"cuda_available={status.cuda_available} and "
            f"visible_cuda_device_count={status.visible_cuda_device_count}."
        )
    return status


def _normalise_utc(now_utc: datetime | None) -> str:
    instant = datetime.now(timezone.utc) if now_utc is None else now_utc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be in the inclusive range 0..2**32-1")


def set_deterministic_seed(
    seed: int,
    *,
    run_id: str,
    record_path: str | os.PathLike[str] | None = None,
    now_utc: datetime | None = None,
) -> SeedRecord:
    """Seed Python and installed numerical backends, and optionally record it.

    The accepted seed range is the unsigned 32-bit range shared by NumPy's
    legacy global generator and the other supported backends.  Setting
    ``PYTHONHASHSEED`` affects child interpreters; the current interpreter's
    hash randomisation is fixed only at process start.
    """

    _validate_seed(seed)
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    enforce_cpu_only_environment()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    numpy_module = _optional_import("numpy")
    if numpy_module is not None:
        numpy_module.random.seed(seed)
        numpy_seeded = True

    torch_seeded = False
    torch_deterministic = False
    torch_module = _optional_import("torch")
    if torch_module is not None:
        require_torch_cpu_only(torch_module)
        torch_module.manual_seed(seed)
        torch_seeded = True
        deterministic = getattr(torch_module, "use_deterministic_algorithms", None)
        if callable(deterministic):
            deterministic(True)
            torch_deterministic = True

    record = SeedRecord(
        run_id=run_id.strip(),
        seed=seed,
        recorded_at_utc=_normalise_utc(now_utc),
        python_random_seeded=True,
        numpy_seeded=numpy_seeded,
        torch_seeded=torch_seeded,
        torch_deterministic_algorithms=torch_deterministic,
        python_hash_seed=str(seed),
    )
    if record_path is not None:
        write_seed_record(record_path, record)
    return record


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_seed_record(
    path: str | os.PathLike[str], record: SeedRecord
) -> None:
    """Write one deterministic-seed record as UTF-8 JSON."""

    _write_json(Path(path), record.to_dict())


def capture_environment(
    package_names: Sequence[str] = DEFAULT_VERSION_PACKAGES,
    *,
    environment: Mapping[str, str] | None = None,
    now_utc: datetime | None = None,
) -> EnvironmentRecord:
    """Capture interpreter, platform, CPU guard, and distribution versions."""

    source_environment = os.environ if environment is None else environment
    versions: list[PackageVersion] = []
    for name in package_names:
        if not name.strip():
            raise ValueError("package names must not be empty")
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        versions.append(PackageVersion(name=name, version=version))

    cpu_environment = tuple(
        (name, source_environment.get(name)) for name in CPU_ONLY_ENVIRONMENT
    )
    return EnvironmentRecord(
        captured_at_utc=_normalise_utc(now_utc),
        python_version=sys.version,
        python_executable=sys.executable,
        python_implementation=sys.implementation.name,
        platform=platform.platform(),
        machine=platform.machine(),
        architecture_bits=64 if sys.maxsize > 2**32 else 32,
        logical_cpu_count=os.cpu_count(),
        cpu_environment=cpu_environment,
        packages=tuple(versions),
    )


def write_environment_record(
    path: str | os.PathLike[str], record: EnvironmentRecord
) -> None:
    """Write an environment/version snapshot as UTF-8 JSON."""

    _write_json(Path(path), record.to_dict())


def primary_test_override_log_path(
    sentinel_path: str | os.PathLike[str],
) -> Path:
    """Return the JSON-lines override log paired with a sentinel path."""

    sentinel = Path(sentinel_path)
    return sentinel.with_name(f"{sentinel.name}.overrides.jsonl")


def claim_primary_test_access(
    sentinel_path: str | os.PathLike[str],
    *,
    run_id: str,
    seed: int,
    override: bool = False,
    override_reason: str | None = None,
    now_utc: datetime | None = None,
) -> PrimaryTestClaim:
    """Atomically claim the write-once primary-test sentinel.

    A second claim raises :class:`PrimaryTestAlreadyClaimedError`.  An explicit
    ``override=True`` requires a non-empty reason, preserves the original
    sentinel, appends an auditable JSON line to a separate override log, and
    emits a warning through this module's logger.
    """

    _validate_seed(seed)
    clean_run_id = run_id.strip()
    if not clean_run_id:
        raise ValueError("run_id must not be empty")

    sentinel = Path(sentinel_path)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    claimed_at = _normalise_utc(now_utc)
    first_payload: dict[str, object] = {
        "claimed_at_utc": claimed_at,
        "run_id": clean_run_id,
        "schema_version": 1,
        "seed": seed,
    }

    try:
        with sentinel.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(first_payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        if not override:
            raise PrimaryTestAlreadyClaimedError(
                f"Primary test already claimed; sentinel exists at {sentinel}"
            ) from error
        reason = "" if override_reason is None else override_reason.strip()
        if not reason.lower().startswith(INFRASTRUCTURE_OVERRIDE_PREFIX):
            raise ValueError(
                "override_reason must begin with "
                f"{INFRASTRUCTURE_OVERRIDE_PREFIX!r} for a primary-test override"
            )

        try:
            predecessor_payload = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as read_error:
            raise RuntimeError(
                "existing primary-test sentinel is not valid JSON"
            ) from read_error
        predecessor_run_id = predecessor_payload.get("run_id")
        if not isinstance(predecessor_run_id, str) or not predecessor_run_id:
            raise RuntimeError("existing primary-test sentinel has no predecessor run_id")

        log_path = primary_test_override_log_path(sentinel)
        override_payload = {
            "claimed_at_utc": claimed_at,
            "predecessor_run_id": predecessor_run_id,
            "predecessor_sentinel_sha256": sha256_file(sentinel),
            "reason": reason,
            "run_id": clean_run_id,
            "schema_version": 1,
            "seed": seed,
        }
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(override_payload, sort_keys=True) + "\n")
        LOGGER.warning(
            "Primary-test access override logged for run_id=%s at %s: %s",
            clean_run_id,
            log_path,
            reason,
        )
        return PrimaryTestClaim(
            sentinel_path=sentinel,
            run_id=clean_run_id,
            seed=seed,
            claimed_at_utc=claimed_at,
            was_override=True,
            override_log_path=log_path,
        )

    return PrimaryTestClaim(
        sentinel_path=sentinel,
        run_id=clean_run_id,
        seed=seed,
        claimed_at_utc=claimed_at,
        was_override=False,
        override_log_path=None,
    )


def sha256_file(
    path: str | os.PathLike[str],
    *,
    chunk_size_bytes: int = DEFAULT_HASH_CHUNK_SIZE_BYTES,
) -> str:
    """Return the lowercase SHA-256 digest of a file.

    ``chunk_size_bytes`` is the maximum number of bytes read per iteration.
    """

    if isinstance(chunk_size_bytes, bool) or not isinstance(chunk_size_bytes, int):
        raise TypeError("chunk_size_bytes must be an integer")
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size_bytes):
            digest.update(block)
    return digest.hexdigest()


def _windows_peak_rss_bytes() -> int:
    """Read Windows ``PeakWorkingSetSize`` for the current process in bytes."""

    from ctypes import wintypes

    size_t = ctypes.c_size_t

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
            ("PrivateUsage", size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(
        get_current_process(), ctypes.byref(counters), counters.cb
    ):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def process_peak_rss_bytes() -> int:
    """Return current-process peak resident-set size in bytes.

    Windows uses ``PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize`` through
    ``ctypes`` and adds no dependency.  POSIX is supported for developer/CI
    portability via ``resource.getrusage``; Linux reports KiB and macOS bytes.
    """

    if sys.platform == "win32":
        return _windows_peak_rss_bytes()

    import resource

    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


def evaluate_peak_rss_gate(
    peak_rss_bytes: int | None = None,
    *,
    limit_bytes: int = PEAK_RSS_LIMIT_BYTES,
) -> MemoryGateResult:
    """Evaluate the strict ``peak RSS < limit`` gate; quantities are bytes."""

    observed = process_peak_rss_bytes() if peak_rss_bytes is None else peak_rss_bytes
    for name, value in (("peak_rss_bytes", observed), ("limit_bytes", limit_bytes)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer byte count")
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if limit_bytes == 0:
        raise ValueError("limit_bytes must be positive")

    passed = observed < limit_bytes
    return MemoryGateResult(
        peak_rss_bytes=observed,
        limit_bytes=limit_bytes,
        passed=passed,
        headroom_bytes=limit_bytes - observed,
    )


def require_peak_rss_below_limit(
    peak_rss_bytes: int | None = None,
    *,
    limit_bytes: int = PEAK_RSS_LIMIT_BYTES,
) -> MemoryGateResult:
    """Return the peak-RSS gate result or raise when it is not strictly below."""

    result = evaluate_peak_rss_gate(peak_rss_bytes, limit_bytes=limit_bytes)
    if not result.passed:
        raise MemoryLimitExceededError(
            "Peak RSS gate failed: "
            f"{result.peak_rss_bytes} bytes is not below {result.limit_bytes} bytes."
        )
    return result


__all__ = [
    "CPU_ONLY_ENVIRONMENT",
    "DEFAULT_HASH_CHUNK_SIZE_BYTES",
    "INFRASTRUCTURE_OVERRIDE_PREFIX",
    "DEFAULT_VERSION_PACKAGES",
    "PEAK_RSS_LIMIT_BYTES",
    "CpuOnlyViolationError",
    "EnvironmentRecord",
    "MemoryGateResult",
    "MemoryLimitExceededError",
    "PackageVersion",
    "PrimaryTestAlreadyClaimedError",
    "PrimaryTestClaim",
    "SeedRecord",
    "TorchDeviceStatus",
    "capture_environment",
    "claim_primary_test_access",
    "enforce_cpu_only_environment",
    "evaluate_peak_rss_gate",
    "inspect_torch_device",
    "primary_test_override_log_path",
    "process_peak_rss_bytes",
    "require_peak_rss_below_limit",
    "require_torch_cpu_only",
    "set_deterministic_seed",
    "sha256_file",
    "write_environment_record",
    "write_seed_record",
]
