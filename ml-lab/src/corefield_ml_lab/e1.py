"""E1 adapters for the frozen CoreField reproduction gate.

This module deliberately contains no thermal-model implementation.  The
synthetic arm calls the imported CoreField reference, and the field arm runs
the existing private analysis in place.  Private telemetry and subprocess
stdout are never returned or copied into the lab.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib import import_module, metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Literal, Mapping


E1_SEEDS: tuple[int, ...] = tuple(range(1000, 1010))

# Frozen in PREREGISTRATION.md section 5.  Only these two headline columns
# participate in the synthetic pass/fail gate; the adapter records the other
# upstream metrics so the headline cannot hide a sign or peak-shape failure.
SYNTHETIC_TARGETS: Mapping[str, Mapping[str, float]] = {
    "A": {"mean_rmse_K": 2.59, "largest_signed_peak_error_K": 6.17},
    "B": {"mean_rmse_K": 1.77, "largest_signed_peak_error_K": 3.17},
    "C": {"mean_rmse_K": 0.11, "largest_signed_peak_error_K": 0.32},
}
FIELD_TARGETS: Mapping[str, float] = {
    "hotspot_rmse_K": 1.55,
    "top_oil_rmse_K": 1.34,
}
FIELD_TOLERANCE_K: float = 0.02

SIGNED_PEAK_CONVENTION: str = (
    "max(predicted hotspot_C) - max(true hotspot_C) [K]; "
    "negative is unsafe-low"
)


class E1Error(RuntimeError):
    """Base class for an E1 adapter failure."""


class SyntheticReferenceError(E1Error):
    """The frozen/imported CoreField reference is unavailable or malformed."""


class PrivateFieldAccessError(E1Error):
    """The external private script, workbook, or interpreter is unavailable."""


class PrivateFieldDependencyError(E1Error):
    """The configured private interpreter cannot import its Excel reader."""


class PrivateFieldExecutionError(E1Error):
    """The private aggregate-only analysis did not complete successfully."""


class PrivateFieldParseError(E1Error):
    """Expected held-out aggregate metrics were absent from private stdout."""


@dataclass(frozen=True)
class VersionMetadata:
    """Runtime and imported-reference metadata; paths identify the code used."""

    python: str
    implementation: str
    platform: str
    corefield: str
    numpy: str
    scipy: str
    corefield_file: str | None
    campaign_file: str | None


@dataclass(frozen=True)
class SyntheticModelMetrics:
    """Seed-aggregated day-C metrics for one upstream model."""

    mean_rmse_K: float
    mean_max_abs_K: float
    mean_signed_peak_error_K: float
    largest_signed_peak_error_K: float
    most_negative_signed_peak_error_K: float
    unsafe_low_seed_fraction: float
    n_seeds: int


@dataclass(frozen=True)
class SyntheticE1Result:
    """Complete synthetic E1 result with provenance metadata."""

    day: str
    seeds: tuple[int, ...]
    signed_peak_convention: str
    models: Mapping[str, SyntheticModelMetrics]
    versions: VersionMetadata

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable aggregate record."""

        return {
            "day": self.day,
            "seeds": list(self.seeds),
            "signed_peak_convention": self.signed_peak_convention,
            "models": {name: asdict(values) for name, values in self.models.items()},
            "versions": asdict(self.versions),
        }


@dataclass(frozen=True)
class ChannelAggregate:
    """One private channel's aggregate errors; no row-level values."""

    rmse_K: float
    bias_K: float
    p95_abs_K: float
    worst_abs_K: float
    n_observations: int


@dataclass(frozen=True)
class PrivateFieldResult:
    """Held-out aggregate metrics parsed from the existing private script."""

    hotspot: ChannelAggregate
    top_oil: ChannelAggregate
    python: str | None = None
    openpyxl: str | None = None
    script_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return only aggregate metrics and non-telemetry provenance."""

        return {
            "hotspot": asdict(self.hotspot),
            "top_oil": asdict(self.top_oil),
            "python": self.python,
            "openpyxl": self.openpyxl,
            "script_sha256": self.script_sha256,
        }


@dataclass(frozen=True)
class PrivateFieldConfig:
    """External locations needed to run the private baseline in place.

    The workbook must remain beside the existing script because that script
    resolves its private input relative to its working directory.  The adapter
    refuses to stage or copy it elsewhere.
    """

    script_path: Path
    workbook_path: Path
    python_executable: str = sys.executable
    timeout_s: float = 180.0


@dataclass(frozen=True)
class GateCheck:
    """One preregistered numeric comparison."""

    label: str
    actual: float
    target: float
    tolerance: float
    passed: bool

    @property
    def absolute_error(self) -> float:
        """Absolute distance from the frozen target [K]."""

        return abs(self.actual - self.target)


@dataclass(frozen=True)
class GateResult:
    """Pass/fail result for one E1 arm."""

    status: Literal["pass", "fail"]
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class OverallE1Gate:
    """Combined gate; absent private access is explicitly ``not_run``."""

    status: Literal["pass", "fail", "not_run"]
    synthetic: GateResult
    field: GateResult | None


def reproduction_tolerance_K(target: float) -> float:
    """Frozen synthetic tolerance ``max(5% * |target|, 0.01 K)``."""

    if not math.isfinite(target):
        raise ValueError("target must be finite")
    return max(0.05 * abs(target), 0.01)


def _distribution_version(name: str, module: ModuleType | Any | None = None) -> str:
    if module is not None:
        value = getattr(module, "__version__", None)
        if value is not None:
            return str(value)
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _optional_import(name: str) -> ModuleType | None:
    try:
        return import_module(name)
    except ModuleNotFoundError:
        return None


def _vendored_reference_root() -> Path | None:
    """Return this checkout's vendor root when the frozen reference exists."""

    candidate = Path(__file__).resolve().parents[2] / "vendor"
    if (candidate / "corefield" / "campaign.py").is_file():
        return candidate
    return None


def _import_default_reference() -> tuple[ModuleType, ModuleType]:
    """Import the checkout's frozen reference, falling back to an installed one."""

    vendor_root = _vendored_reference_root()
    if vendor_root is not None:
        vendor_text = str(vendor_root)
        if vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)
    try:
        corefield_module = import_module("corefield")
        campaign_module = import_module("corefield.campaign")
    except ModuleNotFoundError as exc:
        raise SyntheticReferenceError(
            "CoreField reference is not importable; install or expose the frozen "
            "lab-vendored reference before running E1"
        ) from exc

    if vendor_root is not None:
        campaign_file = getattr(campaign_module, "__file__", None)
        try:
            campaign_path = Path(str(campaign_file)).resolve(strict=True)
            campaign_path.relative_to(vendor_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise SyntheticReferenceError(
                "a non-vendored CoreField package was already imported; start E1 in a "
                "clean interpreter or inject the frozen campaign module explicitly"
            ) from exc
    return corefield_module, campaign_module


def _runtime_versions(
    corefield_module: ModuleType | Any | None,
    campaign_module: ModuleType | Any,
) -> VersionMetadata:
    numpy_module = _optional_import("numpy")
    scipy_module = _optional_import("scipy")
    return VersionMetadata(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        corefield=_distribution_version("corefield", corefield_module),
        numpy=_distribution_version("numpy", numpy_module),
        scipy=_distribution_version("scipy", scipy_module),
        corefield_file=(
            str(getattr(corefield_module, "__file__"))
            if getattr(corefield_module, "__file__", None)
            else None
        ),
        campaign_file=(
            str(getattr(campaign_module, "__file__"))
            if getattr(campaign_module, "__file__", None)
            else None
        ),
    )


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SyntheticReferenceError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise SyntheticReferenceError(f"{label} is not finite")
    return number


def run_synthetic_e1(
    *,
    campaign_module: ModuleType | Any | None = None,
    corefield_module: ModuleType | Any | None = None,
) -> SyntheticE1Result:
    """Run the frozen upstream day-C comparison over seeds 1000--1009.

    ``campaign_module`` and ``corefield_module`` are injection seams for unit
    tests and for an explicitly loaded vendored reference.  If omitted, the
    normal importable ``corefield`` package is used.
    """

    if campaign_module is None:
        imported_corefield, campaign_module = _import_default_reference()
        if corefield_module is None:
            corefield_module = imported_corefield
    if corefield_module is None:
        corefield_module = _optional_import("corefield")

    day_transfer = getattr(campaign_module, "day_transfer", None)
    if not callable(day_transfer):
        raise SyntheticReferenceError("CoreField campaign reference has no callable day_transfer")

    comparison = day_transfer("C", n_seeds=len(E1_SEEDS))
    if str(getattr(comparison, "day", "C")) != "C":
        raise SyntheticReferenceError("CoreField day_transfer returned a non-day-C comparison")
    raw_metrics = getattr(comparison, "metrics", None)
    if not isinstance(raw_metrics, Mapping):
        raise SyntheticReferenceError("CoreField comparison exposes no model-metrics mapping")

    models: dict[str, SyntheticModelMetrics] = {}
    for model in ("A", "B", "C"):
        try:
            per_seed = tuple(raw_metrics[model])
        except (KeyError, TypeError) as exc:
            raise SyntheticReferenceError(f"CoreField comparison is missing Model {model}") from exc
        if len(per_seed) != len(E1_SEEDS):
            raise SyntheticReferenceError(
                f"Model {model} returned {len(per_seed)} seeds; E1 requires {len(E1_SEEDS)}"
            )

        peaks = tuple(
            _finite_float(getattr(item, "peak_error_K", None), f"Model {model} peak error")
            for item in per_seed
        )
        values = SyntheticModelMetrics(
            mean_rmse_K=_finite_float(
                comparison.mean(model, "rmse_K"), f"Model {model} mean RMSE"
            ),
            mean_max_abs_K=_finite_float(
                comparison.mean(model, "max_abs_K"), f"Model {model} mean max-absolute"
            ),
            mean_signed_peak_error_K=_finite_float(
                comparison.mean(model, "peak_error_K"),
                f"Model {model} mean signed peak",
            ),
            largest_signed_peak_error_K=_finite_float(
                comparison.worst_peak(model), f"Model {model} largest signed peak"
            ),
            most_negative_signed_peak_error_K=min(peaks),
            unsafe_low_seed_fraction=sum(value < 0.0 for value in peaks) / len(peaks),
            n_seeds=len(peaks),
        )
        models[model] = values

    return SyntheticE1Result(
        day="C",
        seeds=E1_SEEDS,
        signed_peak_convention=SIGNED_PEAK_CONVENTION,
        models=models,
        versions=_runtime_versions(corefield_module, campaign_module),
    )


def evaluate_synthetic_gate(result: SyntheticE1Result) -> GateResult:
    """Apply the frozen six-cell synthetic E1 gate."""

    checks: list[GateCheck] = []
    for model in ("A", "B", "C"):
        if model not in result.models:
            raise ValueError(f"synthetic result is missing Model {model}")
        actuals = result.models[model]
        for field_name, target in SYNTHETIC_TARGETS[model].items():
            actual = float(getattr(actuals, field_name))
            tolerance = reproduction_tolerance_K(target)
            checks.append(
                GateCheck(
                    label=f"{model}.{field_name}",
                    actual=actual,
                    target=target,
                    tolerance=tolerance,
                    passed=abs(actual - target) <= tolerance,
                )
            )
    return GateResult(
        status="pass" if all(check.passed for check in checks) else "fail",
        checks=tuple(checks),
    )


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _parse_channel(block: str, label: str) -> ChannelAggregate:
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\s+RMSE\s+(?P<rmse>{_FLOAT_PATTERN})\s*K\s+"
        rf"bias\s+(?P<bias>{_FLOAT_PATTERN})\s+p95\s+(?P<p95>{_FLOAT_PATTERN})\s+"
        rf"worst\s+(?P<worst>{_FLOAT_PATTERN})\s+n=(?P<n>\d+)\s*$",
        flags=re.MULTILINE,
    )
    match = pattern.search(block)
    if match is None:
        raise PrivateFieldParseError(
            f"private output has no held-out aggregate line for {label}; no result recorded"
        )
    values = {key: float(match.group(key)) for key in ("rmse", "bias", "p95", "worst")}
    if not all(math.isfinite(value) for value in values.values()):
        raise PrivateFieldParseError(f"private {label} aggregate contains a non-finite value")
    count = int(match.group("n"))
    if count <= 0:
        raise PrivateFieldParseError(f"private {label} aggregate has no observations")
    return ChannelAggregate(
        rmse_K=values["rmse"],
        bias_K=values["bias"],
        p95_abs_K=values["p95"],
        worst_abs_K=values["worst"],
        n_observations=count,
    )


def parse_private_field_aggregates(stdout: str) -> PrivateFieldResult:
    """Parse only the held-out channel aggregates from private script stdout."""

    start = re.search(r"^OUT-OF-SAMPLE\b.*$", stdout, flags=re.MULTILINE)
    if start is None:
        raise PrivateFieldParseError(
            "private output has no OUT-OF-SAMPLE section; no field result recorded"
        )
    tail = stdout[start.end() :]
    next_section = re.search(r"^===", tail, flags=re.MULTILINE)
    block = tail[: next_section.start()] if next_section else tail
    return PrivateFieldResult(
        hotspot=_parse_channel(block, "hot-spot"),
        top_oil=_parse_channel(block, "top-oil"),
    )


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _readable_file(path: Path, label: str) -> Path:
    try:
        candidate = path.expanduser().resolve(strict=True)
        if not candidate.is_file():
            raise PrivateFieldAccessError(f"configured {label} is not a file: {candidate}")
        with candidate.open("rb") as stream:
            stream.read(0)
    except PrivateFieldAccessError:
        raise
    except (OSError, RuntimeError) as exc:
        raise PrivateFieldAccessError(f"configured {label} is unavailable: {path}") from exc
    return candidate


def _resolve_python(executable: str) -> str:
    supplied = Path(executable).expanduser()
    if supplied.parent != Path(".") or supplied.is_absolute():
        try:
            return str(supplied.resolve(strict=True))
        except OSError as exc:
            raise PrivateFieldAccessError(
                f"configured private Python interpreter is unavailable: {executable}"
            ) from exc
    resolved = shutil.which(executable)
    if resolved is None:
        raise PrivateFieldAccessError(
            f"configured private Python interpreter is unavailable: {executable}"
        )
    return resolved


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    vendor_root = _vendored_reference_root()
    if vendor_root is not None:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(vendor_root)
            if not existing
            else str(vendor_root) + os.pathsep + existing
        )
    return environment


def _run_captured(
    runner: Runner,
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env=_subprocess_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise PrivateFieldExecutionError(
            f"private E1 analysis exceeded the configured {timeout_s:g} s timeout"
        ) from exc
    except OSError as exc:
        raise PrivateFieldExecutionError("private E1 subprocess could not be started") from exc


def _probe_private_runtime(
    runner: Runner,
    python_executable: str,
    *,
    cwd: Path,
    timeout_s: float,
) -> tuple[str, str]:
    probe = (
        "import json, openpyxl, platform; "
        "print(json.dumps({'python': platform.python_version(), "
        "'openpyxl': openpyxl.__version__}))"
    )
    completed = _run_captured(
        runner,
        [python_executable, "-B", "-c", probe],
        cwd=cwd,
        timeout_s=min(timeout_s, 30.0),
    )
    if completed.returncode != 0:
        raise PrivateFieldDependencyError(
            "openpyxl is unavailable in the configured private Python interpreter; "
            "install the preregistered lab-only Excel reader before E1"
        )
    try:
        record = json.loads(completed.stdout.strip())
        return str(record["python"]), str(record["openpyxl"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PrivateFieldDependencyError(
            "configured private Python could import openpyxl but did not return version metadata"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_private_field_e1(
    config: PrivateFieldConfig,
    *,
    runner: Runner = subprocess.run,
) -> PrivateFieldResult:
    """Run the existing private field script in place and retain aggregates only.

    Missing private access or ``openpyxl`` raises a dedicated exception.  The
    subprocess uses ``-B`` and ``PYTHONDONTWRITEBYTECODE=1``.  Its stdout is
    parsed in memory and never included in the returned record.
    """

    if not math.isfinite(config.timeout_s) or config.timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and > 0")
    script = _readable_file(Path(config.script_path), "private field script")
    workbook = _readable_file(Path(config.workbook_path), "private field workbook")
    if workbook.parent != script.parent:
        raise PrivateFieldAccessError(
            "private workbook must remain beside the existing field script; "
            "the E1 adapter will not stage or copy it"
        )
    python_executable = _resolve_python(config.python_executable)
    python_version, openpyxl_version = _probe_private_runtime(
        runner,
        python_executable,
        cwd=script.parent,
        timeout_s=config.timeout_s,
    )

    completed = _run_captured(
        runner,
        [python_executable, "-B", script.name],
        cwd=script.parent,
        timeout_s=config.timeout_s,
    )
    if completed.returncode != 0:
        raise PrivateFieldExecutionError(
            f"private E1 analysis exited with status {completed.returncode}; "
            "captured telemetry/output was not retained"
        )
    aggregates = parse_private_field_aggregates(completed.stdout)
    return replace(
        aggregates,
        python=python_version,
        openpyxl=openpyxl_version,
        script_sha256=_sha256(script),
    )


def evaluate_field_gate(result: PrivateFieldResult) -> GateResult:
    """Apply the frozen +/-0.02 K field aggregate gate."""

    actuals = {
        "hotspot_rmse_K": result.hotspot.rmse_K,
        "top_oil_rmse_K": result.top_oil.rmse_K,
    }
    checks = tuple(
        GateCheck(
            label=label,
            actual=actuals[label],
            target=target,
            tolerance=FIELD_TOLERANCE_K,
            passed=abs(actuals[label] - target) <= FIELD_TOLERANCE_K,
        )
        for label, target in FIELD_TARGETS.items()
    )
    return GateResult(
        status="pass" if all(check.passed for check in checks) else "fail",
        checks=checks,
    )


def evaluate_overall_e1_gate(
    synthetic: SyntheticE1Result,
    field: PrivateFieldResult | None,
) -> OverallE1Gate:
    """Combine E1 arms without treating missing private access as a pass."""

    synthetic_gate = evaluate_synthetic_gate(synthetic)
    if not synthetic_gate.passed:
        return OverallE1Gate(status="fail", synthetic=synthetic_gate, field=None)
    if field is None:
        return OverallE1Gate(status="not_run", synthetic=synthetic_gate, field=None)
    field_gate = evaluate_field_gate(field)
    return OverallE1Gate(
        status="pass" if field_gate.passed else "fail",
        synthetic=synthetic_gate,
        field=field_gate,
    )
