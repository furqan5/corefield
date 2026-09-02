"""Focused, private-data-free tests for the frozen E1 adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corefield_ml_lab.e1 import (  # noqa: E402
    ChannelAggregate,
    E1_SEEDS,
    FIELD_TOLERANCE_K,
    GateResult,
    PrivateFieldAccessError,
    PrivateFieldConfig,
    PrivateFieldDependencyError,
    PrivateFieldExecutionError,
    PrivateFieldParseError,
    PrivateFieldResult,
    SIGNED_PEAK_CONVENTION,
    SyntheticE1Result,
    SyntheticModelMetrics,
    VersionMetadata,
    evaluate_field_gate,
    evaluate_overall_e1_gate,
    evaluate_synthetic_gate,
    parse_private_field_aggregates,
    reproduction_tolerance_K,
    run_private_field_e1,
    run_synthetic_e1,
)


@dataclass(frozen=True)
class FakeTrajectoryMetrics:
    rmse_K: float
    max_abs_K: float
    peak_error_K: float
    event_rmse_K: float = 0.0


class FakeComparison:
    day = "C"

    def __init__(self) -> None:
        self.metrics = {
            "A": tuple(FakeTrajectoryMetrics(2.59, 5.76, 5.5 + 0.05 * i) for i in range(10)),
            "B": tuple(FakeTrajectoryMetrics(1.77, 4.82, 2.7 + 0.02 * i) for i in range(10)),
            "C": tuple(
                FakeTrajectoryMetrics(0.11, 0.25, peak)
                for peak in (-0.20, -0.10, -0.02, 0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.32)
            ),
        }

    def mean(self, model: str, field: str) -> float:
        values = [float(getattr(item, field)) for item in self.metrics[model]]
        return round(sum(values) / len(values), 2)

    def worst_peak(self, model: str) -> float:
        return round(max(item.peak_error_K for item in self.metrics[model]), 2)


class FakeCampaign:
    __file__ = "vendored/corefield/campaign.py"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def day_transfer(self, day: str, *, n_seeds: int) -> FakeComparison:
        self.calls.append((day, n_seeds))
        return FakeComparison()


class FakeCorefield:
    __version__ = "test-reference"
    __file__ = "vendored/corefield/__init__.py"


def _exact_synthetic_result() -> SyntheticE1Result:
    models = {
        "A": SyntheticModelMetrics(2.59, 5.76, 5.76, 6.17, 5.1, 0.0, 10),
        "B": SyntheticModelMetrics(1.77, 4.82, 2.72, 3.17, 2.2, 0.0, 10),
        "C": SyntheticModelMetrics(0.11, 0.25, -0.02, 0.32, -0.20, 0.3, 10),
    }
    versions = VersionMetadata(
        python="3.test",
        implementation="CPython",
        platform="test",
        corefield="test-reference",
        numpy="test",
        scipy="test",
        corefield_file="vendored/corefield/__init__.py",
        campaign_file="vendored/corefield/campaign.py",
    )
    return SyntheticE1Result("C", E1_SEEDS, SIGNED_PEAK_CONVENTION, models, versions)


PRIVATE_STDOUT = """\
IN-SAMPLE   (74 d, fitted)   [sentinel-load samples rejected]
  hot-spot                   RMSE  9.90 K   bias +1.00   p95 10.00   worst 20.00   n=9999
  top-oil                    RMSE  8.80 K   bias +2.00   p95  9.00   worst 18.00   n=9999

OUT-OF-SAMPLE (42 d, unseen)   [sentinel-load samples rejected]
  hot-spot                   RMSE  1.55 K   bias -0.68   p95  2.98   worst 19.96   n=5029
  top-oil                    RMSE  1.34 K   bias -0.93   p95  2.70   worst  5.19   n=5029
  -> 2 degC operational criterion: MET

=== what this result does NOT establish ===
  row-level telemetry is never printed here
"""


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def test_synthetic_adapter_calls_frozen_day_and_records_unsafe_sign() -> None:
    campaign = FakeCampaign()

    result = run_synthetic_e1(
        campaign_module=campaign,
        corefield_module=FakeCorefield(),
    )

    assert campaign.calls == [("C", 10)]
    assert result.seeds == tuple(range(1000, 1010))
    assert result.signed_peak_convention.endswith("negative is unsafe-low")
    assert result.models["C"].largest_signed_peak_error_K == pytest.approx(0.32)
    assert result.models["C"].most_negative_signed_peak_error_K == pytest.approx(-0.20)
    assert result.models["C"].unsafe_low_seed_fraction == pytest.approx(0.3)
    assert result.models["C"].n_seeds == 10
    assert result.versions.corefield == "test-reference"
    assert result.versions.campaign_file == "vendored/corefield/campaign.py"
    json.dumps(result.as_dict())


def test_synthetic_tolerance_and_gate_are_frozen() -> None:
    assert reproduction_tolerance_K(2.59) == pytest.approx(0.1295)
    assert reproduction_tolerance_K(0.11) == pytest.approx(0.01)
    assert evaluate_synthetic_gate(_exact_synthetic_result()).status == "pass"

    result = _exact_synthetic_result()
    failed_a = replace(result.models["A"], mean_rmse_K=2.73)
    failed = replace(result, models={**result.models, "A": failed_a})
    gate = evaluate_synthetic_gate(failed)
    assert gate.status == "fail"
    failed_check = next(check for check in gate.checks if check.label == "A.mean_rmse_K")
    assert failed_check.passed is False
    assert failed_check.absolute_error == pytest.approx(0.14)


def test_private_parser_selects_held_out_aggregates_only() -> None:
    result = parse_private_field_aggregates(PRIVATE_STDOUT)

    assert result.hotspot.rmse_K == pytest.approx(1.55)
    assert result.top_oil.rmse_K == pytest.approx(1.34)
    assert result.hotspot.bias_K == pytest.approx(-0.68)
    assert result.hotspot.worst_abs_K == pytest.approx(19.96)
    assert result.hotspot.n_observations == 5029
    assert "stdout" not in result.as_dict()
    assert "9.90" not in json.dumps(result.as_dict())


def test_private_parser_refuses_missing_held_out_section() -> None:
    with pytest.raises(PrivateFieldParseError, match="OUT-OF-SAMPLE"):
        parse_private_field_aggregates("IN-SAMPLE RMSE 1.55 K")


def test_private_adapter_runs_in_place_with_B_and_returns_no_stdout(tmp_path: Path) -> None:
    script = tmp_path / "final_field.py"
    workbook = tmp_path / "private.xlsx"
    script.write_text("# aggregate-only fixture\n", encoding="utf-8")
    workbook.write_bytes(b"fixture; never opened by the fake runner")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if "-c" in command:
            metadata_stdout = json.dumps({"python": "3.test", "openpyxl": "3.test"})
            return _completed(command, stdout=metadata_stdout)
        return _completed(command, stdout=PRIVATE_STDOUT)

    result = run_private_field_e1(
        PrivateFieldConfig(
            script_path=script,
            workbook_path=workbook,
            python_executable=sys.executable,
            timeout_s=5.0,
        ),
        runner=runner,
    )

    assert len(calls) == 2
    assert calls[0][0][1:3] == ["-B", "-c"]
    assert calls[1][0][1:] == ["-B", script.name]
    assert calls[1][1]["cwd"] == str(tmp_path.resolve())
    assert calls[1][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert result.hotspot.rmse_K == pytest.approx(1.55)
    assert result.python == "3.test"
    assert result.openpyxl == "3.test"
    assert result.script_sha256 == hashlib.sha256(script.read_bytes()).hexdigest()
    assert "captured" not in result.as_dict()


def test_private_adapter_fails_explicitly_when_workbook_is_absent(tmp_path: Path) -> None:
    script = tmp_path / "final_field.py"
    script.write_text("# fixture\n", encoding="utf-8")

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("runner must not be called without private access")

    with pytest.raises(PrivateFieldAccessError, match="workbook"):
        run_private_field_e1(
            PrivateFieldConfig(script, tmp_path / "missing.xlsx", sys.executable),
            runner=runner,
        )


def test_private_adapter_fails_explicitly_without_openpyxl(tmp_path: Path) -> None:
    script = tmp_path / "final_field.py"
    workbook = tmp_path / "private.xlsx"
    script.write_text("# fixture\n", encoding="utf-8")
    workbook.write_bytes(b"fixture")
    calls = 0

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _completed(command, returncode=1, stderr="No module named openpyxl")

    with pytest.raises(PrivateFieldDependencyError, match="openpyxl"):
        run_private_field_e1(
            PrivateFieldConfig(script, workbook, sys.executable),
            runner=runner,
        )
    assert calls == 1


def test_private_execution_failure_does_not_echo_captured_output(tmp_path: Path) -> None:
    script = tmp_path / "final_field.py"
    workbook = tmp_path / "private.xlsx"
    script.write_text("# fixture\n", encoding="utf-8")
    workbook.write_bytes(b"fixture")

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "-c" in command:
            return _completed(
                command,
                stdout=json.dumps({"python": "3.test", "openpyxl": "3.test"}),
            )
        return _completed(command, returncode=2, stdout="SENSITIVE_ROW_LEVEL_OUTPUT")

    with pytest.raises(PrivateFieldExecutionError) as exc_info:
        run_private_field_e1(
            PrivateFieldConfig(script, workbook, sys.executable),
            runner=runner,
        )
    assert "SENSITIVE_ROW_LEVEL_OUTPUT" not in str(exc_info.value)


def test_field_and_overall_gate_do_not_convert_not_run_into_pass() -> None:
    channel_hotspot = ChannelAggregate(1.55, -0.68, 2.98, 19.96, 5029)
    channel_oil = ChannelAggregate(1.34, -0.93, 2.70, 5.19, 5029)
    field = PrivateFieldResult(channel_hotspot, channel_oil)

    field_gate = evaluate_field_gate(field)
    assert field_gate == GateResult(status="pass", checks=field_gate.checks)
    assert all(check.tolerance == FIELD_TOLERANCE_K for check in field_gate.checks)

    synthetic = _exact_synthetic_result()
    assert evaluate_overall_e1_gate(synthetic, None).status == "not_run"
    assert evaluate_overall_e1_gate(synthetic, field).status == "pass"

    failed_field = replace(field, hotspot=replace(channel_hotspot, rmse_K=1.58))
    assert evaluate_overall_e1_gate(synthetic, failed_field).status == "fail"
