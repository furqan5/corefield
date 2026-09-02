from __future__ import annotations

import numpy as np
import pytest

from corefield_ml_lab.experiments import (
    E2_REFERENCE_BUDGETS,
    E2_SEEDS,
    E3_SEEDS,
    E3_TARGET_LOADS_PU,
    E4_SEEDS,
    METHOD_NAMES,
    _recorded_primary_command,
    _score_target_plateau,
    aggregate_e2_rows,
    aggregate_e4_rows,
    aggregate_e3_rows,
    e3_configuration,
    e2_configuration,
    e4_configuration,
)
from corefield_ml_lab.synthetic_lab import FeatureFrame


def test_e3_configuration_resolves_frozen_protocol() -> None:
    config = e3_configuration()
    assert tuple(config["seeds"]) == E3_SEEDS
    assert tuple(config["evaluation"]["loads_pu"]) == E3_TARGET_LOADS_PU
    assert tuple(config["methods"]) == METHOD_NAMES
    assert config["reference_budget"] == 20
    assert config["neural_training"]["max_epochs"] == 2000
    assert config["neural_training"]["patience"] == 150
    assert config["physics_mode"] == "structural_mismatch"


def test_e2_configuration_resolves_scarce_reference_protocol() -> None:
    config = e2_configuration()
    assert tuple(config["seeds"]) == E2_SEEDS
    assert tuple(config["reference_budgets"]) == E2_REFERENCE_BUDGETS
    assert config["physics_mode"] == "matched"
    assert config["reference_policy"].startswith("first N")


def test_e4_configuration_fixes_hard_gate_and_effect_size() -> None:
    config = e4_configuration()
    assert tuple(config["seeds"]) == E4_SEEDS
    assert config["reference_budget"] == 20
    assert config["adoption_rule"]["minimum_paired_mean_rmse_improvement_K"] == 0.10
    assert config["adoption_rule"]["outside_hull_residual_must_be_positive_zero_bit_exact"]


def test_recorded_override_command_retains_reason_and_rejects_ambiguous_use() -> None:
    command = _recorded_primary_command(
        "e3", override=True, override_reason="documented infrastructure failure"
    )
    assert command[-3:] == [
        "--override",
        "--override-reason",
        "documented infrastructure failure",
    ]
    with pytest.raises(ValueError, match="required"):
        _recorded_primary_command("e3", override=True, override_reason=None)
    with pytest.raises(ValueError, match="requires"):
        _recorded_primary_command("e3", override=False, override_reason="unused")


def test_plateau_score_excludes_four_hour_warmup() -> None:
    time_s = np.array([0.0, 14_280.0, 14_400.0, 14_520.0])
    frame = FeatureFrame(
        split="e3_test",
        time_s=time_s,
        truth_index=np.arange(4),
        X=np.zeros((4, 9)),
        hotspot_truth_C=np.array([1000.0, 1000.0, 10.0, 12.0]),
        source_time_s=np.zeros((4, 9)),
    )
    score = _score_target_plateau(np.array([0.0, 0.0, 11.0, 13.0]), frame)
    assert score.rmse_K == 1.0
    assert score.signed_peak_error_K == 1.0


def test_e3_aggregation_uses_all_ten_paired_seeds_and_triggers_confirmation() -> None:
    rows = []
    for target in E3_TARGET_LOADS_PU:
        for seed in E3_SEEDS:
            for method in METHOD_NAMES:
                rmse = 1.0
                peak = 0.0
                if method == "pinn" and target == 1.30:
                    rmse = 0.5
                    peak = -0.05
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "target_load_pu": target,
                        "mae_K": rmse,
                        "rmse_K": rmse,
                        "mean_signed_error_K": 0.0,
                        "signed_peak_error_K": peak,
                    }
                )
    result = aggregate_e3_rows(rows)
    cell = result["cells"]["1.30"]["pinn"]
    assert cell["availability_count"] == 10
    assert cell["metrics"]["rmse_K"]["count"] == 10
    assert result["paired_decisions"]["pinn_vs_nls_at_1.30pu"]["beats_nls"]
    assert result["confirmation_required"] == [
        {
            "method": "pinn",
            "target_load_pu": 1.30,
            "reserved_seeds": list(range(51_000, 51_010)),
            "widths": [8, 32],
        }
    ]


def test_e3_greybox_win_and_confirmation_fail_closed_without_complete_invariants() -> None:
    rows = []
    for target in E3_TARGET_LOADS_PU:
        for seed in E3_SEEDS:
            for method in METHOD_NAMES:
                rmse = 1.0
                if method == "greybox" and target == 1.30:
                    rmse = 0.5
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "target_load_pu": target,
                        "mae_K": rmse,
                        "rmse_K": rmse,
                        "mean_signed_error_K": 0.0,
                        "signed_peak_error_K": 0.0,
                    }
                )

    absent = aggregate_e3_rows(rows)
    decision = absent["paired_decisions"]["greybox_vs_nls_at_1.30pu"]
    assert decision["beats_nls_before_outside_hull_invariant"]
    assert not decision["outside_hull_invariant_passed"]
    assert not decision["beats_nls"]
    assert absent["confirmation_required"] == []

    invariants = [
        {
            "all_expected_flags": True,
            "outside_hotspot_bit_exact_nls": True,
            "outside_residual_positive_zero": True,
            "seed": seed,
            "target_load_pu": target,
        }
        for target in E3_TARGET_LOADS_PU
        for seed in E3_SEEDS
    ]
    complete = aggregate_e3_rows(rows, invariant_rows=invariants)
    decision = complete["paired_decisions"]["greybox_vs_nls_at_1.30pu"]
    assert complete["greybox_outside_hull_invariant"][
        "all_expected_cells_present_and_true"
    ]
    assert decision["beats_nls"]
    assert complete["confirmation_required"] == [
        {
            "method": "greybox",
            "target_load_pu": 1.30,
            "reserved_seeds": list(range(51_000, 51_010)),
            "widths": [8, 32],
        }
    ]

    broken = [dict(row) for row in invariants]
    broken[0]["outside_residual_positive_zero"] = False
    failed = aggregate_e3_rows(rows, invariant_rows=broken)
    assert not failed["greybox_outside_hull_invariant"][
        "all_expected_cells_present_and_true"
    ]
    assert not failed["paired_decisions"][
        "greybox_vs_nls_at_1.30pu"
    ]["beats_nls"]
    assert failed["confirmation_required"] == []


def test_e2_aggregation_treats_n3_refusal_as_refusal_and_applies_kill_rule() -> None:
    metric_rows = []
    parameter_rows = []
    availability_rows = []
    truth_params = {
        "delta_theta_or_K": 45.0,
        "tau_o_min": 150.0,
        "delta_theta_hr_K": 22.0,
        "tau_w_min": 7.0,
    }
    zero_errors = {
        name: {"signed_percent": 0.0, "absolute_percent": 0.0}
        for name in truth_params
    }
    zero_errors["median_absolute_percent_across_parameters"] = {"value": 0.0}
    for budget in E2_REFERENCE_BUDGETS:
        for seed in E2_SEEDS:
            for method in ("nls", "plain_nn", "pinn"):
                available = not (method == "nls" and budget == 3)
                availability_rows.append(
                    {
                        "available": available,
                        "budget": budget,
                        "method": method,
                        "seed": seed,
                    }
                )
                if available:
                    rmse = 1.0 if method == "nls" else 2.0
                    metric_rows.append(
                        {
                            "budget": budget,
                            "method": method,
                            "seed": seed,
                            "mae_K": rmse,
                            "rmse_K": rmse,
                            "mean_signed_error_K": 0.0,
                            "signed_peak_error_K": 0.0,
                        }
                    )
                if method in {"nls", "pinn"} and available:
                    parameter_rows.append(
                        {
                            "budget": budget,
                            "errors": zero_errors,
                            "finite_interior": True,
                            "method": method,
                            "parameters": truth_params,
                            "seed": seed,
                        }
                    )
    result = aggregate_e2_rows(metric_rows, parameter_rows, availability_rows)
    assert result["cells"]["3"]["nls"]["availability_rate"] == 0.0
    assert result["cells"]["3"]["pinn"]["availability_rate"] == 1.0
    assert not result["n3_investigate_screen"]["all_ten_held_out_rmse_below_2_K"]
    assert result["physics_informed_identification_classification"] == "reject"
    assert result["pinn_winning_budgets_N_ge_4"] == []


def test_e2_requires_strictly_no_worse_worst_peak_without_e3_tolerance() -> None:
    metric_rows = []
    parameter_rows = []
    availability_rows = []
    zero_errors = {
        name: {"signed_percent": 0.0, "absolute_percent": 0.0}
        for name in (
            "delta_theta_or_K",
            "tau_o_min",
            "delta_theta_hr_K",
            "tau_w_min",
        )
    }
    zero_errors["median_absolute_percent_across_parameters"] = {"value": 0.0}
    parameters = {
        "delta_theta_or_K": 45.0,
        "tau_o_min": 150.0,
        "delta_theta_hr_K": 22.0,
        "tau_w_min": 7.0,
    }
    for budget in E2_REFERENCE_BUDGETS:
        for seed in E2_SEEDS:
            for method in ("nls", "plain_nn", "pinn"):
                available = not (method == "nls" and budget == 3)
                availability_rows.append(
                    {
                        "available": available,
                        "budget": budget,
                        "method": method,
                        "seed": seed,
                    }
                )
                if available:
                    metric_rows.append(
                        {
                            "budget": budget,
                            "method": method,
                            "seed": seed,
                            "mae_K": 0.5 if method == "pinn" else 1.0,
                            "rmse_K": 0.5 if method == "pinn" else 1.0,
                            "mean_signed_error_K": 0.0,
                            # PINN is only 0.05 K worse: allowed by E3, forbidden by E2.
                            "signed_peak_error_K": -0.05 if method == "pinn" else 0.0,
                        }
                    )
                if method in {"nls", "pinn"} and available:
                    parameter_rows.append(
                        {
                            "budget": budget,
                            "errors": zero_errors,
                            "finite_interior": True,
                            "method": method,
                            "parameters": parameters,
                            "seed": seed,
                        }
                    )
    result = aggregate_e2_rows(metric_rows, parameter_rows, availability_rows)
    decision = result["paired_decisions"]["pinn_vs_nls_N4"]
    assert not decision["e2_strict_no_worse_worst_signed_peak"]
    assert not decision["safety_condition"]
    assert not decision["beats_nls"]
    assert result["pinn_winning_budgets_N_ge_4"] == []


def test_e4_aggregation_requires_effect_ci_and_exact_invariant() -> None:
    rows = []
    invariants = []
    for seed in E4_SEEDS:
        for method, rmse in (("nls", 1.0), ("greybox", 0.8)):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "mae_K": rmse,
                    "rmse_K": rmse,
                    "mean_signed_error_K": 0.0,
                    "signed_peak_error_K": 0.0,
                }
            )
        invariants.append(
            {
                "all_expected_flags": True,
                "outside_hotspot_bit_exact_nls": True,
                "outside_residual_positive_zero": True,
                "seed": seed,
            }
        )
    result = aggregate_e4_rows(rows, invariants)
    assert result["classification"] == "adopt for in-range use only"
    broken = [dict(row) for row in invariants]
    broken[0]["outside_hotspot_bit_exact_nls"] = False
    assert aggregate_e4_rows(rows, broken)["classification"] == "reject"
