# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the PolyForm Noncommercial License 1.0.0.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Use is permitted for noncommercial purposes only, as that term is defined by
# the License. Commercial use requires a separate licence from the copyright
# holder. This is a source-available licence, not an open-source one.
#
# Versions of this file released before 2026-09-02 were published under the
# Apache License 2.0 and remain available under those terms; that grant is not
# and cannot be revoked.

"""Stage C regression: sensor-artifact battery, ambient, and the 1.30 pu day.

The two FAIL verdicts here are pinned as hard as the passes. They are
results, not defects:

  wti_calibration_bias  a +3 K calibration reference produces an engine that
                        looks perfectly calibrated against that reference
                        while carrying +4.11 K at the true peak
  ambient_ignored       ignoring a varying ambient UNDER-predicts the peak
                        by 3.09 K -- the one direction a thermal monitor
                        must never err in

If either of these ever starts passing, something has been quietly made
optimistic, and the test must fail loudly.
"""

from __future__ import annotations

import pytest

from conftest import assert_reproduces

from corefield.campaign import GATE_PEAK_K, GATE_RMSE_K


# --------------------------------------------------------------------------
# Artifact battery -- signed parameter errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, d_or, tau_o, d_hr, tau_w, rmse, peak, gate",
    [
        ("baseline", -0.01, -0.22, +0.08, -0.61, 0.13, +0.02, "PASS"),
        ("oil_drift", +2.15, +7.19, -4.01, -8.27, 0.29, -0.02, "PASS"),
        ("telemetry_spikes", 0.00, +0.15, +0.07, -0.74, 0.13, +0.02, "PASS"),
        ("telemetry_spikes_robust", -0.01, -0.17, +0.09, -0.62, 0.12, +0.02, "PASS"),
        ("integer_quantization", +0.01, -0.07, +0.06, -0.68, 0.13, +0.02, "PASS"),
        ("ct_gain_error", -2.59, +0.39, -2.54, -1.66, 0.15, +0.12, "PASS"),
        ("wti_calibration_bias", +0.01, +0.12, +14.52, +10.55, 2.70, +4.11, "FAIL"),
    ],
)
def test_artifact_battery(scenario_results, name, d_or, tau_o, d_hr, tau_w, rmse, peak, gate):
    """The published Stage-C battery table, signed parameter errors included.

    Errors are SIGNED because the object of study is bias, not spread. A
    scenario whose errors cancel in the mean is telling you something quite
    different from one whose errors are small.
    """
    result = scenario_results[name]
    means = result.mean_parameter_errors_pct
    assert_reproduces(means["delta_theta_or"], d_or, f"{name} delta_theta_or")
    assert_reproduces(means["tau_o"], tau_o, f"{name} tau_o")
    assert_reproduces(means["delta_theta_hr"], d_hr, f"{name} delta_theta_hr")
    assert_reproduces(means["tau_w"], tau_w, f"{name} tau_w")
    assert_reproduces(result.mean_rmse_K, rmse, f"{name} RMSE")
    assert_reproduces(result.mean_peak_error_K, peak, f"{name} peak error")
    assert result.gate == gate, (
        f"{name}: gate verdict {result.gate}, published {gate}"
    )


def test_baseline_tau_w_spread(scenario_results):
    """Seed-to-seed tau_w spread at baseline: 2.26 %.

    Pinned because it is the quantity the CRLB comparison rests on -- a
    change here moves the efficiency claim without moving any mean.
    """
    assert_reproduces(scenario_results["baseline"].tau_w_sd_pct, 2.26, "baseline tau_w sd")


# --------------------------------------------------------------------------
# The two failures -- asserted as failures
# --------------------------------------------------------------------------


def test_wti_calibration_bias_fails(scenario_results):
    """A +3 K reference bias must FAIL the gate, and must contaminate dynamics.

    The dynamics point matters commercially: if the bias only shifted the DC
    level, a "we only claim relative trends" positioning would escape it.
    tau_w moving +10.55 % says it does not.
    """
    result = scenario_results["wti_calibration_bias"]
    assert result.gate == "FAIL"
    assert result.mean_peak_error_K > GATE_PEAK_K
    means = result.mean_parameter_errors_pct
    assert means["delta_theta_hr"] > 10.0, "amplitude must be badly biased"
    assert means["tau_w"] > 5.0, (
        "the bias must reshape the DYNAMICS, not just the level -- this is the "
        "finding that kills a 'relative trends only' positioning"
    )


def test_ambient_ignored_fails_in_the_dangerous_direction(scenario_results):
    """Ignoring ambient UNDER-predicts the peak. Assert the sign, not just the size.

    -3.09 K. Under-prediction at the afternoon peak is the failure mode a
    thermal monitor exists to prevent, and it is why `corefield.ingest`
    refuses to fit without an ambient channel rather than warning.
    """
    result = scenario_results["ambient_ignored"]
    assert result.gate == "FAIL"
    assert_reproduces(result.mean_rmse_K, 3.98, "ambient-ignored RMSE")
    assert_reproduces(result.mean_peak_error_K, -3.09, "ambient-ignored peak error")
    assert result.mean_peak_error_K < 0.0, (
        "the peak error must be NEGATIVE -- under-prediction is the dangerous "
        "direction and the whole argument for requiring ambient depends on it"
    )
    means = result.mean_parameter_errors_pct
    assert_reproduces(means["tau_w"], 68.21, "ambient-ignored tau_w error")
    assert means["tau_o"] > 10.0, "oil time constant must also be corrupted"


def test_ambient_measured_passes_cleanly(scenario_results):
    """Given the true ambient, the engine recovers baseline-grade accuracy."""
    result = scenario_results["ambient_measured"]
    assert result.gate == "PASS"
    assert_reproduces(result.mean_rmse_K, 0.08, "ambient-measured RMSE")
    assert result.mean_rmse_K <= GATE_RMSE_K


def test_measuring_ambient_is_worth_roughly_fifty_times_the_error(scenario_results):
    """Measured vs ignored ambient: 0.08 K against 3.98 K RMSE.

    This ratio is the entire commissioning argument for an ambient channel,
    which costs an hourly weather feed.
    """
    measured = scenario_results["ambient_measured"].mean_rmse_K
    ignored = scenario_results["ambient_ignored"].mean_rmse_K
    assert ignored / measured > 20.0


# --------------------------------------------------------------------------
# Robust loss: insurance, not rescue
# --------------------------------------------------------------------------


def test_robust_loss_costs_nothing_at_baseline(scenario_results):
    """soft_l1 on spiked data recovers baseline accuracy: 0.12 vs 0.13 K.

    This pair is the justification for soft_l1 being default-on. The claim
    is NOT that it rescued the tested case -- plain least squares survived
    that too, and the prediction that it would not was logged as a miss
    (P18). The claim is that it costs nothing, so it is worth having
    against heavier-tailed glitch distributions than the one tested.
    """
    plain = scenario_results["telemetry_spikes"].mean_rmse_K
    robust = scenario_results["telemetry_spikes_robust"].mean_rmse_K
    baseline = scenario_results["baseline"].mean_rmse_K
    assert robust <= plain
    assert abs(robust - baseline) <= 0.02


def test_spikes_did_not_degrade_plain_least_squares(scenario_results):
    """P18 was a MISS: 289 dense samples drown symmetric zero-mean glitches.

    Pinned as a negative result. Quietly dropping a failed prediction is how
    a ledger stops being evidence.
    """
    plain = scenario_results["telemetry_spikes"].mean_rmse_K
    baseline = scenario_results["baseline"].mean_rmse_K
    assert abs(plain - baseline) <= 0.02, (
        "spikes were predicted to degrade plain LS and did not; if this ever "
        "starts failing, the glitch model has changed"
    )


def test_quantization_costs_about_one_times_baseline(scenario_results):
    """Integer-degC storage is ~1.0x baseline, not the predicted <=1.4x."""
    quantized = scenario_results["integer_quantization"].mean_rmse_K
    baseline = scenario_results["baseline"].mean_rmse_K
    assert quantized / baseline <= 1.4


def test_drift_poisons_parameters_while_the_trajectory_passes(scenario_results):
    """The quiet killer: gate PASS, parameters badly wrong.

    This is the finding that splits the product in two. The temperature
    trajectory is robust to a drifting oil gauge; a parameter-TREND product
    (aging diagnostics) is not, and needs drift-audited oil sensing.
    """
    result = scenario_results["oil_drift"]
    assert result.gate == "PASS"
    assert result.mean_rmse_K <= GATE_RMSE_K
    means = result.mean_parameter_errors_pct
    assert abs(means["tau_o"]) > 5.0
    assert abs(means["tau_w"]) > 5.0


def test_ct_gain_compensates_in_trajectory_but_not_parameters(scenario_results):
    """A 2 % CT gain error is absorbed by the amplitudes, not the trajectory."""
    result = scenario_results["ct_gain_error"]
    assert result.gate == "PASS"
    assert result.mean_rmse_K <= 0.2
    means = result.mean_parameter_errors_pct
    assert means["delta_theta_or"] < -2.0
    assert means["delta_theta_hr"] < -2.0


# --------------------------------------------------------------------------
# Day C -- the commercial table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, rmse, worst, peak, peak_worst_seed",
    [
        ("A", 2.59, 5.76, +5.76, +6.18),
        ("B", 1.77, 4.82, +2.72, +3.17),
        ("C", 0.11, 0.25, -0.02, +0.32),
    ],
)
def test_day_c_extrapolation_table(day_c_comparison, model, rmse, worst, peak, peak_worst_seed):
    """The 1.30 pu table. This is the commercial argument, so it is pinned hardest.

    Fitted on day A over a 0.6-1.2 pu hull, then asked about 1.30 pu -- the
    emergency-overload case where an operator most wants to know whether the
    extra load is safe.
    """
    assert_reproduces(day_c_comparison.mean(model, "rmse_K"), rmse, f"{model} day-C RMSE")
    assert_reproduces(day_c_comparison.mean(model, "max_abs_K"), worst, f"{model} day-C worst")
    assert_reproduces(day_c_comparison.mean(model, "peak_error_K"), peak, f"{model} day-C peak")
    assert_reproduces(
        day_c_comparison.worst_peak(model), peak_worst_seed, f"{model} day-C worst-seed peak"
    )


def test_day_c_rivals_read_high_at_overload(day_c_comparison):
    """A and B read HIGH at 1.30 pu -- false derating when capacity is worth most.

    The direction is the whole point. A model that reads high at overload
    tells an operator to shed load they did not need to shed.
    """
    assert day_c_comparison.mean("A", "peak_error_K") > 5.0
    assert day_c_comparison.mean("B", "peak_error_K") > 2.0
    assert abs(day_c_comparison.mean("C", "peak_error_K")) < 0.5


def test_day_c_model_c_worst_seed_stays_under_half_a_kelvin(day_c_comparison):
    """Even the worst of 10 seeds holds C within 0.32 K, outside its fitted hull."""
    assert day_c_comparison.worst_peak("C") < 0.5


def test_day_c_separation_is_at_least_an_order_of_magnitude(day_c_comparison):
    """C's peak error is >10x smaller than either rival's, at 1.30 pu."""
    c_peak = abs(day_c_comparison.mean("C", "peak_error_K"))
    for rival in ("A", "B"):
        assert abs(day_c_comparison.mean(rival, "peak_error_K")) > 10 * max(c_peak, 0.02)
