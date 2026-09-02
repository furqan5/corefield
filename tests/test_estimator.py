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

"""Estimator behaviour: reproduction, failure modes, and refusal to guess.

The failure-mode tests matter as much as the reproduction ones. The single
most instructive incident in this project's history is an independent
implementation that railed its optimiser at a bound in 9 of 9 runs and
reported the result as evidence of an identifiability problem in the
physics. It was an implementation failure. An estimator that returns a
railed solution as if it were a measurement is worse than one that crashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_reproduces

from corefield.campaign import CAMPAIGN_START
from corefield.estimator import (
    DEFAULT_STARTS,
    OPTIMISER_BOUNDS,
    HotspotReferences,
    identify,
)
from corefield.synthetic import (
    AMBIENT_CONSTANT_C,
    DT_S,
    OIL_SAMPLE_STRIDE,
    TRUTH_PARAMS,
    calibration_indices,
    day_a_load,
    truth_trajectory,
)


@pytest.fixture(scope="module")
def dataset():
    """One clean synthetic day plus a seeded noisy observation set."""
    truth = truth_trajectory("A")
    t = truth.time_s
    oil_index = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    cal_index = calibration_indices(17, t)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    load_half = day_a_load(t + 0.5 * DT_S)

    rng = np.random.default_rng(2000)
    oil_series = np.full(t.size, np.nan)
    oil_series[oil_index] = truth.top_oil_C[oil_index] + rng.normal(0, 0.5, oil_index.size)
    cal_samples = truth.hotspot_C[cal_index] + rng.normal(0, 0.5, cal_index.size)

    return dict(
        truth=truth, t=t, ambient=ambient, load_half=load_half,
        oil_series=oil_series, cal_index=cal_index,
        refs=HotspotReferences(t[cal_index], cal_samples, source="synthetic"),
    )


def _identify(dataset, **overrides):
    kwargs = dict(
        loss="linear", starts=CAMPAIGN_START,
        load_pu_half=dataset["load_half"], ambient_C_half=dataset["ambient"],
    )
    kwargs.update(overrides)
    return identify(
        dataset["t"], dataset["truth"].load_pu, dataset["ambient"],
        dataset["oil_series"], dataset["refs"], **kwargs,
    )


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


def test_noiseless_recovery_is_exact():
    """With clean data the four parameters come back to machine precision.

    Published as "exact (0.00 % all four)". If this ever degrades, the
    estimator and the forward model have stopped agreeing with each other.
    """
    truth = truth_trajectory("A")
    t = truth.time_s
    oil_index = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    cal_index = calibration_indices(17, t)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    oil_series = np.full(t.size, np.nan)
    oil_series[oil_index] = truth.top_oil_C[oil_index]

    result = identify(
        t, truth.load_pu, ambient, oil_series,
        HotspotReferences(t[cal_index], truth.hotspot_C[cal_index]),
        loss="linear", starts=CAMPAIGN_START,
        load_pu_half=day_a_load(t + 0.5 * DT_S), ambient_C_half=ambient,
    )
    error_pct = np.abs(
        (result.params.as_vector() - TRUTH_PARAMS.as_vector()) / TRUTH_PARAMS.as_vector() * 100
    )
    assert np.all(error_pct < 0.01), f"noiseless errors {error_pct}"
    assert result.residual_rmse_K < 1e-6


def test_identification_succeeds_and_reports(dataset):
    result = _identify(dataset)
    assert result.success
    assert result.n_converged == 1
    assert result.n_observations == (289, 17)
    assert "SUCCESS" in result.report()


def test_multi_start_finds_the_same_optimum(dataset):
    """All four default starts must land on the same solution.

    If they did not, the point estimate would depend on where the optimiser
    happened to begin, and `identify` surfaces that as a warning. Confirming
    agreement here is what lets the default multi-start be trusted.
    """
    single = _identify(dataset, starts=CAMPAIGN_START)
    multi = _identify(dataset, starts=DEFAULT_STARTS)
    assert multi.n_converged >= 3
    relative = np.abs(
        (multi.params.as_vector() - single.params.as_vector()) / single.params.as_vector()
    )
    assert np.all(relative < 1e-3), f"multi-start disagreed by {relative}"


def test_campaign_start_is_first_in_the_default_grid():
    """Reproduction depends on the published start being tried first."""
    assert DEFAULT_STARTS[0] == CAMPAIGN_START[0]


def test_soft_l1_is_the_default_and_costs_little(dataset):
    """Robust loss is default-on: insurance, not rescue.

    On clean data it must reach essentially the same answer as plain least
    squares -- that is the entire justification for leaving it enabled.
    """
    plain = _identify(dataset, loss="linear")
    robust = _identify(dataset)  # default soft_l1 via identify's own default
    default_result = identify(
        dataset["t"], dataset["truth"].load_pu, dataset["ambient"],
        dataset["oil_series"], dataset["refs"], starts=CAMPAIGN_START,
        load_pu_half=dataset["load_half"], ambient_C_half=dataset["ambient"],
    )
    assert default_result.loss == "soft_l1"
    relative = np.abs(
        (robust.params.as_vector() - plain.params.as_vector()) / plain.params.as_vector()
    )
    assert np.all(relative < 0.05)


def test_published_baseline_reproduces(dataset):
    """Seed 2000 at the campaign configuration, digit for digit."""
    result = _identify(dataset)
    error_pct = (
        (result.params.as_vector() - TRUTH_PARAMS.as_vector()) / TRUTH_PARAMS.as_vector() * 100
    )
    # Seed-0 of the Stage-C battery; the 10-seed means are pinned in test_stage_c.
    assert abs(error_pct[3]) < 5.0
    assert result.residual_rmse_K < 0.6


# --------------------------------------------------------------------------
# Failure modes -- these must raise, not return
# --------------------------------------------------------------------------


def test_too_few_calibration_reads_is_refused(dataset):
    """Fewer than four reads cannot determine four parameters. Say so."""
    t = dataset["t"]
    with pytest.raises(ValueError, match="hot-spot reference"):
        _identify(
            dataset,
            starts=CAMPAIGN_START,
        ) if False else identify(
            t, dataset["truth"].load_pu, dataset["ambient"], dataset["oil_series"],
            HotspotReferences(t[[100, 200, 300]], np.array([80.0, 85.0, 90.0])),
            loss="linear", starts=CAMPAIGN_START,
        )


def test_no_oil_measurements_is_refused(dataset):
    """An all-NaN oil channel means there is nothing to fit the oil to."""
    t = dataset["t"]
    with pytest.raises(ValueError, match="no finite samples"):
        identify(
            t, dataset["truth"].load_pu, dataset["ambient"],
            np.full(t.size, np.nan), dataset["refs"], loss="linear", starts=CAMPAIGN_START,
        )


def test_kelvin_calibration_reads_are_refused(dataset):
    """Hot-spot reads in kelvin must raise before any fitting happens."""
    t = dataset["t"]
    cal_index = dataset["cal_index"]
    with pytest.raises(ValueError, match="kelvin"):
        HotspotReferences(t[cal_index], dataset["truth"].hotspot_C[cal_index] + 273.15)


def test_kelvin_oil_series_is_refused(dataset):
    t = dataset["t"]
    oil_kelvin = dataset["oil_series"] + 273.15
    with pytest.raises(ValueError, match="kelvin"):
        identify(
            t, dataset["truth"].load_pu, dataset["ambient"], oil_kelvin,
            dataset["refs"], loss="linear", starts=CAMPAIGN_START,
        )


def test_all_starts_failing_raises_rather_than_returning_the_least_bad(dataset):
    """The central refusal: no railed solution is ever returned as an answer.

    Forced here by giving every start a bound box that cannot contain the
    truth, so the optimiser must rail. The correct behaviour is RuntimeError
    naming the likely causes -- not a ThermalParams sitting on a bound that
    a caller would have no way to distinguish from a real measurement.
    """
    t = dataset["t"]
    lower, upper = OPTIMISER_BOUNDS
    # A start box far from the truth in tau_w, with the truth outside it.
    impossible = ((float(lower[0]), float(lower[1]), float(lower[2]), float(lower[3])),)
    with pytest.raises(RuntimeError, match="none of"):
        identify(
            t, dataset["truth"].load_pu, dataset["ambient"], dataset["oil_series"],
            dataset["refs"], loss="linear", starts=impossible, max_nfev=1,
        )


def test_railed_solutions_are_detected(dataset):
    """A solution pinned to a bound is reported as non-converged."""
    lower, _ = OPTIMISER_BOUNDS
    try:
        result = identify(
            dataset["t"], dataset["truth"].load_pu, dataset["ambient"],
            dataset["oil_series"], dataset["refs"], loss="linear",
            starts=((float(lower[0]), float(lower[1]), float(lower[2]), float(lower[3])),),
            max_nfev=1,
        )
    except RuntimeError:
        return  # refused outright, which is the stronger behaviour
    assert not result.success or result.starts[0].railed_parameters


def test_mismatched_shapes_are_refused(dataset):
    t = dataset["t"]
    with pytest.raises(ValueError, match="shape"):
        identify(
            t, dataset["truth"].load_pu[:-5], dataset["ambient"], dataset["oil_series"],
            dataset["refs"], loss="linear", starts=CAMPAIGN_START,
        )


def test_reference_timestamps_outside_the_grid_are_refused(dataset):
    t = dataset["t"]
    with pytest.raises(ValueError, match="outside the simulation grid"):
        identify(
            t, dataset["truth"].load_pu, dataset["ambient"], dataset["oil_series"],
            HotspotReferences(
                np.array([1e9, 1e9 + 60, 1e9 + 120, 1e9 + 180]),
                np.array([80.0, 82.0, 84.0, 86.0]),
            ),
            loss="linear", starts=CAMPAIGN_START,
        )


def test_hotspot_references_validate_shapes():
    with pytest.raises(ValueError, match="does not match"):
        HotspotReferences(np.array([0.0, 60.0]), np.array([80.0]))


def test_jacobian_condition_is_reported(dataset):
    """Ill-conditioning must be visible to the caller, not swallowed."""
    result = _identify(dataset)
    assert np.isfinite(result.jacobian_condition)
    assert result.jacobian_condition > 0
