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

"""Hot-spot LOCATION observability: the negative result, pinned.

These tests exist to stop the 2D/3D field-reconstruction idea being revived
on optimism. If someone later believes external sensors can locate a hot
spot, this suite is the thing that has to be argued with -- and the
argument has to be a physical one, not a better optimiser.
"""

from __future__ import annotations

import numpy as np
import pytest

from corefield.observability import (
    AxialWindingModel,
    external_location_bound,
    internal_location_bound,
    probes_required_for,
)


def test_top_oil_is_exactly_invariant_to_hot_spot_location():
    """Top-oil does not move AT ALL when the hot spot moves. Machine precision.

    This is the whole argument in one assertion. Every external measurement
    is a function of TOTAL winding loss; moving the hot spot redistributes
    loss without changing its total. The location sits in the exact null
    space of the observation, so no estimator of any kind can recover it.
    """
    model = AxialWindingModel()
    readings = [model.external_observations(z)[0] for z in (0.10, 0.50, 0.90)]
    assert readings[0] == pytest.approx(readings[1], abs=1e-9)
    assert readings[1] == pytest.approx(readings[2], abs=1e-9)


def test_bottom_oil_is_also_exactly_invariant():
    model = AxialWindingModel()
    readings = [model.external_observations(z)[1] for z in (0.10, 0.50, 0.90)]
    assert max(readings) - min(readings) < 1e-9


def test_total_loss_is_conserved_under_relocation():
    """The conservation law the null space follows from."""
    model = AxialWindingModel()
    z = model.height
    totals = [float(np.trapezoid(model.loss_shape(loc), z)) for loc in (0.1, 0.5, 0.9)]
    for total in totals:
        assert total == pytest.approx(1.0, abs=1e-9)


def test_external_measurements_cannot_locate_the_hot_spot():
    """CRLB from external channels is ~40 % of winding height: no information.

    The winding is only 100 % tall. A bound of +/-40 % on a hot spot that
    occupies the top 10 % is indistinguishable from knowing nothing.
    """
    bound = external_location_bound()
    assert not bound.is_identifiable
    assert bound.std_percent_of_height > 20.0
    # Only the stratification-sensitive channels carry anything at all.
    top_oil, bottom_oil = bound.per_channel_sensitivity_K[0], bound.per_channel_sensitivity_K[1]
    assert abs(top_oil) < 1e-6
    assert abs(bottom_oil) < 1e-6


def test_external_route_would_need_implausible_instrumentation():
    """Resolving location to +/-5 % needs ~0.04 K noise -- ~11x better than practical.

    Quoted so the trade-off is explicit rather than rhetorical: the external
    route is not merely difficult, it is gated behind instrumentation that
    does not exist for oil temperature measurement.
    """
    bound = external_location_bound()
    assert bound.noise_needed_for_5pct_K < 0.1
    assert bound.noise_needed_for_5pct_K < AxialWindingModel().sensor_noise_K / 5


def test_two_internal_probes_solve_the_problem():
    """Two probes bracketing the hot spot: ~0.33 % of winding height.

    The contrast is the point. The problem is not hard -- it is hard from
    OUTSIDE. Inside, it needs no machine learning at all.
    """
    bound = internal_location_bound([0.80, 0.95])
    assert bound.is_identifiable
    assert bound.std_percent_of_height < 1.0


def test_internal_beats_external_by_two_orders_of_magnitude():
    outside = external_location_bound().std_percent_of_height
    inside = internal_location_bound([0.80, 0.95]).std_percent_of_height
    assert outside / inside > 50.0


def test_one_probe_is_already_enough_for_practical_purposes():
    """A single probe at the expected location gives ~1.9 % of height."""
    bound = internal_location_bound([0.90])
    assert bound.std_percent_of_height < 5.0


def test_probe_count_saturates_quickly():
    """Beyond two probes the bound barely improves -- diminishing returns.

    Relevant to a sensor-placement product: the value is in WHERE the probes
    go, not how many there are.
    """
    two = internal_location_bound([0.80, 0.95]).std_percent_of_height
    eight = internal_location_bound(np.linspace(0.5, 1.0, 8)).std_percent_of_height
    assert two / eight < 3.0


def test_probes_required_for_reports_a_small_number():
    assert 1 <= probes_required_for(1.0) <= 4


def test_probe_positions_are_validated():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        internal_location_bound([0.5, 1.5])
    with pytest.raises(ValueError, match="non-empty"):
        internal_location_bound([])


# --------------------------------------------------------------------------
# Detecting a hot spot that has moved between windings
#
# On a published 400 MVA ONAF unit the governing hot spot moves from the 120 kV
# winding to the 410 kV winding between 1.00 and 1.29 pu. Fitting a load
# exponent through that handover fits a change of measurement location rather
# than of physics, and on that unit it returned an answer 8 K worse than making
# no correction at all -- confidently, with nothing in the fit complaining.
# --------------------------------------------------------------------------


def test_a_winding_handover_is_detected():
    """The published case: local exponents 1.12, 0.68, 1.67 -- a dip in the middle."""
    from corefield.observability import detect_winding_handover

    load = np.array([0.65, 1.00, 1.29, 1.60])
    # Gradient of the max-of-two-windings series, which crosses at ~1.14 pu.
    gradient = np.array([13.1, 21.2, 25.2, 36.1])
    result = detect_winding_handover(load, gradient)
    assert result.detected
    assert 1.0 < result.load_pu < 1.3
    assert "not one physical location" in result.note


def test_a_single_winding_tracked_throughout_is_not_flagged():
    """The same unit, one winding end to end: exponents rise smoothly, no dip."""
    from corefield.observability import detect_winding_handover

    load = np.array([0.65, 1.00, 1.29, 1.60])
    gradient = np.array([11.0, 18.1, 25.2, 36.1])
    result = detect_winding_handover(load, gradient)
    assert not result.detected
    assert len(result.local_exponents) == 3


def test_a_clean_power_law_is_never_flagged():
    """A pure K**y series has a constant local exponent and must pass."""
    from corefield.observability import detect_winding_handover

    load = np.linspace(0.5, 1.6, 8)
    result = detect_winding_handover(load, 20.0 * load**1.3)
    assert not result.detected
    assert all(abs(e - 1.3) < 1e-9 for e in result.local_exponents)


def test_too_few_points_says_so_rather_than_guessing():
    """Three points give one interior interval and no neighbours to compare."""
    from corefield.observability import detect_winding_handover

    result = detect_winding_handover(
        np.array([0.6, 1.0, 1.4]), np.array([12.0, 20.0, 30.0]))
    assert not result.detected
    assert "not the same as not present" in result.note


def test_unsorted_input_is_handled_and_bad_input_refused():
    from corefield.observability import detect_winding_handover

    load = np.array([1.60, 0.65, 1.29, 1.00])
    gradient = np.array([36.1, 13.1, 25.2, 21.2])
    assert detect_winding_handover(load, gradient).detected

    with pytest.raises(ValueError, match="strictly positive"):
        detect_winding_handover(np.array([0.0, 1.0, 1.2, 1.4]),
                                np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="duplicate values"):
        detect_winding_handover(np.array([1.0, 1.0, 1.2, 1.4]),
                                np.array([1.0, 2.0, 3.0, 4.0]))


# --------------------------------------------------------------------------
# Is the ambient channel telling the truth?
#
# Method due to L. Paulhiac of EDF, given in correspondence: invert the
# steady-state oil model for ambient and compare against the probe. Every
# other calculation in this package trusts the ambient channel without
# testing it, and a mis-sited probe fails in the unsafe direction.
# --------------------------------------------------------------------------

from corefield.iec60076_7 import (  # noqa: E402
    ONAF_MEDIUM_LARGE_POWER,
    ThermalParams,
    simulate,
)
from corefield.observability import (  # noqa: E402
    check_ambient_consistency,
)

_AMB_PARAMS = ThermalParams(
    delta_theta_or_K=38.0, tau_o_min=150.0,
    delta_theta_hr_K=20.0, tau_w_min=7.0,
)


def _settled_record(n_days=6.0, dt_s=120.0, ambient_C=20.0, stage_pattern=None):
    """A record that holds load flat long enough for the oil to settle.

    Two long plateaus at different loads, each many oil time constants, so
    the quasi-steady mask has plenty to work with.
    """
    t = np.arange(0.0, n_days * 86400.0, dt_s)
    load = np.where((t % 86400.0) < 43200.0, 0.60, 0.85)
    amb = np.full(t.size, float(ambient_C)) if np.isscalar(ambient_C) else ambient_C
    traj = simulate(t, load, amb, _AMB_PARAMS, ONAF_MEDIUM_LARGE_POWER)
    stage = None if stage_pattern is None else stage_pattern(t, load)
    return t, load, amb, traj.top_oil_C, stage


def test_a_truthful_ambient_channel_is_not_flagged():
    t, load, amb, oil, _ = _settled_record()
    check = check_ambient_consistency(t, load, amb, oil, _AMB_PARAMS)
    assert not check.suspect
    assert check.n_quasi_steady > 30
    assert abs(check.mean_offset_K) < 0.5
    # The wording must not overclaim: passing is not validation of the probe.
    assert "does not validate the probe" in check.note


def test_a_probe_reading_warm_is_flagged_and_named_as_the_unsafe_direction():
    """A probe in the sun or the cooler exhaust reads high.

    The identified rated oil rise then comes out too small and the loading
    envelope too generous, so this is the direction that matters.
    """
    t, load, amb, oil, _ = _settled_record()
    check = check_ambient_consistency(t, load, amb + 3.0, oil, _AMB_PARAMS)
    assert check.suspect
    # Implied minus measured: a probe reading 3 K warm shows as -3 K.
    assert check.mean_offset_K == pytest.approx(-3.0, abs=0.3)
    assert "unsafe direction" in check.note
    assert "not a correction" in check.note


def test_a_probe_reading_cool_is_flagged_as_conservative():
    t, load, amb, oil, _ = _settled_record()
    check = check_ambient_consistency(t, load, amb - 3.0, oil, _AMB_PARAMS)
    assert check.suspect
    assert check.mean_offset_K == pytest.approx(3.0, abs=0.3)
    assert "conservative direction" in check.note


def test_a_stage_dependent_offset_points_at_the_probe_not_the_model():
    """Fan recirculation onto the probe is the signal nothing else can see.

    A genuine model error is a property of the physics and lands equally on
    every cooling stage. A probe in the exhaust shifts when the fans start.
    Constructed so the MEAN offset is near zero and only the split is large,
    which is precisely the case a mean-only test would miss.
    """
    t, load, amb, oil, _ = _settled_record()
    stage = np.where(load > 0.7, 2, 1)
    # +2 K only while the fans run, -2 K otherwise: mean cancels, split does not.
    corrupted = amb + np.where(stage == 2, 2.0, -2.0)
    check = check_ambient_consistency(
        t, load, corrupted, oil, _AMB_PARAMS, cooling_stage=stage
    )
    assert check.suspect
    assert abs(check.mean_offset_K) < 1.0          # a mean-only test would pass this
    assert check.stage_spread_K == pytest.approx(4.0, abs=0.5)
    assert "points at the probe" in check.note
    assert set(check.per_stage_offset_K) == {1, 2}


def test_a_record_that_never_settles_declines_to_report():
    """No quasi-steady samples means the inversion has nowhere to stand."""
    t = np.arange(0.0, 3 * 86400.0, 120.0)
    # Load never holds still for three oil time constants.
    load = 0.7 + 0.2 * np.sin(2 * np.pi * t / 3600.0)
    amb = np.full(t.size, 20.0)
    traj = simulate(t, load, amb, _AMB_PARAMS, ONAF_MEDIUM_LARGE_POWER)
    check = check_ambient_consistency(t, load, amb, traj.top_oil_C, _AMB_PARAMS)
    assert not check.suspect
    assert check.n_quasi_steady < 30
    assert "NOT CHECKED" in check.note
    assert "not the same as the ambient channel being sound" in check.note


def test_mismatched_shapes_are_refused():
    t, load, amb, oil, _ = _settled_record()
    with pytest.raises(ValueError, match="ambient_C shape"):
        check_ambient_consistency(t, load, amb[:-1], oil, _AMB_PARAMS)


# --------------------------------------------------------------------------
# Do the winding and oil channels share a datum?
#
# Found on a public 40 MVA ONAN record (SINTEF DynaLoad, Zenodo
# 10.5281/zenodo.17223516, CC-BY-4.0), where 74 % of quasi-steady samples show
# the fibre winding probe reading COLDER than the top-oil probe. The IEC form
# cannot produce that at any positive parameter value.
# --------------------------------------------------------------------------

from corefield.observability import check_gradient_datum  # noqa: E402


def _gradient_record(offset_K=0.0, rated_K=25.0, y=1.6, n_per_level=40):
    """Quasi-steady samples across a load range, with an optional datum offset."""
    levels = np.linspace(0.10, 0.60, 12)
    load = np.repeat(levels, n_per_level)
    top_oil = np.full(load.size, 40.0)
    hotspot = top_oil + rated_K * load**y - offset_K
    return load, top_oil, hotspot


def test_a_shared_datum_is_not_flagged():
    load, oil, hot = _gradient_record(offset_K=0.0)
    check = check_gradient_datum(load, oil, hot)
    assert not check.suspect
    assert check.negative_fraction == 0.0
    assert "share a datum" in check.note


def test_a_datum_offset_is_detected_and_its_size_recovered():
    """The offset is the diagnosis; recovering it is what makes the flag useful."""
    load, oil, hot = _gradient_record(offset_K=11.0, rated_K=51.0, y=1.0)
    check = check_gradient_datum(load, oil, hot)
    assert check.suspect
    assert check.negative_fraction > 0.05
    assert check.offset_K == pytest.approx(11.0, abs=0.5)
    assert check.rated_gradient_K == pytest.approx(51.0, rel=0.05)
    assert check.exponent == pytest.approx(1.0, abs=0.1)
    # The whole point: allowing the offset explains the data far better.
    assert check.rmse_with_offset_K < check.rmse_without_offset_K


def test_the_note_says_the_offset_is_never_subtracted():
    """A correction applied silently would invent a measurement."""
    load, oil, hot = _gradient_record(offset_K=11.0)
    note = check_gradient_datum(load, oil, hot).note
    assert "never subtracted" in note
    assert "cannot produce a negative gradient" in note


def test_a_record_that_never_settles_is_not_checked():
    """The gradient relation is a steady-state statement and needs settled load.

    An earlier version of this test used `linspace`, whose 0.002 pu steps sit
    INSIDE the 0.01 pu tolerance, so every sample counted as quasi-steady and
    the test asserted the opposite of what it set up. The load here alternates
    by 0.4 pu every sample, which genuinely never settles.
    """
    load = np.tile([0.20, 0.60], 200)
    oil = np.full(load.size, 40.0)
    hot = oil + 25.0 * load**1.6
    check = check_gradient_datum(load, oil, hot)
    assert not check.suspect
    assert check.n_quasi_steady == 0
    assert "NOT CHECKED" in check.note


def test_gradient_datum_mismatched_shapes_are_refused():
    load, oil, hot = _gradient_record()
    with pytest.raises(ValueError, match="same shape"):
        check_gradient_datum(load, oil, hot[:-1])
