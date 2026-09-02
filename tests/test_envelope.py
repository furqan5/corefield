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

"""Loading envelope: the commercial output, and the refusal to invent limits.

The most important tests here are the ones about PROVENANCE. The limits
decide the temperature at which this software tells an operator it is safe
to overload a transformer, and this repository's IEC text is unverified. A
remembered number at that point in the product would be the single worst
defect the package could ship, so it is made structurally impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from corefield.crlb import cramer_rao_bound
from corefield.envelope import (
    LoadingLimits,
    iec_loading_limits,
    loading_envelope,
    peak_hotspot_at_load,
)
from corefield.iec60076_7 import InitialState
from corefield.synthetic import (
    AMBIENT_CONSTANT_C,
    OIL_SAMPLE_STRIDE,
    TRUTH_PARAMS,
    calibration_indices,
    truth_trajectory,
)

# Explicitly NOT from any standard. The point of this constant is that the
# package refuses to supply one, so the test must supply its own and label it.
TEST_LIMITS = LoadingLimits(
    hotspot_limit_C=120.0,
    top_oil_limit_C=105.0,
    label="illustrative limits for regression testing",
    source="ILLUSTRATIVE TEST VALUE - not from any standard; used only to exercise code",
)

WARM_UNIT = InitialState(top_oil_C=70.0, prior_load_pu=0.9)


@pytest.fixture(scope="module")
def bound():
    truth = truth_trajectory("A")
    t = truth.time_s
    return cramer_rao_bound(
        t,
        truth.load_pu,
        np.full(t.size, AMBIENT_CONSTANT_C),
        TRUTH_PARAMS,
        np.arange(0, t.size, OIL_SAMPLE_STRIDE),
        calibration_indices(17, t),
        0.5,
    )


def _envelope(**overrides):
    kwargs = dict(
        params=TRUTH_PARAMS, limits=TEST_LIMITS, ambient_C=30.0,
        duration_h=2.0, initial_state=WARM_UNIT,
    )
    kwargs.update(overrides)
    return loading_envelope(**kwargs)


# --------------------------------------------------------------------------
# Provenance -- limits may not come from this package
# --------------------------------------------------------------------------


def test_package_refuses_to_supply_iec_limits():
    """There is no default. Looking for one must find the reasoning."""
    with pytest.raises(NotImplementedError) as excinfo:
        iec_loading_limits()
    message = str(excinfo.value)
    assert "UNVERIFIED" in message
    assert "licensed copy" in message


@pytest.mark.parametrize(
    "source",
    ["", "TBD", "IEC", "standard", "from memory", "default", "unknown", "n/a"],
)
def test_placeholder_sources_are_rejected(source):
    """A provenance field that records nothing is worse than none at all."""
    with pytest.raises(ValueError, match="provenance"):
        LoadingLimits(
            hotspot_limit_C=120.0, top_oil_limit_C=105.0, label="x", source=source
        )


def test_implausible_limits_are_rejected():
    with pytest.raises(ValueError, match="plausible range"):
        LoadingLimits(
            hotspot_limit_C=400.0, top_oil_limit_C=105.0,
            label="x", source="a properly written provenance record",
        )


def test_kelvin_style_limit_is_caught_by_the_range_check():
    """393 K would pass as a number and fail as a Celsius limit."""
    with pytest.raises(ValueError, match="degrees Celsius"):
        LoadingLimits(
            hotspot_limit_C=393.15, top_oil_limit_C=378.15,
            label="x", source="a properly written provenance record",
        )


def test_top_oil_limit_above_hotspot_limit_is_rejected():
    """The hot spot is by definition hotter than the oil around it."""
    with pytest.raises(ValueError, match="below"):
        LoadingLimits(
            hotspot_limit_C=105.0, top_oil_limit_C=120.0,
            label="x", source="a properly written provenance record",
        )


def test_source_travels_into_every_summary():
    """A number quoted to an operator must carry where its limit came from."""
    assert TEST_LIMITS.source in _envelope().summary()


# --------------------------------------------------------------------------
# Physics of the envelope
# --------------------------------------------------------------------------


def test_shorter_windows_permit_more_load():
    """Thermal mass buys short-term headroom. This is the product's core claim."""
    half_hour = _envelope(duration_h=0.5).k_max_pu
    two_hours = _envelope(duration_h=2.0).k_max_pu
    eight_hours = _envelope(duration_h=8.0).k_max_pu
    assert half_hour > two_hours > eight_hours


def test_hotter_ambient_reduces_headroom():
    """Ambient enters through the oil low-pass and moves the whole trajectory."""
    cool = _envelope(ambient_C=10.0).k_max_pu
    hot = _envelope(ambient_C=40.0).k_max_pu
    assert cool > hot


def test_warmer_starting_state_reduces_headroom():
    cold_start = _envelope(initial_state=InitialState(50.0, 0.6)).k_max_pu
    warm_start = _envelope(initial_state=InitialState(85.0, 1.1)).k_max_pu
    assert cold_start > warm_start


def test_envelope_respects_the_limit_it_was_given():
    """The returned load must actually keep the peak under the limit."""
    envelope = _envelope()
    peak_hs, peak_oil = peak_hotspot_at_load(
        envelope.k_max_pu, TRUTH_PARAMS, 30.0, 2.0, WARM_UNIT
    )
    assert peak_hs <= TEST_LIMITS.hotspot_limit_C + 1e-6
    assert peak_oil <= TEST_LIMITS.top_oil_limit_C + 1e-6


def test_just_above_the_envelope_violates_the_limit():
    """The answer must be tight, not merely safe -- otherwise it leaves value unused."""
    envelope = _envelope()
    peak_hs, peak_oil = peak_hotspot_at_load(
        envelope.k_max_pu + 0.02, TRUTH_PARAMS, 30.0, 2.0, WARM_UNIT
    )
    assert (
        peak_hs > TEST_LIMITS.hotspot_limit_C
        or peak_oil > TEST_LIMITS.top_oil_limit_C
    )


def test_peak_hotspot_is_non_decreasing_in_load():
    """Bisection needs this. Note it is NON-decreasing, not strictly increasing.

    Below the prior load the unit only cools over the window, so the peak is
    the temperature it started at and does not depend on the new load at all.
    Here the unit starts settled at 0.9 pu, and the peak is identical --
    89.18 degC -- for every load from 0.6 to 0.9 pu.

    A flat region is enough for bisection to be valid, because the feasible
    set {K : peak(K) <= limit} is still an interval anchored at zero. An
    earlier draft of this test asserted STRICT monotonicity and failed; the
    physics was right and the assertion was wrong.
    """
    loads = (0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4)
    peaks = [peak_hotspot_at_load(k, TRUTH_PARAMS, 30.0, 2.0, WARM_UNIT)[0] for k in loads]
    assert all(b >= a - 1e-9 for a, b in zip(peaks, peaks[1:])), "must be non-decreasing"
    # Flat at and below the prior load...
    assert peaks[0] == pytest.approx(peaks[3], abs=1e-9)
    # ...and strictly increasing above it.
    assert all(b > a for a, b in zip(peaks[3:], peaks[4:]))


def test_peak_below_prior_load_is_the_starting_temperature():
    """The flat region is the unit cooling, so the peak is where it began."""
    peak, _ = peak_hotspot_at_load(0.6, TRUTH_PARAMS, 30.0, 2.0, WARM_UNIT)
    initial_gradient = TRUTH_PARAMS.delta_theta_hr_K * WARM_UNIT.prior_load_pu**1.3
    assert peak == pytest.approx(WARM_UNIT.top_oil_C + initial_gradient, abs=1e-6)


def test_top_oil_can_be_the_binding_constraint():
    """With a tight oil limit the answer must say oil, not hot-spot."""
    limits = LoadingLimits(
        hotspot_limit_C=160.0, top_oil_limit_C=90.0,
        label="oil-limited case", source="ILLUSTRATIVE TEST VALUE - not from a standard",
    )
    assert _envelope(limits=limits).limiting_constraint == "top-oil"


def test_already_overloaded_unit_reports_no_headroom():
    """A unit past its limit gets a refusal, not a negative number."""
    limits = LoadingLimits(
        hotspot_limit_C=60.0, top_oil_limit_C=None,
        label="deliberately unreachable", source="ILLUSTRATIVE TEST VALUE - not a standard",
    )
    envelope = _envelope(limits=limits, initial_state=InitialState(95.0, 1.2))
    assert not envelope.feasible
    assert envelope.limiting_constraint == "infeasible"
    assert "NO HEADROOM" in envelope.summary()


def test_unbinding_limit_is_reported_as_a_search_bound():
    """When nothing binds, say so rather than implying a thermal result."""
    limits = LoadingLimits(
        hotspot_limit_C=240.0, top_oil_limit_C=None,
        label="deliberately loose", source="ILLUSTRATIVE TEST VALUE - not a standard",
    )
    envelope = _envelope(limits=limits, search_range=(0.2, 1.5))
    assert envelope.limiting_constraint == "search-bound"
    assert any("search bound" in n for n in envelope.notes)


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def test_conservative_envelope_is_never_above_the_point_estimate(bound):
    envelope = _envelope(bound=bound, n_samples=60)
    assert envelope.k_max_conservative_pu <= envelope.k_max_pu + 1e-9


def test_uncertainty_band_is_labelled_as_parameter_only(bound):
    """Overstating what the band covers would be the dangerous failure here."""
    notes = " ".join(_envelope(bound=bound, n_samples=60).notes)
    assert "PARAMETER error only" in notes
    assert "LOWER bound" in notes
    assert "forecast" in notes


def test_missing_crlb_is_disclosed_rather_than_silently_ignored():
    envelope = _envelope()
    assert envelope.k_max_conservative_pu == envelope.k_max_pu
    assert any("NO parameter-uncertainty margin" in n for n in envelope.notes)


def test_higher_confidence_is_more_conservative(bound):
    lenient = _envelope(bound=bound, confidence=0.60, n_samples=120).k_max_conservative_pu
    strict = _envelope(bound=bound, confidence=0.99, n_samples=120).k_max_conservative_pu
    assert strict <= lenient + 1e-9


def test_confidence_outside_range_is_rejected(bound):
    with pytest.raises(ValueError, match="confidence"):
        _envelope(bound=bound, confidence=1.0)


# --------------------------------------------------------------------------
# The commercial framing
# --------------------------------------------------------------------------


def test_answer_is_given_in_both_forms_a_utility_asks_for():
    """'K pu for N hours' and 'X MVA-hours above nameplate'."""
    summary = _envelope(nameplate_MVA=63.0).summary()
    assert "pu for" in summary
    assert "hours" in summary
    assert "MVA-hours above nameplate" in summary


def test_mva_hours_arithmetic_is_consistent():
    envelope = _envelope(nameplate_MVA=63.0, duration_h=2.0)
    expected = (envelope.k_max_pu - 1.0) * 63.0 * 2.0
    assert envelope.headroom_MVA_h == pytest.approx(expected, rel=1e-9)


def test_constant_voltage_assumption_is_stated():
    """MVA scaling with per-unit current is an assumption, so it is disclosed."""
    assert any("constant voltage" in n for n in _envelope(nameplate_MVA=63.0).notes)


def test_headroom_is_never_negative():
    """A unit with no headroom reports zero MVA-hours, not a negative saving."""
    limits = LoadingLimits(
        hotspot_limit_C=95.0, top_oil_limit_C=None,
        label="tight", source="ILLUSTRATIVE TEST VALUE - not from a standard",
    )
    envelope = _envelope(limits=limits, nameplate_MVA=63.0)
    if envelope.feasible:
        assert envelope.headroom_MVA_h >= 0.0


def test_extrapolation_beyond_the_fitted_hull_is_flagged():
    """An ONAF case-study error must not become another unit's safety margin."""
    envelope = _envelope(fitted_load_hull=(0.6, 1.2))
    assert envelope.k_max_pu > 1.2
    assert any("extrapolation" in n for n in envelope.notes)
    assert any("6.35 K LOW" in n for n in envelope.notes)
    assert any("not transferable" in n for n in envelope.notes)
    assert any("Neither its magnitude nor its sign supplies a safety" in n for n in envelope.notes)
    assert any("non-zero load-slope does not validate" in n for n in envelope.notes)
    assert not any("carry that as margin" in n for n in envelope.notes)


def test_a_fixed_exponent_above_the_hull_says_so_explicitly():
    """A fixed-exponent assumption is reported without asserting a universal bias."""
    envelope = _envelope(fitted_load_hull=(0.6, 1.2))
    assert any("x1 = 0" in n for n in envelope.notes)


def test_no_extrapolation_flag_when_inside_the_hull():
    envelope = _envelope(fitted_load_hull=(0.6, 1.6))
    assert not any("extrapolation" in n for n in envelope.notes)


def test_kelvin_ambient_is_rejected():
    with pytest.raises(ValueError, match="kelvin"):
        _envelope(ambient_C=303.15)


def test_ambient_profile_must_match_the_grid():
    with pytest.raises(ValueError, match="scalar or match"):
        _envelope(ambient_C=np.array([20.0, 21.0, 22.0]))


def test_ambient_profile_is_accepted_when_sized_correctly():
    """A forecast ambient profile is the realistic operational input."""
    n = int(round(2.0 * 3600.0 / 30.0)) + 1
    profile = 25.0 + 5.0 * np.sin(np.linspace(0, np.pi, n))
    envelope = _envelope(ambient_C=profile)
    assert envelope.feasible
    assert np.isfinite(envelope.k_max_pu)


def test_invalid_duration_is_rejected():
    with pytest.raises(ValueError, match="duration_h"):
        _envelope(duration_h=0.0)
