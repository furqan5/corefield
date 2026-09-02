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

"""Staged cooling: per-stage parameters, segmented integration, joint fit.

The test that matters most here is `test_ignoring_staging_is_catastrophic`.
Unmodelled cooling staging costs +6.54 K at the peak — worse than the
structural mismatch that killed Models A and B. Anyone running this package
on a real staged transformer without the stage channel would conclude the
method does not work, when the real fault is a missing input column.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_reproduces

from corefield.estimator import HotspotReferences, identify
from corefield.iec60076_7 import InitialState, ThermalParams, simulate
from corefield.staged import (
    CLEAN_COOLER_PRECONDITION,
    SHARED_BY_DEFAULT,
    StagedThermalParams,
    identify_staged,
    simulate_staged,
    stage_segments,
)
from corefield.synthetic import (
    AMBIENT_CONSTANT_C,
    DT_S,
    FAN_OFF_PU,
    FAN_ON_PU,
    OIL_SAMPLE_STRIDE,
    STAGED_TRUTH_PARAMS,
    calibration_indices,
    day_a_load,
    fan_stage_schedule,
    staged_truth_trajectory,
    truth_trajectory,
)


@pytest.fixture(scope="module")
def staged_day():
    """A staged day-A record with seeded noise on both channels."""
    trajectory, stage = staged_truth_trajectory("A")
    t = trajectory.time_s
    oil_index = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    cal_index = calibration_indices(17, t)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)

    rng = np.random.default_rng(7)
    oil = np.full(t.size, np.nan)
    oil[oil_index] = trajectory.top_oil_C[oil_index] + rng.normal(0, 0.5, oil_index.size)
    cal = trajectory.hotspot_C[cal_index] + rng.normal(0, 0.5, cal_index.size)

    return dict(
        traj=trajectory, stage=stage, t=t, ambient=ambient,
        load_half=day_a_load(t + 0.5 * DT_S), oil=oil,
        refs=HotspotReferences(t[cal_index], cal, source="synthetic"),
    )


def _fit(staged_day, **kwargs):
    options = dict(
        loss="linear",
        load_pu_half=staged_day["load_half"],
        ambient_C_half=staged_day["ambient"],
    )
    options.update(kwargs)
    return identify_staged(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
        staged_day["oil"], staged_day["refs"], staged_day["stage"], **options,
    )


# --------------------------------------------------------------------------
# Segmentation and the schedule
# --------------------------------------------------------------------------


def test_fan_schedule_has_hysteresis():
    """The fans must not chatter at the switching threshold.

    A load hovering near the setpoint would toggle the stage every sample
    without hysteresis, which no control scheme permits and which would make
    the record unfittable.
    """
    ramp = np.concatenate([np.linspace(0.8, 1.0, 50), np.linspace(1.0, 0.8, 50)])
    stage = fan_stage_schedule(ramp)
    changes = int(np.count_nonzero(np.diff(stage) != 0))
    assert changes == 2, f"expected one on and one off transition, got {changes}"
    # Turning off must happen at a LOWER load than turning on.
    assert FAN_OFF_PU < FAN_ON_PU


def test_day_a_produces_three_segments(staged_day):
    """Fans off, on over the afternoon peak, off again."""
    segments = stage_segments(staged_day["stage"])
    assert [s for _, _, s in segments] == [1, 2, 1]


def test_stage_segments_cover_the_record_exactly():
    stage = np.array([1, 1, 2, 2, 2, 1, 3, 3])
    segments = stage_segments(stage)
    assert segments == [(0, 1, 1), (2, 4, 2), (5, 5, 1), (6, 7, 3)]
    covered = sum(b - a + 1 for a, b, _ in segments)
    assert covered == stage.size


# --------------------------------------------------------------------------
# The physics: state carries, parameters jump
# --------------------------------------------------------------------------


def test_temperature_is_continuous_across_a_stage_change(staged_day):
    """Parameters jump at a stage boundary. Stored heat does not.

    The oil is exactly as hot the instant after the fan starts as the instant
    before. A discontinuity here would mean the internal state was rebuilt
    from scratch rather than carried, which is the classic way to get a
    staged model subtly wrong.
    """
    trajectory = staged_day["traj"]
    segments = stage_segments(staged_day["stage"])
    ordinary_step = float(np.max(np.abs(np.diff(trajectory.top_oil_C))))
    for start, _, _ in segments[1:]:
        jump = abs(float(trajectory.top_oil_C[start] - trajectory.top_oil_C[start - 1]))
        assert jump <= ordinary_step + 1e-9, (
            f"top-oil jumped {jump:.4f} K at a stage boundary; state was not carried"
        )


def test_running_the_fans_makes_the_unit_cooler_and_faster():
    """Stage 2 sheds more heat and responds faster, on both time constants.

    The time constants are the standard's own tabulated values for natural
    against forced air on a medium/large unit: tau_o 210 -> 150 min and
    tau_w 10 -> 7 min.
    """
    hot, cool = STAGED_TRUTH_PARAMS[1], STAGED_TRUTH_PARAMS[2]
    assert cool.delta_theta_or_K < hot.delta_theta_or_K
    assert cool.tau_o_min == 150.0 and hot.tau_o_min == 210.0
    assert cool.tau_w_min == 7.0 and hot.tau_w_min == 10.0


def test_single_stage_record_matches_the_unstaged_model():
    """With one stage throughout, staged simulation must equal the plain one.

    Guards the segmented integrator against introducing an offset when there
    is nothing to segment.
    """
    reference = truth_trajectory("A")
    t = reference.time_s
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    params = StagedThermalParams(per_stage={2: STAGED_TRUTH_PARAMS[2]})
    staged = simulate_staged(
        t, reference.load_pu, ambient, np.full(t.size, 2), params,
        load_pu_half=day_a_load(t + 0.5 * DT_S), ambient_C_half=ambient,
    )
    assert np.max(np.abs(staged.hotspot_C - reference.hotspot_C)) < 1e-9


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------


def test_joint_fit_recovers_every_stage(staged_day):
    """All seven free parameters back within 5 %, from one noisy day.

    Seven, not six: tau_w became per-stage when IEC 60076-7 Table 4 showed it
    differs between natural and forced air. The band widened from 2 % to 5 %
    with it, because the extra free parameter is the hardest of the four and
    stage 1's transients have to support their own copy of it. That is a
    re-derivation after a deliberate model correction, not a tolerance
    loosened to hide a failure -- the worst error is 4.4 %, on stage 1 tau_w.
    """
    result = _fit(staged_day)
    assert result.success
    for stage in (1, 2):
        truth = STAGED_TRUTH_PARAMS[stage]
        fitted = result.params.for_stage(stage)
        for name in ("delta_theta_or_K", "tau_o_min", "delta_theta_hr_K", "tau_w_min"):
            error = abs(getattr(fitted, name) - getattr(truth, name)) / getattr(truth, name)
            assert error < 0.05, f"stage {stage} {name} off by {error * 100:.2f} %"


def test_staged_fit_reproduces_the_trajectory(staged_day):
    result = _fit(staged_day)
    fitted = simulate_staged(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
        staged_day["stage"], result.params,
        load_pu_half=staged_day["load_half"], ambient_C_half=staged_day["ambient"],
    )
    error = fitted.hotspot_C - staged_day["traj"].hotspot_C
    assert_reproduces(float(np.sqrt(np.mean(error**2))), 0.128, "staged trajectory RMSE")


def test_ignoring_staging_is_catastrophic(staged_day):
    """THE test. One parameter set on a staged unit fails worse than Model A.

    4.96 K RMSE and +6.54 K at the peak, against 0.13 K and +0.19 K when the
    staging is modelled. For comparison, the single-exponential Model A that
    this package exists to beat reads +5.76 K at 1.30 pu.

    The failure mode matters commercially: it looks exactly like "the method
    does not transfer to real transformers", when the actual fault is one
    missing input column.
    """
    single = identify(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
        staged_day["oil"], staged_day["refs"], loss="linear",
        starts=[(30.0, 6000.0, 15.0, 600.0)],
        load_pu_half=staged_day["load_half"], ambient_C_half=staged_day["ambient"],
    )
    fitted = simulate(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"], single.params,
        load_pu_half=staged_day["load_half"], ambient_C_half=staged_day["ambient"],
    )
    error = fitted.hotspot_C - staged_day["traj"].hotspot_C
    rmse = float(np.sqrt(np.mean(error**2)))
    peak = float(fitted.hotspot_C.max() - staged_day["traj"].hotspot_C.max())

    assert_reproduces(rmse, 4.957, "unstaged RMSE on a staged unit")
    assert_reproduces(peak, 6.541, "unstaged peak error on a staged unit")
    assert rmse > 2.0, "must fail the pre-registered RMSE gate"
    assert peak > 5.7, "must be worse than Model A's +5.76 K at 1.30 pu"


def test_staged_fit_beats_the_unstaged_one_by_an_order_of_magnitude(staged_day):
    staged = _fit(staged_day)
    fitted = simulate_staged(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
        staged_day["stage"], staged.params,
        load_pu_half=staged_day["load_half"], ambient_C_half=staged_day["ambient"],
    )
    staged_rmse = float(np.sqrt(np.mean((fitted.hotspot_C - staged_day["traj"].hotspot_C) ** 2)))
    assert staged_rmse < 0.5


# --------------------------------------------------------------------------
# The shared-parameter contract
# --------------------------------------------------------------------------


def test_only_the_rated_gradient_is_shared(staged_day):
    """dtheta_hr is shared across stages. tau_w is NOT, and must not be.

    Sharing tau_w was this module's original default and it was wrong:
    Table 4 gives 10 min for natural air and 7 min for forced air on the same
    class of unit. The fit must therefore return a DIFFERENT tau_w per stage
    and an identical dtheta_hr.
    """
    result = _fit(staged_day)
    one, two = result.params.for_stage(1), result.params.for_stage(2)
    assert one.delta_theta_hr_K == pytest.approx(two.delta_theta_hr_K)
    assert one.tau_w_min != pytest.approx(two.tau_w_min)
    assert set(result.params.shared) == {"delta_theta_hr_K"}
    assert "tau_w_min" not in SHARED_BY_DEFAULT


def test_declaring_a_parameter_shared_when_it_differs_is_refused():
    """A 'shared' value that differs between stages misdescribes the model."""
    with pytest.raises(ValueError, match="declared shared"):
        StagedThermalParams(
            per_stage={
                1: ThermalParams(60.0, 210.0, 22.0, 7.0),
                2: ThermalParams(45.0, 150.0, 25.0, 7.0),  # dtheta_hr differs
            }
        )


def test_all_four_can_float_per_stage(staged_day):
    """Directed-flow units need the winding pair per stage too."""
    result = _fit(staged_day, shared=())
    assert result.success
    # The truth has them equal, so an unconstrained fit should still find
    # them close -- this checks the packing, not the physics.
    one, two = result.params.for_stage(1), result.params.for_stage(2)
    assert abs(one.delta_theta_hr_K - two.delta_theta_hr_K) < 2.0


def test_unknown_shared_parameter_is_refused():
    with pytest.raises(ValueError, match="unknown shared parameter"):
        StagedThermalParams(per_stage={1: ThermalParams(45.0, 150.0, 22.0, 7.0)},
                            shared=("not_a_parameter",))


def test_simulating_an_unfitted_stage_is_refused(staged_day):
    """A stage that was never identified cannot be simulated."""
    params = StagedThermalParams(per_stage={2: STAGED_TRUTH_PARAMS[2]})
    with pytest.raises(KeyError, match="cooling stage"):
        simulate_staged(
            staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
            staged_day["stage"], params,
        )


def test_empty_stage_map_is_refused():
    with pytest.raises(ValueError, match="at least one cooling stage"):
        StagedThermalParams(per_stage={})


def test_initial_state_is_honoured(staged_day):
    """A warm start must move the opening temperature, not be ignored."""
    params = StagedThermalParams(per_stage=STAGED_TRUTH_PARAMS)
    warm = simulate_staged(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
        staged_day["stage"], params, initial_state=InitialState(95.0, 1.1),
    )
    assert warm.top_oil_C[0] == pytest.approx(95.0)


# --------------------------------------------------------------------------
# Railing against the tau_w < tau_o structural constraint
#
# The box-bound check cannot see this one. ThermalParams requires tau_w <
# tau_o, and `residual` returns a flat penalty wherever that is violated,
# which turns the constraint into an invisible wall in the cost surface: the
# optimiser walks tau_w up to it and stops. Before these tests, a solution
# pressed against that wall was reported as converged and interior, and from
# some starts the fit died with a bare ValueError out of `_unpack` instead of
# the designed refusal.
#
# Found on a 360 MVA ODAF field record whose 10-minute log samples a 7-minute
# winding transient less than once per time constant: stage 3 came back with
# tau_w = tau_o - 1e-6 min and was accepted.
# --------------------------------------------------------------------------


def test_a_solution_pressed_against_the_tau_w_constraint_is_refused():
    """tau_w driven up to tau_o is the constraint talking, not the data.

    Starting the optimiser with the two constants equal puts it on the wall.
    Whatever it returns from there, it must not be a success.
    """
    trajectory, stage = staged_truth_trajectory("A")
    t = trajectory.time_s
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    oil = np.full(t.size, np.nan)
    oil_index = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    # A top-oil channel stuck at one value carries no rate information, so
    # nothing in the record can hold the two time constants apart.
    oil[oil_index] = 60.0
    cal_index = calibration_indices(17, t)
    refs = HotspotReferences(t[cal_index], np.full(cal_index.size, 75.0),
                             source="synthetic-degenerate")

    with pytest.raises(RuntimeError) as excinfo:
        identify_staged(
            t, trajectory.load_pu, ambient, oil, refs, stage,
            loss="linear", start=(30.0, 3000.0, 15.0, 3000.0),
        )
    assert "railed" in str(excinfo.value)


def test_the_constraint_check_names_the_stage_and_both_constants():
    from corefield.staged import STRUCTURAL_MARGIN, _build_packing, _unpack_raw

    slots = _build_packing((1, 2), ("delta_theta_hr_K",))
    # tau_w a hair under tau_o on stage 2: interior to every box bound, and
    # exactly the solution the field record produced.
    vector = np.zeros(len(slots))
    for i, (name, st) in enumerate(slots):
        vector[i] = {"delta_theta_or_K": 40.0, "tau_o_min": 6000.0,
                     "delta_theta_hr_K": 20.0, "tau_w_min": 600.0}[name]
        if name == "tau_w_min" and st == 2:
            vector[i] = 6000.0 * (1.0 - 1e-9)
    raw = _unpack_raw(vector, slots, (1, 2))
    assert raw[2][3] >= (1.0 - STRUCTURAL_MARGIN) * raw[2][1]
    assert raw[1][3] < (1.0 - STRUCTURAL_MARGIN) * raw[1][1]


# --------------------------------------------------------------------------
# Holding a parameter instead of chasing it
# --------------------------------------------------------------------------


def test_a_fixed_parameter_is_held_exactly_and_declared(staged_day):
    """A held parameter must come back untouched, and be reported as held.

    This is the honest route out of the constraint above: a 10-minute log
    cannot inform a 7-minute winding constant, so hold it at the IEC 60076-7
    Table 4 value and say so, rather than reporting whichever limit the
    optimiser reached as a measurement of the unit.
    """
    result = _fit(staged_day, fixed={"tau_w_min": 8.0})
    assert result.success
    for stage in (1, 2):
        assert result.params.for_stage(stage).tau_w_min == pytest.approx(8.0)
    assert result.fixed_parameters == {"tau_w_min": 8.0}
    assert "HELD, NOT IDENTIFIED" in result.report()
    assert "tau_w_min=8" in result.report()


def test_fixing_a_parameter_removes_its_slots_from_the_optimiser(staged_day):
    """Holding tau_w must drop both of its per-stage slots, not just pin them."""
    from corefield.staged import _build_packing

    free = _build_packing((1, 2), ("delta_theta_hr_K",))
    held = _build_packing((1, 2), ("delta_theta_hr_K",), ("tau_w_min",))
    assert len(free) - len(held) == 2
    assert not any(name == "tau_w_min" for name, _ in held)


def test_holding_tau_w_still_recovers_the_parameters_the_data_supports(staged_day):
    """The three identified parameters must not degrade when the fourth is held.

    Held at the truth value, so this isolates the mechanism from any error the
    held value itself introduces.
    """
    truth_tau_w = STAGED_TRUTH_PARAMS[1].tau_w_min
    result = _fit(staged_day, fixed={"tau_w_min": truth_tau_w})
    for stage in (1, 2):
        truth = STAGED_TRUTH_PARAMS[stage]
        fitted = result.params.for_stage(stage)
        for name in ("delta_theta_or_K", "tau_o_min", "delta_theta_hr_K"):
            error = abs(getattr(fitted, name) - getattr(truth, name)) / getattr(truth, name)
            assert error < 0.05, f"stage {stage} {name} off by {error * 100:.2f} %"


@pytest.mark.parametrize(
    "fixed, match",
    [
        ({"tau_z_min": 5.0}, "unknown parameter name"),
        ({"delta_theta_hr_K": 20.0}, "both `fixed` and `shared`"),
        ({"tau_w_min": 999.0}, "outside the physical plausibility bounds"),
    ],
)
def test_malformed_fixed_is_refused(staged_day, fixed, match):
    with pytest.raises(ValueError, match=match):
        _fit(staged_day, fixed=fixed)


def test_fixing_every_parameter_is_refused(staged_day):
    """Holding all four leaves nothing to identify -- that is a simulation.

    `shared` is emptied first so this trips the intended check rather than
    the fixed/shared overlap one.
    """
    with pytest.raises(ValueError, match="nothing to identify"):
        _fit(staged_day, shared=(), fixed={
            "delta_theta_or_K": 40.0, "tau_o_min": 100.0,
            "delta_theta_hr_K": 20.0, "tau_w_min": 7.0,
        })


# --------------------------------------------------------------------------
# Declared preconditions and the scope of a refusal
# --------------------------------------------------------------------------


def test_the_clean_cooler_precondition_appears_on_every_report(staged_day):
    """A fouled cooler is undetectable from the record, so it must be declared.

    External fouling raises the oil-to-air thermal resistance. Identify on a
    fouled cooler and the fouling is absorbed into `delta_theta_or` as though
    it were a property of the transformer; identify clean and apply fouled and
    the model under-predicts the hot spot, which is the unsafe direction. No
    channel here can tell the difference, so the assumption is stated on every
    report rather than checked.
    """
    report = _fit(staged_day).report()
    assert "PRECONDITION:" in report
    assert CLEAN_COOLER_PRECONDITION in report
    # The unsafe direction is the part a reader must not miss.
    assert "under-predict" in report


def test_the_precondition_survives_a_held_parameter(staged_day):
    """Holding a parameter must not displace the standing precondition."""
    report = _fit(staged_day, fixed={"tau_w_min": 7.0}).report()
    assert "HELD, NOT IDENTIFIED" in report
    assert CLEAN_COOLER_PRECONDITION in report


def test_a_railed_refusal_says_it_is_about_the_model_not_the_record():
    """The refusal is a statement about this parameterisation, not the data.

    The IEC two-exponential form is excited by load variation, so a near-flat
    load starves it. A model parameterised on cooling-plant fraction and oil
    viscosity draws excitation from stage switching and oil-temperature range
    instead, and can be identifiable on the very same record. Reporting a
    refusal here as "this record carries no thermal information" would be a
    stronger claim than the estimator is entitled to make.
    """
    t = np.arange(0.0, 6 * 3600.0, 60.0)
    stage = np.where(t < 3 * 3600.0, 1, 2)
    # Constant load: nothing for the exponentiated terms to work against.
    load = np.full(t.size, 0.80)
    ambient = np.full(t.size, 20.0)
    params = StagedThermalParams(per_stage={
        1: ThermalParams(delta_theta_or_K=40.0, tau_o_min=150.0,
                         delta_theta_hr_K=20.0, tau_w_min=7.0),
        2: ThermalParams(delta_theta_or_K=28.0, tau_o_min=100.0,
                         delta_theta_hr_K=20.0, tau_w_min=7.0),
    })
    traj = simulate_staged(t, load, ambient, stage, params)
    refs = HotspotReferences(
        time_s=t[::40][:8], temperature_C=traj.hotspot_C[::40][:8]
    )
    with pytest.raises(RuntimeError) as excinfo:
        identify_staged(t, load, ambient, traj.top_oil_C, refs, stage,
                        shared=(), loss="linear")
    message = str(excinfo.value)
    assert "THIS model form" in message
    assert "not about the" in message
    assert "carries no thermal information" in message
