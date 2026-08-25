# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staged cooling: per-stage parameters, segmented integration, joint fit.

The test that matters most here is `test_ignoring_staging_is_catastrophic`.
Unmodelled cooling staging costs +6.7 K at the peak — worse than the
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


def test_running_the_fans_makes_the_unit_cooler():
    """Sanity: the whole point of stage 2 is that it sheds more heat."""
    hot = STAGED_TRUTH_PARAMS[1]
    cool = STAGED_TRUTH_PARAMS[2]
    assert cool.delta_theta_or_K < hot.delta_theta_or_K
    assert cool.tau_o_min < hot.tau_o_min


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
    """All six free parameters back within 2 %, from one noisy day."""
    result = _fit(staged_day)
    assert result.success
    for stage in (1, 2):
        truth = STAGED_TRUTH_PARAMS[stage]
        fitted = result.params.for_stage(stage)
        for name in ("delta_theta_or_K", "tau_o_min", "delta_theta_hr_K", "tau_w_min"):
            error = abs(getattr(fitted, name) - getattr(truth, name)) / getattr(truth, name)
            assert error < 0.02, f"stage {stage} {name} off by {error * 100:.2f} %"


def test_staged_fit_reproduces_the_trajectory(staged_day):
    result = _fit(staged_day)
    fitted = simulate_staged(
        staged_day["t"], staged_day["traj"].load_pu, staged_day["ambient"],
        staged_day["stage"], result.params,
        load_pu_half=staged_day["load_half"], ambient_C_half=staged_day["ambient"],
    )
    error = fitted.hotspot_C - staged_day["traj"].hotspot_C
    assert_reproduces(float(np.sqrt(np.mean(error**2))), 0.113, "staged trajectory RMSE")


def test_ignoring_staging_is_catastrophic(staged_day):
    """THE test. One parameter set on a staged unit fails worse than Model A.

    4.93 K RMSE and +6.70 K at the peak, against 0.11 K and +0.22 K when the
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

    assert_reproduces(rmse, 4.934, "unstaged RMSE on a staged unit")
    assert_reproduces(peak, 6.703, "unstaged peak error on a staged unit")
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


def test_shared_parameters_are_equal_across_stages(staged_day):
    """The winding pair is shared by default, so it must come back identical."""
    result = _fit(staged_day)
    one, two = result.params.for_stage(1), result.params.for_stage(2)
    assert one.delta_theta_hr_K == pytest.approx(two.delta_theta_hr_K)
    assert one.tau_w_min == pytest.approx(two.tau_w_min)
    assert set(result.params.shared) == set(SHARED_BY_DEFAULT)


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
