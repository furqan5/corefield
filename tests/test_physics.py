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

"""Forward-model regression: truth anchors, integrator equivalence, unit safety.

Every value pinned here was independently re-derived on 24 Aug 2026 from the
legacy notebooks' own code, on NumPy 2.4.6 / SciPy 1.18.0 -- a newer stack
than the campaign's 2.4.4 / 1.17.1. They all reproduced exactly. The
`7.105e-14` style agreements are float64 round-off between algebraically
equivalent formulations and cannot be guessed: they are the strongest
evidence in the repository that the published tables came from this code.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_reproduces

from corefield.iec60076_7 import (
    ONAF_MEDIUM_LARGE_POWER,
    PARAM_BOUNDS,
    CoolingConstants,
    ThermalParams,
    _integrate,
    _integrate_reference,
    hotspot_temperature,
    simulate,
    steady_temperatures,
)
from corefield.synthetic import (
    AMBIENT_CONSTANT_C,
    DT_S,
    TRUTH_PARAMS,
    day_a_load,
    diurnal_ambient,
    time_grid,
    truth_trajectory,
)
from corefield.verification import verify_k_assignment


# --------------------------------------------------------------------------
# Structural verification of the k-assignment
# --------------------------------------------------------------------------


def test_k_assignment_verified():
    """The three structural checks pass. Verifies STRUCTURE, not Table-4 values."""
    check = verify_k_assignment()
    assert check.passed, check.report()
    assert check.closed_form_max_dev_K < 1e-3
    assert_reproduces(check.closed_form_max_dev_K * 1e7, 1.097, "closed-form dev (x1e7)")
    assert_reproduces(check.oil_fraction_at_k11_tau_o, 0.6321, "oil fraction at k11*tau_o")
    assert_reproduces(check.overshoot_fraction * 100, 47.19, "gradient overshoot %")
    assert_reproduces(check.overshoot_time_min, 41.0, "overshoot time (min)")


def test_overshoot_is_absent_without_slow_branch():
    """With k21 = 1 the slow branch vanishes and the overshoot disappears.

    This is the small-distribution-ONAN degeneracy the standard notes, and it
    is also a sanity check on the branch assignment: the overshoot must come
    from the (k21-1) branch and nowhere else.
    """
    onan_like = CoolingConstants(x=0.8, y=1.3, k11=0.5, k21=1.0, k22=2.0, name="test ONAN-like")
    check = verify_k_assignment(constants=onan_like)
    assert check.overshoot_fraction == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Truth-model anchors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "day, peak_C, peak_h, load_min, load_max",
    [
        ("A", 105.56, 15.93, 0.60, 1.20),
        ("B", 100.90, 18.92, 0.70, 1.15),
        ("C", 114.79, 16.95, 0.70, 1.30),
    ],
)
def test_truth_day_anchors(day, peak_C, peak_h, load_min, load_max):
    """Published peak hot-spot and load hull for each synthetic load day."""
    traj = truth_trajectory(day)
    assert_reproduces(traj.peak_hotspot_C, peak_C, f"day-{day} peak hot-spot")
    assert_reproduces(traj.peak_time_h, peak_h, f"day-{day} peak time")
    assert_reproduces(float(traj.load_pu.min()), load_min, f"day-{day} min load")
    assert_reproduces(float(traj.load_pu.max()), load_max, f"day-{day} max load")


def test_diurnal_ambient_raises_the_peak():
    """A +/- 6 K ambient wave lifts the day-A peak from 105.56 to 111.25 degC.

    The ambient maximum lands at 15:00, near the load peak -- deliberately
    adversarial, and the reason ignoring ambient is dangerous rather than
    merely inaccurate.
    """
    traj = truth_trajectory("A", ambient="diurnal")
    assert_reproduces(traj.peak_hotspot_C, 111.25, "diurnal-ambient peak")
    assert_reproduces(traj.peak_time_h, 15.93, "diurnal-ambient peak time")


def test_day_a_gradient_statistics():
    """Hot-spot-over-top-oil gap: max 32.02 K, mean 17.37 K on day A."""
    traj = truth_trajectory("A")
    assert_reproduces(float(traj.gradient_K.max()), 32.02, "day-A gap max")
    assert_reproduces(float(traj.gradient_K.mean()), 17.37, "day-A gap mean")


def test_diurnal_ambient_extremes():
    """The wave peaks at 15:00 and troughs at 03:00, +/- 6 K about 20 degC."""
    t = time_grid()
    ambient = diurnal_ambient(t)
    assert_reproduces(float(t[int(ambient.argmax())] / 3600.0), 15.0, "ambient max hour")
    assert_reproduces(float(t[int(ambient.argmin())] / 3600.0), 3.0, "ambient min hour")
    assert_reproduces(float(ambient.max()), 26.0, "ambient max")
    assert_reproduces(float(ambient.min()), 14.0, "ambient min")


def test_steady_state_is_the_settled_trajectory():
    """Held load must settle onto the closed-form steady state.

    The two-exponential branches cancel to dtheta_hr * K**y at equilibrium,
    so any disagreement here means the branch amplitudes are inconsistent.
    """
    t = time_grid(days=2.0)
    K = np.full(t.size, 0.9)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    traj = simulate(t, K, ambient, TRUTH_PARAMS, load_pu_half=K, ambient_C_half=ambient)
    oil_ss, hs_ss = steady_temperatures(0.9, AMBIENT_CONSTANT_C, TRUTH_PARAMS)
    assert float(traj.top_oil_C[-1]) == pytest.approx(float(oil_ss), abs=1e-6)
    assert float(traj.hotspot_C[-1]) == pytest.approx(float(hs_ss), abs=1e-6)


# --------------------------------------------------------------------------
# Integrator equivalence -- the fast path must not drift
# --------------------------------------------------------------------------


def test_fast_integrator_matches_reference_loop():
    """The lfilter recurrence is an exact restatement of the explicit RK4 loop.

    `_integrate` collapses RK4 into a one-pole recurrence for speed (~44x).
    If that optimisation ever drifts from the literal loop the campaign's
    numbers move silently, so the equivalence is pinned here.
    """
    t = time_grid()
    K = day_a_load(t)
    K_half = day_a_load(t + 0.5 * DT_S)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    kwargs = dict(
        dtor=TRUTH_PARAMS.delta_theta_or_K, tauo_s=TRUTH_PARAMS.tau_o_s,
        dthr=TRUTH_PARAMS.delta_theta_hr_K, tw_s=TRUTH_PARAMS.tau_w_s,
        x=ONAF_MEDIUM_LARGE_POWER.x, y=ONAF_MEDIUM_LARGE_POWER.y,
        k11=ONAF_MEDIUM_LARGE_POWER.k11, k21=ONAF_MEDIUM_LARGE_POWER.k21,
        k22=ONAF_MEDIUM_LARGE_POWER.k22, R=TRUTH_PARAMS.loss_ratio_R,
        K_on=K, K_half=K_half, A_on=ambient, A_half=ambient, dt=DT_S,
    )
    oil_ref, hs_ref = _integrate_reference(**kwargs)
    oil_fast, hs_fast = _integrate(**kwargs, solver="rk4")
    assert np.max(np.abs(oil_ref - oil_fast)) < 1e-10
    assert np.max(np.abs(hs_ref - hs_fast)) < 1e-10


def test_euler_is_available_but_differs_from_rk4():
    """The standard's difference-equation form exists, and is NOT the default.

    Euler is offered for comparison against the standard's own presentation.
    It must differ from RK4 -- if it did not, the solver switch would be a
    no-op and the reproduction path would be ambiguous.
    """
    t = time_grid()
    K = day_a_load(t)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    rk4 = simulate(t, K, ambient, TRUTH_PARAMS, solver="rk4")
    euler = simulate(t, K, ambient, TRUTH_PARAMS, solver="euler")
    assert np.max(np.abs(rk4.hotspot_C - euler.hotspot_C)) > 1e-6
    # ...but at dt = 30 s against tau_w = 420 s the two must stay close.
    assert np.max(np.abs(rk4.hotspot_C - euler.hotspot_C)) < 0.5


# --------------------------------------------------------------------------
# Unit consistency -- feeding kelvin where Celsius is expected must raise
# --------------------------------------------------------------------------


def test_kelvin_ambient_is_rejected():
    """Ambient in kelvin must raise, not silently shift the model by 273 K."""
    t = time_grid()
    K = day_a_load(t)
    ambient_kelvin = np.full(t.size, AMBIENT_CONSTANT_C + 273.15)
    with pytest.raises(ValueError, match="kelvin"):
        simulate(t, K, ambient_kelvin, TRUTH_PARAMS)


def test_kelvin_scalar_ambient_is_rejected():
    """The same check applies to a scalar ambient."""
    t = time_grid()
    with pytest.raises(ValueError, match="kelvin"):
        hotspot_temperature(t, day_a_load(t), 293.15, TRUTH_PARAMS)


def test_celsius_ambient_is_accepted():
    """A plausible Celsius ambient must NOT trip the kelvin guard."""
    t = time_grid()
    traj = simulate(t, day_a_load(t), 45.0, TRUTH_PARAMS)
    assert np.all(np.isfinite(traj.hotspot_C))


def test_non_uniform_grid_is_rejected():
    """Fixed-step RK4 on a non-uniform grid would be silently wrong."""
    t = time_grid()
    t_bad = t.copy()
    t_bad[100] += 5.0
    with pytest.raises(ValueError, match="uniformly spaced"):
        simulate(t_bad, day_a_load(t), AMBIENT_CONSTANT_C, TRUTH_PARAMS)


def test_too_coarse_grid_is_rejected():
    """A step above tau_w/2 numerically damps the transient. Refuse it."""
    t = np.arange(0.0, 24 * 3600.0 + 600.0, 600.0)  # 10 min steps vs tau_w = 7 min
    with pytest.raises(ValueError, match="winding time constant"):
        simulate(t, day_a_load(t), AMBIENT_CONSTANT_C, TRUTH_PARAMS)


def test_negative_load_is_rejected():
    t = time_grid()
    K = day_a_load(t)
    K[10] = -0.1
    with pytest.raises(ValueError, match="negative"):
        simulate(t, K, AMBIENT_CONSTANT_C, TRUTH_PARAMS)


def test_nan_load_is_rejected():
    t = time_grid()
    K = day_a_load(t)
    K[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        simulate(t, K, AMBIENT_CONSTANT_C, TRUTH_PARAMS)


# --------------------------------------------------------------------------
# ThermalParams validation
# --------------------------------------------------------------------------


def test_params_reject_out_of_bounds():
    """Physically implausible parameters are refused, not clamped."""
    with pytest.raises(ValueError, match="outside the physical plausibility bounds"):
        ThermalParams(
            delta_theta_or_K=500.0, tau_o_min=150.0,
            delta_theta_hr_K=22.0, tau_w_min=7.0,
        )


def test_params_reject_inverted_time_constants():
    """tau_w >= tau_o means the branch labels have swapped meaning."""
    with pytest.raises(ValueError, match="fast"):
        ThermalParams(
            delta_theta_or_K=45.0, tau_o_min=50.0,
            delta_theta_hr_K=22.0, tau_w_min=60.0,
        )


def test_params_round_trip_through_vector():
    """as_vector/from_vector must be exactly inverse -- the optimiser relies on it."""
    restored = ThermalParams.from_vector(TRUTH_PARAMS.as_vector())
    assert restored.delta_theta_or_K == pytest.approx(TRUTH_PARAMS.delta_theta_or_K)
    assert restored.tau_o_min == pytest.approx(TRUTH_PARAMS.tau_o_min)
    assert restored.delta_theta_hr_K == pytest.approx(TRUTH_PARAMS.delta_theta_hr_K)
    assert restored.tau_w_min == pytest.approx(TRUTH_PARAMS.tau_w_min)


def test_vector_units_are_seconds_not_minutes():
    """A silent minutes/seconds swap would move tau_o by 60x. Pin the convention."""
    vector = TRUTH_PARAMS.as_vector()
    assert vector[1] == pytest.approx(150.0 * 60.0)
    assert vector[3] == pytest.approx(7.0 * 60.0)


def test_settled_constants_are_unchanged():
    """ONAF constants are settled per CLAUDE.md. Changing one must break a test."""
    c = ONAF_MEDIUM_LARGE_POWER
    assert (c.x, c.y, c.k11, c.k21, c.k22) == (0.8, 1.3, 0.5, 2.0, 2.0)


def test_param_bounds_cover_the_synthetic_unit():
    """The plausibility bounds must not exclude the unit the campaign used."""
    for name, (low, high) in PARAM_BOUNDS.items():
        value = getattr(TRUTH_PARAMS, name)
        assert low <= value <= high, f"{name}={value} outside [{low}, {high}]"
