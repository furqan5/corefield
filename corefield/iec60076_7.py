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

"""Two-exponential transformer thermal model, IEC 60076-7 structure.

============================================================================
IEC PROVENANCE -- MIRROR-SOURCED, UNVERIFIED AGAINST A LICENSED COPY

(a, project provenance) The earlier 25 Aug 2026 check used mirror-sourced
text. It does not satisfy the project's licensed-source verification gate.
Retain this warning until an authorised copy has been checked and recorded.
The settled ONAF constants are unchanged; they are not a compliance claim.
This repository does not redistribute the standard's text, tables or figures.

(a, implementation tests) `corefield.verification` checks numerical
consistency of the implemented branch assignment, including closed-form/RK4
agreement and step responses. Self-consistency is not independent evidence
of standards compliance or physical validity on a transformer.
============================================================================

Model
-----
Three states, all in kelvin:

    theta_o      absolute top-oil temperature            [degC]
    dtheta_h1    fast (winding) gradient branch          [K]
    dtheta_h2    slow (oil-side) gradient branch         [K]

    d(theta_o)/dt  = (dtheta_or * f(K) + theta_a(t) - theta_o) / (k11 * tau_o)
    d(dtheta_h1)/dt = (k21 * dtheta_hr * K**y - dtheta_h1) / (k22 * tau_w)
    d(dtheta_h2)/dt = ((k21 - 1) * dtheta_hr * K**y - dtheta_h2) / (tau_o / k22)

    f(K) = ((1 + R * K**2) / (1 + R)) ** x
    theta_h(t) = theta_o(t) + dtheta_h1(t) - dtheta_h2(t)

Ambient enters through the oil equation only, and therefore reaches the hot
spot through the k11*tau_o = 75 min oil low-pass. Integrating theta_o in
ABSOLUTE form (rather than as a rise over ambient) is what makes a
time-varying ambient correct; for constant ambient the two forms agree to
8.5e-14 K, which is float64 round-off.

Difference-equation form
------------------------
The standard presents these as difference equations, which is forward Euler.
This module integrates with classical RK4 on a uniform grid instead. Two
reasons, both deliberate:

  1. Accuracy. At dt = 30 s against tau_w = 420 s, Euler carries a visible
     local truncation error at load transitions; RK4 does not. The published
     campaign used RK4 and its closed-form check passes at 1.10e-7 K.
  2. Reproduction. Every published CoreField number was produced by RK4 at
     dt = 30 s. Switching integrator would silently move the numbers the
     regression suite exists to pin.

`solver="euler"` is available for comparison against the standard's own
presentation, but it is NOT the reproduction path and is not what the
regression tests pin.

Units
-----
SI throughout, with two documented exceptions carried for engineering
readability and flagged in every signature:
    - temperatures in degrees Celsius (degC), temperature *differences* and
      *rises* in kelvin (K)
    - time constants in MINUTES in the public API (`ThermalParams`), because
      that is how nameplate and heat-run data are quoted; converted to
      seconds internally at exactly one place.
Load is per-unit (pu) of rated current, dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:  # SciPy carries the C-speed one-pole recurrence used by the fast path.
    from scipy.signal import lfilter as _lfilter
except ImportError:  # pragma: no cover - keeps the physics usable without SciPy
    _lfilter = None  # type: ignore[assignment]

__all__ = [
    "CoolingConstants",
    "ONAF_MEDIUM_LARGE_POWER",
    "ONAN_MEDIUM_LARGE_POWER",
    "ONAN_SMALL",
    "OD_MEDIUM_LARGE_POWER",
    "ThermalParams",
    "PARAM_BOUNDS",
    "oil_exponent",
    "winding_exponent",
    "InitialState",
    "ThermalTrajectory",
    "steady_top_oil_rise",
    "steady_hotspot_gradient",
    "steady_temperatures",
    "top_oil_rise",
    "hotspot_rise",
    "hotspot_temperature",
    "simulate",
]

Solver = Literal["rk4", "euler"]


# --------------------------------------------------------------------------
# Cooling-class constants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CoolingConstants:
    """Empirical exponents and time-constant multipliers for one cooling class.

    All dimensionless. The constants remain mirror-sourced and UNVERIFIED
    against a licensed standard; see the module provenance banner.

    Attributes
    ----------
    x : oil exponent, applied to the loss ratio in f(K)      [-]
    y : winding exponent, applied to K in the gradient drive [-]
    k11 : multiplier on tau_o in the oil equation            [-]
    k21 : amplitude split between the two gradient branches  [-]
    k22 : multiplier on tau_w (fast branch) and divisor on
          tau_o (slow branch)                                [-]
    name : human-readable cooling-class label
    """

    x: float
    y: float
    k11: float
    k21: float
    k22: float
    name: str
    #: Load-slope of the oil exponent, per unit of load: x(K) = x + x1*(K-1).
    #: Zero reproduces the standard's fixed exponent exactly, and is the
    #: default, so nothing changes unless a caller asks for it.
    #:
    #: (b) Experimental extension motivated by exploratory fits to the ONAF
    #: test in Nordman and Lahtinen (2003), DOI 10.1109/TPWRD.2002.807747.
    #: Calculated interval exponents are not direct measurements of this x1.
    #: Neither their magnitude nor a physical cause is established across
    #: other units or cooling classes. The caller supplies the slope; the
    #: production four-parameter estimator does not identify it automatically.
    x1: float = 0.0
    #: Load-slope of the winding exponent: y(K) = y + y1*(K-1). Same rationale.
    y1: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "k11", "k21", "k22"):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"CoolingConstants.{field_name} must be finite and > 0, got {value!r}"
                )
        # The slopes may be zero or negative -- a cooling system can plausibly
        # become relatively more effective with load -- so only finiteness is
        # required of them.
        for field_name in ("x1", "y1"):
            value = getattr(self, field_name)
            if not np.isfinite(value):
                raise ValueError(
                    f"CoolingConstants.{field_name} must be finite, got {value!r}"
                )
        if self.k21 < 1.0:
            # k21 < 1 would make the slow branch amplitude (k21-1)*dthr negative,
            # inverting the overshoot into an undershoot -- unphysical.
            raise ValueError(f"k21 must be >= 1.0, got {self.k21!r}")


#: Medium & large power transformers, ONAF. Settled per CLAUDE.md -- do not
#: change without an explicit instruction. Mirror-sourced; UNVERIFIED against
#: a licensed standard. The 25 Aug 2026 mirror check does not close that gate.
#:
#: Note on portability (label (c), from methods v4 section 9.3): other cooling
#: classes are a column swap. Small distribution ONAN has k21 = 1.0, which
#: makes the slow-branch amplitude (k21-1)*dtheta_hr vanish identically, and
#: Model C degenerates to overshoot-free single-exponential behaviour.
ONAF_MEDIUM_LARGE_POWER = CoolingConstants(
    x=0.8, y=1.3, k11=0.5, k21=2.0, k22=2.0, name="ONAF, medium & large power"
)

#: Natural air over the same size of unit -- the other half of a two-stage
#: ON../ON.F transformer. The exponents and k-constants are unchanged from
#: ONAF; only the two time constants differ (tau_o 210 vs 150 min, tau_w 10
#: vs 7 min), and those live in ThermalParams rather than here.
ONAN_MEDIUM_LARGE_POWER = CoolingConstants(
    x=0.8, y=1.3, k11=0.5, k21=2.0, k22=2.0, name="ONAN, medium & large power"
)

#: Small distribution transformers, natural air.
#:
#: k21 = 1.0 makes the slow-branch amplitude (k21-1)*dtheta_hr vanish
#: identically, so the two-exponential model collapses to overshoot-free
#: single-exponential behaviour. This is why small units show no hot-spot
#: overshoot on a load step, and why the A/B/C model comparison has nothing
#: to separate on them.
ONAN_SMALL = CoolingConstants(
    x=0.8, y=1.6, k11=1.0, k21=1.0, k22=2.0, name="ONAN, small transformers"
)

#: Directed oil flow (OD..., including ODAF).
#:
#: READ THIS BEFORE INTERPRETING ANY OD RESULT. k21 = 1.0, exactly as for
#: small units, so the slow branch vanishes and the transient hot-spot
#: overshoot disappears. The two-exponential structure degenerates to
#: single-exponential behaviour, and the day-C separation between Models A,
#: B and C -- +6.17 K against +0.32 K at 1.30 pu -- does NOT transfer to a
#: directed-flow unit. That result belongs to ON.. classes where k21 = 2.0.
#:
#: The exponents also differ sharply from ONAF: x = 1.0 rather than 0.8, and
#: y = 2.0 rather than 1.3. A model fitted with ONAF constants to an OD unit
#: is wrong in the drive terms, not merely mistuned.
OD_MEDIUM_LARGE_POWER = CoolingConstants(
    x=1.0, y=2.0, k11=1.0, k21=1.0, k22=1.0, name="OD (directed oil flow)"
)


# --------------------------------------------------------------------------
# Identified parameters
# --------------------------------------------------------------------------

#: Physical plausibility bounds for the four identified parameters, as
#: (low, high) in the units of the corresponding ThermalParams field.
#:
#: Provenance, stated honestly (label (b)/(c)):
#:   dtheta_or, tau_o, dtheta_hr  -- the optimiser bounds used throughout the
#:     published campaign (v4 cells 20/22), which bracket the ONAF-scale
#:     illustrative parameter set with wide margin.
#:   tau_w  -- widened from 60 min to 120 min midway through the campaign,
#:     when the truth model moved to the IEC two-exponential structure. The
#:     honest reason is NOT a physical argument: under structural mismatch a
#:     single-exponential model's effective tau_w can run high, and a 60 min
#:     ceiling would have railed it, producing a bounded-solution artifact
#:     rather than a result. See AUDIT.md section 3.1.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "delta_theta_or_K": (10.0, 90.0),
    "tau_o_min": (30.0, 600.0),
    "delta_theta_hr_K": (5.0, 60.0),
    "tau_w_min": (1.0, 120.0),
}


@dataclass(frozen=True)
class ThermalParams:
    """The four per-unit thermal parameters this package identifies.

    These are exactly the quantities IEC 60076-7 says can otherwise only be
    obtained from a prolonged heat-run test on a transformer fitted with
    fibre-optic sensors.

    Parameters
    ----------
    delta_theta_or_K : rated top-oil rise over ambient, at rated load  [K]
    tau_o_min : oil time constant                                      [min]
    delta_theta_hr_K : rated hot-spot rise over top-oil                [K]
    tau_w_min : winding time constant                                  [min]
    loss_ratio_R : load loss / no-load loss at rated load              [-]
        NOT identified by this package -- held fixed, taken from the
        nameplate or the heat-run certificate. Carried here because the
        physics needs it. Default 6.0 is the illustrative ONAF-scale value
        used throughout the synthetic campaign; it is not a real unit.

    Raises
    ------
    ValueError
        If any value is non-finite, non-positive, or outside PARAM_BOUNDS.
    """

    delta_theta_or_K: float
    tau_o_min: float
    delta_theta_hr_K: float
    tau_w_min: float
    loss_ratio_R: float = 6.0

    def __post_init__(self) -> None:
        for field_name, (low, high) in PARAM_BOUNDS.items():
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"ThermalParams.{field_name} must be finite, got {value!r}")
            if not (low <= value <= high):
                raise ValueError(
                    f"ThermalParams.{field_name} = {value!r} is outside the physical "
                    f"plausibility bounds [{low}, {high}]. If this is a real unit whose "
                    f"value genuinely lies outside, widen PARAM_BOUNDS deliberately and "
                    f"record why -- do not silently clamp."
                )
        if not np.isfinite(self.loss_ratio_R) or self.loss_ratio_R <= 0.0:
            raise ValueError(
                f"ThermalParams.loss_ratio_R must be finite and > 0, got {self.loss_ratio_R!r}"
            )
        # A winding faster than the oil is the physical ordering this two-
        # exponential structure assumes (the fast branch is the winding). If
        # tau_w exceeds tau_o the branch labels have swapped meaning and every
        # downstream interpretation is wrong, so refuse rather than warn.
        if self.tau_w_min >= self.tau_o_min:
            raise ValueError(
                f"tau_w_min ({self.tau_w_min}) must be < tau_o_min ({self.tau_o_min}): "
                f"the two-exponential structure assumes the winding branch is the fast "
                f"one. Equal or inverted time constants mean the branch assignment is "
                f"wrong, not merely unusual."
            )

    # -- convenience ------------------------------------------------------

    @property
    def tau_o_s(self) -> float:
        """Oil time constant [s]."""
        return self.tau_o_min * 60.0

    @property
    def tau_w_s(self) -> float:
        """Winding time constant [s]."""
        return self.tau_w_min * 60.0

    def as_vector(self) -> NDArray[np.float64]:
        """Pack as [dtheta_or (K), tau_o (s), dtheta_hr (K), tau_w (s)].

        Seconds, not minutes -- this is the optimiser's working vector and the
        ordering the published campaign used. Do not reorder.
        """
        return np.array(
            [self.delta_theta_or_K, self.tau_o_s, self.delta_theta_hr_K, self.tau_w_s],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(
        cls, vector: NDArray[np.float64], loss_ratio_R: float = 6.0
    ) -> "ThermalParams":
        """Unpack [dtheta_or (K), tau_o (s), dtheta_hr (K), tau_w (s)]."""
        if np.shape(vector) != (4,):
            raise ValueError(f"expected a length-4 vector, got shape {np.shape(vector)}")
        return cls(
            delta_theta_or_K=float(vector[0]),
            tau_o_min=float(vector[1]) / 60.0,
            delta_theta_hr_K=float(vector[2]),
            tau_w_min=float(vector[3]) / 60.0,
            loss_ratio_R=loss_ratio_R,
        )

    def replace(self, **changes: float) -> "ThermalParams":
        """Return a copy with fields replaced, re-validated."""
        return replace(self, **changes)


@dataclass(frozen=True)
class InitialState:
    """The unit's thermal state at the start of a simulation.

    By default `simulate` starts at thermal equilibrium for the opening load
    and ambient, which is right for a synthetic study and wrong for an
    operational question. A dispatcher asking "how much more can this unit
    take?" is asking about a transformer that is already warm, and starting
    it from equilibrium would either flatter or penalise the answer
    depending on which way the unit was drifting.

    Both fields are things a control room actually knows.

    Attributes
    ----------
    top_oil_C : measured top-oil temperature right now [degC]
    prior_load_pu : the load the unit has been carrying [pu]. Sets the two
        winding gradient branches, which are assumed settled at that load --
        reasonable because the fast branch settles in ~15 min and the slow
        one tracks the oil.
    """

    top_oil_C: float
    prior_load_pu: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.top_oil_C):
            raise ValueError(f"top_oil_C must be finite, got {self.top_oil_C!r}")
        if self.top_oil_C > 200.0:
            raise ValueError(
                f"top_oil_C = {self.top_oil_C} looks like kelvin; this API takes degrees "
                f"Celsius."
            )
        if not np.isfinite(self.prior_load_pu) or self.prior_load_pu < 0.0:
            raise ValueError(
                f"prior_load_pu must be finite and >= 0, got {self.prior_load_pu!r}"
            )


@dataclass(frozen=True)
class ThermalTrajectory:
    """Result of a forward simulation on a uniform time grid.

    Attributes
    ----------
    time_s : uniform time grid                                    [s]
    load_pu : load current driving the simulation                 [pu]
    ambient_C : ambient temperature                               [degC]
    top_oil_C : top-oil temperature                               [degC]
    hotspot_C : winding hot-spot temperature                      [degC]
    gradient_K : hot-spot rise over top-oil, = hotspot - top_oil  [K]
    """

    time_s: NDArray[np.float64]
    load_pu: NDArray[np.float64]
    ambient_C: NDArray[np.float64]
    top_oil_C: NDArray[np.float64]
    hotspot_C: NDArray[np.float64]
    gradient_K: NDArray[np.float64]

    @property
    def peak_hotspot_C(self) -> float:
        """Maximum hot-spot temperature over the trajectory [degC]."""
        return float(self.hotspot_C.max())

    @property
    def peak_time_h(self) -> float:
        """Time of the hot-spot maximum [h]."""
        return float(self.time_s[int(self.hotspot_C.argmax())] / 3600.0)


# --------------------------------------------------------------------------
# Steady state
# --------------------------------------------------------------------------


def oil_exponent(
    load_pu: NDArray[np.float64] | float, constants: CoolingConstants
) -> NDArray[np.float64]:
    """The oil exponent at a given load [-]: x(K) = x + x1*(K-1).

    Constant at the tabulated value unless `constants.x1` is non-zero.
    """
    K = np.asarray(load_pu, dtype=np.float64)
    return constants.x + constants.x1 * (K - 1.0)


def winding_exponent(
    load_pu: NDArray[np.float64] | float, constants: CoolingConstants
) -> NDArray[np.float64]:
    """The winding exponent at a given load [-]: y(K) = y + y1*(K-1)."""
    K = np.asarray(load_pu, dtype=np.float64)
    return constants.y + constants.y1 * (K - 1.0)


def _loss_factor(
    load_pu: NDArray[np.float64] | float, params: ThermalParams, constants: CoolingConstants
) -> NDArray[np.float64]:
    """f(K) = ((1 + R*K^2) / (1 + R))^x(K)  [-], vectorised over load."""
    K = np.asarray(load_pu, dtype=np.float64)
    R = params.loss_ratio_R
    return ((1.0 + R * K**2) / (1.0 + R)) ** oil_exponent(K, constants)


def steady_top_oil_rise(
    load_pu: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
) -> NDArray[np.float64]:
    """Steady-state top-oil rise over ambient if the load were held [K].

    Parameters
    ----------
    load_pu : per-unit load current [pu], scalar or array
    params : identified thermal parameters
    constants : cooling-class constants

    Returns
    -------
    Top-oil rise over ambient [K], same shape as `load_pu`.
    """
    return params.delta_theta_or_K * _loss_factor(load_pu, params, constants)


def steady_hotspot_gradient(
    load_pu: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
) -> NDArray[np.float64]:
    """Steady-state hot-spot rise over top-oil if the load were held [K].

    At steady state the two exponential branches settle to k21*g and
    (k21-1)*g, whose difference is g = dtheta_hr * K^y. The overshoot is a
    purely transient phenomenon and does not appear here.
    """
    K = np.asarray(load_pu, dtype=np.float64)
    return params.delta_theta_hr_K * K ** winding_exponent(K, constants)


def steady_temperatures(
    load_pu: NDArray[np.float64] | float,
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Steady-state (top-oil, hot-spot) temperatures if the load were held.

    Parameters
    ----------
    load_pu : per-unit load current [pu]
    ambient_C : ambient temperature [degC] -- Celsius, NOT kelvin
    params, constants : as above

    Returns
    -------
    (top_oil_C, hotspot_C), both [degC].
    """
    theta_a = np.asarray(ambient_C, dtype=np.float64)
    oil = theta_a + steady_top_oil_rise(load_pu, params, constants)
    return oil, oil + steady_hotspot_gradient(load_pu, params, constants)


# --------------------------------------------------------------------------
# Forward integration
# --------------------------------------------------------------------------


def _rk4_affine_coefficients(
    tau_s: float,
    dt: float,
    drive_on: NDArray[np.float64],
    drive_half: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Collapse one RK4 step of ds/dt = (D(t) - s)/tau into s -> a*s + b.

    Each of the three states in this model obeys a first-order LINEAR ODE
    with a constant coefficient and a fully known forcing D(t) -- the
    nonlinearity in K enters only through D, which is computed before
    integration begins. An RK4 step of such an ODE is therefore an AFFINE
    map in s, and its multiplier `a` is the same at every step.

    That fact is what makes the fast path possible: the whole trajectory is
    a first-order linear recurrence, which `scipy.signal.lfilter` evaluates
    in C rather than in a Python loop.

    `a` and `b` are obtained by evaluating the RK4 stages rather than by
    expanding them symbolically -- same arithmetic, no algebra to get wrong.

    Returns
    -------
    (a, b) where b has one entry per step (length n-1 is used).
    """
    inv_tau = 1.0 / tau_s
    half, sixth = 0.5 * dt, dt / 6.0

    # b: the response to the forcing from a zero initial state.
    d0, dh, d1 = drive_on[:-1], drive_half[:-1], drive_on[1:]
    k1 = d0 * inv_tau
    k2 = (dh - half * k1) * inv_tau
    k3 = (dh - half * k2) * inv_tau
    k4 = (d1 - dt * k3) * inv_tau
    b = sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # a: the response to a unit initial state with zero forcing.
    s = 1.0
    a1 = -s * inv_tau
    a2 = -(s + half * a1) * inv_tau
    a3 = -(s + half * a2) * inv_tau
    a4 = -(s + dt * a3) * inv_tau
    a = s + sixth * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

    return float(a), b


def _linear_recurrence(a: float, b: NDArray[np.float64], s0: float, n: int) -> NDArray[np.float64]:
    """Evaluate s[i+1] = a*s[i] + b[i] with s[0] = s0, returning all n samples.

    Uses scipy.signal.lfilter, which is the same recurrence expressed as a
    one-pole IIR filter and runs in C. Falls back to a Python loop if SciPy
    is unavailable, so the physics module keeps working without it.
    """
    if _lfilter is None:  # pragma: no cover - SciPy is a hard dependency in practice
        out = np.empty(n, dtype=np.float64)
        s = s0
        out[0] = s
        for i in range(n - 1):
            s = a * s + b[i]
            out[i + 1] = s
        return out

    driving = np.empty(n, dtype=np.float64)
    driving[0] = s0
    driving[1:] = b[: n - 1]
    return np.asarray(_lfilter([1.0], [1.0, -a], driving), dtype=np.float64)


def _integrate(
    dtor: float,
    tauo_s: float,
    dthr: float,
    tw_s: float,
    x: float,
    y: float,
    k11: float,
    k21: float,
    k22: float,
    R: float,
    K_on: NDArray[np.float64],
    K_half: NDArray[np.float64],
    A_on: NDArray[np.float64],
    A_half: NDArray[np.float64],
    dt: float,
    solver: Solver,
    initial_state: "InitialState | None" = None,
    raw_initial: tuple[float, float, float] | None = None,
    return_state: bool = False,
    x1: float = 0.0,
    y1: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[
    NDArray[np.float64], NDArray[np.float64], tuple[float, float, float]
]:
    """Core integrator. Returns (top_oil_C, hotspot_C).

    `raw_initial` sets the three internal states directly as
    (top_oil_C, fast_branch_K, slow_branch_K), overriding `initial_state`.
    `return_state` additionally returns their final values. Both exist for
    staged cooling: when fans switch, the parameters change but the
    transformer's stored heat does not, so a segmented integration has to
    hand the exact internal state across the boundary. Only the DIFFERENCE
    of the two gradient branches is observable, so the pair cannot be
    recovered from the output and must be carried explicitly.

    Performance note. The obvious implementation -- a Python loop carrying a
    3-vector through RK4 -- costs about 1.5 s per identification fit, which
    puts the full regression campaign well past any reasonable test budget.
    Because all three states are linear with known forcing (see
    `_rk4_affine_coefficients`), the loop collapses to three independent
    one-pole recurrences evaluated in C, ~40x faster.

    This is an exact restatement of the same RK4 scheme, not an
    approximation of it. `tests/test_physics.py` pins the fast path against
    an explicit reference loop so the equivalence cannot silently drift.
    """
    n = K_on.size
    top_oil = np.empty(n, dtype=np.float64)
    hotspot = np.empty(n, dtype=np.float64)

    # Precompute the two load-dependent drive terms. Same values as computing
    # them inside the loop, just not 4n times over.
    # Exponents may vary with load. x1 = y1 = 0 gives the fixed-exponent form
    # exactly, which is the default everywhere.
    fK_on = ((1.0 + R * K_on**2) / (1.0 + R)) ** (x + x1 * (K_on - 1.0))
    fK_half = ((1.0 + R * K_half**2) / (1.0 + R)) ** (x + x1 * (K_half - 1.0))
    Ky_on = K_on ** (y + y1 * (K_on - 1.0))
    Ky_half = K_half ** (y + y1 * (K_half - 1.0))

    inv_tau_oil = 1.0 / (k11 * tauo_s)
    inv_tau_fast = 1.0 / (k22 * tw_s)
    inv_tau_slow = 1.0 / (tauo_s / k22)
    amp_fast = k21 * dthr
    amp_slow = (k21 - 1.0) * dthr

    # Initial condition: thermal equilibrium at the opening load and ambient,
    # unless the caller supplies the unit's actual current state.
    if raw_initial is not None:
        s0, s1, s2 = (float(v) for v in raw_initial)
    elif initial_state is None:
        s0 = float(A_on[0] + dtor * fK_on[0])
        s1 = float(amp_fast * Ky_on[0])
        s2 = float(amp_slow * Ky_on[0])
    else:
        prior_K = float(initial_state.prior_load_pu)
        prior_Ky = prior_K ** (y + y1 * (prior_K - 1.0))
        s0 = float(initial_state.top_oil_C)
        s1 = float(amp_fast * prior_Ky)
        s2 = float(amp_slow * prior_Ky)
    top_oil[0] = s0
    hotspot[0] = s0 + s1 - s2

    if solver == "euler":
        # The standard's own difference-equation presentation. Provided for
        # comparison only -- NOT the reproduction path (see module docstring).
        for i in range(n - 1):
            d0 = (dtor * fK_on[i] + A_on[i] - s0) * inv_tau_oil
            d1 = (amp_fast * Ky_on[i] - s1) * inv_tau_fast
            d2 = (amp_slow * Ky_on[i] - s2) * inv_tau_slow
            s0 += dt * d0
            s1 += dt * d1
            s2 += dt * d2
            top_oil[i + 1] = s0
            hotspot[i + 1] = s0 + s1 - s2
        if return_state:
            return top_oil, hotspot, (s0, s1, s2)
        return top_oil, hotspot

    # Fast path: three independent one-pole recurrences, evaluated in C.
    drive_oil_on = dtor * fK_on + A_on
    drive_oil_half = dtor * fK_half + A_half
    a_oil, b_oil = _rk4_affine_coefficients(k11 * tauo_s, dt, drive_oil_on, drive_oil_half)
    a_fast, b_fast = _rk4_affine_coefficients(
        k22 * tw_s, dt, amp_fast * Ky_on, amp_fast * Ky_half
    )
    a_slow, b_slow = _rk4_affine_coefficients(
        tauo_s / k22, dt, amp_slow * Ky_on, amp_slow * Ky_half
    )

    oil_state = _linear_recurrence(a_oil, b_oil, s0, n)
    fast_state = _linear_recurrence(a_fast, b_fast, s1, n)
    slow_state = _linear_recurrence(a_slow, b_slow, s2, n)

    hotspot_out = oil_state + fast_state - slow_state
    if return_state:
        final = (float(oil_state[-1]), float(fast_state[-1]), float(slow_state[-1]))
        return oil_state, hotspot_out, final
    return oil_state, hotspot_out


def _integrate_reference(
    dtor: float,
    tauo_s: float,
    dthr: float,
    tw_s: float,
    x: float,
    y: float,
    k11: float,
    k21: float,
    k22: float,
    R: float,
    K_on: NDArray[np.float64],
    K_half: NDArray[np.float64],
    A_on: NDArray[np.float64],
    A_half: NDArray[np.float64],
    dt: float,
    x1: float = 0.0,
    y1: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Explicit RK4 loop, kept as the reference the fast path is checked against.

    This is the literal transcription of the published campaign's integrator:
    a Python loop carrying the three-state vector through four RK4 stages.
    It is slow on purpose -- it exists so that `_integrate`'s recurrence
    formulation has something independent to be verified against, and so
    that anyone reading the fast path can see what it is a restatement of.
    Not used in production. See `tests/test_physics.py`.
    """
    n = K_on.size
    top_oil = np.empty(n, dtype=np.float64)
    hotspot = np.empty(n, dtype=np.float64)

    # Exponents may vary with load. x1 = y1 = 0 gives the fixed-exponent form
    # exactly, which is the default everywhere.
    fK_on = ((1.0 + R * K_on**2) / (1.0 + R)) ** (x + x1 * (K_on - 1.0))
    fK_half = ((1.0 + R * K_half**2) / (1.0 + R)) ** (x + x1 * (K_half - 1.0))
    Ky_on = K_on ** (y + y1 * (K_on - 1.0))
    Ky_half = K_half ** (y + y1 * (K_half - 1.0))

    inv_tau_oil = 1.0 / (k11 * tauo_s)
    inv_tau_fast = 1.0 / (k22 * tw_s)
    inv_tau_slow = 1.0 / (tauo_s / k22)
    amp_fast = k21 * dthr
    amp_slow = (k21 - 1.0) * dthr

    s0 = float(A_on[0] + dtor * fK_on[0])
    s1 = float(amp_fast * Ky_on[0])
    s2 = float(amp_slow * Ky_on[0])
    top_oil[0] = s0
    hotspot[0] = s0 + s1 - s2

    half = 0.5 * dt
    sixth = dt / 6.0
    for i in range(n - 1):
        fk0, fkh, fk1 = fK_on[i], fK_half[i], fK_on[i + 1]
        ky0, kyh, ky1 = Ky_on[i], Ky_half[i], Ky_on[i + 1]
        a0, ah, a1 = A_on[i], A_half[i], A_on[i + 1]

        k1_0 = (dtor * fk0 + a0 - s0) * inv_tau_oil
        k1_1 = (amp_fast * ky0 - s1) * inv_tau_fast
        k1_2 = (amp_slow * ky0 - s2) * inv_tau_slow

        k2_0 = (dtor * fkh + ah - (s0 + half * k1_0)) * inv_tau_oil
        k2_1 = (amp_fast * kyh - (s1 + half * k1_1)) * inv_tau_fast
        k2_2 = (amp_slow * kyh - (s2 + half * k1_2)) * inv_tau_slow

        k3_0 = (dtor * fkh + ah - (s0 + half * k2_0)) * inv_tau_oil
        k3_1 = (amp_fast * kyh - (s1 + half * k2_1)) * inv_tau_fast
        k3_2 = (amp_slow * kyh - (s2 + half * k2_2)) * inv_tau_slow

        k4_0 = (dtor * fk1 + a1 - (s0 + dt * k3_0)) * inv_tau_oil
        k4_1 = (amp_fast * ky1 - (s1 + dt * k3_1)) * inv_tau_fast
        k4_2 = (amp_slow * ky1 - (s2 + dt * k3_2)) * inv_tau_slow

        s0 = s0 + sixth * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
        s1 = s1 + sixth * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
        s2 = s2 + sixth * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)

        top_oil[i + 1] = s0
        hotspot[i + 1] = s0 + s1 - s2

    return top_oil, hotspot


def _prepare_inputs(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    load_pu_half: NDArray[np.float64] | None,
    ambient_C_half: NDArray[np.float64] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Validate and broadcast simulation inputs. Returns arrays + dt."""
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"time_s must be 1-D with >= 2 samples, got shape {t.shape}")

    steps = np.diff(t)
    dt = float(steps[0])
    if dt <= 0.0:
        raise ValueError(f"time_s must be strictly increasing; first step is {dt} s")
    # A non-uniform grid would silently invalidate the fixed-step RK4 below.
    # corefield.ingest resamples to a uniform grid precisely so this holds.
    if not np.allclose(steps, dt, rtol=0.0, atol=1e-9):
        worst = float(np.max(np.abs(steps - dt)))
        raise ValueError(
            f"time_s must be uniformly spaced for fixed-step RK4; largest deviation "
            f"from the first step ({dt} s) is {worst} s. Resample first -- "
            f"corefield.ingest.load_telemetry does this with a documented method."
        )

    K = np.asarray(load_pu, dtype=np.float64)
    if K.shape != t.shape:
        raise ValueError(f"load_pu shape {K.shape} does not match time_s shape {t.shape}")
    if np.any(~np.isfinite(K)):
        raise ValueError("load_pu contains non-finite values (NaN/inf); gap-fill before simulating")
    if np.any(K < 0.0):
        raise ValueError("load_pu contains negative values; per-unit load current cannot be < 0")

    A = np.asarray(ambient_C, dtype=np.float64)
    if A.ndim == 0:
        A = np.full(t.shape, float(A))
    elif A.shape != t.shape:
        raise ValueError(f"ambient_C shape {A.shape} does not match time_s shape {t.shape}")
    if np.any(~np.isfinite(A)):
        raise ValueError("ambient_C contains non-finite values (NaN/inf)")
    _reject_kelvin(A, "ambient_C")

    # Half-step values for RK4 stages 2 and 3. Interpolating the endpoints is
    # second-order and would degrade RK4; when the analytic load/ambient
    # function is available the caller should pass exact half-step samples.
    if load_pu_half is None:
        Kh = 0.5 * (K[:-1] + K[1:])
        Kh = np.append(Kh, K[-1])
    else:
        Kh = np.asarray(load_pu_half, dtype=np.float64)
        if Kh.shape != t.shape:
            raise ValueError(f"load_pu_half shape {Kh.shape} does not match time_s shape {t.shape}")

    if ambient_C_half is None:
        Ah = 0.5 * (A[:-1] + A[1:])
        Ah = np.append(Ah, A[-1])
    else:
        Ah = np.asarray(ambient_C_half, dtype=np.float64)
        if Ah.shape != t.shape:
            raise ValueError(
                f"ambient_C_half shape {Ah.shape} does not match time_s shape {t.shape}"
            )
        _reject_kelvin(Ah, "ambient_C_half")

    return t, K, Kh, A, Ah, dt


def _reject_kelvin(values: NDArray[np.float64], name: str) -> None:
    """Refuse an absolute-temperature series that is obviously in kelvin.

    Ambient temperatures on Earth do not exceed 60 degC. A series whose
    minimum is above 200 is kelvin, and feeding kelvin where Celsius is
    expected shifts the whole thermal model by 273 K -- an error that
    produces plausible-looking, entirely wrong output. Fail loudly instead.
    """
    if values.size and float(np.min(values)) > 200.0:
        raise ValueError(
            f"{name} looks like kelvin (minimum {float(np.min(values)):.1f}), but this "
            f"API takes degrees CELSIUS. Subtract 273.15. Absolute temperatures are "
            f"degC; only differences and rises are in K."
        )


def simulate(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    *,
    load_pu_half: NDArray[np.float64] | None = None,
    ambient_C_half: NDArray[np.float64] | None = None,
    solver: Solver = "rk4",
    initial_state: InitialState | None = None,
) -> ThermalTrajectory:
    """Integrate the two-exponential model forward on a uniform time grid.

    Parameters
    ----------
    time_s : uniform, strictly increasing time grid [s]
    load_pu : per-unit load current at each grid point [pu]
    ambient_C : ambient temperature [degC], scalar (constant) or per-sample.
        Celsius, NOT kelvin -- a kelvin series is detected and rejected.
    params : the four identified thermal parameters
    constants : cooling-class constants (default ONAF medium & large power)
    load_pu_half, ambient_C_half : optional exact half-step samples for the
        RK4 interior stages. When the driving signals come from an analytic
        function, passing these preserves fourth-order accuracy; omitted,
        they are averaged from the endpoints, which is what real telemetry
        allows.
    solver : "rk4" (default, the reproduction path) or "euler" (the
        standard's difference-equation presentation, for comparison only)
    initial_state : the unit's current thermal state. Omit to start from
        equilibrium at the opening load, which is right for a synthetic
        study and wrong for an operational question.

    Returns
    -------
    ThermalTrajectory with top-oil and hot-spot temperatures in degC.

    Raises
    ------
    ValueError
        On a non-uniform grid, shape mismatch, non-finite or negative load,
        or an ambient series that appears to be in kelvin.
    """
    if solver not in ("rk4", "euler"):
        raise ValueError(f"solver must be 'rk4' or 'euler', got {solver!r}")

    t, K, Kh, A, Ah, dt = _prepare_inputs(
        time_s, load_pu, ambient_C, load_pu_half, ambient_C_half
    )

    # The standard's own guidance is a step below half the smallest time
    # constant. Warn-by-exception rather than silently producing a trajectory
    # whose transients are numerically damped.
    if dt > 0.5 * params.tau_w_s:
        raise ValueError(
            f"time step {dt:.1f} s exceeds half the winding time constant "
            f"({0.5 * params.tau_w_s:.1f} s). The winding transient -- the thing this "
            f"model exists to resolve -- would be numerically damped. Resample to a "
            f"finer grid, or identify a slower unit."
        )

    top_oil, hotspot = _integrate(
        dtor=params.delta_theta_or_K,
        tauo_s=params.tau_o_s,
        dthr=params.delta_theta_hr_K,
        tw_s=params.tau_w_s,
        x=constants.x,
        y=constants.y,
        x1=constants.x1,
        y1=constants.y1,
        k11=constants.k11,
        k21=constants.k21,
        k22=constants.k22,
        R=params.loss_ratio_R,
        K_on=K,
        K_half=Kh,
        A_on=A,
        A_half=Ah,
        dt=dt,
        solver=solver,
        initial_state=initial_state,
    )

    return ThermalTrajectory(
        time_s=t,
        load_pu=K,
        ambient_C=A,
        top_oil_C=top_oil,
        hotspot_C=hotspot,
        gradient_K=hotspot - top_oil,
    )


def top_oil_rise(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    **kwargs: object,
) -> NDArray[np.float64]:
    """Top-oil rise over ambient, as a trajectory [K].

    Thin wrapper over `simulate`; see it for parameter documentation.
    Returns theta_o(t) - theta_a(t), in kelvin.
    """
    traj = simulate(time_s, load_pu, ambient_C, params, constants, **kwargs)  # type: ignore[arg-type]
    return traj.top_oil_C - traj.ambient_C


def hotspot_rise(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    **kwargs: object,
) -> NDArray[np.float64]:
    """Hot-spot rise over TOP-OIL, as a trajectory [K].

    This is the gradient theta_h(t) - theta_o(t), not the rise over ambient.
    It is the quantity that exhibits the two-exponential overshoot.
    """
    traj = simulate(time_s, load_pu, ambient_C, params, constants, **kwargs)  # type: ignore[arg-type]
    return traj.gradient_K


def hotspot_temperature(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    **kwargs: object,
) -> NDArray[np.float64]:
    """Absolute winding hot-spot temperature, as a trajectory [degC].

    This is the hidden quantity the whole package exists to reconstruct.
    """
    traj = simulate(time_s, load_pu, ambient_C, params, constants, **kwargs)  # type: ignore[arg-type]
    return traj.hotspot_C
