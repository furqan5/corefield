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

"""Synthetic truth generator and the Stage-C corruption battery.

============================================================================
EVERYTHING IN THIS MODULE IS SYNTHETIC.
No measurement from a real transformer has ever entered this package. The
parameter set below is an illustrative ONAF-scale unit, not a real one, and
the corruption magnitudes are plausible instrument bounds (label (b)), not
values measured on any installed asset. Nothing here constitutes field
validation. See LIMITATIONS in README.md.
============================================================================

Contents
--------
Three load days, reproducing the published campaign exactly:

  day A  0.60 - 1.20 pu, events at 6/12/16/20 h   -- the fitting day
  day B  0.70 - 1.15 pu, events at 5/11/14/19 h   -- unseen transfer day,
         INSIDE the day-A load hull; tests schedule generalisation
  day C  0.70 - 1.30 pu, 2 h at 1.30 pu (15-17 h) -- emergency overload,
         OUTSIDE the day-A hull; tests extrapolation. This is the day that
         separates the models, and the reason day C exists as a separate
         case is that extrapolation beyond the fitted hull is precisely
         where the single-exponential models fail dangerously.

Load edges are tanh ramps of half-width 90 s rather than instantaneous
steps: a real feeder does not step, and a discontinuity puts a slope kink in
the truth that no smooth model can represent.

The corruption scenarios are the Stage-C battery. Each returns a
`CorruptionScenario` describing what the sensors do to the data and what the
model is consequently told. Their published gate verdicts are recorded on
the scenario itself -- including the two that FAIL, which are results and
not defects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .iec60076_7 import (
    ONAF_MEDIUM_LARGE_POWER,
    CoolingConstants,
    ThermalParams,
    ThermalTrajectory,
    simulate,
)

__all__ = [
    "TRUTH_PARAMS",
    "AMBIENT_CONSTANT_C",
    "DT_S",
    "DAY_S",
    "day_a_load",
    "day_b_load",
    "day_c_load",
    "diurnal_ambient",
    "time_grid",
    "truth_trajectory",
    "calibration_indices",
    "OIL_SAMPLE_STRIDE",
    "STAGED_TRUTH_PARAMS",
    "FAN_ON_PU",
    "FAN_OFF_PU",
    "fan_stage_schedule",
    "staged_truth_trajectory",
    "CorruptionScenario",
    "baseline",
    "oil_drift",
    "telemetry_spikes",
    "telemetry_spikes_robust",
    "integer_quantization",
    "ct_gain_error",
    "wti_calibration_bias",
    "ambient_measured",
    "ambient_ignored",
    "ALL_SCENARIOS",
    "scenario_by_name",
]

# --------------------------------------------------------------------------
# The synthetic unit
# --------------------------------------------------------------------------

#: Illustrative ONAF-scale parameter set used throughout the campaign.
#: Label (b): these are NOT a real transformer. tau_o = 150 min and
#: tau_w = 7 min happen to coincide with the mirror-sourced Table-4 ONAF
#: defaults; dtheta_or = 45 K, dtheta_hr = 22 K and R = 6 are engineering
#: scale choices with no nameplate behind them.
TRUTH_PARAMS = ThermalParams(
    delta_theta_or_K=45.0,
    tau_o_min=150.0,
    delta_theta_hr_K=22.0,
    tau_w_min=7.0,
    loss_ratio_R=6.0,
)

AMBIENT_CONSTANT_C: float = 20.0
DT_S: float = 30.0
DAY_S: float = 24 * 3600.0

#: Dense top-oil channel sampling: every 10th 30-s sample = 5 min, giving
#: 289 samples per day. This is SCADA-grade logging, deliberately -- the
#: product claim is that no new instrumentation is needed.
OIL_SAMPLE_STRIDE: int = 10

#: Ramp half-width [h]. 0.025 h = 90 s, giving a 10-90 % edge of ~3.3 min.
_RAMP_A: float = 0.025


def time_grid(days: float = 1.0, dt_s: float = DT_S) -> NDArray[np.float64]:
    """Uniform time grid [s], inclusive of the endpoint.

    Parameters
    ----------
    days : span in days
    dt_s : step [s]
    """
    return np.arange(0.0, days * DAY_S + dt_s, dt_s)


# --------------------------------------------------------------------------
# Load days
# --------------------------------------------------------------------------


def day_a_load(time_s: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Day A: the fitting day. 0.60 - 1.20 pu, events at 6/12/16/20 h [pu]."""
    h = (np.asarray(time_s, dtype=np.float64) / 3600.0) % 24.0
    a = _RAMP_A
    return (
        0.6
        + 0.15 * (1 + np.tanh((h - 6) / a))
        + 0.15 * (1 + np.tanh((h - 12) / a))
        - 0.15 * (1 + np.tanh((h - 16) / a))
        - 0.15 * (1 + np.tanh((h - 20) / a))
    )


def day_b_load(time_s: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Day B: unseen transfer day. 0.70 - 1.15 pu, events 5/11/14/19 h [pu].

    Inside the day-A hull. Tests whether a model fitted on one day's event
    structure survives a different one -- not whether it extrapolates.
    """
    h = (np.asarray(time_s, dtype=np.float64) / 3600.0) % 24.0
    a = _RAMP_A
    return (
        0.70
        + 0.175 * (1 + np.tanh((h - 5) / a))
        - 0.10 * (1 + np.tanh((h - 11) / a))
        + 0.15 * (1 + np.tanh((h - 14) / a))
        - 0.225 * (1 + np.tanh((h - 19) / a))
    )


def day_c_load(time_s: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Day C: emergency overload. 0.70 - 1.30 pu, 2 h at 1.30 (15-17 h) [pu].

    OUTSIDE the day-A fitting hull of 0.6-1.2 pu. This is the commercial
    case: it asks what each model says when the operator most wants to know
    whether extra load is safe.
    """
    h = (np.asarray(time_s, dtype=np.float64) / 3600.0) % 24.0
    a = _RAMP_A
    return (
        0.70
        + 0.125 * (1 + np.tanh((h - 6) / a))
        + 0.10 * (1 + np.tanh((h - 12) / a))
        + 0.075 * (1 + np.tanh((h - 15) / a))
        - 0.15 * (1 + np.tanh((h - 17) / a))
        - 0.15 * (1 + np.tanh((h - 21) / a))
    )


def diurnal_ambient(
    time_s: NDArray[np.float64] | float,
    amplitude_K: float = 6.0,
    mean_C: float = AMBIENT_CONSTANT_C,
) -> NDArray[np.float64]:
    """Diurnal ambient wave [degC]: minimum ~03:00, maximum ~15:00.

    The 15:00 maximum is deliberately adversarial -- it lands near the
    afternoon load peak, so ignoring ambient produces its largest error at
    exactly the worst moment.

    Parameters
    ----------
    time_s : time [s]
    amplitude_K : peak deviation from the mean [K]
    mean_C : daily mean ambient [degC]
    """
    h = np.asarray(time_s, dtype=np.float64) / 3600.0
    return mean_C + amplitude_K * np.sin(2 * np.pi * (h - 9.0) / 24.0)


_LOAD_DAYS: dict[str, Callable[[NDArray[np.float64] | float], NDArray[np.float64]]] = {
    "A": day_a_load,
    "B": day_b_load,
    "C": day_c_load,
}


def truth_trajectory(
    day: Literal["A", "B", "C"] = "A",
    *,
    params: ThermalParams = TRUTH_PARAMS,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    ambient: Literal["constant", "diurnal"] = "constant",
    dt_s: float = DT_S,
) -> ThermalTrajectory:
    """Generate the synthetic ground truth for one load day.

    Half-step load and ambient samples are passed analytically, preserving
    RK4's fourth-order accuracy -- the truth generator can do this because
    it knows the driving functions; real telemetry cannot.

    Parameters
    ----------
    day : "A", "B" or "C"
    params : thermal parameters of the synthetic unit
    constants : cooling-class constants
    ambient : "constant" (20 degC) or "diurnal" (+/- 6 K wave)
    dt_s : integration step [s]

    Returns
    -------
    ThermalTrajectory with the hidden hot-spot trajectory filled in.
    """
    if day not in _LOAD_DAYS:
        raise ValueError(f"day must be one of {sorted(_LOAD_DAYS)}, got {day!r}")

    t = time_grid(1.0, dt_s)
    load_fn = _LOAD_DAYS[day]
    K = load_fn(t)
    K_half = load_fn(t + 0.5 * dt_s)

    if ambient == "constant":
        A = np.full(t.size, AMBIENT_CONSTANT_C)
        A_half = A
    elif ambient == "diurnal":
        A = diurnal_ambient(t)
        A_half = diurnal_ambient(t + 0.5 * dt_s)
    else:
        raise ValueError(f"ambient must be 'constant' or 'diurnal', got {ambient!r}")

    return simulate(
        t, K, A, params, constants, load_pu_half=K_half, ambient_C_half=A_half
    )


# --------------------------------------------------------------------------
# Staged cooling
# --------------------------------------------------------------------------

#: Load at which the fan bank starts, and the lower load at which it stops.
#: The gap is hysteresis: without it a load hovering at the threshold would
#: chatter the fans on and off every sample, which no real control scheme
#: permits and which would make the record unfittable.
FAN_ON_PU: float = 0.95
FAN_OFF_PU: float = 0.85

#: Two-stage parameter set. Stage 1 is fans off (natural air), stage 2 is
#: fans running (forced air), for the same medium/large power transformer.
#:
#: The two TIME CONSTANTS are the standard's own tabulated values for those
#: two cooling classes -- tau_o 210 -> 150 min and tau_w 10 -> 7 min. An
#: earlier version of this scenario held tau_w equal across stages on the
#: reasoning that tank fans do not touch the winding path; checking the
#: standard showed that is not what it says.
#:
#: The rated oil rise (60 -> 45 K) is label (b), an engineering estimate:
#: it is unit-specific and not tabulated anywhere. A quarter reduction is
#: the direction and rough magnitude a fan bank produces.
#:
#: The rated gradient is deliberately IDENTICAL across stages -- the one
#: assumption `corefield.staged.SHARED_BY_DEFAULT` still encodes, and the
#: thing this scenario exists to test.
STAGED_TRUTH_PARAMS: dict[int, ThermalParams] = {
    1: ThermalParams(
        delta_theta_or_K=60.0, tau_o_min=210.0,
        delta_theta_hr_K=22.0, tau_w_min=10.0, loss_ratio_R=6.0,
    ),
    2: ThermalParams(
        delta_theta_or_K=45.0, tau_o_min=150.0,
        delta_theta_hr_K=22.0, tau_w_min=7.0, loss_ratio_R=6.0,
    ),
}


def fan_stage_schedule(load_pu: NDArray[np.float64]) -> NDArray[np.int_]:
    """Cooling stage per sample, from load with hysteresis.

    Returns 1 where the fans are off and 2 where they run. The schedule is a
    deterministic function of the load, so the record is exactly
    reproducible and carries no feedback from temperature back into the
    staging.

    Real units commonly stage on winding or oil temperature rather than
    load, which does introduce that feedback. This is a simplification, and
    it is the conservative one for testing the estimator: temperature-driven
    staging correlates stage changes with thermal transients, which gives the
    fit MORE information about the difference between stages, not less.
    """
    K = np.asarray(load_pu, dtype=np.float64)
    stage = np.ones(K.size, dtype=np.int_)
    running = False
    for i, value in enumerate(K):
        if running and value < FAN_OFF_PU:
            running = False
        elif not running and value > FAN_ON_PU:
            running = True
        stage[i] = 2 if running else 1
    return stage


def staged_truth_trajectory(
    day: Literal["A", "B", "C"] = "A",
    *,
    staged_params: dict[int, ThermalParams] | None = None,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    dt_s: float = DT_S,
) -> tuple[ThermalTrajectory, NDArray[np.int_]]:
    """Ground truth for a unit whose fans switch during the day.

    Returns
    -------
    (trajectory, stage) -- the trajectory and the cooling stage per sample.

    Notes
    -----
    Imported lazily from `corefield.staged` to keep the dependency one-way:
    `staged` builds on `synthetic`'s parameter sets, not the reverse.
    """
    from .staged import StagedThermalParams, simulate_staged

    t = time_grid(1.0, dt_s)
    load_fn = _LOAD_DAYS[day]
    K = load_fn(t)
    K_half = load_fn(t + 0.5 * dt_s)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    stage = fan_stage_schedule(K)

    params = StagedThermalParams(
        per_stage=staged_params or STAGED_TRUTH_PARAMS,
        constants=constants,
    )
    trajectory = simulate_staged(
        t, K, ambient, stage, params, load_pu_half=K_half, ambient_C_half=ambient
    )
    return trajectory, stage


# --------------------------------------------------------------------------
# Calibration schedule
# --------------------------------------------------------------------------

#: Which load events are sampled, per calibration budget.
_CAL_EVENTS: dict[int, list[float]] = {
    5: [12],
    9: [12, 16],
    13: [6, 12, 16],
    17: [6, 12, 16, 20],
    25: [6, 12, 16, 20],
}


def calibration_indices(n_cal: int, time_s: NDArray[np.float64]) -> NDArray[np.intp]:
    """Grid indices of the event-triggered hot-spot calibration reads.

    The observability law this implements (label (a), the central result of
    the v1 campaign): amplitude parameters are observable from quasi-steady
    operation, rate parameters only from transients -- and each sampled
    transient must ALSO anchor its own settled asymptote, or the amplitude
    and the rate stay correlated and the optimiser walks a degenerate valley.

    Hence 3/8/18 min after each event (0.35/0.68/0.92 of the full rise, so
    the rate is observed) plus 48 min (6.9 tau_w, 99.9 % settled, so the
    amplitude at that event's own load is anchored), plus one 4 h
    quasi-steady anchor.

    Parameters
    ----------
    n_cal : calibration budget; one of 5, 9, 13, 17, 25
    time_s : the uniform time grid to index into [s]

    Returns
    -------
    Sorted, unique grid indices.
    """
    if n_cal not in _CAL_EVENTS:
        raise ValueError(
            f"n_cal must be one of {sorted(_CAL_EVENTS)}, got {n_cal!r}. These are the "
            f"budgets the published campaign characterised; an arbitrary count has no "
            f"CRLB floor on record."
        )
    dt = float(time_s[1] - time_s[0])
    events = _CAL_EVENTS[n_cal]
    minutes = [1.5, 3, 5, 8, 12, 48] if n_cal == 25 else [3, 8, 18, 48]
    cal_h = [e + m / 60.0 for e in events for m in minutes] + [4.0]
    idx = np.unique((np.array(sorted(cal_h)) * 3600.0 / dt).round().astype(np.intp))
    return idx[idx < time_s.size]


# --------------------------------------------------------------------------
# Corruption battery
# --------------------------------------------------------------------------

# Signature of a corruption function: (values, rng, time_s_of_values) -> values
CorruptionFn = Callable[
    [NDArray[np.float64], np.random.Generator, NDArray[np.float64]], NDArray[np.float64]
]


@dataclass(frozen=True)
class CorruptionScenario:
    """One sensor/signal corruption case from the Stage-C battery.

    Attributes
    ----------
    name : scenario identifier
    description : what the instrument is doing wrong, in plain language
    oil_corruption : applied to the sampled top-oil series, or None
    cal_corruption : applied to the hot-spot calibration reads, or None
    load_gain : multiplier the MEASURED load carries relative to truth [-].
        1.02 means the CT reads 2 % high; the model both fits and runs on
        the wrong signal, which is what deployment actually looks like.
    truth_ambient : ambient profile the TRUTH is generated with
    model_ambient : ambient profile the MODEL is told about. "ignored" means
        the model assumes a constant 20 degC while the truth swings +/- 6 K.
    robust_loss : whether the fit uses soft_l1 rather than plain least squares
    seed_base : RNG seed family; scenarios with different truths use
        different families so their noise draws never collide
    published_gate : the campaign's recorded verdict, "PASS" or "FAIL"
    published_note : what the scenario demonstrated
    """

    name: str
    description: str
    oil_corruption: CorruptionFn | None = None
    cal_corruption: CorruptionFn | None = None
    load_gain: float = 1.0
    truth_ambient: Literal["constant", "diurnal"] = "constant"
    model_ambient: Literal["true", "ignored"] = "true"
    robust_loss: bool = False
    seed_base: int = 2000
    published_gate: Literal["PASS", "FAIL"] = "PASS"
    published_note: str = ""
    published_errors_pct: dict[str, float] = field(default_factory=dict)


# -- the corruption functions ---------------------------------------------


def _drift(
    values: NDArray[np.float64], rng: np.random.Generator, time_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """+1.5 K linear drift over 24 h -- a poorly maintained gauge loop."""
    return values + 1.5 * (time_s / (24 * 3600.0))


def _spikes(
    values: NDArray[np.float64], rng: np.random.Generator, time_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """2 % of samples displaced by +/- 8 K -- telemetry glitches."""
    mask = rng.random(values.size) < 0.02
    return np.where(mask, values + rng.choice([-8.0, 8.0], values.size), values)


def _quantize(
    values: NDArray[np.float64], rng: np.random.Generator, time_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rounded to whole degrees -- an integer-degC historian format."""
    return np.round(values)


def _wti_bias(
    values: NDArray[np.float64], rng: np.random.Generator, time_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """+3 K constant offset on the calibration reference -- WTI replica set hot.

    The most commercially dangerous corruption in the battery, because the
    resulting engine looks perfectly calibrated AGAINST THE REPLICA while
    carrying a large error against the real hot spot.
    """
    return values + 3.0


# -- the named scenarios ---------------------------------------------------


def baseline() -> CorruptionScenario:
    """Clean instruments: 0.5 K Gaussian noise on both channels, nothing else."""
    return CorruptionScenario(
        name="baseline",
        description="Clean instrumentation; 0.5 K Gaussian noise only.",
        published_gate="PASS",
        published_note="Reference case for every other scenario.",
        published_errors_pct={
            "delta_theta_or": -0.01, "tau_o": -0.22,
            "delta_theta_hr": 0.08, "tau_w": -0.61,
        },
    )


def oil_drift() -> CorruptionScenario:
    """Top-oil gauge drifting +1.5 K over 24 h.

    The quiet killer. The trajectory still gates PASS -- least squares
    redistributes the ramp into the parameters rather than the peak -- but
    tau_o moves +7.2 % and tau_w -8.3 %. Invisible on any plot. Harmless for
    the temperature product; disqualifying for a parameter-TREND product
    such as aging diagnostics.
    """
    return CorruptionScenario(
        name="oil_drift",
        description="Top-oil sensor drifts +1.5 K linearly over 24 h.",
        oil_corruption=_drift,
        published_gate="PASS",
        published_note="Trajectory survives; parameters are poisoned.",
        published_errors_pct={
            "delta_theta_or": 2.15, "tau_o": 7.19,
            "delta_theta_hr": -4.01, "tau_w": -8.27,
        },
    )


def telemetry_spikes() -> CorruptionScenario:
    """2 % of top-oil samples displaced by +/- 8 K, fitted with plain least squares.

    Registered as a prediction that this would degrade plain LS. It did not
    -- 289 dense samples drown symmetric zero-mean glitches. Logged as a
    MISS (P18) rather than quietly dropped.
    """
    return CorruptionScenario(
        name="telemetry_spikes",
        description="2 % of top-oil samples displaced by +/- 8 K; plain least squares.",
        oil_corruption=_spikes,
        robust_loss=False,
        published_gate="PASS",
        published_note="No measurable degradation -- prediction P18 was over-called.",
        published_errors_pct={
            "delta_theta_or": 0.00, "tau_o": 0.15,
            "delta_theta_hr": 0.07, "tau_w": -0.74,
        },
    )


def telemetry_spikes_robust() -> CorruptionScenario:
    """The same spikes, fitted with soft_l1.

    Recovers the baseline numbers almost exactly. This pair is the evidence
    for the robust loss being insurance rather than rescue: it costs nothing
    when unneeded, so it stays default-on for glitch distributions heavier
    than the one tested.
    """
    return CorruptionScenario(
        name="telemetry_spikes_robust",
        description="2 % of top-oil samples displaced by +/- 8 K; soft_l1 robust loss.",
        oil_corruption=_spikes,
        robust_loss=True,
        published_gate="PASS",
        published_note="Insurance, not rescue -- costs nothing at baseline.",
        published_errors_pct={
            "delta_theta_or": -0.01, "tau_o": -0.17,
            "delta_theta_hr": 0.09, "tau_w": -0.62,
        },
    )


def integer_quantization() -> CorruptionScenario:
    """Top-oil stored as whole degrees Celsius by the historian."""
    return CorruptionScenario(
        name="integer_quantization",
        description="Top-oil quantised to integer degC by the historian.",
        oil_corruption=_quantize,
        published_gate="PASS",
        published_note="About 1.0x baseline -- costs nothing.",
        published_errors_pct={
            "delta_theta_or": 0.01, "tau_o": -0.07,
            "delta_theta_hr": 0.06, "tau_w": -0.68,
        },
    )


def ct_gain_error() -> CorruptionScenario:
    """Current transformer reading 2 % high.

    The amplitude parameters absorb the gain error (-2.6 %) while the
    trajectory stays compensated to 0.15 K, because the model is deployed on
    the same wrong load signal it was fitted with. The error budget
    therefore applies to the parameter-trend tier, not the temperature tier.
    """
    return CorruptionScenario(
        name="ct_gain_error",
        description="CT reads 2 % high; model is fitted and deployed on the same wrong K.",
        load_gain=1.02,
        published_gate="PASS",
        published_note="Trajectory compensated; amplitude parameters carry -2.6 %.",
        published_errors_pct={
            "delta_theta_or": -2.59, "tau_o": 0.39,
            "delta_theta_hr": -2.54, "tau_w": -1.66,
        },
    )


def wti_calibration_bias() -> CorruptionScenario:
    """Calibration reference reading +3 K high. EXPECTED TO FAIL THE GATE.

    This FAIL is a result, not a bug. A winding-temperature-indicator
    replica set 3 K hot produces an engine that appears perfectly calibrated
    against that replica while carrying +14.5 % on dtheta_hr, +10.6 % on
    tau_w and +4.1 K at the true peak. Critically the bias reshapes the
    DYNAMICS, not merely the DC level, so a "relative trends only"
    positioning does not escape it.

    Commissioning consequence: at least one bias-audited hot-spot reference
    per unit -- portable fibre during a commissioning window, or the factory
    heat-run certificate as the anchor.
    """
    return CorruptionScenario(
        name="wti_calibration_bias",
        description="Hot-spot calibration reference biased +3 K (WTI replica set hot).",
        cal_corruption=_wti_bias,
        published_gate="FAIL",
        published_note="Contaminates dynamics, not just level. Drives the "
        "bias-audited-reference commissioning requirement.",
        published_errors_pct={
            "delta_theta_or": 0.01, "tau_o": 0.12,
            "delta_theta_hr": 14.52, "tau_w": 10.55,
        },
    )


def ambient_measured() -> CorruptionScenario:
    """Ambient swings +/- 6 K and the model is told about it. Passes cleanly."""
    return CorruptionScenario(
        name="ambient_measured",
        description="Ambient swings +/- 6 K diurnally; the model is given the true ambient.",
        truth_ambient="diurnal",
        model_ambient="true",
        seed_base=2100,
        published_gate="PASS",
        published_note="RMSE 0.08 K. An hourly weather feed is adequate: ambient "
        "enters through the 75-min oil low-pass.",
    )


def ambient_ignored() -> CorruptionScenario:
    """Ambient swings +/- 6 K and the model assumes it constant. EXPECTED TO FAIL.

    The single most important negative result in the campaign, because the
    error is in the dangerous direction: the peak is UNDER-predicted by
    3.09 K, and the ambient maximum coincides with the load peak. A thermal
    monitor that reads low at the afternoon peak is worse than no monitor.

    This is why `corefield.ingest` refuses to fit without an ambient channel
    rather than warning and proceeding.
    """
    return CorruptionScenario(
        name="ambient_ignored",
        description="Ambient swings +/- 6 K diurnally; the model assumes a constant 20 degC.",
        truth_ambient="diurnal",
        model_ambient="ignored",
        seed_base=2100,
        published_gate="FAIL",
        published_note="Peak UNDER-predicted by 3.09 K -- the dangerous direction. "
        "Parameters unusable (tau_w +68 %, tau_o +12 %).",
    )


#: The full battery, in the order the campaign ran it.
ALL_SCENARIOS: tuple[Callable[[], CorruptionScenario], ...] = (
    baseline,
    oil_drift,
    telemetry_spikes,
    telemetry_spikes_robust,
    integer_quantization,
    ct_gain_error,
    wti_calibration_bias,
    ambient_measured,
    ambient_ignored,
)


def scenario_by_name(name: str) -> CorruptionScenario:
    """Look up a scenario by its `name` field."""
    for factory in ALL_SCENARIOS:
        scenario = factory()
        if scenario.name == name:
            return scenario
    available = ", ".join(sorted(f().name for f in ALL_SCENARIOS))
    raise KeyError(f"unknown scenario {name!r}; available: {available}")
