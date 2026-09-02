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

"""Reproduction of the published synthetic campaign.

This module exists so that the regression suite, the Streamlit demo and any
future reviewer all run the SAME code path to produce the campaign's tables.
The legacy notebooks stored zero cell outputs across all 73 code cells, so
before this module existed every published number was a transcription that
nobody could re-derive. Now they are re-derived on every test run.

Everything here is synthetic. See `corefield.synthetic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .crlb import PARAMETER_NAMES
from .estimator import HotspotReferences, identify
from .iec60076_7 import ThermalParams, simulate
from .models_ab import fit_single_exponential, simulate_single_exponential
from .synthetic import (
    AMBIENT_CONSTANT_C,
    DT_S,
    OIL_SAMPLE_STRIDE,
    TRUTH_PARAMS,
    CorruptionScenario,
    calibration_indices,
    day_a_load,
    day_b_load,
    day_c_load,
    diurnal_ambient,
    truth_trajectory,
)

__all__ = [
    "SENSOR_NOISE_K",
    "GATE_RMSE_K",
    "GATE_PEAK_K",
    "TrajectoryMetrics",
    "ScenarioResult",
    "ModelComparison",
    "DayComparisonTrajectories",
    "run_scenario",
    "stage_b_gate",
    "day_transfer",
    "day_transfer_trajectories",
]

#: Sensor noise standard deviation used throughout the campaign [K].
#: SCADA-grade; the product claim depends on NOT needing better than this.
SENSOR_NOISE_K: float = 0.5

#: Pre-registered gate: a model passes only if BOTH hold, against the TRUE
#: hot-spot trajectory. The peak criterion is separate from RMSE on purpose
#: -- a model can flatter its RMSE while missing the one moment that matters.
GATE_RMSE_K: float = 2.0
GATE_PEAK_K: float = 2.0

#: The single optimiser start the published campaign used. Passing this to
#: `identify` disables multi-start, which is what reproduction requires.
CAMPAIGN_START: tuple[tuple[float, float, float, float], ...] = ((30.0, 6000.0, 15.0, 600.0),)

#: Load-event hours on day A, and the post-event window scored by `ev_rmse`.
_EVENT_HOURS: tuple[int, ...] = (6, 12, 16, 20)
_EVENT_WINDOW_MIN: float = 45.0


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Scoring of a fitted trajectory against the true hot spot.

    All values are rounded to 2 decimal places BEFORE aggregation, matching
    the published campaign. Rounding first then averaging is not the same as
    averaging then rounding, and the published tables did the former.

    Attributes
    ----------
    rmse_K : root-mean-square error over the whole day [K]
    max_abs_K : largest absolute error at any instant [K]
    peak_error_K : signed error at the PEAK -- fitted peak minus true peak.
        Positive means reading HIGH, which causes false derating. Negative
        means reading LOW, which is the direction a thermal monitor must
        never err in.
    event_rmse_K : RMSE restricted to the 45 min after each load event [K]
    """

    rmse_K: float
    max_abs_K: float
    peak_error_K: float
    event_rmse_K: float

    def passes_gate(self) -> bool:
        """Whether this trajectory meets the pre-registered gate."""
        return self.rmse_K <= GATE_RMSE_K and abs(self.peak_error_K) <= GATE_PEAK_K


@dataclass(frozen=True)
class ScenarioResult:
    """Aggregated outcome of one corruption scenario over several seeds."""

    name: str
    n_seeds: int
    parameter_errors_pct: NDArray[np.float64]  # (n_seeds, 4), signed
    metrics: tuple[TrajectoryMetrics, ...]
    published_gate: str

    @property
    def mean_parameter_errors_pct(self) -> dict[str, float]:
        """Signed mean parameter error per name [%]. Signed: the object is bias."""
        means = self.parameter_errors_pct.mean(axis=0)
        return {n: round(float(v), 2) for n, v in zip(PARAMETER_NAMES, means)}

    @property
    def tau_w_sd_pct(self) -> float:
        """Standard deviation of the tau_w error across seeds [%]."""
        return round(float(self.parameter_errors_pct[:, 3].std(ddof=1)), 2)

    @property
    def mean_rmse_K(self) -> float:
        return round(float(np.mean([m.rmse_K for m in self.metrics])), 2)

    @property
    def mean_peak_error_K(self) -> float:
        return round(float(np.mean([m.peak_error_K for m in self.metrics])), 2)

    @property
    def gate(self) -> Literal["PASS", "FAIL"]:
        """Gate verdict on the seed-mean metrics."""
        ok = self.mean_rmse_K <= GATE_RMSE_K and abs(self.mean_peak_error_K) <= GATE_PEAK_K
        return "PASS" if ok else "FAIL"

    @property
    def reproduces_published_gate(self) -> bool:
        return self.gate == self.published_gate


@dataclass(frozen=True)
class ModelComparison:
    """Model A / B / C scored on one load day."""

    day: str
    metrics: dict[str, tuple[TrajectoryMetrics, ...]]

    def mean(self, model: str, field: str) -> float:
        return round(float(np.mean([getattr(m, field) for m in self.metrics[model]])), 2)

    def worst_peak(self, model: str) -> float:
        """Largest per-seed peak error [K] -- the worst-case a utility would see."""
        return round(max(round(m.peak_error_K, 2) for m in self.metrics[model]), 2)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _event_mask(time_s: NDArray[np.float64]) -> NDArray[np.bool_]:
    mask = np.zeros(time_s.size, dtype=bool)
    for hour in _EVENT_HOURS:
        start = hour * 3600.0
        mask |= (time_s >= start) & (time_s <= start + _EVENT_WINDOW_MIN * 60.0)
    return mask


def _score(
    fitted: NDArray[np.float64], truth: NDArray[np.float64], mask: NDArray[np.bool_]
) -> TrajectoryMetrics:
    err = fitted - truth
    return TrajectoryMetrics(
        rmse_K=round(float(np.sqrt(np.mean(err**2))), 2),
        max_abs_K=round(float(np.max(np.abs(err))), 2),
        peak_error_K=round(float(fitted.max() - truth.max()), 2),
        event_rmse_K=round(float(np.sqrt(np.mean(err[mask] ** 2))), 2),
    )


def _day_load(day: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(on-grid, half-step) load for a named day."""
    fn = {"A": day_a_load, "B": day_b_load, "C": day_c_load}[day]
    t = truth_trajectory(day).time_s
    return fn(t), fn(t + 0.5 * DT_S)


# --------------------------------------------------------------------------
# Stage C: corruption battery
# --------------------------------------------------------------------------


def run_scenario(
    scenario: CorruptionScenario,
    n_seeds: int = 10,
    *,
    noise_K: float = SENSOR_NOISE_K,
    n_cal: int = 17,
) -> ScenarioResult:
    """Run one corruption scenario over `n_seeds` noise realisations.

    Evaluation is DEPLOYMENT-CONSISTENT: the fitted model is scored running
    on the same possibly-corrupted signals it will see in service, against
    the TRUE hot-spot trajectory. That is what distinguishes a scenario the
    engine survives from one it merely appears to survive -- a CT gain error
    is compensated in the trajectory but not in the parameters, and only
    deployment-consistent scoring shows that.

    Parameters
    ----------
    scenario : the corruption case
    n_seeds : number of noise realisations
    noise_K : sensor noise standard deviation [K]
    n_cal : calibration budget

    Returns
    -------
    ScenarioResult
    """
    truth = truth_trajectory("A", ambient=scenario.truth_ambient)
    t = truth.time_s
    mask = _event_mask(t)
    oil_index = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    cal_index = calibration_indices(n_cal, t)

    K_on, K_half = _day_load("A")
    K_model = K_on * scenario.load_gain
    K_model_half = K_half * scenario.load_gain

    if scenario.truth_ambient == "diurnal":
        ambient_true = diurnal_ambient(t)
        ambient_true_half = diurnal_ambient(t + 0.5 * DT_S)
    else:
        ambient_true = np.full(t.size, AMBIENT_CONSTANT_C)
        ambient_true_half = ambient_true

    if scenario.model_ambient == "ignored":
        ambient_model = np.full(t.size, AMBIENT_CONSTANT_C)
        ambient_model_half = ambient_model
    else:
        ambient_model = ambient_true
        ambient_model_half = ambient_true_half

    truth_vector = TRUTH_PARAMS.as_vector()
    errors = np.empty((n_seeds, 4), dtype=np.float64)
    scores: list[TrajectoryMetrics] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(scenario.seed_base + seed)
        oil_samples = truth.top_oil_C[oil_index] + rng.normal(0, noise_K, oil_index.size)
        cal_samples = truth.hotspot_C[cal_index] + rng.normal(0, noise_K, cal_index.size)

        if scenario.oil_corruption is not None:
            oil_samples = scenario.oil_corruption(oil_samples, rng, t[oil_index])
        if scenario.cal_corruption is not None:
            cal_samples = scenario.cal_corruption(cal_samples, rng, t[cal_index])

        oil_series = np.full(t.size, np.nan)
        oil_series[oil_index] = oil_samples

        result = identify(
            t, K_model, ambient_model, oil_series,
            HotspotReferences(t[cal_index], cal_samples, source="synthetic"),
            loss="soft_l1" if scenario.robust_loss else "linear",
            starts=CAMPAIGN_START,
            load_pu_half=K_model_half, ambient_C_half=ambient_model_half,
        )

        deployed = simulate(
            t, K_model, ambient_model, result.params,
            load_pu_half=K_model_half, ambient_C_half=ambient_model_half,
        )
        errors[seed] = (result.params.as_vector() - truth_vector) / truth_vector * 100.0
        scores.append(_score(deployed.hotspot_C, truth.hotspot_C, mask))

    return ScenarioResult(
        name=scenario.name,
        n_seeds=n_seeds,
        parameter_errors_pct=errors,
        metrics=tuple(scores),
        published_gate=scenario.published_gate,
    )


# --------------------------------------------------------------------------
# Stage B: model selection
# --------------------------------------------------------------------------


def _fit_models_on_day_a(
    n_seeds: int, noise_K: float, n_cal: int
) -> tuple[list, list, list[ThermalParams]]:
    """Fit A, B and C on day A. Returns (A fits, B fits, C parameter sets).

    RNG discipline: Models A and B consume an oil-noise draw they then
    discard, so that their calibration draw lands at the same position in
    the stream as Model C's. Without that alignment the three models would
    be scored on different noise and the comparison would be meaningless.
    """
    truth = truth_trajectory("A")
    t = truth.time_s
    oil_index = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    cal_index = calibration_indices(n_cal, t)
    K_on, K_half = _day_load("A")
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)

    fits_a, fits_b, params_c = [], [], []
    for seed in range(n_seeds):
        # Models A and B: oil draw over the FULL grid, discarded.
        rng = np.random.default_rng(1000 + seed)
        _ = rng.normal(0, max(noise_K, 1e-9), t.size)
        cal_samples = truth.hotspot_C[cal_index] + rng.normal(0, noise_K, cal_index.size)
        fits_a.append(
            fit_single_exponential("A", truth.top_oil_C, K_on, cal_index, cal_samples,
                                   DT_S, load_pu_half=K_half)
        )
        fits_b.append(
            fit_single_exponential("B", truth.top_oil_C, K_on, cal_index, cal_samples,
                                   DT_S, load_pu_half=K_half)
        )

        # Model C: same stream, but the oil draw is USED.
        rng = np.random.default_rng(1000 + seed)
        oil_noise = rng.normal(0, max(noise_K, 1e-9), t.size)
        cal_samples_c = truth.hotspot_C[cal_index] + rng.normal(0, noise_K, cal_index.size)
        oil_series = np.full(t.size, np.nan)
        oil_series[oil_index] = (truth.top_oil_C + oil_noise)[oil_index]
        params_c.append(
            identify(
                t, K_on, ambient, oil_series,
                HotspotReferences(t[cal_index], cal_samples_c, source="synthetic"),
                loss="linear", starts=CAMPAIGN_START,
                load_pu_half=K_half, ambient_C_half=ambient,
            ).params
        )
    return fits_a, fits_b, params_c


def stage_b_gate(
    n_seeds: int = 10, *, noise_K: float = SENSOR_NOISE_K, n_cal: int = 17
) -> ModelComparison:
    """Fit A, B and C on day A and score all three on day A."""
    return day_transfer("A", n_seeds=n_seeds, noise_K=noise_K, n_cal=n_cal)


def day_transfer(
    day: Literal["A", "B", "C"],
    n_seeds: int = 10,
    *,
    noise_K: float = SENSOR_NOISE_K,
    n_cal: int = 17,
) -> ModelComparison:
    """Fit A, B and C on day A, then score all three on `day`.

    day="A" is the fitting-day gate. day="B" tests schedule generalisation
    inside the fitted load hull. day="C" tests extrapolation to 1.30 pu,
    outside it -- the case that separates the models commercially.
    """
    fits_a, fits_b, params_c = _fit_models_on_day_a(n_seeds, noise_K, n_cal)

    target = truth_trajectory(day)
    t = target.time_s
    mask = _event_mask(t)
    K_on, K_half = _day_load(day)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)

    metrics: dict[str, tuple[TrajectoryMetrics, ...]] = {}
    for name, fits in (("A", fits_a), ("B", fits_b)):
        metrics[name] = tuple(
            _score(
                simulate_single_exponential(
                    target.top_oil_C, K_on, f.delta_theta_hr_eff_K, f.tau_w_eff_min,
                    f.y_eff, DT_S, load_pu_half=K_half,
                ),
                target.hotspot_C, mask,
            )
            for f in fits
        )
    metrics["C"] = tuple(
        _score(
            simulate(t, K_on, ambient, p, load_pu_half=K_half, ambient_C_half=ambient).hotspot_C,
            target.hotspot_C, mask,
        )
        for p in params_c
    )
    return ModelComparison(day=day, metrics=metrics)


@dataclass(frozen=True)
class DayComparisonTrajectories:
    """Trajectories behind a ModelComparison, for plotting.

    Attributes
    ----------
    day : which load day was scored
    time_h : time axis [h]
    load_pu : load profile of the scored day [pu]
    truth_hotspot_C : the true hot-spot trajectory [degC]
    mean_hotspot_C : seed-mean fitted trajectory per model [degC]
    peak_error_K : seed-mean peak error per model [K]
    worst_peak_error_K : worst single-seed peak error per model [K]
    """

    day: str
    time_h: NDArray[np.float64]
    load_pu: NDArray[np.float64]
    truth_hotspot_C: NDArray[np.float64]
    mean_hotspot_C: dict[str, NDArray[np.float64]]
    peak_error_K: dict[str, float]
    worst_peak_error_K: dict[str, float]


def day_transfer_trajectories(
    day: Literal["A", "B", "C"] = "C",
    n_seeds: int = 10,
    *,
    noise_K: float = SENSOR_NOISE_K,
    n_cal: int = 17,
) -> DayComparisonTrajectories:
    """Fit A, B and C on day A and return their trajectories on `day`.

    Same computation as `day_transfer`, but keeping the trajectories rather
    than only their summary metrics. Used by the demo application, which
    needs to show the curves and not merely the table.
    """
    fits_a, fits_b, params_c = _fit_models_on_day_a(n_seeds, noise_K, n_cal)

    target = truth_trajectory(day)
    t = target.time_s
    K_on, K_half = _day_load(day)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)

    curves: dict[str, list[NDArray[np.float64]]] = {"A": [], "B": [], "C": []}
    for name, fits in (("A", fits_a), ("B", fits_b)):
        for f in fits:
            curves[name].append(
                simulate_single_exponential(
                    target.top_oil_C, K_on, f.delta_theta_hr_eff_K, f.tau_w_eff_min,
                    f.y_eff, DT_S, load_pu_half=K_half,
                )
            )
    for p in params_c:
        curves["C"].append(
            simulate(t, K_on, ambient, p, load_pu_half=K_half, ambient_C_half=ambient).hotspot_C
        )

    truth_peak = float(target.hotspot_C.max())
    mean_curves: dict[str, NDArray[np.float64]] = {}
    peak_error: dict[str, float] = {}
    worst_peak: dict[str, float] = {}
    for name, stack in curves.items():
        arr = np.vstack(stack)
        mean_curves[name] = arr.mean(axis=0)
        per_seed = [round(float(c.max() - truth_peak), 2) for c in stack]
        peak_error[name] = round(float(np.mean(per_seed)), 2)
        worst_peak[name] = round(max(per_seed), 2)

    return DayComparisonTrajectories(
        day=day,
        time_h=t / 3600.0,
        load_pu=K_on,
        truth_hotspot_C=target.hotspot_C,
        mean_hotspot_C=mean_curves,
        peak_error_K=peak_error,
        worst_peak_error_K=worst_peak,
    )
