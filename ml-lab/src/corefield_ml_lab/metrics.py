"""Frozen trajectory metrics and paired decision rules.

The sign convention is always ``prediction - truth``.  A negative signed
peak error is therefore unsafe-low for the overload question.  This module
contains no data-generation or model-fitting code and cannot access a primary
test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 902_010


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Scores for one noise-free hidden-truth trajectory."""

    mae_K: float
    rmse_K: float
    mean_signed_error_K: float
    signed_peak_error_K: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Five-number stochastic-cell summary plus the sample standard deviation."""

    count: int
    mean: float
    sample_std: float
    median: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CellMetrics:
    """Per-method/load aggregation across seeds."""

    mae_K: DistributionSummary
    rmse_K: DistributionSummary
    mean_signed_error_K: DistributionSummary
    signed_peak_error_K: DistributionSummary
    most_negative_signed_peak_error_K: float
    unsafe_low_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mae_K": self.mae_K.as_dict(),
            "rmse_K": self.rmse_K.as_dict(),
            "mean_signed_error_K": self.mean_signed_error_K.as_dict(),
            "signed_peak_error_K": self.signed_peak_error_K.as_dict(),
            "most_negative_signed_peak_error_K": (
                self.most_negative_signed_peak_error_K
            ),
            "unsafe_low_fraction": self.unsafe_low_fraction,
        }


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile interval for a paired mean difference."""

    mean_difference: float
    lower_95: float
    upper_95: float
    resamples: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class E3WinDecision:
    """Frozen E3 comparison of one method against NLS at one load."""

    paired_difference_method_minus_nls_K: BootstrapInterval
    lower_mean_rmse: bool
    safety_condition: bool
    confidence_interval_excludes_zero_in_favour: bool
    beats_nls: bool

    def as_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["paired_difference_method_minus_nls_K"] = (
            self.paired_difference_method_minus_nls_K.as_dict()
        )
        return output


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def trajectory_metrics(
    prediction_C: Sequence[float] | np.ndarray,
    truth_C: Sequence[float] | np.ndarray,
) -> TrajectoryMetrics:
    """Score one prediction against noise-free truth in Celsius.

    Differences of Celsius temperatures have units kelvin.  Signed peak error
    is the difference of the two trajectory maxima, not the error at either
    trajectory's peak index.
    """

    prediction = _finite_vector(prediction_C, name="prediction_C")
    truth = _finite_vector(truth_C, name="truth_C")
    if prediction.shape != truth.shape:
        raise ValueError("prediction_C and truth_C must have identical shape")
    error = prediction - truth
    return TrajectoryMetrics(
        mae_K=float(np.mean(np.abs(error))),
        rmse_K=float(np.sqrt(np.mean(error**2))),
        mean_signed_error_K=float(np.mean(error)),
        signed_peak_error_K=float(np.max(prediction) - np.max(truth)),
    )


def distribution_summary(
    values: Sequence[float] | np.ndarray,
) -> DistributionSummary:
    """Return the frozen summary; sample SD uses ``ddof=1``."""

    array = _finite_vector(values, name="values")
    sample_std = 0.0 if array.size == 1 else float(np.std(array, ddof=1))
    return DistributionSummary(
        count=int(array.size),
        mean=float(np.mean(array)),
        sample_std=sample_std,
        median=float(np.median(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )


def aggregate_trajectory_metrics(
    metrics: Iterable[TrajectoryMetrics],
) -> CellMetrics:
    """Aggregate a stochastic cell without selecting a best seed."""

    rows = tuple(metrics)
    if not rows:
        raise ValueError("metrics must contain at least one seed")
    peak = np.asarray([row.signed_peak_error_K for row in rows], dtype=np.float64)
    return CellMetrics(
        mae_K=distribution_summary([row.mae_K for row in rows]),
        rmse_K=distribution_summary([row.rmse_K for row in rows]),
        mean_signed_error_K=distribution_summary(
            [row.mean_signed_error_K for row in rows]
        ),
        signed_peak_error_K=distribution_summary(peak),
        most_negative_signed_peak_error_K=float(np.min(peak)),
        unsafe_low_fraction=float(np.mean(peak < 0.0)),
    )


def paired_mean_bootstrap_interval(
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Two-sided 95% percentile CI for paired ``candidate-reference`` means."""

    candidate_array = _finite_vector(candidate, name="candidate")
    reference_array = _finite_vector(reference, name="reference")
    if candidate_array.shape != reference_array.shape:
        raise ValueError("candidate and reference must be paired and equally sized")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    difference = candidate_array - reference_array
    rng = np.random.default_rng(seed)
    # Ten primary seeds make this only 0.8 MB at the frozen 10,000 resamples.
    indices = rng.integers(0, difference.size, size=(resamples, difference.size))
    bootstrap_means = np.mean(difference[indices], axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return BootstrapInterval(
        mean_difference=float(np.mean(difference)),
        lower_95=float(lower),
        upper_95=float(upper),
        resamples=resamples,
        seed=seed,
    )


def e3_win_decision(
    method_rmse_K: Sequence[float] | np.ndarray,
    nls_rmse_K: Sequence[float] | np.ndarray,
    method_signed_peak_error_K: Sequence[float] | np.ndarray,
    nls_signed_peak_error_K: Sequence[float] | np.ndarray,
) -> E3WinDecision:
    """Apply the preregistered E3 win rule exactly."""

    method_peak = _finite_vector(
        method_signed_peak_error_K, name="method_signed_peak_error_K"
    )
    nls_peak = _finite_vector(nls_signed_peak_error_K, name="nls_signed_peak_error_K")
    if method_peak.shape != nls_peak.shape:
        raise ValueError("method and NLS signed-peak arrays must be paired")
    interval = paired_mean_bootstrap_interval(method_rmse_K, nls_rmse_K)
    lower_mean = interval.mean_difference < 0.0
    safety = float(np.min(method_peak)) >= float(np.min(nls_peak)) - 0.10
    favourable_ci = interval.upper_95 < 0.0
    return E3WinDecision(
        paired_difference_method_minus_nls_K=interval,
        lower_mean_rmse=lower_mean,
        safety_condition=safety,
        confidence_interval_excludes_zero_in_favour=favourable_ci,
        beats_nls=lower_mean and safety and favourable_ci,
    )


def e4_adoption_decision(
    greybox_rmse_K: Sequence[float] | np.ndarray,
    nls_rmse_K: Sequence[float] | np.ndarray,
    *,
    exact_outside_hull_invariant: bool,
) -> Mapping[str, object]:
    """Apply the in-range E4 rule using paired ``NLS-greybox`` improvement."""

    improvement = paired_mean_bootstrap_interval(nls_rmse_K, greybox_rmse_K)
    at_least_point_one = improvement.mean_difference >= 0.10
    excludes_zero = improvement.lower_95 > 0.0
    adopt = bool(exact_outside_hull_invariant and at_least_point_one and excludes_zero)
    return {
        "paired_improvement_nls_minus_greybox_K": improvement.as_dict(),
        "mean_improvement_at_least_0.10_K": at_least_point_one,
        "confidence_interval_excludes_zero": excludes_zero,
        "exact_outside_hull_invariant": bool(exact_outside_hull_invariant),
        "adopt_in_range_only": adopt,
    }


def accuracy_rank(cells: Mapping[str, CellMetrics]) -> tuple[str, ...]:
    """Rank methods by increasing mean RMSE."""

    return tuple(sorted(cells, key=lambda name: (cells[name].rmse_K.mean, name)))


def safety_rank(cells: Mapping[str, CellMetrics]) -> tuple[str, ...]:
    """Apply the frozen lexicographic overload safety ranking."""

    return tuple(
        sorted(
            cells,
            key=lambda name: (
                cells[name].unsafe_low_fraction > 0.0,
                -cells[name].most_negative_signed_peak_error_K,
                -cells[name].signed_peak_error_K.mean,
                cells[name].rmse_K.mean,
                name,
            ),
        )
    )


def assert_at_least_ten_seeds(seed_values: Sequence[int]) -> tuple[int, ...]:
    """Validate the universal reporting rule and reject repeated seeds."""

    seeds = tuple(int(value) for value in seed_values)
    if len(seeds) < 10:
        raise ValueError("every stochastic cell requires at least 10 seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed values must be unique")
    if any(value < 0 or value > 0xFFFFFFFF for value in seeds):
        raise ValueError("seed values must be in 0..2**32-1")
    return seeds


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BootstrapInterval",
    "CellMetrics",
    "DistributionSummary",
    "E3WinDecision",
    "TrajectoryMetrics",
    "accuracy_rank",
    "aggregate_trajectory_metrics",
    "assert_at_least_ten_seeds",
    "distribution_summary",
    "e3_win_decision",
    "e4_adoption_decision",
    "paired_mean_bootstrap_interval",
    "safety_rank",
    "trajectory_metrics",
]
