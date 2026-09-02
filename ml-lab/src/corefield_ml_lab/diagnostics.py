"""Preregistered E5 conformal and E6 null-space diagnostics.

The conformal functions operate on independent episode-level scalar outcomes.
They implement the higher-rank split-conformal rule in PREREGISTRATION.md and
the target-point mass at ``+infinity`` used by weighted conformal prediction
under covariate shift.  Estimated KDE ratios are diagnostics only; this module
does not attach an exact finite-sample guarantee to estimated weights.

The E6 generator deliberately duplicates every externally observable history
while assigning independently permuted winding-height labels.  Its fixed,
small NumPy probe is a leakage detector, not a field-reconstruction model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta as beta_distribution
from scipy.stats import gaussian_kde


DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95
DEFAULT_E6_SEEDS: tuple[int, ...] = tuple(range(71_000, 71_010))
E6_FEATURE_NAMES: tuple[str, ...] = (
    "load_now_pu",
    "load_lag_6_min_pu",
    "load_lag_16_min_pu",
    "load_lag_60_min_pu",
    "load_lag_180_min_pu",
    "ambient_now_degC",
    "top_oil_now_degC",
    "top_oil_lag_16_min_degC",
    "top_oil_lag_60_min_degC",
)
E6_HISTORY_CHANNELS: tuple[str, ...] = (
    "load_pu",
    "ambient_degC",
    "top_oil_degC",
    "total_loss_relative",
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Coverage, exact binomial interval, availability, and finite widths."""

    n_total: int
    n_covered: int
    empirical_coverage: float
    confidence_level: float
    exact_ci_lower: float
    exact_ci_upper: float
    n_finite: int
    finite_availability: float
    mean_finite_width: float
    median_finite_width: float


@dataclass(frozen=True, slots=True)
class WeightedConformalResult:
    """One weighted upper limit for one target covariate."""

    upper_limit: float
    correction_quantile: float
    finite: bool
    refusal_reason: str | None
    calibration_weight_sum: float
    target_weight: float
    effective_sample_size: float
    target_infinity_mass: float


@dataclass(frozen=True, slots=True)
class KDERatioWeights:
    """Scott-bandwidth Gaussian-KDE density-ratio evaluations."""

    evaluation_loads: FloatArray
    density_ratio: FloatArray
    calibration_density: FloatArray
    target_density: FloatArray
    calibration_hull: tuple[float, float]


@dataclass(frozen=True, slots=True)
class NullSpacePairs:
    """Paired E6 observations with identical external histories and features."""

    external_features_a: FloatArray
    external_features_b: FloatArray
    external_histories_a: FloatArray
    external_histories_b: FloatArray
    location_labels_a: FloatArray
    location_labels_b: FloatArray
    feature_names: tuple[str, ...] = E6_FEATURE_NAMES
    history_channels: tuple[str, ...] = E6_HISTORY_CHANNELS


@dataclass(frozen=True, slots=True)
class PairEqualityDiagnostics:
    """Machine-precision equality checks for one E6 paired data set."""

    features_equal: bool
    histories_equal: bool
    max_feature_absolute_difference: float
    max_history_absolute_difference: float
    labels_are_distinct: bool

    @property
    def passed(self) -> bool:
        """Return whether all external inputs match and labels remain distinct."""

        return self.features_equal and self.histories_equal and self.labels_are_distinct


@dataclass(frozen=True, slots=True)
class LocationProbeResult:
    """Held-out E6 probe metrics for one seed."""

    seed: int
    n_train: int
    n_test: int
    r2: float
    mae: float
    constant_median_mae: float
    relative_mae_improvement: float
    leakage_detected: bool


@dataclass(frozen=True, slots=True)
class NullSpaceProbeSummary:
    """All preregistered E6 seeds and the joint leakage decision."""

    seeds: tuple[int, ...]
    pair_diagnostics: tuple[PairEqualityDiagnostics, ...]
    probe_results: tuple[LocationProbeResult, ...]
    passed: bool
    maximum_r2: float
    maximum_relative_mae_improvement: float


def _one_dimensional_finite(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_probability(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def one_sided_higher_quantile(
    conformity_scores: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Return the split-conformal higher quantile, possibly ``+infinity``.

    For ``n`` calibration scores, the one-indexed rank is
    ``ceil((n + 1) * (1 - alpha))``.  A rank above ``n`` is represented by the
    standard appended ``+infinity`` point rather than by clipping to the
    largest observed score.
    """

    scores = _one_dimensional_finite(conformity_scores, name="conformity_scores")
    error_rate = _validate_probability(alpha, name="alpha")
    rank = ceil((scores.size + 1) * (1.0 - error_rate))
    if rank > scores.size:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def split_conformal_upper_limits(
    predicted_peaks: ArrayLike,
    conformity_scores: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> FloatArray:
    """Add the finite-sample one-sided correction to predicted episode peaks."""

    predictions = _one_dimensional_finite(predicted_peaks, name="predicted_peaks")
    correction = one_sided_higher_quantile(conformity_scores, alpha=alpha)
    return predictions + correction


def exact_binomial_confidence_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Return the equal-tailed Clopper-Pearson exact binomial interval."""

    if isinstance(successes, bool) or not isinstance(successes, (int, np.integer)):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise TypeError("trials must be an integer")
    successes_int = int(successes)
    trials_int = int(trials)
    if trials_int <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes_int <= trials_int:
        raise ValueError("successes must lie in the inclusive range 0..trials")
    level = _validate_probability(confidence, name="confidence")
    tail = (1.0 - level) / 2.0
    lower = (
        0.0
        if successes_int == 0
        else float(
            beta_distribution.ppf(
                tail, successes_int, trials_int - successes_int + 1
            )
        )
    )
    upper = (
        1.0
        if successes_int == trials_int
        else float(
            beta_distribution.ppf(
                1.0 - tail, successes_int + 1, trials_int - successes_int
            )
        )
    )
    return lower, upper


def coverage_metrics(
    true_peaks: ArrayLike,
    predicted_peaks: ArrayLike,
    upper_limits: ArrayLike,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> CoverageMetrics:
    """Summarise formal coverage and the availability/width of finite limits."""

    truths = _one_dimensional_finite(true_peaks, name="true_peaks")
    predictions = _one_dimensional_finite(predicted_peaks, name="predicted_peaks")
    limits = np.asarray(upper_limits, dtype=np.float64)
    if limits.ndim != 1 or limits.size == 0:
        raise ValueError("upper_limits must be a non-empty one-dimensional array")
    if np.any(np.isnan(limits)) or np.any(np.isneginf(limits)):
        raise ValueError("upper_limits may be finite or +infinity, but not NaN/-infinity")
    if not (truths.shape == predictions.shape == limits.shape):
        raise ValueError("true_peaks, predicted_peaks, and upper_limits must align")

    covered = truths <= limits
    n_covered = int(np.count_nonzero(covered))
    ci_lower, ci_upper = exact_binomial_confidence_interval(
        n_covered, truths.size, confidence=confidence
    )
    finite = np.isfinite(limits)
    finite_widths = limits[finite] - predictions[finite]
    if finite_widths.size:
        mean_width = float(np.mean(finite_widths))
        median_width = float(np.median(finite_widths))
    else:
        mean_width = float("nan")
        median_width = float("nan")
    return CoverageMetrics(
        n_total=int(truths.size),
        n_covered=n_covered,
        empirical_coverage=float(n_covered / truths.size),
        confidence_level=float(confidence),
        exact_ci_lower=ci_lower,
        exact_ci_upper=ci_upper,
        n_finite=int(np.count_nonzero(finite)),
        finite_availability=float(np.mean(finite)),
        mean_finite_width=mean_width,
        median_finite_width=median_width,
    )


def effective_sample_size(weights: ArrayLike) -> float:
    """Return Kish's weight effective sample size ``(sum w)^2 / sum(w^2)``."""

    values = _one_dimensional_finite(weights, name="weights")
    if np.any(values < 0.0):
        raise ValueError("weights must be non-negative")
    denominator = float(np.dot(values, values))
    if denominator == 0.0:
        return 0.0
    total = float(np.sum(values))
    return total * total / denominator


def beta_5_2_mapped_density_ratio(
    loads: ArrayLike | float,
    *,
    lower: float = 0.60,
    upper: float = 0.90,
) -> FloatArray | float:
    """Return the known Beta(5,2)-to-Uniform density ratio on ``[lower, upper]``.

    Both distributions are mapped to the same interval, so their Jacobians
    cancel and the ratio is the Beta(5,2) density ``30*z**4*(1-z)``.
    Values outside the common support have ratio zero.
    """

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("upper must be finite and greater than lower")
    raw = np.asarray(loads, dtype=np.float64)
    if np.any(~np.isfinite(raw)):
        raise ValueError("loads must be finite")
    z = (raw - lower) / (upper - lower)
    ratio = np.where((z >= 0.0) & (z <= 1.0), 30.0 * z**4 * (1.0 - z), 0.0)
    if raw.ndim == 0:
        return float(ratio)
    return np.asarray(ratio, dtype=np.float64)


def gaussian_kde_scott_density_ratio(
    calibration_loads: ArrayLike,
    unlabeled_target_loads: ArrayLike,
    evaluation_loads: ArrayLike,
) -> KDERatioWeights:
    """Estimate ``q(x)/p(x)`` with two Gaussian KDEs using Scott bandwidth.

    The returned calibration density is left unmodified.  In particular, a
    caller can identify numerical zero and refuse an interval instead of
    flooring or clipping the denominator into a finite pseudo-guarantee.
    """

    calibration = _one_dimensional_finite(calibration_loads, name="calibration_loads")
    target = _one_dimensional_finite(
        unlabeled_target_loads, name="unlabeled_target_loads"
    )
    evaluation = _one_dimensional_finite(evaluation_loads, name="evaluation_loads")
    if calibration.size < 2 or np.ptp(calibration) == 0.0:
        raise ValueError("calibration_loads need at least two distinct values for KDE")
    if target.size < 2 or np.ptp(target) == 0.0:
        raise ValueError("unlabeled_target_loads need at least two distinct values for KDE")

    calibration_kde = gaussian_kde(calibration, bw_method="scott")
    target_kde = gaussian_kde(target, bw_method="scott")
    calibration_density = np.asarray(calibration_kde(evaluation), dtype=np.float64)
    target_density = np.asarray(target_kde(evaluation), dtype=np.float64)
    ratio = np.full(evaluation.shape, np.inf, dtype=np.float64)
    positive = calibration_density > np.finfo(np.float64).tiny
    np.divide(target_density, calibration_density, out=ratio, where=positive)
    return KDERatioWeights(
        evaluation_loads=evaluation.copy(),
        density_ratio=ratio,
        calibration_density=calibration_density,
        target_density=target_density,
        calibration_hull=(float(np.min(calibration)), float(np.max(calibration))),
    )


def weighted_conformal_upper_limit(
    predicted_peak: float,
    conformity_scores: ArrayLike,
    calibration_weights: ArrayLike,
    target_weight: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    query_load: float | None = None,
    calibration_loads: ArrayLike | None = None,
    calibration_density_at_query: float | None = None,
) -> WeightedConformalResult:
    """Return one weighted upper limit with target mass retained at infinity.

    When a query lies outside the observed calibration hull, or its estimated
    calibration density is numerically zero, this function refuses a finite
    interval and assigns the target all mass at ``+infinity``.
    """

    prediction = float(predicted_peak)
    if not np.isfinite(prediction):
        raise ValueError("predicted_peak must be finite")
    scores = _one_dimensional_finite(conformity_scores, name="conformity_scores")
    weights = _one_dimensional_finite(calibration_weights, name="calibration_weights")
    if scores.shape != weights.shape:
        raise ValueError("conformity_scores and calibration_weights must align")
    if np.any(weights < 0.0):
        raise ValueError("calibration_weights must be non-negative")
    target_weight_value = float(target_weight)
    if np.isnan(target_weight_value) or target_weight_value < 0.0:
        raise ValueError("target_weight must be non-negative and not NaN")
    error_rate = _validate_probability(alpha, name="alpha")
    weight_sum = float(np.sum(weights))
    ess = effective_sample_size(weights)
    if weight_sum <= 0.0:
        raise ValueError("calibration_weights must have a positive sum")

    refusal_reason: str | None = None
    if (query_load is None) != (calibration_loads is None):
        raise ValueError("query_load and calibration_loads must be supplied together")
    if query_load is not None and calibration_loads is not None:
        calibration_covariates = _one_dimensional_finite(
            calibration_loads, name="calibration_loads"
        )
        query = float(query_load)
        if not np.isfinite(query):
            raise ValueError("query_load must be finite")
        if query < float(np.min(calibration_covariates)) or query > float(
            np.max(calibration_covariates)
        ):
            refusal_reason = "query_outside_calibration_hull"
    if calibration_density_at_query is not None:
        density = float(calibration_density_at_query)
        if np.isnan(density) or density < 0.0:
            raise ValueError("calibration_density_at_query must be non-negative")
        if density <= np.finfo(np.float64).tiny:
            refusal_reason = "calibration_density_numerically_zero"

    if refusal_reason is not None:
        return WeightedConformalResult(
            upper_limit=float("inf"),
            correction_quantile=float("inf"),
            finite=False,
            refusal_reason=refusal_reason,
            calibration_weight_sum=weight_sum,
            target_weight=target_weight_value,
            effective_sample_size=ess,
            target_infinity_mass=1.0,
        )

    # Positive infinity is an honest density-ratio outcome only when one of
    # the support/refusal checks above makes the interval unbounded.  It must
    # never enter the finite weighted quantile arithmetic.
    if not np.isfinite(target_weight_value):
        raise ValueError(
            "non-finite target_weight requires an outside-hull or zero-density refusal"
        )

    normalizer = weight_sum + target_weight_value
    if normalizer <= 0.0:
        raise ValueError("combined calibration and target weight must be positive")
    target_infinity_mass = target_weight_value / normalizer
    required_unnormalized_mass = (1.0 - error_rate) * normalizer
    if required_unnormalized_mass > weight_sum:
        correction = float("inf")
    else:
        order = np.argsort(scores, kind="stable")
        cumulative = np.cumsum(weights[order])
        index = int(np.searchsorted(cumulative, required_unnormalized_mass, side="left"))
        correction = float(scores[order[min(index, scores.size - 1)]])
    finite = bool(np.isfinite(correction))
    return WeightedConformalResult(
        upper_limit=prediction + correction,
        correction_quantile=correction,
        finite=finite,
        refusal_reason=None if finite else "target_infinity_mass_exceeds_alpha",
        calibration_weight_sum=weight_sum,
        target_weight=target_weight_value,
        effective_sample_size=ess,
        target_infinity_mass=target_infinity_mass,
    )


def weighted_conformal_upper_limits(
    predicted_peaks: ArrayLike,
    conformity_scores: ArrayLike,
    calibration_weights: ArrayLike,
    target_weights: ArrayLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    query_loads: ArrayLike | None = None,
    calibration_loads: ArrayLike | None = None,
    calibration_densities_at_queries: ArrayLike | None = None,
) -> tuple[WeightedConformalResult, ...]:
    """Vector wrapper around :func:`weighted_conformal_upper_limit`."""

    predictions = _one_dimensional_finite(predicted_peaks, name="predicted_peaks")
    query_weights = np.asarray(target_weights, dtype=np.float64)
    if query_weights.ndim != 1 or query_weights.size == 0:
        raise ValueError("target_weights must be a non-empty one-dimensional array")
    if np.any(np.isnan(query_weights)) or np.any(query_weights < 0.0):
        raise ValueError("target_weights must be non-negative and not NaN")
    if predictions.shape != query_weights.shape:
        raise ValueError("predicted_peaks and target_weights must align")
    if query_loads is None:
        loads: list[float | None] = [None] * predictions.size
    else:
        query_array = _one_dimensional_finite(query_loads, name="query_loads")
        if query_array.shape != predictions.shape:
            raise ValueError("query_loads and predicted_peaks must align")
        loads = [float(value) for value in query_array]
    if calibration_densities_at_queries is None:
        densities: list[float | None] = [None] * predictions.size
    else:
        density_array = np.asarray(calibration_densities_at_queries, dtype=np.float64)
        if density_array.ndim != 1 or density_array.shape != predictions.shape:
            raise ValueError(
                "calibration_densities_at_queries and predicted_peaks must align"
            )
        densities = [float(value) for value in density_array]
    return tuple(
        weighted_conformal_upper_limit(
            float(prediction),
            conformity_scores,
            calibration_weights,
            float(target_weight),
            alpha=alpha,
            query_load=query,
            calibration_loads=calibration_loads,
            calibration_density_at_query=density,
        )
        for prediction, target_weight, query, density in zip(
            predictions, query_weights, loads, densities, strict=True
        )
    )


def _synthetic_external_histories(
    n_pairs: int,
    *,
    rng: np.random.Generator,
    history_steps: int,
) -> FloatArray:
    """Create external histories only; no location variable enters this path."""

    innovations = rng.normal(0.0, 0.035, size=(n_pairs, history_steps))
    load = np.empty_like(innovations)
    load[:, 0] = rng.uniform(0.60, 0.95, size=n_pairs)
    for index in range(1, history_steps):
        load[:, index] = 0.96 * load[:, index - 1] + 0.04 * 0.775 + innovations[:, index]
    np.clip(load, 0.55, 1.00, out=load)

    minutes = (np.arange(history_steps, dtype=np.float64) - history_steps + 1) * 2.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_pairs, 1))
    ambient = 20.0 + 6.0 * np.sin(2.0 * np.pi * minutes / 1440.0 + phase)
    ambient += rng.normal(0.0, 0.15, size=ambient.shape)

    total_loss = (1.0 + 6.0 * load**2) / 7.0
    oil_drive = 45.0 * total_loss**0.8
    top_oil = np.empty_like(oil_drive)
    top_oil[:, 0] = ambient[:, 0] + oil_drive[:, 0]
    time_fraction = 2.0 / 150.0
    for index in range(1, history_steps):
        equilibrium = ambient[:, index] + oil_drive[:, index]
        top_oil[:, index] = top_oil[:, index - 1] + time_fraction * (
            equilibrium - top_oil[:, index - 1]
        )
    top_oil += rng.normal(0.0, 0.5, size=top_oil.shape)
    return np.stack((load, ambient, top_oil, total_loss), axis=2)


def _probe_features_from_histories(histories: FloatArray) -> FloatArray:
    """Extract the fixed nine plain-NN inputs from 2 min histories."""

    load = histories[:, :, 0]
    ambient = histories[:, :, 1]
    oil = histories[:, :, 2]
    return np.column_stack(
        (
            load[:, -1],
            load[:, -4],
            load[:, -9],
            load[:, -31],
            load[:, -91],
            ambient[:, -1],
            oil[:, -1],
            oil[:, -9],
            oil[:, -31],
        )
    )


def generate_null_space_pairs(
    n_pairs: int,
    *,
    seed: int,
    history_steps: int = 91,
) -> NullSpacePairs:
    """Generate E6 pairs with bit-identical externals and permuted labels."""

    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int):
        raise TypeError("n_pairs must be an integer")
    if n_pairs < 4:
        raise ValueError("n_pairs must be at least four")
    if isinstance(history_steps, bool) or not isinstance(history_steps, int):
        raise TypeError("history_steps must be an integer")
    if history_steps < 91:
        raise ValueError("history_steps must cover the frozen 180 min lag")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    histories = _synthetic_external_histories(
        n_pairs, rng=rng, history_steps=history_steps
    )
    features = _probe_features_from_histories(histories)
    locations = (np.arange(n_pairs, dtype=np.float64) + 0.5) / n_pairs
    labels_a = rng.permutation(locations)
    labels_b = rng.permutation(locations)
    if np.array_equal(labels_a, labels_b):
        labels_b = np.roll(labels_b, 1)
    return NullSpacePairs(
        external_features_a=features.copy(),
        external_features_b=features.copy(),
        external_histories_a=histories.copy(),
        external_histories_b=histories.copy(),
        location_labels_a=labels_a,
        location_labels_b=labels_b,
    )


def check_null_space_pair_equality(pairs: NullSpacePairs) -> PairEqualityDiagnostics:
    """Check every paired external value for exact machine equality."""

    if pairs.external_features_a.shape != pairs.external_features_b.shape:
        raise ValueError("paired external feature shapes differ")
    if pairs.external_histories_a.shape != pairs.external_histories_b.shape:
        raise ValueError("paired external history shapes differ")
    if pairs.location_labels_a.shape != pairs.location_labels_b.shape:
        raise ValueError("paired location-label shapes differ")
    feature_difference = np.abs(
        pairs.external_features_a - pairs.external_features_b
    )
    history_difference = np.abs(
        pairs.external_histories_a - pairs.external_histories_b
    )
    return PairEqualityDiagnostics(
        features_equal=bool(np.array_equal(pairs.external_features_a, pairs.external_features_b)),
        histories_equal=bool(
            np.array_equal(pairs.external_histories_a, pairs.external_histories_b)
        ),
        max_feature_absolute_difference=float(np.max(feature_difference, initial=0.0)),
        max_history_absolute_difference=float(np.max(history_difference, initial=0.0)),
        labels_are_distinct=not np.array_equal(
            pairs.location_labels_a, pairs.location_labels_b
        ),
    )


def _initial_probe_parameters(
    input_width: int, rng: np.random.Generator
) -> list[FloatArray]:
    """Initialize the fixed 9-16-16-1 tanh probe."""

    hidden_width = 16
    w1 = rng.normal(0.0, np.sqrt(2.0 / (input_width + hidden_width)), (input_width, hidden_width))
    b1 = np.zeros(hidden_width, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / (2 * hidden_width)), (hidden_width, hidden_width))
    b2 = np.zeros(hidden_width, dtype=np.float64)
    w3 = rng.normal(0.0, np.sqrt(2.0 / (hidden_width + 1)), (hidden_width, 1))
    b3 = np.zeros(1, dtype=np.float64)
    return [w1, b1, w2, b2, w3, b3]


def _probe_forward(
    features: FloatArray, parameters: Sequence[FloatArray]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    w1, b1, w2, b2, w3, b3 = parameters
    hidden_1 = np.tanh(features @ w1 + b1)
    hidden_2 = np.tanh(hidden_1 @ w2 + b2)
    prediction = hidden_2 @ w3 + b3
    return hidden_1, hidden_2, prediction[:, 0]


def fit_location_probe(
    external_features: ArrayLike,
    location_labels: ArrayLike,
    *,
    seed: int,
    train_fraction: float = 0.70,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
) -> LocationProbeResult:
    """Fit and score the fixed CPU 9-16-16-1 tanh leakage probe."""

    features = np.asarray(external_features, dtype=np.float64)
    labels = _one_dimensional_finite(location_labels, name="location_labels")
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("external_features must be 2-D and align with labels")
    if features.shape[1] != len(E6_FEATURE_NAMES):
        raise ValueError(f"external_features must have {len(E6_FEATURE_NAMES)} columns")
    if not np.all(np.isfinite(features)):
        raise ValueError("external_features must be finite")
    if np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("location_labels must lie in [0, 1]")
    fraction = float(train_fraction)
    if not 0.5 <= fraction < 1.0:
        raise ValueError("train_fraction must lie in [0.5, 1.0)")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(labels.size)
    n_train = int(np.floor(fraction * labels.size))
    if n_train < 2 or labels.size - n_train < 2:
        raise ValueError("train and test partitions must each contain at least two rows")
    train_index, test_index = order[:n_train], order[n_train:]
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale

    parameters = _initial_probe_parameters(features.shape[1], rng)
    parameters[-1][0] = float(np.median(y_train))
    first_moment = [np.zeros_like(parameter) for parameter in parameters]
    second_moment = [np.zeros_like(parameter) for parameter in parameters]
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        hidden_1, hidden_2, prediction = _probe_forward(x_train, parameters)
        error = prediction - y_train
        output_gradient = (2.0 / n_train) * error[:, None]
        w1, _b1, w2, _b2, w3, _b3 = parameters
        grad_w3 = hidden_2.T @ output_gradient + 2.0 * weight_decay * w3
        grad_b3 = np.sum(output_gradient, axis=0)
        hidden_2_gradient = output_gradient @ w3.T
        preactivation_2_gradient = hidden_2_gradient * (1.0 - hidden_2**2)
        grad_w2 = hidden_1.T @ preactivation_2_gradient + 2.0 * weight_decay * w2
        grad_b2 = np.sum(preactivation_2_gradient, axis=0)
        hidden_1_gradient = preactivation_2_gradient @ w2.T
        preactivation_1_gradient = hidden_1_gradient * (1.0 - hidden_1**2)
        grad_w1 = x_train.T @ preactivation_1_gradient + 2.0 * weight_decay * w1
        grad_b1 = np.sum(preactivation_1_gradient, axis=0)
        gradients = [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]

        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            first_moment[index] = beta_1 * first_moment[index] + (1.0 - beta_1) * gradient
            second_moment[index] = beta_2 * second_moment[index] + (1.0 - beta_2) * gradient**2
            corrected_first = first_moment[index] / (1.0 - beta_1**step)
            corrected_second = second_moment[index] / (1.0 - beta_2**step)
            parameter -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )

    _hidden_1, _hidden_2, test_prediction = _probe_forward(x_test, parameters)
    squared_error = float(np.sum((test_prediction - y_test) ** 2))
    total_variation = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float("nan") if total_variation == 0.0 else 1.0 - squared_error / total_variation
    mae = float(np.mean(np.abs(test_prediction - y_test)))
    median_prediction = float(np.median(y_train))
    constant_median_mae = float(np.mean(np.abs(median_prediction - y_test)))
    relative_improvement = (
        float("nan")
        if constant_median_mae == 0.0
        else (constant_median_mae - mae) / constant_median_mae
    )
    leakage = bool(
        not np.isfinite(r2)
        or not np.isfinite(relative_improvement)
        or r2 > 0.02
        or relative_improvement > 0.02
    )
    return LocationProbeResult(
        seed=int(seed),
        n_train=n_train,
        n_test=int(labels.size - n_train),
        r2=r2,
        mae=mae,
        constant_median_mae=constant_median_mae,
        relative_mae_improvement=relative_improvement,
        leakage_detected=leakage,
    )


def probe_null_space_pairs(
    pairs: NullSpacePairs,
    *,
    seed: int,
    epochs: int = 300,
) -> LocationProbeResult:
    """Fit the fixed probe to both branches of one paired E6 data set."""

    pair_check = check_null_space_pair_equality(pairs)
    if not pair_check.features_equal or not pair_check.histories_equal:
        raise ValueError("external pair invariant failed before probe fitting")
    features = np.vstack((pairs.external_features_a, pairs.external_features_b))
    labels = np.concatenate((pairs.location_labels_a, pairs.location_labels_b))
    return fit_location_probe(features, labels, seed=seed, epochs=epochs)


def run_null_space_probe_seeds(
    *,
    n_pairs: int = 512,
    seeds: Iterable[int] = DEFAULT_E6_SEEDS,
    epochs: int = 300,
) -> NullSpaceProbeSummary:
    """Run the fixed E6 generator/probe protocol over the declared seeds."""

    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("seeds must not be empty")
    pair_results: list[PairEqualityDiagnostics] = []
    probe_results: list[LocationProbeResult] = []
    for seed in seed_tuple:
        pairs = generate_null_space_pairs(n_pairs, seed=seed)
        pair_result = check_null_space_pair_equality(pairs)
        pair_results.append(pair_result)
        probe_results.append(probe_null_space_pairs(pairs, seed=seed, epochs=epochs))
    maximum_r2 = max(result.r2 for result in probe_results)
    maximum_improvement = max(
        result.relative_mae_improvement for result in probe_results
    )
    passed = all(result.passed for result in pair_results) and not any(
        result.leakage_detected for result in probe_results
    )
    return NullSpaceProbeSummary(
        seeds=seed_tuple,
        pair_diagnostics=tuple(pair_results),
        probe_results=tuple(probe_results),
        passed=passed,
        maximum_r2=maximum_r2,
        maximum_relative_mae_improvement=maximum_improvement,
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_E6_SEEDS",
    "E6_FEATURE_NAMES",
    "E6_HISTORY_CHANNELS",
    "CoverageMetrics",
    "KDERatioWeights",
    "LocationProbeResult",
    "NullSpacePairs",
    "NullSpaceProbeSummary",
    "PairEqualityDiagnostics",
    "WeightedConformalResult",
    "beta_5_2_mapped_density_ratio",
    "check_null_space_pair_equality",
    "coverage_metrics",
    "effective_sample_size",
    "exact_binomial_confidence_interval",
    "fit_location_probe",
    "gaussian_kde_scott_density_ratio",
    "generate_null_space_pairs",
    "one_sided_higher_quantile",
    "probe_null_space_pairs",
    "run_null_space_probe_seeds",
    "split_conformal_upper_limits",
    "weighted_conformal_upper_limit",
    "weighted_conformal_upper_limits",
]
