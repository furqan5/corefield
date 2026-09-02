from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from corefield_ml_lab import diagnostics


def test_one_sided_higher_quantile_uses_preregistered_rank() -> None:
    scores = np.arange(1.0, 201.0)

    observed = diagnostics.one_sided_higher_quantile(scores, alpha=0.05)

    # ceil((200 + 1) * 0.95) = 191, using a one-indexed order statistic.
    assert observed == 191.0
    shuffled = np.random.default_rng(4).permutation(scores)
    assert diagnostics.one_sided_higher_quantile(shuffled, alpha=0.05) == 191.0
    assert np.isinf(
        diagnostics.one_sided_higher_quantile([1.0, 2.0, 3.0], alpha=0.05)
    )


def test_split_limits_and_ordinary_coverage_report_exact_interval() -> None:
    calibration_scores = np.arange(1.0, 20.0)
    predictions = np.full(100, 50.0)
    truths = np.concatenate((np.full(95, 68.0), np.full(5, 70.0)))

    limits = diagnostics.split_conformal_upper_limits(
        predictions, calibration_scores, alpha=0.05
    )
    metrics = diagnostics.coverage_metrics(truths, predictions, limits)

    assert np.all(limits == 69.0)
    assert metrics.n_covered == 95
    assert metrics.empirical_coverage == 0.95
    assert metrics.exact_ci_lower < 0.95 < metrics.exact_ci_upper
    assert metrics.finite_availability == 1.0
    assert metrics.mean_finite_width == 19.0
    assert metrics.median_finite_width == 19.0


@pytest.mark.parametrize(
    ("successes", "expected_boundary"),
    [(0, (0.0, None)), (10, (None, 1.0))],
)
def test_exact_binomial_interval_handles_boundaries(
    successes: int, expected_boundary: tuple[float | None, float | None]
) -> None:
    lower, upper = diagnostics.exact_binomial_confidence_interval(successes, 10)

    assert 0.0 <= lower <= upper <= 1.0
    if expected_boundary[0] is not None:
        assert lower == expected_boundary[0]
    if expected_boundary[1] is not None:
        assert upper == expected_boundary[1]


def test_weighted_conformal_retains_target_mass_at_infinity() -> None:
    scores = np.array([1.0, 2.0])
    weights = np.ones(2)

    finite = diagnostics.weighted_conformal_upper_limit(
        100.0, scores, weights, 0.1, alpha=0.20
    )
    unbounded = diagnostics.weighted_conformal_upper_limit(
        100.0, scores, weights, 1.0, alpha=0.20
    )

    assert finite.finite
    assert finite.correction_quantile == 2.0
    assert finite.upper_limit == 102.0
    assert finite.target_infinity_mass == pytest.approx(0.1 / 2.1)
    assert not unbounded.finite
    assert np.isinf(unbounded.upper_limit)
    assert unbounded.target_infinity_mass == pytest.approx(1.0 / 3.0)
    assert unbounded.refusal_reason == "target_infinity_mass_exceeds_alpha"


def test_strict_support_mismatch_is_unbounded_with_zero_finite_availability() -> None:
    calibration_loads = np.linspace(0.60, 0.90, 20)
    predictions = np.array([90.0, 100.0, 120.0, 160.0])
    queries = np.array([0.975, 1.125, 1.275, 1.575])
    target_weights = np.ones(queries.size)

    results = diagnostics.weighted_conformal_upper_limits(
        predictions,
        np.linspace(-1.0, 2.0, calibration_loads.size),
        np.ones(calibration_loads.size),
        target_weights,
        query_loads=queries,
        calibration_loads=calibration_loads,
    )
    limits = np.array([result.upper_limit for result in results])
    metrics = diagnostics.coverage_metrics(
        true_peaks=predictions + 50.0,
        predicted_peaks=predictions,
        upper_limits=limits,
    )

    assert all(not result.finite for result in results)
    assert all(
        result.refusal_reason == "query_outside_calibration_hull"
        for result in results
    )
    assert all(result.target_infinity_mass == 1.0 for result in results)
    assert metrics.empirical_coverage == 1.0
    assert metrics.finite_availability == 0.0
    assert np.isnan(metrics.mean_finite_width)


def test_numerically_zero_calibration_density_refuses_finite_limit() -> None:
    result = diagnostics.weighted_conformal_upper_limit(
        80.0,
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 1.0],
        0.1,
        alpha=0.20,
        calibration_density_at_query=0.0,
    )

    assert not result.finite
    assert np.isinf(result.upper_limit)
    assert result.refusal_reason == "calibration_density_numerically_zero"
    assert result.target_infinity_mass == 1.0


def test_kde_style_infinite_ratio_reaches_zero_density_refusal_in_vector_path() -> None:
    results = diagnostics.weighted_conformal_upper_limits(
        [80.0],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 1.0],
        [float("inf")],
        alpha=0.20,
        query_loads=[0.80],
        calibration_loads=[0.60, 0.75, 0.90],
        calibration_densities_at_queries=[0.0],
    )
    assert len(results) == 1
    assert not results[0].finite
    assert results[0].refusal_reason == "calibration_density_numerically_zero"
    assert results[0].target_infinity_mass == 1.0


def test_infinite_ratio_without_support_refusal_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires"):
        diagnostics.weighted_conformal_upper_limits(
            [80.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [float("inf")],
            alpha=0.20,
            query_loads=[0.80],
            calibration_loads=[0.60, 0.75, 0.90],
            calibration_densities_at_queries=[1.0],
        )


def test_known_beta_ratio_and_effective_sample_size() -> None:
    loads = np.array([0.59, 0.60, 0.75, 0.90, 0.91])

    ratio = diagnostics.beta_5_2_mapped_density_ratio(loads)

    assert np.array_equal(ratio[[0, 1, 3, 4]], np.zeros(4))
    assert ratio[2] == pytest.approx(0.9375)
    assert diagnostics.effective_sample_size(np.ones(20)) == pytest.approx(20.0)
    assert diagnostics.effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_scott_kde_ratio_is_deterministic_and_reports_finite_overlap_weights() -> None:
    rng = np.random.default_rng(61_000)
    calibration = rng.uniform(0.60, 0.90, size=300)
    target = 0.60 + 0.30 * rng.beta(5.0, 2.0, size=600)
    evaluation = np.concatenate((calibration, np.array([0.65, 0.75, 0.85])))

    first = diagnostics.gaussian_kde_scott_density_ratio(
        calibration, target, evaluation
    )
    second = diagnostics.gaussian_kde_scott_density_ratio(
        calibration, target, evaluation
    )

    assert np.array_equal(first.evaluation_loads, evaluation)
    assert np.array_equal(first.density_ratio, second.density_ratio)
    assert np.all(np.isfinite(first.density_ratio))
    assert np.all(first.density_ratio > 0.0)
    assert first.calibration_hull == (float(np.min(calibration)), float(np.max(calibration)))
    estimated_ess = diagnostics.effective_sample_size(
        first.density_ratio[: calibration.size]
    )
    assert 0.0 < estimated_ess <= calibration.size


def test_null_space_generator_is_exactly_paired_and_deterministic() -> None:
    first = diagnostics.generate_null_space_pairs(64, seed=71_000)
    second = diagnostics.generate_null_space_pairs(64, seed=71_000)
    check = diagnostics.check_null_space_pair_equality(first)

    assert check.passed
    assert check.max_feature_absolute_difference == 0.0
    assert check.max_history_absolute_difference == 0.0
    assert np.array_equal(first.external_features_a, first.external_features_b)
    assert np.array_equal(first.external_histories_a, first.external_histories_b)
    assert np.array_equal(first.external_features_a, second.external_features_a)
    assert np.array_equal(first.location_labels_a, second.location_labels_a)
    assert np.array_equal(
        np.sort(first.location_labels_a), np.sort(first.location_labels_b)
    )
    assert not np.array_equal(first.location_labels_a, first.location_labels_b)


def test_pair_checker_detects_external_feature_leakage() -> None:
    clean = diagnostics.generate_null_space_pairs(16, seed=71_001)
    leaked_features = clean.external_features_b.copy()
    leaked_features[:, 0] = clean.location_labels_b
    leaked = replace(clean, external_features_b=leaked_features)

    check = diagnostics.check_null_space_pair_equality(leaked)

    assert not check.passed
    assert not check.features_equal
    assert check.max_feature_absolute_difference > 0.0


def test_location_probe_is_deterministic_on_null_space_pairs() -> None:
    pairs = diagnostics.generate_null_space_pairs(96, seed=71_002)

    first = diagnostics.probe_null_space_pairs(pairs, seed=71_002, epochs=40)
    second = diagnostics.probe_null_space_pairs(pairs, seed=71_002, epochs=40)

    assert first == second
    assert np.isfinite(first.r2)
    assert np.isfinite(first.mae)
    assert first.n_train + first.n_test == 192


def test_location_probe_flags_an_intentionally_leaked_label() -> None:
    rng = np.random.default_rng(71_003)
    features = rng.normal(size=(500, len(diagnostics.E6_FEATURE_NAMES)))
    labels = 0.5 + 0.45 * np.tanh(1.5 * features[:, 0])

    result = diagnostics.fit_location_probe(
        features,
        labels,
        seed=71_003,
        epochs=300,
    )

    assert result.leakage_detected
    assert result.r2 > 0.02
    assert result.relative_mae_improvement > 0.02
