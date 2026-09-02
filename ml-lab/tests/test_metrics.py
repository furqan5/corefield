from __future__ import annotations

import numpy as np
import pytest

from corefield_ml_lab.metrics import (
    accuracy_rank,
    aggregate_trajectory_metrics,
    assert_at_least_ten_seeds,
    e3_win_decision,
    e4_adoption_decision,
    paired_mean_bootstrap_interval,
    safety_rank,
    trajectory_metrics,
)


def test_trajectory_sign_and_peak_convention() -> None:
    truth = np.array([10.0, 12.0, 11.0])
    low = np.array([9.0, 10.0, 10.5])
    scores = trajectory_metrics(low, truth)
    assert scores.mean_signed_error_K < 0.0
    assert scores.signed_peak_error_K == -1.5
    assert scores.rmse_K == pytest.approx(np.sqrt((1.0 + 4.0 + 0.25) / 3.0))


def test_peak_error_compares_maxima_not_one_index() -> None:
    scores = trajectory_metrics([0.0, 9.0, 7.0], [10.0, 0.0, 0.0])
    assert scores.signed_peak_error_K == -1.0


def test_cell_summary_uses_sample_sd_and_unsafe_fraction() -> None:
    rows = [
        trajectory_metrics([1.0, 2.0], [1.0, 2.0]),
        trajectory_metrics([0.0, 1.0], [1.0, 2.0]),
    ]
    cell = aggregate_trajectory_metrics(rows)
    assert cell.rmse_K.mean == 0.5
    assert cell.rmse_K.sample_std == pytest.approx(np.sqrt(0.5))
    assert cell.most_negative_signed_peak_error_K == -1.0
    assert cell.unsafe_low_fraction == 0.5


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    candidate = np.arange(10.0)
    reference = candidate + 1.0
    first = paired_mean_bootstrap_interval(candidate, reference)
    second = paired_mean_bootstrap_interval(candidate, reference)
    assert first == second
    assert first.mean_difference == -1.0
    assert first.lower_95 == -1.0
    assert first.upper_95 == -1.0


def test_e3_win_requires_accuracy_ci_and_safety() -> None:
    nls_rmse = np.ones(10)
    method_rmse = np.full(10, 0.8)
    nls_peak = np.zeros(10)
    safe_method_peak = np.full(10, -0.05)
    assert e3_win_decision(
        method_rmse, nls_rmse, safe_method_peak, nls_peak
    ).beats_nls
    unsafe_method_peak = np.full(10, -0.1000001)
    assert not e3_win_decision(
        method_rmse, nls_rmse, unsafe_method_peak, nls_peak
    ).beats_nls


def test_e4_requires_effect_ci_and_invariant() -> None:
    nls = np.ones(10)
    grey = np.full(10, 0.85)
    assert e4_adoption_decision(
        grey, nls, exact_outside_hull_invariant=True
    )["adopt_in_range_only"]
    assert not e4_adoption_decision(
        grey, nls, exact_outside_hull_invariant=False
    )["adopt_in_range_only"]


def test_accuracy_and_safety_rank_can_disagree() -> None:
    safe = aggregate_trajectory_metrics(
        [trajectory_metrics([2.0, 2.0], [1.0, 1.0])] * 10
    )
    accurate_but_low = aggregate_trajectory_metrics(
        [trajectory_metrics([0.9, 0.9], [1.0, 1.0])] * 10
    )
    cells = {"safe": safe, "low": accurate_but_low}
    assert accuracy_rank(cells) == ("low", "safe")
    assert safety_rank(cells) == ("safe", "low")


def test_seed_rule() -> None:
    assert assert_at_least_ten_seeds(range(10)) == tuple(range(10))
    with pytest.raises(ValueError, match="at least 10"):
        assert_at_least_ten_seeds(range(9))
    with pytest.raises(ValueError, match="unique"):
        assert_at_least_ten_seeds([0] * 10)
