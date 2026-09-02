"""Unit tests for the preregistered synthetic mechanism-test substrate.

These tests exercise construction and invariants only.  They do not execute
an E2--E6 primary test cell or compute a reported experiment metric.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from corefield_ml_lab.synthetic_lab import (
    AMBIENT_AMPLITUDE_K,
    AMBIENT_MEAN_C,
    FEATURE_NAMES,
    FEATURE_SOURCE_LAGS_MIN,
    MODEL_DT_S,
    REFERENCE_BUDGETS,
    REFERENCE_OFFSETS_MIN,
    SENSOR_NOISE_SIGMA_K,
    STRUCTURAL_MISMATCH_CONSTANTS,
    TOP_OIL_SAMPLE_S,
    TRUTH_DT_S,
    TruthRecord,
    build_feature_frame,
    fit_train_standardizer,
    feature_matrix_at_times,
    make_e3_schedule,
    make_in_range_test_schedule,
    make_schedule,
    make_train_schedule,
    make_validation_schedule,
    observe_hotspot_references,
    observe_record,
    physics_constants,
    reference_candidate_indices,
    simulate_truth,
    sparse_reference_indices,
)


@pytest.fixture(scope="module")
def train_schedule():
    return make_train_schedule()


@pytest.fixture(scope="module")
def validation_schedule():
    return make_validation_schedule()


@pytest.fixture(scope="module")
def train_truth(train_schedule):
    # Training data are permitted implementation fixtures, not hidden test access.
    return simulate_truth(train_schedule, physics_mode="matched")


@pytest.fixture(scope="module")
def validation_truth(validation_schedule):
    return simulate_truth(validation_schedule, physics_mode="matched")


def test_units_grid_nominal_constants_and_temperature_identity(train_schedule, train_truth) -> None:
    assert np.all(np.diff(train_schedule.time_s) == TRUTH_DT_S)
    assert train_schedule.duration_h == 48.0
    assert MODEL_DT_S == 120.0
    assert TOP_OIL_SAMPLE_S == 300.0
    assert SENSOR_NOISE_SIGMA_K == 0.5

    assert np.min(train_schedule.ambient_C) == pytest.approx(
        AMBIENT_MEAN_C - AMBIENT_AMPLITUDE_K
    )
    assert np.max(train_schedule.ambient_C) == pytest.approx(
        AMBIENT_MEAN_C + AMBIENT_AMPLITUDE_K
    )
    assert np.allclose(
        train_truth.gradient_K,
        train_truth.hotspot_C - train_truth.top_oil_C,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(train_truth.gradient_K > 0.0)
    assert train_truth.top_oil_C.min() > -50.0  # degC sanity, not a design limit.


def test_schedule_bounds_splits_and_distinctness(train_schedule, validation_schedule) -> None:
    in_range = make_in_range_test_schedule()

    assert train_schedule.split == "train"
    assert train_schedule.load_hull_pu == pytest.approx((0.60, 0.95))
    assert validation_schedule.split == "validation"
    assert validation_schedule.duration_h == 24.0
    assert validation_schedule.load_pu.min() >= 0.65
    assert validation_schedule.load_pu.max() <= 0.92
    assert in_range.split == "in_range_test"
    assert in_range.load_pu.min() >= 0.62
    assert in_range.load_pu.max() <= 0.94
    assert not np.array_equal(validation_schedule.load_pu, in_range.load_pu)
    assert not np.shares_memory(train_schedule.load_pu, validation_schedule.load_pu)
    assert not train_schedule.load_pu.flags.writeable


@pytest.mark.parametrize("target", (1.00, 1.15, 1.30, 1.60))
def test_e3_schedule_has_exact_four_hour_step_without_running_truth(target: float) -> None:
    schedule = make_e3_schedule(target)
    boundary = int(4.0 * 3600.0 / TRUTH_DT_S)

    assert schedule.duration_h == 8.0
    assert np.all(schedule.load_pu[:boundary] == 0.75)
    assert np.all(schedule.load_pu[boundary:] == target)
    assert schedule.event_time_s.tolist() == [4.0 * 3600.0]
    assert schedule.target_load_pu == target


def test_schedule_dispatch_refuses_ambiguous_e3_target() -> None:
    assert make_schedule("train").split == "train"
    with pytest.raises(ValueError, match="requires e3_target_load_pu"):
        make_schedule("e3_test")
    with pytest.raises(ValueError, match="one of"):
        make_e3_schedule(1.31)
    with pytest.raises(ValueError, match="only valid"):
        make_schedule("validation", e3_target_load_pu=1.30)


def test_structural_mismatch_is_only_the_declared_oil_exponent_slope() -> None:
    matched = physics_constants("matched")
    mismatch = physics_constants("structural_mismatch")

    assert mismatch is STRUCTURAL_MISMATCH_CONSTANTS
    assert mismatch.x1 == pytest.approx(0.21)
    assert mismatch.y1 == 0.0
    assert mismatch.x + mismatch.x1 * (1.60 - 1.0) == pytest.approx(0.926)
    assert (mismatch.y, mismatch.k11, mismatch.k21, mismatch.k22) == (
        matched.y,
        matched.k11,
        matched.k21,
        matched.k22,
    )


def test_sensor_noise_is_seed_deterministic_and_five_minute_sampled(train_truth) -> None:
    first = observe_record(train_truth, seed=31000)
    repeated = observe_record(train_truth, seed=31000)
    different = observe_record(train_truth, seed=31001)

    assert np.array_equal(first.top_oil_C, repeated.top_oil_C)
    assert not np.array_equal(first.top_oil_C, different.top_oil_C)
    assert np.all(np.diff(first.top_oil_time_s) == TOP_OIL_SAMPLE_S)
    assert np.array_equal(first.top_oil_time_s, train_truth.schedule.time_s[first.top_oil_index])
    assert first.top_oil_C.size < train_truth.top_oil_C.size
    assert not np.shares_memory(first.top_oil_C, train_truth.top_oil_C)


def test_reference_budgets_are_unique_nested_and_placed_before_noise(
    train_schedule, train_truth
) -> None:
    candidates = reference_candidate_indices(train_schedule)
    assert candidates.size == max(REFERENCE_BUDGETS)
    assert np.unique(candidates).size == candidates.size

    previous: set[int] = set()
    for budget in REFERENCE_BUDGETS:
        selected = sparse_reference_indices(train_schedule, budget)
        current = set(map(int, selected))
        assert selected.size == budget
        assert previous <= current
        previous = current

    # Offset-major round robin: all seven 3 min reads precede the 8 min reads.
    n_events = train_schedule.event_time_s.size
    expected_first_round = (
        train_schedule.event_time_s + REFERENCE_OFFSETS_MIN[0] * 60.0
    ) / TRUTH_DT_S
    assert np.array_equal(candidates[:n_events], expected_first_round.astype(np.intp))

    refs_20 = observe_hotspot_references(train_truth, budget=20, seed=31000)
    refs_50 = observe_hotspot_references(train_truth, budget=50, seed=31000)
    values_50 = dict(zip(refs_50.index.tolist(), refs_50.temperature_C.tolist(), strict=True))
    assert all(values_50[int(i)] == value for i, value in zip(refs_20.index, refs_20.temperature_C))
    assert np.allclose(
        refs_20.temperature_C - refs_20.truth_temperature_C,
        observe_hotspot_references(train_truth, budget=20, seed=31000).temperature_C
        - refs_20.truth_temperature_C,
    )


def test_reference_noise_stream_is_independent_of_top_oil_stream(train_truth) -> None:
    observed = observe_record(train_truth, seed=1234)
    refs = observe_hotspot_references(train_truth, budget=50, seed=1234)
    oil_noise = observed.top_oil_C - train_truth.top_oil_C[observed.top_oil_index]
    ref_noise = refs.temperature_C - refs.truth_temperature_C

    assert not np.array_equal(oil_noise[: ref_noise.size], ref_noise)


def test_noise_substreams_are_independent_across_splits_and_schedules(
    train_truth, validation_truth
) -> None:
    seed = 31000
    train_oil = observe_record(train_truth, seed=seed)
    validation_oil = observe_record(validation_truth, seed=seed)
    assert not np.array_equal(
        train_oil.top_oil_C[: validation_oil.top_oil_C.size]
        - train_truth.top_oil_C[train_oil.top_oil_index[: validation_oil.top_oil_C.size]],
        validation_oil.top_oil_C
        - validation_truth.top_oil_C[validation_oil.top_oil_index],
    )

    train_refs = observe_hotspot_references(train_truth, budget=20, seed=seed)
    validation_refs = observe_hotspot_references(
        validation_truth, budget=20, seed=seed
    )
    assert not np.array_equal(
        train_refs.temperature_C - train_refs.truth_temperature_C,
        validation_refs.temperature_C - validation_refs.truth_temperature_C,
    )

    first_e3 = simulate_truth(make_e3_schedule(1.00), physics_mode="structural_mismatch")
    second_e3 = simulate_truth(make_e3_schedule(1.15), physics_mode="structural_mismatch")
    first_noise = observe_record(first_e3, seed=seed).top_oil_C - first_e3.top_oil_C[
        observe_record(first_e3, seed=seed).top_oil_index
    ]
    second_noise = observe_record(second_e3, seed=seed).top_oil_C - second_e3.top_oil_C[
        observe_record(second_e3, seed=seed).top_oil_index
    ]
    assert not np.array_equal(first_noise, second_noise)


def test_features_have_exact_lags_and_use_only_noisy_sampled_top_oil(train_truth) -> None:
    observed = observe_record(train_truth, seed=31000)
    frame = build_feature_frame(observed)
    row = 37
    time_s = frame.time_s[row]

    assert frame.X.shape[1] == 9
    assert frame.feature_names == FEATURE_NAMES
    assert frame.time_s[0] == 0.0
    assert np.all(np.diff(frame.time_s) == MODEL_DT_S)
    assert np.allclose(
        time_s - frame.source_time_s[row],
        np.minimum(time_s, np.asarray(FEATURE_SOURCE_LAGS_MIN) * 60.0),
    )
    for column, lag_min in enumerate((0.0, 6.0, 16.0, 60.0, 180.0)):
        expected = np.interp(
            time_s - lag_min * 60.0,
            train_truth.schedule.time_s,
            train_truth.schedule.load_pu,
        )
        assert frame.X[row, column] == pytest.approx(expected)
    assert frame.X[row, 5] == pytest.approx(
        np.interp(time_s, train_truth.schedule.time_s, train_truth.schedule.ambient_C)
    )
    for column, lag_min in zip((6, 7, 8), (0.0, 16.0, 60.0), strict=True):
        expected = np.interp(
            time_s - lag_min * 60.0,
            observed.top_oil_time_s,
            observed.top_oil_C,
        )
        assert frame.X[row, column] == pytest.approx(expected)

    # Alter dense hidden oil between five-minute samples.  Features cannot see it.
    modified_dense_oil = train_truth.top_oil_C.copy()
    unsampled = np.ones(modified_dense_oil.size, dtype=bool)
    unsampled[observed.top_oil_index] = False
    modified_dense_oil[unsampled] += 100.0
    modified_truth = TruthRecord(
        schedule=train_truth.schedule,
        physics_mode=train_truth.physics_mode,
        top_oil_C=modified_dense_oil,
        hotspot_C=modified_dense_oil + train_truth.gradient_K,
        gradient_K=train_truth.gradient_K,
    )
    modified_observed = replace(observed, truth=modified_truth)
    assert np.array_equal(build_feature_frame(modified_observed).X, frame.X)


def test_sparse_reference_features_are_contemporaneous(train_truth) -> None:
    observed = observe_record(train_truth, seed=31000)
    references = observe_hotspot_references(train_truth, budget=3, seed=31000)
    features, source_times = feature_matrix_at_times(observed, references.time_s)
    assert features.shape == (3, 9)
    assert np.array_equal(source_times[:, 0], references.time_s)
    assert np.array_equal(source_times[:, 5], references.time_s)
    assert np.array_equal(source_times[:, 6], references.time_s)
    assert np.allclose(
        features[:, 0],
        np.interp(
            references.time_s,
            train_truth.schedule.time_s,
            train_truth.schedule.load_pu,
        ),
    )


def test_train_only_standardization_and_split_isolation(train_truth, validation_truth) -> None:
    train_frame = build_feature_frame(observe_record(train_truth, seed=31000))
    validation_frame = build_feature_frame(observe_record(validation_truth, seed=31000))

    standardizer = fit_train_standardizer(train_frame)
    standardized_train = standardizer.transform(train_frame)
    standardized_validation = standardizer.transform(validation_frame)

    assert standardizer.fitted_split == "train"
    assert np.allclose(np.mean(standardized_train, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(standardized_train, axis=0), 1.0, atol=1e-12)
    assert not np.allclose(np.mean(standardized_validation, axis=0), 0.0, atol=1e-3)
    assert train_frame.split == "train"
    assert validation_frame.split == "validation"
    assert not np.shares_memory(train_frame.X, validation_frame.X)
    with pytest.raises(ValueError, match="split='train'"):
        fit_train_standardizer(validation_frame)


def test_seed_and_budget_validation(train_truth) -> None:
    with pytest.raises(TypeError, match="integer"):
        observe_record(train_truth, seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        observe_record(train_truth, seed=-1)
    with pytest.raises(ValueError, match="budget"):
        sparse_reference_indices(train_truth.schedule, 5)
