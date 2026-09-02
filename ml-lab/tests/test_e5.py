from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from corefield_ml_lab import e5
from corefield_ml_lab.classical import NLSModel, NLSRefusal
from corefield_ml_lab.synthetic_lab import NOMINAL_PARAMS


def _model() -> NLSModel:
    return NLSModel(
        params=NOMINAL_PARAMS,
        residual_rmse_K=1.0,
        oil_residual_rmse_K=0.5,
        hotspot_residual_rmse_K=1.0,
        jacobian_condition=10.0,
        warnings=(),
    )


def test_configuration_resolves_every_preregistered_count_and_band() -> None:
    config = e5.e5_configuration()

    assert config["alpha"] == 0.05
    assert config["fit"]["fit_count"] == 1
    assert config["fit"]["reference_budget"] == 20
    assert config["fit"]["seed"] == 61_000
    assert config["ordinary"]["calibration"]["episodes"] == 200
    assert config["ordinary"]["exchangeable_test"]["episodes"] == 1_000
    bands = config["ordinary"]["strict_shift_bands"]
    assert [band["centre_pu"] for band in bands] == [1.0, 1.15, 1.3, 1.6]
    assert all(band["upper_pu"] - band["lower_pu"] == pytest.approx(0.05) for band in bands)
    assert config["weighted"]["unlabeled_target_episodes"] == 2_000


def test_episode_draws_are_deterministic_independent_streams_and_in_support() -> None:
    first = e5.draw_e5_episode_loads(
        calibration_episodes=12,
        test_episodes=15,
        unlabeled_target_episodes=20,
    )
    second = e5.draw_e5_episode_loads(
        calibration_episodes=12,
        test_episodes=15,
        unlabeled_target_episodes=20,
    )

    assert np.array_equal(first.calibration_loads_pu, second.calibration_loads_pu)
    assert np.array_equal(first.weighted_test_loads_pu, second.weighted_test_loads_pu)
    assert not np.array_equal(
        first.calibration_loads_pu, first.exchangeable_test_loads_pu[:12]
    )
    assert np.all((first.calibration_loads_pu >= 0.60) & (first.calibration_loads_pu < 0.90))
    assert np.all((first.weighted_test_loads_pu >= 0.60) & (first.weighted_test_loads_pu <= 0.90))
    assert first.unlabeled_target_loads_pu.size == 20
    for centre, loads in first.strict_test_loads_by_centre:
        assert loads.size == 15
        assert np.all(loads >= centre - 0.025)
        assert np.all(loads < centre + 0.025)
        assert not loads.flags.writeable


def test_primary_draw_validation_checks_every_frozen_support_and_finiteness() -> None:
    draws = e5.draw_e5_episode_loads()
    e5.validate_primary_episode_draws(draws)

    calibration = draws.calibration_loads_pu.copy()
    calibration[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        e5.validate_primary_episode_draws(
            replace(draws, calibration_loads_pu=calibration)
        )

    exchangeable = draws.exchangeable_test_loads_pu.copy()
    exchangeable[0] = 0.900001
    with pytest.raises(RuntimeError, match="outside frozen support"):
        e5.validate_primary_episode_draws(
            replace(draws, exchangeable_test_loads_pu=exchangeable)
        )

    unlabeled = draws.unlabeled_target_loads_pu.copy()
    unlabeled[0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        e5.validate_primary_episode_draws(
            replace(draws, unlabeled_target_loads_pu=unlabeled)
        )

    weighted = draws.weighted_test_loads_pu.copy()
    weighted[0] = 0.599999
    with pytest.raises(RuntimeError, match="outside frozen support"):
        e5.validate_primary_episode_draws(
            replace(draws, weighted_test_loads_pu=weighted)
        )

    strict = list(draws.strict_test_loads_by_centre)
    strict_loads = strict[2][1].copy()
    strict_loads[0] = strict[2][0] + 0.025001
    strict[2] = (strict[2][0], strict_loads)
    with pytest.raises(RuntimeError, match="outside frozen support"):
        e5.validate_primary_episode_draws(
            replace(draws, strict_test_loads_by_centre=tuple(strict))
        )


def test_primary_draw_validation_rejects_nonvector_even_with_correct_count() -> None:
    draws = e5.draw_e5_episode_loads()
    reshaped = draws.calibration_loads_pu.reshape(20, 10)
    with pytest.raises(RuntimeError, match="shape/count"):
        e5.validate_primary_episode_draws(
            replace(draws, calibration_loads_pu=reshaped)
        )


def test_episode_schedule_reuses_e3_grid_and_steps_at_four_hours() -> None:
    schedule = e5.make_e5_episode_schedule(0.83)

    step = int(4.0 * 3600.0 / 30.0)
    assert schedule.duration_h == 8.0
    assert schedule.target_load_pu == 0.83
    assert np.all(schedule.load_pu[:step] == 0.75)
    assert np.all(schedule.load_pu[step:] == 0.83)
    assert np.all(schedule.load_pu_half[:step] == 0.75)
    assert np.all(schedule.load_pu_half[step:] == 0.83)
    assert not schedule.load_pu.flags.writeable


def test_episode_evaluator_uses_full_trajectory_max_with_monkeypatched_physics() -> None:
    @dataclass
    class FakeSchedule:
        target_load_pu: float

    calls: list[tuple[str, float, str | None]] = []

    def schedule_factory(load: float):
        calls.append(("schedule", load, None))
        return FakeSchedule(load)

    def truth_simulator(schedule, *, physics_mode: str):
        calls.append(("truth", schedule.target_load_pu, physics_mode))
        return SimpleNamespace(hotspot_C=np.array([10.0, 20.0 + schedule.target_load_pu]))

    def predictor(_model, schedule):
        calls.append(("prediction", schedule.target_load_pu, None))
        return SimpleNamespace(hotspot_C=np.array([9.0, 18.0 + schedule.target_load_pu]))

    batch = e5.evaluate_episode_loads(
        [0.70, 1.10],
        _model(),
        schedule_factory=schedule_factory,
        truth_simulator=truth_simulator,
        model_predictor=predictor,
    )

    assert np.allclose(batch.true_peaks_C, [20.70, 21.10])
    assert np.allclose(batch.predicted_peaks_C, [18.70, 19.10])
    assert np.allclose(batch.conformity_scores_K, [2.0, 2.0])
    assert [call[0] for call in calls] == [
        "schedule", "truth", "prediction", "schedule", "truth", "prediction"
    ]


def test_fit_once_uses_structural_mismatch_seed_and_twenty_references() -> None:
    calls: list[tuple[str, object]] = []
    model = _model()

    def truth_simulator(schedule, *, physics_mode: str):
        calls.append(("truth_mode", physics_mode))
        return SimpleNamespace(schedule=schedule)

    def record_observer(truth, *, seed: int):
        calls.append(("record_seed", seed))
        return "observed"

    def reference_observer(truth, *, budget: int, seed: int):
        calls.append(("reference", (budget, seed)))
        return "references"

    def fitter(observed, references):
        calls.append(("fit", (observed, references)))
        return model

    result = e5.fit_e5_nls_once(
        truth_simulator=truth_simulator,
        record_observer=record_observer,
        reference_observer=reference_observer,
        fitter=fitter,
    )

    assert result is model
    assert calls == [
        ("truth_mode", "structural_mismatch"),
        ("record_seed", 61_000),
        ("reference", (20, 61_000)),
        ("fit", ("observed", "references")),
    ]


def test_fit_once_turns_nls_refusal_into_explicit_failure() -> None:
    with pytest.raises(RuntimeError, match="insufficient"):
        e5.fit_e5_nls_once(
            truth_simulator=lambda schedule, physics_mode: "truth",
            record_observer=lambda truth, seed: "observed",
            reference_observer=lambda truth, budget, seed: "references",
            fitter=lambda observed, references: NLSRefusal(
                "ValueError", "insufficient information"
            ),
        )


def test_ordinary_case_uses_one_correction_and_is_json_strict() -> None:
    batch = e5.EpisodeBatch(
        target_loads_pu=np.array([0.7, 0.8, 0.9]),
        true_peaks_C=np.array([102.0, 104.0, 106.0]),
        predicted_peaks_C=np.array([100.0, 101.0, 102.0]),
    )
    # At alpha=0.25 and n=7, rank ceil(8 * .75)=6 => correction 6 K.
    payload = e5.ordinary_case_payload(
        batch, np.arange(1.0, 8.0), alpha=0.25
    )

    assert payload["correction_quantile_K"] == 6.0
    assert payload["metrics"]["empirical_coverage"] == 1.0
    assert all(row["upper_width_K"] == 6.0 for row in payload["rows"])
    json.dumps(payload, allow_nan=False)


def test_ratio_comparison_reports_signed_and_absolute_error() -> None:
    payload = e5.ratio_comparison_payload([1.0, 2.0, 3.0], [1.5, 2.0, 2.5])

    assert payload["bias"] == pytest.approx(0.0)
    assert payload["mean_absolute_error"] == pytest.approx(1.0 / 3.0)
    assert payload["root_mean_square_error"] == pytest.approx(np.sqrt(1.0 / 6.0))
    assert payload["correlation"] == pytest.approx(1.0)


def test_strict_weighted_support_is_unbounded_json_safe_and_formally_contains() -> None:
    calibration_loads = np.linspace(0.60, 0.90, 20)
    calibration_scores = np.linspace(-1.0, 2.0, 20)
    batch = e5.EpisodeBatch(
        target_loads_pu=np.array([0.98, 1.12, 1.29]),
        true_peaks_C=np.array([110.0, 120.0, 130.0]),
        predicted_peaks_C=np.array([100.0, 105.0, 110.0]),
    )

    payload = e5.strict_weighted_support_payload(
        batch,
        calibration_scores,
        calibration_loads,
    )

    assert payload["formal_containment_only_via_unbounded_limits"] == 1.0
    assert payload["metrics"]["empirical_coverage"] == 1.0
    assert payload["metrics"]["finite_availability"] == 0.0
    assert payload["metrics"]["mean_finite_upper_width_K"] is None
    assert payload["calibration_weight_effective_sample_size"] is None
    assert payload["effective_sample_size_status"].startswith("not defined")
    assert all(row["upper_limit_C"] is None for row in payload["rows"])
    assert all(row["target_infinity_mass"] == 1.0 for row in payload["rows"])
    json.dumps(payload, allow_nan=False)


def test_weighted_overlap_fixture_uses_estimated_weights_and_reports_ess() -> None:
    calibration_loads = np.linspace(0.60, 0.90, 20)
    calibration_scores = np.linspace(-2.0, 3.0, 20)
    calibration_weights = np.linspace(0.5, 1.5, 20)
    batch = e5.EpisodeBatch(
        target_loads_pu=np.array([0.70, 0.80]),
        true_peaks_C=np.array([101.0, 103.0]),
        predicted_peaks_C=np.array([100.0, 101.0]),
    )

    payload = e5.weighted_case_payload(
        batch,
        calibration_scores,
        calibration_loads,
        calibration_weights,
        target_weights=np.array([0.5, 0.75]),
        calibration_densities_at_queries=np.array([1.0, 1.0]),
        alpha=0.20,
    )

    assert payload["calibration_weight_effective_sample_size"] == pytest.approx(
        e5.effective_sample_size(calibration_weights)
    )
    assert payload["metrics"]["n_total"] == 2
    assert all(row["interval_status"] == "finite" for row in payload["rows"])
    json.dumps(payload, allow_nan=False)


def test_primary_runner_reaches_prerequisites_then_write_once_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(e5, "enforce_cpu_only_environment", lambda: events.append("cpu"))
    monkeypatch.setattr(e5, "require_e1_passed", lambda root: events.append("e1"))
    monkeypatch.setattr(
        e5,
        "require_completed_primary",
        lambda *args, **kwargs: events.append(str(args[1])),
    )

    class ReachedClaim(RuntimeError):
        pass

    def begin(*args, **kwargs):
        events.append("claim")
        assert kwargs["experiment"] == "e5"
        assert kwargs["seeds"] == [61_000]
        assert kwargs["command"][-1] == "e5"
        raise ReachedClaim

    monkeypatch.setattr(e5, "begin_primary_run", begin)
    with pytest.raises(ReachedClaim):
        e5.run_e5_primary(tmp_path)
    assert events == ["cpu", "e1", "e3", "e2", "e4", "claim"]


def test_primary_runner_executes_one_fit_and_full_preregistered_episode_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(e5, "require_e1_passed", lambda root: None)
    monkeypatch.setattr(e5, "require_completed_primary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        e5,
        "begin_primary_run",
        lambda *args, **kwargs: SimpleNamespace(run_id="e5-integration"),
    )
    fit_seeds: list[int] = []

    def fit_once(*, seed: int):
        fit_seeds.append(seed)
        return _model()

    evaluated_sizes: list[int] = []

    def evaluate(loads, model):
        load = np.asarray(loads, dtype=np.float64)
        evaluated_sizes.append(load.size)
        predicted = 70.0 + 20.0 * load
        # A deterministic conditional outcome keeps episode draws iid while
        # making the orchestration test independent of the expensive ODE.
        truth = predicted + 4.0 * (load - 0.75)
        return e5.EpisodeBatch(load, truth, predicted)

    persisted: dict[str, object] = {}

    def finish(run, aggregate):
        persisted.update(aggregate)
        return {"status": "completed"}

    monkeypatch.setattr(e5, "fit_e5_nls_once", fit_once)
    monkeypatch.setattr(e5, "evaluate_episode_loads", evaluate)
    monkeypatch.setattr(e5, "finish_primary_run", finish)
    result = e5.run_e5_primary(tmp_path)

    assert fit_seeds == [61_000]
    assert evaluated_sizes == [200, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000]
    assert result["run_id"] == "e5-integration"
    assert persisted["configuration"]["ordinary"]["calibration"]["episodes"] == 200
    json.dumps(persisted, allow_nan=False)
