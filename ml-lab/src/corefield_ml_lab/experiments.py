"""Preregistered experiment orchestration.

Model construction uses train and validation records only.  Primary test
truth is generated exclusively inside the corresponding ``run_*_primary``
function, after its write-once access sentinel has been created.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .classical import (
    NLSModel,
    NLSRefusal,
    fit_nls,
    predict_generic,
    predict_nls,
    prediction_on_feature_frame,
    parameter_percent_errors,
)
from .e1 import (
    E1_SEEDS,
    PrivateFieldConfig,
    evaluate_overall_e1_gate,
    run_private_field_e1,
    run_synthetic_e1,
)
from .evidence import require_completed_primary, require_e1_passed
from .metrics import (
    CellMetrics,
    TrajectoryMetrics,
    accuracy_rank,
    aggregate_trajectory_metrics,
    assert_at_least_ten_seeds,
    e4_adoption_decision,
    e3_win_decision,
    safety_rank,
    trajectory_metrics,
)
from .ml_methods import (
    HullAwareResidualFit,
    PinnFit,
    PinnTrajectory,
    SupervisedFit,
    TrainingConfig,
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    fit_pinn,
    fit_plain_nn,
    fit_residual_nn,
)
from .runstore import begin_primary_run, finish_primary_run, record_primary_failures
from .runtime import enforce_cpu_only_environment, require_torch_cpu_only
from .synthetic_lab import (
    FeatureFrame,
    TruthRecord,
    build_feature_frame,
    feature_matrix_at_times,
    make_e3_schedule,
    make_in_range_test_schedule,
    make_train_schedule,
    make_validation_schedule,
    observe_hotspot_references,
    observe_record,
    simulate_truth,
)


E3_SEEDS: tuple[int, ...] = tuple(range(31_000, 31_010))
E3_TARGET_LOADS_PU: tuple[float, ...] = (1.00, 1.15, 1.30, 1.60)
E3_REFERENCE_BUDGET = 20
E2_SEEDS: tuple[int, ...] = tuple(range(41_000, 41_010))
E2_REFERENCE_BUDGETS: tuple[int, ...] = (3, 4, 6, 10, 20, 50)
E4_SEEDS: tuple[int, ...] = E3_SEEDS
E4_REFERENCE_BUDGET = 20
METHOD_NAMES: tuple[str, ...] = (
    "nls",
    "plain_nn",
    "pinn",
    "greybox",
    "generic_iec",
)


def _recorded_primary_command(
    experiment: str,
    *,
    override: bool,
    override_reason: str | None,
) -> list[str]:
    """Build the auditable CLI invocation while omitting private E1 paths."""

    command = [sys.executable, "-m", "corefield_ml_lab", experiment]
    if override:
        reason = "" if override_reason is None else override_reason.strip()
        if not reason:
            raise ValueError("override_reason is required when override=True")
        command.extend(["--override", "--override-reason", reason])
    elif override_reason is not None:
        raise ValueError("override_reason requires override=True")
    return command


@record_primary_failures
def run_e1_primary(
    repository: str | Path,
    *,
    private_config: PrivateFieldConfig | None = None,
    override: bool = False,
    override_reason: str | None = None,
) -> Mapping[str, object]:
    """Execute and persist the synthetic and optional private E1 gate."""

    enforce_cpu_only_environment()
    repository_path = Path(repository).resolve()
    configuration = {
        "day": "C",
        "private_field_access_configured": private_config is not None,
        "private_output_policy": "aggregate metrics only; no stdout or rows retained",
        "seeds": list(E1_SEEDS),
    }
    run = begin_primary_run(
        repository_path,
        experiment="e1",
        configuration=configuration,
        seeds=E1_SEEDS,
        command=_recorded_primary_command(
            "e1", override=override, override_reason=override_reason
        ),
        override=override,
        override_reason=override_reason,
    )
    synthetic = run_synthetic_e1()
    field = run_private_field_e1(private_config) if private_config is not None else None
    gate = evaluate_overall_e1_gate(synthetic, field)
    aggregate = {
        "configuration": configuration,
        "field": None if field is None else field.as_dict(),
        "gate": {
            "field": None
            if gate.field is None
            else {
                "checks": [
                    {
                        "absolute_error_K": check.absolute_error,
                        "actual_K": check.actual,
                        "label": check.label,
                        "passed": check.passed,
                        "target_K": check.target,
                        "tolerance_K": check.tolerance,
                    }
                    for check in gate.field.checks
                ],
                "status": gate.field.status,
            },
            "overall_status": gate.status,
            "synthetic": {
                "checks": [
                    {
                        "absolute_error_K": check.absolute_error,
                        "actual_K": check.actual,
                        "label": check.label,
                        "passed": check.passed,
                        "target_K": check.target,
                        "tolerance_K": check.tolerance,
                    }
                    for check in gate.synthetic.checks
                ],
                "status": gate.synthetic.status,
            },
        },
        "schema_version": 1,
        "synthetic": synthetic.as_dict(),
    }
    final = finish_primary_run(run, aggregate)
    return {"aggregate": aggregate, "manifest": dict(final), "run_id": run.run_id}


@dataclass(slots=True)
class SeedMethodFits:
    """One seed's fitted handles and serialisable train/validation evidence."""

    seed: int
    nls: NLSModel | NLSRefusal
    plain_nn: SupervisedFit | None
    pinn: PinnFit | None
    greybox: HullAwareResidualFit | None
    failures: dict[str, Mapping[str, str]]
    training_record: dict[str, object]


def e3_configuration() -> dict[str, object]:
    """Complete resolved configuration hashed into the E3 access claim."""

    config = TrainingConfig()
    return {
        "evaluation": {
            "loads_pu": list(E3_TARGET_LOADS_PU),
            "score_window": "time_s >= 14400 (four-hour target plateau)",
            "truth": "noise-free hidden hot-spot temperature",
        },
        "features": {
            "grid_s": 120.0,
            "pre_record_lags": "opening value (declared settled prehistory)",
            "reference_alignment": "features evaluated at exact reference timestamps",
            "top_oil": "linear interpolation of noisy five-minute samples",
            "top_oil_loss": "state evaluated at each actual five-minute sample",
        },
        "noise_substreams": "SHA-256-stable independent seed/sensor/schedule streams",
        "methods": list(METHOD_NAMES),
        "neural_training": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "minimum_normalized_improvement": config.minimum_normalized_improvement,
            "intraop_threads": config.intraop_threads,
        },
        "physics_mode": "structural_mismatch",
        "reference_budget": E3_REFERENCE_BUDGET,
        "schedule_amplitudes_frozen_after_prereg_before_primary": {
            "train_post_event_loads_pu": [0.95, 0.65, 0.90, 0.70, 0.93, 0.62, 0.88],
            "validation_post_event_loads_pu": [0.90, 0.65, 0.92, 0.70],
        },
        "seeds": list(E3_SEEDS),
    }


def e2_configuration() -> dict[str, object]:
    """Complete resolved E2 configuration hashed into its access claim."""

    config = TrainingConfig()
    return {
        "evaluation_split": "in_range_test_24h, noise-free hidden truth",
        "methods": ["nls", "plain_nn", "pinn"],
        "neural_training": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "minimum_normalized_improvement": config.minimum_normalized_improvement,
            "intraop_threads": config.intraop_threads,
        },
        "physics_mode": "matched",
        "noise_substreams": "SHA-256-stable independent seed/sensor/schedule streams",
        "measurement_losses": {
            "hotspot": "features evaluated at exact sparse-reference timestamps",
            "top_oil": "PINN state evaluated at every actual five-minute sample",
        },
        "reference_budgets": list(E2_REFERENCE_BUDGETS),
        "reference_policy": "first N candidates, nested; validation uses same N",
        "seeds": list(E2_SEEDS),
    }


def e4_configuration() -> dict[str, object]:
    """Resolved hull-aware residual protocol."""

    config = TrainingConfig()
    return {
        "adoption_rule": {
            "minimum_paired_mean_rmse_improvement_K": 0.10,
            "paired_bootstrap_ci_must_exclude_zero": True,
            "outside_hull_residual_must_be_positive_zero_bit_exact": True,
        },
        "evaluation_split": "in_range_test_24h, noise-free hidden truth",
        "methods": ["nls", "greybox"],
        "neural_training": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "minimum_normalized_improvement": config.minimum_normalized_improvement,
            "intraop_threads": config.intraop_threads,
        },
        "physics_mode": "structural_mismatch",
        "noise_substreams": "SHA-256-stable independent seed/sensor/schedule streams",
        "measurement_losses": {
            "hotspot": "features evaluated at exact sparse-reference timestamps"
        },
        "reference_budget": E4_REFERENCE_BUDGET,
        "seeds": list(E4_SEEDS),
    }


def _failure(error: BaseException) -> Mapping[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _reference_rmse(prediction: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(labels)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((prediction[mask] - labels[mask]) ** 2)))


def _pinn_trajectory(
    frame: FeatureFrame,
    truth: TruthRecord,
    reference_features: np.ndarray,
    reference_hotspot_C: np.ndarray,
    top_oil_measurement_features: np.ndarray,
    top_oil_measurement_values_C: np.ndarray,
) -> PinnTrajectory:
    index = frame.truth_index
    return PinnTrajectory(
        features=frame.X,
        time_s=frame.time_s,
        load_pu=truth.schedule.load_pu[index],
        ambient_C=truth.schedule.ambient_C[index],
        # Column 6 is current top oil interpolated only from noisy 5 min samples.
        measured_top_oil_C=frame.X[:, 6],
        hotspot_reference_C=np.full(frame.time_s.shape, np.nan, dtype=np.float64),
        reference_features=reference_features,
        reference_hotspot_C=reference_hotspot_C,
        top_oil_measurement_features=top_oil_measurement_features,
        top_oil_measurement_values_C=top_oil_measurement_values_C,
    )


def fit_seed_methods(
    seed: int,
    *,
    physics_mode: str,
    reference_budget: int,
    training_config: TrainingConfig | None = None,
    train_truth: TruthRecord | None = None,
    validation_truth: TruthRecord | None = None,
) -> SeedMethodFits:
    """Fit all declared methods using only train/validation data."""

    config = TrainingConfig() if training_config is None else training_config
    train = (
        simulate_truth(make_train_schedule(), physics_mode=physics_mode)
        if train_truth is None
        else train_truth
    )
    validation = (
        simulate_truth(make_validation_schedule(), physics_mode=physics_mode)
        if validation_truth is None
        else validation_truth
    )
    if train.split != "train" or validation.split != "validation":
        raise ValueError("fit_seed_methods requires train and validation truth")

    observed_train = observe_record(train, seed=seed)
    observed_validation = observe_record(validation, seed=seed)
    references_train = observe_hotspot_references(
        train, budget=reference_budget, seed=seed
    )
    references_validation = observe_hotspot_references(
        validation, budget=reference_budget, seed=seed
    )
    frame_train = build_feature_frame(observed_train)
    frame_validation = build_feature_frame(observed_validation)
    reference_features_train, _ = feature_matrix_at_times(
        observed_train, references_train.time_s
    )
    reference_features_validation, _ = feature_matrix_at_times(
        observed_validation, references_validation.time_s
    )
    oil_features_train, _ = feature_matrix_at_times(
        observed_train, observed_train.top_oil_time_s
    )

    failures: dict[str, Mapping[str, str]] = {}
    record: dict[str, object] = {
        "reference_alignment_max_horizon_s": 0.0,
        "reference_budget": reference_budget,
        "seed": seed,
        "train_feature_rows": int(frame_train.X.shape[0]),
        "validation_feature_rows": int(frame_validation.X.shape[0]),
    }

    nls = fit_nls(observed_train, references_train)
    if isinstance(nls, NLSModel):
        record["nls"] = nls.as_dict()
    else:
        record["nls"] = {"available": False, "refusal": nls.as_dict()}

    plain: SupervisedFit | None = None
    try:
        plain = fit_plain_nn(
            reference_features_train,
            references_train.temperature_C,
            reference_features_validation,
            references_validation.temperature_C,
            seed=seed,
            config=config,
            standardizer_features=frame_train.X,
        )
        record["plain_nn"] = {
            "available": True,
            "best_epoch": plain.best_epoch,
            "epochs_ran": plain.epochs_ran,
            "stopped_early": plain.stopped_early,
            "validation_reference_rmse_K": _reference_rmse(
                plain.predict(reference_features_validation),
                references_validation.temperature_C,
            ),
        }
    except (ValueError, RuntimeError, FloatingPointError) as error:
        failures["plain_nn"] = _failure(error)
        record["plain_nn"] = {"available": False, "failure": failures["plain_nn"]}

    pinn: PinnFit | None = None
    try:
        pinn = fit_pinn(
            _pinn_trajectory(
                frame_train,
                train,
                reference_features_train,
                references_train.temperature_C,
                oil_features_train,
                observed_train.top_oil_C,
            ),
            reference_features_validation,
            references_validation.temperature_C,
            seed=seed,
            config=config,
        )
        pinn_validation = pinn.predict_states(reference_features_validation)[
            "hotspot_C"
        ]
        record["pinn"] = {
            "available": True,
            "best_epoch": pinn.best_epoch,
            "epochs_ran": pinn.epochs_ran,
            "parameters": dict(pinn.parameter_values()),
            "stopped_early": pinn.stopped_early,
            "validation_reference_rmse_K": _reference_rmse(
                pinn_validation, references_validation.temperature_C
            ),
        }
    except (ValueError, RuntimeError, FloatingPointError) as error:
        failures["pinn"] = _failure(error)
        record["pinn"] = {"available": False, "failure": failures["pinn"]}

    greybox: HullAwareResidualFit | None = None
    if isinstance(nls, NLSModel):
        try:
            nls_train_trajectory = predict_nls(nls, train.schedule)
            nls_validation_trajectory = predict_nls(nls, validation.schedule)
            nls_train_references = np.asarray(nls_train_trajectory.hotspot_C)[
                references_train.index
            ]
            nls_validation_references = np.asarray(
                nls_validation_trajectory.hotspot_C
            )[references_validation.index]
            greybox = fit_residual_nn(
                reference_features_train,
                references_train.temperature_C - nls_train_references,
                reference_features_validation,
                references_validation.temperature_C - nls_validation_references,
                train_load_pu=train.schedule.load_pu,
                seed=seed,
                config=config,
                standardizer_features=frame_train.X,
            )
            grey_validation = greybox.predict(
                reference_features_validation,
                validation.schedule.load_pu[references_validation.index],
                nls_hotspot_C=nls_validation_references,
            ).hotspot_C
            assert grey_validation is not None
            record["greybox"] = {
                "available": True,
                "best_epoch": greybox.fit.best_epoch,
                "epochs_ran": greybox.fit.epochs_ran,
                "hull_max_pu": greybox.hull_max_pu,
                "hull_min_pu": greybox.hull_min_pu,
                "stopped_early": greybox.fit.stopped_early,
                "validation_reference_rmse_K": _reference_rmse(
                    grey_validation, references_validation.temperature_C
                ),
            }
        except (ValueError, RuntimeError, FloatingPointError) as error:
            failures["greybox"] = _failure(error)
            record["greybox"] = {
                "available": False,
                "failure": failures["greybox"],
            }
    else:
        failures["greybox"] = {
            "type": "NLSRefusal",
            "message": "grey-box requires an available NLS core",
        }
        record["greybox"] = {"available": False, "failure": failures["greybox"]}

    return SeedMethodFits(
        seed=seed,
        nls=nls,
        plain_nn=plain,
        pinn=pinn,
        greybox=greybox,
        failures=failures,
        training_record=record,
    )


def _score_target_plateau(
    prediction_C: np.ndarray,
    frame: FeatureFrame,
) -> TrajectoryMetrics:
    mask = frame.time_s >= 4.0 * 3600.0
    if not np.any(mask):
        raise ValueError("E3 feature frame contains no target plateau")
    return trajectory_metrics(prediction_C[mask], frame.hotspot_truth_C[mask])


def evaluate_e3_seed(
    fits: SeedMethodFits,
    test_truth_by_load: Mapping[float, TruthRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate one fitted seed on all four already-authorised E3 truths."""

    rows: list[dict[str, object]] = []
    invariants: list[dict[str, object]] = []
    for target in E3_TARGET_LOADS_PU:
        truth = test_truth_by_load[target]
        observed = observe_record(truth, seed=fits.seed)
        frame = build_feature_frame(observed)
        load = truth.schedule.load_pu[frame.truth_index]

        predictions: dict[str, np.ndarray] = {}
        nls_prediction: np.ndarray | None = None
        if isinstance(fits.nls, NLSModel):
            nls_prediction = prediction_on_feature_frame(
                predict_nls(fits.nls, truth.schedule), frame
            )
            predictions["nls"] = nls_prediction
        if fits.plain_nn is not None:
            predictions["plain_nn"] = fits.plain_nn.predict(frame.X)
        if fits.pinn is not None:
            predictions["pinn"] = fits.pinn.predict_states(frame.X)["hotspot_C"]
        if fits.greybox is not None and nls_prediction is not None:
            hull = fits.greybox.predict(
                frame.X, load, nls_hotspot_C=nls_prediction
            )
            assert hull.hotspot_C is not None
            predictions["greybox"] = hull.hotspot_C
            outside = (load < fits.greybox.hull_min_pu) | (
                load > fits.greybox.hull_max_pu
            )
            invariants.append(
                {
                    "all_expected_flags": bool(
                        np.array_equal(hull.extrapolation_flag, outside)
                    ),
                    "outside_hotspot_bit_exact_nls": bool(
                        np.array_equal(hull.hotspot_C[outside], nls_prediction[outside])
                    ),
                    "outside_residual_positive_zero": bool(
                        np.all(hull.residual_K[outside].view(np.uint64) == 0)
                    ),
                    "seed": fits.seed,
                    "target_load_pu": target,
                }
            )
        predictions["generic_iec"] = prediction_on_feature_frame(
            predict_generic(truth.schedule), frame
        )

        for method, prediction in predictions.items():
            scores = _score_target_plateau(prediction, frame)
            rows.append(
                {
                    "method": method,
                    "seed": fits.seed,
                    "target_load_pu": target,
                    **scores.as_dict(),
                }
            )
    return rows, invariants


def _metrics_from_row(row: Mapping[str, object]) -> TrajectoryMetrics:
    return TrajectoryMetrics(
        mae_K=float(row["mae_K"]),
        rmse_K=float(row["rmse_K"]),
        mean_signed_error_K=float(row["mean_signed_error_K"]),
        signed_peak_error_K=float(row["signed_peak_error_K"]),
    )


def _resolve_e3_greybox_invariants(
    invariant_rows: Sequence[Mapping[str, object]] | None,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Require one true outside-hull invariant row per E3 seed/load cell."""

    expected = {
        (int(seed), float(target))
        for seed in seeds
        for target in E3_TARGET_LOADS_PU
    }
    by_cell: dict[tuple[int, float], Mapping[str, object]] = {}
    duplicate_cells: list[tuple[int, float]] = []
    unexpected_count = 0
    for row in () if invariant_rows is None else invariant_rows:
        try:
            cell = (int(row["seed"]), float(row["target_load_pu"]))
        except (KeyError, TypeError, ValueError):
            unexpected_count += 1
            continue
        if cell not in expected:
            unexpected_count += 1
            continue
        if cell in by_cell:
            duplicate_cells.append(cell)
            continue
        by_cell[cell] = row

    missing = sorted(expected.difference(by_cell))
    false_cells = sorted(
        cell
        for cell, row in by_cell.items()
        if not (
            row.get("all_expected_flags") is True
            and row.get("outside_hotspot_bit_exact_nls") is True
            and row.get("outside_residual_positive_zero") is True
        )
    )
    passed = bool(
        not missing
        and not duplicate_cells
        and unexpected_count == 0
        and not false_cells
        and len(by_cell) == len(expected)
    )
    serialise = lambda cell: {
        "seed": cell[0],
        "target_load_pu": cell[1],
    }
    return {
        "all_expected_cells_present_and_true": passed,
        "duplicate_cells": [serialise(cell) for cell in sorted(set(duplicate_cells))],
        "expected_cell_count": len(expected),
        "false_cells": [serialise(cell) for cell in false_cells],
        "missing_cells": [serialise(cell) for cell in missing],
        "recorded_unique_cell_count": len(by_cell),
        "unexpected_or_malformed_row_count": unexpected_count,
    }


def aggregate_e3_rows(
    rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]] | None = None,
    *,
    seeds: Sequence[int] = E3_SEEDS,
) -> dict[str, object]:
    """Aggregate all cells, ranks, paired rules, and confirmation triggers."""

    seed_tuple = assert_at_least_ten_seeds(seeds)
    seed_set = set(seed_tuple)
    cell_output: dict[str, object] = {}
    decisions: dict[str, object] = {}
    confirmation_required: list[dict[str, object]] = []
    greybox_invariant = _resolve_e3_greybox_invariants(invariant_rows, seed_tuple)
    greybox_invariant_passed = bool(
        greybox_invariant["all_expected_cells_present_and_true"]
    )

    for target in E3_TARGET_LOADS_PU:
        target_rows = [row for row in rows if float(row["target_load_pu"]) == target]
        cells: dict[str, CellMetrics] = {}
        row_map: dict[str, dict[int, Mapping[str, object]]] = {}
        target_payload: dict[str, object] = {}
        for method in METHOD_NAMES:
            method_rows = [row for row in target_rows if row["method"] == method]
            by_seed = {int(row["seed"]): row for row in method_rows}
            row_map[method] = by_seed
            available = sorted(seed_set.intersection(by_seed))
            payload: dict[str, object] = {
                "availability_count": len(available),
                "availability_rate": len(available) / len(seed_tuple),
            }
            if set(available) == seed_set:
                cell = aggregate_trajectory_metrics(
                    _metrics_from_row(by_seed[seed]) for seed in seed_tuple
                )
                cells[method] = cell
                payload["metrics"] = cell.as_dict()
            target_payload[method] = payload
        target_payload["accuracy_rank"] = list(accuracy_rank(cells))
        target_payload["safety_rank"] = list(safety_rank(cells))
        cell_output[f"{target:.2f}"] = target_payload

        if "nls" in cells:
            for method in ("plain_nn", "pinn", "greybox", "generic_iec"):
                if method not in cells:
                    continue
                method_rows = row_map[method]
                nls_rows = row_map["nls"]
                decision = e3_win_decision(
                    [float(method_rows[seed]["rmse_K"]) for seed in seed_tuple],
                    [float(nls_rows[seed]["rmse_K"]) for seed in seed_tuple],
                    [
                        float(method_rows[seed]["signed_peak_error_K"])
                        for seed in seed_tuple
                    ],
                    [
                        float(nls_rows[seed]["signed_peak_error_K"])
                        for seed in seed_tuple
                    ],
                )
                key = f"{method}_vs_nls_at_{target:.2f}pu"
                decision_payload = decision.as_dict()
                effective_beats_nls = decision.beats_nls
                if method == "greybox":
                    decision_payload["beats_nls_before_outside_hull_invariant"] = (
                        decision.beats_nls
                    )
                    decision_payload["outside_hull_invariant_passed"] = (
                        greybox_invariant_passed
                    )
                    effective_beats_nls = bool(
                        decision.beats_nls and greybox_invariant_passed
                    )
                    decision_payload["beats_nls"] = effective_beats_nls
                decisions[key] = decision_payload
                if (
                    method in {"plain_nn", "pinn", "greybox"}
                    and target in {1.30, 1.60}
                    and effective_beats_nls
                ):
                    confirmation_required.append(
                        {
                            "method": method,
                            "target_load_pu": target,
                            "reserved_seeds": list(range(51_000, 51_010)),
                            "widths": [8, 32],
                        }
                    )
    return {
        "cells": cell_output,
        "confirmation_required": confirmation_required,
        "greybox_outside_hull_invariant": greybox_invariant,
        "paired_decisions": decisions,
    }


def _parameter_error_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for name in PARAMETER_NAMES:
        signed = [
            float(row["errors"][name]["signed_percent"])  # type: ignore[index]
            for row in rows
        ]
        absolute = [
            float(row["errors"][name]["absolute_percent"])  # type: ignore[index]
            for row in rows
        ]
        from .metrics import distribution_summary

        output[name] = {
            "signed_percent": distribution_summary(signed).as_dict(),
            "absolute_percent": distribution_summary(absolute).as_dict(),
        }
    medians = [
        float(
            row["errors"]["median_absolute_percent_across_parameters"]["value"]  # type: ignore[index]
        )
        for row in rows
    ]
    from .metrics import distribution_summary

    output["median_absolute_percent_across_parameters"] = distribution_summary(
        medians
    ).as_dict()
    return output


def aggregate_e2_rows(
    metric_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    availability_rows: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int] = E2_SEEDS,
) -> dict[str, object]:
    """Aggregate E2 availability, recovery, held-out scores, and kill rule."""

    seed_tuple = assert_at_least_ten_seeds(seeds)
    seed_set = set(seed_tuple)
    cells: dict[str, object] = {}
    decisions: dict[str, object] = {}
    pinn_winning_budgets: list[int] = []
    confirmation_required: list[dict[str, object]] = []

    for budget in E2_REFERENCE_BUDGETS:
        budget_payload: dict[str, object] = {}
        metric_map: dict[str, dict[int, Mapping[str, object]]] = {}
        for method in ("nls", "plain_nn", "pinn"):
            availability = [
                row
                for row in availability_rows
                if int(row["budget"]) == budget and row["method"] == method
            ]
            available_seeds = {
                int(row["seed"]) for row in availability if bool(row["available"])
            }
            method_metrics = {
                int(row["seed"]): row
                for row in metric_rows
                if int(row["budget"]) == budget and row["method"] == method
            }
            metric_map[method] = method_metrics
            payload: dict[str, object] = {
                "availability_count": len(available_seeds),
                "availability_rate": len(available_seeds) / len(seed_tuple),
                "refusal_or_failure_count": len(seed_tuple) - len(available_seeds),
            }
            if available_seeds == seed_set and set(method_metrics) == seed_set:
                payload["held_out_metrics"] = aggregate_trajectory_metrics(
                    _metrics_from_row(method_metrics[seed]) for seed in seed_tuple
                ).as_dict()
            if method in {"nls", "pinn"}:
                recovery = [
                    row
                    for row in parameter_rows
                    if int(row["budget"]) == budget and row["method"] == method
                ]
                if len(recovery) == len(seed_tuple):
                    payload["parameter_recovery"] = _parameter_error_summary(recovery)
            else:
                payload["parameter_recovery"] = "N/A"
            budget_payload[method] = payload

        if budget >= 4 and set(metric_map["nls"]) == seed_set and set(
            metric_map["pinn"]
        ) == seed_set:
            nls_map = metric_map["nls"]
            pinn_map = metric_map["pinn"]
            decision = e3_win_decision(
                [float(pinn_map[seed]["rmse_K"]) for seed in seed_tuple],
                [float(nls_map[seed]["rmse_K"]) for seed in seed_tuple],
                [float(pinn_map[seed]["signed_peak_error_K"]) for seed in seed_tuple],
                [float(nls_map[seed]["signed_peak_error_K"]) for seed in seed_tuple],
            )
            # E2 froze the stricter phrase "no worse worst signed peak";
            # unlike E3, it has no 0.10 K tolerance.  Keep the shared E3
            # diagnostics but replace the final E2 safety/win flags exactly.
            strict_no_worse_peak = min(
                float(pinn_map[seed]["signed_peak_error_K"])
                for seed in seed_tuple
            ) >= min(
                float(nls_map[seed]["signed_peak_error_K"])
                for seed in seed_tuple
            )
            decision_payload = decision.as_dict()
            decision_payload["e2_strict_no_worse_worst_signed_peak"] = bool(
                strict_no_worse_peak
            )
            decision_payload["safety_condition"] = bool(strict_no_worse_peak)
            decision_payload["beats_nls"] = bool(
                decision.lower_mean_rmse
                and decision.confidence_interval_excludes_zero_in_favour
                and strict_no_worse_peak
            )
            decisions[f"pinn_vs_nls_N{budget}"] = decision_payload
            if decision_payload["beats_nls"]:
                pinn_winning_budgets.append(budget)
                confirmation_required.append(
                    {
                        "experiment": "e2",
                        "method": "pinn",
                        "reference_budget": budget,
                        "reserved_seeds": list(range(51_000, 51_010)),
                        "widths": [8, 32],
                    }
                )
        cells[str(budget)] = budget_payload

    n3_pinn = {
        int(row["seed"]): row
        for row in metric_rows
        if int(row["budget"]) == 3 and row["method"] == "pinn"
    }
    n3_parameter = [
        row
        for row in parameter_rows
        if int(row["budget"]) == 3 and row["method"] == "pinn"
    ]
    n3_all_finite_interior = len(n3_parameter) == len(seed_tuple) and all(
        bool(row["finite_interior"]) for row in n3_parameter
    )
    n3_all_rmse_below_2 = set(n3_pinn) == seed_set and all(
        float(n3_pinn[seed]["rmse_K"]) < 2.0 for seed in seed_tuple
    )
    if n3_all_finite_interior and n3_all_rmse_below_2:
        confirmation_required.append(
            {
                "experiment": "e2",
                "method": "pinn",
                "reference_budget": 3,
                "reserved_seeds": list(range(51_000, 51_010)),
                "widths": [8, 32],
            }
        )

    return {
        "cells": cells,
        "confirmation_required": confirmation_required,
        "n3_investigate_screen": {
            "all_ten_finite_and_interior": n3_all_finite_interior,
            "all_ten_held_out_rmse_below_2_K": n3_all_rmse_below_2,
            "passes_before_reserved_confirmation": (
                n3_all_finite_interior and n3_all_rmse_below_2
            ),
        },
        "paired_decisions": decisions,
        "physics_informed_identification_classification": (
            "investigate further pending reserved confirmation"
            if pinn_winning_budgets
            or (n3_all_finite_interior and n3_all_rmse_below_2)
            else "reject"
        ),
        "pinn_winning_budgets_N_ge_4": pinn_winning_budgets,
    }


@record_primary_failures
def run_e2_primary(
    repository: str | Path,
    *,
    override: bool = False,
    override_reason: str | None = None,
) -> Mapping[str, object]:
    """Execute the write-once scarce-reference sweep after E3."""

    enforce_cpu_only_environment()
    torch_status = require_torch_cpu_only()
    repository_path = Path(repository).resolve()
    config = e2_configuration()
    require_e1_passed(repository_path)
    require_completed_primary(
        repository_path,
        "e3",
        required_configuration=e3_configuration(),
        required_seeds=E3_SEEDS,
    )
    run = begin_primary_run(
        repository_path,
        experiment="e2",
        configuration=config,
        seeds=E2_SEEDS,
        command=_recorded_primary_command(
            "e2", override=override, override_reason=override_reason
        ),
        override=override,
        override_reason=override_reason,
    )
    train_truth = simulate_truth(make_train_schedule(), physics_mode="matched")
    validation_truth = simulate_truth(
        make_validation_schedule(), physics_mode="matched"
    )
    # Hidden in-range truth is generated only after the sentinel above.
    test_truth = simulate_truth(
        make_in_range_test_schedule(), physics_mode="matched"
    )

    metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for budget in E2_REFERENCE_BUDGETS:
        for seed in E2_SEEDS:
            observed_train = observe_record(train_truth, seed=seed)
            observed_validation = observe_record(validation_truth, seed=seed)
            observed_test = observe_record(test_truth, seed=seed)
            references_train = observe_hotspot_references(
                train_truth, budget=budget, seed=seed
            )
            references_validation = observe_hotspot_references(
                validation_truth, budget=budget, seed=seed
            )
            train_frame = build_feature_frame(observed_train)
            validation_frame = build_feature_frame(observed_validation)
            test_frame = build_feature_frame(observed_test)
            train_reference_features, _ = feature_matrix_at_times(
                observed_train, references_train.time_s
            )
            validation_reference_features, _ = feature_matrix_at_times(
                observed_validation, references_validation.time_s
            )
            train_oil_features, _ = feature_matrix_at_times(
                observed_train, observed_train.top_oil_time_s
            )
            cell_training: dict[str, object] = {"budget": budget, "seed": seed}

            nls = fit_nls(observed_train, references_train)
            if isinstance(nls, NLSModel):
                availability_rows.append(
                    {"available": True, "budget": budget, "method": "nls", "seed": seed}
                )
                nls_prediction = prediction_on_feature_frame(
                    predict_nls(nls, test_truth.schedule), test_frame
                )
                scores = trajectory_metrics(
                    nls_prediction, test_frame.hotspot_truth_C
                )
                metric_rows.append(
                    {
                        "budget": budget,
                        "method": "nls",
                        "seed": seed,
                        **scores.as_dict(),
                    }
                )
                parameters = dict(nls.parameter_mapping())
                parameter_rows.append(
                    {
                        "budget": budget,
                        "errors": parameter_percent_errors(parameters),
                        "finite_interior": True,
                        "method": "nls",
                        "parameters": parameters,
                        "seed": seed,
                    }
                )
                cell_training["nls"] = nls.as_dict()
            else:
                availability_rows.append(
                    {
                        "available": False,
                        "budget": budget,
                        "method": "nls",
                        "refusal": nls.as_dict(),
                        "seed": seed,
                    }
                )
                cell_training["nls"] = {
                    "available": False,
                    "refusal": nls.as_dict(),
                }

            try:
                plain = fit_plain_nn(
                    train_reference_features,
                    references_train.temperature_C,
                    validation_reference_features,
                    references_validation.temperature_C,
                    seed=seed,
                    standardizer_features=train_frame.X,
                )
                plain_prediction = plain.predict(test_frame.X)
                scores = trajectory_metrics(
                    plain_prediction, test_frame.hotspot_truth_C
                )
                metric_rows.append(
                    {
                        "budget": budget,
                        "method": "plain_nn",
                        "seed": seed,
                        **scores.as_dict(),
                    }
                )
                availability_rows.append(
                    {
                        "available": True,
                        "budget": budget,
                        "method": "plain_nn",
                        "seed": seed,
                    }
                )
                cell_training["plain_nn"] = {
                    "available": True,
                    "best_epoch": plain.best_epoch,
                    "epochs_ran": plain.epochs_ran,
                    "validation_reference_rmse_K": _reference_rmse(
                        plain.predict(validation_reference_features),
                        references_validation.temperature_C,
                    ),
                }
            except (ValueError, RuntimeError, FloatingPointError) as error:
                plain = None
                failure = _failure(error)
                availability_rows.append(
                    {
                        "available": False,
                        "budget": budget,
                        "failure": failure,
                        "method": "plain_nn",
                        "seed": seed,
                    }
                )
                cell_training["plain_nn"] = {
                    "available": False,
                    "failure": failure,
                }

            try:
                pinn = fit_pinn(
                    _pinn_trajectory(
                        train_frame,
                        train_truth,
                        train_reference_features,
                        references_train.temperature_C,
                        train_oil_features,
                        observed_train.top_oil_C,
                    ),
                    validation_reference_features,
                    references_validation.temperature_C,
                    seed=seed,
                )
                pinn_prediction = pinn.predict_states(test_frame.X)["hotspot_C"]
                scores = trajectory_metrics(
                    pinn_prediction, test_frame.hotspot_truth_C
                )
                metric_rows.append(
                    {
                        "budget": budget,
                        "method": "pinn",
                        "seed": seed,
                        **scores.as_dict(),
                    }
                )
                parameters = dict(pinn.parameter_values())
                values = np.asarray(
                    [parameters[name] for name in PARAMETER_NAMES], dtype=np.float64
                )
                finite_interior = bool(
                    np.all(np.isfinite(values))
                    and np.all(values > np.asarray(PARAMETER_LOWER))
                    and np.all(values < np.asarray(PARAMETER_UPPER))
                )
                parameter_rows.append(
                    {
                        "budget": budget,
                        "errors": parameter_percent_errors(parameters),
                        "finite_interior": finite_interior,
                        "method": "pinn",
                        "parameters": parameters,
                        "seed": seed,
                    }
                )
                availability_rows.append(
                    {"available": True, "budget": budget, "method": "pinn", "seed": seed}
                )
                cell_training["pinn"] = {
                    "available": True,
                    "best_epoch": pinn.best_epoch,
                    "epochs_ran": pinn.epochs_ran,
                    "finite_interior": finite_interior,
                    "parameters": parameters,
                    "validation_reference_rmse_K": _reference_rmse(
                        pinn.predict_states(validation_reference_features)[
                            "hotspot_C"
                        ],
                        references_validation.temperature_C,
                    ),
                }
            except (ValueError, RuntimeError, FloatingPointError) as error:
                pinn = None
                failure = _failure(error)
                availability_rows.append(
                    {
                        "available": False,
                        "budget": budget,
                        "failure": failure,
                        "method": "pinn",
                        "seed": seed,
                    }
                )
                cell_training["pinn"] = {
                    "available": False,
                    "failure": failure,
                }
            training_rows.append(cell_training)
            print(
                f"E2 completed N={budget}, seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            del plain, pinn, nls
            gc.collect()

    aggregate = {
        "availability_rows": availability_rows,
        "configuration": config,
        "parameter_rows": parameter_rows,
        "resolved": aggregate_e2_rows(
            metric_rows, parameter_rows, availability_rows
        ),
        "rows": metric_rows,
        "schema_version": 1,
        "torch": {
            "cpu_only": torch_status.cpu_only,
            "cuda_available": torch_status.cuda_available,
            "version": torch_status.version,
            "visible_cuda_device_count": torch_status.visible_cuda_device_count,
        },
        "training": training_rows,
    }
    final = finish_primary_run(run, aggregate)
    return {"aggregate": aggregate, "manifest": dict(final), "run_id": run.run_id}


def aggregate_e4_rows(
    metric_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int] = E4_SEEDS,
) -> dict[str, object]:
    """Apply the frozen in-range improvement and outside-hull invariant rule."""

    seed_tuple = assert_at_least_ten_seeds(seeds)
    maps = {
        method: {
            int(row["seed"]): row
            for row in metric_rows
            if row["method"] == method
        }
        for method in ("nls", "greybox")
    }
    cells: dict[str, object] = {}
    for method, by_seed in maps.items():
        payload: dict[str, object] = {
            "availability_count": len(by_seed),
            "availability_rate": len(by_seed) / len(seed_tuple),
        }
        if set(by_seed) == set(seed_tuple):
            payload["metrics"] = aggregate_trajectory_metrics(
                _metrics_from_row(by_seed[seed]) for seed in seed_tuple
            ).as_dict()
        cells[method] = payload
    exact_invariant = len(invariant_rows) == len(seed_tuple) and all(
        bool(row["all_expected_flags"])
        and bool(row["outside_hotspot_bit_exact_nls"])
        and bool(row["outside_residual_positive_zero"])
        for row in invariant_rows
    )
    if all(set(maps[method]) == set(seed_tuple) for method in maps):
        decision = dict(
            e4_adoption_decision(
                [float(maps["greybox"][seed]["rmse_K"]) for seed in seed_tuple],
                [float(maps["nls"][seed]["rmse_K"]) for seed in seed_tuple],
                exact_outside_hull_invariant=exact_invariant,
            )
        )
    else:
        decision = {
            "adopt_in_range_only": False,
            "exact_outside_hull_invariant": exact_invariant,
            "reason": "fewer than ten paired available seeds",
        }
    return {
        "cells": cells,
        "decision": decision,
        "classification": (
            "adopt for in-range use only"
            if bool(decision["adopt_in_range_only"])
            else "reject"
        ),
    }


@record_primary_failures
def run_e4_primary(
    repository: str | Path,
    *,
    override: bool = False,
    override_reason: str | None = None,
) -> Mapping[str, object]:
    """Execute the write-once hard-gated residual experiment."""

    enforce_cpu_only_environment()
    torch_status = require_torch_cpu_only()
    repository_path = Path(repository).resolve()
    config = e4_configuration()
    require_e1_passed(repository_path)
    require_completed_primary(
        repository_path,
        "e3",
        required_configuration=e3_configuration(),
        required_seeds=E3_SEEDS,
    )
    require_completed_primary(
        repository_path,
        "e2",
        required_configuration=e2_configuration(),
        required_seeds=E2_SEEDS,
    )
    run = begin_primary_run(
        repository_path,
        experiment="e4",
        configuration=config,
        seeds=E4_SEEDS,
        command=_recorded_primary_command(
            "e4", override=override, override_reason=override_reason
        ),
        override=override,
        override_reason=override_reason,
    )
    train_truth = simulate_truth(
        make_train_schedule(), physics_mode="structural_mismatch"
    )
    validation_truth = simulate_truth(
        make_validation_schedule(), physics_mode="structural_mismatch"
    )
    test_truth = simulate_truth(
        make_in_range_test_schedule(), physics_mode="structural_mismatch"
    )
    metric_rows: list[dict[str, object]] = []
    invariant_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for seed in E4_SEEDS:
        observed_train = observe_record(train_truth, seed=seed)
        observed_validation = observe_record(validation_truth, seed=seed)
        observed_test = observe_record(test_truth, seed=seed)
        references_train = observe_hotspot_references(
            train_truth, budget=E4_REFERENCE_BUDGET, seed=seed
        )
        references_validation = observe_hotspot_references(
            validation_truth, budget=E4_REFERENCE_BUDGET, seed=seed
        )
        train_frame = build_feature_frame(observed_train)
        validation_frame = build_feature_frame(observed_validation)
        test_frame = build_feature_frame(observed_test)
        train_reference_features, _ = feature_matrix_at_times(
            observed_train, references_train.time_s
        )
        validation_reference_features, _ = feature_matrix_at_times(
            observed_validation, references_validation.time_s
        )
        nls = fit_nls(observed_train, references_train)
        record: dict[str, object] = {"seed": seed}
        if not isinstance(nls, NLSModel):
            record["nls"] = {"available": False, "refusal": nls.as_dict()}
            record["greybox"] = {
                "available": False,
                "failure": "NLS core unavailable",
            }
            training_rows.append(record)
            print(f"E4 seed={seed}: NLS refusal", file=sys.stderr, flush=True)
            continue
        record["nls"] = nls.as_dict()
        nls_train_trajectory = predict_nls(nls, train_truth.schedule)
        nls_validation_trajectory = predict_nls(nls, validation_truth.schedule)
        nls_train_references = np.asarray(nls_train_trajectory.hotspot_C)[
            references_train.index
        ]
        nls_validation_references = np.asarray(
            nls_validation_trajectory.hotspot_C
        )[references_validation.index]
        nls_validation = prediction_on_feature_frame(
            nls_validation_trajectory, validation_frame
        )
        nls_test = prediction_on_feature_frame(
            predict_nls(nls, test_truth.schedule), test_frame
        )
        nls_scores = trajectory_metrics(nls_test, test_frame.hotspot_truth_C)
        metric_rows.append(
            {"method": "nls", "seed": seed, **nls_scores.as_dict()}
        )
        try:
            greybox = fit_residual_nn(
                train_reference_features,
                references_train.temperature_C - nls_train_references,
                validation_reference_features,
                references_validation.temperature_C - nls_validation_references,
                train_load_pu=train_truth.schedule.load_pu,
                seed=seed,
                standardizer_features=train_frame.X,
            )
            test_load = test_truth.schedule.load_pu[test_frame.truth_index]
            grey_test_result = greybox.predict(
                test_frame.X, test_load, nls_hotspot_C=nls_test
            )
            assert grey_test_result.hotspot_C is not None
            if bool(np.any(grey_test_result.extrapolation_flag)):
                raise RuntimeError("in-range test unexpectedly leaves the train hull")
            grey_scores = trajectory_metrics(
                grey_test_result.hotspot_C, test_frame.hotspot_truth_C
            )
            metric_rows.append(
                {"method": "greybox", "seed": seed, **grey_scores.as_dict()}
            )

            # Test hard gating independently of outcome truth by presenting a
            # scalar load array strictly beyond the fitted hull.
            probe_load = np.full(
                validation_frame.time_s.shape, greybox.hull_max_pu + 0.10
            )
            probe = greybox.predict(
                validation_frame.X,
                probe_load,
                nls_hotspot_C=nls_validation,
            )
            assert probe.hotspot_C is not None
            invariant_rows.append(
                {
                    "all_expected_flags": bool(
                        np.all(probe.extrapolation_flag)
                    ),
                    "outside_hotspot_bit_exact_nls": bool(
                        np.array_equal(probe.hotspot_C, nls_validation)
                    ),
                    "outside_residual_positive_zero": bool(
                        np.all(probe.residual_K.view(np.uint64) == 0)
                    ),
                    "seed": seed,
                }
            )
            record["greybox"] = {
                "available": True,
                "best_epoch": greybox.fit.best_epoch,
                "epochs_ran": greybox.fit.epochs_ran,
                "hull_max_pu": greybox.hull_max_pu,
                "hull_min_pu": greybox.hull_min_pu,
                "validation_reference_rmse_K": _reference_rmse(
                    greybox.predict(
                        validation_reference_features,
                        validation_truth.schedule.load_pu[
                            references_validation.index
                        ],
                        nls_hotspot_C=nls_validation_references,
                    ).hotspot_C,
                    references_validation.temperature_C,
                ),
            }
        except (ValueError, RuntimeError, FloatingPointError) as error:
            record["greybox"] = {
                "available": False,
                "failure": _failure(error),
            }
        training_rows.append(record)
        print(f"E4 completed seed={seed}", file=sys.stderr, flush=True)
        del nls
        if "greybox" in locals():
            del greybox
        gc.collect()

    aggregate = {
        "configuration": config,
        "greybox_outside_hull_invariants": invariant_rows,
        "resolved": aggregate_e4_rows(metric_rows, invariant_rows),
        "rows": metric_rows,
        "schema_version": 1,
        "torch": {
            "cpu_only": torch_status.cpu_only,
            "cuda_available": torch_status.cuda_available,
            "version": torch_status.version,
            "visible_cuda_device_count": torch_status.visible_cuda_device_count,
        },
        "training": training_rows,
    }
    final = finish_primary_run(run, aggregate)
    return {"aggregate": aggregate, "manifest": dict(final), "run_id": run.run_id}


@record_primary_failures
def run_e3_primary(
    repository: str | Path,
    *,
    override: bool = False,
    override_reason: str | None = None,
) -> Mapping[str, object]:
    """Execute the write-once E3 primary run in the frozen seed order."""

    enforce_cpu_only_environment()
    torch_status = require_torch_cpu_only()
    repository_path = Path(repository).resolve()
    config = e3_configuration()
    require_e1_passed(repository_path)
    run = begin_primary_run(
        repository_path,
        experiment="e3",
        configuration=config,
        seeds=E3_SEEDS,
        command=_recorded_primary_command(
            "e3", override=override, override_reason=override_reason
        ),
        override=override,
        override_reason=override_reason,
    )

    # Test truth is generated only after begin_primary_run has durably claimed
    # its hash-stamped access sentinel.
    test_truth = {
        target: simulate_truth(
            make_e3_schedule(target), physics_mode="structural_mismatch"
        )
        for target in E3_TARGET_LOADS_PU
    }
    train_truth = simulate_truth(
        make_train_schedule(), physics_mode="structural_mismatch"
    )
    validation_truth = simulate_truth(
        make_validation_schedule(), physics_mode="structural_mismatch"
    )
    all_rows: list[dict[str, object]] = []
    all_invariants: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []
    for seed in E3_SEEDS:
        fits = fit_seed_methods(
            seed,
            physics_mode="structural_mismatch",
            reference_budget=E3_REFERENCE_BUDGET,
            train_truth=train_truth,
            validation_truth=validation_truth,
        )
        training_records.append(fits.training_record)
        rows, invariants = evaluate_e3_seed(fits, test_truth)
        all_rows.extend(rows)
        all_invariants.extend(invariants)
        del fits
        gc.collect()

    aggregate = {
        "configuration": config,
        "resolved": aggregate_e3_rows(all_rows, invariant_rows=all_invariants),
        "rows": all_rows,
        "schema_version": 1,
        "torch": {
            "cpu_only": torch_status.cpu_only,
            "cuda_available": torch_status.cuda_available,
            "version": torch_status.version,
            "visible_cuda_device_count": torch_status.visible_cuda_device_count,
        },
        "training": training_records,
        "greybox_outside_hull_invariants": all_invariants,
    }
    final = finish_primary_run(run, aggregate)
    return {"aggregate": aggregate, "manifest": dict(final), "run_id": run.run_id}


__all__ = [
    "E2_REFERENCE_BUDGETS",
    "E2_SEEDS",
    "E3_REFERENCE_BUDGET",
    "E3_SEEDS",
    "E3_TARGET_LOADS_PU",
    "E4_REFERENCE_BUDGET",
    "E4_SEEDS",
    "METHOD_NAMES",
    "SeedMethodFits",
    "aggregate_e2_rows",
    "aggregate_e3_rows",
    "aggregate_e4_rows",
    "e2_configuration",
    "e3_configuration",
    "e4_configuration",
    "evaluate_e3_seed",
    "fit_seed_methods",
    "run_e1_primary",
    "run_e2_primary",
    "run_e3_primary",
    "run_e4_primary",
]
