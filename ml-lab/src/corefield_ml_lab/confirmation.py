"""Reserved positive-result confirmations for E2 and E3.

This module is deliberately separate from the primary experiment runners.  A
confirmation can be claimed only when a completed, hash-verified primary
aggregate contains the exact preregistered trigger.  The hidden confirmation
truth is generated only after :func:`begin_primary_run` has durably created a
trigger-specific write-once sentinel.

Temperatures are degrees Celsius (absolute) or kelvin (differences), time is
seconds, and load is per unit.  All neural fitting is CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np

from .classical import (
    NLSModel,
    fit_nls,
    parameter_percent_errors,
    predict_nls,
    prediction_on_feature_frame,
)
from .metrics import (
    TrajectoryMetrics,
    aggregate_trajectory_metrics,
    assert_at_least_ten_seeds,
    e3_win_decision,
    paired_mean_bootstrap_interval,
    trajectory_metrics,
)
from .ml_methods import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    HullAwareResidualFit,
    PinnFit,
    PinnTrajectory,
    SupervisedFit,
    TrainingConfig,
    fit_pinn,
    fit_plain_nn,
    fit_residual_nn,
)
from .runstore import (
    begin_primary_run,
    canonical_sha256,
    finish_primary_run,
    record_primary_failures,
)
from .runtime import (
    enforce_cpu_only_environment,
    require_torch_cpu_only,
    sha256_file,
)
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


RESERVED_CONFIRMATION_SEEDS: tuple[int, ...] = tuple(range(51_000, 51_010))
RESERVED_HIDDEN_WIDTHS: tuple[int, ...] = (8, 32)
E3_CONFIRMATION_LOADS_PU: tuple[float, ...] = (1.30, 1.60)
E3_CONFIRMABLE_METHODS: tuple[str, ...] = ("plain_nn", "pinn", "greybox")
E2_CONFIRMATION_BUDGETS: tuple[int, ...] = (3, 4, 6, 10, 20, 50)
E3_REFERENCE_BUDGET = 20

_SAFE_RUN_ID = re.compile(r"[a-z0-9_-]+\Z")


@dataclass(frozen=True, slots=True)
class ConfirmationTrigger:
    """One exact positive-result trigger emitted by E2 or E3."""

    experiment: str
    method: str
    target_load_pu: float | None = None
    reference_budget: int | None = None

    def __post_init__(self) -> None:
        if self.experiment == "e3":
            if self.method not in E3_CONFIRMABLE_METHODS:
                raise ValueError(
                    f"E3 method must be one of {E3_CONFIRMABLE_METHODS}"
                )
            if self.target_load_pu not in E3_CONFIRMATION_LOADS_PU:
                raise ValueError(
                    f"E3 confirmation load must be one of {E3_CONFIRMATION_LOADS_PU}"
                )
            if self.reference_budget is not None:
                raise ValueError("E3 trigger cannot specify reference_budget")
            return
        if self.experiment == "e2":
            if self.method != "pinn":
                raise ValueError("E2 confirmation is defined only for the PINN")
            if self.reference_budget not in E2_CONFIRMATION_BUDGETS:
                raise ValueError(
                    f"E2 reference budget must be one of {E2_CONFIRMATION_BUDGETS}"
                )
            if self.target_load_pu is not None:
                raise ValueError("E2 trigger cannot specify target_load_pu")
            return
        raise ValueError("confirmation experiment must be 'e2' or 'e3'")

    @classmethod
    def e3(cls, method: str, target_load_pu: float) -> "ConfirmationTrigger":
        return cls("e3", method, target_load_pu=float(target_load_pu))

    @classmethod
    def e2(cls, reference_budget: int) -> "ConfirmationTrigger":
        return cls("e2", "pinn", reference_budget=int(reference_budget))

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "experiment": self.experiment,
            "method": self.method,
        }
        if self.target_load_pu is not None:
            payload["target_load_pu"] = self.target_load_pu
        if self.reference_budget is not None:
            payload["reference_budget"] = self.reference_budget
        return payload

    @property
    def sentinel_experiment(self) -> str:
        """Stable, trigger-specific experiment name for the write-once claim."""

        if self.experiment == "e3":
            assert self.target_load_pu is not None
            load_code = int(round(self.target_load_pu * 100.0))
            return f"e3_confirmation_{self.method}_{load_code}"
        assert self.reference_budget is not None
        return f"e2_confirmation_pinn_n{self.reference_budget}"


@dataclass(frozen=True, slots=True)
class VerifiedPrimaryTrigger:
    """Hash-checked provenance for a trigger in a completed primary run."""

    experiment: str
    run_id: str
    aggregate_sha256: str
    config_sha256: str
    protocol_sha256: str
    final_manifest_path: str

    def as_dict(self) -> dict[str, str]:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "aggregate_sha256": self.aggregate_sha256,
            "config_sha256": self.config_sha256,
            "protocol_sha256": self.protocol_sha256,
            "final_manifest_path": self.final_manifest_path,
        }


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read verified evidence file {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence file must contain a JSON object: {path}")
    return value


def _normalise_trigger_entry(entry: Mapping[str, object]) -> dict[str, object] | None:
    """Return the preregistered trigger fields, or ``None`` if malformed."""

    method = entry.get("method")
    raw_seeds = entry.get("reserved_seeds")
    raw_widths = entry.get("widths")
    if not isinstance(method, str):
        return None
    if not isinstance(raw_seeds, list) or not isinstance(raw_widths, list):
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_seeds):
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_widths):
        return None
    output: dict[str, object] = {
        "method": method,
        "reserved_seeds": list(raw_seeds),
        "widths": list(raw_widths),
    }
    if "experiment" in entry:
        if not isinstance(entry["experiment"], str):
            return None
        output["experiment"] = entry["experiment"]
    if "target_load_pu" in entry:
        target = entry["target_load_pu"]
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            return None
        target_value = float(target)
        if not math.isfinite(target_value):
            return None
        output["target_load_pu"] = target_value
    if "reference_budget" in entry:
        budget = entry["reference_budget"]
        if isinstance(budget, bool) or not isinstance(budget, int):
            return None
        output["reference_budget"] = budget
    return output


def _entry_matches_trigger(
    entry: Mapping[str, object], trigger: ConfirmationTrigger
) -> bool:
    normalised = _normalise_trigger_entry(entry)
    if normalised is None:
        return False
    if normalised["reserved_seeds"] != list(RESERVED_CONFIRMATION_SEEDS):
        return False
    if normalised["widths"] != list(RESERVED_HIDDEN_WIDTHS):
        return False
    if normalised["method"] != trigger.method:
        return False
    if trigger.experiment == "e3":
        return (
            normalised.get("experiment", "e3") == "e3"
            and normalised.get("target_load_pu") == trigger.target_load_pu
            and "reference_budget" not in normalised
        )
    return (
        normalised.get("experiment") == "e2"
        and normalised.get("reference_budget") == trigger.reference_budget
        and "target_load_pu" not in normalised
    )


def verify_completed_primary_trigger(
    repository: str | Path,
    *,
    trigger: ConfirmationTrigger,
    prior_run_id: str,
) -> VerifiedPrimaryTrigger:
    """Verify a completed prior aggregate and its exact confirmation trigger.

    The aggregate content hash, final/start manifest identities, current frozen
    preregistration hash, resource gate, and exact seed/width trigger are all
    checked before any confirmation sentinel may be claimed.
    """

    if not prior_run_id or _SAFE_RUN_ID.fullmatch(prior_run_id) is None:
        raise ValueError("prior_run_id contains unsafe characters")
    root = Path(repository).resolve()
    run_directory = (root / "runs" / trigger.experiment / prior_run_id).resolve()
    expected_parent = (root / "runs" / trigger.experiment).resolve()
    if run_directory.parent != expected_parent:
        raise ValueError("prior_run_id does not resolve inside the expected run directory")

    start_path = run_directory / "manifest.start.json"
    final_path = run_directory / "manifest.final.json"
    aggregate_path = run_directory / "aggregate.json"
    start = _load_json_object(start_path)
    final = _load_json_object(final_path)
    aggregate = _load_json_object(aggregate_path)

    for name, manifest in (("start", start), ("final", final)):
        if manifest.get("experiment") != trigger.experiment:
            raise RuntimeError(f"{name} manifest experiment does not match trigger")
        if manifest.get("run_id") != prior_run_id:
            raise RuntimeError(f"{name} manifest run_id does not match directory")
    if final.get("status") != "completed":
        raise RuntimeError("prior primary run is not completed")
    memory_gate = final.get("memory_gate")
    if not isinstance(memory_gate, dict) or memory_gate.get("passed") is not True:
        raise RuntimeError("prior primary run did not pass its memory gate")

    actual_aggregate_hash = sha256_file(aggregate_path)
    if final.get("aggregate_sha256") != actual_aggregate_hash:
        raise RuntimeError("prior aggregate hash does not match its final manifest")
    config_hash = final.get("config_sha256")
    if not isinstance(config_hash, str) or start.get("config_sha256") != config_hash:
        raise RuntimeError("prior start/final configuration hashes disagree")
    config_payload = start.get("config_payload")
    if not isinstance(config_payload, dict):
        raise RuntimeError("prior start manifest has no configuration payload")
    if canonical_sha256(config_payload) != config_hash:
        raise RuntimeError("prior start configuration payload hash is invalid")
    base_run_id = f"{trigger.experiment}-{config_hash[:16]}"
    if prior_run_id != base_run_id and re.fullmatch(
        re.escape(base_run_id) + r"-attempt[0-9]{2,}", prior_run_id
    ) is None:
        raise RuntimeError("prior run id is not stamped with its configuration hash")
    if config_payload.get("experiment") != trigger.experiment:
        raise RuntimeError("prior configuration payload experiment is inconsistent")
    from .experiments import (
        E2_SEEDS,
        E3_SEEDS,
        e2_configuration,
        e3_configuration,
    )

    expected_configuration = (
        e3_configuration() if trigger.experiment == "e3" else e2_configuration()
    )
    expected_seeds = E3_SEEDS if trigger.experiment == "e3" else E2_SEEDS
    if config_payload.get("configuration") != expected_configuration:
        raise RuntimeError(
            "prior primary run does not use the current corrected frozen configuration"
        )
    if tuple(config_payload.get("seeds", ())) != tuple(expected_seeds):
        raise RuntimeError("prior primary run does not use the frozen primary seeds")
    aggregate_configuration = aggregate.get("configuration")
    if aggregate_configuration != config_payload.get("configuration"):
        raise RuntimeError("prior aggregate and start configurations disagree")
    protocol_hash = sha256_file(root / "PREREGISTRATION.md")
    if final.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("prior final manifest does not use the current frozen protocol")
    if start.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("prior start manifest does not use the current frozen protocol")
    vendor_hash = sha256_file(root / "vendor" / "manifest.json")
    if (
        start.get("vendor_manifest_sha256") != vendor_hash
        or config_payload.get("vendor_manifest_sha256") != vendor_hash
    ):
        raise RuntimeError("prior primary run does not use the current vendor manifest")

    resolved = aggregate.get("resolved")
    if not isinstance(resolved, dict):
        raise RuntimeError("prior aggregate has no resolved result object")
    required = resolved.get("confirmation_required")
    if not isinstance(required, list):
        raise RuntimeError("prior aggregate has no confirmation_required list")
    exact_matches = [
        item
        for item in required
        if isinstance(item, dict) and _entry_matches_trigger(item, trigger)
    ]
    if len(exact_matches) != 1:
        raise RuntimeError(
            "exactly one matching preregistered trigger is required in the prior aggregate"
        )

    return VerifiedPrimaryTrigger(
        experiment=trigger.experiment,
        run_id=prior_run_id,
        aggregate_sha256=actual_aggregate_hash,
        config_sha256=config_hash,
        protocol_sha256=protocol_hash,
        final_manifest_path=str(final_path.relative_to(root)).replace("\\", "/"),
    )


def confirmation_configuration(
    trigger: ConfirmationTrigger,
    source: VerifiedPrimaryTrigger,
) -> dict[str, object]:
    """Return the fully resolved configuration hashed into the access claim."""

    training = TrainingConfig()
    configuration: dict[str, object] = {
        "confirmation_for": trigger.as_dict(),
        "exact_reference_features": True,
        "hidden_test_truth_generated_after_access_claim": True,
        "hidden_widths": list(RESERVED_HIDDEN_WIDTHS),
        "measurement_noise": {
            "hotspot_reference_sigma_K": 0.5,
            "independent_seed_sensor_schedule_substreams": True,
            "top_oil_actual_sample_interval_s": 300.0,
            "top_oil_sigma_K": 0.5,
        },
        "neural_training": {
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
            "max_epochs": training.max_epochs,
            "patience": training.patience,
            "minimum_normalized_improvement": (
                training.minimum_normalized_improvement
            ),
            "intraop_threads": training.intraop_threads,
        },
        "reserved_seeds": list(RESERVED_CONFIRMATION_SEEDS),
        "source_primary_evidence": source.as_dict(),
    }
    if trigger.experiment == "e3":
        configuration.update(
            {
                "comparison": "triggered neural method versus NLS",
                "physics_mode": "structural_mismatch",
                "reference_budget": E3_REFERENCE_BUDGET,
                "score_window": "time_s >= 14400 s target plateau",
                "success_rule": {
                    "both_widths_preserve_safety_tolerance_K": 0.10,
                    "at_least_one_width_preserves_lower_paired_mean_rmse": True,
                    "paired_bootstrap_resamples": 10_000,
                    "paired_bootstrap_two_sided_confidence": 0.95,
                },
            }
        )
    else:
        assert trigger.reference_budget is not None
        if trigger.reference_budget == 3:
            rule: dict[str, object] = {
                "both_widths_all_ten_finite_and_interior": True,
                "both_widths_every_held_out_rmse_below_K": 2.0,
                "nls_refusal_is_not_an_accuracy_win": True,
            }
        else:
            rule = {
                "both_widths_strict_no_worse_worst_signed_peak": True,
                "at_least_one_width_preserves_lower_paired_mean_rmse": True,
                "paired_bootstrap_resamples": 10_000,
                "paired_bootstrap_two_sided_confidence": 0.95,
            }
        configuration.update(
            {
                "comparison": "PINN versus NLS" if trigger.reference_budget >= 4 else "PINN N=3 screen",
                "evaluation_split": "in_range_test_24h, noise-free hidden truth",
                "physics_mode": "matched",
                "reference_budget": trigger.reference_budget,
                "success_rule": rule,
            }
        )
    return configuration


def recorded_confirmation_command(
    trigger: ConfirmationTrigger,
    *,
    prior_run_id: str,
    override: bool = False,
    override_reason: str | None = None,
) -> list[str]:
    """Return the complete reproducible CLI invocation for one trigger."""

    command = [
        sys.executable,
        "-m",
        "corefield_ml_lab",
        "confirmation",
        trigger.experiment,
        "--method",
        trigger.method,
        "--prior-run-id",
        prior_run_id,
    ]
    if trigger.target_load_pu is not None:
        command.extend(["--target-load", f"{trigger.target_load_pu:.2f}"])
    if trigger.reference_budget is not None:
        command.extend(["--reference-budget", str(trigger.reference_budget)])
    if override:
        reason = "" if override_reason is None else override_reason.strip()
        if not reason:
            raise ValueError("override_reason is required when override=True")
        command.extend(["--override", "--override-reason", reason])
    elif override_reason is not None:
        raise ValueError("override_reason requires override=True")
    return command


def _metric_from_row(row: Mapping[str, object]) -> TrajectoryMetrics:
    return TrajectoryMetrics(
        mae_K=float(row["mae_K"]),
        rmse_K=float(row["rmse_K"]),
        mean_signed_error_K=float(row["mean_signed_error_K"]),
        signed_peak_error_K=float(row["signed_peak_error_K"]),
    )


def _rows_by_seed(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    width: int,
    seeds: Sequence[int],
) -> dict[int, Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if row.get("method") == method and int(row.get("hidden_width", -1)) == width
    ]
    output: dict[int, Mapping[str, object]] = {}
    for row in selected:
        seed = int(row["seed"])
        if seed in output:
            raise ValueError(f"duplicate {method}, width={width}, seed={seed} row")
        output[seed] = row
    allowed = set(int(seed) for seed in seeds)
    if not set(output).issubset(allowed):
        raise ValueError("confirmation rows contain an unregistered seed")
    return output


def _cell_payload(
    rows: Mapping[int, Mapping[str, object]], seeds: Sequence[int]
) -> dict[str, object]:
    seed_tuple = tuple(int(seed) for seed in seeds)
    payload: dict[str, object] = {
        "availability_count": len(rows),
        "availability_rate": len(rows) / len(seed_tuple),
    }
    if set(rows) == set(seed_tuple):
        payload["metrics"] = aggregate_trajectory_metrics(
            _metric_from_row(rows[seed]) for seed in seed_tuple
        ).as_dict()
    return payload


def aggregate_confirmation_rows(
    trigger: ConfirmationTrigger,
    metric_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int] = RESERVED_CONFIRMATION_SEEDS,
    widths: Sequence[int] = RESERVED_HIDDEN_WIDTHS,
) -> dict[str, object]:
    """Apply the frozen confirmation rule to complete per-seed rows."""

    seed_tuple = assert_at_least_ten_seeds(seeds)
    width_tuple = tuple(int(width) for width in widths)
    if seed_tuple != RESERVED_CONFIRMATION_SEEDS:
        raise ValueError("confirmation must use exactly seeds 51000..51009")
    if width_tuple != RESERVED_HIDDEN_WIDTHS:
        raise ValueError("confirmation must use exactly hidden widths 8 and 32")

    cells: dict[str, object] = {}
    width_results: dict[str, object] = {}
    safety_by_width: list[bool] = []
    accuracy_by_width: list[bool] = []

    for width in width_tuple:
        candidate = _rows_by_seed(
            metric_rows,
            method=trigger.method,
            width=width,
            seeds=seed_tuple,
        )
        nls = _rows_by_seed(
            metric_rows, method="nls", width=width, seeds=seed_tuple
        )
        cells[str(width)] = {
            trigger.method: _cell_payload(candidate, seed_tuple),
            "nls": _cell_payload(nls, seed_tuple),
        }

        if trigger.experiment == "e2" and trigger.reference_budget == 3:
            parameter_map: dict[int, Mapping[str, object]] = {}
            for row in parameter_rows:
                if (
                    row.get("method") == "pinn"
                    and int(row.get("hidden_width", -1)) == width
                ):
                    seed = int(row["seed"])
                    if seed in parameter_map:
                        raise ValueError(
                            f"duplicate PINN parameter row at width={width}, seed={seed}"
                        )
                    parameter_map[seed] = row
            complete = set(candidate) == set(seed_tuple)
            finite_interior = set(parameter_map) == set(seed_tuple) and all(
                bool(parameter_map[seed].get("finite_interior"))
                for seed in seed_tuple
            )
            every_rmse_below = complete and all(
                float(candidate[seed]["rmse_K"]) < 2.0 for seed in seed_tuple
            )
            passed = bool(finite_interior and every_rmse_below)
            width_results[str(width)] = {
                "all_ten_available": complete,
                "all_ten_finite_and_interior": finite_interior,
                "every_held_out_rmse_strictly_below_2_K": every_rmse_below,
                "passes_n3_reserved_size_screen": passed,
            }
            safety_by_width.append(passed)
            accuracy_by_width.append(passed)
            continue

        paired = set(candidate) == set(seed_tuple) and set(nls) == set(seed_tuple)
        if not paired:
            width_results[str(width)] = {
                "paired_all_ten_available": False,
                "accuracy_win_preserved": False,
                "safety_preserved": False,
            }
            safety_by_width.append(False)
            accuracy_by_width.append(False)
            continue

        candidate_rmse = [float(candidate[seed]["rmse_K"]) for seed in seed_tuple]
        nls_rmse = [float(nls[seed]["rmse_K"]) for seed in seed_tuple]
        candidate_peak = [
            float(candidate[seed]["signed_peak_error_K"]) for seed in seed_tuple
        ]
        nls_peak = [float(nls[seed]["signed_peak_error_K"]) for seed in seed_tuple]

        if trigger.experiment == "e3":
            decision = e3_win_decision(
                candidate_rmse, nls_rmse, candidate_peak, nls_peak
            )
            safety = bool(decision.safety_condition)
            accuracy = bool(
                decision.lower_mean_rmse
                and decision.confidence_interval_excludes_zero_in_favour
            )
            result: dict[str, object] = {
                **decision.as_dict(),
                "accuracy_win_preserved_without_safety_component": accuracy,
                "paired_all_ten_available": True,
                "safety_preserved": safety,
            }
        else:
            interval = paired_mean_bootstrap_interval(candidate_rmse, nls_rmse)
            strict_safety = min(candidate_peak) >= min(nls_peak)
            accuracy = bool(
                interval.mean_difference < 0.0 and interval.upper_95 < 0.0
            )
            safety = bool(strict_safety)
            result = {
                "accuracy_win_preserved_without_safety_component": accuracy,
                "paired_all_ten_available": True,
                "paired_difference_pinn_minus_nls_K": interval.as_dict(),
                "strict_no_worse_worst_signed_peak": safety,
                "pinn_worst_signed_peak_error_K": min(candidate_peak),
                "nls_worst_signed_peak_error_K": min(nls_peak),
                "safety_preserved": safety,
            }
        width_results[str(width)] = result
        safety_by_width.append(safety)
        accuracy_by_width.append(accuracy)

    if trigger.experiment == "e2" and trigger.reference_budget == 3:
        passed = all(safety_by_width)
        consequence = "investigate further" if passed else "reject"
        rule = {
            "both_widths_pass_n3_screen": passed,
            "reserved_size_confirmation_succeeds": passed,
        }
    else:
        both_safe = all(safety_by_width)
        one_accuracy = any(accuracy_by_width)
        passed = bool(both_safe and one_accuracy)
        consequence = (
            "positive result confirmed; eligible for higher-level decision checks"
            if passed
            else "investigate further"
        )
        rule = {
            "both_widths_preserve_safety": both_safe,
            "at_least_one_width_preserves_paired_rmse_win": one_accuracy,
            "confirmation_passed": passed,
        }

    return {
        "cells": cells,
        "confirmation_passed": passed,
        "preregistered_consequence": consequence,
        "rule": rule,
        "trigger": trigger.as_dict(),
        "width_results": width_results,
    }


def _fit_history(fit: SupervisedFit | PinnFit) -> list[dict[str, object]]:
    return [
        {
            "epoch": record.epoch,
            "training_loss": record.training_loss,
            "validation_loss": record.validation_loss,
            "pinn_terms": {name: value for name, value in record.pinn_terms},
        }
        for record in fit.history
    ]


def _fit_record(fit: SupervisedFit | PinnFit) -> dict[str, object]:
    return {
        "available": True,
        "best_epoch": fit.best_epoch,
        "epochs_ran": fit.epochs_ran,
        "history": _fit_history(fit),
        "seed": fit.seed,
        "stopped_early": fit.stopped_early,
    }


def _failure(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _reference_rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(prediction, dtype=np.float64)
    labels = np.asarray(reference, dtype=np.float64)
    if values.shape != labels.shape or values.ndim != 1:
        raise ValueError("reference prediction and labels must be aligned vectors")
    return float(np.sqrt(np.mean((values - labels) ** 2)))


def _pinn_trajectory(
    frame: FeatureFrame,
    truth: TruthRecord,
    reference_features: np.ndarray,
    reference_hotspot_C: np.ndarray,
    top_oil_features: np.ndarray,
    top_oil_values_C: np.ndarray,
) -> PinnTrajectory:
    index = frame.truth_index
    return PinnTrajectory(
        features=frame.X,
        time_s=frame.time_s,
        load_pu=truth.schedule.load_pu[index],
        ambient_C=truth.schedule.ambient_C[index],
        measured_top_oil_C=frame.X[:, 6],
        hotspot_reference_C=np.full(frame.time_s.shape, np.nan, dtype=np.float64),
        reference_features=reference_features,
        reference_hotspot_C=reference_hotspot_C,
        top_oil_measurement_features=top_oil_features,
        top_oil_measurement_values_C=top_oil_values_C,
    )


def _score_e3_plateau(prediction_C: np.ndarray, frame: FeatureFrame) -> TrajectoryMetrics:
    plateau = frame.time_s >= 4.0 * 3600.0
    if not np.any(plateau):
        raise ValueError("E3 confirmation frame has no target plateau")
    return trajectory_metrics(prediction_C[plateau], frame.hotspot_truth_C[plateau])


def _finite_interior_parameters(parameters: Mapping[str, float]) -> bool:
    values = np.asarray([parameters[name] for name in PARAMETER_NAMES], dtype=np.float64)
    return bool(
        np.all(np.isfinite(values))
        and np.all(values > np.asarray(PARAMETER_LOWER))
        and np.all(values < np.asarray(PARAMETER_UPPER))
    )


def _execute_e3_confirmation(trigger: ConfirmationTrigger) -> dict[str, object]:
    """Execute E3 fits after the caller has already claimed test access."""

    assert trigger.target_load_pu is not None
    # All truth, including the hidden target episode, is intentionally created
    # inside this post-claim function.
    train_truth = simulate_truth(
        make_train_schedule(), physics_mode="structural_mismatch"
    )
    validation_truth = simulate_truth(
        make_validation_schedule(), physics_mode="structural_mismatch"
    )
    test_truth = simulate_truth(
        make_e3_schedule(trigger.target_load_pu),
        physics_mode="structural_mismatch",
    )

    metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for seed in RESERVED_CONFIRMATION_SEEDS:
        observed_train = observe_record(train_truth, seed=seed)
        observed_validation = observe_record(validation_truth, seed=seed)
        observed_test = observe_record(test_truth, seed=seed)
        references_train = observe_hotspot_references(
            train_truth, budget=E3_REFERENCE_BUDGET, seed=seed
        )
        references_validation = observe_hotspot_references(
            validation_truth, budget=E3_REFERENCE_BUDGET, seed=seed
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

        nls = fit_nls(observed_train, references_train)
        nls_prediction: np.ndarray | None = None
        if isinstance(nls, NLSModel):
            nls_prediction = prediction_on_feature_frame(
                predict_nls(nls, test_truth.schedule), test_frame
            )
            nls_score = _score_e3_plateau(nls_prediction, test_frame)
            nls_record: dict[str, object] = nls.as_dict()
        else:
            nls_score = None
            nls_record = {"available": False, "refusal": nls.as_dict()}

        for width in RESERVED_HIDDEN_WIDTHS:
            base = {
                "hidden_width": width,
                "method": trigger.method,
                "reference_budget": E3_REFERENCE_BUDGET,
                "reference_feature_alignment_max_horizon_s": 0.0,
                "seed": seed,
                "top_oil_measurement_interval_s": 300.0,
            }
            training: dict[str, object] = {**base, "nls": nls_record}
            if nls_score is not None:
                metric_rows.append(
                    {
                        "hidden_width": width,
                        "method": "nls",
                        "seed": seed,
                        "target_load_pu": trigger.target_load_pu,
                        **nls_score.as_dict(),
                    }
                )
                availability_rows.append(
                    {
                        "available": True,
                        "hidden_width": width,
                        "method": "nls",
                        "seed": seed,
                    }
                )
            else:
                availability_rows.append(
                    {
                        "available": False,
                        "hidden_width": width,
                        "method": "nls",
                        "refusal": nls.as_dict(),
                        "seed": seed,
                    }
                )

            try:
                if trigger.method == "plain_nn":
                    fit: SupervisedFit | PinnFit | HullAwareResidualFit = fit_plain_nn(
                        train_reference_features,
                        references_train.temperature_C,
                        validation_reference_features,
                        references_validation.temperature_C,
                        seed=seed,
                        hidden_width=width,
                        standardizer_features=train_frame.X,
                    )
                    prediction = fit.predict(test_frame.X)
                    validation_prediction = fit.predict(
                        validation_reference_features
                    )
                    training["fit"] = _fit_record(fit)
                elif trigger.method == "pinn":
                    fit = fit_pinn(
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
                        hidden_width=width,
                    )
                    prediction = fit.predict_states(test_frame.X)["hotspot_C"]
                    validation_prediction = fit.predict_states(
                        validation_reference_features
                    )["hotspot_C"]
                    fit_payload = _fit_record(fit)
                    fit_payload["parameters"] = dict(fit.parameter_values())
                    training["fit"] = fit_payload
                else:
                    if not isinstance(nls, NLSModel) or nls_prediction is None:
                        raise RuntimeError("grey-box confirmation requires available NLS")
                    nls_train = predict_nls(nls, train_truth.schedule)
                    nls_validation = predict_nls(nls, validation_truth.schedule)
                    nls_train_references = np.asarray(nls_train.hotspot_C)[
                        references_train.index
                    ]
                    nls_validation_references = np.asarray(nls_validation.hotspot_C)[
                        references_validation.index
                    ]
                    fit = fit_residual_nn(
                        train_reference_features,
                        references_train.temperature_C - nls_train_references,
                        validation_reference_features,
                        references_validation.temperature_C
                        - nls_validation_references,
                        train_load_pu=train_truth.schedule.load_pu,
                        seed=seed,
                        hidden_width=width,
                        standardizer_features=train_frame.X,
                    )
                    load = test_truth.schedule.load_pu[test_frame.truth_index]
                    result = fit.predict(
                        test_frame.X, load, nls_hotspot_C=nls_prediction
                    )
                    assert result.hotspot_C is not None
                    prediction = result.hotspot_C
                    validation_result = fit.predict(
                        validation_reference_features,
                        validation_truth.schedule.load_pu[
                            references_validation.index
                        ],
                        nls_hotspot_C=nls_validation_references,
                    )
                    assert validation_result.hotspot_C is not None
                    validation_prediction = validation_result.hotspot_C
                    fit_payload = _fit_record(fit.fit)
                    fit_payload.update(
                        {
                            "hull_min_pu": fit.hull_min_pu,
                            "hull_max_pu": fit.hull_max_pu,
                        }
                    )
                    training["fit"] = fit_payload

                score = _score_e3_plateau(
                    np.asarray(prediction, dtype=np.float64), test_frame
                )
                metric_rows.append(
                    {
                        "hidden_width": width,
                        "method": trigger.method,
                        "seed": seed,
                        "target_load_pu": trigger.target_load_pu,
                        **score.as_dict(),
                    }
                )
                availability_rows.append(
                    {
                        "available": True,
                        "hidden_width": width,
                        "method": trigger.method,
                        "seed": seed,
                    }
                )
                training["validation_reference_rmse_K"] = _reference_rmse(
                    np.asarray(validation_prediction, dtype=np.float64),
                    references_validation.temperature_C,
                )
            except (ValueError, RuntimeError, FloatingPointError) as error:
                failure = _failure(error)
                availability_rows.append(
                    {
                        "available": False,
                        "failure": failure,
                        "hidden_width": width,
                        "method": trigger.method,
                        "seed": seed,
                    }
                )
                training["fit"] = {"available": False, "failure": failure}
            training_rows.append(training)
            print(
                f"E3 confirmation {trigger.method}, "
                f"K={trigger.target_load_pu:.2f}, width={width}, seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            if "fit" in locals():
                del fit
            gc.collect()

    return {
        "availability_rows": availability_rows,
        "parameter_rows": parameter_rows,
        "resolved": aggregate_confirmation_rows(
            trigger, metric_rows, parameter_rows
        ),
        "rows": metric_rows,
        "training": training_rows,
    }


def _execute_e2_confirmation(trigger: ConfirmationTrigger) -> dict[str, object]:
    """Execute E2 PINN confirmation after test access has been claimed."""

    assert trigger.reference_budget is not None
    budget = trigger.reference_budget
    train_truth = simulate_truth(make_train_schedule(), physics_mode="matched")
    validation_truth = simulate_truth(
        make_validation_schedule(), physics_mode="matched"
    )
    test_truth = simulate_truth(
        make_in_range_test_schedule(), physics_mode="matched"
    )

    metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for seed in RESERVED_CONFIRMATION_SEEDS:
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

        nls = fit_nls(observed_train, references_train)
        if isinstance(nls, NLSModel):
            nls_prediction = prediction_on_feature_frame(
                predict_nls(nls, test_truth.schedule), test_frame
            )
            nls_score = trajectory_metrics(
                nls_prediction, test_frame.hotspot_truth_C
            )
            nls_parameters = dict(nls.parameter_mapping())
            nls_record: dict[str, object] = nls.as_dict()
        else:
            nls_score = None
            nls_parameters = None
            nls_record = {"available": False, "refusal": nls.as_dict()}

        for width in RESERVED_HIDDEN_WIDTHS:
            base = {
                "hidden_width": width,
                "method": "pinn",
                "reference_budget": budget,
                "reference_feature_alignment_max_horizon_s": 0.0,
                "seed": seed,
                "top_oil_measurement_interval_s": 300.0,
            }
            training: dict[str, object] = {**base, "nls": nls_record}
            if nls_score is not None and nls_parameters is not None:
                metric_rows.append(
                    {
                        "hidden_width": width,
                        "method": "nls",
                        "reference_budget": budget,
                        "seed": seed,
                        **nls_score.as_dict(),
                    }
                )
                parameter_rows.append(
                    {
                        "errors": parameter_percent_errors(nls_parameters),
                        "finite_interior": True,
                        "hidden_width": width,
                        "method": "nls",
                        "parameters": nls_parameters,
                        "reference_budget": budget,
                        "seed": seed,
                    }
                )
                availability_rows.append(
                    {
                        "available": True,
                        "hidden_width": width,
                        "method": "nls",
                        "seed": seed,
                    }
                )
            else:
                availability_rows.append(
                    {
                        "available": False,
                        "hidden_width": width,
                        "method": "nls",
                        "refusal": nls.as_dict(),
                        "seed": seed,
                    }
                )

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
                    hidden_width=width,
                )
                prediction = pinn.predict_states(test_frame.X)["hotspot_C"]
                score = trajectory_metrics(
                    prediction, test_frame.hotspot_truth_C
                )
                parameters = dict(pinn.parameter_values())
                finite_interior = _finite_interior_parameters(parameters)
                metric_rows.append(
                    {
                        "hidden_width": width,
                        "method": "pinn",
                        "reference_budget": budget,
                        "seed": seed,
                        **score.as_dict(),
                    }
                )
                parameter_rows.append(
                    {
                        "errors": parameter_percent_errors(parameters),
                        "finite_interior": finite_interior,
                        "hidden_width": width,
                        "method": "pinn",
                        "parameters": parameters,
                        "reference_budget": budget,
                        "seed": seed,
                    }
                )
                availability_rows.append(
                    {
                        "available": True,
                        "hidden_width": width,
                        "method": "pinn",
                        "seed": seed,
                    }
                )
                fit_payload = _fit_record(pinn)
                fit_payload.update(
                    {
                        "finite_interior": finite_interior,
                        "parameters": parameters,
                        "validation_reference_rmse_K": _reference_rmse(
                            pinn.predict_states(validation_reference_features)[
                                "hotspot_C"
                            ],
                            references_validation.temperature_C,
                        ),
                    }
                )
                training["fit"] = fit_payload
            except (ValueError, RuntimeError, FloatingPointError) as error:
                failure = _failure(error)
                availability_rows.append(
                    {
                        "available": False,
                        "failure": failure,
                        "hidden_width": width,
                        "method": "pinn",
                        "seed": seed,
                    }
                )
                training["fit"] = {"available": False, "failure": failure}
            training_rows.append(training)
            print(
                f"E2 confirmation N={budget}, width={width}, seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            if "pinn" in locals():
                del pinn
            gc.collect()

    return {
        "availability_rows": availability_rows,
        "parameter_rows": parameter_rows,
        "resolved": aggregate_confirmation_rows(
            trigger, metric_rows, parameter_rows
        ),
        "rows": metric_rows,
        "training": training_rows,
    }


def _execute_confirmation_after_claim(
    trigger: ConfirmationTrigger,
) -> dict[str, object]:
    if trigger.experiment == "e3":
        return _execute_e3_confirmation(trigger)
    return _execute_e2_confirmation(trigger)


@record_primary_failures
def run_reserved_confirmation(
    repository: str | Path,
    *,
    trigger: ConfirmationTrigger,
    prior_run_id: str,
    override: bool = False,
    override_reason: str | None = None,
) -> Mapping[str, object]:
    """Run one preregistered confirmation after verifying its exact trigger."""

    enforce_cpu_only_environment()
    torch_status = require_torch_cpu_only()
    repository_path = Path(repository).resolve()

    # Evidence verification intentionally precedes the sentinel.  A missing or
    # merely plausible trigger must not consume reserved confirmation access.
    source = verify_completed_primary_trigger(
        repository_path, trigger=trigger, prior_run_id=prior_run_id
    )
    configuration = confirmation_configuration(trigger, source)
    run = begin_primary_run(
        repository_path,
        experiment=trigger.sentinel_experiment,
        configuration=configuration,
        seeds=RESERVED_CONFIRMATION_SEEDS,
        command=recorded_confirmation_command(
            trigger,
            prior_run_id=prior_run_id,
            override=override,
            override_reason=override_reason,
        ),
        override=override,
        override_reason=override_reason,
    )

    # No hidden truth is generated until the durable claim above has returned.
    execution = _execute_confirmation_after_claim(trigger)
    aggregate = {
        "configuration": configuration,
        **execution,
        "schema_version": 1,
        "torch": {
            "cpu_only": torch_status.cpu_only,
            "cuda_available": torch_status.cuda_available,
            "version": torch_status.version,
            "visible_cuda_device_count": torch_status.visible_cuda_device_count,
        },
    }
    final = finish_primary_run(run, aggregate)
    return {"aggregate": aggregate, "manifest": dict(final), "run_id": run.run_id}


__all__ = [
    "ConfirmationTrigger",
    "E2_CONFIRMATION_BUDGETS",
    "E3_CONFIRMABLE_METHODS",
    "E3_CONFIRMATION_LOADS_PU",
    "RESERVED_CONFIRMATION_SEEDS",
    "RESERVED_HIDDEN_WIDTHS",
    "VerifiedPrimaryTrigger",
    "aggregate_confirmation_rows",
    "confirmation_configuration",
    "recorded_confirmation_command",
    "run_reserved_confirmation",
    "verify_completed_primary_trigger",
]
