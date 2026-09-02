"""Adapters for the frozen CoreField nonlinear least-squares baseline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .synthetic_lab import (
    FeatureFrame,
    MATCHED_CONSTANTS,
    NOMINAL_PARAMS,
    ObservedRecord,
    Schedule,
    SparseHotspotReferences,
)


def _load_estimator():
    module = import_module("corefield.estimator")
    expected = (
        Path(__file__).resolve().parents[2] / "vendor" / "corefield" / "estimator.py"
    ).resolve()
    actual = Path(module.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"NLS resolved to {actual}, expected frozen file {expected}")
    return module


_estimator = _load_estimator()
_physics = import_module("corefield.iec60076_7")


@dataclass(frozen=True, slots=True)
class NLSRefusal:
    """An explicit unavailable result; never converted to a numeric score."""

    reason_type: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"reason_type": self.reason_type, "message": self.message}


@dataclass(frozen=True, slots=True)
class NLSModel:
    """Successful frozen four-parameter NLS fit."""

    params: Any
    residual_rmse_K: float
    oil_residual_rmse_K: float
    hotspot_residual_rmse_K: float
    jacobian_condition: float
    warnings: tuple[str, ...]

    def parameter_mapping(self) -> Mapping[str, float]:
        return {
            "delta_theta_or_K": float(self.params.delta_theta_or_K),
            "tau_o_min": float(self.params.tau_o_min),
            "delta_theta_hr_K": float(self.params.delta_theta_hr_K),
            "tau_w_min": float(self.params.tau_w_min),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "available": True,
            "parameters": dict(self.parameter_mapping()),
            "residual_rmse_K": self.residual_rmse_K,
            "oil_residual_rmse_K": self.oil_residual_rmse_K,
            "hotspot_residual_rmse_K": self.hotspot_residual_rmse_K,
            "jacobian_condition": self.jacobian_condition,
            "warnings": list(self.warnings),
        }


NLSOutcome = NLSModel | NLSRefusal


def fit_nls(
    observed: ObservedRecord,
    references: SparseHotspotReferences,
    *,
    identifier: Callable[..., Any] | None = None,
) -> NLSOutcome:
    """Fit the frozen estimator or return its explicit refusal.

    Top oil is dense only at the five-minute measurement indices and NaN at
    all other 30 s truth-grid rows.  No hidden top-oil or hot-spot truth is
    passed to the estimator.
    """

    if observed.split != "train" or references.split != "train":
        raise ValueError("NLS fitting is confined to the train split")
    if observed.seed != references.seed:
        raise ValueError("observed top-oil and hot-spot references must share a seed")
    schedule = observed.truth.schedule
    if np.any(references.time_s < schedule.time_s[0]) or np.any(
        references.time_s > schedule.time_s[-1]
    ):
        raise ValueError("hot-spot references lie outside the observed record")

    top_oil = np.full(schedule.time_s.shape, np.nan, dtype=np.float64)
    top_oil[observed.top_oil_index] = observed.top_oil_C
    hotspot_refs = _estimator.HotspotReferences(
        time_s=np.asarray(references.time_s, dtype=np.float64),
        temperature_C=np.asarray(references.temperature_C, dtype=np.float64),
    )
    identify = _estimator.identify if identifier is None else identifier
    try:
        result = identify(
            schedule.time_s,
            schedule.load_pu,
            schedule.ambient_C,
            top_oil,
            hotspot_refs,
            constants=MATCHED_CONSTANTS,
            loss_ratio_R=6.0,
            loss="linear",
            load_pu_half=schedule.load_pu_half,
            ambient_C_half=schedule.ambient_C_half,
        )
    except (ValueError, RuntimeError) as error:
        return NLSRefusal(type(error).__name__, str(error))
    if not bool(result.success):
        return NLSRefusal("non_convergence", "identifier returned success=False")
    return NLSModel(
        params=result.params,
        residual_rmse_K=float(result.residual_rmse_K),
        oil_residual_rmse_K=float(result.oil_residual_rmse_K),
        hotspot_residual_rmse_K=float(result.hotspot_residual_rmse_K),
        jacobian_condition=float(result.jacobian_condition),
        warnings=tuple(str(value) for value in result.warnings),
    )


def simulate_model(
    schedule: Schedule,
    params: Any,
) -> Any:
    """Run fixed-form CoreField physics on one declared schedule."""

    return _physics.simulate(
        schedule.time_s,
        schedule.load_pu,
        schedule.ambient_C,
        params,
        MATCHED_CONSTANTS,
        load_pu_half=schedule.load_pu_half,
        ambient_C_half=schedule.ambient_C_half,
        solver="rk4",
    )


def predict_nls(model: NLSModel, schedule: Schedule) -> Any:
    """Predict with one successful NLS model."""

    return simulate_model(schedule, model.params)


def predict_generic(schedule: Schedule) -> Any:
    """Generic nominal-parameter comparator with no identification."""

    return simulate_model(schedule, NOMINAL_PARAMS)


def prediction_on_feature_frame(trajectory: Any, frame: FeatureFrame) -> NDArray[np.float64]:
    """Sample a full 30 s model trajectory on a frame's common 2 min rows."""

    prediction = np.asarray(trajectory.hotspot_C, dtype=np.float64)
    if np.max(frame.truth_index) >= prediction.size:
        raise ValueError("feature frame indices exceed trajectory length")
    return prediction[frame.truth_index].copy()


def references_on_feature_frame(
    references: SparseHotspotReferences,
    frame: FeatureFrame,
) -> NDArray[np.float64]:
    """Place sparse labels on the latest non-future 2 min feature row.

    Candidate times at 3 min offsets can fall between the fixed 2 min grid.
    Flooring prevents future-feature leakage; the maximum label horizon is
    strictly less than 2 min and is recorded by the experiment runner.
    """

    if references.split != frame.split:
        raise ValueError("reference and feature splits differ")
    labels = np.full(frame.time_s.shape, np.nan, dtype=np.float64)
    for time_s, temperature_C in zip(
        references.time_s, references.temperature_C, strict=True
    ):
        row = int(np.searchsorted(frame.time_s, time_s, side="right") - 1)
        if row < 0:
            raise ValueError("a reference precedes the complete-lag feature frame")
        if labels[row] == labels[row]:
            raise ValueError("two references map to the same 2 min feature row")
        labels[row] = float(temperature_C)
    return labels


def parameter_percent_errors(
    estimated: Mapping[str, float],
) -> Mapping[str, Mapping[str, float]]:
    """Signed and absolute percent error against frozen synthetic truth."""

    truth = {
        "delta_theta_or_K": 45.0,
        "tau_o_min": 150.0,
        "delta_theta_hr_K": 22.0,
        "tau_w_min": 7.0,
    }
    output: dict[str, Mapping[str, float]] = {}
    absolute: list[float] = []
    for name, truth_value in truth.items():
        value = float(estimated[name])
        signed = 100.0 * (value - truth_value) / truth_value
        magnitude = abs(signed)
        absolute.append(magnitude)
        output[name] = {
            "signed_percent": signed,
            "absolute_percent": magnitude,
        }
    output["median_absolute_percent_across_parameters"] = {
        "value": float(np.median(absolute))
    }
    return output


__all__ = [
    "NLSModel",
    "NLSOutcome",
    "NLSRefusal",
    "fit_nls",
    "parameter_percent_errors",
    "predict_generic",
    "predict_nls",
    "prediction_on_feature_frame",
    "references_on_feature_frame",
    "simulate_model",
]
