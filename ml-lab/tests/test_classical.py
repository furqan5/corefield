from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from corefield_ml_lab.classical import (
    NLSModel,
    NLSRefusal,
    fit_nls,
    parameter_percent_errors,
    predict_generic,
    prediction_on_feature_frame,
    references_on_feature_frame,
)
from corefield_ml_lab.synthetic_lab import (
    NOMINAL_PARAMS,
    build_feature_frame,
    make_train_schedule,
    observe_hotspot_references,
    observe_record,
    simulate_truth,
)


@pytest.fixture(scope="module")
def train_inputs():
    truth = simulate_truth(make_train_schedule(), physics_mode="matched")
    observed = observe_record(truth, seed=31000)
    references = observe_hotspot_references(truth, budget=20, seed=31000)
    return truth, observed, references


def test_adapter_passes_only_sparse_measured_oil_and_frozen_settings(train_inputs) -> None:
    _truth, observed, references = train_inputs
    captured = {}

    def identifier(time_s, load, ambient, top_oil, refs, **kwargs):
        captured.update(
            time_s=time_s,
            top_oil=top_oil,
            refs=refs,
            kwargs=kwargs,
        )
        return SimpleNamespace(
            success=True,
            params=NOMINAL_PARAMS,
            residual_rmse_K=0.1,
            oil_residual_rmse_K=0.1,
            hotspot_residual_rmse_K=0.1,
            jacobian_condition=10.0,
            warnings=(),
        )

    outcome = fit_nls(observed, references, identifier=identifier)
    assert isinstance(outcome, NLSModel)
    finite = np.flatnonzero(np.isfinite(captured["top_oil"]))
    assert np.array_equal(finite, observed.top_oil_index)
    assert np.array_equal(captured["top_oil"][finite], observed.top_oil_C)
    assert captured["kwargs"]["loss"] == "linear"
    assert captured["kwargs"]["loss_ratio_R"] == 6.0
    assert np.array_equal(captured["refs"].temperature_C, references.temperature_C)


def test_nls_refusal_is_explicit(train_inputs) -> None:
    _truth, observed, references = train_inputs

    def refuse(*args, **kwargs):
        raise ValueError("fewer than four")

    outcome = fit_nls(observed, references, identifier=refuse)
    assert isinstance(outcome, NLSRefusal)
    assert outcome.reason_type == "ValueError"
    assert "fewer than four" in outcome.message


def test_reference_mapping_floors_to_avoid_future_features(train_inputs) -> None:
    truth, observed, references = train_inputs
    frame = build_feature_frame(observed)
    labels = references_on_feature_frame(references, frame)
    rows = np.flatnonzero(np.isfinite(labels))
    assert rows.size == 20
    for reference_time in references.time_s:
        row = np.searchsorted(frame.time_s, reference_time, side="right") - 1
        assert frame.time_s[row] <= reference_time
        assert reference_time - frame.time_s[row] < 120.0


def test_generic_and_feature_sampling_share_declared_rows(train_inputs) -> None:
    truth, observed, _references = train_inputs
    frame = build_feature_frame(observed)
    trajectory = predict_generic(truth.schedule)
    sampled = prediction_on_feature_frame(trajectory, frame)
    assert np.array_equal(sampled, trajectory.hotspot_C[frame.truth_index])


def test_parameter_percent_errors_preserve_sign_and_units() -> None:
    errors = parameter_percent_errors(
        {
            "delta_theta_or_K": 49.5,
            "tau_o_min": 135.0,
            "delta_theta_hr_K": 22.0,
            "tau_w_min": 7.7,
        }
    )
    assert errors["delta_theta_or_K"]["signed_percent"] == pytest.approx(10.0)
    assert errors["tau_o_min"]["signed_percent"] == pytest.approx(-10.0)
    assert errors["median_absolute_percent_across_parameters"]["value"] == pytest.approx(10.0)
