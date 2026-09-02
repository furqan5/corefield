from __future__ import annotations

import os

import numpy as np
import pytest


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch = pytest.importorskip("torch")

from corefield_ml_lab.ml_methods import (  # noqa: E402
    CPU_DEVICE,
    NOMINAL_PARAMETERS,
    PARAMETER_LOWER,
    PARAMETER_UPPER,
    PINN_LOSS_KEYS,
    FeatureStandardizer,
    PhysicsInformedNetwork,
    PinnTrajectory,
    PlainHotspotNetwork,
    ResidualNetwork,
    TrainingConfig,
    centered_finite_difference,
    fit_pinn,
    fit_plain_nn,
    fit_residual_nn,
    pinn_loss_terms,
    pinn_total_loss,
)


def _features(n_rows: int, *, offset: float = 0.0) -> np.ndarray:
    row = np.linspace(-1.0, 1.0, n_rows, dtype=np.float64)[:, None]
    column = np.arange(9, dtype=np.float64)[None, :]
    return np.sin(row * (column + 1.0)) + offset


def _plain_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = _features(24)
    train_y = 65.0 + 1.5 * train_x[:, 0] - 0.4 * train_x[:, 4]
    train_y[[1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22]] = np.nan
    validation_x = _features(12, offset=0.15)
    validation_y = 64.0 + validation_x[:, 0]
    validation_y[[0, 2, 3, 5, 6, 8, 9, 11]] = np.nan
    return train_x, train_y, validation_x, validation_y


def _pinn_data() -> tuple[PinnTrajectory, np.ndarray, np.ndarray]:
    n_rows = 10
    features = _features(n_rows)
    time_s = np.arange(n_rows, dtype=np.float64) * 120.0
    load = np.linspace(0.70, 0.90, n_rows, dtype=np.float64)
    ambient = 20.0 + np.linspace(0.0, 0.5, n_rows, dtype=np.float64)
    top_oil = ambient + 28.0
    top_oil[[1, 3, 5, 7, 9]] = np.nan
    hotspot = ambient + 45.0
    hotspot[[0, 2, 3, 5, 6, 8, 9]] = np.nan
    trajectory = PinnTrajectory(
        features=features,
        time_s=time_s,
        load_pu=load,
        ambient_C=ambient,
        measured_top_oil_C=top_oil,
        hotspot_reference_C=hotspot,
    )
    validation_x = _features(6, offset=0.1)
    validation_y = np.array([65.0, np.nan, 66.0, np.nan, 67.0, np.nan])
    return trajectory, validation_x, validation_y


def _linear_widths(model: torch.nn.Module) -> list[tuple[int, int]]:
    return [
        (layer.in_features, layer.out_features)
        for layer in model.modules()
        if isinstance(layer, torch.nn.Linear)
    ]


def _state_dict_arrays(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


def test_frozen_training_defaults() -> None:
    config = TrainingConfig()
    assert config.learning_rate == 1.0e-3
    assert config.weight_decay == 1.0e-6
    assert config.max_epochs == 2_000
    assert config.patience == 150
    assert config.minimum_normalized_improvement == 1.0e-4
    assert config.intraop_threads == 1


@pytest.mark.parametrize(
    ("model_type", "expected_widths", "output_width"),
    [
        (PlainHotspotNetwork, [(9, 16), (16, 16), (16, 1)], 1),
        (PhysicsInformedNetwork, [(9, 16), (16, 16), (16, 3)], 3),
        (ResidualNetwork, [(9, 8), (8, 1)], 1),
    ],
)
def test_frozen_architectures_are_cpu_only(
    model_type: type[torch.nn.Module],
    expected_widths: list[tuple[int, int]],
    output_width: int,
) -> None:
    model = model_type().to(CPU_DEVICE)
    assert _linear_widths(model) == expected_widths
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    output = model(torch.zeros((5, 9), dtype=torch.float32, device=CPU_DEVICE))
    expected_shape = (5, 3) if output_width == 3 else (5,)
    assert output.shape == expected_shape
    assert output.device.type == "cpu"


@pytest.mark.parametrize("width", [8, 32])
def test_reserved_confirmation_widths_propagate_to_all_neural_architectures(
    width: int,
) -> None:
    assert _linear_widths(PlainHotspotNetwork(width)) == [
        (9, width),
        (width, width),
        (width, 1),
    ]
    assert _linear_widths(PhysicsInformedNetwork(width)) == [
        (9, width),
        (width, width),
        (width, 3),
    ]
    assert _linear_widths(ResidualNetwork(width)) == [(9, width), (width, 1)]


def test_feature_standardizer_uses_train_rows_only() -> None:
    train = _features(8)
    validation = _features(5, offset=100.0)
    scaler = FeatureStandardizer.fit(train)
    scaler.transform(validation)
    np.testing.assert_array_equal(scaler.mean, train.mean(axis=0))
    assert not np.allclose(scaler.mean, np.vstack((train, validation)).mean(axis=0))


def test_plain_fit_is_bitwise_deterministic_for_same_seed() -> None:
    train_x, train_y, validation_x, validation_y = _plain_data()
    config = TrainingConfig(max_epochs=7, patience=7)
    first = fit_plain_nn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=31_000,
        config=config,
    )
    second = fit_plain_nn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=31_000,
        config=config,
    )
    assert first.device.type == second.device.type == "cpu"
    for name, first_value in _state_dict_arrays(first.model).items():
        np.testing.assert_array_equal(first_value, _state_dict_arrays(second.model)[name])
    np.testing.assert_array_equal(
        first.predict(validation_x), second.predict(validation_x)
    )
    assert first.history == second.history


def test_sparse_supervised_fit_uses_full_train_frame_for_standardization() -> None:
    train_x, train_y, validation_x, validation_y = _plain_data()
    full_train_frame = np.vstack((train_x, train_x + 20.0))
    fit = fit_plain_nn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=123,
        config=TrainingConfig(max_epochs=1, patience=1),
        standardizer_features=full_train_frame,
    )
    np.testing.assert_allclose(
        fit.standardizer.mean,
        full_train_frame.mean(axis=0),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(fit.standardizer.mean, train_x.mean(axis=0))


def test_validation_labels_do_not_enter_plain_gradient_update() -> None:
    train_x, train_y, validation_x, validation_y = _plain_data()
    alternate_validation = validation_y.copy()
    alternate_validation[np.isfinite(alternate_validation)] += 1_000.0
    config = TrainingConfig(max_epochs=1, patience=1)
    first = fit_plain_nn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=123,
        config=config,
    )
    second = fit_plain_nn(
        train_x,
        train_y,
        validation_x,
        alternate_validation,
        seed=123,
        config=config,
    )
    for name, first_value in _state_dict_arrays(first.model).items():
        np.testing.assert_array_equal(first_value, _state_dict_arrays(second.model)[name])
    assert first.history[0].training_loss == second.history[0].training_loss
    assert first.history[0].validation_loss != second.history[0].validation_loss


def test_sigmoid_parameterization_stays_inside_frozen_bounds() -> None:
    model = PhysicsInformedNetwork()
    nominal = model.bounded_parameters().detach().cpu().numpy()
    np.testing.assert_allclose(nominal, NOMINAL_PARAMETERS, rtol=0.0, atol=2.0e-5)

    with torch.no_grad():
        model.raw_parameters.copy_(
            torch.tensor([-100.0, 100.0, -20.0, 20.0], dtype=torch.float32)
        )
    bounded = model.bounded_parameters().detach().cpu().numpy()
    assert np.all(bounded >= np.asarray(PARAMETER_LOWER))
    assert np.all(bounded <= np.asarray(PARAMETER_UPPER))
    assert bounded[0] == PARAMETER_LOWER[0]
    assert bounded[1] == PARAMETER_UPPER[1]


def test_centered_finite_difference_is_centered_and_unit_consistent() -> None:
    time_s = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0], dtype=torch.float64)
    temperature_K = time_s**2
    derivative_K_per_s = centered_finite_difference(temperature_K, time_s)
    torch.testing.assert_close(
        derivative_K_per_s,
        torch.tensor([4.0, 8.0, 12.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_pinn_exposes_exactly_five_finite_dimensionless_terms() -> None:
    trajectory, _, _ = _pinn_data()
    scaler = FeatureStandardizer.fit(trajectory.features)
    features = torch.as_tensor(scaler.transform(trajectory.features))
    model = PhysicsInformedNetwork()
    terms = pinn_loss_terms(
        model,
        features,
        torch.as_tensor(trajectory.time_s, dtype=torch.float64),
        torch.as_tensor(trajectory.load_pu, dtype=torch.float64),
        torch.as_tensor(trajectory.ambient_C, dtype=torch.float64),
        torch.as_tensor(trajectory.measured_top_oil_C, dtype=torch.float64),
        torch.as_tensor(trajectory.hotspot_reference_C, dtype=torch.float64),
    )
    assert tuple(terms) == PINN_LOSS_KEYS
    assert len(terms) == 5
    assert all(value.ndim == 0 for value in terms.values())
    assert all(torch.isfinite(value) for value in terms.values())
    total = pinn_total_loss(terms)
    torch.testing.assert_close(
        total,
        torch.stack(tuple(terms.values())).mean(),
        rtol=0.0,
        atol=0.0,
    )
    total.backward()
    assert model.raw_parameters.grad is not None
    assert torch.all(torch.isfinite(model.raw_parameters.grad))


def test_pinn_measurement_terms_use_separate_exact_timestamp_feature_rows() -> None:
    class FeatureEchoStates(PhysicsInformedNetwork):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            zero = torch.zeros_like(features[:, 0])
            return torch.stack((features[:, 0], zero, zero), dim=1)

    model = FeatureEchoStates()
    n_rows = 5
    main_features = torch.zeros((n_rows, 9), dtype=torch.float32)
    time = torch.arange(n_rows, dtype=torch.float64) * 120.0
    load = torch.full((n_rows,), 0.8, dtype=torch.float64)
    ambient = torch.full((n_rows,), 20.0, dtype=torch.float64)
    # These deliberately wrong aligned labels must be ignored when separate
    # exact-timestamp inputs are supplied.
    aligned_wrong = torch.full((n_rows,), 999.0, dtype=torch.float64)
    oil_features = torch.zeros((2, 9), dtype=torch.float32)
    oil_features[:, 0] = torch.tensor([1.0, 2.0])
    oil_values = torch.tensor([1.5, 1.5], dtype=torch.float64)
    reference_features = torch.zeros((2, 9), dtype=torch.float32)
    reference_features[:, 0] = torch.tensor([3.0, 4.0])
    reference_values = torch.tensor([3.0, 3.5], dtype=torch.float64)
    terms = pinn_loss_terms(
        model,
        main_features,
        time,
        load,
        ambient,
        aligned_wrong,
        aligned_wrong,
        reference_features,
        reference_values,
        oil_features,
        oil_values,
    )
    # sigma=0.5 K: oil normalized errors [-1,+1], hotspot [0,+1].
    torch.testing.assert_close(
        terms["top_oil_measurement"], torch.tensor(1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        terms["hotspot_reference"], torch.tensor(0.5, dtype=torch.float64)
    )


def test_opening_equilibrium_is_part_of_each_ode_term() -> None:
    class PrescribedStates(PhysicsInformedNetwork):
        def __init__(self, states: torch.Tensor) -> None:
            super().__init__()
            self.states = states

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.states

    n_rows = 5
    load = torch.full((n_rows,), 0.8, dtype=torch.float64)
    ambient = torch.full((n_rows,), 20.0, dtype=torch.float64)
    time = torch.arange(n_rows, dtype=torch.float64) * 120.0
    base_model = PhysicsInformedNetwork()
    params = base_model.bounded_parameters().detach().to(torch.float64)
    dtheta_or, _, dtheta_hr, _ = params
    loss_factor = ((1.0 + 6.0 * load**2) / 7.0) ** 0.8
    winding_factor = load**1.3
    equilibrium = torch.stack(
        (
            ambient + dtheta_or * loss_factor,
            2.0 * dtheta_hr * winding_factor,
            dtheta_hr * winding_factor,
        ),
        dim=1,
    ).to(torch.float32)
    perturbed = equilibrium.clone()
    perturbed[0, :] += torch.tensor([1.0, 1.0, 1.0])
    model = PrescribedStates(perturbed)
    with torch.no_grad():
        model.raw_parameters.copy_(base_model.raw_parameters)
    hotspot = equilibrium[:, 0] + equilibrium[:, 1] - equilibrium[:, 2]
    terms = pinn_loss_terms(
        model,
        torch.zeros((n_rows, 9), dtype=torch.float32),
        time,
        load,
        ambient,
        equilibrium[:, 0].to(torch.float64),
        hotspot.to(torch.float64),
    )
    assert terms["oil_ode"] > 0.0
    assert terms["winding_fast_ode"] > 0.0
    assert terms["winding_slow_ode"] > 0.0


def test_validation_labels_do_not_enter_pinn_gradient_update() -> None:
    trajectory, validation_x, validation_y = _pinn_data()
    alternate_validation = validation_y.copy()
    alternate_validation[np.isfinite(alternate_validation)] -= 500.0
    config = TrainingConfig(max_epochs=1, patience=1)
    first = fit_pinn(
        trajectory,
        validation_x,
        validation_y,
        seed=31_000,
        config=config,
    )
    second = fit_pinn(
        trajectory,
        validation_x,
        alternate_validation,
        seed=31_000,
        config=config,
    )
    for name, first_value in _state_dict_arrays(first.model).items():
        np.testing.assert_array_equal(first_value, _state_dict_arrays(second.model)[name])
    assert first.history[0].training_loss == second.history[0].training_loss
    assert first.history[0].validation_loss != second.history[0].validation_loss
    assert tuple(name for name, _ in first.history[0].pinn_terms) == PINN_LOSS_KEYS


def test_hull_gate_is_inclusive_and_outside_values_are_bit_exact() -> None:
    train_x, train_y, validation_x, validation_y = _plain_data()
    train_load = np.linspace(0.60, 0.95, train_x.shape[0])
    fit = fit_residual_nn(
        train_x,
        train_y - 65.0,
        validation_x,
        validation_y - 65.0,
        train_load_pu=train_load,
        seed=31_000,
        config=TrainingConfig(max_epochs=2, patience=2),
    )
    query_x = _features(4, offset=0.05)
    query_load = np.array([0.59, 0.60, 0.95, 0.96], dtype=np.float64)
    baseline = np.array([-0.0, 60.0, 70.0, -123.25], dtype=np.float64)
    prediction = fit.predict(
        query_x,
        query_load,
        nls_hotspot_C=baseline,
    )
    np.testing.assert_array_equal(
        prediction.extrapolation_flag,
        np.array([True, False, False, True]),
    )
    outside = prediction.extrapolation_flag
    assert prediction.residual_K[outside].view(np.uint64).tolist() == [0, 0]
    assert prediction.hotspot_C is not None
    np.testing.assert_array_equal(
        prediction.hotspot_C[outside].view(np.uint64),
        baseline[outside].view(np.uint64),
    )
    assert _linear_widths(fit.fit.model) == [(9, 8), (8, 1)]
    assert fit.fit.device.type == "cpu"


def test_pinn_total_loss_rejects_missing_or_reordered_components() -> None:
    values = {key: torch.tensor(1.0) for key in reversed(PINN_LOSS_KEYS)}
    with pytest.raises(ValueError, match="exactly"):
        pinn_total_loss(values)
