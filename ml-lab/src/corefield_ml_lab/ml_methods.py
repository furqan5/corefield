"""Frozen CPU-only neural methods for the CoreField falsification harness.

The models in this module implement the architectures and training defaults
frozen in ``PREREGISTRATION.md``.  They deliberately do not load, generate, or
inspect any primary-test data.  Callers supply train and validation arrays and
retain responsibility for the write-once test-access boundary.

Temperatures are in degrees Celsius (absolute) or kelvin (differences), time
is in seconds at the physics-loss boundary, and load is per-unit.
"""

from __future__ import annotations

import copy
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

# This must precede the optional dependency import.  A caller that imported
# Torch earlier still receives CPU tensors/models from every public function.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from torch import Tensor, nn


INPUT_FEATURES = 9
PLAIN_HIDDEN_WIDTH = 16
PINN_HIDDEN_WIDTH = 16
RESIDUAL_HIDDEN_WIDTH = 8
CPU_DEVICE = torch.device("cpu")

PARAMETER_NAMES: tuple[str, ...] = (
    "delta_theta_or_K",
    "tau_o_min",
    "delta_theta_hr_K",
    "tau_w_min",
)
PARAMETER_LOWER = (10.0, 30.0, 5.0, 1.0)
PARAMETER_UPPER = (90.0, 600.0, 60.0, 120.0)
NOMINAL_PARAMETERS = (45.0, 150.0, 22.0, 7.0)

# Frozen ONAF constants and fixed synthetic loss ratio.
LOSS_RATIO_R = 6.0
OIL_EXPONENT_X = 0.8
WINDING_EXPONENT_Y = 1.3
K11 = 0.5
K21 = 2.0
K22 = 2.0
MEASUREMENT_SIGMA_K = 0.5

PINN_LOSS_KEYS: tuple[str, ...] = (
    "top_oil_measurement",
    "hotspot_reference",
    "oil_ode",
    "winding_fast_ode",
    "winding_slow_ode",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Frozen optimizer and early-stopping settings.

    Tests may pass a shorter ``max_epochs`` or ``patience``.  Primary runs
    must use the defaults.
    """

    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    max_epochs: int = 2_000
    patience: int = 150
    minimum_normalized_improvement: float = 1.0e-4
    intraop_threads: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if isinstance(self.max_epochs, bool) or self.max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer")
        if isinstance(self.patience, bool) or self.patience < 1:
            raise ValueError("patience must be a positive integer")
        if (
            not math.isfinite(self.minimum_normalized_improvement)
            or self.minimum_normalized_improvement < 0.0
        ):
            raise ValueError(
                "minimum_normalized_improvement must be finite and non-negative"
            )
        if isinstance(self.intraop_threads, bool) or self.intraop_threads < 1:
            raise ValueError("intraop_threads must be a positive integer")


@dataclass(frozen=True, slots=True)
class FeatureStandardizer:
    """Train-only feature location and scale for the nine neural inputs."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train_features: np.ndarray | Sequence[Sequence[float]]) -> "FeatureStandardizer":
        values = _feature_array(train_features, name="train_features")
        mean = values.mean(axis=0, dtype=np.float64)
        scale = values.std(axis=0, ddof=0, dtype=np.float64)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(
        self, features: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        values = _feature_array(features, name="features")
        if self.mean.shape != (INPUT_FEATURES,) or self.scale.shape != (
            INPUT_FEATURES,
        ):
            raise ValueError("standardizer statistics must each have shape (9,)")
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_loss: float
    pinn_terms: tuple[tuple[str, float], ...] = ()


@dataclass(slots=True)
class SupervisedFit:
    """A fitted plain or residual network plus its train-only scaler."""

    model: nn.Module
    standardizer: FeatureStandardizer
    history: tuple[EpochRecord, ...]
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    seed: int

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict(
        self, features: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        x = _as_feature_tensor(self.standardizer.transform(features))
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)
        return prediction.detach().cpu().numpy().astype(np.float64)


@dataclass(slots=True)
class PinnFit:
    """A fitted PINN plus the five-term training trace."""

    model: "PhysicsInformedNetwork"
    standardizer: FeatureStandardizer
    history: tuple[EpochRecord, ...]
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    seed: int

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict_states(
        self, features: np.ndarray | Sequence[Sequence[float]]
    ) -> Mapping[str, np.ndarray]:
        x = _as_feature_tensor(self.standardizer.transform(features))
        self.model.eval()
        with torch.no_grad():
            states = self.model(x)
            hotspot = states[:, 0] + states[:, 1] - states[:, 2]
        values = states.detach().cpu().numpy().astype(np.float64)
        return {
            "theta_o_C": values[:, 0],
            "h1_K": values[:, 1],
            "h2_K": values[:, 2],
            "hotspot_C": hotspot.detach().cpu().numpy().astype(np.float64),
        }

    def parameter_values(self) -> Mapping[str, float]:
        return self.model.parameter_mapping()


@dataclass(frozen=True, slots=True)
class PinnTrajectory:
    """Training trajectory supplied to the discrete PINN.

    Missing measured temperatures are represented by NaN.  Physics residuals
    use every row; measurement terms use only finite rows.
    """

    features: np.ndarray
    time_s: np.ndarray
    load_pu: np.ndarray
    ambient_C: np.ndarray
    measured_top_oil_C: np.ndarray
    hotspot_reference_C: np.ndarray
    reference_features: np.ndarray | None = None
    reference_hotspot_C: np.ndarray | None = None
    top_oil_measurement_features: np.ndarray | None = None
    top_oil_measurement_values_C: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class HullPrediction:
    residual_K: np.ndarray
    extrapolation_flag: np.ndarray
    hotspot_C: np.ndarray | None = None


@dataclass(slots=True)
class HullAwareResidualFit:
    """Fitted 9-8-1 residual model and immutable scalar load hull."""

    fit: SupervisedFit
    hull_min_pu: float
    hull_max_pu: float

    def predict(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        load_pu: np.ndarray | Sequence[float],
        *,
        nls_hotspot_C: np.ndarray | Sequence[float] | None = None,
    ) -> HullPrediction:
        raw = self.fit.predict(features)
        load = _vector_array(load_pu, name="load_pu", length=raw.size)
        outside = (load < self.hull_min_pu) | (load > self.hull_max_pu)

        # Assignment, rather than multiplication by a Boolean mask, gives a
        # canonical +0.0 bit pattern even if the raw output were NaN or -0.0.
        gated = raw.copy()
        gated[outside] = 0.0

        hotspot: np.ndarray | None = None
        if nls_hotspot_C is not None:
            baseline = _vector_array(
                nls_hotspot_C, name="nls_hotspot_C", length=raw.size
            )
            # Preserve outside-hull values byte-for-byte by modifying only
            # inside-hull indices on a copy of the baseline.
            hotspot = baseline.copy()
            inside = ~outside
            hotspot[inside] = hotspot[inside] + gated[inside]
        return HullPrediction(
            residual_K=gated,
            extrapolation_flag=outside,
            hotspot_C=hotspot,
        )


class PlainHotspotNetwork(nn.Module):
    """Fixed fully connected 9 -> 16 -> 16 -> 1 tanh network."""

    layer_widths = (INPUT_FEATURES, 16, 16, 1)

    def __init__(self, hidden_width: int = PLAIN_HIDDEN_WIDTH) -> None:
        super().__init__()
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int) or hidden_width < 1:
            raise ValueError("hidden_width must be a positive integer")
        self.hidden_width = hidden_width
        self.layer_widths = (INPUT_FEATURES, hidden_width, hidden_width, 1)
        self.network = nn.Sequential(
            nn.Linear(INPUT_FEATURES, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        _check_tensor_features(features)
        return self.network(features).squeeze(-1)


class ResidualNetwork(nn.Module):
    """Fixed fully connected 9 -> 8 -> 1 tanh residual network."""

    layer_widths = (INPUT_FEATURES, 8, 1)

    def __init__(self, hidden_width: int = RESIDUAL_HIDDEN_WIDTH) -> None:
        super().__init__()
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int) or hidden_width < 1:
            raise ValueError("hidden_width must be a positive integer")
        self.hidden_width = hidden_width
        self.layer_widths = (INPUT_FEATURES, hidden_width, 1)
        self.network = nn.Sequential(
            nn.Linear(INPUT_FEATURES, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        _check_tensor_features(features)
        return self.network(features).squeeze(-1)


class PhysicsInformedNetwork(nn.Module):
    """Fixed 9 -> 16 -> 16 -> 3 state network with bounded parameters."""

    layer_widths = (INPUT_FEATURES, 16, 16, 3)

    def __init__(self, hidden_width: int = PINN_HIDDEN_WIDTH) -> None:
        super().__init__()
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int) or hidden_width < 1:
            raise ValueError("hidden_width must be a positive integer")
        self.hidden_width = hidden_width
        self.layer_widths = (INPUT_FEATURES, hidden_width, hidden_width, 3)
        self.network = nn.Sequential(
            nn.Linear(INPUT_FEATURES, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 3),
        )
        lower = torch.tensor(PARAMETER_LOWER, dtype=torch.float32)
        upper = torch.tensor(PARAMETER_UPPER, dtype=torch.float32)
        nominal = torch.tensor(NOMINAL_PARAMETERS, dtype=torch.float32)
        fraction = (nominal - lower) / (upper - lower)
        initial_raw = torch.logit(fraction)
        self.raw_parameters = nn.Parameter(initial_raw)
        self.register_buffer("parameter_lower", lower)
        self.register_buffer("parameter_upper", upper)

    def forward(self, features: Tensor) -> Tensor:
        _check_tensor_features(features)
        return self.network(features)

    def bounded_parameters(self) -> Tensor:
        """Return [dtheta_or K, tau_o min, dtheta_hr K, tau_w min]."""

        return self.parameter_lower + (
            self.parameter_upper - self.parameter_lower
        ) * torch.sigmoid(self.raw_parameters)

    def parameter_mapping(self) -> Mapping[str, float]:
        values = self.bounded_parameters().detach().cpu().tolist()
        return dict(zip(PARAMETER_NAMES, (float(value) for value in values)))


def _feature_array(
    values: np.ndarray | Sequence[Sequence[float]], *, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != INPUT_FEATURES:
        raise ValueError(f"{name} must have shape (n, 9), got {array.shape}")
    if array.shape[0] == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _vector_array(
    values: np.ndarray | Sequence[float], *, name: str, length: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if length is not None and array.size != length:
        raise ValueError(f"{name} length {array.size} does not match {length}")
    return array


def _as_feature_tensor(features: np.ndarray) -> Tensor:
    return torch.as_tensor(features, dtype=torch.float32, device=CPU_DEVICE)


def _check_tensor_features(features: Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != INPUT_FEATURES:
        raise ValueError(f"features must have shape (n, 9), got {tuple(features.shape)}")
    if features.device.type != "cpu":
        raise RuntimeError("CPU-only execution is required")


def _configure_determinism(seed: int, intraop_threads: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be in the inclusive range 0..2**32-1")
    torch.set_num_threads(intraop_threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _finite_label_tensor(
    values: np.ndarray | Sequence[float], *, name: str, length: int
) -> tuple[Tensor, Tensor]:
    array = _vector_array(values, name=name, length=length)
    mask = np.isfinite(array)
    if not np.any(mask):
        raise ValueError(f"{name} contains no finite labels")
    tensor = torch.as_tensor(array[mask], dtype=torch.float32, device=CPU_DEVICE)
    index = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=CPU_DEVICE)
    return tensor, index


def _normalized_improvement(best: float, candidate: float) -> float:
    if not math.isfinite(best):
        return math.inf
    return (best - candidate) / max(abs(best), torch.finfo(torch.float64).eps)


def _supervised_fit(
    model: nn.Module,
    train_features: np.ndarray | Sequence[Sequence[float]],
    train_targets: np.ndarray | Sequence[float],
    validation_features: np.ndarray | Sequence[Sequence[float]],
    validation_targets: np.ndarray | Sequence[float],
    *,
    seed: int,
    config: TrainingConfig,
    standardizer_features: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> SupervisedFit:
    _configure_determinism(seed, config.intraop_threads)
    scaler = FeatureStandardizer.fit(
        train_features if standardizer_features is None else standardizer_features
    )
    x_train_np = scaler.transform(train_features)
    x_validation_np = scaler.transform(validation_features)
    x_train = _as_feature_tensor(x_train_np)
    x_validation = _as_feature_tensor(x_validation_np)
    y_train, train_index = _finite_label_tensor(
        train_targets, name="train_targets", length=x_train.shape[0]
    )
    y_validation, validation_index = _finite_label_tensor(
        validation_targets,
        name="validation_targets",
        length=x_validation.shape[0],
    )

    model.to(device=CPU_DEVICE, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    records: list[EpochRecord] = []

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)[train_index]
        train_loss = torch.mean((prediction - y_train) ** 2)
        train_loss.backward()
        optimizer.step()

        # Validation is checkpoint selection only: no gradient graph is built,
        # and no validation value enters the optimizer objective above.
        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation)[validation_index]
            validation_loss = torch.mean(
                (validation_prediction - y_validation) ** 2
            )
        train_value = float(train_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        if not math.isfinite(train_value) or not math.isfinite(validation_value):
            raise FloatingPointError("non-finite neural training loss")
        records.append(EpochRecord(epoch, train_value, validation_value))

        if (
            _normalized_improvement(best_loss, validation_value)
            >= config.minimum_normalized_improvement
        ):
            best_loss = validation_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return SupervisedFit(
        model=model,
        standardizer=scaler,
        history=tuple(records),
        best_epoch=best_epoch,
        epochs_ran=len(records),
        stopped_early=len(records) < config.max_epochs,
        seed=seed,
    )


def fit_plain_nn(
    train_features: np.ndarray | Sequence[Sequence[float]],
    train_hotspot_C: np.ndarray | Sequence[float],
    validation_features: np.ndarray | Sequence[Sequence[float]],
    validation_hotspot_C: np.ndarray | Sequence[float],
    *,
    seed: int,
    config: TrainingConfig | None = None,
    hidden_width: int = PLAIN_HIDDEN_WIDTH,
    standardizer_features: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> SupervisedFit:
    """Fit the fixed plain network using finite hot-spot references only."""

    actual_config = TrainingConfig() if config is None else config
    _configure_determinism(seed, actual_config.intraop_threads)
    model = PlainHotspotNetwork(hidden_width=hidden_width)
    return _supervised_fit(
        model,
        train_features,
        train_hotspot_C,
        validation_features,
        validation_hotspot_C,
        seed=seed,
        config=actual_config,
        standardizer_features=standardizer_features,
    )


def fit_residual_nn(
    train_features: np.ndarray | Sequence[Sequence[float]],
    train_reference_minus_nls_K: np.ndarray | Sequence[float],
    validation_features: np.ndarray | Sequence[Sequence[float]],
    validation_reference_minus_nls_K: np.ndarray | Sequence[float],
    *,
    train_load_pu: np.ndarray | Sequence[float],
    seed: int,
    config: TrainingConfig | None = None,
    hidden_width: int = RESIDUAL_HIDDEN_WIDTH,
    standardizer_features: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> HullAwareResidualFit:
    """Fit the fixed residual net and freeze the scalar training-load hull."""

    train_x = _feature_array(train_features, name="train_features")
    load = _vector_array(train_load_pu, name="train_load_pu")
    if np.any(~np.isfinite(load)):
        raise ValueError("train_load_pu must be finite")
    actual_config = TrainingConfig() if config is None else config
    _configure_determinism(seed, actual_config.intraop_threads)
    model = ResidualNetwork(hidden_width=hidden_width)
    fit = _supervised_fit(
        model,
        train_x,
        train_reference_minus_nls_K,
        validation_features,
        validation_reference_minus_nls_K,
        seed=seed,
        config=actual_config,
        standardizer_features=standardizer_features,
    )
    return HullAwareResidualFit(
        fit=fit,
        hull_min_pu=float(np.min(load)),
        hull_max_pu=float(np.max(load)),
    )


def _validated_pinn_trajectory(
    trajectory: PinnTrajectory,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    features = _feature_array(trajectory.features, name="trajectory.features")
    n_rows = features.shape[0]
    if n_rows < 3:
        raise ValueError("PINN trajectory needs at least three rows for centered differences")
    time_s = _vector_array(trajectory.time_s, name="trajectory.time_s", length=n_rows)
    load = _vector_array(trajectory.load_pu, name="trajectory.load_pu", length=n_rows)
    ambient = _vector_array(
        trajectory.ambient_C, name="trajectory.ambient_C", length=n_rows
    )
    top_oil = _vector_array(
        trajectory.measured_top_oil_C,
        name="trajectory.measured_top_oil_C",
        length=n_rows,
    )
    hotspot = _vector_array(
        trajectory.hotspot_reference_C,
        name="trajectory.hotspot_reference_C",
        length=n_rows,
    )
    if np.any(~np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("trajectory.time_s must be finite and strictly increasing")
    if np.any(~np.isfinite(load)) or np.any(load < 0.0):
        raise ValueError("trajectory.load_pu must be finite and non-negative")
    if np.any(~np.isfinite(ambient)):
        raise ValueError("trajectory.ambient_C must be finite")
    separate_features = trajectory.reference_features
    separate_hotspot = trajectory.reference_hotspot_C
    if (separate_features is None) != (separate_hotspot is None):
        raise ValueError(
            "trajectory.reference_features and reference_hotspot_C must be supplied together"
        )
    reference_features: np.ndarray | None = None
    reference_hotspot: np.ndarray | None = None
    if separate_features is not None and separate_hotspot is not None:
        reference_features = _feature_array(
            separate_features, name="trajectory.reference_features"
        )
        reference_hotspot = _vector_array(
            separate_hotspot,
            name="trajectory.reference_hotspot_C",
            length=reference_features.shape[0],
        )
        if not np.any(np.isfinite(reference_hotspot)):
            raise ValueError("trajectory.reference_hotspot_C contains no finite values")
    elif not np.any(np.isfinite(hotspot)):
        raise ValueError("trajectory.hotspot_reference_C contains no finite values")
    separate_oil_features = trajectory.top_oil_measurement_features
    separate_oil_values = trajectory.top_oil_measurement_values_C
    if (separate_oil_features is None) != (separate_oil_values is None):
        raise ValueError(
            "top_oil_measurement_features and values must be supplied together"
        )
    oil_features: np.ndarray | None = None
    oil_values: np.ndarray | None = None
    if separate_oil_features is not None and separate_oil_values is not None:
        oil_features = _feature_array(
            separate_oil_features,
            name="trajectory.top_oil_measurement_features",
        )
        oil_values = _vector_array(
            separate_oil_values,
            name="trajectory.top_oil_measurement_values_C",
            length=oil_features.shape[0],
        )
        if not np.all(np.isfinite(oil_values)):
            raise ValueError("top_oil_measurement_values_C must be finite")
    elif not np.any(np.isfinite(top_oil)):
        raise ValueError("trajectory.measured_top_oil_C contains no finite values")
    return (
        features,
        time_s,
        load,
        ambient,
        top_oil,
        hotspot,
        reference_features,
        reference_hotspot,
        oil_features,
        oil_values,
    )


def centered_finite_difference(values: Tensor, time_s: Tensor) -> Tensor:
    """Centered derivative at rows 1..n-2 on a strictly increasing grid."""

    if values.ndim != 1 or time_s.ndim != 1 or values.shape != time_s.shape:
        raise ValueError("values and time_s must be one-dimensional with equal shape")
    if values.numel() < 3:
        raise ValueError("at least three samples are required")
    intervals = time_s[2:] - time_s[:-2]
    if bool(torch.any(~torch.isfinite(intervals))) or bool(torch.any(intervals <= 0.0)):
        raise ValueError("time_s must be finite and strictly increasing")
    return (values[2:] - values[:-2]) / intervals


def _masked_dimensionless_mse(
    prediction: Tensor, observation: Tensor, scale_K: float
) -> Tensor:
    mask = torch.isfinite(observation)
    if not bool(torch.any(mask)):
        raise ValueError("measurement term contains no finite observations")
    error = (prediction[mask] - observation[mask]) / scale_K
    return torch.mean(error**2)


def pinn_loss_terms(
    model: PhysicsInformedNetwork,
    standardized_features: Tensor,
    time_s: Tensor,
    load_pu: Tensor,
    ambient_C: Tensor,
    measured_top_oil_C: Tensor,
    hotspot_reference_C: Tensor,
    standardized_reference_features: Tensor | None = None,
    reference_hotspot_C: Tensor | None = None,
    standardized_top_oil_features: Tensor | None = None,
    top_oil_measurement_values_C: Tensor | None = None,
) -> "OrderedDict[str, Tensor]":
    """Return exactly five dimensionless PINN loss terms.

    The three ODE terms use centered finite differences at interior rows.
    Each also prepends its opening-equilibrium residual, so the initial state
    is constrained without creating a sixth loss term.  Physics arithmetic is
    float64 while neural weights and outputs remain float32.
    """

    _check_tensor_features(standardized_features)
    n_rows = standardized_features.shape[0]
    vectors = (time_s, load_pu, ambient_C, measured_top_oil_C, hotspot_reference_C)
    if any(vector.ndim != 1 or vector.shape[0] != n_rows for vector in vectors):
        raise ValueError("all PINN trajectory vectors must have shape (n,)")
    if n_rows < 3:
        raise ValueError("PINN trajectory needs at least three rows")

    states32 = model(standardized_features)
    hotspot32 = states32[:, 0] + states32[:, 1] - states32[:, 2]
    if (standardized_top_oil_features is None) != (
        top_oil_measurement_values_C is None
    ):
        raise ValueError(
            "standardized_top_oil_features and measurement values must be supplied together"
        )
    if standardized_top_oil_features is None:
        top_oil_term = _masked_dimensionless_mse(
            states32[:, 0],
            measured_top_oil_C.to(torch.float32),
            MEASUREMENT_SIGMA_K,
        )
    else:
        _check_tensor_features(standardized_top_oil_features)
        oil_prediction = model(standardized_top_oil_features)[:, 0]
        top_oil_term = _masked_dimensionless_mse(
            oil_prediction,
            top_oil_measurement_values_C.to(torch.float32),
            MEASUREMENT_SIGMA_K,
        )
    if (standardized_reference_features is None) != (reference_hotspot_C is None):
        raise ValueError(
            "standardized_reference_features and reference_hotspot_C must be supplied together"
        )
    if standardized_reference_features is None:
        hotspot_term = _masked_dimensionless_mse(
            hotspot32, hotspot_reference_C.to(torch.float32), MEASUREMENT_SIGMA_K
        )
    else:
        _check_tensor_features(standardized_reference_features)
        reference_states = model(standardized_reference_features)
        reference_prediction = (
            reference_states[:, 0] + reference_states[:, 1] - reference_states[:, 2]
        )
        hotspot_term = _masked_dimensionless_mse(
            reference_prediction,
            reference_hotspot_C.to(torch.float32),
            MEASUREMENT_SIGMA_K,
        )

    states = states32.to(torch.float64)
    time = time_s.to(torch.float64)
    load = load_pu.to(torch.float64)
    ambient = ambient_C.to(torch.float64)
    if bool(torch.any(~torch.isfinite(time))) or bool(torch.any(torch.diff(time) <= 0.0)):
        raise ValueError("time_s must be finite and strictly increasing")
    if bool(torch.any(~torch.isfinite(load))) or bool(torch.any(load < 0.0)):
        raise ValueError("load_pu must be finite and non-negative")
    if bool(torch.any(~torch.isfinite(ambient))):
        raise ValueError("ambient_C must be finite")

    params = model.bounded_parameters().to(torch.float64)
    dtheta_or, tau_o_min, dtheta_hr, tau_w_min = params.unbind()
    tau_o_s = tau_o_min * 60.0
    tau_w_s = tau_w_min * 60.0
    theta_o, h1, h2 = states.unbind(dim=1)

    loss_factor = ((1.0 + LOSS_RATIO_R * load**2) / (1.0 + LOSS_RATIO_R)) ** OIL_EXPONENT_X
    winding_drive_factor = load**WINDING_EXPONENT_Y

    dtheta_o_dt = centered_finite_difference(theta_o, time)
    dh1_dt = centered_finite_difference(h1, time)
    dh2_dt = centered_finite_difference(h2, time)

    oil_dynamic = (
        K11 * tau_o_s * dtheta_o_dt
        - (dtheta_or * loss_factor[1:-1] + ambient[1:-1] - theta_o[1:-1])
    ) / dtheta_or
    h1_dynamic = (
        K22 * tau_w_s * dh1_dt
        - (K21 * dtheta_hr * winding_drive_factor[1:-1] - h1[1:-1])
    ) / dtheta_hr
    h2_dynamic = (
        (tau_o_s / K22) * dh2_dt
        - ((K21 - 1.0) * dtheta_hr * winding_drive_factor[1:-1] - h2[1:-1])
    ) / dtheta_hr

    # Opening equilibrium belongs to its corresponding state-equation term.
    oil_opening = (
        theta_o[0] - (ambient[0] + dtheta_or * loss_factor[0])
    ) / dtheta_or
    h1_opening = (
        h1[0] - K21 * dtheta_hr * winding_drive_factor[0]
    ) / dtheta_hr
    h2_opening = (
        h2[0] - (K21 - 1.0) * dtheta_hr * winding_drive_factor[0]
    ) / dtheta_hr

    terms: "OrderedDict[str, Tensor]" = OrderedDict()
    terms[PINN_LOSS_KEYS[0]] = top_oil_term.to(torch.float64)
    terms[PINN_LOSS_KEYS[1]] = hotspot_term.to(torch.float64)
    terms[PINN_LOSS_KEYS[2]] = torch.mean(
        torch.cat((oil_opening.reshape(1), oil_dynamic)) ** 2
    )
    terms[PINN_LOSS_KEYS[3]] = torch.mean(
        torch.cat((h1_opening.reshape(1), h1_dynamic)) ** 2
    )
    terms[PINN_LOSS_KEYS[4]] = torch.mean(
        torch.cat((h2_opening.reshape(1), h2_dynamic)) ** 2
    )
    return terms


def pinn_total_loss(terms: Mapping[str, Tensor]) -> Tensor:
    """Return the fixed unweighted mean of the five named PINN terms."""

    if tuple(terms.keys()) != PINN_LOSS_KEYS:
        raise ValueError(f"PINN terms must be exactly {PINN_LOSS_KEYS}")
    return torch.stack(tuple(terms.values())).mean()


def fit_pinn(
    train_trajectory: PinnTrajectory,
    validation_features: np.ndarray | Sequence[Sequence[float]],
    validation_hotspot_C: np.ndarray | Sequence[float],
    *,
    seed: int,
    config: TrainingConfig | None = None,
    hidden_width: int = PINN_HIDDEN_WIDTH,
) -> PinnFit:
    """Fit the discrete PINN on train physics/data and validation labels only."""

    actual_config = TrainingConfig() if config is None else config
    _configure_determinism(seed, actual_config.intraop_threads)
    (
        features,
        time,
        load,
        ambient,
        top_oil,
        hotspot,
        reference_features,
        reference_hotspot,
        oil_features,
        oil_values,
    ) = _validated_pinn_trajectory(train_trajectory)
    scaler = FeatureStandardizer.fit(features)
    x_train = _as_feature_tensor(scaler.transform(features))
    x_validation = _as_feature_tensor(scaler.transform(validation_features))
    y_validation, validation_index = _finite_label_tensor(
        validation_hotspot_C,
        name="validation_hotspot_C",
        length=x_validation.shape[0],
    )
    # Synthetic records are intentionally immutable NumPy arrays.  Copy into
    # Torch-owned storage to avoid undefined behaviour warnings from tensors
    # that alias non-writable arrays.
    tensor = lambda values: torch.tensor(values, dtype=torch.float64, device=CPU_DEVICE)
    time_t = tensor(time)
    load_t = tensor(load)
    ambient_t = tensor(ambient)
    top_oil_t = tensor(top_oil)
    hotspot_t = tensor(hotspot)

    if reference_features is None:
        x_reference = None
        reference_hotspot_t = None
    else:
        x_reference = _as_feature_tensor(scaler.transform(reference_features))
        assert reference_hotspot is not None
        reference_hotspot_t = tensor(reference_hotspot)

    if oil_features is None:
        x_oil_measurements = None
        oil_measurement_values_t = None
    else:
        x_oil_measurements = _as_feature_tensor(scaler.transform(oil_features))
        assert oil_values is not None
        oil_measurement_values_t = tensor(oil_values)

    model = PhysicsInformedNetwork(hidden_width=hidden_width).to(
        device=CPU_DEVICE, dtype=torch.float32
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=actual_config.learning_rate,
        weight_decay=actual_config.weight_decay,
    )
    best_loss = math.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    records: list[EpochRecord] = []

    for epoch in range(actual_config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        terms = pinn_loss_terms(
            model,
            x_train,
            time_t,
            load_t,
            ambient_t,
            top_oil_t,
            hotspot_t,
            x_reference,
            reference_hotspot_t,
            x_oil_measurements,
            oil_measurement_values_t,
        )
        train_loss = pinn_total_loss(terms)
        train_loss.backward()
        optimizer.step()

        # Validation references select an epoch only; validation physics and
        # validation top-oil measurements never enter this objective.
        model.eval()
        with torch.no_grad():
            states = model(x_validation)
            predicted_hotspot = states[:, 0] + states[:, 1] - states[:, 2]
            validation_loss = torch.mean(
                (predicted_hotspot[validation_index] - y_validation) ** 2
            )
        train_value = float(train_loss.detach().cpu())
        validation_value = float(validation_loss.detach().cpu())
        if not math.isfinite(train_value) or not math.isfinite(validation_value):
            raise FloatingPointError("non-finite PINN training loss")
        term_record = tuple(
            (name, float(value.detach().cpu())) for name, value in terms.items()
        )
        records.append(
            EpochRecord(epoch, train_value, validation_value, term_record)
        )

        if (
            _normalized_improvement(best_loss, validation_value)
            >= actual_config.minimum_normalized_improvement
        ):
            best_loss = validation_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= actual_config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return PinnFit(
        model=model,
        standardizer=scaler,
        history=tuple(records),
        best_epoch=best_epoch,
        epochs_ran=len(records),
        stopped_early=len(records) < actual_config.max_epochs,
        seed=seed,
    )


__all__ = [
    "CPU_DEVICE",
    "FeatureStandardizer",
    "HullAwareResidualFit",
    "HullPrediction",
    "INPUT_FEATURES",
    "NOMINAL_PARAMETERS",
    "PARAMETER_LOWER",
    "PARAMETER_NAMES",
    "PARAMETER_UPPER",
    "PINN_LOSS_KEYS",
    "PinnFit",
    "PinnTrajectory",
    "PhysicsInformedNetwork",
    "PlainHotspotNetwork",
    "ResidualNetwork",
    "SupervisedFit",
    "TrainingConfig",
    "centered_finite_difference",
    "fit_pinn",
    "fit_plain_nn",
    "fit_residual_nn",
    "pinn_loss_terms",
    "pinn_total_loss",
]
