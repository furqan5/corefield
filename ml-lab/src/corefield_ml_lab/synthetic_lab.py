"""Deterministic synthetic records shared by the mechanism experiments.

This module is an adapter around the frozen, vendored CoreField equations.  It
defines signals, observations, reference placement, and neural features; it
does not contain a second thermal-model implementation.  Absolute
temperatures are in degrees Celsius, temperature differences are in kelvin,
time is in seconds unless a name says otherwise, and load is dimensionless
per unit.

The schedules below are engineering stress-test settings frozen for this lab,
not representations of an installed transformer.  Test outcomes must always
be scored against ``TruthRecord.hotspot_C`` rather than noisy references.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from pathlib import Path
import sys
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray


def _load_vendored_physics():
    """Load this checkout's frozen CoreField physics, never a shadow copy."""

    vendor_root = Path(__file__).resolve().parents[2] / "vendor"
    expected_package = (vendor_root / "corefield" / "__init__.py").resolve()
    expected_module = (vendor_root / "corefield" / "iec60076_7.py").resolve()
    if not expected_module.is_file():
        raise RuntimeError(f"frozen CoreField physics is missing: {expected_module}")

    loaded_package = sys.modules.get("corefield")
    if loaded_package is not None:
        loaded_file = getattr(loaded_package, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != expected_package:
            raise RuntimeError(
                "a non-vendored 'corefield' package was imported before synthetic_lab; "
                "restart with this checkout's vendor directory first on sys.path"
            )

    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    module = import_module("corefield.iec60076_7")
    resolved = Path(module.__file__).resolve()
    if resolved != expected_module:
        raise RuntimeError(
            f"synthetic truth resolved to {resolved}, expected frozen file {expected_module}"
        )
    return module


_physics = _load_vendored_physics()
CoolingConstants = _physics.CoolingConstants
ThermalParams = _physics.ThermalParams


TRUTH_DT_S: float = 30.0
MODEL_DT_S: float = 120.0
TOP_OIL_SAMPLE_S: float = 300.0
SENSOR_NOISE_SIGMA_K: float = 0.5
AMBIENT_MEAN_C: float = 20.0
AMBIENT_AMPLITUDE_K: float = 6.0
RAMP_HALF_WIDTH_H: float = 0.025  # 90 s; the frozen CoreField convention.

REFERENCE_BUDGETS: tuple[int, ...] = (3, 4, 6, 10, 20, 50)
REFERENCE_OFFSETS_MIN: tuple[float, ...] = (3.0, 8.0, 18.0, 48.0)
REFERENCE_SETTLING_MIN: float = 120.0
REFERENCE_EVENT_CLEARANCE_MIN: float = 30.0

FEATURE_NAMES: tuple[str, ...] = (
    "load_current_pu",
    "load_lag_6min_pu",
    "load_lag_16min_pu",
    "load_lag_60min_pu",
    "load_lag_180min_pu",
    "ambient_current_C",
    "top_oil_current_C",
    "top_oil_lag_16min_C",
    "top_oil_lag_60min_C",
)
FEATURE_SOURCE_LAGS_MIN: tuple[float, ...] = (
    0.0,
    6.0,
    16.0,
    60.0,
    180.0,
    0.0,
    0.0,
    16.0,
    60.0,
)
MAX_FEATURE_LAG_MIN: float = 180.0

NOMINAL_PARAMS = ThermalParams(
    delta_theta_or_K=45.0,
    tau_o_min=150.0,
    delta_theta_hr_K=22.0,
    tau_w_min=7.0,
    loss_ratio_R=6.0,
)
MATCHED_CONSTANTS = CoolingConstants(
    x=0.8,
    y=1.3,
    k11=0.5,
    k21=2.0,
    k22=2.0,
    name="ONAF mechanism-test matched truth",
    x1=0.0,
    y1=0.0,
)
STRUCTURAL_MISMATCH_CONSTANTS = CoolingConstants(
    x=0.8,
    y=1.3,
    k11=0.5,
    k21=2.0,
    k22=2.0,
    name="ONAF mechanism-test structural-mismatch truth",
    x1=0.21,
    y1=0.0,
)

SplitName = Literal["train", "validation", "in_range_test", "e3_test"]
PhysicsMode = Literal["matched", "structural_mismatch"]


def _readonly(values: NDArray[np.floating] | Sequence[float]) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _readonly_index(values: NDArray[np.integer] | Sequence[int]) -> NDArray[np.intp]:
    array = np.array(values, dtype=np.intp, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Schedule:
    """One immutable driving schedule on the 30 s truth grid."""

    name: str
    split: SplitName
    time_s: NDArray[np.float64]
    load_pu: NDArray[np.float64]
    ambient_C: NDArray[np.float64]
    load_pu_half: NDArray[np.float64]
    ambient_C_half: NDArray[np.float64]
    event_time_s: NDArray[np.float64]
    target_load_pu: float | None = None

    def __post_init__(self) -> None:
        shape = self.time_s.shape
        if self.time_s.ndim != 1 or self.time_s.size < 2:
            raise ValueError("schedule time_s must be a one-dimensional grid")
        for name in ("load_pu", "ambient_C", "load_pu_half", "ambient_C_half"):
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(f"{name} shape {values.shape} != time_s shape {shape}")
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        if not np.allclose(np.diff(self.time_s), TRUTH_DT_S, rtol=0.0, atol=1e-12):
            raise ValueError(f"truth schedule must use exactly {TRUTH_DT_S:g} s steps")
        if np.any(self.load_pu < 0.0) or np.any(self.load_pu_half < 0.0):
            raise ValueError("load_pu cannot be negative")
        if np.any(np.diff(self.event_time_s) <= 0.0):
            raise ValueError("event_time_s must be strictly increasing")
        if np.any(self.event_time_s <= self.time_s[0]) or np.any(
            self.event_time_s >= self.time_s[-1]
        ):
            raise ValueError("events must lie strictly inside the schedule")

    @property
    def duration_h(self) -> float:
        """Record duration [h]."""

        return float((self.time_s[-1] - self.time_s[0]) / 3600.0)

    @property
    def load_hull_pu(self) -> tuple[float, float]:
        """Observed scalar load hull [pu]."""

        return float(np.min(self.load_pu)), float(np.max(self.load_pu))


@dataclass(frozen=True)
class TruthRecord:
    """Noise-free hidden truth produced by vendored CoreField RK4."""

    schedule: Schedule
    physics_mode: PhysicsMode
    top_oil_C: NDArray[np.float64]
    hotspot_C: NDArray[np.float64]
    gradient_K: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected = self.schedule.time_s.shape
        for name in ("top_oil_C", "hotspot_C", "gradient_K"):
            values = getattr(self, name)
            if values.shape != expected or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {expected}")
        if not np.allclose(
            self.gradient_K, self.hotspot_C - self.top_oil_C, rtol=0.0, atol=1e-10
        ):
            raise ValueError("gradient_K must equal hotspot_C - top_oil_C")

    @property
    def split(self) -> SplitName:
        return self.schedule.split


@dataclass(frozen=True)
class ObservedRecord:
    """Five-minute noisy top-oil observations for one hidden truth record."""

    truth: TruthRecord
    seed: int
    top_oil_index: NDArray[np.intp]
    top_oil_time_s: NDArray[np.float64]
    top_oil_C: NDArray[np.float64]

    @property
    def split(self) -> SplitName:
        return self.truth.split


@dataclass(frozen=True)
class SparseHotspotReferences:
    """Nested noisy calibration references; hidden truth is retained for audit."""

    split: SplitName
    seed: int
    budget: int
    index: NDArray[np.intp]
    time_s: NDArray[np.float64]
    temperature_C: NDArray[np.float64]
    truth_temperature_C: NDArray[np.float64]


@dataclass(frozen=True)
class FeatureFrame:
    """Nine preregistered inputs and hidden noise-free targets on a 2 min grid."""

    split: SplitName
    time_s: NDArray[np.float64]
    truth_index: NDArray[np.intp]
    X: NDArray[np.float64]
    hotspot_truth_C: NDArray[np.float64]
    source_time_s: NDArray[np.float64]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        n = self.time_s.size
        if self.X.shape != (n, len(FEATURE_NAMES)):
            raise ValueError(f"X must have shape ({n}, {len(FEATURE_NAMES)})")
        if self.source_time_s.shape != self.X.shape:
            raise ValueError("source_time_s must have the same shape as X")
        if self.truth_index.shape != (n,) or self.hotspot_truth_C.shape != (n,):
            raise ValueError("truth_index and hotspot_truth_C must align with feature rows")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("feature order differs from the frozen nine-feature protocol")
        if np.any(self.source_time_s > self.time_s[:, None] + 1e-12):
            raise ValueError("a feature is labelled with a source time after its row time")


@dataclass(frozen=True)
class TrainStandardizer:
    """Feature standardization fitted exclusively on the train split."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    fitted_split: Literal["train"] = "train"

    def transform(self, frame_or_array: FeatureFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply frozen train statistics without inspecting another split."""

        if isinstance(frame_or_array, FeatureFrame):
            if frame_or_array.feature_names != self.feature_names:
                raise ValueError("feature names/order do not match the fitted standardizer")
            values = frame_or_array.X
        else:
            values = np.asarray(frame_or_array, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have shape (n, {len(self.feature_names)}), got {values.shape}"
            )
        return np.asarray((values - self.mean) / self.scale, dtype=np.float64)


def _time_grid(duration_h: float) -> NDArray[np.float64]:
    n_steps = int(round(duration_h * 3600.0 / TRUTH_DT_S))
    if not np.isclose(n_steps * TRUTH_DT_S, duration_h * 3600.0):
        raise ValueError("duration must be an integer multiple of the 30 s truth step")
    return np.arange(n_steps + 1, dtype=np.float64) * TRUTH_DT_S


def _diurnal_ambient(time_s: NDArray[np.float64]) -> NDArray[np.float64]:
    time_h = np.asarray(time_s, dtype=np.float64) / 3600.0
    return AMBIENT_MEAN_C + AMBIENT_AMPLITUDE_K * np.sin(
        2.0 * np.pi * (time_h - 9.0) / 24.0
    )


def _smoothed_levels(
    time_s: NDArray[np.float64],
    initial: float,
    event_h: Sequence[float],
    post_event_levels: Sequence[float],
) -> NDArray[np.float64]:
    if len(event_h) != len(post_event_levels):
        raise ValueError("every event requires one post-event level")
    time_h = np.asarray(time_s, dtype=np.float64) / 3600.0
    result = np.full(time_h.shape, float(initial), dtype=np.float64)
    previous = float(initial)
    for event, level in zip(event_h, post_event_levels, strict=True):
        change = float(level) - previous
        result += 0.5 * change * (
            1.0 + np.tanh((time_h - float(event)) / RAMP_HALF_WIDTH_H)
        )
        previous = float(level)
    return result


def _schedule_from_function(
    *,
    name: str,
    split: SplitName,
    duration_h: float,
    event_h: Sequence[float],
    load_function,
    target_load_pu: float | None = None,
) -> Schedule:
    time_s = _time_grid(duration_h)
    half_time_s = time_s + 0.5 * TRUTH_DT_S
    return Schedule(
        name=name,
        split=split,
        time_s=_readonly(time_s),
        load_pu=_readonly(load_function(time_s)),
        ambient_C=_readonly(_diurnal_ambient(time_s)),
        load_pu_half=_readonly(load_function(half_time_s)),
        ambient_C_half=_readonly(_diurnal_ambient(half_time_s)),
        event_time_s=_readonly(np.asarray(event_h, dtype=np.float64) * 3600.0),
        target_load_pu=target_load_pu,
    )


def make_train_schedule() -> Schedule:
    """Frozen 48 h fit record, confined to 0.60--0.95 pu."""

    events = (6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0)
    levels = (0.95, 0.65, 0.90, 0.70, 0.93, 0.62, 0.88)
    load = lambda t: _smoothed_levels(t, 0.60, events, levels)
    return _schedule_from_function(
        name="train_48h", split="train", duration_h=48.0, event_h=events, load_function=load
    )


def make_validation_schedule() -> Schedule:
    """Frozen, distinct 24 h early-stopping record, 0.65--0.92 pu."""

    events = (4.5, 9.5, 14.0, 19.5)
    levels = (0.90, 0.65, 0.92, 0.70)
    load = lambda t: _smoothed_levels(t, 0.68, events, levels)
    return _schedule_from_function(
        name="validation_24h",
        split="validation",
        duration_h=24.0,
        event_h=events,
        load_function=load,
    )


def make_in_range_test_schedule() -> Schedule:
    """Frozen 24 h hidden test record, confined to 0.62--0.94 pu."""

    events = (3.0, 8.0, 13.5, 18.0, 21.5)
    levels = (0.94, 0.62, 0.86, 0.68, 0.91)
    load = lambda t: _smoothed_levels(t, 0.72, events, levels)
    return _schedule_from_function(
        name="in_range_test_24h",
        split="in_range_test",
        duration_h=24.0,
        event_h=events,
        load_function=load,
    )


def make_e3_schedule(target_load_pu: float) -> Schedule:
    """Eight-hour E3 episode: 4 h at 0.75 pu, then 4 h at target."""

    target = float(target_load_pu)
    allowed = (1.00, 1.15, 1.30, 1.60)
    if target not in allowed:
        raise ValueError(f"E3 target must be one of {allowed}, got {target_load_pu!r}")

    def load(time_s: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(np.asarray(time_s) < 4.0 * 3600.0, 0.75, target)

    return _schedule_from_function(
        name=f"e3_{target:.2f}pu",
        split="e3_test",
        duration_h=8.0,
        event_h=(4.0,),
        load_function=load,
        target_load_pu=target,
    )


def make_schedule(split: SplitName, *, e3_target_load_pu: float | None = None) -> Schedule:
    """Dispatch to one of the four frozen schedule constructors."""

    if split == "train":
        if e3_target_load_pu is not None:
            raise ValueError("e3_target_load_pu is only valid for split='e3_test'")
        return make_train_schedule()
    if split == "validation":
        if e3_target_load_pu is not None:
            raise ValueError("e3_target_load_pu is only valid for split='e3_test'")
        return make_validation_schedule()
    if split == "in_range_test":
        if e3_target_load_pu is not None:
            raise ValueError("e3_target_load_pu is only valid for split='e3_test'")
        return make_in_range_test_schedule()
    if split == "e3_test":
        if e3_target_load_pu is None:
            raise ValueError("split='e3_test' requires e3_target_load_pu")
        return make_e3_schedule(e3_target_load_pu)
    raise ValueError(f"unknown split {split!r}")


def physics_constants(mode: PhysicsMode) -> CoolingConstants:
    """Return the frozen constants for matched or hidden-mismatch truth."""

    if mode == "matched":
        return MATCHED_CONSTANTS
    if mode == "structural_mismatch":
        return STRUCTURAL_MISMATCH_CONSTANTS
    raise ValueError(f"unknown physics mode {mode!r}")


def simulate_truth(schedule: Schedule, *, physics_mode: PhysicsMode) -> TruthRecord:
    """Run the vendored three-state CoreField model with 30 s RK4."""

    constants = physics_constants(physics_mode)
    trajectory = _physics.simulate(
        schedule.time_s,
        schedule.load_pu,
        schedule.ambient_C,
        NOMINAL_PARAMS,
        constants,
        load_pu_half=schedule.load_pu_half,
        ambient_C_half=schedule.ambient_C_half,
        solver="rk4",
    )
    return TruthRecord(
        schedule=schedule,
        physics_mode=physics_mode,
        top_oil_C=_readonly(trajectory.top_oil_C),
        hotspot_C=_readonly(trajectory.hotspot_C),
        gradient_K=_readonly(trajectory.gradient_K),
    )


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be non-negative")
    return value


def _stable_record_code(record_name: str) -> tuple[int, int]:
    """Return a process-independent 64-bit code for one declared record."""

    if not record_name.strip():
        raise ValueError("record_name must not be empty")
    digest = hashlib.sha256(record_name.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[:4], byteorder="little", signed=False),
        int.from_bytes(digest[4:8], byteorder="little", signed=False),
    )


def _stream_rng(seed: int, stream: int, record_name: str) -> np.random.Generator:
    """Independent deterministic substream by seed, sensor, and record."""

    record_code = _stable_record_code(record_name)
    return np.random.default_rng(
        np.random.SeedSequence([_validated_seed(seed), stream, *record_code])
    )


def observe_record(truth: TruthRecord, *, seed: int) -> ObservedRecord:
    """Sample top oil every 5 min and add independent 0.5 K Gaussian noise."""

    stride = int(round(TOP_OIL_SAMPLE_S / TRUTH_DT_S))
    indices = np.arange(0, truth.schedule.time_s.size, stride, dtype=np.intp)
    rng = _stream_rng(seed, 0, truth.schedule.name)
    measured = truth.top_oil_C[indices] + rng.normal(
        loc=0.0, scale=SENSOR_NOISE_SIGMA_K, size=indices.size
    )
    return ObservedRecord(
        truth=truth,
        seed=int(seed),
        top_oil_index=_readonly_index(indices),
        top_oil_time_s=_readonly(truth.schedule.time_s[indices]),
        top_oil_C=_readonly(measured),
    )


def reference_candidate_indices(schedule: Schedule) -> NDArray[np.intp]:
    """Return all 50 ordered candidates before noise is generated.

    Transient candidates are offset-major: every event at 3 min, then every
    event at 8 min, and so on.  Remaining candidates are evenly distributed
    over portions of plateaux at least 120 min after the preceding event and
    30 min before the next event.  The opening plateau is already at the
    synthetic equilibrium and is eligible.
    """

    dt = TRUTH_DT_S
    last_time = float(schedule.time_s[-1])
    transient: list[int] = []
    for offset_min in REFERENCE_OFFSETS_MIN:
        for event_s in schedule.event_time_s:
            time_s = float(event_s) + offset_min * 60.0
            if time_s <= last_time:
                transient.append(int(round(time_s / dt)))
    transient = list(dict.fromkeys(transient))

    required = max(REFERENCE_BUDGETS) - len(transient)
    if required < 0:
        transient = transient[: max(REFERENCE_BUDGETS)]
        required = 0

    all_indices = np.arange(schedule.time_s.size, dtype=np.intp)
    eligible = np.ones(schedule.time_s.size, dtype=bool)
    for event_s in schedule.event_time_s:
        after = (schedule.time_s >= event_s) & (
            schedule.time_s < event_s + REFERENCE_SETTLING_MIN * 60.0
        )
        before = (schedule.time_s < event_s) & (
            schedule.time_s > event_s - REFERENCE_EVENT_CLEARANCE_MIN * 60.0
        )
        eligible[after | before] = False
    eligible[transient] = False
    pool = all_indices[eligible]
    if pool.size < required:
        raise ValueError("schedule has too little quasi-steady support for 50 references")
    if required:
        positions = np.rint(np.linspace(0, pool.size - 1, required)).astype(np.intp)
        anchors = pool[positions].tolist()
    else:
        anchors = []
    candidates = np.asarray(transient + anchors, dtype=np.intp)
    if candidates.size != max(REFERENCE_BUDGETS) or np.unique(candidates).size != candidates.size:
        raise RuntimeError("reference candidate construction did not produce 50 unique indices")
    candidates.setflags(write=False)
    return candidates


def sparse_reference_indices(schedule: Schedule, budget: int) -> NDArray[np.intp]:
    """Select the first N frozen candidates and return them in time order."""

    if budget not in REFERENCE_BUDGETS:
        raise ValueError(f"budget must be one of {REFERENCE_BUDGETS}, got {budget!r}")
    selected = np.sort(reference_candidate_indices(schedule)[:budget])
    selected.setflags(write=False)
    return selected


def observe_hotspot_references(
    truth: TruthRecord, *, budget: int, seed: int
) -> SparseHotspotReferences:
    """Add an independent 0.5 K noise stream to nested reference candidates."""

    if budget not in REFERENCE_BUDGETS:
        raise ValueError(f"budget must be one of {REFERENCE_BUDGETS}, got {budget!r}")
    candidates = reference_candidate_indices(truth.schedule)
    rng = _stream_rng(seed, 1, truth.schedule.name)
    candidate_noise = rng.normal(
        loc=0.0, scale=SENSOR_NOISE_SIGMA_K, size=max(REFERENCE_BUDGETS)
    )
    selected_candidates = candidates[:budget]
    order = np.argsort(selected_candidates)
    indices = np.asarray(selected_candidates[order], dtype=np.intp)
    measured = truth.hotspot_C[selected_candidates] + candidate_noise[:budget]
    measured = measured[order]
    hidden = truth.hotspot_C[indices]
    indices.setflags(write=False)
    return SparseHotspotReferences(
        split=truth.split,
        seed=int(seed),
        budget=budget,
        index=indices,
        time_s=_readonly(truth.schedule.time_s[indices]),
        temperature_C=_readonly(measured),
        truth_temperature_C=_readonly(hidden),
    )


def build_feature_frame(observed: ObservedRecord) -> FeatureFrame:
    """Build the frozen nine inputs without exposing dense true top oil.

    The preregistered linear interpolation is performed solely on the noisy
    five-minute top-oil observations.  Before the record begins, lagged load
    and top oil use the opening value.  This represents the declared settled
    pre-record condition, keeps the complete 8 h E3 episode scoreable, and
    avoids discarding early quasi-steady reference candidates.
    """

    schedule = observed.truth.schedule
    model_stride = int(round(MODEL_DT_S / TRUTH_DT_S))
    model_indices = np.arange(0, schedule.time_s.size, model_stride, dtype=np.intp)
    row_time = schedule.time_s[model_indices]

    features, source_times = feature_matrix_at_times(observed, row_time)
    return FeatureFrame(
        split=observed.split,
        time_s=_readonly(row_time),
        truth_index=_readonly_index(model_indices),
        X=_readonly(features),
        hotspot_truth_C=_readonly(observed.truth.hotspot_C[model_indices]),
        source_time_s=_readonly(source_times),
    )


def feature_matrix_at_times(
    observed: ObservedRecord,
    query_time_s: NDArray[np.float64] | Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the nine declared inputs at exact in-record query times.

    This is used for contemporaneous sparse-reference losses whose timestamps
    need not lie on the regular 2 min physics grid.  Standardization is still
    fitted on :func:`build_feature_frame` for the complete train record.
    """

    schedule = observed.truth.schedule
    row_time = np.asarray(query_time_s, dtype=np.float64)
    if row_time.ndim != 1 or row_time.size == 0 or np.any(~np.isfinite(row_time)):
        raise ValueError("query_time_s must be a non-empty finite 1-D array")
    if np.any(row_time < schedule.time_s[0]) or np.any(row_time > schedule.time_s[-1]):
        raise ValueError("query_time_s must lie inside the observed record")

    load_lags_min = (0.0, 6.0, 16.0, 60.0, 180.0)
    load_columns = [
        np.interp(row_time - lag * 60.0, schedule.time_s, schedule.load_pu)
        for lag in load_lags_min
    ]
    ambient_current = np.interp(row_time, schedule.time_s, schedule.ambient_C)
    oil_lags_min = (0.0, 16.0, 60.0)
    oil_columns = [
        np.interp(
            row_time - lag * 60.0,
            observed.top_oil_time_s,
            observed.top_oil_C,
        )
        for lag in oil_lags_min
    ]
    features = np.column_stack(
        [*load_columns, ambient_current, *oil_columns]
    ).astype(np.float64, copy=False)
    source_times = np.maximum(
        schedule.time_s[0],
        row_time[:, None]
        - 60.0 * np.asarray(FEATURE_SOURCE_LAGS_MIN, dtype=np.float64)[None, :],
    )
    return features, source_times


def fit_train_standardizer(train_frame: FeatureFrame) -> TrainStandardizer:
    """Fit population mean/scale, refusing validation or test input."""

    if train_frame.split != "train":
        raise ValueError(
            f"standardization statistics must be fitted on split='train', got {train_frame.split!r}"
        )
    mean = np.mean(train_frame.X, axis=0)
    scale = np.std(train_frame.X, axis=0, ddof=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return TrainStandardizer(mean=_readonly(mean), scale=_readonly(scale))


__all__ = [
    "AMBIENT_AMPLITUDE_K",
    "AMBIENT_MEAN_C",
    "FEATURE_NAMES",
    "FEATURE_SOURCE_LAGS_MIN",
    "FeatureFrame",
    "MATCHED_CONSTANTS",
    "MAX_FEATURE_LAG_MIN",
    "MODEL_DT_S",
    "NOMINAL_PARAMS",
    "ObservedRecord",
    "PhysicsMode",
    "REFERENCE_BUDGETS",
    "REFERENCE_OFFSETS_MIN",
    "SENSOR_NOISE_SIGMA_K",
    "STRUCTURAL_MISMATCH_CONSTANTS",
    "Schedule",
    "SparseHotspotReferences",
    "SplitName",
    "TOP_OIL_SAMPLE_S",
    "TRUTH_DT_S",
    "TrainStandardizer",
    "TruthRecord",
    "build_feature_frame",
    "fit_train_standardizer",
    "feature_matrix_at_times",
    "make_e3_schedule",
    "make_in_range_test_schedule",
    "make_schedule",
    "make_train_schedule",
    "make_validation_schedule",
    "observe_hotspot_references",
    "observe_record",
    "physics_constants",
    "reference_candidate_indices",
    "simulate_truth",
    "sparse_reference_indices",
]
