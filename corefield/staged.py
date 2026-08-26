# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staged cooling: transformers whose fans and pumps switch during operation.

WHY THIS EXISTS
---------------
A real transformer with multi-stage cooling does not have one set of thermal
parameters. When a fan bank starts, the oil sheds heat faster: the rated oil
rise falls and the oil time constant shortens, discontinuously, at the moment
of the switch. Fitting a single constant parameter set across a record that
spans a stage change produces a compromise that fits neither stage and looks
like the method failing on real data.

The published EDF ODAF dataset ships a cooling-stage channel, and its
companion paper is titled for stage representation -- which is a strong hint
that a single parameter set does not survive contact with a staged unit.

WHAT CHANGES BETWEEN STAGES, AND WHAT DOES NOT
----------------------------------------------
Not all four parameters need to be per-stage, and making them all per-stage
costs conditioning for nothing. But fewer of them are shared than intuition
suggests.

`delta_theta_or` and `tau_o` are per-stage: fans act on the oil-to-air path
and change how far and how fast the oil cools.

`tau_w` is ALSO per-stage. An earlier version of this module shared it, on
the reasoning that tank fans do not touch the winding-to-oil path. That
reasoning is wrong: IEC 60076-7 Table 4 gives different winding time
constants for the two cooling classes of the same transformer (10 min
against 7 min for medium and large power units, natural against forced air).
The standard was checked and the assumption did not survive it.

`delta_theta_hr` is shared by default. It is the product of the hot-spot
factor and the rated winding-to-oil gradient, both properties of the winding
geometry and of oil circulation driven by buoyancy in either stage, so tank
fans leave it alone. Table 4 does not tabulate it -- it is unit-specific --
so this remains an assumption rather than a checked value, and it is
recorded on `StagedThermalParams.shared` for exactly that reason.

For directed-flow (OD) classes, pumps change the flow through the winding
itself and therefore the gradient too. Set `shared=()` there and let all
four float per stage, at the cost of needing enough data in every stage.

STATE IS CARRIED, PARAMETERS ARE NOT
------------------------------------
At a stage boundary the parameters jump but the stored heat does not: the
oil is exactly as hot the instant after the fan starts as the instant before.
Integration is therefore segmented, with the three internal states handed
across each boundary unchanged. Only the DIFFERENCE of the two gradient
branches is observable, so the pair has to be carried explicitly rather than
reconstructed from the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .estimator import HotspotReferences, OPTIMISER_BOUNDS, _railed_parameters
from .iec60076_7 import (
    ONAF_MEDIUM_LARGE_POWER,
    PARAM_BOUNDS,
    CoolingConstants,
    InitialState,
    ThermalParams,
    ThermalTrajectory,
    _integrate,
    _reject_kelvin,
)

__all__ = [
    "SHARED_BY_DEFAULT",
    "STRUCTURAL_MARGIN",
    "StagedThermalParams",
    "StagedIdentificationResult",
    "stage_segments",
    "simulate_staged",
    "identify_staged",
]

#: Parameters held common across cooling stages unless told otherwise.
#: Only the rated gradient. tau_w is NOT shared -- see the module docstring:
#: IEC 60076-7 Table 4 gives different winding time constants for natural and
#: forced-air cooling of the same unit, which contradicted the assumption an
#: earlier version of this module made.
SHARED_BY_DEFAULT: tuple[str, ...] = ("delta_theta_hr_K",)

_PARAM_ORDER: tuple[str, ...] = (
    "delta_theta_or_K",
    "tau_o_min",
    "delta_theta_hr_K",
    "tau_w_min",
)
#: Index of each parameter in the optimiser's working vector [K, s, K, s].
_VECTOR_INDEX = {name: i for i, name in enumerate(_PARAM_ORDER)}


@dataclass(frozen=True)
class StagedThermalParams:
    """One thermal parameter set per cooling stage.

    Attributes
    ----------
    per_stage : {stage label: ThermalParams}
    shared : names of `ThermalParams` fields held equal across all stages.
        Recorded on the object because a reader has to know whether a
        difference between stages was fitted or merely assumed away.
    constants : cooling-class constants, common to all stages. The cooling
        CLASS does not change when a fan starts within that class; only the
        parameter values do.
    """

    per_stage: Mapping[int, ThermalParams]
    shared: tuple[str, ...] = SHARED_BY_DEFAULT
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER

    def __post_init__(self) -> None:
        if not self.per_stage:
            raise ValueError("per_stage must contain at least one cooling stage")
        for name in self.shared:
            if name not in _VECTOR_INDEX:
                raise ValueError(
                    f"unknown shared parameter {name!r}; expected one of {_PARAM_ORDER}"
                )
        # A "shared" parameter that differs between stages is a lie about the
        # model, and would silently mislead anyone reading the result.
        for name in self.shared:
            values = {round(float(getattr(p, name)), 9) for p in self.per_stage.values()}
            if len(values) > 1:
                raise ValueError(
                    f"{name!r} is declared shared across stages but takes {len(values)} "
                    f"different values: {sorted(values)}."
                )

    @property
    def stages(self) -> tuple[int, ...]:
        """Stage labels, sorted."""
        return tuple(sorted(self.per_stage))

    def for_stage(self, stage: int) -> ThermalParams:
        """Parameters for one stage."""
        if stage not in self.per_stage:
            raise KeyError(
                f"no parameters for cooling stage {stage!r}; identified stages are "
                f"{self.stages}. A record containing a stage that was never fitted "
                f"cannot be simulated."
            )
        return self.per_stage[stage]

    def report(self) -> str:
        """Human-readable summary."""
        lines = [f"Staged thermal parameters ({len(self.per_stage)} stage(s))",
                 f"  shared across stages: {', '.join(self.shared) or 'nothing'}",
                 ""]
        header = f"  {'stage':>6}  {'dtheta_or [K]':>14}  {'tau_o [min]':>12}  " \
                 f"{'dtheta_hr [K]':>14}  {'tau_w [min]':>12}"
        lines.append(header)
        for stage in self.stages:
            p = self.per_stage[stage]
            lines.append(
                f"  {stage:>6}  {p.delta_theta_or_K:>14.3f}  {p.tau_o_min:>12.3f}  "
                f"{p.delta_theta_hr_K:>14.3f}  {p.tau_w_min:>12.3f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class StagedIdentificationResult:
    """Outcome of a staged identification."""

    params: StagedThermalParams
    success: bool
    cost: float
    residual_rmse_K: float
    n_observations: tuple[int, int]
    stage_sample_counts: Mapping[int, int]
    railed_parameters: tuple[str, ...]
    #: Parameters held at a supplied value instead of identified, in that
    #: field's own units. A held parameter is NOT a measurement of this unit
    #: and must not be reported as one.
    fixed_parameters: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def report(self) -> str:
        lines = [
            f"Staged identification: {'SUCCESS' if self.success else 'FAILED'}",
            f"  residual RMSE : {self.residual_rmse_K:.4f} K",
            f"  observations  : {self.n_observations[0]} top-oil, "
            f"{self.n_observations[1]} hot-spot",
            "  samples/stage : "
            + ", ".join(f"{s}={n}" for s, n in sorted(self.stage_sample_counts.items())),
            "",
            self.params.report(),
        ]
        if self.fixed_parameters:
            lines.append(
                "  HELD, NOT IDENTIFIED: "
                + ", ".join(f"{k}={v:g}" for k, v in sorted(self.fixed_parameters.items()))
                + "  (tabulated values, not measurements of this unit)"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def stage_segments(stage: NDArray[np.int_]) -> list[tuple[int, int, int]]:
    """Split a stage channel into runs.

    Returns
    -------
    List of (start_index, end_index_inclusive, stage_label), covering the
    whole series in order.
    """
    labels = np.asarray(stage)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("stage must be a non-empty 1-D array")
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries - 1, [labels.size - 1]])
    return [(int(a), int(b), int(labels[a])) for a, b in zip(starts, ends)]


def _integrate_staged(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64],
    load_half: NDArray[np.float64],
    ambient_half: NDArray[np.float64],
    stage: NDArray[np.int_],
    params: StagedThermalParams,
    dt: float,
    raw_initial: tuple[float, float, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Segmented integration, carrying internal state across stage boundaries."""
    c = params.constants
    n = time_s.size
    top_oil = np.empty(n, dtype=np.float64)
    hotspot = np.empty(n, dtype=np.float64)

    state = raw_initial
    cursor = 0
    for start, end, label in stage_segments(stage):
        p = params.for_stage(label)
        # Each segment after the first re-integrates from the previous
        # segment's LAST sample, so the boundary point is shared and the
        # state hands over exactly.
        lo = start if cursor == 0 else start - 1
        hi = end + 1
        oil_seg, hs_seg, state = _integrate(  # type: ignore[misc]
            dtor=p.delta_theta_or_K, tauo_s=p.tau_o_s,
            dthr=p.delta_theta_hr_K, tw_s=p.tau_w_s,
            x=c.x, y=c.y, x1=c.x1, y1=c.y1, k11=c.k11, k21=c.k21, k22=c.k22,
            R=p.loss_ratio_R,
            K_on=load_pu[lo:hi], K_half=load_half[lo:hi],
            A_on=ambient_C[lo:hi], A_half=ambient_half[lo:hi],
            dt=dt, solver="rk4", raw_initial=state, return_state=True,
        )
        keep = slice(0, None) if cursor == 0 else slice(1, None)
        top_oil[cursor : end + 1] = oil_seg[keep]
        hotspot[cursor : end + 1] = hs_seg[keep]
        cursor = end + 1

    return top_oil, hotspot


def _prepare(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    stage: Sequence[int] | NDArray[np.int_],
    load_pu_half: NDArray[np.float64] | None,
    ambient_C_half: NDArray[np.float64] | None,
):
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"time_s must be 1-D with >= 2 samples, got shape {t.shape}")
    steps = np.diff(t)
    dt = float(steps[0])
    if dt <= 0 or not np.allclose(steps, dt, rtol=0.0, atol=1e-9):
        raise ValueError("time_s must be uniformly spaced and increasing")

    K = np.asarray(load_pu, dtype=np.float64)
    A = np.asarray(ambient_C, dtype=np.float64)
    if A.ndim == 0:
        A = np.full(t.shape, float(A))
    S = np.asarray(stage)
    if S.dtype.kind == "f":
        if np.any(~np.isfinite(S)):
            raise ValueError("cooling_stage contains non-finite values")
        S = S.astype(np.int_)
    for name, arr in (("load_pu", K), ("ambient_C", A), ("stage", S)):
        if arr.shape != t.shape:
            raise ValueError(f"{name} shape {arr.shape} != time_s shape {t.shape}")
    _reject_kelvin(A[np.isfinite(A)], "ambient_C")

    Kh = load_pu_half if load_pu_half is not None else np.append(
        0.5 * (K[:-1] + K[1:]), K[-1]
    )
    Ah = ambient_C_half if ambient_C_half is not None else np.append(
        0.5 * (A[:-1] + A[1:]), A[-1]
    )
    return t, K, A, S, np.asarray(Kh, dtype=np.float64), np.asarray(Ah, dtype=np.float64), dt


def simulate_staged(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    stage: Sequence[int] | NDArray[np.int_],
    params: StagedThermalParams,
    *,
    load_pu_half: NDArray[np.float64] | None = None,
    ambient_C_half: NDArray[np.float64] | None = None,
    initial_state: InitialState | None = None,
) -> ThermalTrajectory:
    """Integrate a record whose cooling stage changes during it.

    Parameters
    ----------
    time_s : uniform time grid [s]
    load_pu : per-unit load current [pu]
    ambient_C : ambient temperature [degC]
    stage : cooling-stage label per sample. Any integer labels; they only
        have to match the keys of `params.per_stage`.
    params : one parameter set per stage
    load_pu_half, ambient_C_half : optional analytic half-step samples
    initial_state : the unit's state at t=0, or None for equilibrium at the
        opening load under the opening stage's parameters

    Returns
    -------
    ThermalTrajectory
    """
    t, K, A, S, Kh, Ah, dt = _prepare(
        time_s, load_pu, ambient_C, stage, load_pu_half, ambient_C_half
    )

    raw = None
    if initial_state is not None:
        first = params.for_stage(int(S[0]))
        prior_ky = float(initial_state.prior_load_pu) ** params.constants.y
        raw = (
            float(initial_state.top_oil_C),
            params.constants.k21 * first.delta_theta_hr_K * prior_ky,
            (params.constants.k21 - 1.0) * first.delta_theta_hr_K * prior_ky,
        )

    top_oil, hotspot = _integrate_staged(t, K, A, Kh, Ah, S, params, dt, raw)
    return ThermalTrajectory(
        time_s=t, load_pu=K, ambient_C=A,
        top_oil_C=top_oil, hotspot_C=hotspot, gradient_K=hotspot - top_oil,
    )


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------


def _build_packing(
    stages: tuple[int, ...],
    shared: tuple[str, ...],
    fixed: tuple[str, ...] = (),
):
    """Map (stage, parameter) onto positions in a flat optimiser vector.

    Names in `fixed` get no slot at all: they are held at a caller-supplied
    value and never seen by the optimiser.
    """
    slots: list[tuple[str, int | None]] = []
    for name in _PARAM_ORDER:
        if name in fixed:
            continue
        if name in shared:
            slots.append((name, None))
        else:
            slots.extend((name, s) for s in stages)
    return slots


def _unpack_raw(vector, slots, stages, fixed=None) -> dict[int, list[float]]:
    """Flat vector -> {stage: [dtheta_or (K), tau_o (s), dtheta_hr (K), tau_w (s)]}.

    Separate from `_unpack` because the structural-constraint check in
    `identify_staged` has to inspect a solution that `ThermalParams` would
    refuse to construct.

    `fixed` maps a parameter name to its held value in WORKING units (K for
    the rises, seconds for the time constants).
    """
    values: dict[int, list[float]] = {s: [0.0, 0.0, 0.0, 0.0] for s in stages}
    for name, held in (fixed or {}).items():
        for s in stages:
            values[s][_VECTOR_INDEX[name]] = float(held)
    for position, (name, stage) in enumerate(slots):
        index = _VECTOR_INDEX[name]
        targets = stages if stage is None else (stage,)
        for s in targets:
            values[s][index] = float(vector[position])
    return values


def _unpack(vector, slots, stages, loss_ratio_R, fixed=None):
    """Turn the flat vector back into one ThermalParams per stage."""
    return {
        s: ThermalParams.from_vector(np.array(v), loss_ratio_R=loss_ratio_R)
        for s, v in _unpack_raw(vector, slots, stages, fixed).items()
    }


#: How close tau_w may sit to tau_o before the solution counts as pinned
#: against the structural constraint rather than determined by the record.
#:
#: (c) Judgement, not a measured constant. The two-exponential structure only
#: carries information while the branches are separated; as tau_w -> tau_o the
#: winding branch has stopped being the fast one and the value the optimiser
#: reports is set by the constraint, not by the data.
STRUCTURAL_MARGIN: float = 0.01


def identify_staged(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    top_oil_C: NDArray[np.float64],
    hotspot_refs: HotspotReferences,
    stage: Sequence[int] | NDArray[np.int_],
    *,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    shared: tuple[str, ...] = SHARED_BY_DEFAULT,
    fixed: Mapping[str, float] | None = None,
    loss_ratio_R: float = 6.0,
    loss: str = "soft_l1",
    start: tuple[float, float, float, float] = (30.0, 6000.0, 15.0, 600.0),
    load_pu_half: NDArray[np.float64] | None = None,
    ambient_C_half: NDArray[np.float64] | None = None,
    max_nfev: int | None = None,
) -> StagedIdentificationResult:
    """Identify one parameter set per cooling stage, in a single joint fit.

    Fitting stages independently would throw away the boundaries, which are
    the most informative samples in the record: at a stage change the load is
    unchanged while the cooling is not, so the response isolates the
    parameters that differ. A joint fit with shared state across boundaries
    uses them.

    Parameters
    ----------
    time_s, load_pu, ambient_C : as `simulate_staged`
    top_oil_C : measured top-oil on the grid, NaN where unmeasured
    hotspot_refs : sparse hot-spot calibration reads
    stage : cooling-stage label per sample
    constants : cooling-class constants
    shared : parameters held equal across stages. See SHARED_BY_DEFAULT and
        the module docstring -- the default suits tank-fan staging and is
        wrong for directed-flow classes.
    fixed : parameters HELD at a supplied value and never fitted, keyed by
        `ThermalParams` field name in that field's own units (K, minutes).
        Use this when the record cannot inform a parameter: a 10-minute log
        samples a 7-minute winding transient less than once per time
        constant, so tau_w is not estimable from it and letting the optimiser
        chase it produces a value set by whichever bound or constraint it
        reaches first. Holding it at the IEC 60076-7 Table 4 value is the
        honest alternative, and the report says which parameters were held.
    loss_ratio_R : load loss / no-load loss, held fixed
    loss : robust loss for `least_squares`
    start : initial guess, applied to every stage
    load_pu_half, ambient_C_half : optional analytic half-step samples
    max_nfev : optimiser evaluation cap

    Returns
    -------
    StagedIdentificationResult

    Raises
    ------
    ValueError
        On malformed input, or too few observations to support the number of
        free parameters.
    RuntimeError
        If the optimiser fails to converge to an interior solution.
    """
    t, K, A, S, Kh, Ah, dt = _prepare(
        time_s, load_pu, ambient_C, stage, load_pu_half, ambient_C_half
    )
    oil = np.asarray(top_oil_C, dtype=np.float64)
    if oil.shape != t.shape:
        raise ValueError(f"top_oil_C shape {oil.shape} != time_s shape {t.shape}")
    oil_index = np.flatnonzero(np.isfinite(oil))
    if oil_index.size == 0:
        raise ValueError("top_oil_C contains no finite samples")
    _reject_kelvin(oil[oil_index], "top_oil_C")

    cal_index = np.round(
        (np.asarray(hotspot_refs.time_s, dtype=np.float64) - t[0]) / dt
    ).astype(np.intp)
    if np.any(cal_index < 0) or np.any(cal_index >= t.size):
        raise ValueError("hot-spot reference timestamps fall outside the grid")
    cal_values = np.asarray(hotspot_refs.temperature_C, dtype=np.float64)

    fixed = dict(fixed or {})
    unknown = set(fixed) - set(_PARAM_ORDER)
    if unknown:
        raise ValueError(
            f"fixed contains unknown parameter name(s) {sorted(unknown)}; "
            f"expected any of {list(_PARAM_ORDER)}"
        )
    overlap = set(fixed) & set(shared)
    if overlap:
        raise ValueError(
            f"parameter(s) {sorted(overlap)} appear in both `fixed` and `shared`; "
            f"a held parameter is already common to every stage."
        )
    # Working units: K for the rises, seconds for the time constants.
    fixed_working = {
        name: (value * 60.0 if name.endswith("_min") else value)
        for name, value in fixed.items()
    }
    for name, value in fixed.items():
        low, high = PARAM_BOUNDS[name]
        if not (low <= value <= high):
            raise ValueError(
                f"fixed[{name!r}] = {value} is outside the physical plausibility "
                f"bounds [{low}, {high}]."
            )
    if len(fixed) == len(_PARAM_ORDER):
        raise ValueError("every parameter is fixed; there is nothing to identify.")

    stages = tuple(sorted(set(int(v) for v in S)))
    slots = _build_packing(stages, shared, tuple(fixed))
    counts = {s: int(np.sum(S == s)) for s in stages}

    notes: list[str] = []
    thin = [s for s, n in counts.items() if n < 0.02 * t.size]
    if thin:
        notes.append(
            f"stage(s) {thin} occupy under 2 % of the record; their parameters rest on "
            f"very little data and their uncertainty will be correspondingly large."
        )
    if len(slots) > cal_index.size + oil_index.size:
        raise ValueError(
            f"{len(slots)} free parameters exceed the {cal_index.size + oil_index.size} "
            f"observations available."
        )
    if cal_index.size < 4:
        raise ValueError(
            f"only {cal_index.size} hot-spot reference(s); the winding parameters cannot "
            f"be identified from fewer than four."
        )

    lower_full, upper_full = OPTIMISER_BOUNDS
    lower = np.array([lower_full[_VECTOR_INDEX[n]] for n, _ in slots])
    upper = np.array([upper_full[_VECTOR_INDEX[n]] for n, _ in slots])
    x0 = np.clip(np.array([start[_VECTOR_INDEX[n]] for n, _ in slots]), lower, upper)

    def residual(vector):
        try:
            per_stage = _unpack(vector, slots, stages, loss_ratio_R, fixed_working)
        except ValueError:
            # The optimiser stepped somewhere physically invalid; steer it back
            # rather than crashing the fit.
            return np.full(oil_index.size + cal_index.size, 1e3)
        staged = StagedThermalParams(per_stage=per_stage, shared=shared, constants=constants)
        model_oil, model_hs = _integrate_staged(t, K, A, Kh, Ah, S, staged, dt, None)
        return np.concatenate(
            [model_oil[oil_index] - oil[oil_index], model_hs[cal_index] - cal_values]
        )

    result = least_squares(
        residual, x0=x0, bounds=(lower, upper), method="trf", loss=loss, max_nfev=max_nfev
    )

    railed: list[str] = []
    for i, (name, stage_label) in enumerate(slots):
        span = upper[i] - lower[i]
        if abs(result.x[i] - lower[i]) <= 1e-3 * span:
            railed.append(f"{name}@lower(stage {stage_label})")
        elif abs(result.x[i] - upper[i]) <= 1e-3 * span:
            railed.append(f"{name}@upper(stage {stage_label})")

    # A parameter can rail against a STRUCTURAL constraint as well as a box
    # bound, and the loop above cannot see it. `ThermalParams` requires
    # tau_w < tau_o, and `residual` returns a flat penalty wherever that is
    # violated -- which makes the constraint an invisible wall in the cost
    # surface. The optimiser walks tau_w up to the wall and stops against it,
    # and the result was previously reported as a converged interior solution.
    #
    # Not hypothetical: on a 360 MVA ODAF field record, stage 3 returned
    # tau_w = tau_o - 1e-6 min and was accepted. From other starts the
    # optimiser finished ON the wall and `_unpack` raised a bare ValueError
    # out of the middle of the fit instead of the designed refusal.
    for stage_label, v in sorted(
        _unpack_raw(result.x, slots, stages, fixed_working).items()
    ):
        tau_o_s, tau_w_s = v[1], v[3]
        if tau_w_s >= (1.0 - STRUCTURAL_MARGIN) * tau_o_s:
            railed.append(
                f"tau_w_min@tau_o-constraint(stage {stage_label}: "
                f"tau_w={tau_w_s / 60.0:.2f} min vs tau_o={tau_o_s / 60.0:.2f} min)"
            )

    if result.status <= 0 or railed:
        raise RuntimeError(
            f"staged identification failed: status={result.status}, "
            f"railed={railed or 'none'}, message={result.message!r}.\n"
            "A railed solution is an optimiser artifact, not a measurement. Usual "
            "causes: a stage with too few samples, no load variation within a stage, "
            "or `shared` set too loose for the data available. A parameter railed "
            "against the tau_w < tau_o constraint means the winding branch has no "
            "support in this record: the value reported is set by the constraint, "
            "not measured."
        )

    params = StagedThermalParams(
        per_stage=_unpack(result.x, slots, stages, loss_ratio_R, fixed_working),
        shared=shared, constants=constants,
    )
    final = residual(result.x)
    return StagedIdentificationResult(
        params=params,
        success=True,
        cost=float(result.cost),
        residual_rmse_K=float(np.sqrt(np.mean(final**2))),
        n_observations=(int(oil_index.size), int(cal_index.size)),
        stage_sample_counts=counts,
        railed_parameters=tuple(railed),
        fixed_parameters=dict(fixed),
        warnings=tuple(notes),
    )
