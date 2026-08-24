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

"""Identification of the four IEC thermal parameters by nonlinear least squares.

This is the production engine. It is classical NLS, not a neural network,
and that is a measured result rather than a preference: under structural
mismatch the single-exponential alternatives read the hot spot several
kelvin HIGH at high load, triggering derating exactly when capacity is worth
the most. See README.md for the comparison table.

Observation model
-----------------
Two channels, deliberately asymmetric:

  DENSE  top-oil temperature, SCADA-grade, typically every 5 min. Carries
         all the information about the oil parameters and none at all about
         the winding -- the oil equation contains no winding terms, so
         d(theta_o)/d{dtheta_hr, tau_w} = 0 identically.

  SPARSE hot-spot calibration reads. Because of the above, some direct
         hot-spot data is unavoidable. WHEN those reads are taken decides
         whether the problem is well-posed at all: amplitude parameters are
         observable from quasi-steady operation, rate parameters only from
         transients, and each sampled transient must also anchor its own
         settled asymptote or the two stay correlated and the optimiser
         walks a degenerate valley. `synthetic.calibration_indices`
         implements the schedule that follows from this.

Integration grid vs observation grid
------------------------------------
These are NOT the same grid and must not be conflated. The model integrates
at a fine step (30 s by default) because the winding time constant is
minutes; observations arrive far more slowly. Pass a fine `time_s` grid with
`top_oil_C` set to NaN wherever no measurement exists.
`corefield.ingest.load_telemetry` produces exactly this layout.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .iec60076_7 import (
    ONAF_MEDIUM_LARGE_POWER,
    PARAM_BOUNDS,
    CoolingConstants,
    ThermalParams,
    _integrate,
    _reject_kelvin,
)

__all__ = [
    "HotspotReferences",
    "StartOutcome",
    "IdentificationResult",
    "DEFAULT_STARTS",
    "OPTIMISER_BOUNDS",
    "identify",
]

Loss = Literal["soft_l1", "linear", "huber", "cauchy", "arctan"]


# --------------------------------------------------------------------------
# Optimiser configuration
# --------------------------------------------------------------------------

#: Optimiser bounds as (lower, upper) in the working vector's units:
#: [dtheta_or (K), tau_o (s), dtheta_hr (K), tau_w (s)].
#:
#: Derived from PARAM_BOUNDS in iec60076_7, which carries the provenance --
#: including the honest note that the tau_w ceiling was widened mid-campaign
#: to stop a mismatched model railing, which is an optimiser-hygiene reason
#: and not a physical one.
OPTIMISER_BOUNDS: tuple[NDArray[np.float64], NDArray[np.float64]] = (
    np.array(
        [
            PARAM_BOUNDS["delta_theta_or_K"][0],
            PARAM_BOUNDS["tau_o_min"][0] * 60.0,
            PARAM_BOUNDS["delta_theta_hr_K"][0],
            PARAM_BOUNDS["tau_w_min"][0] * 60.0,
        ]
    ),
    np.array(
        [
            PARAM_BOUNDS["delta_theta_or_K"][1],
            PARAM_BOUNDS["tau_o_min"][1] * 60.0,
            PARAM_BOUNDS["delta_theta_hr_K"][1],
            PARAM_BOUNDS["tau_w_min"][1] * 60.0,
        ]
    ),
)

#: Coarse multi-start grid, as [dtheta_or (K), tau_o (s), dtheta_hr (K), tau_w (s)].
#:
#: The FIRST entry is the single start the published campaign used. It is
#: first so that when it converges to the global optimum -- which it does on
#: every synthetic case tested -- the multi-start machinery returns exactly
#: the campaign's answer and the regression suite pins a real reproduction
#: rather than a lucky restart.
#:
#: The remaining entries bracket the plausible envelope: a cool/fast unit, a
#: hot/slow unit, and a mid-range unit with a deliberately wrong tau_w. Four
#: starts is a compromise -- each costs a full fit, and the objective has not
#: shown multiple minima on any case examined, so this is insurance against
#: real data rather than a response to observed multimodality.
DEFAULT_STARTS: tuple[tuple[float, float, float, float], ...] = (
    (30.0, 6000.0, 15.0, 600.0),
    (20.0, 3600.0, 10.0, 240.0),
    (60.0, 12000.0, 35.0, 900.0),
    (45.0, 9000.0, 22.0, 1800.0),
)

#: Relative distance to a bound below which a solution is called "railed".
_RAIL_RTOL = 1e-3


# --------------------------------------------------------------------------
# Inputs and outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HotspotReferences:
    """Sparse hot-spot calibration reads.

    Attributes
    ----------
    time_s : timestamps of the reads [s], on the same origin as the
        simulation grid
    temperature_C : measured hot-spot temperature [degC]
    source : provenance of the reference, free text. This matters: a
        winding-temperature-indicator replica carries its own calibration
        bias straight into the identified parameters (+14.5 % on dtheta_hr
        for a +3 K replica offset), so the commissioning requirement is at
        least one BIAS-AUDITED reference per unit.
    """

    time_s: NDArray[np.float64]
    temperature_C: NDArray[np.float64]
    source: str = "unspecified"

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=np.float64)
        v = np.asarray(self.temperature_C, dtype=np.float64)
        if t.shape != v.shape:
            raise ValueError(
                f"time_s shape {t.shape} does not match temperature_C shape {v.shape}"
            )
        if t.ndim != 1 or t.size == 0:
            raise ValueError("hot-spot references must be a non-empty 1-D series")
        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(v)):
            raise ValueError("hot-spot references contain non-finite values")
        _reject_kelvin(v, "HotspotReferences.temperature_C")

    def __len__(self) -> int:
        return int(np.asarray(self.time_s).size)


@dataclass(frozen=True)
class StartOutcome:
    """What happened from one multi-start initial guess."""

    start: tuple[float, float, float, float]
    converged: bool
    cost: float
    status: int
    message: str
    railed_parameters: tuple[str, ...]
    params: ThermalParams | None


@dataclass(frozen=True)
class IdentificationResult:
    """Outcome of an identification run.

    Attributes
    ----------
    params : the identified parameters (lowest-cost converged start)
    success : whether at least one start converged without railing
    cost : final least-squares cost of the selected solution
    residual_rmse_K : RMS of the selected solution's residual [K]
    oil_residual_rmse_K : RMS over the dense top-oil channel only [K]
    hotspot_residual_rmse_K : RMS over the sparse calibration channel only [K]
    n_observations : (dense count, sparse count)
    starts : per-start outcomes, in the order attempted
    loss : the robust loss actually used
    jacobian_condition : condition number of the Jacobian at the solution.
        Large values (> 1e8) mean the parameters are poorly separated by
        this particular record -- usually too few load events.
    warnings : non-fatal problems worth surfacing to the user
    """

    params: ThermalParams
    success: bool
    cost: float
    residual_rmse_K: float
    oil_residual_rmse_K: float
    hotspot_residual_rmse_K: float
    n_observations: tuple[int, int]
    starts: tuple[StartOutcome, ...]
    loss: str
    jacobian_condition: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_converged(self) -> int:
        """How many starts converged."""
        return sum(1 for s in self.starts if s.converged)

    def report(self) -> str:
        """Human-readable summary for the console or the demo app."""
        p = self.params
        lines = [
            f"Identification: {'SUCCESS' if self.success else 'FAILED'}",
            f"  starts converged      : {self.n_converged}/{len(self.starts)}",
            f"  loss                  : {self.loss}",
            f"  observations          : {self.n_observations[0]} top-oil, "
            f"{self.n_observations[1]} hot-spot reference(s)",
            "",
            f"  delta_theta_or        : {p.delta_theta_or_K:8.3f} K",
            f"  tau_o                 : {p.tau_o_min:8.3f} min",
            f"  delta_theta_hr        : {p.delta_theta_hr_K:8.3f} K",
            f"  tau_w                 : {p.tau_w_min:8.3f} min",
            "",
            f"  residual RMSE         : {self.residual_rmse_K:.4f} K "
            f"(oil {self.oil_residual_rmse_K:.4f}, "
            f"hot-spot {self.hotspot_residual_rmse_K:.4f})",
            f"  Jacobian condition    : {self.jacobian_condition:.2e}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------


def _nearest_indices(
    grid_s: NDArray[np.float64], times_s: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Map observation timestamps onto the nearest simulation-grid indices."""
    dt = float(grid_s[1] - grid_s[0])
    idx = np.round((times_s - grid_s[0]) / dt).astype(np.intp)
    if np.any(idx < 0) or np.any(idx >= grid_s.size):
        raise ValueError(
            "hot-spot reference timestamps fall outside the simulation grid "
            f"[{grid_s[0]:.0f}, {grid_s[-1]:.0f}] s"
        )
    return idx


def identify(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    top_oil_C: NDArray[np.float64],
    hotspot_refs: HotspotReferences,
    *,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    loss_ratio_R: float = 6.0,
    loss: Loss = "soft_l1",
    f_scale: float = 1.0,
    starts: Sequence[Sequence[float]] | None = None,
    max_nfev: int | None = None,
    load_pu_half: NDArray[np.float64] | None = None,
    ambient_C_half: NDArray[np.float64] | None = None,
) -> IdentificationResult:
    """Identify the four thermal parameters from telemetry plus calibration reads.

    Parameters
    ----------
    time_s : uniform simulation grid [s]. Must be fine enough to resolve the
        winding transient (step <= tau_w/2); 30 s is the campaign default.
    load_pu : per-unit load current on the grid [pu]
    ambient_C : ambient temperature [degC], scalar or per-sample. NOT
        optional and NOT ignorable -- ignoring a varying ambient
        under-predicts the afternoon peak by ~3 K, in the dangerous
        direction. `corefield.ingest` refuses to fit without it.
    top_oil_C : measured top-oil temperature on the grid [degC], with NaN at
        every sample where no measurement exists. The dense channel.
    hotspot_refs : sparse hot-spot calibration reads. The only source of
        winding information -- see the module docstring.
    constants : cooling-class constants
    loss_ratio_R : load loss / no-load loss [-], from the nameplate. Held
        fixed, not identified.
    loss : robust loss for `scipy.optimize.least_squares`. Defaults to
        `soft_l1`. This is INSURANCE, NOT RESCUE -- it costs nothing at
        baseline (0.12 K vs 0.13 K RMSE on the tested glitch model), so it
        stays on by default against heavier-tailed glitch distributions than
        the ones tested. Pass `loss="linear"` to reproduce published
        campaign numbers, which were produced with plain least squares.
    f_scale : soft margin between inlier and outlier residuals [K]
    starts : multi-start initial guesses as [dtheta_or (K), tau_o (s),
        dtheta_hr (K), tau_w (s)]. Defaults to DEFAULT_STARTS.
    max_nfev : optimiser evaluation cap per start
    load_pu_half, ambient_C_half : optional exact half-step samples for the
        RK4 interior stages. Real telemetry cannot supply these and they
        default to endpoint averages, which is second-order accurate and
        fine at 30 s against a 7 min winding constant. They exist because
        the synthetic campaign DID have the analytic driving functions and
        used them; reproducing its numbers requires passing them.

    Returns
    -------
    IdentificationResult

    Raises
    ------
    ValueError
        On malformed input, a non-uniform grid, a too-coarse grid, an
        ambient series in kelvin, or too few hot-spot references.
    RuntimeError
        If EVERY start fails to converge. The failure is reported rather
        than returning the best of a bad set -- a railed or non-converged
        solution that is silently returned is indistinguishable from a
        real answer, and this package will not manufacture one.
    """
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"time_s must be 1-D with >= 2 samples, got shape {t.shape}")
    steps = np.diff(t)
    dt = float(steps[0])
    if dt <= 0 or not np.allclose(steps, dt, rtol=0.0, atol=1e-9):
        raise ValueError(
            "time_s must be uniformly spaced and increasing; resample first "
            "(corefield.ingest.load_telemetry does this)."
        )

    K = np.asarray(load_pu, dtype=np.float64)
    oil = np.asarray(top_oil_C, dtype=np.float64)
    if K.shape != t.shape:
        raise ValueError(f"load_pu shape {K.shape} != time_s shape {t.shape}")
    if oil.shape != t.shape:
        raise ValueError(f"top_oil_C shape {oil.shape} != time_s shape {t.shape}")
    if np.any(~np.isfinite(K)):
        raise ValueError("load_pu contains non-finite values; gap-fill before identifying")

    A = np.asarray(ambient_C, dtype=np.float64)
    if A.ndim == 0:
        A = np.full(t.shape, float(A))
    elif A.shape != t.shape:
        raise ValueError(f"ambient_C shape {A.shape} != time_s shape {t.shape}")
    _reject_kelvin(A[np.isfinite(A)], "ambient_C")
    if np.any(~np.isfinite(A)):
        raise ValueError("ambient_C contains non-finite values; gap-fill before identifying")

    oil_index = np.flatnonzero(np.isfinite(oil))
    if oil_index.size == 0:
        raise ValueError("top_oil_C contains no finite samples -- nothing to fit the oil to")
    _reject_kelvin(oil[oil_index], "top_oil_C")

    cal_index = _nearest_indices(t, np.asarray(hotspot_refs.time_s, dtype=np.float64))
    cal_values = np.asarray(hotspot_refs.temperature_C, dtype=np.float64)
    if cal_index.size < 4:
        raise ValueError(
            f"only {cal_index.size} hot-spot reference(s) supplied. Four free parameters "
            f"cannot be identified from fewer than four independent reads, and in "
            f"practice a single load event (5 reads) already sits at a 13.5 % CRLB floor "
            f"on tau_w. Commission on at least two load events."
        )

    oil_measured = oil[oil_index]
    if load_pu_half is None:
        K_half = np.append(0.5 * (K[:-1] + K[1:]), K[-1])
    else:
        K_half = np.asarray(load_pu_half, dtype=np.float64)
        if K_half.shape != t.shape:
            raise ValueError(f"load_pu_half shape {K_half.shape} != time_s shape {t.shape}")
    if ambient_C_half is None:
        A_half = np.append(0.5 * (A[:-1] + A[1:]), A[-1])
    else:
        A_half = np.asarray(ambient_C_half, dtype=np.float64)
        if A_half.shape != t.shape:
            raise ValueError(f"ambient_C_half shape {A_half.shape} != time_s shape {t.shape}")

    x, y = constants.x, constants.y
    k11, k21, k22 = constants.k11, constants.k21, constants.k22

    def _model(vector: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _integrate(
            dtor=float(vector[0]), tauo_s=float(vector[1]),
            dthr=float(vector[2]), tw_s=float(vector[3]),
            x=x, y=y, k11=k11, k21=k21, k22=k22, R=loss_ratio_R,
            K_on=K, K_half=K_half, A_on=A, A_half=A_half, dt=dt, solver="rk4",
        )

    def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        model_oil, model_hs = _model(vector)
        return np.concatenate(
            [model_oil[oil_index] - oil_measured, model_hs[cal_index] - cal_values]
        )

    lower, upper = OPTIMISER_BOUNDS
    start_grid = tuple(tuple(float(v) for v in s) for s in (starts or DEFAULT_STARTS))
    for s in start_grid:
        if len(s) != 4:
            raise ValueError(f"each start must have 4 entries, got {s!r}")

    outcomes: list[StartOutcome] = []
    best: tuple[float, NDArray[np.float64], object] | None = None

    for s in start_grid:
        x0 = np.clip(np.array(s, dtype=np.float64), lower, upper)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = least_squares(
                    residual, x0=x0, bounds=(lower, upper), method="trf",
                    loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                )
            except (ValueError, FloatingPointError) as exc:
                outcomes.append(
                    StartOutcome(s, False, float("inf"), -99, f"raised {exc!r}", (), None)
                )
                continue

        railed = _railed_parameters(result.x, lower, upper)
        converged = bool(result.status > 0) and not railed
        try:
            params = ThermalParams.from_vector(result.x, loss_ratio_R=loss_ratio_R)
        except ValueError:
            params = None
            converged = False

        outcomes.append(
            StartOutcome(
                start=s, converged=converged, cost=float(result.cost),
                status=int(result.status), message=str(result.message),
                railed_parameters=railed, params=params,
            )
        )
        if converged and (best is None or result.cost < best[0]):
            best = (float(result.cost), result.x, result)

    if best is None:
        detail = "\n".join(
            f"    start {o.start}: status={o.status}, railed={o.railed_parameters or 'none'}, "
            f"{o.message}"
            for o in outcomes
        )
        raise RuntimeError(
            f"identification failed: none of {len(start_grid)} starts converged to an "
            f"interior solution.\n{detail}\n"
            "A solution pinned to a bound is an optimiser artifact, not a measurement. "
            "The usual causes are: too few load events in the record (the rate "
            "parameters are then unobservable), a missing or mis-scaled ambient "
            "channel, or load in amperes where per-unit was expected. Returning the "
            "best of a bad set would hide this, so it is refused."
        )

    best_cost, best_x, best_result = best
    model_oil, model_hs = _model(best_x)
    oil_resid = model_oil[oil_index] - oil_measured
    hs_resid = model_hs[cal_index] - cal_values
    full_resid = np.concatenate([oil_resid, hs_resid])

    condition = _jacobian_condition(best_result)
    notes: list[str] = []
    if condition > 1e8:
        notes.append(
            f"Jacobian condition number {condition:.1e} -- the parameters are poorly "
            f"separated by this record. Most often too few load events; the rate "
            f"parameters need transients that anchor both rise and settled value."
        )
    n_failed = len(outcomes) - sum(1 for o in outcomes if o.converged)
    if n_failed:
        notes.append(
            f"{n_failed} of {len(outcomes)} starts did not converge to an interior "
            f"solution; the reported answer is the best of those that did."
        )
    spread = _converged_spread(outcomes)
    if spread is not None and spread > 0.05:
        notes.append(
            f"converged starts disagree by up to {spread * 100:.1f} % on at least one "
            f"parameter, which suggests more than one local minimum. Inspect "
            f"`starts` before trusting the point estimate."
        )

    return IdentificationResult(
        params=ThermalParams.from_vector(best_x, loss_ratio_R=loss_ratio_R),
        success=True,
        cost=best_cost,
        residual_rmse_K=float(np.sqrt(np.mean(full_resid**2))),
        oil_residual_rmse_K=float(np.sqrt(np.mean(oil_resid**2))),
        hotspot_residual_rmse_K=float(np.sqrt(np.mean(hs_resid**2))),
        n_observations=(int(oil_index.size), int(cal_index.size)),
        starts=tuple(outcomes),
        loss=str(loss),
        jacobian_condition=condition,
        warnings=tuple(notes),
    )


def _railed_parameters(
    x: NDArray[np.float64], lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> tuple[str, ...]:
    """Names of parameters sitting on a bound.

    A railed parameter is the signature of a broken fit, not a tight one:
    least squares has run out of road rather than found a minimum. The
    independent implementation audited during the campaign railed tau_w at
    its 2.0-min bound in 9 of 9 runs and reported the result as an
    identifiability problem -- it was an implementation failure. Detect it.
    """
    names = ("delta_theta_or_K", "tau_o_min", "delta_theta_hr_K", "tau_w_min")
    hits: list[str] = []
    for i, name in enumerate(names):
        span = upper[i] - lower[i]
        if abs(x[i] - lower[i]) <= _RAIL_RTOL * span:
            hits.append(f"{name}@lower")
        elif abs(x[i] - upper[i]) <= _RAIL_RTOL * span:
            hits.append(f"{name}@upper")
    return tuple(hits)


def _jacobian_condition(result: object) -> float:
    """Condition number of the optimiser's final Jacobian, or inf if unavailable."""
    jac = getattr(result, "jac", None)
    if jac is None:
        return float("inf")
    try:
        singular = np.linalg.svd(np.asarray(jac), compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf")
    if singular.size == 0 or singular[-1] <= 0.0:
        return float("inf")
    return float(singular[0] / singular[-1])


def _converged_spread(outcomes: Sequence[StartOutcome]) -> float | None:
    """Largest relative disagreement between converged starts, or None."""
    vectors = [o.params.as_vector() for o in outcomes if o.converged and o.params is not None]
    if len(vectors) < 2:
        return None
    stack = np.vstack(vectors)
    reference = np.median(stack, axis=0)
    reference = np.where(reference == 0.0, 1.0, reference)
    return float(np.max(np.abs(stack - reference) / np.abs(reference)))
