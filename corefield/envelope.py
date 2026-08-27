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

"""Dynamic loading envelope -- how much more can this unit carry, and for how long.

This is the commercial output. Everything upstream exists to make this
number trustworthy.

THE LIMITS ARE NOT IN THIS FILE, AND THAT IS DELIBERATE
-------------------------------------------------------
IEC 60076-7 states loading limits for each loading type and transformer
category. This repository's copy of the standard is mirror-sourced and
UNVERIFIED, so hard-coding those numbers from memory would put an
unverifiable temperature limit at the exact point where the software tells
an operator it is safe to overload a transformer. That is the worst possible
place for a remembered number.

So `LoadingLimits` has no defaults. The caller constructs it, states where
the values came from, and that provenance string travels with every result
and appears in every summary. `iec_loading_limits()` exists only to raise a
NotImplementedError explaining this, so that someone looking for the
defaults finds the reasoning rather than silence.

WHAT THE UNCERTAINTY BAND DOES AND DOES NOT COVER
--------------------------------------------------
The conservative envelope propagates PARAMETER uncertainty from the
Cramer-Rao bound through the thermal model by Monte Carlo, and reports the
load at which the requested confidence quantile of the peak hot spot still
respects the limit.

Three honest caveats, all of which make the true uncertainty LARGER than
what is reported:

  1. The CRLB is a LOWER bound on estimator covariance. A real estimator
     matches it at best, so sampling from it understates parameter spread.
  2. Structural error is not included. The model is assumed correct. On
     synthetic data it was; on a real transformer that is an assumption,
     and no field validation exists yet.
  3. Ambient and load FORECAST error is not included. The envelope answers
     "if ambient follows this profile", not "whatever the weather does".

Do not present the conservative number as a safety guarantee. It is a
parameter-uncertainty band, and it is labelled as one in `summary()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .crlb import CRLBResult
from .iec60076_7 import (
    ONAF_MEDIUM_LARGE_POWER,
    CoolingConstants,
    InitialState,
    ThermalParams,
    _integrate,
)

__all__ = [
    "LoadingLimits",
    "LoadingEnvelope",
    "iec_loading_limits",
    "peak_hotspot_at_load",
    "loading_envelope",
]

_PLACEHOLDER_SOURCES = {
    "", "tbd", "todo", "unknown", "n/a", "na", "none", "?", "iec", "standard",
    "from memory", "default", "placeholder", "example",
}


@dataclass(frozen=True)
class LoadingLimits:
    """Thermal limits the envelope must respect. No defaults, by design.

    Parameters
    ----------
    hotspot_limit_C : maximum permitted winding hot-spot temperature [degC]
    top_oil_limit_C : maximum permitted top-oil temperature [degC], or None
        to leave top-oil unconstrained
    label : the loading case these limits describe, e.g. "normal cyclic
        loading, medium power transformer"
    source : WHERE THE NUMBERS CAME FROM. Required, and checked against a
        list of placeholders. Write something a reviewer could follow up:
        the clause of a licensed copy of the standard, a utility loading
        policy document, or a manufacturer's letter.

        This package's copy of IEC 60076-7 is mirror-sourced and UNVERIFIED.
        Take these values from a licensed copy of the standard, or from your
        own asset policy -- not from this software, and not from memory.

    Raises
    ------
    ValueError
        On a non-finite or implausible limit, or a missing/placeholder source.
    """

    hotspot_limit_C: float
    top_oil_limit_C: float | None
    label: str
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.hotspot_limit_C):
            raise ValueError("hotspot_limit_C must be finite")
        if not (40.0 <= self.hotspot_limit_C <= 250.0):
            raise ValueError(
                f"hotspot_limit_C = {self.hotspot_limit_C} is outside any plausible range "
                f"(40-250 degC). Check the units -- this is degrees Celsius, not kelvin."
            )
        if self.top_oil_limit_C is not None:
            if not np.isfinite(self.top_oil_limit_C):
                raise ValueError("top_oil_limit_C must be finite or None")
            if self.top_oil_limit_C >= self.hotspot_limit_C:
                raise ValueError(
                    f"top_oil_limit_C ({self.top_oil_limit_C}) must be below "
                    f"hotspot_limit_C ({self.hotspot_limit_C}): the hot spot is by "
                    f"definition hotter than the oil it sits in."
                )
        if not self.label.strip():
            raise ValueError("label must describe the loading case")
        if self.source.strip().lower() in _PLACEHOLDER_SOURCES or len(self.source.strip()) < 8:
            raise ValueError(
                f"source = {self.source!r} is not a usable provenance record. State where "
                f"these limits came from, specifically enough that a reviewer could check "
                f"it -- a licensed copy of the standard, a utility loading policy, or a "
                f"manufacturer's letter. This package's own IEC text is UNVERIFIED and "
                f"must not be cited as the source."
            )


def iec_loading_limits(*_args: object, **_kwargs: object) -> LoadingLimits:
    """Deliberately unimplemented. Read the error message.

    Raises
    ------
    NotImplementedError
        Always. The loading limits must be sourced by the caller.
    """
    raise NotImplementedError(
        "CoreField does not ship IEC 60076-7 loading limits.\n"
        "\n"
        "The limits decide the temperature at which this software tells an operator it "
        "is safe to overload a transformer. This repository's copy of IEC 60076-7 was "
        "mirror-sourced and is UNVERIFIED, so supplying those numbers here would put an "
        "unchecked value at the single most consequential point in the product.\n"
        "\n"
        "Construct LoadingLimits yourself, from a licensed copy of the standard for your "
        "loading type and transformer category, or from your own asset-management policy:\n"
        "\n"
        "    LoadingLimits(\n"
        "        hotspot_limit_C=<from your source>,\n"
        "        top_oil_limit_C=<from your source, or None>,\n"
        "        label='normal cyclic loading, medium power',\n"
        "        source='IEC 60076-7:2018 Table N, licensed copy, checked 2026-08-24 by AB',\n"
        "    )\n"
    )


@dataclass(frozen=True)
class LoadingEnvelope:
    """How much load this unit can carry, for how long, and what that is worth.

    Attributes
    ----------
    k_max_pu : largest sustainable load at the point-estimate parameters [pu]
    k_max_conservative_pu : largest load at which the `confidence` quantile
        of the peak hot spot still respects the limits [pu]
    duration_h : the window the answer applies to [h]
    limits : the limits used, with their provenance
    limiting_constraint : "hot-spot", "top-oil", or "search-bound"
    peak_hotspot_C, peak_top_oil_C : peaks reached at `k_max_pu` [degC]
    confidence : quantile used for the conservative bound
    nameplate_MVA : nameplate rating, if supplied [MVA]
    headroom_MVA_h, conservative_headroom_MVA_h : energy above nameplate
        over the window [MVA-hours], if `nameplate_MVA` was supplied
    feasible : False when the unit is already at or above its limit
    notes : caveats a reader must see
    """

    k_max_pu: float
    k_max_conservative_pu: float
    duration_h: float
    limits: LoadingLimits
    limiting_constraint: Literal["hot-spot", "top-oil", "search-bound", "infeasible"]
    peak_hotspot_C: float
    peak_top_oil_C: float
    confidence: float
    nameplate_MVA: float | None = None
    headroom_MVA_h: float | None = None
    conservative_headroom_MVA_h: float | None = None
    feasible: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """The answer in the two forms a utility actually asks for."""
        if not self.feasible:
            return (
                f"NO HEADROOM. At its current thermal state this unit already reaches "
                f"{self.peak_hotspot_C:.1f} degC against a limit of "
                f"{self.limits.hotspot_limit_C:.1f} degC ({self.limits.label}).\n"
                f"Limits source: {self.limits.source}"
            )

        lines = [
            f"This unit can carry {self.k_max_conservative_pu:.3f} pu for "
            f"{self.duration_h:.1f} hours",
            f"  (point estimate {self.k_max_pu:.3f} pu; "
            f"{self.confidence * 100:.0f} % parameter-confidence value quoted first).",
            f"  Binding constraint: {self.limiting_constraint} "
            f"({self.limits.hotspot_limit_C:.1f} degC hot-spot"
            + (
                f", {self.limits.top_oil_limit_C:.1f} degC top-oil)"
                if self.limits.top_oil_limit_C is not None
                else ")"
            ),
        ]
        if self.conservative_headroom_MVA_h is not None:
            lines.append(
                f"  That is {self.conservative_headroom_MVA_h:.1f} MVA-hours above nameplate "
                f"(point estimate {self.headroom_MVA_h:.1f})."
            )
        lines.append(f"  Limits source: {self.limits.source}")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def _ambient_arrays(
    ambient_C: float | Sequence[float] | NDArray[np.float64], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ambient = np.asarray(ambient_C, dtype=np.float64)
    if ambient.ndim == 0:
        series = np.full(n, float(ambient))
    elif ambient.size == n:
        series = ambient.astype(np.float64)
    else:
        raise ValueError(
            f"ambient_C must be a scalar or match the {n}-point envelope grid, "
            f"got {ambient.size} values"
        )
    if series.size and float(np.min(series)) > 200.0:
        raise ValueError(
            "ambient_C looks like kelvin; this API takes degrees Celsius."
        )
    half = np.append(0.5 * (series[:-1] + series[1:]), series[-1])
    return series, half


def peak_hotspot_at_load(
    load_pu: float,
    params: ThermalParams,
    ambient_C: float | Sequence[float] | NDArray[np.float64],
    duration_h: float,
    initial_state: InitialState,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    dt_s: float = 30.0,
) -> tuple[float, float]:
    """Peak (hot-spot, top-oil) reached holding `load_pu` for `duration_h`.

    Returns
    -------
    (peak_hotspot_C, peak_top_oil_C)
    """
    n = max(2, int(round(duration_h * 3600.0 / dt_s)) + 1)
    load = np.full(n, float(load_pu))
    ambient, ambient_half = _ambient_arrays(ambient_C, n)
    top_oil, hotspot = _integrate(
        dtor=params.delta_theta_or_K, tauo_s=params.tau_o_s,
        dthr=params.delta_theta_hr_K, tw_s=params.tau_w_s,
        x=constants.x, y=constants.y, x1=constants.x1, y1=constants.y1,
        k11=constants.k11,
        k21=constants.k21, k22=constants.k22, R=params.loss_ratio_R,
        K_on=load, K_half=load, A_on=ambient, A_half=ambient_half,
        dt=dt_s, solver="rk4", initial_state=initial_state,
    )
    return float(hotspot.max()), float(top_oil.max())


def _sample_parameters(
    params: ThermalParams,
    bound: CRLBResult,
    n_samples: int,
    rng: np.random.Generator,
) -> list[ThermalParams]:
    """Draw parameter sets from the CRLB covariance, rejecting unphysical ones.

    The CRLB is a lower bound on estimator covariance, so this UNDERSTATES
    the real spread. Recorded here rather than only in the module docstring
    because this is the function someone will read when they wonder how
    conservative the conservative number really is.
    """
    mean = params.as_vector()
    samples: list[ThermalParams] = []
    attempts = 0
    while len(samples) < n_samples and attempts < 20 * n_samples:
        attempts += 1
        draw = rng.multivariate_normal(mean, bound.covariance)
        try:
            samples.append(ThermalParams.from_vector(draw, loss_ratio_R=params.loss_ratio_R))
        except ValueError:
            continue  # outside physical bounds, or tau_w >= tau_o
    if not samples:
        raise RuntimeError(
            "could not draw any physically valid parameter samples from the CRLB "
            "covariance. The bound is implausibly wide for this record, which usually "
            "means too few load events or too few calibration reads."
        )
    return samples


def loading_envelope(
    params: ThermalParams,
    limits: LoadingLimits,
    ambient_C: float | Sequence[float] | NDArray[np.float64],
    duration_h: float,
    initial_state: InitialState,
    *,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    bound: CRLBResult | None = None,
    confidence: float = 0.95,
    n_samples: int = 200,
    nameplate_MVA: float | None = None,
    fitted_load_hull: tuple[float, float] | None = None,
    dt_s: float = 30.0,
    search_range: tuple[float, float] = (0.2, 2.0),
    tolerance_pu: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> LoadingEnvelope:
    """Largest load this unit can sustain for `duration_h` within `limits`.

    Parameters
    ----------
    params : identified thermal parameters
    limits : thermal limits, with provenance. See `LoadingLimits`.
    ambient_C : ambient over the window [degC]; scalar or per-grid-point
    duration_h : how long the load must be sustained [h]
    initial_state : the unit's thermal state now
    constants : cooling-class constants
    bound : CRLB for the identified parameters. Supply it to get a
        conservative envelope; without it the conservative value equals the
        point estimate and a note says so.
    confidence : quantile of the peak hot spot that must respect the limit
    n_samples : Monte Carlo draws for the uncertainty band
    nameplate_MVA : rating, to express the answer in MVA-hours
    fitted_load_hull : (min, max) load the parameters were identified over.
        If the envelope exceeds it, a note flags the extrapolation -- that
        is exactly where the falsified single-exponential models erred by
        +5.76 K.
    dt_s : integration step [s]
    search_range : bisection bracket on load [pu]
    tolerance_pu : bisection tolerance [pu]
    rng : generator for the Monte Carlo draws

    Returns
    -------
    LoadingEnvelope
    """
    if duration_h <= 0:
        raise ValueError(f"duration_h must be > 0, got {duration_h}")
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"confidence must be in [0.5, 1.0), got {confidence}")
    low, high = search_range
    if not 0 < low < high:
        raise ValueError(f"search_range must satisfy 0 < low < high, got {search_range}")

    generator = rng or np.random.default_rng(0)
    notes: list[str] = []

    def within(peaks_hs: float, peaks_oil: float) -> bool:
        if peaks_hs > limits.hotspot_limit_C:
            return False
        if limits.top_oil_limit_C is not None and peaks_oil > limits.top_oil_limit_C:
            return False
        return True

    def point_ok(load: float) -> bool:
        return within(*peak_hotspot_at_load(
            load, params, ambient_C, duration_h, initial_state, constants, dt_s
        ))

    # The unit may already be over its limit at the lowest load considered.
    if not point_ok(low):
        hs, oil = peak_hotspot_at_load(
            low, params, ambient_C, duration_h, initial_state, constants, dt_s
        )
        return LoadingEnvelope(
            k_max_pu=float("nan"), k_max_conservative_pu=float("nan"),
            duration_h=duration_h, limits=limits, limiting_constraint="infeasible",
            peak_hotspot_C=hs, peak_top_oil_C=oil, confidence=confidence,
            nameplate_MVA=nameplate_MVA, feasible=False,
            notes=("The unit exceeds its limit even at the lowest load searched "
                   f"({low:.2f} pu). Reduce load or improve cooling.",),
        )

    search_limited = point_ok(high)
    if search_limited:
        k_max = high
        notes.append(
            f"The limit is not binding up to the search ceiling of {high:.2f} pu, so "
            f"{high:.2f} pu is a search bound and not a thermal one. Other constraints "
            f"(bushings, tap-changer, cables, protection settings) will bind first and "
            f"are outside this model."
        )
    else:
        lo, hi = low, high
        while hi - lo > tolerance_pu:
            mid = 0.5 * (lo + hi)
            if point_ok(mid):
                lo = mid
            else:
                hi = mid
        k_max = lo

    peak_hs, peak_oil = peak_hotspot_at_load(
        k_max, params, ambient_C, duration_h, initial_state, constants, dt_s
    )

    # Which limit actually binds? Bisection converges from BELOW, so at k_max
    # the binding limit is still satisfied -- by a margin set by the bisection
    # tolerance, not by anything physical. Comparing peaks against limits at
    # k_max therefore mis-attributes whenever that margin exceeds the
    # comparison tolerance. Probe just ABOVE k_max instead and see which limit
    # gives way first; that is the question "which constraint binds?" actually
    # means.
    if search_limited:
        constraint: Literal["hot-spot", "top-oil", "search-bound", "infeasible"] = "search-bound"
    else:
        over_hs, over_oil = peak_hotspot_at_load(
            k_max + max(tolerance_pu, 1e-3), params, ambient_C, duration_h,
            initial_state, constants, dt_s,
        )
        hs_excess = (over_hs - limits.hotspot_limit_C) / limits.hotspot_limit_C
        oil_excess = (
            (over_oil - limits.top_oil_limit_C) / limits.top_oil_limit_C
            if limits.top_oil_limit_C is not None
            else -np.inf
        )
        constraint = "top-oil" if oil_excess > hs_excess else "hot-spot"

    # -- conservative envelope --------------------------------------------
    if bound is None:
        k_conservative = k_max
        notes.append(
            "No CRLB supplied, so the conservative value equals the point estimate and "
            "carries NO parameter-uncertainty margin. Pass `bound=` to get one."
        )
    else:
        samples = _sample_parameters(params, bound, n_samples, generator)
        if len(samples) < n_samples:
            notes.append(
                f"{len(samples)} of {n_samples} parameter draws were physically valid; "
                f"the rest fell outside the plausibility bounds and were rejected."
            )

        def quantile_ok(load: float) -> bool:
            hs = np.empty(len(samples))
            oil = np.empty(len(samples))
            for i, sample in enumerate(samples):
                hs[i], oil[i] = peak_hotspot_at_load(
                    load, sample, ambient_C, duration_h, initial_state, constants, dt_s
                )
            if float(np.quantile(hs, confidence)) > limits.hotspot_limit_C:
                return False
            if (
                limits.top_oil_limit_C is not None
                and float(np.quantile(oil, confidence)) > limits.top_oil_limit_C
            ):
                return False
            return True

        if not quantile_ok(low):
            k_conservative = float("nan")
            notes.append(
                "Parameter uncertainty alone puts the unit over its limit at the lowest "
                "load searched. The identification is too imprecise to support an "
                "envelope; commission over more load events."
            )
        else:
            lo, hi = low, min(high, k_max)
            if quantile_ok(hi):
                k_conservative = hi
            else:
                while hi - lo > tolerance_pu:
                    mid = 0.5 * (lo + hi)
                    if quantile_ok(mid):
                        lo = mid
                    else:
                        hi = mid
                k_conservative = lo
        notes.append(
            "The uncertainty band covers PARAMETER error only. It excludes model "
            "structural error and ambient/load forecast error, and it is drawn from the "
            "Cramer-Rao bound, which is a LOWER bound on estimator covariance. True "
            "uncertainty is therefore larger than shown."
        )

    if fitted_load_hull is not None:
        hull_low, hull_high = fitted_load_hull
        if k_max > hull_high:
            notes.append(
                f"{k_max:.3f} pu is ABOVE the {hull_low:.2f}-{hull_high:.2f} pu hull the "
                f"parameters were identified over. This is extrapolation, not a validated "
                f"operating rating. An exploratory fit to one published ONAF test read "
                f"6.35 K LOW at 1.60 pu, but that error is not transferable to this unit "
                f"or cooling class. Neither its magnitude nor its sign supplies a safety "
                f"margin. A non-zero load-slope does not validate extrapolation. Keep "
                f"this result in research/shadow mode pending unit-specific validation "
                f"and utility approval; even the fitted hull is not a certified domain."
            )
        if k_max > hull_high and getattr(constants, "x1", 0.0) == 0.0:
            notes.append(
                "The cooling constants in use have x1 = 0, i.e. a FIXED oil exponent. That "
                "assumption has not been validated outside this fitted load hull. "
                "`crlb.load_slope_identifiability` is a conditional precision diagnostic, "
                "not an estimator, a cooling-class transfer rule, or a safety certificate."
            )

    headroom = conservative_headroom = None
    if nameplate_MVA is not None:
        if nameplate_MVA <= 0:
            raise ValueError(f"nameplate_MVA must be > 0, got {nameplate_MVA}")
        # Load is per-unit CURRENT; at constant voltage apparent power scales
        # with it, so MVA = K * nameplate. Stated because it is an assumption.
        headroom = float(max(0.0, k_max - 1.0) * nameplate_MVA * duration_h)
        if np.isfinite(k_conservative):
            conservative_headroom = float(
                max(0.0, k_conservative - 1.0) * nameplate_MVA * duration_h
            )
        notes.append(
            "MVA-hours assume constant voltage, so apparent power scales with per-unit "
            "current."
        )

    return LoadingEnvelope(
        k_max_pu=float(k_max),
        k_max_conservative_pu=float(k_conservative),
        duration_h=float(duration_h),
        limits=limits,
        limiting_constraint=constraint,
        peak_hotspot_C=peak_hs,
        peak_top_oil_C=peak_oil,
        confidence=confidence,
        nameplate_MVA=nameplate_MVA,
        headroom_MVA_h=headroom,
        conservative_headroom_MVA_h=conservative_headroom,
        feasible=True,
        notes=tuple(notes),
    )
