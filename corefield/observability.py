# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the PolyForm Noncommercial License 1.0.0.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Use is permitted for noncommercial purposes only, as that term is defined by
# the License. Commercial use requires a separate licence from the copyright
# holder. This is a source-available licence, not an open-source one.
#
# Versions of this file released before 2026-09-02 were published under the
# Apache License 2.0 and remain available under those terms; that grant is not
# and cannot be revoked.

"""Can the hot-spot LOCATION be identified from external measurements?

Short answer: no, and the reason is conservation of energy rather than any
shortcoming of a particular method. This module computes the bound.

WHY THIS MODULE EXISTS
----------------------
A natural extension of the 0-D product is to reconstruct the internal
temperature FIELD and report where in the winding the hot spot actually
sits. It is an attractive idea -- a utility would genuinely value it -- and
it is the stated goal of the withdrawn geometry-PINN branch (see
WITHDRAWN.md).

Before spending months on it, the question worth asking is not "can a neural
network learn this?" but "is the information present in the measurements at
all?". No estimator, classical or learned, can extract information a
measurement does not contain. That is what a Fisher information analysis
answers, and it answers it for every possible method at once.

THE RESULT
----------
Top-oil and bottom-oil temperature are EXACTLY invariant to hot-spot
location. Not approximately -- exactly, to machine precision. Moving the hot
spot from 10 % to 90 % of winding height changes the top-oil reading by
0.0000000 K.

The mechanism is structural. Every external measurement is a function of the
TOTAL heat the winding delivers to the oil. Redistributing that heat along
the winding height changes where the hot spot is without changing how much
heat there is, so the location lies in the exact null space of the external
observation map.

A second-order effect survives: oil heats cumulatively as it rises past the
losses, so the oil profile SHAPE (and hence tank-surface temperature)
carries a weak signal -- about 0.09 K per 10 % of winding height. Against
0.5 K class instrumentation that yields a Cramer-Rao bound of roughly
+/- 40 % of winding height, which is no information at all about a hot spot
that occupies the top 10 %. Resolving location to +/- 5 % would require
sensor noise below 0.044 K, about 11x better than the best practical oil
instrumentation.

With INTERNAL fibre-optic probes the picture inverts completely: two probes
bracketing the expected hot spot give a bound of ~0.33 % of winding height.
The problem is not hard, it simply requires sensors inside the winding --
which can only be installed during manufacture, and which by existing
practice are already placed near 90 % of winding height by design analysis.

MODEL SCOPE AND HONEST LIMITS
-----------------------------
This is a 1-D axial model with a prescribed oil-rise profile, not a CFD
solution. Real windings vary radially, disc to disc, and suffer oil-flow
maldistribution. What makes the conclusion robust despite that simplicity is
that the leading term is an exact cancellation from energy conservation:
ANY model in which the oil circuit is the only thermal path from winding to
the outside world inherits it. A finer model changes the size of the
second-order term, not the fact that the first-order term is zero.

Label (b): the numbers here are an engineering analysis on a simplified
geometry with stated assumptions, not a validated field result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AxialWindingModel",
    "LocationBound",
    "external_location_bound",
    "internal_location_bound",
    "probes_required_for",
    "HandoverCheck",
    "detect_winding_handover",
    "AmbientProbeCheck",
    "check_ambient_consistency",
    "GradientDatumCheck",
    "check_gradient_datum",
]


@dataclass(frozen=True)
class AxialWindingModel:
    """Simplified 1-D axial winding-plus-oil model.

    Attributes
    ----------
    n_nodes : discretisation of the winding height
    bottom_oil_C : oil temperature entering the winding at z = 0 [degC]
    top_oil_C : oil temperature leaving at z = 1 [degC]
    delta_theta_hr_K : rated hot-spot gradient over oil [K]
    bump_width : Gaussian width of the loss concentration, in units of
        winding height [-]
    uniform_base : uniform loss floor relative to the bump peak [-]
    sensor_noise_K : measurement noise standard deviation [K]
    """

    n_nodes: int = 401
    bottom_oil_C: float = 60.0
    top_oil_C: float = 78.0
    delta_theta_hr_K: float = 22.0
    bump_width: float = 0.12
    uniform_base: float = 0.6
    sensor_noise_K: float = 0.5

    @property
    def height(self) -> NDArray[np.float64]:
        """Normalised winding height coordinate z in [0, 1] [-]."""
        return np.linspace(0.0, 1.0, self.n_nodes)

    def loss_shape(self, location: float) -> NDArray[np.float64]:
        """Normalised loss-density profile with a concentration at `location`.

        Normalised so its integral over height is 1 for ANY location. This is
        the physically essential constraint: moving the hot spot redistributes
        loss, it does not create or destroy any.
        """
        z = self.height
        bump = np.exp(-0.5 * ((z - location) / self.bump_width) ** 2)
        profile = self.uniform_base + bump
        return profile / float(np.trapezoid(profile, z))

    def oil_profile(self, location: float) -> NDArray[np.float64]:
        """Oil temperature up the winding [degC].

        Oil heats cumulatively as it rises past the losses, so the profile
        SHAPE depends on where those losses sit -- while its endpoints do
        not. This is the stricter, more favourable-to-detection formulation;
        a well-mixed-oil model would give exactly zero external sensitivity.
        """
        z = self.height
        shape = self.loss_shape(location)
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (shape[1:] + shape[:-1]) * np.diff(z))]
        )
        span = self.top_oil_C - self.bottom_oil_C
        return self.bottom_oil_C + span * cumulative / cumulative[-1]

    def winding_profile(self, location: float) -> NDArray[np.float64]:
        """Winding temperature up the height [degC]."""
        return self.oil_profile(location) + self.delta_theta_hr_K * self.loss_shape(location)

    def external_observations(self, location: float) -> NDArray[np.float64]:
        """Every channel a utility can read without opening the transformer.

        Top-oil, bottom-oil, tank surface (approximated by the mixed-oil
        mean) and the winding-average a WTI replica effectively integrates.
        """
        z = self.height
        oil = self.oil_profile(location)
        winding = self.winding_profile(location)
        return np.array(
            [
                oil[-1],
                oil[0],
                float(np.trapezoid(oil, z)),
                float(np.trapezoid(winding, z)),
            ]
        )

    def internal_observations(
        self, location: float, probe_positions: Sequence[float]
    ) -> NDArray[np.float64]:
        """Fibre-optic probe readings at known normalised heights [degC]."""
        return np.interp(
            np.asarray(probe_positions, dtype=np.float64),
            self.height,
            self.winding_profile(location),
        )


@dataclass(frozen=True)
class LocationBound:
    """Cramer-Rao bound on hot-spot location.

    Attributes
    ----------
    std_percent_of_height : lower bound on the standard deviation of any
        unbiased location estimate, as a percentage of winding height.
        `inf` means the location is not identifiable at all.
    max_sensitivity_K : largest |d(reading)/d(location)| across channels,
        in K per unit of winding height. Zero means exact unobservability.
    per_channel_sensitivity_K : sensitivity of each channel
    noise_needed_for_5pct_K : sensor noise that would be required to resolve
        location to +/- 5 % of height, or `inf` if no noise level suffices
    """

    std_percent_of_height: float
    max_sensitivity_K: float
    per_channel_sensitivity_K: NDArray[np.float64]
    noise_needed_for_5pct_K: float

    @property
    def is_identifiable(self) -> bool:
        """Whether the location can be pinned to better than +/- 10 % of height."""
        return self.std_percent_of_height < 10.0


def _bound(
    observe: Callable[[float], NDArray[np.float64]],
    location: float,
    sensor_noise_K: float,
    step: float = 1e-4,
) -> LocationBound:
    """CRLB on a scalar location parameter from an observation map."""
    derivative = (observe(location + step) - observe(location - step)) / (2.0 * step)
    fisher = float(derivative @ derivative) / sensor_noise_K**2
    if fisher <= 1e-30:
        std = float("inf")
        needed = float("inf")
    else:
        std = float(np.sqrt(1.0 / fisher) * 100.0)
        # Noise scales the bound linearly: std ~ noise / |d|.
        needed = float(sensor_noise_K * 5.0 / std)
    return LocationBound(
        std_percent_of_height=std,
        max_sensitivity_K=float(np.max(np.abs(derivative))),
        per_channel_sensitivity_K=derivative,
        noise_needed_for_5pct_K=needed,
    )


def external_location_bound(
    model: AxialWindingModel | None = None, location: float = 0.90
) -> LocationBound:
    """CRLB on hot-spot location from EXTERNAL measurements only.

    Parameters
    ----------
    model : the axial winding model
    location : true hot-spot location, as a fraction of winding height.
        0.90 is the value existing practice assumes when placing probes.

    Returns
    -------
    LocationBound. Expect ~40 % of winding height -- i.e. no useful
    information -- with top-oil and bottom-oil contributing exactly nothing.
    """
    model = model or AxialWindingModel()
    return _bound(model.external_observations, location, model.sensor_noise_K)


def internal_location_bound(
    probe_positions: Sequence[float],
    model: AxialWindingModel | None = None,
    location: float = 0.90,
) -> LocationBound:
    """CRLB on hot-spot location from fibre-optic probes at known heights.

    Two probes bracketing the expected hot spot are enough to reach ~0.33 %
    of winding height. The problem is not hard once a sensor is inside the
    winding -- which is precisely the constraint that makes it a
    manufacturer-facing capability rather than a retrofit one.
    """
    model = model or AxialWindingModel()
    positions = np.asarray(probe_positions, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("probe_positions must be a non-empty 1-D sequence")
    if np.any(positions < 0.0) or np.any(positions > 1.0):
        raise ValueError("probe positions are normalised heights and must lie in [0, 1]")
    return _bound(
        lambda loc: model.internal_observations(loc, positions),
        location,
        model.sensor_noise_K,
    )


def probes_required_for(
    target_percent: float,
    model: AxialWindingModel | None = None,
    location: float = 0.90,
    max_probes: int = 8,
) -> int:
    """Fewest evenly-spread probes over the top half reaching `target_percent`.

    Parameters
    ----------
    target_percent : desired location accuracy, as a percentage of height
    model : the axial winding model
    location : true hot-spot location [-]
    max_probes : search limit

    Returns
    -------
    Probe count, or -1 if `max_probes` is insufficient.
    """
    model = model or AxialWindingModel()
    for count in range(1, max_probes + 1):
        positions = np.linspace(0.5, 1.0, count + 2)[1:-1] if count > 1 else np.array([location])
        if internal_location_bound(positions, model, location).std_percent_of_height <= target_percent:
            return count
    return -1


# --------------------------------------------------------------------------
# Detecting that the hot spot has moved between windings
#
# A transformer with two instrumented main windings does not necessarily keep
# its hottest point in the same one. On a published 400 MVA ONAF unit the hot
# spot moves from the 120 kV winding to the 410 kV winding between 1.00 and
# 1.29 pu, because the two are cooled differently and their gradients grow at
# different rates.
#
# That matters because the governing hot spot -- the maximum over windings --
# is then not a single physical location, and its apparent load exponent is a
# blend of two. Fitting a power law through the handover fits a change of
# MEASUREMENT LOCATION rather than of physics. On that unit, doing so returned
# an answer 8 K worse than making no correction at all, and it did so
# confidently: nothing in the fit complained.
#
# The signature is specific. Below the handover the series follows one
# winding's exponent, above it the other's, and across the crossover the
# maximum grows more slowly than either -- because the winding taking over
# starts from below. So the local exponent DIPS in the crossing interval and
# recovers after it. A single winding tracked throughout gives a local exponent
# that varies smoothly and does not dip.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HandoverCheck:
    """Whether a load-versus-gradient series crosses a winding handover.

    Attributes
    ----------
    detected : whether a handover signature was found
    load_pu : load at the centre of the suspect interval [pu], or None
    local_exponents : the exponent implied by each consecutive pair [-]
    note : what was found and what to do about it
    """

    detected: bool
    load_pu: float | None
    local_exponents: tuple[float, ...]
    note: str


def detect_winding_handover(
    load_pu: NDArray[np.float64],
    gradient_K: NDArray[np.float64],
    *,
    dip_fraction: float = 0.75,
) -> HandoverCheck:
    """Flag a hot-spot-to-top-oil gradient series that crosses windings.

    Parameters
    ----------
    load_pu : load at each observation [pu], strictly positive
    gradient_K : hot-spot rise above top oil at each load [K], strictly positive
    dip_fraction : an interval is suspect when its local exponent falls below
        this fraction of BOTH neighbours. **(c)** Judgement, not a measured
        constant. 0.75 flags the published 400 MVA case, whose crossing
        interval sits at 0.61 of its lower neighbour, while leaving a
        single-winding series whose exponents rise monotonically untouched.

    Returns
    -------
    HandoverCheck

    Notes
    -----
    Needs at least four observations: three intervals, so that a middle one has
    a neighbour on each side. With fewer, a handover cannot be distinguished
    from an exponent that is simply changing, and the check says so rather than
    guessing.

    This detects a handover; it does not repair one. The repair is to track a
    single winding throughout, which requires sensors in both.
    """
    K = np.asarray(load_pu, dtype=np.float64).ravel()
    g = np.asarray(gradient_K, dtype=np.float64).ravel()
    if K.shape != g.shape:
        raise ValueError(f"load_pu shape {K.shape} != gradient_K shape {g.shape}")
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(g))):
        raise ValueError("load_pu and gradient_K must be finite")
    if np.any(K <= 0.0) or np.any(g <= 0.0):
        raise ValueError("load_pu and gradient_K must be strictly positive")

    order = np.argsort(K)
    K, g = K[order], g[order]
    if np.any(np.diff(K) <= 0.0):
        raise ValueError("load_pu contains duplicate values; cannot form intervals")

    if K.size < 4:
        return HandoverCheck(
            False, None, (),
            f"Only {K.size} observations. A handover needs at least four to be "
            f"distinguished from an exponent that is simply changing with load. "
            f"Not checked -- which is not the same as not present.",
        )

    exponents = np.log(g[1:] / g[:-1]) / np.log(K[1:] / K[:-1])
    for i in range(1, exponents.size - 1):
        left, mid, right = exponents[i - 1], exponents[i], exponents[i + 1]
        if mid < dip_fraction * left and mid < dip_fraction * right:
            centre = float(np.sqrt(K[i] * K[i + 1]))
            return HandoverCheck(
                True, centre, tuple(float(e) for e in exponents),
                f"Handover signature near {centre:.2f} pu: the local exponent dips to "
                f"{mid:.2f} between neighbours of {left:.2f} and {right:.2f}. The "
                f"governing hot spot is most likely moving between windings here, so "
                f"this series is not one physical location. Fitting a load exponent "
                f"through it fits a change of measurement location, not of physics. "
                f"Track a single winding across the whole range instead.",
            )

    return HandoverCheck(
        False, None, tuple(float(e) for e in exponents),
        f"No handover signature. Local exponents {np.round(exponents, 2).tolist()} "
        f"vary without a dip, consistent with one physical location throughout.",
    )


# --------------------------------------------------------------------------
# Is the ambient channel telling the truth?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AmbientProbeCheck:
    """Whether the ambient channel is consistent with the oil it should explain.

    Attributes
    ----------
    suspect : whether an offset or a stage-dependence exceeded tolerance
    mean_offset_K : mean of (implied ambient - measured ambient) over the
        quasi-steady samples [K]. Positive means the probe reads COLD relative
        to the oil it is supposed to explain
    per_stage_offset_K : {cooling stage: mean offset [K]}, empty when no stage
        channel was supplied
    stage_spread_K : largest difference between any two per-stage offsets [K],
        or None when fewer than two stages carry quasi-steady samples
    n_quasi_steady : how many samples the test actually used
    note : what was found and what it does and does not mean
    """

    suspect: bool
    mean_offset_K: float
    per_stage_offset_K: Mapping[int, float]
    stage_spread_K: float | None
    n_quasi_steady: int
    note: str


def check_ambient_consistency(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64],
    top_oil_C: NDArray[np.float64],
    params: "ThermalParams",
    *,
    constants: "CoolingConstants | None" = None,
    cooling_stage: NDArray[np.int_] | None = None,
    settle_factor: float = 3.0,
    load_tol_pu: float = 0.02,
    tolerance_K: float = 1.0,
    min_samples: int = 30,
) -> AmbientProbeCheck:
    """Invert the steady-state oil model for ambient and compare with the probe.

    At quasi-steady the model says top oil sits `steady_top_oil_rise(K)` above
    ambient. Rearranged, the record implies an ambient temperature. Comparing
    that against the measured ambient isolates the ambient channel, which every
    other calculation in this package trusts without ever testing it.

    The method is due to L. Paulhiac of EDF, given in correspondence.

    Parameters
    ----------
    time_s : uniformly spaced sample times [s]
    load_pu : per-unit load current at each sample [pu]
    ambient_C : measured ambient temperature [degC]
    top_oil_C : measured top-oil temperature [degC]
    params : identified thermal parameters for this unit
    constants : cooling-class constants; defaults to the ONAF medium/large set
    cooling_stage : cooling-stage label per sample, or None. Supplying it is
        what makes fan recirculation onto the probe detectable
    settle_factor : how many oil time constants the load must have been steady
        for a sample to count as quasi-steady. **(c)** Judgement. Three time
        constants leaves about 5 % of a step uncompleted
    load_tol_pu : the load must stay within this band over the settling window
        for the oil to be treated as settled [pu]
    tolerance_K : offsets larger than this in magnitude are flagged [K]
    min_samples : below this many quasi-steady samples the test declines to
        report rather than reporting noise

    Returns
    -------
    AmbientProbeCheck

    Notes
    -----
    **This is a flag, never a correction.** The offset absorbs EVERY
    steady-state error in the model, not only the probe: a wrong loss ratio, a
    wrong oil exponent, solar gain on the tank, or a rated oil rise identified
    on a fouled cooler will all land in it. A non-zero offset says the record
    and the model disagree at steady state; it does not say which is wrong.
    Do not adjust the ambient channel with this number.

    The stage-dependent part is the sharper signal, and it is the one that is
    unavailable any other way. A genuine model error is a property of the
    physics and does not know which fans are running, so it lands roughly
    equally on every cooling stage. An ambient probe sitting in the cooler
    exhaust does know: its reading shifts when the fans start. A large
    `stage_spread_K` against a small `mean_offset_K` points at the probe
    rather than at the model.

    Direction matters for safety. A probe reading HIGH -- in the exhaust, in
    the sun, against a warm wall -- makes the identified rated oil rise too
    SMALL, which makes the loading envelope too generous. That is the unsafe
    direction, and it presents here as a negative `mean_offset_K`.
    """
    from .iec60076_7 import ONAF_MEDIUM_LARGE_POWER, steady_top_oil_rise

    if constants is None:
        constants = ONAF_MEDIUM_LARGE_POWER

    t = np.asarray(time_s, dtype=np.float64).ravel()
    K = np.asarray(load_pu, dtype=np.float64).ravel()
    amb = np.asarray(ambient_C, dtype=np.float64).ravel()
    oil = np.asarray(top_oil_C, dtype=np.float64).ravel()
    for name, arr in (("load_pu", K), ("ambient_C", amb), ("top_oil_C", oil)):
        if arr.shape != t.shape:
            raise ValueError(f"{name} shape {arr.shape} != time_s shape {t.shape}")
    if t.size < 2:
        raise ValueError("time_s must have at least two samples")
    dt = float(np.diff(t).mean())
    if dt <= 0.0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-9):
        raise ValueError("time_s must be uniformly spaced and increasing")
    if cooling_stage is not None:
        stage = np.asarray(cooling_stage).ravel()
        if stage.shape != t.shape:
            raise ValueError(
                f"cooling_stage shape {stage.shape} != time_s shape {t.shape}"
            )
        stage = stage.astype(np.int_)
    else:
        stage = None

    # A sample is quasi-steady when the load has been flat for long enough that
    # the oil has effectively caught up. That is the physical condition, so it
    # is tested directly rather than through a slope threshold on the oil.
    window = int(round(settle_factor * params.tau_o_min * 60.0 / dt))
    if window < 1:
        window = 1
    finite = np.isfinite(K) & np.isfinite(amb) & np.isfinite(oil)
    settled = np.zeros(t.size, dtype=bool)
    for i in range(window, t.size):
        seg = K[i - window: i + 1]
        if not np.all(finite[i - window: i + 1]):
            continue
        if float(np.nanmax(seg) - np.nanmin(seg)) <= load_tol_pu:
            settled[i] = True

    n = int(settled.sum())
    if n < min_samples:
        return AmbientProbeCheck(
            False, float("nan"), {}, None, n,
            f"Only {n} quasi-steady samples against a minimum of {min_samples}. "
            f"The load never holds within {load_tol_pu:.2f} pu for "
            f"{settle_factor:g} oil time constants ({settle_factor * params.tau_o_min:.0f} "
            f"min), so the steady-state inversion has nowhere to stand. NOT CHECKED "
            f"-- which is not the same as the ambient channel being sound.",
        )

    implied = oil[settled] - steady_top_oil_rise(K[settled], params, constants)
    offset = implied - amb[settled]
    mean_offset = float(np.mean(offset))

    per_stage: dict[int, float] = {}
    spread: float | None = None
    if stage is not None:
        for s in sorted(set(int(v) for v in stage[settled])):
            m = stage[settled] == s
            if int(m.sum()) >= min_samples:
                per_stage[s] = float(np.mean(offset[m]))
        if len(per_stage) >= 2:
            values = list(per_stage.values())
            spread = float(max(values) - min(values))

    offset_suspect = abs(mean_offset) > tolerance_K
    spread_suspect = spread is not None and spread > tolerance_K
    suspect = offset_suspect or spread_suspect

    direction = (
        "reads WARM relative to the oil, which biases the identified rated oil "
        "rise LOW and the loading envelope HIGH -- the unsafe direction"
        if mean_offset < 0 else
        "reads COOL relative to the oil, which biases the identified rated oil "
        "rise HIGH and the loading envelope LOW -- the conservative direction"
    )

    if not suspect:
        note = (
            f"Consistent. Over {n} quasi-steady samples the implied ambient sits "
            f"{mean_offset:+.2f} K from the measured one, inside the {tolerance_K:.2f} K "
            f"tolerance"
            + (f", and the spread across cooling stages is {spread:.2f} K"
               if spread is not None else "")
            + ". This does not validate the probe; it says the record and the model "
              "do not disagree at steady state by more than the tolerance."
        )
    elif spread_suspect and abs(mean_offset) <= tolerance_K:
        note = (
            f"SUSPECT, and it points at the probe. Over {n} quasi-steady samples the "
            f"mean offset is only {mean_offset:+.2f} K, but it differs by {spread:.2f} K "
            f"between cooling stages ({', '.join(f'{s}: {v:+.2f} K' for s, v in sorted(per_stage.items()))}). "
            f"A model error does not know which fans are running; an ambient probe in "
            f"the cooler exhaust does. Check the probe siting before trusting a fit "
            f"from this record."
        )
    else:
        note = (
            f"SUSPECT. Over {n} quasi-steady samples the implied ambient sits "
            f"{mean_offset:+.2f} K from the measured one, beyond the {tolerance_K:.2f} K "
            f"tolerance: the probe {direction}"
            + (f". Spread across cooling stages is {spread:.2f} K"
               if spread is not None else "")
            + ". This offset absorbs every steady-state model error as well as the "
              "probe -- a wrong loss ratio, a wrong exponent, solar gain, or a rated "
              "oil rise identified on a fouled cooler. It is a flag for "
              "investigation, not a correction to apply."
        )

    return AmbientProbeCheck(suspect, mean_offset, per_stage, spread, n, note)


# --------------------------------------------------------------------------
# Do the winding and oil channels share a datum?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GradientDatumCheck:
    """Whether the hot-spot and top-oil channels are referenced to each other.

    Attributes
    ----------
    suspect : whether a material fraction of the record has the winding
        reading COLDER than the top oil
    negative_fraction : fraction of quasi-steady samples with hot spot below
        top oil [-]
    n_quasi_steady : samples the test used
    offset_K : fitted datum offset C [K], or None when not estimated. Positive
        means the winding channel reads low relative to the oil channel
    rated_gradient_K : fitted rated gradient once the offset is allowed [K].
        An EXTRAPOLATION whenever the record stops well below 1.00 pu, and on a
        light-load record it is barely constrained at all
    exponent : fitted winding exponent once the offset is allowed [-]
    crossing_load_pu : the load at which the fitted gradient equals the offset,
        so the winding channel crosses the oil channel [pu], or None when not
        estimated. **This is the robust quantity on a light-load record.** It is
        `(C / delta_theta_hr) ** (1 / y)`, a ratio, and unlike its three
        ingredients it barely moves when the exponent is wrong. Trust it in
        preference to `rated_gradient_K` whenever the hull stops short of rated
    rmse_with_offset_K, rmse_without_offset_K : residual of the two forms [K]
    note : what was found and what it means for identification
    """

    suspect: bool
    negative_fraction: float
    n_quasi_steady: int
    offset_K: float | None
    rated_gradient_K: float | None
    exponent: float | None
    crossing_load_pu: float | None
    rmse_with_offset_K: float | None
    rmse_without_offset_K: float | None
    note: str


def check_gradient_datum(
    load_pu: NDArray[np.float64],
    top_oil_C: NDArray[np.float64],
    hotspot_C: NDArray[np.float64],
    *,
    negative_fraction_threshold: float = 0.05,
    load_step_tol_pu: float = 0.01,
    min_samples: int = 50,
) -> GradientDatumCheck:
    """Flag a record whose winding channel reads colder than its oil channel.

    The IEC form sets hot spot = top oil + `delta_theta_hr * K**y` with both
    parameters positive, so the model **cannot produce a negative gradient at
    any parameter value**. A record that shows one is not describable by the
    model, and fitting it anyway drives `delta_theta_hr` to a bound or absorbs
    a constant datum error into a rated parameter.

    The usual cause is not a fault, and specifically **it is not a calibration
    offset**. A fibre probe sits at a fixed height inside the winding while the
    top-oil channel is taken at the top of the oil, so the two sensors sit at
    different heights in the same oil column and see different local oil
    temperatures. That height difference contributes a constant C which the
    model has no term for:

        measured gradient = delta_theta_hr * K**y  -  C

    **(a) Confirmed on a public record.** On the SINTEF DynaLoad 40 MVA ONAN
    dataset (Zenodo 10.5281/zenodo.17223516, CC-BY-4.0) the measured deficit is
    not constant: it shrinks monotonically with load and crosses zero inside the
    measured hull, which a calibration offset cannot do. Fitted C is 8.93-10.41 K
    across four winding channels against a measured mean top-minus-bottom oil
    span of 10.37 K, and channels known to sit lower in the winding show a larger
    C. The quantity absorbed is the oil column, not an instrument error.

    Parameters
    ----------
    load_pu : per-unit load current at each sample [pu], strictly positive
    top_oil_C : measured top-oil temperature [degC]
    hotspot_C : measured winding hot-spot temperature [degC]
    negative_fraction_threshold : flag when more than this fraction of
        quasi-steady samples show a negative gradient [-]. **(c)** Judgement.
        A few negative samples are noise around a small gradient at light load;
        5 % is not.
    load_step_tol_pu : a sample counts as quasi-steady when the load moved less
        than this from the previous sample [pu]. The gradient relation is a
        steady-state statement
    min_samples : below this many quasi-steady samples the offset is not
        estimated, because a three-parameter fit on less is not informative

    Returns
    -------
    GradientDatumCheck

    Notes
    -----
    **This is a flag and a diagnosis, never a correction.** The offset is not
    subtracted from anyone's data. A fitted C absorbs every constant difference
    between the two channels, including genuine probe placement, calibration
    error and a hot spot that simply is not where the probe is. Reporting it
    tells an engineer where to look; applying it would invent a measurement.

    The rated gradient reported here is an EXTRAPOLATION whenever the record's
    load hull stops well below 1.00 pu, and must be labelled as one.
    """
    K = np.asarray(load_pu, dtype=np.float64).ravel()
    oil = np.asarray(top_oil_C, dtype=np.float64).ravel()
    hot = np.asarray(hotspot_C, dtype=np.float64).ravel()
    if not (K.shape == oil.shape == hot.shape):
        raise ValueError(
            f"load_pu {K.shape}, top_oil_C {oil.shape} and hotspot_C {hot.shape} "
            f"must have the same shape"
        )
    if K.size < 2:
        raise ValueError("need at least two samples")

    gradient = hot - oil
    # The first sample has no predecessor, so it has no evidence of being
    # settled. Prepending its own value would fabricate a zero step and count
    # it as quasi-steady on an assumption the record does not support.
    step = np.full(K.size, np.inf)
    step[1:] = np.abs(np.diff(K))
    usable = (
        np.isfinite(gradient) & np.isfinite(K) & (K > 0.0) & (step < load_step_tol_pu)
    )
    n = int(usable.sum())
    if n == 0:
        return GradientDatumCheck(
            False, float("nan"), 0, None, None, None, None, None, None,
            "No quasi-steady samples: the load never holds still long enough for the "
            "steady-state gradient relation to apply. NOT CHECKED.",
        )

    Kq, gq = K[usable], gradient[usable]
    negative_fraction = float(np.mean(gq < 0.0))

    if negative_fraction <= negative_fraction_threshold:
        return GradientDatumCheck(
            False, negative_fraction, n, None, None, None, None, None, None,
            f"Consistent. {100 * negative_fraction:.1f} % of {n} quasi-steady samples "
            f"show the winding below the top oil, within the "
            f"{100 * negative_fraction_threshold:.0f} % tolerance. The two channels "
            f"appear to share a datum.",
        )

    offset = rated = exponent = crossing = rmse_with = rmse_without = None
    if n >= min_samples:
        from scipy.optimize import least_squares

        with_offset = least_squares(
            lambda p: p[0] * Kq ** p[1] - p[2] - gq,
            [max(float(np.ptp(gq)), 1.0), 1.6, 1.0],
            bounds=([1e-3, 0.2, -50.0], [500.0, 3.0, 100.0]),
        )
        rated, exponent, offset = (float(v) for v in with_offset.x)
        rmse_with = float(np.sqrt(np.mean(with_offset.fun**2)))
        # The load at which the fitted gradient cancels the offset, so the
        # winding channel crosses the oil channel. Unlike rated, exponent and
        # offset individually, this ratio is well determined on a light-load
        # record: it moves by a few per cent across the whole plausible
        # exponent range, where the rated gradient moves by a factor of two or
        # more. Only defined for a positive offset; a negative one means the
        # winding never reads low and there is nothing to cross.
        if offset > 0.0:
            crossing = float((offset / rated) ** (1.0 / exponent))

        without = least_squares(
            lambda p: p[0] * Kq ** p[1] - gq,
            [max(float(np.ptp(gq)), 1.0), 1.6],
            bounds=([1e-3, 0.2], [500.0, 3.0]),
        )
        rmse_without = float(np.sqrt(np.mean(without.fun**2)))

    head = (
        f"SUSPECT. {100 * negative_fraction:.1f} % of {n} quasi-steady samples have the "
        f"winding hot spot BELOW the top oil. The IEC form cannot produce a negative "
        f"gradient at any positive parameter value, so this record is not describable "
        f"by the model as it stands."
    )
    if offset is None:
        return GradientDatumCheck(
            True, negative_fraction, n, None, None, None, None, None, None,
            head + f" Too few quasi-steady samples ({n} < {min_samples}) to estimate the "
                   f"offset.",
        )

    improved = rmse_without - rmse_with
    if crossing is None:
        crossing_note = ""
    elif crossing <= float(Kq.max()):
        crossing_note = (
            f" The channels cross at K = {crossing:.3f} pu, which is INSIDE this "
            f"record's load hull and is therefore measured rather than extrapolated. "
            f"Prefer it to the rated gradient: the crossing is a ratio and survives a "
            f"wrong exponent, while the rated gradient does not."
        )
    else:
        crossing_note = (
            f" The fit puts the crossing at K = {crossing:.3f} pu, ABOVE this record's "
            f"maximum of {float(Kq.max()):.3f} pu, so the winding never actually "
            f"overtakes the oil anywhere in the data and the crossing is itself an "
            f"extrapolation."
        )
    return GradientDatumCheck(
        True, negative_fraction, n, offset, rated, exponent, crossing,
        rmse_with, rmse_without,
        head
        + f" Allowing a constant datum offset C fits the same data far better: "
          f"residual {rmse_without:.2f} -> {rmse_with:.2f} K for C = {offset:.2f} K, "
          f"rated gradient {rated:.1f} K, exponent {exponent:.2f}. An offset of that "
          f"size is the signature of a winding probe and a top-oil probe sitting at "
          f"different heights in the oil column, which is a placement property and "
          f"NOT a calibration error. Identify the gradient branch from this record "
          f"and the offset is absorbed into the rated parameter instead. The offset "
          f"is reported, never subtracted."
        + crossing_note
        + (f" Note the improvement is only {improved:.2f} K, so the offset reading is "
           f"weakly supported here." if improved < 0.5 else "")
    )
