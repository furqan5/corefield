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

"""Fisher information and the Cramer-Rao bound for the four thermal parameters.

Why this module exists
----------------------
The CRLB is the floor no unbiased estimator can beat for a given record. It
converts "our estimator is accurate" -- an unfalsifiable claim -- into "our
estimator is within X of the information-theoretic limit for this data",
which is checkable and which also tells a utility something they can act on:
how many load events they must commission over. A single load event puts a
13.5 % floor under tau_w no method can improve; two events drop it to 4.9 %.
That is a scheduling decision, not an algorithm decision.

Two conventions -- read this before comparing numbers
-----------------------------------------------------
"Efficiency ratio" is used in the literature and in this project's own
history to mean two DIFFERENT quantities:

  FOLDED  mean(|error|) / (0.798 * CRLB)
          For an unbiased Gaussian estimator, E|error| = sqrt(2/pi)*sigma
          = 0.798*sigma. This ratio compares the observed MEAN ABSOLUTE
          error against that folded expectation. It is what the published
          CoreField campaign reported.

  STD     std(estimates) / CRLB
          The textbook definition: the estimator's actual dispersion
          against the bound.

They agree only asymptotically, for an unbiased estimator, with enough
seeds. At 10 seeds they differ noticeably. `efficiency_ratio` therefore
requires you to name the convention rather than guessing, and states which
one produced any number it returns.

Provenance warning on the headline figure
-----------------------------------------
The claim "all four parameters recovered at 0.99-1.02x CRLB" that circulated
in the project's own documents is NOT supported by its data. The 0.99-1.02x
range comes from a TWO-parameter table on the older single-exponential truth
model, and covers 6 of its 8 cells (the other two are 1.13x and 0.68x). The
genuine FOUR-parameter figures, in the folded convention, are 0.84 / 0.82 /
0.82 / 0.91. See AUDIT.md section 5.1. Ratios below 1.0 mean the observed
error came in UNDER the folded expectation, which at 10 seeds is ordinary
sampling variation and is a good result -- the problem was only that a
number from a different experiment was carried across.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .iec60076_7 import (
    ONAF_MEDIUM_LARGE_POWER,
    CoolingConstants,
    ThermalParams,
    _integrate,
    simulate,
)

__all__ = [
    "PARAMETER_NAMES",
    "FOLDED_FACTOR",
    "CRLBResult",
    "fisher_information",
    "cramer_rao_bound",
    "efficiency_ratio",
    "LoadSlopeIdentifiability",
    "load_slope_identifiability",
    "OvershootIdentifiability",
    "overshoot_identifiability",
]

#: Parameter order everywhere in this package.
PARAMETER_NAMES: tuple[str, str, str, str] = (
    "delta_theta_or",
    "tau_o",
    "delta_theta_hr",
    "tau_w",
)

#: E|X| / sigma for a zero-mean Gaussian: sqrt(2/pi).
FOLDED_FACTOR: float = float(np.sqrt(2.0 / np.pi))

#: Relative step for the central-difference Jacobian, matching the campaign.
_JACOBIAN_RSTEP: float = 1e-3


@dataclass(frozen=True)
class CRLBResult:
    """Cramer-Rao bound for the four parameters at a given operating record.

    Attributes
    ----------
    std_absolute : lower bound on estimator standard deviation, in the
        working vector's units: [K, s, K, s]
    std_percent : the same, as a percentage of the true parameter value
    correlation : 4x4 parameter correlation matrix implied by the bound.
        rho(tau_w, tau_o) near zero is what refutes the claim that the
        four-parameter problem is degenerate -- the dense oil record pins
        the oil pair nearly independently of the winding pair.
    covariance : full 4x4 covariance lower bound
    condition : condition number of the Fisher information matrix
    names : parameter names, in order
    """

    std_absolute: NDArray[np.float64]
    std_percent: NDArray[np.float64]
    correlation: NDArray[np.float64]
    covariance: NDArray[np.float64]
    condition: float
    names: tuple[str, str, str, str] = PARAMETER_NAMES

    def as_dict(self) -> dict[str, float]:
        """Relative bounds keyed by parameter name [%]."""
        return {n: float(v) for n, v in zip(self.names, self.std_percent)}

    def report(self) -> str:
        """Human-readable summary."""
        lines = ["Cramer-Rao lower bound (no unbiased estimator can beat this):"]
        for name, pct in zip(self.names, self.std_percent):
            lines.append(f"  {name:>16}: {pct:6.2f} %")
        rho = float(self.correlation[1, 3])
        lines.append(f"  rho(tau_o, tau_w) = {rho:+.3f}")
        lines.append(f"  Fisher matrix condition number: {self.condition:.2e}")
        if self.condition > 1e10:
            lines.append(
                "  WARNING: near-singular Fisher matrix -- this record does not "
                "separate the parameters. Add load events."
            )
        return "\n".join(lines)


def _observation_jacobian(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64],
    params: ThermalParams,
    oil_index: NDArray[np.intp],
    hotspot_index: NDArray[np.intp],
    constants: CoolingConstants,
) -> NDArray[np.float64]:
    """d(observations)/d(parameters), by central differences [obs x 4].

    Observations are stacked as [top-oil at oil_index, hot-spot at
    hotspot_index] -- the same ordering the estimator's residual uses.
    """
    t = np.asarray(time_s, dtype=np.float64)
    dt = float(t[1] - t[0])
    K = np.asarray(load_pu, dtype=np.float64)
    A = np.asarray(ambient_C, dtype=np.float64)
    K_half = np.append(0.5 * (K[:-1] + K[1:]), K[-1])
    A_half = np.append(0.5 * (A[:-1] + A[1:]), A[-1])

    def observe(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        oil, hs = _integrate(
            dtor=float(vector[0]), tauo_s=float(vector[1]),
            dthr=float(vector[2]), tw_s=float(vector[3]),
            x=constants.x, y=constants.y, x1=constants.x1, y1=constants.y1,
            k11=constants.k11,
            k21=constants.k21, k22=constants.k22, R=params.loss_ratio_R,
            K_on=K, K_half=K_half, A_on=A, A_half=A_half, dt=dt, solver="rk4",
        )
        return np.concatenate([oil[oil_index], hs[hotspot_index]])

    p0 = params.as_vector()
    n_obs = oil_index.size + hotspot_index.size
    jacobian = np.zeros((n_obs, 4), dtype=np.float64)
    for j in range(4):
        step = _JACOBIAN_RSTEP * p0[j]
        dp = np.zeros(4)
        dp[j] = step
        jacobian[:, j] = (observe(p0 + dp) - observe(p0 - dp)) / (2.0 * step)
    return jacobian


def fisher_information(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    oil_index: NDArray[np.intp],
    hotspot_index: NDArray[np.intp],
    sigma_K: float,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
) -> NDArray[np.float64]:
    """Fisher information matrix for the four parameters [4 x 4].

    Assumes independent Gaussian measurement noise of standard deviation
    `sigma_K` on BOTH channels. That assumption is the model's, not the
    world's: real sensors drift, spike and correlate. Drift in particular
    is not covered by this bound -- it biases the estimate rather than
    inflating its variance, and a CRLB says nothing about bias.

    Parameters
    ----------
    time_s : uniform simulation grid [s]
    load_pu : per-unit load current on the grid [pu]
    ambient_C : ambient temperature [degC], scalar or per-sample
    params : parameters at which to evaluate the information
    oil_index : grid indices carrying a top-oil measurement
    hotspot_index : grid indices carrying a hot-spot calibration read
    sigma_K : measurement noise standard deviation [K]
    constants : cooling-class constants

    Returns
    -------
    Fisher information matrix, in the working vector's units.
    """
    if sigma_K <= 0.0 or not np.isfinite(sigma_K):
        raise ValueError(f"sigma_K must be finite and > 0, got {sigma_K!r}")

    t = np.asarray(time_s, dtype=np.float64)
    A = np.asarray(ambient_C, dtype=np.float64)
    if A.ndim == 0:
        A = np.full(t.shape, float(A))

    oil_index = np.asarray(oil_index, dtype=np.intp)
    hotspot_index = np.asarray(hotspot_index, dtype=np.intp)
    if hotspot_index.size < 4:
        raise ValueError(
            f"{hotspot_index.size} hot-spot observation(s) cannot support a 4-parameter "
            f"bound: the Fisher matrix is rank-deficient by construction."
        )

    jacobian = _observation_jacobian(
        t, np.asarray(load_pu, dtype=np.float64), A, params,
        oil_index, hotspot_index, constants,
    )
    return (jacobian.T @ jacobian) / (sigma_K**2)


def cramer_rao_bound(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    oil_index: NDArray[np.intp],
    hotspot_index: NDArray[np.intp],
    sigma_K: float,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
) -> CRLBResult:
    """Cramer-Rao lower bound on the four parameters for this record.

    Parameters are as `fisher_information`.

    Returns
    -------
    CRLBResult, with bounds both absolute and as a percentage of the true
    parameter values.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the Fisher matrix is singular -- meaning the record genuinely
        does not identify all four parameters. This is raised rather than
        pseudo-inverted, because a pseudo-inverse would silently return a
        finite-looking bound for an unidentifiable problem.
    """
    information = fisher_information(
        time_s, load_pu, ambient_C, params, oil_index, hotspot_index, sigma_K, constants
    )
    condition = float(np.linalg.cond(information))
    covariance = np.linalg.inv(information)
    variances = np.diag(covariance)
    if np.any(variances <= 0.0):
        raise np.linalg.LinAlgError(
            "Fisher information yields a non-positive variance; the record does not "
            "identify all four parameters."
        )
    std = np.sqrt(variances)
    correlation = covariance / np.outer(std, std)
    return CRLBResult(
        std_absolute=std,
        std_percent=std / params.as_vector() * 100.0,
        correlation=correlation,
        covariance=covariance,
        condition=condition,
    )


def efficiency_ratio(
    estimates: Sequence[ThermalParams] | NDArray[np.float64],
    truth: ThermalParams,
    bound: CRLBResult,
    *,
    convention: Literal["folded", "std"],
) -> dict[str, float]:
    """Estimator efficiency against the CRLB, per parameter.

    Parameters
    ----------
    estimates : the per-seed identified parameter sets
    truth : the true parameters the estimates are compared against. Only
        available on synthetic data -- on real telemetry there is no truth
        and this function does not apply.
    bound : the CRLB for the same record
    convention : which ratio to compute. REQUIRED -- there is no sensible
        default, because the two conventions disagree at realistic seed
        counts and the published CoreField numbers used "folded" while the
        textbook definition is "std". See the module docstring.

        "folded" -> mean(|estimate - truth|) / (0.798 * CRLB)
        "std"    -> std(estimates)           / CRLB

    Returns
    -------
    Ratio per parameter name. 1.0 means the estimator sits exactly on the
    bound under the chosen convention; below 1.0 means it did better than
    the convention's expectation on this sample.
    """
    if convention not in ("folded", "std"):
        raise ValueError(f"convention must be 'folded' or 'std', got {convention!r}")

    if isinstance(estimates, np.ndarray):
        stack = np.asarray(estimates, dtype=np.float64)
    else:
        stack = np.vstack([p.as_vector() for p in estimates])
    if stack.ndim != 2 or stack.shape[1] != 4:
        raise ValueError(f"expected an (n_seeds, 4) array of estimates, got {stack.shape}")
    if stack.shape[0] < 2 and convention == "std":
        raise ValueError("the 'std' convention needs at least 2 seeds")

    truth_vector = truth.as_vector()

    if convention == "folded":
        observed = np.mean(np.abs(stack - truth_vector), axis=0)
        reference = FOLDED_FACTOR * bound.std_absolute
    else:
        observed = np.std(stack, axis=0, ddof=1)
        reference = bound.std_absolute

    return {name: float(o / r) for name, o, r in zip(PARAMETER_NAMES, observed, reference)}


# --------------------------------------------------------------------------
# Identifiability of the load-slope of the oil exponent
#
# The steady oil rise is dtheta_or * g(K)^x(K) with g = (1+R K^2)/(1+R). Under
# a load-dependent exponent x(K) = x0 + x1*(K-1) the two sensitivities are
#
#     d/dx0 = dtheta_or * g^x * ln g
#     d/dx1 = dtheta_or * g^x * ln g * (K - 1)
#
# They differ ONLY by the factor (K-1). The two Jacobian columns are therefore
# exactly collinear at a single load. Load diversity affects rank and
# conditioning, but independent sample count and measurement noise also affect
# precision. Repeating a full-rank design improves the ideal IID bound; it does
# not establish the validity of extrapolation to unobserved loads. (a, algebra)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadSlopeIdentifiability:
    """Conditional precision of a load-dependent oil exponent for a record.

    Attributes
    ----------
    std_x1 : Cramer-Rao standard deviation on the load-slope x1 [per pu]
    correlation_x0_x1 : correlation between the two exponent terms [-]. As this
        approaches 1 the pair becomes indistinguishable.
    load_hull : (min, max) of the load actually present [pu]
    supported : whether the assumed design meets the chosen precision threshold;
        not a validation of model structure or of a loading envelope
    note : plain statement of what the numbers mean
    """

    std_x1: float
    correlation_x0_x1: float
    load_hull: tuple[float, float]
    supported: bool
    note: str


def load_slope_identifiability(
    load_pu: NDArray[np.float64],
    params: ThermalParams,
    sigma_K: float,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    *,
    reference_x1: float | None = None,
    tolerance: float = 0.25,
) -> LoadSlopeIdentifiability:
    """Evaluate a conditional steady-state oil-slope precision bound.

    Parameters
    ----------
    load_pu : the load actually present in the record [pu]. Steady-state
        observations are assumed; this bounds the amplitude information, which
        is where the exponent lives.
    params : thermal parameters, for dtheta_or and the loss ratio
    sigma_K : independent, identically distributed top-oil noise std [K]
    constants : cooling-class constants
    reference_x1 : the slope magnitude to judge the bound against [per pu].
        Positive magnitude, not an estimated slope. If omitted, 0.21 is used
        only with the unchanged ONAF example constants. It is an illustrative
        engineering estimate (b), motivated by an exploratory analysis of
        Nordman and Lahtinen's 2003 ONAF test, DOI 10.1109/TPWRD.2002.807747.
        Interval power-law exponents are not direct measurements of x1 in
        this model. Other constants require an explicit reference; no slope
        magnitude or direction is established for ODAF or ONAN by that test.
    tolerance : positive maximum std_x1 / reference_x1 called supported [-]

    Returns
    -------
    LoadSlopeIdentifiability

    Notes
    -----
    Fits the information for three parameters: oil-rise amplitude, x0, x1.
    The loss ratio and all other model assumptions are treated as known.
    Correlated observations, sensor bias and structural mismatch are excluded.
    A full-rank record can become more precise with more independent samples;
    repeating a rank-deficient design cannot add missing parameter directions.
    Neither result demonstrates safe operation outside the validated domain.

    Returns `supported=False` for inadequate information rather than raising.
    This is a diagnostic; it does not fit x1 or automatically gate the estimator.
    """
    K = np.asarray(load_pu, dtype=np.float64)
    if K.ndim != 1:
        raise ValueError("load_pu must be a one-dimensional array")
    if K.size < 2:
        raise ValueError("load_pu must contain at least two samples")
    if not np.all(np.isfinite(K)):
        raise ValueError("load_pu contains non-finite values")
    if np.any(K < 0.0):
        raise ValueError("load_pu contains negative load")
    if not np.isfinite(sigma_K) or sigma_K <= 0.0:
        raise ValueError(f"sigma_K must be finite and > 0, got {sigma_K!r}")
    if reference_x1 is None:
        if constants != ONAF_MEDIUM_LARGE_POWER:
            raise ValueError(
                "reference_x1 must be supplied for constants other than the default "
                "ONAF example; its illustrative slope must not transfer to another "
                "cooling class or calibration."
            )
        reference_x1 = 0.21  # (b) Historical ONAF sensitivity example, not a unit measurement.
    if not np.isfinite(reference_x1) or reference_x1 <= 0.0:
        raise ValueError("reference_x1 must be a finite positive magnitude [per pu]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and > 0")

    R = params.loss_ratio_R
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g = (1.0 + R * K**2) / (1.0 + R)
        f = g ** (constants.x + constants.x1 * (K - 1.0))
        common = params.delta_theta_or_K * f * np.log(g)
        jacobian = np.column_stack([f, common, common * (K - 1.0)]) / sigma_K
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("load-slope sensitivity is non-finite; check load and noise scales")

    hull = (float(K.min()), float(K.max()))
    # SVD avoids squaring the condition number in J.T @ J. Its numerical-rank
    # check also handles two distinct levels repeated many times: three
    # parameters remain underdetermined regardless of the number of rows. (a)
    try:
        _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=False)
        rank_tolerance = max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if rank < 3:
            raise np.linalg.LinAlgError
        inverse_weighted = vh.T / singular_values
        covariance = inverse_weighted @ inverse_weighted.T
        variances = np.diag(covariance)
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        return LoadSlopeIdentifiability(
            std_x1=float("inf"), correlation_x0_x1=1.0, load_hull=hull, supported=False,
            note=(
                f"Singular or numerically unresolved design over {hull[0]:.2f}-{hull[1]:.2f} "
                f"pu: the three-parameter bound is undetermined. Repeating the same "
                f"rank-deficient load levels cannot add missing parameter directions. "
                f"Use distinct informative levels or independently justified fixed parameters."
            ),
        )

    std = np.sqrt(variances)
    rho = float(np.clip(covariance[1, 2] / (std[1] * std[2]), -1.0, 1.0))
    std_x1 = float(std[2])
    ratio = std_x1 / reference_x1
    supported = ratio <= tolerance

    if supported:
        note = (
            f"Supported: over {hull[0]:.2f}-{hull[1]:.2f} pu the load-slope is bounded to "
            f"{100 * ratio:.0f} % of the supplied/example reference, "
            f"rho(x0,x1) = {rho:.3f}, under the stated IID model assumptions. "
            f"This is a precision diagnostic, not validation of an overload rating."
        )
    else:
        note = (
            f"NOT supported: over {hull[0]:.2f}-{hull[1]:.2f} pu the bound on the "
            f"load-slope is {100 * ratio:.0f} % of the supplied/example reference, with "
            f"rho(x0,x1) = {rho:.3f}. More informative load levels, lower noise, or more "
            f"independent observations can improve this full-rank bound. This result "
            f"does not establish the direction or magnitude of extrapolation error. "
            f"It bounds the load-slope of THIS exponent parameterisation and says "
            f"nothing about whether some other parameterisation of the same record "
            f"is identifiable."
        )
    return LoadSlopeIdentifiability(std_x1, rho, hull, supported, note)


# --------------------------------------------------------------------------
# Identifiability of the overshoot constant k21
#
# k21 sets the amplitude of the fast gradient branch, and with it the overshoot
# that is the entire reason the two-exponential form exists. IEC 60076-7 states
# it "can be defined only if the transformer is equipped with fibre optic
# sensors", and the package holds it at the tabulated value everywhere.
#
# The reason no steady-state record can inform it is structural rather than
# statistical: at steady state the two branches settle to k21*g and (k21-1)*g,
# whose DIFFERENCE is g regardless of k21. The constant cancels exactly. So the
# information about k21 lives in the transient. Informative observation times
# depend on both branch time constants and their amplitudes; a peak time from
# one ONAF example is not a universal commissioning schedule. (a, model algebra)
#
# An exactly steady-state record carries no information. Late but not fully
# settled observations may carry weak information. (a, model algebra)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OvershootIdentifiability:
    """Whether a record's transient content can inform the overshoot constant.

    Attributes
    ----------
    std_k21 : Cramer-Rao standard deviation on k21 [-]
    relative : std_k21 as a fraction of the tabulated k21 [-]
    supported : whether the conditional one-parameter precision threshold is met;
        this does not establish joint identifiability
    note : plain statement of what the numbers mean
    """

    std_k21: float
    relative: float
    supported: bool
    note: str


def overshoot_identifiability(
    time_s: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    ambient_C: NDArray[np.float64] | float,
    params: ThermalParams,
    hotspot_index: NDArray[np.intp],
    sigma_K: float,
    constants: CoolingConstants = ONAF_MEDIUM_LARGE_POWER,
    *,
    tolerance: float = 0.25,
    step: float = 1e-3,
) -> OvershootIdentifiability:
    """Evaluate k21 precision conditional on all other parameters being known.

    Parameters
    ----------
    time_s, load_pu, ambient_C : record in seconds, per-unit current and degC,
        respectively, as for `fisher_information`
    params : thermal parameters to evaluate the sensitivity at
    hotspot_index : indices of the hot-spot observations. k21 is invisible in
        top-oil, so only these carry information about it
    sigma_K : hot-spot measurement noise [K]
    constants : cooling-class constants, evaluated at their tabulated k21
    tolerance : the largest std_k21 / k21 still called supported
    step : relative finite-difference step on k21

    Returns
    -------
    OvershootIdentifiability

    Notes
    -----
    Uses a one-parameter bound: it asks whether the record could pin k21 with
    everything else known. Failure means the chosen precision threshold is not
    met even under those assumptions. Passing is not evidence that a joint fit
    can separate k21, k22 and the winding time constant. The noise model assumes
    independent Gaussian observations and excludes bias and structural error.
    """
    idx = np.asarray(hotspot_index, dtype=np.intp)
    if idx.size == 0:
        return OvershootIdentifiability(
            float("inf"), float("inf"), False,
            "No hot-spot observations. k21 is invisible in top-oil because it "
            "does not enter the oil equation. This measurement set cannot "
            "inform k21 in the implemented model.",
        )
    if not np.isfinite(sigma_K) or sigma_K <= 0.0:
        raise ValueError(f"sigma_K must be finite and > 0, got {sigma_K!r}")

    def hotspot_at(k21: float) -> NDArray[np.float64]:
        c = CoolingConstants(x=constants.x, y=constants.y, k11=constants.k11,
                             k21=k21, k22=constants.k22, name="perturbed",
                             x1=constants.x1, y1=constants.y1)
        return simulate(time_s, load_pu, ambient_C, params, constants=c).hotspot_C[idx]

    k0 = constants.k21
    h = max(step * k0, 1e-4)
    # k21 >= 1 is enforced by CoolingConstants, so step upward from the bound.
    lo = max(k0 - h, 1.0 + 1e-9)
    derivative = (hotspot_at(k0 + h) - hotspot_at(lo)) / (k0 + h - lo)

    information = float(derivative @ derivative) / sigma_K**2
    if information <= 0.0 or not np.isfinite(information):
        return OvershootIdentifiability(
            float("inf"), float("inf"), False,
            "The hot-spot observations are completely insensitive to k21. The "
            "record contains no step this package can see, or every observation "
            "falls long after the transients have settled.",
        )

    std = float(np.sqrt(1.0 / information))
    relative = std / k0
    supported = relative <= tolerance
    if supported:
        note = (
            f"Supported: k21 bounded to {100 * relative:.0f} % of its tabulated "
            f"{k0:g}, with all other parameters held known. The observations "
            f"carry transient sensitivity. This is not joint identification "
            f"or validation of an overload rating."
        )
    else:
        note = (
            f"NOT supported: the bound on k21 is {100 * relative:.0f} % of its "
            f"reference {k0:g}, even with other parameters known. An informative "
            f"window near an overshoot peak depends on this unit's branch time "
            f"constants; there is no universal 40-minute sampling rule. Different "
            f"timing or more independent transient observations may improve precision."
        )
    return OvershootIdentifiability(std, relative, supported, note)
