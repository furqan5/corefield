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

"""Models A and B -- the falsified single-exponential candidates.

WHY FALSIFIED CODE LIVES IN THE PACKAGE
---------------------------------------
These two engines lost the Stage-B structural-mismatch gate and are NOT
production code. They are kept, and tested, because the comparison against
them is the commercial argument: at 1.30 pu they read the hot spot several
kelvin HIGH, which triggers derating exactly when spare capacity is worth
the most. A claim that CoreField is better than the simple alternative is
only checkable if the simple alternative is here to check against.

Do not use these to estimate anything. `corefield.estimator.identify` is the
production path.

Structure
---------
Both integrate a single-exponential winding ODE driven by the MEASURED
top-oil signal -- the standard's alternative on-line formulation for when
theta_o is available as an electrical signal:

    tau_w * d(theta_h)/dt = dtheta_hr * K**y - (theta_h - theta_o(t))

    Model A   free {dtheta_hr, tau_w},      y fixed at 2.0
    Model B   free {dtheta_hr, tau_w, y}

Model C -- the production engine -- instead integrates the coupled oil plus
two-branch winding system and identifies four parameters.

How they fail, which is the interesting part
--------------------------------------------
A: least squares chases the two-exponential overshoot hump by crushing
   tau_w,eff to 4.84 min against a true 7, then over-predicts the settled
   peak by +4.03 K on the fitting day and +5.76 K at 1.30 pu.

B: the free exponent absorbs dynamics error rather than finding the
   steady-state exponent -- y_eff = 1.665 against a true 1.30. Effective
   parameters under structural mismatch are not physical parameters, and
   this is the cleanest demonstration of that in the whole campaign.

A note on the handicap, which runs the WRONG way
------------------------------------------------
In the published campaign these two models were driven by the NOISE-FREE
true top-oil series, while Model C had to fit a noisy one. That asymmetry
favours A and B: they receive a perfect input and need only add the winding
gradient. Their failure is therefore measured under conditions kinder than
Model C's, and the gap between them is if anything understated. The legacy
methods reports described this channel as "measured", which was wrong; see
AUDIT.md section 4.5. `drive_is_noise_free` records which was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .iec60076_7 import _rk4_affine_coefficients, _linear_recurrence

__all__ = ["SingleExponentialFit", "simulate_single_exponential", "fit_single_exponential"]


@dataclass(frozen=True)
class SingleExponentialFit:
    """Result of fitting Model A or Model B.

    Attributes
    ----------
    model : "A" or "B"
    delta_theta_hr_eff_K : effective rated gradient [K]. "Effective" is not
        a hedge -- under structural mismatch this is a curve-fitting
        coefficient and NOT the physical parameter of the same name.
    tau_w_eff_min : effective winding time constant [min]
    y_eff : effective winding exponent [-] (2.0 for Model A, fitted for B)
    converged : optimiser status > 0
    drive_is_noise_free : whether the driving top-oil series was the
        noise-free truth (the campaign's choice) or a measured signal
    """

    model: Literal["A", "B"]
    delta_theta_hr_eff_K: float
    tau_w_eff_min: float
    y_eff: float
    converged: bool
    drive_is_noise_free: bool


def simulate_single_exponential(
    top_oil_C: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    delta_theta_hr_K: float,
    tau_w_min: float,
    y: float,
    dt_s: float,
    *,
    load_pu_half: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Integrate the single-exponential winding ODE driven by a top-oil series.

    Parameters
    ----------
    top_oil_C : driving top-oil temperature on the grid [degC]
    load_pu : per-unit load current on the grid [pu]
    delta_theta_hr_K : rated hot-spot rise over top-oil [K]
    tau_w_min : winding time constant [min]
    y : winding exponent [-]
    dt_s : grid step [s]
    load_pu_half : optional analytic half-step load samples [pu]

    Returns
    -------
    Hot-spot temperature trajectory [degC].

    Notes
    -----
    Like the two-exponential model, this ODE is linear in theta_h with a
    fully known forcing D(t) = dtheta_hr * K(t)**y + theta_o(t), so the same
    affine-recurrence fast path applies. The half-step oil value is the
    endpoint average, because a measured series has no analytic midpoint.
    """
    oil = np.asarray(top_oil_C, dtype=np.float64)
    K = np.asarray(load_pu, dtype=np.float64)
    n = oil.size
    if K.shape != oil.shape:
        raise ValueError(f"load_pu shape {K.shape} != top_oil_C shape {oil.shape}")
    if tau_w_min <= 0.0:
        raise ValueError(f"tau_w_min must be > 0, got {tau_w_min!r}")

    Ky_on = K**y
    if load_pu_half is None:
        K_half = np.append(0.5 * (K[:-1] + K[1:]), K[-1])
    else:
        K_half = np.asarray(load_pu_half, dtype=np.float64)
    Ky_half = K_half**y

    oil_half = np.append(0.5 * (oil[:-1] + oil[1:]), oil[-1])

    drive_on = delta_theta_hr_K * Ky_on + oil
    drive_half = delta_theta_hr_K * Ky_half + oil_half

    tau_s = tau_w_min * 60.0
    a, b = _rk4_affine_coefficients(tau_s, dt_s, drive_on, drive_half)
    s0 = float(oil[0] + delta_theta_hr_K * Ky_on[0])
    return _linear_recurrence(a, b, s0, n)


def fit_single_exponential(
    model: Literal["A", "B"],
    top_oil_C: NDArray[np.float64],
    load_pu: NDArray[np.float64],
    hotspot_ref_index: NDArray[np.intp],
    hotspot_ref_C: NDArray[np.float64],
    dt_s: float,
    *,
    load_pu_half: NDArray[np.float64] | None = None,
    loss: str = "linear",
) -> SingleExponentialFit:
    """Fit Model A or Model B to sparse hot-spot reads.

    Bounds and the initial guess match the published campaign exactly, so
    that the reproduced effective parameters (tau_w,eff = 4.84 min for A;
    y_eff = 1.665 for B) are comparable digit for digit.

    Parameters
    ----------
    model : "A" (y fixed at 2.0) or "B" (y free)
    top_oil_C : driving top-oil series [degC]
    load_pu : per-unit load on the grid [pu]
    hotspot_ref_index : grid indices of the calibration reads
    hotspot_ref_C : calibration read values [degC]
    dt_s : grid step [s]
    load_pu_half : optional analytic half-step load samples
    loss : robust loss; "linear" reproduces the campaign

    Returns
    -------
    SingleExponentialFit
    """
    if model not in ("A", "B"):
        raise ValueError(f"model must be 'A' or 'B', got {model!r}")

    idx = np.asarray(hotspot_ref_index, dtype=np.intp)
    target = np.asarray(hotspot_ref_C, dtype=np.float64)

    def trajectory(dthr: float, tw_s: float, y: float) -> NDArray[np.float64]:
        return simulate_single_exponential(
            top_oil_C, load_pu, dthr, tw_s / 60.0, y, dt_s, load_pu_half=load_pu_half
        )

    if model == "A":
        def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
            return trajectory(p[0], p[1], 2.0)[idx] - target

        result = least_squares(
            residual, x0=[15.0, 600.0], bounds=([5.0, 60.0], [60.0, 7200.0]),
            method="trf", loss=loss,
        )
        dthr, tw_s, y_eff = float(result.x[0]), float(result.x[1]), 2.0
    else:
        def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
            return trajectory(p[0], p[1], p[2])[idx] - target

        result = least_squares(
            residual, x0=[15.0, 600.0, 2.0],
            bounds=([5.0, 60.0, 0.8], [60.0, 7200.0, 3.0]),
            method="trf", loss=loss,
        )
        dthr, tw_s, y_eff = float(result.x[0]), float(result.x[1]), float(result.x[2])

    return SingleExponentialFit(
        model=model,
        delta_theta_hr_eff_K=dthr,
        tau_w_eff_min=tw_s / 60.0,
        y_eff=y_eff,
        converged=bool(result.status > 0),
        drive_is_noise_free=True,
    )
