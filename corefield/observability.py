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
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AxialWindingModel",
    "LocationBound",
    "external_location_bound",
    "internal_location_bound",
    "probes_required_for",
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
