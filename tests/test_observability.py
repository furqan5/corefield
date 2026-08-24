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

"""Hot-spot LOCATION observability: the negative result, pinned.

These tests exist to stop the 2D/3D field-reconstruction idea being revived
on optimism. If someone later believes external sensors can locate a hot
spot, this suite is the thing that has to be argued with -- and the
argument has to be a physical one, not a better optimiser.
"""

from __future__ import annotations

import numpy as np
import pytest

from corefield.observability import (
    AxialWindingModel,
    external_location_bound,
    internal_location_bound,
    probes_required_for,
)


def test_top_oil_is_exactly_invariant_to_hot_spot_location():
    """Top-oil does not move AT ALL when the hot spot moves. Machine precision.

    This is the whole argument in one assertion. Every external measurement
    is a function of TOTAL winding loss; moving the hot spot redistributes
    loss without changing its total. The location sits in the exact null
    space of the observation, so no estimator of any kind can recover it.
    """
    model = AxialWindingModel()
    readings = [model.external_observations(z)[0] for z in (0.10, 0.50, 0.90)]
    assert readings[0] == pytest.approx(readings[1], abs=1e-9)
    assert readings[1] == pytest.approx(readings[2], abs=1e-9)


def test_bottom_oil_is_also_exactly_invariant():
    model = AxialWindingModel()
    readings = [model.external_observations(z)[1] for z in (0.10, 0.50, 0.90)]
    assert max(readings) - min(readings) < 1e-9


def test_total_loss_is_conserved_under_relocation():
    """The conservation law the null space follows from."""
    model = AxialWindingModel()
    z = model.height
    totals = [float(np.trapezoid(model.loss_shape(loc), z)) for loc in (0.1, 0.5, 0.9)]
    for total in totals:
        assert total == pytest.approx(1.0, abs=1e-9)


def test_external_measurements_cannot_locate_the_hot_spot():
    """CRLB from external channels is ~40 % of winding height: no information.

    The winding is only 100 % tall. A bound of +/-40 % on a hot spot that
    occupies the top 10 % is indistinguishable from knowing nothing.
    """
    bound = external_location_bound()
    assert not bound.is_identifiable
    assert bound.std_percent_of_height > 20.0
    # Only the stratification-sensitive channels carry anything at all.
    top_oil, bottom_oil = bound.per_channel_sensitivity_K[0], bound.per_channel_sensitivity_K[1]
    assert abs(top_oil) < 1e-6
    assert abs(bottom_oil) < 1e-6


def test_external_route_would_need_implausible_instrumentation():
    """Resolving location to +/-5 % needs ~0.04 K noise -- ~11x better than practical.

    Quoted so the trade-off is explicit rather than rhetorical: the external
    route is not merely difficult, it is gated behind instrumentation that
    does not exist for oil temperature measurement.
    """
    bound = external_location_bound()
    assert bound.noise_needed_for_5pct_K < 0.1
    assert bound.noise_needed_for_5pct_K < AxialWindingModel().sensor_noise_K / 5


def test_two_internal_probes_solve_the_problem():
    """Two probes bracketing the hot spot: ~0.33 % of winding height.

    The contrast is the point. The problem is not hard -- it is hard from
    OUTSIDE. Inside, it needs no machine learning at all.
    """
    bound = internal_location_bound([0.80, 0.95])
    assert bound.is_identifiable
    assert bound.std_percent_of_height < 1.0


def test_internal_beats_external_by_two_orders_of_magnitude():
    outside = external_location_bound().std_percent_of_height
    inside = internal_location_bound([0.80, 0.95]).std_percent_of_height
    assert outside / inside > 50.0


def test_one_probe_is_already_enough_for_practical_purposes():
    """A single probe at the expected location gives ~1.9 % of height."""
    bound = internal_location_bound([0.90])
    assert bound.std_percent_of_height < 5.0


def test_probe_count_saturates_quickly():
    """Beyond two probes the bound barely improves -- diminishing returns.

    Relevant to a sensor-placement product: the value is in WHERE the probes
    go, not how many there are.
    """
    two = internal_location_bound([0.80, 0.95]).std_percent_of_height
    eight = internal_location_bound(np.linspace(0.5, 1.0, 8)).std_percent_of_height
    assert two / eight < 3.0


def test_probes_required_for_reports_a_small_number():
    assert 1 <= probes_required_for(1.0) <= 4


def test_probe_positions_are_validated():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        internal_location_bound([0.5, 1.5])
    with pytest.raises(ValueError, match="non-empty"):
        internal_location_bound([])
