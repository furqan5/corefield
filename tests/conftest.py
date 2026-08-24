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

"""Shared fixtures and the tolerance policy for the regression suite.

TOLERANCE POLICY -- read this before changing any number in these tests.

The published campaign values are quoted to 2 decimal places. A tolerance
tighter than the quoting precision is not a strict test, it is a broken one:
it would fail on the rounding of the reference value itself. So every
comparison uses

    tolerance = max(0.05 * |published| , 0.01)

i.e. the +/-5 % the project brief specifies, with an absolute floor of 0.01
that exists solely to cover 2-dp rounding of the reference. The floor is NOT
a loosening for convenience -- for a published value of 2.59 the 5 % term
(0.13) dominates and the floor never engages. It engages only for values
near zero, where 5 % of 0.02 K would be 0.001 K, far below the precision at
which 0.02 was ever written down.

If a test here fails, the correct response is to investigate, not to widen
this. Every deviation found during the Stage-3 build is documented in
REPRODUCTION.md with its cause.
"""

from __future__ import annotations

import numpy as np
import pytest

from corefield.campaign import day_transfer, run_scenario
from corefield.synthetic import ALL_SCENARIOS

#: Relative tolerance from the project brief.
REL_TOL = 0.05
#: Absolute floor covering 2-dp rounding of the published reference values.
ABS_FLOOR = 0.01


def tolerance(published: float) -> float:
    """Comparison tolerance for a published value. See module docstring."""
    return max(REL_TOL * abs(published), ABS_FLOOR)


def assert_reproduces(actual: float, published: float, label: str) -> None:
    """Assert `actual` reproduces `published` within the documented tolerance."""
    tol = tolerance(published)
    delta = abs(actual - published)
    assert delta <= tol, (
        f"{label}: reproduced {actual:.4f}, published {published:.4f}, "
        f"delta {delta:.4f} exceeds tolerance {tol:.4f}. "
        f"Do NOT widen the tolerance -- investigate and record the cause."
    )


# --------------------------------------------------------------------------
# Session-scoped campaign fixtures
#
# The full campaign is ~180 optimiser fits. Running it once per session and
# sharing the result keeps the suite inside its 3-minute budget with room to
# spare; these fixtures take about 8 s in total.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def day_a_comparison():
    """Models A/B/C fitted on day A, scored on day A (the Stage-B gate)."""
    return day_transfer("A")


@pytest.fixture(scope="session")
def day_b_comparison():
    """Models A/B/C fitted on day A, scored on the unseen day B."""
    return day_transfer("B")


@pytest.fixture(scope="session")
def day_c_comparison():
    """Models A/B/C fitted on day A, scored at 1.30 pu on day C."""
    return day_transfer("C")


@pytest.fixture(scope="session")
def scenario_results() -> dict:
    """Every Stage-C corruption scenario, keyed by name."""
    return {factory().name: run_scenario(factory()) for factory in ALL_SCENARIOS}


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A fixed-seed generator for tests that need one."""
    return np.random.default_rng(0)
