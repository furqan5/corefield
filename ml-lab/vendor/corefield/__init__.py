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

"""CoreField -- transformer winding hot-spot estimation and dynamic loading envelopes.

Identifies the four IEC 60076-7 thermal parameters (dtheta_or, tau_o,
dtheta_hr, tau_w) from three signals a utility already logs -- load current,
ambient temperature, top-oil temperature -- then runs the identified model in
service to answer the question that carries commercial value: how much extra
load can this specific unit carry, and for how long.

    IEC PROVENANCE: the structure and ONAF constants were checked against
    IEC 60076-7:2018 Edition 2.0 (25 Aug 2026) and match. No standard text
    is reproduced here. Users claiming standards compliance must hold their
    own licensed copy from an authorised distributor.

    FIELD VALIDATION: NONE. Every number this package reproduces was
    generated from synthetic data. No measurement from a real transformer
    has entered it. See LIMITATIONS in README.md.

The production engine is classical nonlinear least squares on the IEC
two-exponential structure. That is a result, not a preference: under
structural mismatch the single-exponential alternatives read the hot spot
several kelvin HIGH at high load, which triggers derating exactly when
capacity is worth the most.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .iec60076_7 import (
    OD_MEDIUM_LARGE_POWER,
    ONAF_MEDIUM_LARGE_POWER,
    ONAN_MEDIUM_LARGE_POWER,
    ONAN_SMALL,
    PARAM_BOUNDS,
    CoolingConstants,
    ThermalParams,
    ThermalTrajectory,
    hotspot_rise,
    hotspot_temperature,
    simulate,
    steady_hotspot_gradient,
    steady_temperatures,
    steady_top_oil_rise,
    top_oil_rise,
)

__all__ = [
    "__version__",
    "CoolingConstants",
    "ONAF_MEDIUM_LARGE_POWER",
    "ONAN_MEDIUM_LARGE_POWER",
    "ONAN_SMALL",
    "OD_MEDIUM_LARGE_POWER",
    "ThermalParams",
    "ThermalTrajectory",
    "PARAM_BOUNDS",
    "simulate",
    "top_oil_rise",
    "hotspot_rise",
    "hotspot_temperature",
    "steady_top_oil_rise",
    "steady_hotspot_gradient",
    "steady_temperatures",
]
