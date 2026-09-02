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

"""Ingestion of real utility telemetry, and the validation gate before fitting.

WHAT THIS MODULE IS FOR
-----------------------
Everything else in this package has only ever seen synthetic data. This is
the boundary where real measurements enter, and it is deliberately the most
suspicious code in the repository. A historian export arrives with
out-of-order rows, duplicate timestamps, gaps, mixed units, integer-rounded
temperatures, and sometimes no ambient channel at all. Each of those either
gets handled explicitly here or stops the run.

TWO RULES THAT DRIVE THE DESIGN
-------------------------------
1. **Inputs may be interpolated. Observations may not.**
   Load and ambient are *inputs* to the thermal model: the integrator needs
   a value at every grid point, so they are linearly interpolated onto the
   grid and the gaps are reported. Top-oil is an *observation*: interpolating
   it would invent measurements that were never taken and correlate their
   noise, which silently corrupts the least-squares weighting and makes the
   residual RMSE look better than the data deserves. Measured top-oil is
   snapped to the nearest grid point; every other grid point stays NaN.

2. **No ambient, no fit.**
   Ignoring a varying ambient under-predicts the afternoon peak by 3.09 K --
   the one direction a thermal monitor must never err in, and worst exactly
   when ambient maximum coincides with load peak. This module raises rather
   than warns. An hourly public weather feed is sufficient, because ambient
   reaches the winding through a ~75-minute oil low-pass.

THE LOAD HULL
-------------
The validation report states the span of load in the record, because that
span is the region the identified parameters are actually supported over.
Extrapolation beyond it is where the single-exponential models failed by
+5.76 K at 1.30 pu. A record that never exceeds 0.9 pu cannot certify
behaviour at 1.3 pu, regardless of how good the fit looks.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .estimator import HotspotReferences

__all__ = [
    "REQUIRED_COLUMNS",
    "COLUMN_ALIASES",
    "TEMPLATE_COLUMNS",
    "STUCK_CHANNEL_HOURS",
    "AmbientMissingError",
    "ValidationReport",
    "TelemetryFrame",
    "load_telemetry",
    "write_template",
]

#: Canonical column names this module works in.
REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "load_pu", "ambient_C", "top_oil_C")

#: Optional canonical columns.
OPTIONAL_COLUMNS: tuple[str, ...] = ("hotspot_C",)

#: Accepted header spellings, lowercased and stripped of separators. Utility
#: historians name these columns whatever the commissioning engineer typed in
#: 1998, so accept the common variants rather than making a human rename them.
COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date", "ts", "utc", "localtime"),
    "load_pu": ("loadpu", "load", "k", "kpu", "loadfactor", "perunitload", "puload"),
    "current_A": ("currenta", "current", "amps", "amperes", "iload", "loadcurrent", "ia"),
    "ambient_C": ("ambientc", "ambient", "tambient", "ambienttemp", "airtemp", "toa", "ta"),
    "top_oil_C": ("topoilc", "topoil", "oil", "toptemp", "oiltemp", "to", "top_oil_temp"),
    "hotspot_C": ("hotspotc", "hotspot", "hst", "windingtemp", "fibre", "fiber", "wti"),
    "cooling_stage": ("coolingstage", "stage", "coolerstage", "fanstage", "coolingmode",
                      "coolers", "fans", "coolingstep"),
}

#: Columns written by `write_template`, in order, with their units.
TEMPLATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("timestamp", "ISO 8601, e.g. 2026-03-01T00:00:00"),
    ("load_pu", "per-unit load current [pu]; or supply current_A instead"),
    ("ambient_C", "ambient air temperature [degC] - REQUIRED, see notes"),
    ("top_oil_C", "top-oil temperature [degC]"),
    ("hotspot_C", "hot-spot calibration reads [degC] - leave blank except where measured"),
    ("cooling_stage", "cooling stage index, e.g. 1 = fans off, 2 = fans on - see notes"),
)

#: Longest gap in an INPUT channel that will be interpolated across [s].
#: Chosen as roughly the oil low-pass time constant: interpolating ambient
#: across less than that is invisible to the thermal model, across more than
#: it is a fabrication the model will respond to.
DEFAULT_MAX_GAP_S: float = 3600.0

#: A load change of at least this size counts toward a load event [pu].
_EVENT_THRESHOLD_PU: float = 0.05
#: Minimum separation between distinct load events [s].
_EVENT_SEPARATION_S: float = 1800.0


class AmbientMissingError(ValueError):
    """Raised when a record has no usable ambient channel.

    Deliberately its own exception type so a caller can catch precisely this
    and prompt for a weather feed, rather than catching every ValueError.
    """


@dataclass(frozen=True)
class ValidationReport:
    """What the user must see BEFORE any fitting happens.

    Attributes
    ----------
    n_rows_in : rows read from the file
    n_rows_used : rows surviving cleaning
    n_duplicates_dropped : duplicate timestamps collapsed
    n_out_of_order : rows that were not in chronological order
    span_hours : record length [h]
    median_interval_s : median raw sampling interval [s]
    min_interval_s, max_interval_s : extremes of the raw interval [s]
    grid_step_s : uniform grid step the record was resampled onto [s]
    oil_coverage_pct : share of grid points carrying a real top-oil reading
    load_min_pu, load_max_pu : the LOAD HULL the fit is supported over
    n_gaps : input-channel gaps longer than the grid step
    longest_gap_s : longest such gap [s]
    n_load_events : detected load transitions
    ambient_present : whether a usable ambient channel exists
    temperature_quantisation_K : detected rounding of the top-oil channel
    n_hotspot_refs : count of hot-spot calibration reads
    cooling_stages : distinct cooling stages present, or () if not logged
    n_stage_changes : how many times the cooling stage switched
    warnings : non-fatal problems the user should read
    """

    n_rows_in: int
    n_rows_used: int
    n_duplicates_dropped: int
    n_out_of_order: int
    span_hours: float
    median_interval_s: float
    min_interval_s: float
    max_interval_s: float
    grid_step_s: float
    oil_coverage_pct: float
    load_min_pu: float
    load_max_pu: float
    n_gaps: int
    longest_gap_s: float
    n_load_events: int
    ambient_present: bool
    temperature_quantisation_K: float
    n_hotspot_refs: int
    cooling_stages: tuple[int, ...] = ()
    n_stage_changes: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_fittable(self) -> bool:
        """Whether the record can support a four-parameter identification.

        Four free parameters need at least four independent hot-spot reads,
        and the rate parameters need transients: a single load event leaves
        a ~12 % floor under tau_w that no method can beat, while two events
        drop it to ~4 %.
        """
        return (
            self.ambient_present
            and self.n_hotspot_refs >= 4
            and self.n_load_events >= 1
            and self.oil_coverage_pct > 0.0
        )

    def report(self) -> str:
        """The human-readable gate. Print this before fitting anything."""
        lines = [
            "=" * 70,
            "TELEMETRY VALIDATION REPORT",
            "=" * 70,
            f"  rows read                : {self.n_rows_in}",
            f"  rows used                : {self.n_rows_used}"
            + (
                f"  ({self.n_duplicates_dropped} duplicate timestamps collapsed)"
                if self.n_duplicates_dropped
                else ""
            ),
        ]
        if self.n_out_of_order:
            lines.append(f"  out-of-order rows sorted : {self.n_out_of_order}")
        lines += [
            f"  record span              : {self.span_hours:.1f} h",
            f"  raw sampling interval    : median {self.median_interval_s:.0f} s "
            f"(min {self.min_interval_s:.0f}, max {self.max_interval_s:.0f})",
            f"  resampled onto grid      : {self.grid_step_s:.0f} s",
            f"  top-oil coverage         : {self.oil_coverage_pct:.1f} % of grid points",
            "",
            f"  LOAD HULL                : {self.load_min_pu:.3f} - {self.load_max_pu:.3f} pu",
            "     Identified parameters are supported over this span only.",
            "     Extrapolating above it is exactly where single-exponential",
            "     models read +5.76 K HIGH at 1.30 pu.",
            "",
            f"  load events detected     : {self.n_load_events}",
            f"  hot-spot calibration reads: {self.n_hotspot_refs}",
            f"  ambient channel          : {'present' if self.ambient_present else 'MISSING'}",
        ]
        if self.cooling_stages:
            lines.append(
                f"  cooling stages           : {list(self.cooling_stages)} "
                f"({self.n_stage_changes} change(s))"
            )
        if self.n_gaps:
            lines.append(
                f"  input gaps               : {self.n_gaps} "
                f"(longest {self.longest_gap_s / 60:.1f} min, interpolated)"
            )
        if self.temperature_quantisation_K > 0.5:
            lines.append(
                f"  top-oil quantisation     : {self.temperature_quantisation_K:.2f} K "
                f"(integer-degC historian; costs ~1.0x baseline, harmless)"
            )
        lines.append("")
        lines.append(f"  FITTABLE                 : {'YES' if self.is_fittable else 'NO'}")
        for warning in self.warnings:
            for i, chunk in enumerate(textwrap.wrap(warning, 64)):
                lines.append(("  WARNING: " if i == 0 else "           ") + chunk)
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass(frozen=True)
class TelemetryFrame:
    """Cleaned, uniformly-resampled telemetry ready for identification.

    Attributes
    ----------
    time_s : uniform grid, seconds from the first sample [s]
    timestamps : the corresponding wall-clock times
    load_pu : per-unit load current, interpolated onto the grid [pu]
    ambient_C : ambient temperature, interpolated onto the grid [degC]
    top_oil_C : measured top-oil, NaN where no measurement exists [degC]
    hotspot_refs : sparse hot-spot calibration reads, or None
    cooling_stage : cooling-stage label per grid point, or None if the
        record does not log it. Nearest-neighbour resampled, never
        interpolated -- a stage is a discrete control state and a
        "stage 1.4" would be meaningless.
    report : the validation report
    """

    time_s: NDArray[np.float64]
    timestamps: pd.DatetimeIndex
    load_pu: NDArray[np.float64]
    ambient_C: NDArray[np.float64]
    top_oil_C: NDArray[np.float64]
    hotspot_refs: HotspotReferences | None
    cooling_stage: NDArray[np.int_] | None
    report: ValidationReport

    def require_fittable(self) -> None:
        """Raise unless this record can support an identification.

        Call before `identify`. Refusing here, with a reason, is better than
        returning parameters from a record that could never have determined
        them -- a fit that converges on inadequate data still produces
        numbers, and those numbers are indistinguishable from real ones.
        """
        if not self.report.ambient_present:
            raise AmbientMissingError(_AMBIENT_MESSAGE)
        problems = []
        if self.report.n_hotspot_refs < 4:
            problems.append(
                f"only {self.report.n_hotspot_refs} hot-spot calibration read(s); four free "
                f"parameters need at least 4, and two load events' worth (9 reads) is the "
                f"point where the tau_w information floor drops from ~12 % to ~4 %"
            )
        if self.report.n_load_events < 1:
            problems.append(
                "no load events detected; rate parameters (tau_o, tau_w) are unobservable "
                "from quasi-steady operation -- they require transients that anchor both "
                "the rise and the settled value"
            )
        if self.report.oil_coverage_pct <= 0.0:
            problems.append("no top-oil measurements present")
        if problems:
            raise ValueError(
                "this record cannot support a four-parameter identification:\n  - "
                + "\n  - ".join(problems)
            )


_AMBIENT_MESSAGE = (
    "no ambient temperature channel found, and this package will not fit without one.\n"
    "\n"
    "Ignoring a varying ambient UNDER-predicts the hot-spot peak by 3.09 K "
    "(RMSE 3.98 K vs 0.08 K when ambient is supplied), and it does so in the "
    "dangerous direction: the ambient maximum coincides with the afternoon load "
    "peak, so the model reads LOW exactly when the transformer is hottest. It also "
    "corrupts the identified parameters -- tau_w by +68 %, tau_o by +12 % -- so the "
    "result is not salvageable by a later correction.\n"
    "\n"
    "This is cheap to fix. Ambient reaches the winding through a ~75-minute oil "
    "low-pass, so an HOURLY public weather feed from the nearest station is "
    "sufficient. Add an 'ambient_C' column in degrees Celsius and re-run."
)


# --------------------------------------------------------------------------
# Column resolution
# --------------------------------------------------------------------------


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map canonical names onto the file's actual headers."""
    lookup: dict[str, str] = {}
    normalised = {_normalise(c): c for c in frame.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised:
                lookup[canonical] = normalised[alias]
                break
    return lookup


def _to_celsius(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Convert an absolute temperature series to Celsius if it is in kelvin.

    Detection is by magnitude: no ambient or oil temperature on Earth sits
    above 200 in Celsius, and none sits below 200 in kelvin. Silently
    accepting kelvin would shift the entire thermal model by 273 K while
    still producing a plausible-looking trajectory.
    """
    finite = values[np.isfinite(values)]
    if finite.size and float(np.min(finite)) > 200.0:
        return values - 273.15
    return values


def _detect_quantisation(values: NDArray[np.float64]) -> float:
    """Detect the rounding step of a temperature channel [K]."""
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        return 0.0
    if np.allclose(finite, np.round(finite), atol=1e-9):
        return 1.0
    for step in (0.5, 0.1, 0.01):
        if np.allclose(finite / step, np.round(finite / step), atol=1e-9):
            return step
    return 0.0


#: How long an input channel may hold one exact value before the record is
#: called stuck rather than steady, in hours.
#:
#: (b) Engineering estimate from the only field corpus available: three
#: directed-flow units, four clean segments logged every 10 minutes at two
#: decimal places. The longest constant-load run that is plainly genuine is
#: 34.8 h; the longest defective one is 169.8 h, a load channel pinned at
#: 0.01 pu for the last week of a record while ambient still swung 12 K, the
#: top-oil still swung 11 K and the cooling control kept switching fan
#: stages. Nothing in the corpus falls between 35 h and 169 h, so 48 h sits
#: in an empty gap rather than on a boundary. Revisit it against a wider
#: corpus -- a unit genuinely held at constant load for three days would trip
#: this, and that warning would be wrong.
STUCK_CHANNEL_HOURS: float = 48.0


def _stuck_run_hours(values: NDArray[np.float64], seconds: NDArray[np.float64]) -> float:
    """Longest span [h] over which `values` holds one exact value.

    Exact equality is deliberate. A channel that has genuinely settled still
    dithers in its last logged digit; one that is reporting a stuck sentinel
    does not move at all. Comparing with a tolerance would blur the two.
    """
    if values.size < 2:
        return 0.0
    change = np.flatnonzero(np.diff(values) != 0.0)
    edges = np.concatenate(([0], change + 1, [values.size - 1]))
    spans = seconds[edges[1:]] - seconds[edges[:-1]]
    return float(spans.max() / 3600.0) if spans.size else 0.0


def _count_load_events(time_s: NDArray[np.float64], load_pu: NDArray[np.float64]) -> int:
    """Count distinct load transitions.

    A load event is a change of at least `_EVENT_THRESHOLD_PU` sustained
    across a window, with events at least `_EVENT_SEPARATION_S` apart. The
    count matters because it sets the information floor on the rate
    parameters: one event leaves ~12 % on tau_w, two drops it to ~4 %.
    """
    if time_s.size < 3:
        return 0
    step = float(time_s[1] - time_s[0])
    window = max(1, int(round(900.0 / step)))  # 15-minute window
    if window >= load_pu.size:
        return 0
    change = np.abs(load_pu[window:] - load_pu[:-window])
    candidates = np.flatnonzero(change >= _EVENT_THRESHOLD_PU)
    if candidates.size == 0:
        return 0
    events = 1
    last = time_s[candidates[0]]
    for index in candidates[1:]:
        if time_s[index] - last >= _EVENT_SEPARATION_S:
            events += 1
            last = time_s[index]
    return events


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_telemetry(
    path: str | Path,
    *,
    grid_step_s: float = 30.0,
    rated_current_A: float | None = None,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    duplicate_policy: Literal["median", "first", "last"] = "median",
    column_map: Mapping[str, str] | None = None,
) -> TelemetryFrame:
    """Read a utility telemetry CSV and prepare it for identification.

    Handles, explicitly: out-of-order rows, duplicate timestamps, gaps,
    amperes-vs-per-unit load, kelvin-vs-Celsius temperatures, and
    integer-quantised historian output.

    Parameters
    ----------
    path : CSV file. See `write_template` for the expected columns.
    grid_step_s : uniform grid to resample onto [s]. Default 30 s, which
        resolves a winding time constant down to 1 minute. The model
        integrates on this grid; observations keep their own timing.
    rated_current_A : nameplate rated current [A]. Required only when the
        file carries current in amperes rather than per-unit.
    max_gap_s : longest gap in an INPUT channel that will be interpolated
        across [s]. Longer gaps are reported and the affected grid points
        are left unusable rather than fabricated.
    duplicate_policy : how to collapse duplicate timestamps. "median" is
        the default because it is robust to a single bad repeated row.
    column_map : explicit {canonical: file_header} overrides, for files
        whose headers the alias table does not recognise.

    Returns
    -------
    TelemetryFrame. Inspect `.report.report()` before fitting, and call
    `.require_fittable()` to enforce the gate.

    Raises
    ------
    FileNotFoundError, ValueError
        On a missing file, unreadable CSV, missing required columns, or
        load in amperes without `rated_current_A`.
    AmbientMissingError
        If no ambient channel is present. This is a refusal, not a warning:
        see the module docstring.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"telemetry file not found: {source}")

    raw = pd.read_csv(source)
    if raw.empty:
        raise ValueError(f"{source} contains no data rows")
    n_rows_in = len(raw)

    columns = dict(_resolve_columns(raw))
    if column_map:
        columns.update(column_map)

    warnings: list[str] = []

    if "timestamp" not in columns:
        raise ValueError(
            f"no timestamp column found in {source.name}. Recognised spellings: "
            f"{', '.join(COLUMN_ALIASES['timestamp'])}. Pass column_map to override."
        )

    # -- ambient: refuse early, before doing any other work ----------------
    ambient_present = "ambient_C" in columns
    if ambient_present:
        ambient_raw = pd.to_numeric(raw[columns["ambient_C"]], errors="coerce").to_numpy(float)
        if not np.any(np.isfinite(ambient_raw)):
            ambient_present = False
    if not ambient_present:
        raise AmbientMissingError(_AMBIENT_MESSAGE)

    if "top_oil_C" not in columns:
        raise ValueError(
            f"no top-oil temperature column found in {source.name}. Recognised spellings: "
            f"{', '.join(COLUMN_ALIASES['top_oil_C'])}."
        )

    # -- load: per-unit, or amperes plus a nameplate rating ----------------
    if "load_pu" in columns:
        load_raw = pd.to_numeric(raw[columns["load_pu"]], errors="coerce").to_numpy(float)
        finite = load_raw[np.isfinite(load_raw)]
        if finite.size and float(np.nanmax(finite)) > 10.0:
            raise ValueError(
                f"column '{columns['load_pu']}' was read as per-unit load but reaches "
                f"{float(np.nanmax(finite)):.1f} pu. Per-unit load is normally 0-1.5. This "
                f"looks like amperes -- rename the column to 'current_A' and pass "
                f"rated_current_A, so the conversion is explicit and recorded."
            )
    elif "current_A" in columns:
        if rated_current_A is None or rated_current_A <= 0:
            raise ValueError(
                f"{source.name} carries load in amperes ('{columns['current_A']}') but no "
                f"rated_current_A was supplied. Per-unit load cannot be derived without the "
                f"nameplate rated current, and guessing it would silently rescale every "
                f"identified parameter."
            )
        load_raw = (
            pd.to_numeric(raw[columns["current_A"]], errors="coerce").to_numpy(float)
            / rated_current_A
        )
    else:
        raise ValueError(
            f"no load column found in {source.name}. Supply either 'load_pu' or "
            f"'current_A' together with rated_current_A."
        )

    # -- assemble, sort, de-duplicate --------------------------------------
    stamps = pd.to_datetime(raw[columns["timestamp"]], errors="coerce", format="mixed")
    table = pd.DataFrame(
        {
            "timestamp": stamps,
            "load_pu": load_raw,
            "ambient_C": _to_celsius(ambient_raw, "ambient_C"),
            "top_oil_C": _to_celsius(
                pd.to_numeric(raw[columns["top_oil_C"]], errors="coerce").to_numpy(float),
                "top_oil_C",
            ),
        }
    )
    if "cooling_stage" in columns:
        table["cooling_stage"] = pd.to_numeric(
            raw[columns["cooling_stage"]], errors="coerce"
        ).to_numpy(float)
    if "hotspot_C" in columns:
        table["hotspot_C"] = _to_celsius(
            pd.to_numeric(raw[columns["hotspot_C"]], errors="coerce").to_numpy(float),
            "hotspot_C",
        )

    unparsed = int(table["timestamp"].isna().sum())
    if unparsed:
        warnings.append(f"{unparsed} row(s) had an unparseable timestamp and were dropped.")
        table = table.dropna(subset=["timestamp"])
    if table.empty:
        raise ValueError(f"no rows in {source.name} had a parseable timestamp")

    n_out_of_order = int((table["timestamp"].diff() < pd.Timedelta(0)).sum())
    table = table.sort_values("timestamp", kind="mergesort")

    n_before = len(table)
    if table["timestamp"].duplicated().any():
        aggregator = {"median": "median", "first": "first", "last": "last"}[duplicate_policy]
        table = table.groupby("timestamp", as_index=False).agg(aggregator)
    n_duplicates = n_before - len(table)

    times = table["timestamp"].to_numpy()
    seconds = (times - times[0]) / np.timedelta64(1, "s")
    seconds = seconds.astype(float)
    intervals = np.diff(seconds)
    if intervals.size == 0:
        raise ValueError(f"{source.name} has only one usable row after cleaning")

    # -- uniform grid -------------------------------------------------------
    grid = np.arange(0.0, float(seconds[-1]) + grid_step_s, grid_step_s)
    if grid.size < 2:
        raise ValueError(
            f"record spans {seconds[-1]:.0f} s, shorter than the {grid_step_s:.0f} s grid step"
        )

    # INPUTS are interpolated -- the integrator needs a value everywhere.
    load_grid = np.interp(grid, seconds, table["load_pu"].to_numpy(float))
    ambient_grid = np.interp(grid, seconds, table["ambient_C"].to_numpy(float))

    # Gaps longer than max_gap_s are fabrications, not interpolations.
    gap_mask = intervals > max(max_gap_s, grid_step_s)
    n_gaps = int(gap_mask.sum())
    longest_gap = float(intervals.max()) if intervals.size else 0.0
    if n_gaps:
        warnings.append(
            f"{n_gaps} input gap(s) longer than {max_gap_s / 60:.0f} min were interpolated "
            f"across (longest {longest_gap / 60:.0f} min). The thermal model responds to "
            f"interpolated ambient and load as though they were measured; consider splitting "
            f"the record at these gaps."
        )

    # OBSERVATIONS are snapped, never interpolated.
    oil_grid = np.full(grid.size, np.nan)
    oil_values = table["top_oil_C"].to_numpy(float)
    measured = np.isfinite(oil_values)
    if np.any(measured):
        indices = np.round(seconds[measured] / grid_step_s).astype(int)
        valid = (indices >= 0) & (indices < grid.size)
        oil_grid[indices[valid]] = oil_values[measured][valid]

    # A cooling stage is a discrete control state, so it is carried onto the
    # grid by nearest-neighbour hold, never interpolated. Averaging stage 1
    # and stage 2 into 1.5 would invent a cooling configuration that does
    # not exist on any transformer.
    stage_grid = None
    stages: tuple[int, ...] = ()
    n_stage_changes = 0
    if "cooling_stage" in table:
        raw_stage = table["cooling_stage"].to_numpy(float)
        present = np.isfinite(raw_stage)
        if present.sum() >= 2:
            positions = np.searchsorted(seconds[present], grid, side="right") - 1
            positions = np.clip(positions, 0, present.sum() - 1)
            stage_grid = raw_stage[present][positions].astype(np.int_)
            stages = tuple(sorted(set(int(v) for v in stage_grid)))
            n_stage_changes = int(np.count_nonzero(np.diff(stage_grid) != 0))

    hotspot_refs: HotspotReferences | None = None
    n_hotspot_refs = 0
    if "hotspot_C" in table:
        hs_values = table["hotspot_C"].to_numpy(float)
        hs_mask = np.isfinite(hs_values)
        n_hotspot_refs = int(hs_mask.sum())
        if n_hotspot_refs:
            hs_indices = np.round(seconds[hs_mask] / grid_step_s).astype(int)
            hs_valid = (hs_indices >= 0) & (hs_indices < grid.size)
            hotspot_refs = HotspotReferences(
                time_s=grid[hs_indices[hs_valid]],
                temperature_C=hs_values[hs_mask][hs_valid],
                source=f"{source.name}:hotspot_C",
            )

    load_min, load_max = float(np.nanmin(load_grid)), float(np.nanmax(load_grid))
    n_events = _count_load_events(grid, load_grid)
    coverage = 100.0 * float(np.isfinite(oil_grid).sum()) / grid.size

    # A channel stuck on one value is not a measurement, and it is invisible
    # to every other check here: the row count is right, the timestamps are
    # regular, the value is in range. It reached the field campaign undetected
    # and took the headline out-of-sample score with it.
    for channel, series in (("load", load_grid), ("ambient", ambient_grid)):
        held = _stuck_run_hours(series, grid)
        if held >= STUCK_CHANNEL_HOURS:
            warnings.append(
                f"the {channel} channel holds one exact value for {held:.1f} h "
                f"({100.0 * held * 3600.0 / max(seconds[-1], 1.0):.0f} % of the record). "
                f"A stuck channel is a missing-data sentinel wearing a plausible number, "
                f"not a steady operating point -- check it against the temperature "
                f"channels before fitting or scoring anything on this span."
            )

    if n_events == 0:
        warnings.append(
            "no load events detected. The rate parameters tau_o and tau_w are unobservable "
            "from quasi-steady operation; only the amplitudes can be identified from this "
            "record."
        )
    elif n_events == 1:
        warnings.append(
            "only one load event detected. This puts a ~12 % information floor under tau_w "
            "that no estimator can beat. A second event drops it to ~4 %."
        )
    if load_max < 1.0:
        warnings.append(
            f"load never exceeds {load_max:.2f} pu. Parameters identified here are supported "
            f"only over {load_min:.2f}-{load_max:.2f} pu; any loading-envelope result above "
            f"that span is extrapolation and must be labelled as such."
        )
    if len(stages) > 1:
        warnings.append(
            f"{len(stages)} cooling stages are present with {n_stage_changes} change(s). "
            f"Fitting ONE parameter set across a stage change is wrong: on synthetic "
            f"data it costs 4.96 K RMSE and +6.54 K at the peak, worse than the "
            f"single-exponential models this package exists to beat. Use "
            f"corefield.staged.identify_staged."
        )
    if n_hotspot_refs == 0:
        warnings.append(
            "no hot-spot calibration reads. Top-oil carries ZERO information about the "
            "winding parameters (the oil equation contains no winding terms), so some direct "
            "hot-spot data is unavoidable. Commissioning needs at least one bias-audited "
            "reference -- a WTI replica alone transfers its own calibration bias into the "
            "identified parameters."
        )

    report = ValidationReport(
        n_rows_in=n_rows_in,
        n_rows_used=len(table),
        n_duplicates_dropped=n_duplicates,
        n_out_of_order=n_out_of_order,
        span_hours=float(seconds[-1]) / 3600.0,
        median_interval_s=float(np.median(intervals)),
        min_interval_s=float(intervals.min()),
        max_interval_s=float(intervals.max()),
        grid_step_s=grid_step_s,
        oil_coverage_pct=coverage,
        load_min_pu=load_min,
        load_max_pu=load_max,
        n_gaps=n_gaps,
        longest_gap_s=longest_gap,
        n_load_events=n_events,
        ambient_present=True,
        temperature_quantisation_K=_detect_quantisation(oil_values),
        n_hotspot_refs=n_hotspot_refs,
        cooling_stages=stages,
        n_stage_changes=n_stage_changes,
        warnings=tuple(warnings),
    )

    return TelemetryFrame(
        time_s=grid,
        timestamps=pd.DatetimeIndex(times[0] + (grid * 1e9).astype("timedelta64[ns]")),
        load_pu=load_grid,
        ambient_C=ambient_grid,
        top_oil_C=oil_grid,
        hotspot_refs=hotspot_refs,
        cooling_stage=stage_grid,
        report=report,
    )


def write_template(path: str | Path, *, example_rows: int = 0) -> Path:
    """Write a blank CSV with the columns this package requires.

    This is the file a utility engineer gets handed. It carries its own
    instructions in a comment header, because the person filling it in will
    not have read this docstring.

    Parameters
    ----------
    path : destination CSV
    example_rows : if > 0, include this many illustrative rows. They are
        clearly marked as examples and must be deleted before use.

    Returns
    -------
    The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# CoreField telemetry template",
        "#",
        "# One row per sample. Rows may be out of order and may contain gaps or",
        "# duplicate timestamps - all three are handled on import.",
        "#",
        "# COLUMNS",
    ]
    for name, description in TEMPLATE_COLUMNS:
        header.append(f"#   {name:<12} {description}")
    header += [
        "#",
        "# AMBIENT IS REQUIRED. Omitting it under-predicts the hot-spot peak by",
        "# 3.09 K, in the dangerous direction (ambient maximum coincides with the",
        "# afternoon load peak). An HOURLY reading from the nearest weather station",
        "# is sufficient - ambient reaches the winding through a ~75 min oil low-pass.",
        "#",
        "# HOT-SPOT READS: leave blank except where a real measurement was taken.",
        "# Four free parameters need at least 4 reads. Take them 3, 8, 18 and 48",
        "# minutes after a load change, plus one after ~4 h of steady load, and",
        "# repeat over at least TWO load events - that schedule drops the tau_w",
        "# information floor from ~12 % to ~4 %. Use a bias-audited reference:",
        "# a winding-temperature-indicator replica transfers its own bias into the",
        "# identified parameters (+3 K bias -> +14.5 % on the gradient parameter).",
        "#",
        "# COOLING STAGE: if this unit has switchable fans or pumps, log which",
        "# stage is running as an integer (1 = fans off, 2 = first bank, and so on).",
        "# It matters more than it looks. When a fan bank starts, the oil sheds heat",
        "# faster and the thermal parameters change discontinuously. Fitting a single",
        "# parameter set across a stage change costs 4.96 K RMSE and +6.54 K at the peak",
        "# on synthetic data - worse than the simplified models this tool exists to",
        "# beat. If the unit has only one cooling configuration, leave the column out.",
        "#",
        "# LOAD: per-unit of rated current. If you have amperes instead, name the",
        "# column current_A and supply the nameplate rated current at import.",
        "#",
        "# TEMPERATURES: degrees Celsius.",
        "",
    ]

    columns = [name for name, _ in TEMPLATE_COLUMNS]
    lines = header + [",".join(columns)]

    if example_rows > 0:
        start = pd.Timestamp("2026-03-01T00:00:00")
        for i in range(example_rows):
            stamp = (start + pd.Timedelta(minutes=5 * i)).isoformat()
            lines.append(f"{stamp},EXAMPLE-DELETE,EXAMPLE-DELETE,EXAMPLE-DELETE,")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
