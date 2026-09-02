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

"""Ingestion: the messy-data boundary, and the refusal to fit without ambient.

Every failure mode listed in the Stage-4 brief gets a test that proves it is
handled rather than merely mentioned: out-of-order rows, duplicates, gaps,
amperes vs per-unit, kelvin vs Celsius, integer quantisation.

The end-to-end round trip at the bottom is the most important test in this
file. It is the only place where the full chain -- synthetic truth, written
to a realistically messy CSV, read back through ingestion, identified --
runs as one piece.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corefield.campaign import CAMPAIGN_START
from corefield.cli import main as cli_main
from corefield.estimator import identify
from corefield.ingest import (
    AmbientMissingError,
    TEMPLATE_COLUMNS,
    load_telemetry,
    write_template,
)
from corefield.synthetic import (
    AMBIENT_CONSTANT_C,
    TRUTH_PARAMS,
    calibration_indices,
    truth_trajectory,
)

START = pd.Timestamp("2026-03-01T00:00:00")


def _build_csv(
    path,
    *,
    load_stride_min: float = 1.0,
    oil_stride_min: float = 5.0,
    seed: int = 11,
    include_ambient: bool = True,
    include_hotspot: bool = True,
    kelvin: bool = False,
    amperes: float | None = None,
    quantise_oil: bool = False,
    shuffle: bool = False,
    duplicate_rows: int = 0,
    drop_hours: tuple[float, float] | None = None,
):
    """Write a synthetic day-A record as a deliberately imperfect CSV."""
    truth = truth_trajectory("A")
    t = truth.time_s
    rng = np.random.default_rng(seed)
    cal = calibration_indices(17, t)

    load_idx = np.arange(0, t.size, max(1, int(load_stride_min * 2)))
    oil_idx = np.arange(0, t.size, max(1, int(oil_stride_min * 2)))

    rows: dict[int, dict] = {}

    def touch(i: int) -> dict:
        if i not in rows:
            rows[i] = {
                "timestamp": (START + pd.Timedelta(seconds=float(t[i]))).isoformat(),
                "load_pu": round(float(truth.load_pu[i]), 4),
                "ambient_C": AMBIENT_CONSTANT_C,
                "top_oil_C": None,
                "hotspot_C": None,
            }
        return rows[i]

    for i in load_idx:
        touch(int(i))
    for i in oil_idx:
        value = float(truth.top_oil_C[i] + rng.normal(0, 0.5))
        touch(int(i))["top_oil_C"] = round(value, 0 if quantise_oil else 1)
    if include_hotspot:
        for i in cal:
            touch(int(i))["hotspot_C"] = round(
                float(truth.hotspot_C[i] + rng.normal(0, 0.5)), 2
            )

    table = pd.DataFrame([rows[k] for k in sorted(rows)])

    if drop_hours is not None:
        seconds = np.array(
            [(pd.Timestamp(s) - START).total_seconds() for s in table["timestamp"]]
        )
        keep = ~((seconds >= drop_hours[0] * 3600) & (seconds < drop_hours[1] * 3600))
        table = table[keep]

    if kelvin:
        for column in ("ambient_C", "top_oil_C", "hotspot_C"):
            table[column] = pd.to_numeric(table[column], errors="coerce") + 273.15
    if amperes is not None:
        table["current_A"] = table["load_pu"] * amperes
        table = table.drop(columns=["load_pu"])
    if not include_ambient:
        table = table.drop(columns=["ambient_C"])
    if not include_hotspot:
        table = table.drop(columns=["hotspot_C"])
    if duplicate_rows:
        table = pd.concat([table, table.head(duplicate_rows)], ignore_index=True)
    if shuffle:
        table = table.sample(frac=1.0, random_state=3).reset_index(drop=True)

    table.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------


def test_missing_ambient_is_refused_not_warned(tmp_path):
    """No ambient, no fit. The message must carry the number that justifies it."""
    path = _build_csv(tmp_path / "no_ambient.csv", include_ambient=False)
    with pytest.raises(AmbientMissingError) as excinfo:
        load_telemetry(path)
    message = str(excinfo.value)
    assert "3.09" in message, "the refusal must state the cost of ignoring ambient"
    assert "hourly" in message.lower(), "it must also say how cheap the fix is"


def test_all_nan_ambient_column_counts_as_missing(tmp_path):
    """A present-but-empty ambient column is not an ambient channel."""
    path = _build_csv(tmp_path / "blank_ambient.csv")
    table = pd.read_csv(path)
    table["ambient_C"] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(AmbientMissingError):
        load_telemetry(path)


# --------------------------------------------------------------------------
# Messy input handling
# --------------------------------------------------------------------------


def test_out_of_order_rows_are_sorted(tmp_path):
    path = _build_csv(tmp_path / "shuffled.csv", shuffle=True)
    frame = load_telemetry(path)
    assert frame.report.n_out_of_order > 0
    assert np.all(np.diff(frame.time_s) > 0)


def test_duplicate_timestamps_are_collapsed(tmp_path):
    path = _build_csv(tmp_path / "dupes.csv", duplicate_rows=25)
    frame = load_telemetry(path)
    assert frame.report.n_duplicates_dropped == 25


def test_kelvin_temperatures_are_converted(tmp_path):
    """Kelvin is detected and converted, not passed through as Celsius.

    A 273 K offset would still produce a plausible-looking trajectory, which
    is exactly what makes it dangerous.
    """
    celsius = load_telemetry(_build_csv(tmp_path / "c.csv"))
    kelvin = load_telemetry(_build_csv(tmp_path / "k.csv", kelvin=True))
    assert np.allclose(celsius.ambient_C, kelvin.ambient_C, atol=1e-6)
    finite = np.isfinite(celsius.top_oil_C)
    assert np.allclose(celsius.top_oil_C[finite], kelvin.top_oil_C[finite], atol=1e-6)


def test_amperes_without_rated_current_is_refused(tmp_path):
    """Guessing the nameplate rating would rescale every identified parameter."""
    path = _build_csv(tmp_path / "amps.csv", amperes=800.0)
    with pytest.raises(ValueError, match="rated_current_A"):
        load_telemetry(path)


def test_amperes_with_rated_current_converts(tmp_path):
    path = _build_csv(tmp_path / "amps_ok.csv", amperes=800.0)
    frame = load_telemetry(path, rated_current_A=800.0)
    assert frame.report.load_max_pu == pytest.approx(1.2, abs=0.02)


def test_amperes_mislabelled_as_per_unit_is_caught(tmp_path):
    """A column called load_pu holding hundreds of amps must not be believed."""
    path = _build_csv(tmp_path / "mislabelled.csv")
    table = pd.read_csv(path)
    table["load_pu"] = table["load_pu"] * 800.0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="looks like amperes"):
        load_telemetry(path)


def test_gaps_are_reported_and_flagged(tmp_path):
    path = _build_csv(tmp_path / "gappy.csv", drop_hours=(9.0, 12.0))
    frame = load_telemetry(path)
    assert frame.report.n_gaps >= 1
    assert frame.report.longest_gap_s > 2 * 3600
    assert any("gap" in w.lower() for w in frame.report.warnings)


def test_integer_quantisation_is_detected(tmp_path):
    path = _build_csv(tmp_path / "quantised.csv", quantise_oil=True)
    frame = load_telemetry(path)
    assert frame.report.temperature_quantisation_K == pytest.approx(1.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_telemetry(tmp_path / "nope.csv")


# --------------------------------------------------------------------------
# Inputs are interpolated; observations are not
# --------------------------------------------------------------------------


def test_observations_are_not_interpolated(tmp_path):
    """Top-oil must stay NaN between measurements.

    Interpolating an observation invents data that was never measured and
    correlates its noise, which makes the residual RMSE look better than
    the record deserves. Inputs are a different case: the integrator needs
    load and ambient at every grid point, so those ARE interpolated.
    """
    frame = load_telemetry(_build_csv(tmp_path / "sparse.csv", oil_stride_min=5.0))
    assert np.isnan(frame.top_oil_C).any()
    assert frame.report.oil_coverage_pct == pytest.approx(10.0, abs=1.0)
    # Inputs, by contrast, are complete.
    assert np.all(np.isfinite(frame.load_pu))
    assert np.all(np.isfinite(frame.ambient_C))


def test_grid_is_uniform_and_fine_enough_to_integrate(tmp_path):
    frame = load_telemetry(_build_csv(tmp_path / "grid.csv"), grid_step_s=30.0)
    steps = np.diff(frame.time_s)
    assert np.allclose(steps, 30.0)
    assert frame.report.grid_step_s == 30.0


# --------------------------------------------------------------------------
# The validation gate
# --------------------------------------------------------------------------


def test_report_states_the_load_hull(tmp_path):
    """The hull is the span the parameters are supported over."""
    frame = load_telemetry(_build_csv(tmp_path / "hull.csv"))
    assert frame.report.load_min_pu == pytest.approx(0.60, abs=0.02)
    assert frame.report.load_max_pu == pytest.approx(1.20, abs=0.02)
    assert "LOAD HULL" in frame.report.report()
    assert "+5.76" in frame.report.report()


def test_load_events_are_counted(tmp_path):
    """Day A has four load events; the count drives the tau_w information floor."""
    frame = load_telemetry(_build_csv(tmp_path / "events.csv"))
    assert frame.report.n_load_events == 4


def test_record_without_hotspot_reads_is_not_fittable(tmp_path):
    """Top-oil carries zero winding information, so reads are unavoidable."""
    frame = load_telemetry(_build_csv(tmp_path / "no_hs.csv", include_hotspot=False))
    assert not frame.report.is_fittable
    with pytest.raises(ValueError, match="hot-spot calibration read"):
        frame.require_fittable()
    assert any("ZERO information" in w for w in frame.report.warnings)


def test_fittable_record_passes_the_gate(tmp_path):
    frame = load_telemetry(_build_csv(tmp_path / "good.csv"))
    assert frame.report.is_fittable
    frame.require_fittable()


def test_report_is_printable_and_mentions_the_essentials(tmp_path):
    text = load_telemetry(_build_csv(tmp_path / "print.csv")).report.report()
    for expected in ("LOAD HULL", "load events detected", "ambient channel", "FITTABLE"):
        assert expected in text


# --------------------------------------------------------------------------
# Template and CLI
# --------------------------------------------------------------------------


def test_template_has_the_required_columns(tmp_path):
    path = write_template(tmp_path / "template.csv")
    table = pd.read_csv(path, comment="#")
    assert list(table.columns) == [name for name, _ in TEMPLATE_COLUMNS]
    assert table.empty


def test_template_explains_why_ambient_is_required(tmp_path):
    """The engineer filling this in will not have read the docs."""
    text = write_template(tmp_path / "t.csv").read_text(encoding="utf-8")
    assert "3.09" in text
    assert "AMBIENT IS REQUIRED" in text
    assert "bias-audited" in text


def test_cli_writes_a_template(tmp_path, capsys):
    target = tmp_path / "cli_template.csv"
    assert cli_main(["--template", str(target)]) == 0
    assert target.exists()
    assert "template" in capsys.readouterr().out.lower()


def test_cli_validate_accepts_a_good_file(tmp_path, capsys):
    path = _build_csv(tmp_path / "cli_good.csv")
    assert cli_main(["validate", str(path)]) == 0
    assert "LOAD HULL" in capsys.readouterr().out


def test_cli_validate_refuses_a_file_without_ambient(tmp_path, capsys):
    path = _build_csv(tmp_path / "cli_bad.csv", include_ambient=False)
    assert cli_main(["validate", str(path)]) == 2
    assert "3.09" in capsys.readouterr().err


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_round_trip_recovers_the_truth(tmp_path):
    """Synthetic truth -> messy CSV -> ingest -> identify.

    Shuffled rows, duplicate timestamps, integer-quantised oil, sparse
    hot-spot reads. The whole chain in one test.
    """
    path = _build_csv(
        tmp_path / "roundtrip.csv",
        load_stride_min=1.0, oil_stride_min=5.0,
        shuffle=True, duplicate_rows=10, quantise_oil=True,
    )
    frame = load_telemetry(path)
    frame.require_fittable()
    result = identify(
        frame.time_s, frame.load_pu, frame.ambient_C, frame.top_oil_C,
        frame.hotspot_refs, loss="linear", starts=CAMPAIGN_START,
    )
    error_pct = np.abs(
        (result.params.as_vector() - TRUTH_PARAMS.as_vector()) / TRUTH_PARAMS.as_vector() * 100
    )
    assert error_pct[0] < 1.0, "delta_theta_or"
    assert error_pct[1] < 2.0, "tau_o"
    assert error_pct[2] < 3.0, "delta_theta_hr"
    assert error_pct[3] < 6.0, "tau_w"
    assert result.residual_rmse_K < 1.0


def test_load_sampling_rate_matters_more_than_oil_sampling_rate(tmp_path):
    """A commissioning specification, derived rather than assumed.

    Logging load at 5-minute resolution aliases the ~90 s load ramps that
    carry the tau_w information: the error rises from +2.1 % to +8.4 %.
    Logging top-oil at 5 minutes instead of 1 costs almost nothing, because
    the oil low-pass is ~75 minutes.

    Practical consequence for a pilot: insist on 1-minute LOAD CURRENT
    logging; 5-minute top-oil is fine. That is usually free -- the historian
    is already sampling current faster than it stores it.
    """

    def tau_w_error(load_stride_min: float, oil_stride_min: float) -> float:
        path = _build_csv(
            tmp_path / f"rate_{load_stride_min}_{oil_stride_min}.csv",
            load_stride_min=load_stride_min, oil_stride_min=oil_stride_min,
        )
        frame = load_telemetry(path)
        result = identify(
            frame.time_s, frame.load_pu, frame.ambient_C, frame.top_oil_C,
            frame.hotspot_refs, loss="linear", starts=CAMPAIGN_START,
        )
        truth = TRUTH_PARAMS.as_vector()
        return abs(float((result.params.as_vector()[3] - truth[3]) / truth[3] * 100))

    fast_load = tau_w_error(1.0, 5.0)
    slow_load = tau_w_error(5.0, 5.0)
    finer_oil = tau_w_error(1.0, 1.0)

    assert fast_load < 5.0, "1-minute load logging must keep tau_w usable"
    assert slow_load > 2 * fast_load, "5-minute load logging must visibly degrade tau_w"
    assert abs(finer_oil - fast_load) < slow_load - fast_load, (
        "refining OIL sampling must help less than refining LOAD sampling"
    )


# --------------------------------------------------------------------------
# Cooling stage
# --------------------------------------------------------------------------


def _add_stage_column(path, stage_values):
    table = pd.read_csv(path)
    table["cooling_stage"] = stage_values(len(table))
    table.to_csv(path, index=False)
    return path


def test_cooling_stage_is_read_and_reported(tmp_path):
    """A staged record must surface its stages before anyone fits it."""
    path = _build_csv(tmp_path / "staged.csv")
    _add_stage_column(path, lambda n: [1] * (n // 2) + [2] * (n - n // 2))
    frame = load_telemetry(path)
    assert frame.cooling_stage is not None
    assert frame.report.cooling_stages == (1, 2)
    assert frame.report.n_stage_changes == 1
    assert "cooling stages" in frame.report.report()


def test_multiple_stages_trigger_a_warning_naming_the_cost(tmp_path):
    """The warning must state what ignoring staging costs, not just that it exists."""
    path = _build_csv(tmp_path / "staged_warn.csv")
    _add_stage_column(path, lambda n: [1] * (n // 2) + [2] * (n - n // 2))
    frame = load_telemetry(path)
    warnings = " ".join(frame.report.warnings)
    assert "cooling stages" in warnings
    assert "6.54" in warnings, "the warning must quantify the peak error"
    assert "identify_staged" in warnings, "and point at the fix"


def test_cooling_stage_is_never_interpolated(tmp_path):
    """A stage is a discrete control state; 1.5 is not a cooling configuration."""
    path = _build_csv(tmp_path / "stage_discrete.csv", load_stride_min=5.0)
    _add_stage_column(path, lambda n: [1] * (n // 2) + [2] * (n - n // 2))
    frame = load_telemetry(path, grid_step_s=30.0)
    assert set(np.unique(frame.cooling_stage)) <= {1, 2}


def test_single_stage_record_gets_no_staging_warning(tmp_path):
    path = _build_csv(tmp_path / "one_stage.csv")
    _add_stage_column(path, lambda n: [2] * n)
    frame = load_telemetry(path)
    assert frame.report.cooling_stages == (2,)
    assert frame.report.n_stage_changes == 0
    assert not any("cooling stages" in w for w in frame.report.warnings)


def test_record_without_a_stage_column_still_works(tmp_path):
    """Most units have one cooling configuration; the column stays optional."""
    frame = load_telemetry(_build_csv(tmp_path / "no_stage.csv"))
    assert frame.cooling_stage is None
    assert frame.report.cooling_stages == ()


def test_template_explains_the_cooling_stage_column(tmp_path):
    text = write_template(tmp_path / "t2.csv").read_text(encoding="utf-8")
    assert "COOLING STAGE" in text
    assert "6.54" in text


# --------------------------------------------------------------------------
# Stuck input channels
#
# A channel pinned on one value is invisible to every other check in this
# module: the row count is right, the timestamps are regular, the value is in
# range and physically plausible. One reached a field campaign undetected -- a
# load channel holding exactly 0.01 pu for the last 169.7 h of a record while
# ambient still swung 12 K, top-oil still swung 11 K and the cooling control
# kept switching fan stages -- and took the headline out-of-sample score with
# it. The record was not of a transformer sitting idle; the channel had died.
# --------------------------------------------------------------------------


def _stuck_channel_frame(tmp_path, stuck_hours, total_hours=336.0, step_min=10.0):
    """A clean record whose load channel freezes for the last `stuck_hours`."""
    import numpy as np
    import pandas as pd

    n = int(total_hours * 60.0 / step_min)
    t = pd.date_range("2026-01-01", periods=n, freq=f"{int(step_min)}min")
    hours = np.arange(n) * step_min / 60.0
    # An ordinary daily load cycle, rounded to two decimals as loggers do.
    load = np.round(0.70 + 0.15 * np.sin(2 * np.pi * hours / 24.0), 2)
    ambient = np.round(15.0 + 8.0 * np.sin(2 * np.pi * (hours - 6.0) / 24.0), 1)
    frozen = hours > (total_hours - stuck_hours)
    load[frozen] = 0.01
    frame = pd.DataFrame({
        "timestamp": t,
        "load_pu": load,
        "ambient_C": ambient,
        # Temperatures keep moving: that is what makes the load channel a lie.
        "top_oil_C": np.round(ambient + 30.0 + 3.0 * np.sin(2 * np.pi * hours / 24.0), 1),
        "hotspot_C": np.round(ambient + 50.0 + 4.0 * np.sin(2 * np.pi * hours / 24.0), 1),
    })
    path = tmp_path / "stuck.csv"
    frame.to_csv(path, index=False)
    return path


def test_a_stuck_load_channel_is_reported(tmp_path):
    from corefield.ingest import STUCK_CHANNEL_HOURS, load_telemetry

    path = _stuck_channel_frame(tmp_path, stuck_hours=STUCK_CHANNEL_HOURS + 24.0)
    report = load_telemetry(path).report
    stuck = [w for w in report.warnings if "holds one exact value" in w]
    assert stuck, f"stuck channel not reported; warnings were {report.warnings}"
    assert "load" in stuck[0]
    assert "sentinel" in stuck[0]


def test_a_stuck_channel_shows_up_in_the_printed_report(tmp_path):
    from corefield.ingest import STUCK_CHANNEL_HOURS, load_telemetry

    path = _stuck_channel_frame(tmp_path, stuck_hours=STUCK_CHANNEL_HOURS + 24.0)
    assert "holds one exact value" in load_telemetry(path).report.report()


def test_an_ordinary_flat_spell_is_not_called_stuck(tmp_path):
    """A day and a half of genuinely constant load must not trip the check.

    The longest plainly-genuine constant-load run in the field corpus is
    34.8 h. Warning on that would make the check noise, and a check that
    cries wolf is one the user learns to skip past.
    """
    from corefield.ingest import load_telemetry

    path = _stuck_channel_frame(tmp_path, stuck_hours=34.8)
    report = load_telemetry(path).report
    assert not [w for w in report.warnings if "holds one exact value" in w]
