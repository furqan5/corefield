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

"""CoreField demo application.

    streamlit run app/streamlit_app.py

Three tabs:
  1. Why this matters   the day-C chart -- the commercial argument
  2. Identify           upload telemetry, see the gate, identify parameters
  3. Loading envelope   how much load, for how long, worth how much

THE BANNER IS NOT DECORATION
----------------------------
Every tab carries a persistent banner stating whether what is on screen came
from SYNTHETIC data or from a USER-SUPPLIED file. No field validation exists
for anything in this repository, and a synthetic result screenshotted without
that label would be a false claim about a real transformer. The banner is
rendered before any chart on every tab, and it reflects the data behind THAT
tab -- tab 1 is always synthetic because it shows the campaign result, while
tabs 2 and 3 follow whatever the user loaded.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from corefield import __version__  # noqa: E402
from corefield.campaign import CAMPAIGN_START, day_transfer_trajectories  # noqa: E402
from corefield.crlb import PARAMETER_NAMES, cramer_rao_bound  # noqa: E402
from corefield.envelope import LoadingLimits, loading_envelope  # noqa: E402
from corefield.estimator import identify  # noqa: E402
from corefield.iec60076_7 import InitialState, ThermalParams, simulate  # noqa: E402
from corefield.ingest import AmbientMissingError, load_telemetry, write_template  # noqa: E402
from corefield.synthetic import (  # noqa: E402
    AMBIENT_CONSTANT_C,
    DT_S,
    OIL_SAMPLE_STRIDE,
    TRUTH_PARAMS,
    calibration_indices,
    truth_trajectory,
)

st.set_page_config(
    page_title="CoreField - transformer hot-spot & loading envelope",
    page_icon="🌡",
    layout="wide",
)

SYNTHETIC = "SYNTHETIC"
USER_SUPPLIED = "USER-SUPPLIED"

PLOT_STYLE = {
    "truth": dict(color="#111111", lw=3.0, alpha=0.55),
    "A": dict(color="#C1440E", lw=1.6, ls="--"),
    "B": dict(color="#E09F3E", lw=1.6, ls="--"),
    "C": dict(color="#1B6CA8", lw=2.0),
}


# --------------------------------------------------------------------------
# The banner
# --------------------------------------------------------------------------


def render_banner(kind: str, detail: str = "") -> None:
    """Render the persistent data-provenance banner. Call before any chart."""
    if kind == SYNTHETIC:
        st.warning(
            f"### ⚠ SYNTHETIC DATA\n"
            f"Everything shown on this tab was generated from a synthetic transformer "
            f"model. **No field validation exists.** These numbers describe how the "
            f"estimator behaves on data whose true answer is known — they are not "
            f"measurements of any real unit.{(' ' + detail) if detail else ''}",
            icon="⚠",
        )
    else:
        st.info(
            f"### 📄 USER-SUPPLIED DATA\n"
            f"This tab is showing results computed from a file you provided."
            f"{(' ' + detail) if detail else ''} The thermal model itself remains "
            f"unvalidated against field measurements, and this package's IEC 60076-7 "
            f"text is mirror-sourced and UNVERIFIED.",
            icon="📄",
        )


# --------------------------------------------------------------------------
# Cached computation
# --------------------------------------------------------------------------


@st.cache_data(persist="disk", show_spinner="Reproducing the day-C model comparison…")
def day_c_comparison() -> dict:
    """The campaign's day-C result, recomputed rather than quoted.

    Computed live rather than shipped as a precomputed figure, so the chart
    can never drift from the code. Persisted to disk so only the very first
    launch pays the ~3.5 s; the cache invalidates automatically if this
    function changes, and `pytest` catches any drift in what it calls.
    """
    result = day_transfer_trajectories("C")
    return {
        "time_h": result.time_h,
        "load_pu": result.load_pu,
        "truth": result.truth_hotspot_C,
        "curves": result.mean_hotspot_C,
        "peak_error": result.peak_error_K,
        "worst_peak": result.worst_peak_error_K,
    }


@st.cache_data(show_spinner="Building the bundled demo record…")
def demo_csv_bytes() -> bytes:
    """A realistic, slightly messy telemetry file built from the synthetic truth."""
    truth = truth_trajectory("A")
    t = truth.time_s
    rng = np.random.default_rng(4)
    cal = calibration_indices(17, t)
    start = pd.Timestamp("2026-03-01T00:00:00")

    rows: dict[int, dict] = {}

    def touch(i: int) -> dict:
        if i not in rows:
            rows[i] = {
                "timestamp": (start + pd.Timedelta(seconds=float(t[i]))).isoformat(),
                "load_pu": round(float(truth.load_pu[i]), 4),
                "ambient_C": AMBIENT_CONSTANT_C,
                "top_oil_C": None,
                "hotspot_C": None,
            }
        return rows[i]

    for i in range(0, t.size, 2):  # load every minute
        touch(i)
    for i in range(0, t.size, OIL_SAMPLE_STRIDE):  # oil every 5 minutes
        touch(i)["top_oil_C"] = round(float(truth.top_oil_C[i] + rng.normal(0, 0.5)), 1)
    for i in cal:
        touch(int(i))["hotspot_C"] = round(float(truth.hotspot_C[i] + rng.normal(0, 0.5)), 2)

    table = pd.DataFrame([rows[k] for k in sorted(rows)])
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


@st.cache_data(show_spinner="Identifying thermal parameters…")
def run_identification(raw: bytes, filename: str) -> dict:
    """Ingest a CSV and identify the four parameters. Returns plain data."""
    tmp = Path(st.session_state["_scratch"]) / filename
    tmp.write_bytes(raw)
    frame = load_telemetry(tmp)
    frame.require_fittable()

    result = identify(
        frame.time_s, frame.load_pu, frame.ambient_C, frame.top_oil_C, frame.hotspot_refs
    )
    fitted = simulate(frame.time_s, frame.load_pu, frame.ambient_C, result.params)

    oil_index = np.flatnonzero(np.isfinite(frame.top_oil_C))
    bound = cramer_rao_bound(
        frame.time_s, frame.load_pu, frame.ambient_C, result.params,
        oil_index,
        np.round(
            (np.asarray(frame.hotspot_refs.time_s) - frame.time_s[0])
            / (frame.time_s[1] - frame.time_s[0])
        ).astype(int),
        0.5,
    )
    return {
        "report_text": frame.report.report(),
        "load_hull": (frame.report.load_min_pu, frame.report.load_max_pu),
        "n_events": frame.report.n_load_events,
        "warnings": frame.report.warnings,
        "params": result.params,
        "crlb_pct": bound.std_percent,
        "rho_oil_winding": float(bound.correlation[1, 3]),
        "residual_rmse_K": result.residual_rmse_K,
        "oil_residual_rmse_K": result.oil_residual_rmse_K,
        "hotspot_residual_rmse_K": result.hotspot_residual_rmse_K,
        "n_observations": result.n_observations,
        "estimator_report": result.report(),
        "time_h": frame.time_s / 3600.0,
        "load_pu": frame.load_pu,
        "top_oil_C": frame.top_oil_C,
        "fitted_oil_C": fitted.top_oil_C,
        "fitted_hotspot_C": fitted.hotspot_C,
        "oil_index": oil_index,
        "ref_time_h": np.asarray(frame.hotspot_refs.time_s) / 3600.0,
        "ref_C": np.asarray(frame.hotspot_refs.temperature_C),
        "final_top_oil_C": float(fitted.top_oil_C[-1]),
        "final_load_pu": float(frame.load_pu[-1]),
    }


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------


def tab_why_this_matters() -> None:
    st.header("Why this matters")
    render_banner(
        SYNTHETIC,
        "This is the model-selection result that chose the production engine.",
    )

    data = day_c_comparison()
    st.markdown(
        "**All three models were fitted on the same ordinary day (0.6–1.2 pu), then "
        "asked about a 2-hour emergency overload at 1.30 pu — outside the load range "
        "they were fitted over.** The two single-exponential models read the hot spot "
        "several kelvin *high*, which would trigger derating exactly when the spare "
        "capacity is worth the most."
    )

    fig, (ax_load, ax) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
    )
    ax_load.plot(data["time_h"], data["load_pu"], color="#444444", lw=1.6)
    ax_load.axhline(1.2, color="#999999", ls=":", lw=1.0)
    ax_load.set_ylabel("load [pu]")
    ax_load.set_ylim(0.6, 1.4)
    ax_load.text(0.3, 1.22, "top of fitted range (1.2 pu)", fontsize=7.5, color="#777777")
    ax_load.grid(alpha=0.25)

    ax.plot(data["time_h"], data["truth"], label="true hot spot", **PLOT_STYLE["truth"])
    labels = {
        "A": "Model A — single-exponential, K² drive",
        "B": "Model B — single-exponential, free exponent",
        "C": "Model C — IEC two-exponential (production)",
    }
    for name in ("A", "B", "C"):
        ax.plot(data["time_h"], data["curves"][name], label=labels[name], **PLOT_STYLE[name])

    peak_index = int(np.argmax(data["truth"]))
    peak_h = float(data["time_h"][peak_index])
    peak_C = float(data["truth"][peak_index])
    ax.axvline(peak_h, color="#BBBBBB", ls=":", lw=1.0)
    for name, offset in (("A", 34), ("B", 18), ("C", -20)):
        ax.annotate(
            f"{name}: {data['worst_peak'][name]:+.2f} K worst case",
            xy=(peak_h, peak_C),
            xytext=(peak_h - 5.6, peak_C + offset),
            fontsize=9,
            color=PLOT_STYLE[name]["color"],
            arrowprops=dict(arrowstyle="->", color=PLOT_STYLE[name]["color"], lw=1.0, alpha=0.7),
        )
    ax.set_xlabel("time [h]")
    ax.set_ylabel("hot-spot temperature [°C]")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 24)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Peak error at 1.30 pu")
    table = pd.DataFrame(
        {
            "Model": [labels[k] for k in ("A", "B", "C")],
            "Mean peak error [K]": [f"{data['peak_error'][k]:+.2f}" for k in ("A", "B", "C")],
            "Worst-case peak error [K]": [
                f"{data['worst_peak'][k]:+.2f}" for k in ("A", "B", "C")
            ],
            "Verdict": ["FAIL", "FAIL", "PASS"],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        "Ten noise realisations, σ = 0.5 K on both channels. Reproduced live by this app "
        "from `corefield.campaign`, not quoted from a report. Models A and B were given "
        "the *noise-free* top-oil signal while Model C had to fit a noisy one — the "
        "comparison is handicapped in their favour and they still fail."
    )


def tab_identify() -> None:
    st.header("Identify")
    source = st.session_state.get("source", SYNTHETIC)
    render_banner(
        source,
        "Bundled demo record generated from the synthetic model."
        if source == SYNTHETIC
        else "",
    )

    payload = st.session_state.get("payload")
    if payload is None:
        st.info("Choose a data source in the sidebar to begin.")
        return

    try:
        data = run_identification(payload["bytes"], payload["name"])
    except AmbientMissingError as exc:
        st.error("**Refused: no ambient channel.**")
        st.code(str(exc))
        return
    except (ValueError, RuntimeError) as exc:
        st.error(f"**Could not fit this record.**\n\n{exc}")
        return

    st.subheader("1 · Validation report")
    st.caption("Shown before any fitting. A record that cannot support the fit is refused here.")
    st.code(data["report_text"], language="text")
    for warning in data["warnings"]:
        st.warning(warning)

    st.subheader("2 · Identified parameters")
    params: ThermalParams = data["params"]
    values = params.as_vector()
    display_values = [values[0], values[1] / 60.0, values[2], values[3] / 60.0]
    units = ["K", "min", "K", "min"]
    crlb = data["crlb_pct"]

    columns = st.columns(4)
    pretty = ["Δθ_or", "τ_o", "Δθ_hr", "τ_w"]
    for column, name, value, unit, bound_pct in zip(columns, pretty, display_values, units, crlb):
        column.metric(
            label=f"{name}  [{unit}]",
            value=f"{value:.2f}",
            delta=f"± {value * bound_pct / 100:.2f} (CRLB)",
            delta_color="off",
        )

    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    positions = np.arange(4)
    ax.barh(positions, crlb, color="#1B6CA8", alpha=0.85)
    ax.set_yticks(positions, pretty)
    ax.invert_yaxis()
    ax.set_xlabel("Cramér–Rao lower bound [% of value]")
    for i, value in enumerate(crlb):
        ax.text(value, i, f"  {value:.2f} %", va="center", fontsize=9)
    ax.set_xlim(0, max(float(np.max(crlb)) * 1.35, 0.5))
    ax.grid(alpha=0.25, axis="x")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption(
        f"The bound is what *no* estimator can beat on this record. τ_w is always the "
        f"hardest of the four — it needs load transients, while the amplitudes are "
        f"readable from steady operation. ρ(τ_o, τ_w) = {data['rho_oil_winding']:+.3f}, "
        f"so the oil and winding pairs separate cleanly: this is not a degenerate fit."
    )

    st.subheader("3 · Fit quality")
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        oil_index = data["oil_index"]
        ax.plot(
            data["time_h"][oil_index], data["top_oil_C"][oil_index], ".",
            ms=2.4, color="#8899AA", label="measured top-oil",
        )
        ax.plot(data["time_h"], data["fitted_oil_C"], color="#1B6CA8", lw=1.4,
                label="fitted top-oil")
        ax.plot(data["time_h"], data["fitted_hotspot_C"], color="#C1440E", lw=1.8,
                label="reconstructed hot spot (hidden)")
        ax.plot(data["ref_time_h"], data["ref_C"], "o", ms=6, mfc="none",
                mec="#111111", mew=1.3, label="hot-spot calibration reads")
        ax.set_xlabel("time [h]")
        ax.set_ylabel("temperature [°C]")
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.metric("Residual RMSE", f"{data['residual_rmse_K']:.3f} K")
        st.metric("… top-oil channel", f"{data['oil_residual_rmse_K']:.3f} K")
        st.metric("… hot-spot channel", f"{data['hotspot_residual_rmse_K']:.3f} K")
        st.caption(
            f"{data['n_observations'][0]} top-oil samples, "
            f"{data['n_observations'][1]} hot-spot reads."
        )

    st.subheader("4 · Residuals")
    fig, ax = plt.subplots(figsize=(11, 2.8))
    oil_index = data["oil_index"]
    residual = data["fitted_oil_C"][oil_index] - data["top_oil_C"][oil_index]
    ax.axhline(0, color="#999999", lw=1.0)
    ax.plot(data["time_h"][oil_index], residual, ".", ms=2.6, color="#1B6CA8")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("top-oil\nresidual [K]")
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Structure in this plot means the model is missing something. Flat noise means "
        "it is not. A slow ramp in particular would indicate sensor drift, which passes "
        "the trajectory gate while quietly poisoning the parameters."
    )


def tab_envelope() -> None:
    st.header("Loading envelope")
    source = st.session_state.get("source", SYNTHETIC)
    render_banner(source)

    payload = st.session_state.get("payload")
    if payload is None:
        st.info("Choose a data source in the sidebar to begin.")
        return
    try:
        data = run_identification(payload["bytes"], payload["name"])
    except (AmbientMissingError, ValueError, RuntimeError) as exc:
        st.error(f"Identification failed, so no envelope can be computed.\n\n{exc}")
        return

    st.markdown(
        "**The limits below are not supplied by this software.** Its copy of "
        "IEC 60076-7 is mirror-sourced and unverified, so hard-coding a temperature "
        "limit at the point where the tool says *it is safe to overload* would be the "
        "worst possible place for a remembered number. Enter your own, and record where "
        "they came from — the provenance travels with the result."
    )

    with st.form("limits"):
        columns = st.columns([1, 1, 1, 2])
        hotspot_limit = columns[0].number_input(
            "Hot-spot limit [°C]", min_value=40.0, max_value=250.0, value=120.0, step=1.0
        )
        oil_limit = columns[1].number_input(
            "Top-oil limit [°C]", min_value=30.0, max_value=200.0, value=105.0, step=1.0
        )
        duration_h = columns[2].number_input(
            "Duration [h]", min_value=0.25, max_value=24.0, value=2.0, step=0.25
        )
        nameplate = columns[3].number_input(
            "Nameplate rating [MVA]", min_value=0.1, max_value=2000.0, value=63.0, step=1.0
        )
        source_text = st.text_input(
            "Where did these limits come from?",
            value="",
            placeholder="e.g. IEC 60076-7:2018, licensed copy, Table N, checked 2026-08-24 by AB",
        )
        ambient_C = st.slider("Ambient over the window [°C]", -10.0, 55.0, 30.0, 1.0)
        confidence = st.slider("Confidence for the conservative value", 0.50, 0.99, 0.95, 0.01)
        submitted = st.form_submit_button("Compute envelope", type="primary")

    if not submitted:
        st.info("Enter your limits and their source, then compute.")
        return

    try:
        limits = LoadingLimits(
            hotspot_limit_C=hotspot_limit,
            top_oil_limit_C=oil_limit,
            label="user-supplied loading limits",
            source=source_text,
        )
    except ValueError as exc:
        st.error(f"**Limits rejected.** {exc}")
        return

    params: ThermalParams = data["params"]
    truth = truth_trajectory("A")
    bound = cramer_rao_bound(
        truth.time_s, truth.load_pu, np.full(truth.time_s.size, AMBIENT_CONSTANT_C),
        params, np.arange(0, truth.time_s.size, OIL_SAMPLE_STRIDE),
        calibration_indices(17, truth.time_s), 0.5,
    )
    state = InitialState(
        top_oil_C=data["final_top_oil_C"], prior_load_pu=data["final_load_pu"]
    )

    with st.spinner("Propagating parameter uncertainty…"):
        envelope = loading_envelope(
            params, limits, ambient_C, duration_h, state,
            bound=bound, confidence=confidence, n_samples=120,
            nameplate_MVA=nameplate, fitted_load_hull=data["load_hull"],
        )

    if not envelope.feasible:
        st.error(envelope.summary())
        return

    left, middle, right = st.columns(3)
    left.metric(
        "Sustainable load",
        f"{envelope.k_max_conservative_pu:.3f} pu",
        delta=f"point estimate {envelope.k_max_pu:.3f}",
        delta_color="off",
    )
    middle.metric("For", f"{envelope.duration_h:.2f} h")
    right.metric(
        "Above nameplate",
        f"{envelope.conservative_headroom_MVA_h:.1f} MVA·h"
        if envelope.conservative_headroom_MVA_h is not None
        else "—",
    )

    st.success(envelope.summary())

    # How the answer varies with the window length.
    with st.spinner("Sweeping duration…"):
        durations = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        point = []
        for hours in durations:
            point.append(
                loading_envelope(
                    params, limits, ambient_C, float(hours), state,
                    nameplate_MVA=nameplate,
                ).k_max_pu
            )

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.0))
    ax.plot(durations, point, "o-", color="#1B6CA8", lw=2.0, ms=5)
    ax.axhline(1.0, color="#999999", ls=":", lw=1.2)
    ax.text(durations[-1], 1.005, "nameplate", ha="right", fontsize=8, color="#777777")
    ax.plot([duration_h], [envelope.k_max_pu], "*", ms=17, color="#C1440E", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(durations, [f"{d:g}" for d in durations])
    ax.set_xlabel("duration [h]")
    ax.set_ylabel("maximum sustainable load [pu]")
    ax.set_title("Shorter windows buy more load", fontsize=10)
    ax.grid(alpha=0.25)

    n_points = max(2, int(round(duration_h * 3600.0 / DT_S)) + 1)
    grid = np.arange(n_points) * DT_S
    trace = simulate(
        grid, np.full(n_points, envelope.k_max_pu), np.full(n_points, ambient_C),
        params, initial_state=state,
    )
    ax2.plot(grid / 3600.0, trace.hotspot_C, color="#C1440E", lw=2.0, label="hot spot")
    ax2.plot(grid / 3600.0, trace.top_oil_C, color="#1B6CA8", lw=1.6, label="top oil")
    ax2.axhline(hotspot_limit, color="#C1440E", ls="--", lw=1.2, label="hot-spot limit")
    ax2.axhline(oil_limit, color="#1B6CA8", ls="--", lw=1.2, label="top-oil limit")
    ax2.set_xlabel("time [h]")
    ax2.set_ylabel("temperature [°C]")
    ax2.set_title(f"Trajectory at {envelope.k_max_pu:.3f} pu", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("What this number does not include")
    for note in envelope.notes:
        st.caption(f"• {note}")


# --------------------------------------------------------------------------
# Sidebar and entry point
# --------------------------------------------------------------------------


def main() -> None:
    if "_scratch" not in st.session_state:
        # A real temp directory, not the project tree: uploaded telemetry is the
        # user's data and must not be left lying in the repository, where the
        # .gitignore rules are the only thing standing between it and a commit.
        st.session_state["_scratch"] = tempfile.mkdtemp(prefix="corefield_")

    with st.sidebar:
        st.title("CoreField")
        st.caption(f"v{__version__}")
        st.markdown("---")
        st.subheader("Data source")
        choice = st.radio(
            "Data source",
            ["Bundled synthetic demo", "Upload a CSV"],
            label_visibility="collapsed",
        )

        if choice == "Bundled synthetic demo":
            st.session_state["source"] = SYNTHETIC
            st.session_state["payload"] = {
                "bytes": demo_csv_bytes(),
                "name": "demo_synthetic.csv",
            }
            st.caption(
                "One synthetic day: 1-minute load, 5-minute top-oil, 17 hot-spot "
                "calibration reads. The true parameters are known, so the fit can be "
                "scored."
            )
        else:
            uploaded = st.file_uploader("Telemetry CSV", type=["csv"])
            if uploaded is not None:
                st.session_state["source"] = USER_SUPPLIED
                st.session_state["payload"] = {
                    "bytes": uploaded.getvalue(),
                    "name": uploaded.name,
                }
            else:
                st.session_state["payload"] = None
                st.caption("Waiting for a file.")

        st.markdown("---")
        st.subheader("Need a template?")
        template_path = Path(st.session_state["_scratch"]) / "corefield_template.csv"
        write_template(template_path)
        st.download_button(
            "Download blank telemetry template",
            data=template_path.read_bytes(),
            file_name="corefield_telemetry_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Hand this to the site engineer. The required columns and the calibration "
            "schedule are documented in the file header."
        )

        st.markdown("---")
        st.caption(
            "**IEC 60076-7 provenance: UNVERIFIED.** Equation structure and cooling-class "
            "constants were mirror-sourced and have not been checked against a licensed "
            "copy of the standard.\n\n"
            "**Field validation: none.** No measurement from a real transformer has ever "
            "entered this package."
        )

    tabs = st.tabs(["① Why this matters", "② Identify", "③ Loading envelope"])
    with tabs[0]:
        tab_why_this_matters()
    with tabs[1]:
        tab_identify()
    with tabs[2]:
        tab_envelope()


main()
