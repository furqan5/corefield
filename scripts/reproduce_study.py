"""Reproduce the synthetic study only; no operational data are loaded.

Run from the repository root with ``python -m scripts.reproduce_study --out DIR``.
Temperatures/errors are degC/K, time is seconds and load is per-unit current.
The JSON and plots are generated evidence, not a field-loading certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from corefield.campaign import CAMPAIGN_START, day_transfer, day_transfer_trajectories, run_scenario
from corefield.crlb import cramer_rao_bound, efficiency_ratio
from corefield.estimator import HotspotReferences, identify
from corefield.synthetic import (
    AMBIENT_CONSTANT_C, DT_S, OIL_SAMPLE_STRIDE, TRUTH_PARAMS,
    ALL_SCENARIOS, calibration_indices, day_a_load, truth_trajectory,
)


def reproduce(out: Path, *, figures_only: bool = False) -> dict:
    """Write synthetic study metrics and plots; physical units are stated in keys."""
    out.mkdir(parents=True, exist_ok=True)
    if figures_only:
        # This mode renders an existing synthetic result, without rerunning fits.
        result = json.loads((out / "study_results.json").read_text(encoding="utf-8"))
        write_tables(out, result)
        write_figures(out)
        return result
    truth = truth_trajectory("A")
    t = truth.time_s
    oil_idx = np.arange(0, t.size, OIL_SAMPLE_STRIDE)
    cal_idx = calibration_indices(17, t)
    ambient = np.full(t.size, AMBIENT_CONSTANT_C)
    half = day_a_load(t + DT_S / 2)
    bound = cramer_rao_bound(t, truth.load_pu, ambient, TRUTH_PARAMS, oil_idx, cal_idx, 0.5)
    estimates = []
    for seed in range(400):
        rng = np.random.default_rng(2000 + seed)
        oil = np.full(t.size, np.nan)
        oil[oil_idx] = truth.top_oil_C[oil_idx] + rng.normal(0, 0.5, oil_idx.size)
        refs = truth.hotspot_C[cal_idx] + rng.normal(0, 0.5, cal_idx.size)
        estimates.append(identify(
            t, truth.load_pu, ambient, oil, HotspotReferences(t[cal_idx], refs),
            loss="linear", starts=CAMPAIGN_START, load_pu_half=half, ambient_C_half=ambient,
        ).params)
    vectors = np.vstack([p.as_vector() for p in estimates])
    truth_vector = TRUTH_PARAMS.as_vector()
    result = {
        "scope": "synthetic; Model C structure matched to truth; no operational data",
        "environment": {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
        "truth_vector_K_s_K_s":truth_vector.tolist(), "loss_ratio_R":TRUTH_PARAMS.loss_ratio_R,
        "noise_sd_K":0.5,"n_mc":400,"seed_start":2000,
        "crlb_sd_percent":bound.as_dict(), "correlation":bound.correlation.tolist(),
        "folded_ratio":efficiency_ratio(estimates,TRUTH_PARAMS,bound,convention="folded"),
        "sd_ratio":efficiency_ratio(estimates,TRUTH_PARAMS,bound,convention="std"),
        "bias_percent":dict(zip(bound.names,((vectors.mean(axis=0)-truth_vector)/truth_vector*100).tolist())),
        "schedules":{}, "transfer":{}, "corruptions":{},
    }
    for n in (5,9,17):
        result["schedules"][str(n)] = cramer_rao_bound(
            t, truth.load_pu, ambient, TRUTH_PARAMS, oil_idx, calibration_indices(n,t),0.5
        ).as_dict()
    for day in ("A","B","C"):
        comparison=day_transfer(day)
        result["transfer"][day]={name:{
            "mean_rmse_K":comparison.mean(name,"rmse_K"),
            "mean_peak_error_K":comparison.mean(name,"peak_error_K"),
            "largest_signed_peak_error_K":comparison.worst_peak(name),
        } for name in ("A","B","C")}
    for factory in ALL_SCENARIOS:
        scenario=run_scenario(factory())
        result["corruptions"][scenario.name]={
            "parameter_bias_percent":scenario.mean_parameter_errors_pct,
            "mean_rmse_K":scenario.mean_rmse_K,"mean_peak_error_K":scenario.mean_peak_error_K,
            "gate":scenario.gate,"n_seeds":scenario.n_seeds,
        }
    (out/"study_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    write_tables(out, result)
    write_figures(out)
    return result


def write_tables(out: Path, result: dict) -> None:
    """Write LaTeX table rows from synthetic results; errors are K or percent."""
    labels = (r"$\Delta\theta_{or}$", r"$\tau_o$", r"$\Delta\theta_{hr}$", r"$\tau_w$")
    rows = []
    for name, label in zip(result["crlb_sd_percent"], labels):
        values = [result[k][name] for k in ("crlb_sd_percent", "folded_ratio", "sd_ratio", "bias_percent")]
        rows.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    (out / "precision_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    labels = ("Baseline", "Oil drift", "Oil spikes", "Spikes, robust loss", "Integer rounding",
              "Current gain error", "Reference bias", "Ambient supplied", "Ambient omitted")
    rows = []
    for label, scenario in zip(labels, result["corruptions"].values()):
        rows.append(f"{label} & {scenario['mean_rmse_K']:.2f} & {scenario['mean_peak_error_K']:+.2f}" + r" \\")
    (out / "corruption_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_figures(out: Path) -> None:
    """Render original synthetic temperature [degC] and error [K] figures."""
    truth = truth_trajectory("A")
    t = truth.time_s
    cal_idx = calibration_indices(17, t)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"savefig.bbox":"tight"})
    fig,axes=plt.subplots(2,1,figsize=(3.45,3.8),sharex=True,constrained_layout=True)
    axes[0].plot(t/3600,truth.load_pu,color="#1B6CA8")
    axes[0].set_ylabel("Load [pu]")
    axes[1].plot(t/3600,truth.top_oil_C,label="True top-oil",color="#1B6CA8")
    axes[1].plot(t/3600,truth.hotspot_C,label="True hot-spot",color="#A94014")
    axes[1].scatter(t[cal_idx]/3600,truth.hotspot_C[cal_idx],s=14,marker="x",color="black",label="Reference times")
    axes[1].set(xlabel="Time [h]",ylabel="Temperature [degC]")
    axes[1].legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.02),
                   ncol=2, borderaxespad=0)
    fig.savefig(out/"calibration_design.pdf")
    plt.close(fig)

    curves=day_transfer_trajectories("C")
    fig,axes=plt.subplots(2,1,figsize=(3.45,3.8),sharex=True,constrained_layout=True)
    axes[0].plot(curves.time_h,curves.truth_hotspot_C,color="black",label="Truth",linewidth=1.4)
    for name,color in (("A","#A94014"),("B","#1B6CA8"),("C","#39734B")):
        axes[0].plot(curves.time_h,curves.mean_hotspot_C[name],color=color,label=f"Model {name}",linewidth=0.9)
        axes[1].plot(curves.time_h,curves.mean_hotspot_C[name]-curves.truth_hotspot_C,color=color,label=name)
    axes[0].set_ylabel("Hot-spot [degC]")
    axes[0].legend(fontsize=8.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02),
                   borderaxespad=0)
    axes[1].axhline(0,color="black",linewidth=0.5)
    axes[1].set(xlabel="Time [h]",ylabel="Mean prediction error [K]")
    fig.savefig(out/"synthetic_transfer.pdf")
    plt.close(fig)


def main() -> None:
    """CLI; generated outputs require an explicit destination directory."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--figures-only", action="store_true", help="Render tables/figures from existing study_results.json")
    args=parser.parse_args()
    result=reproduce(args.out, figures_only=args.figures_only)
    print(json.dumps({k:result[k] for k in ("crlb_sd_percent","folded_ratio","bias_percent","transfer")},indent=2))


if __name__=="__main__":
    main()
