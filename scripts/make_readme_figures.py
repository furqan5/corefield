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

"""Regenerate the figures embedded in README.md.

    python scripts/make_readme_figures.py

Both are computed from the package at run time, so a README figure can never
drift from the code it illustrates. Requires the app extra.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from corefield.campaign import day_transfer_trajectories  # noqa: E402

STYLE = {
    "truth": dict(color="#111111", lw=3.2, alpha=0.5),
    "A": dict(color="#C1440E", lw=1.7, ls="--"),
    "B": dict(color="#E09F3E", lw=1.7, ls="-."),
    "C": dict(color="#1B6CA8", lw=2.2),
}
LABELS = {
    "A": "Model A — single-exponential, $K^2$ drive",
    "B": "Model B — single-exponential, free exponent",
    "C": "Model C — IEC two-exponential (production)",
}


def day_c_figure(destination: Path) -> Path:
    """The extrapolation result: three models asked about 1.30 pu."""
    d = day_transfer_trajectories("C")

    fig, (ax_load, ax, ax_err) = plt.subplots(
        3, 1, figsize=(10.5, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.8, 1.4], "hspace": 0.10},
    )
    fig.patch.set_facecolor("white")

    ax_load.fill_between(d.time_h, 0.6, 1.2, color="#2E7D32", alpha=0.10)
    ax_load.plot(d.time_h, d.load_pu, color="#333", lw=1.9)
    ax_load.axhline(1.30, color="#C1440E", ls=":", lw=1.3)
    ax_load.text(0.35, 1.325, "1.30 pu emergency loading", fontsize=9, color="#C1440E")
    # Placed late in the day, where the load trace sits well above it.
    ax_load.text(12.6, 0.615, "hull the models were fitted over, 0.6–1.2 pu",
                 fontsize=9, color="#2E7D32")
    ax_load.set_ylabel("load\n[pu]", fontsize=11)
    ax_load.set_ylim(0.55, 1.48)
    ax_load.grid(alpha=0.25)
    ax_load.set_title(
        "Fitted on an ordinary day. Asked about a 2-hour emergency overload.",
        fontsize=13.5, fontweight="bold", pad=11,
    )

    ax.plot(d.time_h, d.truth_hotspot_C, label="true hot spot", **STYLE["truth"])
    for name in ("A", "B", "C"):
        ax.plot(d.time_h, d.mean_hotspot_C[name], label=LABELS[name], **STYLE[name])
    ax.set_ylabel("hot-spot temperature [°C]", fontsize=11.5)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(alpha=0.25)

    for name in ("A", "B", "C"):
        err = d.mean_hotspot_C[name] - d.truth_hotspot_C
        ax_err.plot(d.time_h, err, **STYLE[name])
    ax_err.axhline(0, color="#333", lw=1.1)
    ax_err.fill_between(d.time_h, -2, 2, color="#999", alpha=0.15)
    ax_err.text(0.35, 2.4, "±2 K pre-registered gate", fontsize=9, color="#666")
    peak_h = float(d.time_h[int(np.argmax(d.truth_hotspot_C))])
    for name, dy in (("A", 1.1), ("B", -0.4), ("C", -2.4)):
        ax_err.annotate(
            f"{d.worst_peak_error_K[name]:+.2f} K worst case",
            xy=(peak_h, d.peak_error_K[name]),
            xytext=(peak_h - 8.4, d.peak_error_K[name] + dy),
            fontsize=9.5, fontweight="bold", color=STYLE[name]["color"],
            arrowprops=dict(arrowstyle="-|>", color=STYLE[name]["color"], lw=1.4),
        )
    ax_err.set_ylabel("error\n[K]", fontsize=11)
    ax_err.set_xlabel("time [h]", fontsize=11.5)
    ax_err.set_xlim(0, 24)
    ax_err.set_ylim(-4.2, 8.2)
    ax_err.grid(alpha=0.25)
    ax_err.text(
        0.985, 0.06, "above zero = reads HIGH = false derating",
        transform=ax_err.transAxes, ha="right", fontsize=9.5, color="#C1440E",
        fontweight="bold",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    print("Wrote", day_c_figure(root / "docs" / "day_c_extrapolation.png"))
    from make_linkedin_figure import render  # noqa: E402

    print("Wrote", render(root / "docs" / "hotspot_location_invariance.png"))
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
