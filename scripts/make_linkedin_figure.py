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

"""Render the hot-spot-location observability result as a figure.

    python scripts/make_linkedin_figure.py [output.png]

Shows the same winding with its hot spot near the bottom and near the top,
and the top-oil temperature that an external sensor reads in each case. The
two readings are identical, which is the whole point: location sits in the
exact null space of every measurement available outside the tank.

Every value drawn here is computed from `corefield.observability` at run
time. Nothing is typed into the plotting code, so the figure cannot drift
away from the model it claims to depict.

Requires the `app` extra: pip install -e ".[app]"
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from corefield.observability import (  # noqa: E402
    AxialWindingModel,
    external_location_bound,
    internal_location_bound,
)

LOW, HIGH = 0.10, 0.90
XLIM = (56, 126)
PROBES = (0.80, 0.95)


def render(destination: Path) -> Path:
    """Draw the figure and write it to `destination`."""
    model = AxialWindingModel()
    z = model.height
    external = external_location_bound()
    internal = internal_location_bound(list(PROBES))

    reading = model.external_observations(LOW)[0]
    if abs(reading - model.external_observations(HIGH)[0]) > 1e-9:
        raise AssertionError(
            "top-oil is no longer invariant to hot-spot location; the figure's "
            "central claim has stopped being true and must not be published."
        )

    def at(profile, height: float) -> float:
        return float(profile[int(np.argmin(np.abs(z - height)))])

    fig = plt.figure(figsize=(10, 10), dpi=120)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.958, "Move the hot spot 80% of the way up a transformer winding.",
             ha="center", fontsize=18.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.918, "The sensor outside the tank reads the same number.",
             ha="center", fontsize=18.5, color="#C1440E", fontweight="bold")

    for x0, location, colour, title in [
        (0.115, LOW, "#2E7D32", "Hot spot near the BOTTOM"),
        (0.545, HIGH, "#C1440E", "Hot spot near the TOP"),
    ]:
        ax = fig.add_axes([x0, 0.415, 0.355, 0.44])
        winding = model.winding_profile(location)
        oil = model.oil_profile(location)

        ax.fill_betweenx(z, oil, winding, color=colour, alpha=0.13, zorder=1)
        ax.plot(oil, z, color="#5B6B7A", lw=1.7, ls=":", zorder=2)
        ax.plot(winding, z, color=colour, lw=3.4, zorder=3)

        peak = int(np.argmax(winding))
        px, pz = float(winding[peak]), float(z[peak])
        ax.plot(px, pz, "o", ms=12, mfc=colour, mec="white", mew=2.2, zorder=5)

        # Place the label in space the curve does not occupy: right of the
        # curve when the hot spot is low, left of it when high.
        tx, tz = (px + 3.0, pz + 0.34) if location < 0.5 else (at(winding, 0.62) - 20.0, 0.60)
        ax.annotate("hot spot", xy=(px, pz), xytext=(tx, tz), fontsize=13,
                    fontweight="bold", color=colour, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                    shrinkA=6, shrinkB=8), zorder=6)

        ax.plot([reading], [1.0], "o", ms=12, mfc="white", mec="#111111", mew=2.6, zorder=6)
        ax.annotate("top-oil\nsensor", xy=(reading, 1.0), xytext=(reading - 20, 0.80),
                    fontsize=11.5, fontweight="bold", color="#111111", ha="center",
                    linespacing=1.3,
                    arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.7,
                                    shrinkA=4, shrinkB=7), zorder=7)

        ax.set_title(title, fontsize=14.5, fontweight="bold", color="#333333", pad=11)
        ax.set_xlim(*XLIM)
        ax.set_ylim(-0.04, 1.12)
        ax.set_yticks([0, 0.5, 1.0], ["bottom", "mid", "top"], fontsize=12)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.set_xlabel("temperature  [\u00b0C]", fontsize=12.5, labelpad=6)
        ax.grid(alpha=0.20, lw=0.9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if x0 > 0.4:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("height up the winding", fontsize=12.5, labelpad=8)
            ax.text(at(oil, 0.44) - 7.5, 0.44, "oil", fontsize=12, color="#5B6B7A",
                    fontweight="bold", rotation=66, va="center")
            ax.text(at(winding, 0.80) + 3.5, 0.80, "winding", fontsize=12,
                    color=colour, fontweight="bold", va="center")

    for centre in (0.293, 0.723):
        fig.text(centre, 0.338, "top-oil sensor reads", ha="center",
                 fontsize=12.5, color="#666666")
        fig.text(centre, 0.278, f"{reading:.6f} \u00b0C", ha="center", fontsize=26,
                 fontweight="bold", color="#111111")
    fig.text(0.5, 0.222, "identical to nine decimal places", ha="center",
             fontsize=14.5, style="italic", color="#C1440E")

    fig.patches.append(
        FancyBboxPatch((0.055, 0.032), 0.89, 0.135, transform=fig.transFigure,
                       boxstyle="round,pad=0.010", facecolor="#F4F6F8",
                       edgecolor="#DCE1E6", lw=1.3, zorder=0)
    )
    fig.text(0.5, 0.132, "Every external measurement depends on TOTAL winding loss.",
             ha="center", fontsize=13, color="#444444")
    fig.text(0.5, 0.093,
             f"Cram\u00e9r\u2013Rao bound on hot-spot location, from outside:  "
             f"\u00b1 {external.std_percent_of_height:.0f}% of winding height",
             ha="center", fontsize=14, fontweight="bold", color="#C1440E")
    fig.text(0.5, 0.052,
             f"From two fibre-optic probes inside:  "
             f"\u00b1 {internal.std_percent_of_height:.2f}%"
             f"     \u00b7     1-D axial model, synthetic     \u00b7     CoreField",
             ha="center", fontsize=11.5, color="#666666")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120, facecolor="white")
    plt.close(fig)
    return destination


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hotspot_invariance.png")
    written = render(target)
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
