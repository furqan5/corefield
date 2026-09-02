"""E8 -- can a cheap array on the OUTSIDE of the tank locate the hot spot?

WHY THIS EXPERIMENT EXISTS
--------------------------
E6 and E7 closed two doors and it is worth being precise about which.

E6 showed a model cannot infer location from load, ambient and top oil. E7 showed
the four-channel external bundle in `corefield.observability` gives 56.6 % of
winding height at 0.5 K -- useless.

But that bundle collapses the entire tank to ONE scalar: the mixed-oil mean,
`trapezoid(oil, z)`. And the model's own docstring says the thing that matters:

    "Oil heats cumulatively as it rises past the losses, so the profile SHAPE
     depends on where those losses sit -- while its endpoints do not."

A single mean destroys shape. **An array of sensors up the tank wall samples it.**
That is a different observable, it has never been tested here, and it is exactly
what a cheap portable external instrument would measure.

So the question is not "can external sensing work" -- it is "how many external
sensors, how good, and how well coupled to the oil".

WHAT MAKES THIS HARDER THAN IT LOOKS
------------------------------------
A sensor on the tank wall does not read oil temperature. The wall sits between
the oil and the weather, so its reading is attenuated toward ambient:

    wall = coupling * oil + (1 - coupling) * ambient

`coupling` below 1 scales the location signal down directly. A well-insulated
tank in still air might reach 0.7; a windy site with sun on the tank is far
worse. This experiment sweeps it rather than assuming it.

**On common-mode error.** Solar gain, wind and ambient drift hit every sensor on
one tank at once. Those largely cancel when what you estimate is the SHAPE of the
profile, which is a difference between sensors. The Fisher calculation below uses
independent per-sensor noise, which is the right model for the part that does not
cancel. An array is therefore better placed against outdoor conditions than a
single external sensor is -- that is an argument for the array, not against it.

LABEL
-----
**(b)** Engineering analysis on the simplified 1-D axial model, under its stated
assumptions plus a linear wall-coupling model that is itself an assumption. Not a
validated field result. No instrument has been built or tested.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from corefield.observability import AxialWindingModel

SENSOR_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
NOISE_LEVELS_K: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
#: Fraction of the oil-to-ambient temperature difference that reaches the wall
#: sensor. 1.0 is a perfect thermal contact and is optimistic by construction.
COUPLINGS: tuple[float, ...] = (1.0, 0.7, 0.5, 0.3)

TRUE_LOCATION: float = 0.90
AMBIENT_C: float = 25.0

USELESS_PERCENT: float = 25.0
ACTIONABLE_PERCENT: float = 10.0
EXCELLENT_PERCENT: float = 2.0

_STEP_LOC: float = 1e-4
_STEP_DHR: float = 1e-3


def wall_array(model: AxialWindingModel, location: float, dhr: float,
               heights: np.ndarray, coupling: float) -> np.ndarray:
    """Tank-wall readings at normalised heights, attenuated toward ambient."""
    m = replace(model, delta_theta_hr_K=dhr)
    oil = np.interp(heights, m.height, m.oil_profile(location))
    return coupling * oil + (1.0 - coupling) * AMBIENT_C


def _shape(model: AxialWindingModel, loc: float, heights: np.ndarray) -> np.ndarray:
    """Normalised cumulative oil rise f(z; loc): 0 at the bottom, 1 at the top."""
    prof = model.oil_profile(loc)
    f = (prof - model.bottom_oil_C) / (model.top_oil_C - model.bottom_oil_C)
    return np.interp(heights, model.height, f)


def joint_location_std_percent(model: AxialWindingModel, heights: np.ndarray,
                               coupling: float, location: float = TRUE_LOCATION,
                               *, coupling_tilt: bool = False) -> float:
    """Marginal location std [%H] for a tank-wall array that knows nothing else.

    WHY THE NUISANCES ARE (OFFSET, SCALE) AND NOT WHAT E7 USED

    Two earlier versions of this function were wrong in the same way, and both
    wrongly reported that nothing works.

    The first carried `delta_theta_hr` as the nuisance, copying E7. But
    `oil_profile` does not depend on the rated gradient at all -- only
    `winding_profile` does -- so that column was exactly zero and made the
    information matrix singular. A parameter with no effect on the observable is
    absent, not confounded.

    The second carried oil endpoints AND wall coupling as three separate
    nuisances. Writing the observation out shows why that is rank-deficient:

        obs = c*[bot + (top-bot)*f(z;loc)] + (1-c)*amb
            = [c*bot + (1-c)*amb] + [c*(top-bot)] * f(z;loc)
            =         A           +       B       * f(z;loc)

    Bottom oil, top oil and coupling enter ONLY through the offset A and the
    scale B. Three unknowns, two degrees of freedom, so the nuisance block is
    singular by construction. That degeneracy sits entirely inside the
    nuisances; location enters through f(z;loc), a separate direction, and
    condemning it for a nuisance-only degeneracy was an analysis error.

    Parameterising by (location, A, B) is therefore both full rank and the
    honest description of a portable instrument: it knows neither the oil
    temperatures nor its own thermal contact, and fits both away.

    `coupling_tilt` adds the real risk. The A + B*f form assumes ONE coupling
    for every sensor. Radiators, fins, paint and sun on one side make contact
    vary with height, which eats directly into the shape signal that carries
    location. The tilt parameter is a linear-in-height coupling variation.
    """
    span = model.top_oil_C - model.bottom_oil_C
    B = coupling * span
    d_loc = B * (_shape(model, location + _STEP_LOC, heights)
                 - _shape(model, location - _STEP_LOC, heights)) / (2 * _STEP_LOC)
    cols = [d_loc, np.ones_like(heights), _shape(model, location, heights)]
    if coupling_tilt:
        # d(obs)/d(tilt) for obs = (A + t*(z-0.5)) + B*f: the offset acquires a
        # height-dependent term the array must separate from the profile shape.
        cols.append(heights - 0.5)

    J = np.column_stack(cols)
    fim = (J.T @ J) / model.sensor_noise_K**2
    if J.shape[0] < J.shape[1] or not np.all(np.isfinite(fim)) or np.linalg.cond(fim) > 1e12:
        return float("inf")
    cov = np.linalg.inv(fim)
    return float(np.sqrt(cov[0, 0]) * 100.0) if cov[0, 0] > 0 else float("inf")


def _verdict(p: float) -> str:
    if not np.isfinite(p) or p >= USELESS_PERCENT:
        return "useless"
    if p <= EXCELLENT_PERCENT:
        return "excellent"
    if p <= ACTIONABLE_PERCENT:
        return "actionable"
    return "weak"


@dataclass(frozen=True)
class Row:
    n_sensors: int
    coupling: float
    noise_K: float
    percent: float
    verdict: str


def run() -> dict:
    base = AxialWindingModel()

    print("=" * 84)
    print("E8  A CHEAP EXTERNAL ARRAY -- does sampling the tank profile locate the hot spot?")
    print("=" * 84)
    print("  The four-channel external bundle collapses the tank to one mixed-oil mean.")
    print("  An array samples the profile SHAPE, which is where the location signal lives.")
    print()

    # How big is the raw signal, before any sensor is chosen?
    z = base.height
    hi = base.oil_profile(TRUE_LOCATION + 0.05)
    lo = base.oil_profile(TRUE_LOCATION - 0.05)
    diff = hi - lo
    k = int(np.argmax(np.abs(diff)))
    print("  RAW SIGNAL, oil profile, for a 10 % shift in hot-spot location:")
    print(f"    largest change {np.max(np.abs(diff)):.3f} K, at height {z[k]:.2f}")
    print(f"    change at the top (z=1.00) {diff[-1]:+.4f} K   "
          f"at the bottom (z=0.00) {diff[0]:+.4f} K")
    print("    The endpoints barely move. That is the null space. The middle does move,")
    print("    and that is what an array can see and a single mean cannot.")
    print()

    rows: list[Row] = []
    for coupling in COUPLINGS:
        print(f"  wall coupling {coupling:.1f}  "
              f"(1.0 = sensor reads oil exactly; lower = attenuated toward ambient)")
        print(f"    {'sensors':>8} " + " ".join(f"{n:>9.2f}K" for n in NOISE_LEVELS_K))
        for n in SENSOR_COUNTS:
            heights = np.linspace(0.02, 0.98, n) if n > 1 else np.array([0.5])
            cells = []
            for noise in NOISE_LEVELS_K:
                m = replace(base, sensor_noise_K=noise)
                p = joint_location_std_percent(m, heights, coupling)
                rows.append(Row(n, coupling, noise, p, _verdict(p)))
                cells.append(f"{p:>9.1f} " if np.isfinite(p) and p < 1e4 else f"{'inf':>9} ")
            print(f"    {n:>8} " + " ".join(cells))
        print()

    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    ok = [r for r in rows if r.verdict in ("actionable", "excellent")]
    if not ok:
        print("  NOTHING tested reaches even an actionable bound.")
        best = min(rows, key=lambda r: r.percent)
        print(f"  Best case anywhere: {best.n_sensors} sensors, coupling {best.coupling}, "
              f"noise {best.noise_K} K -> {best.percent:.1f} %H ({best.verdict})")
        print("  And that best case assumes the most favourable coupling and the most")
        print("  expensive sensor in the sweep.")
    else:
        print(f"  {len(ok)} configuration(s) reach actionable or better:")
        for r in sorted(ok, key=lambda r: r.percent)[:12]:
            print(f"    {r.n_sensors:>3} sensors, coupling {r.coupling:.1f}, "
                  f"noise {r.noise_K:>4} K -> {r.percent:6.2f} %H ({r.verdict})")
        cheap = max(ok, key=lambda r: (r.noise_K, -r.n_sensors, -r.coupling))
        print()
        print(f"  Loosest sensor that still works: {cheap.n_sensors} sensors, "
              f"coupling {cheap.coupling:.1f}, noise {cheap.noise_K} K -> {cheap.percent:.2f} %H")

    print()
    print("  For contrast, from E7 with the magnitude also unknown:")
    print("    2 internal probes bracketing the hot spot, 2.0 K noise -> 1.07 %H")
    print("    the 4-channel external bundle,             0.5 K noise -> 56.62 %H")

    return {
        "label": "(b) simplified 1-D axial model plus an assumed linear wall-coupling model",
        "bound": "joint CRLB over (location, bottom_oil, top_oil); coupling variant adds it",
        "ambient_C": AMBIENT_C,
        "raw_signal_K_per_10pct_shift": float(np.max(np.abs(diff))),
        "raw_signal_peak_height": float(z[k]),
        "rows": [
            {"n_sensors": r.n_sensors, "coupling": r.coupling, "noise_K": r.noise_K,
             "location_std_percent_of_height": r.percent, "verdict": r.verdict}
            for r in rows
        ],
    }


def main() -> None:
    out = run()
    d = Path("runs/e8")
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print()
    print("written: runs/e8/aggregate.json")


if __name__ == "__main__":
    main()
