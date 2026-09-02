"""E7 -- what instrument makes hot-spot LOCATION identifiable, and how cheap?

WHY THIS EXPERIMENT EXISTS
--------------------------
E6 settled the negative half empirically. Given histories that are bit-identical
in every external channel and differ only in the hidden location label, the best
probe reaches R^2 = -0.018 -- worse than predicting the mean. No estimator
extracts information a measurement does not contain, so the useful question is
not "which model?" but:

    WHAT MEASUREMENT WOULD MAKE LOCATION IDENTIFIABLE, AND WHAT WOULD IT COST?

That turns an impossibility into an instrument specification, which can be built.

TWO QUANTITIES THAT GET CONFLATED
---------------------------------
  hot-spot TEMPERATURE -- observable from outside. The production engine already
                          identifies it, at 1.55 K out-of-sample on a real unit.
                          No new hardware needed.
  hot-spot LOCATION    -- not observable from outside at any practical sensor
                          quality. Needs a sensor inside the winding.

WHY THIS SUPERSEDES A SINGLE-PARAMETER BOUND
--------------------------------------------
`corefield.observability` bounds location as a SCALAR parameter: every other
quantity, including how hot the hot spot actually is, is assumed known exactly.
That flatters any design badly, and the first run of this experiment showed how.
Optimising placement under the scalar bound put both probes ABOVE the hot spot,
at 0.97 and 1.00, and claimed 0.05 % of winding height from a single probe.

A probe reading high cannot, on its own, distinguish "the hot spot moved closer"
from "the hot spot got hotter". Those two are confounded, and a bound that
assumes the magnitude away cannot see the confounding.

So this experiment computes the JOINT Fisher information over

    (location, delta_theta_hr)

and reports the marginal standard deviation of location with magnitude treated
as unknown -- which is the situation any real deployment is in. The scalar bound
is reported alongside, to show the size of the flattery.

LABEL
-----
**(b)** Engineering analysis on the simplified 1-D axial model in
`corefield.observability`, under its stated assumptions. Not a validated field
result. No instrument has been built or tested. These are design numbers to be
checked, not measurements.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from corefield.observability import AxialWindingModel

#: Sensor noise [K]. The cost axis: cheaper sensing is noisier. 0.1 K is
#: laboratory-grade; 0.5 K is the module default and representative of good
#: practical instrumentation; 2.0 K is a cheap sensor in an awkward position.
NOISE_LEVELS_K: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
PROBE_COUNTS: tuple[int, ...] = (0, 1, 2, 3, 4)
TRUE_LOCATION: float = 0.90

USELESS_PERCENT: float = 25.0
ACTIONABLE_PERCENT: float = 10.0
EXCELLENT_PERCENT: float = 2.0

_STEP_LOC: float = 1e-4
_STEP_DHR: float = 1e-3


def _observations(model: AxialWindingModel, loc: float, dhr: float,
                  positions: tuple[float, ...] | None) -> np.ndarray:
    m = replace(model, delta_theta_hr_K=dhr)
    if positions is None or len(positions) == 0:
        return m.external_observations(loc)
    return np.concatenate([m.external_observations(loc),
                           m.internal_observations(loc, list(positions))])


def joint_location_std_percent(
    model: AxialWindingModel, positions: tuple[float, ...] | None,
    location: float = TRUE_LOCATION,
) -> tuple[float, float]:
    """Marginal location std with magnitude unknown, and with it known [%H].

    Returns (joint, scalar). The joint figure inverts the 2x2 Fisher matrix over
    (location, delta_theta_hr) and takes the location diagonal, so it carries the
    cost of not knowing how hot the hot spot is. The scalar figure assumes the
    magnitude away, and is reported only to show how much that assumption buys.
    """
    dhr = model.delta_theta_hr_K
    d_loc = (_observations(model, location + _STEP_LOC, dhr, positions)
             - _observations(model, location - _STEP_LOC, dhr, positions)) / (2 * _STEP_LOC)
    d_dhr = (_observations(model, location, dhr + _STEP_DHR, positions)
             - _observations(model, location, dhr - _STEP_DHR, positions)) / (2 * _STEP_DHR)

    scalar_fisher = float(d_loc @ d_loc) / model.sensor_noise_K**2
    scalar = float(np.sqrt(1.0 / scalar_fisher) * 100.0) if scalar_fisher > 1e-30 else float("inf")

    J = np.column_stack([d_loc, d_dhr])
    fim = (J.T @ J) / model.sensor_noise_K**2
    # A singular or near-singular matrix means the two parameters are not jointly
    # identifiable from this configuration -- report that rather than a number.
    if not np.all(np.isfinite(fim)) or np.linalg.cond(fim) > 1e12:
        return float("inf"), scalar
    cov = np.linalg.inv(fim)
    if cov[0, 0] <= 0:
        return float("inf"), scalar
    return float(np.sqrt(cov[0, 0]) * 100.0), scalar


def _verdict(std_percent: float) -> str:
    if not np.isfinite(std_percent) or std_percent >= USELESS_PERCENT:
        return "useless"
    if std_percent <= EXCELLENT_PERCENT:
        return "excellent"
    if std_percent <= ACTIONABLE_PERCENT:
        return "actionable"
    return "weak"


@dataclass(frozen=True)
class Configuration:
    n_probes: int
    positions: tuple[float, ...]
    noise_K: float
    joint_percent: float
    scalar_percent: float
    verdict: str


def best_positions(n_probes: int, noise_K: float, *, grid: int = 21):
    """Search placements, ranking by the JOINT bound rather than the scalar one."""
    model = AxialWindingModel(sensor_noise_K=noise_K)
    if n_probes == 0:
        j, s = joint_location_std_percent(model, None)
        return (), j, s
    candidates = np.linspace(0.50, 1.00, grid)
    best = (float("inf"), float("inf"), ())
    for combo in itertools.combinations(candidates, n_probes):
        pos = tuple(round(float(c), 3) for c in combo)
        j, s = joint_location_std_percent(model, pos)
        if np.isfinite(j) and j < best[0]:
            best = (j, s, pos)
    return best[2], best[0], best[1]


def run() -> dict:
    print("=" * 82)
    print("E7  INSTRUMENT DESIGN -- what makes hot-spot LOCATION identifiable")
    print("=" * 82)
    print(f"  hot spot at {TRUE_LOCATION:.2f} of winding height. Magnitude treated as UNKNOWN.")
    print(f"  verdicts on the joint bound: <={EXCELLENT_PERCENT:.0f} % excellent | "
          f"<={ACTIONABLE_PERCENT:.0f} % actionable | >={USELESS_PERCENT:.0f} % useless")
    print()
    print(f"  {'probes':>6} {'noise K':>8} {'best positions':<26} "
          f"{'joint %H':>10} {'scalar %H':>11} {'verdict':>11}")
    print("  " + "-" * 78)

    rows: list[Configuration] = []
    for n in PROBE_COUNTS:
        for noise in NOISE_LEVELS_K:
            pos, j, s = best_positions(n, noise)
            v = _verdict(j)
            rows.append(Configuration(n, pos, noise, j, s, v))
            shown = "external only" if n == 0 else ", ".join(f"{p:.2f}" for p in pos)
            js = f"{j:>10.2f}" if np.isfinite(j) else f"{'inf':>10}"
            print(f"  {n:>6} {noise:>8.2f} {shown:<26} {js} {s:>11.2f} {v:>11}")
        print()

    print("=" * 82)
    print("WHAT THE JOINT BOUND CHANGES")
    print("=" * 82)
    for n in PROBE_COUNTS:
        r = next(x for x in rows if x.n_probes == n and x.noise_K == 0.5)
        ratio = (r.joint_percent / r.scalar_percent) if np.isfinite(r.joint_percent) and r.scalar_percent > 0 else float("inf")
        rs = f"{ratio:.1f}x worse" if np.isfinite(ratio) else "not identifiable"
        print(f"  {n} probe(s) at 0.5 K: scalar {r.scalar_percent:7.2f} %H -> "
              f"joint {r.joint_percent if np.isfinite(r.joint_percent) else float('inf'):7.2f} %H   ({rs})")
    print()
    print("  Not knowing how hot the hot spot is costs real precision. A design")
    print("  chosen under the scalar bound is chosen against the wrong objective.")

    print()
    print("=" * 82)
    print("THE SPECIFICATION")
    print("=" * 82)
    ok = [r for r in rows if r.verdict in ("actionable", "excellent")]
    if not ok:
        print("  Nothing tested reaches an actionable joint bound.")
    else:
        cheapest = max(ok, key=lambda r: (r.noise_K, -r.n_probes))
        fewest = min(ok, key=lambda r: (r.n_probes, -r.noise_K))
        print(f"  Loosest sensor that still works : {cheapest.n_probes} probe(s) at "
              f"{cheapest.positions}, {cheapest.noise_K} K -> {cheapest.joint_percent:.2f} %H")
        print(f"  Fewest probes that still work   : {fewest.n_probes} probe(s) at "
              f"{fewest.positions}, {fewest.noise_K} K -> {fewest.joint_percent:.2f} %H")
    print()
    print("  External channels only, joint bound, every sensor quality tested:")
    for r in [x for x in rows if x.n_probes == 0]:
        js = f"{r.joint_percent:8.2f}" if np.isfinite(r.joint_percent) else "     inf"
        print(f"    noise {r.noise_K:>4} K -> {js} %H  ({r.verdict})")

    return {
        "label": "(b) engineering analysis, simplified 1-D axial model, not validated",
        "bound": "joint 2-parameter CRLB over (location, delta_theta_hr); scalar reported for contrast",
        "true_location_fraction": TRUE_LOCATION,
        "thresholds_percent_of_height": {
            "excellent": EXCELLENT_PERCENT, "actionable": ACTIONABLE_PERCENT,
            "useless": USELESS_PERCENT,
        },
        "configurations": [
            {"n_probes": r.n_probes, "positions": list(r.positions), "noise_K": r.noise_K,
             "joint_location_std_percent_of_height": r.joint_percent,
             "scalar_location_std_percent_of_height": r.scalar_percent,
             "verdict": r.verdict}
            for r in rows
        ],
    }


def main() -> None:
    out = run()
    d = Path("runs/e7")
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print()
    print("written: runs/e7/aggregate.json")


if __name__ == "__main__":
    main()
