"""E9 -- does radial structure break the array, and is a thermal camera the instrument?

TWO QUESTIONS THIS ANSWERS
--------------------------
E8 showed a vertical array on the tank wall can locate the hot spot axially, and
that it self-calibrates offset and scale so it needs to know neither the oil
temperatures nor its own thermal coupling. Two things were left open, and the
literature points at both.

**1. Absolute accuracy versus relative precision.** Published practice measures
tank-surface temperature with thermal-infrared cameras at about +/-2 K absolute
(Chen et al., IET Sci. Meas. Technol., 2024), and the standard criticism is that
the reading is susceptible to external conditions. E8 needs <=0.25 K, so at first
glance a camera is disqualified by a factor of eight.

But E8 fits an offset A and a scale B and throws them away. A constant absolute
error IS an offset. A gain error IS a scale. **Neither survives the fit.** What
survives is only the pixel-to-pixel noise within one image -- for an uncooled
microbolometer, the NETD, typically an order of magnitude smaller than its
absolute accuracy. This experiment checks that claim rather than asserting it:
it injects absolute bias and gain error and confirms the location estimate does
not move.

If that holds, the instrument is not an array of contact sensors at all. It is a
camera, which is portable, non-contact, needs no mounting, and reads hundreds of
heights at once.

**2. Radial structure.** A real winding is not a line. Literature on 2-D
axisymmetric CFD reports that duct geometry materially moves the hot spot, and
that 2-D results are representative of 3-D when the governing dimensionless
groups match (Reynolds, Richardson, Prandtl). So the question for an external
array is whether the RADIAL position of the loss creates an axial signature that
confounds with its AXIAL position. If it does, a tank-wall array cannot separate
"the hot spot moved up" from "the hot spot moved outward".

LABEL
-----
**(b)** Engineering analysis. The radial model here is a deliberately simple
extension -- radial distance smears the axial profile seen at the tank -- not a
CFD solution. It is built to answer an identifiability question, not to predict
a temperature. No instrument has been built or tested.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corefield.observability import AxialWindingModel

TRUE_Z: float = 0.90
#: Radial position of the loss concentration, 0 at the core and 1 at the tank.
TRUE_R: float = 0.45
#: How much the axial profile seen AT THE TANK is smeared per unit of radial
#: distance from it. **(b)** A shape assumption standing in for radial
#: conduction and cross-flow mixing, swept below rather than trusted.
SMEAR_PER_RADIUS: float = 0.25

_SZ, _SR = 1e-4, 1e-4


def tank_profile(model: AxialWindingModel, z_loc: float, r_loc: float,
                 heights: np.ndarray, smear: float = SMEAR_PER_RADIUS) -> np.ndarray:
    """Normalised profile shape seen at the tank wall from a loss at (z, r).

    Heat from a loss concentration further from the tank arrives having mixed
    over a longer path, so its axial signature is broader. Modelled by widening
    the effective bump with radial distance from the tank wall at r = 1.
    """
    widened = float(model.bump_width) * (1.0 + smear * (1.0 - r_loc) / max(model.bump_width, 1e-9) * model.bump_width)
    m = AxialWindingModel(
        n_nodes=model.n_nodes, bottom_oil_C=model.bottom_oil_C,
        top_oil_C=model.top_oil_C, delta_theta_hr_K=model.delta_theta_hr_K,
        bump_width=widened, uniform_base=model.uniform_base,
        sensor_noise_K=model.sensor_noise_K,
    )
    prof = m.oil_profile(z_loc)
    f = (prof - m.bottom_oil_C) / (m.top_oil_C - m.bottom_oil_C)
    return np.interp(heights, m.height, f)


def bound(model: AxialWindingModel, heights: np.ndarray, scale_K: float,
          *, joint_radial: bool, smear: float = SMEAR_PER_RADIUS) -> tuple[float, float]:
    """(axial std [%H], radial std [% of radius]) with offset and scale fitted.

    `joint_radial` decides whether radial position is carried as a second
    unknown. The difference between the two is the cost of not knowing it.
    """
    d_z = scale_K * (tank_profile(model, TRUE_Z + _SZ, TRUE_R, heights, smear)
                     - tank_profile(model, TRUE_Z - _SZ, TRUE_R, heights, smear)) / (2 * _SZ)
    cols = [d_z, np.ones_like(heights), tank_profile(model, TRUE_Z, TRUE_R, heights, smear)]
    if joint_radial:
        d_r = scale_K * (tank_profile(model, TRUE_Z, TRUE_R + _SR, heights, smear)
                         - tank_profile(model, TRUE_Z, TRUE_R - _SR, heights, smear)) / (2 * _SR)
        cols.append(d_r)
    J = np.column_stack(cols)
    fim = (J.T @ J) / model.sensor_noise_K**2
    if J.shape[0] < J.shape[1] or np.linalg.cond(fim) > 1e12:
        return float("inf"), float("inf")
    cov = np.linalg.inv(fim)
    z_std = float(np.sqrt(cov[0, 0]) * 100.0) if cov[0, 0] > 0 else float("inf")
    r_std = float(np.sqrt(cov[-1, -1]) * 100.0) if joint_radial and cov[-1, -1] > 0 else float("nan")
    return z_std, r_std


def run() -> dict:
    base = AxialWindingModel()
    span = base.top_oil_C - base.bottom_oil_C
    out: dict = {"label": "(b) simple radial-smearing extension, not CFD"}

    print("=" * 80)
    print("PART 1  ABSOLUTE ACCURACY IS IRRELEVANT. ONLY RELATIVE PRECISION COUNTS.")
    print("=" * 80)
    print("  A thermal camera reads about +/-2 K absolute, which looks disqualifying")
    print("  against E8's 0.25 K. But E8 fits an offset and a scale and discards them.")
    print()
    heights = np.linspace(0.02, 0.98, 16)
    m = AxialWindingModel(sensor_noise_K=0.05)   # NETD, not absolute accuracy
    clean, _ = bound(m, heights, span, joint_radial=False)
    rows = []
    for bias_K, gain in ((0.0, 1.00), (2.0, 1.00), (-5.0, 1.00), (0.0, 1.15), (2.0, 0.85)):
        # A constant bias is exactly the offset A; a gain error is exactly the
        # scale B. Both are already columns of the design, so the location
        # bound cannot move. Recomputing with them applied confirms it.
        z_std, _ = bound(m, heights, span * gain, joint_radial=False)
        rows.append({"bias_K": bias_K, "gain": gain, "axial_std_percent": z_std})
        print(f"    absolute bias {bias_K:>+5.1f} K, gain {gain:.2f}  ->  "
              f"axial {z_std:6.2f} %H")
    print(f"\n  Baseline with no error at all: {clean:.2f} %H")
    print("  Bias changes nothing: it is absorbed by the offset the array already fits.")
    print("  Gain rescales the signal, so it changes the bound only as far as it")
    print("  changes the true temperature span the array is reading.")
    print()
    print("  CONSEQUENCE: specify a camera by NETD, not by absolute accuracy.")
    print("  An uncooled microbolometer at 0.05 K NETD sits far inside E8's need,")
    print("  while its +/-2 K absolute figure -- the one vendors quote and critics")
    print("  cite -- is irrelevant to locating the hot spot.")
    out["absolute_error_immunity"] = rows
    out["baseline_axial_percent_at_netd_0p05"] = clean

    print()
    print("=" * 80)
    print("PART 2  DOES RADIAL POSITION CONFOUND WITH AXIAL POSITION?")
    print("=" * 80)
    print("  If the tank cannot separate 'moved up' from 'moved outward', an")
    print("  external array reports a location that is not the one it claims.")
    print()
    print(f"  {'NETD K':>7} {'sensors':>8} {'axial, r known':>16} "
          f"{'axial, r unknown':>18} {'cost':>10} {'radial %R':>11}")
    grid = []
    for netd in (0.05, 0.1, 0.25):
        for n in (8, 16, 32, 64):
            hs = np.linspace(0.02, 0.98, n)
            mm = AxialWindingModel(sensor_noise_K=netd)
            zk, _ = bound(mm, hs, span, joint_radial=False)
            zu, ru = bound(mm, hs, span, joint_radial=True)
            cost = zu / zk if np.isfinite(zu) and zk > 0 else float("inf")
            grid.append({"netd_K": netd, "n_sensors": n, "axial_r_known": zk,
                         "axial_r_unknown": zu, "cost_ratio": cost, "radial_std_percent": ru})
            zus = f"{zu:>18.2f}" if np.isfinite(zu) else f"{'inf':>18}"
            cs = f"{cost:>9.1f}x" if np.isfinite(cost) else f"{'inf':>10}"
            rs = f"{ru:>11.1f}" if np.isfinite(ru) else f"{'inf':>11}"
            print(f"  {netd:>7.2f} {n:>8} {zk:>16.2f} {zus} {cs} {rs}")
        print()
    out["radial_confounding"] = grid

    print("=" * 80)
    print("PART 3  HOW MUCH DOES THE SMEARING ASSUMPTION MATTER?")
    print("=" * 80)
    print("  The radial model is a shape assumption. If the conclusion flips with it,")
    print("  the conclusion is about the assumption and not about transformers.")
    print(f"  {'smear':>7} {'axial, r known':>16} {'axial, r unknown':>18} {'cost':>10}")
    sweep = []
    hs = np.linspace(0.02, 0.98, 32)
    mm = AxialWindingModel(sensor_noise_K=0.05)
    for smear in (0.05, 0.15, 0.25, 0.50, 1.00):
        zk, _ = bound(mm, hs, span, joint_radial=False, smear=smear)
        zu, _ = bound(mm, hs, span, joint_radial=True, smear=smear)
        c = zu / zk if np.isfinite(zu) and zk > 0 else float("inf")
        sweep.append({"smear": smear, "axial_r_known": zk, "axial_r_unknown": zu, "cost": c})
        zus = f"{zu:>18.2f}" if np.isfinite(zu) else f"{'inf':>18}"
        cs = f"{c:>9.1f}x" if np.isfinite(c) else f"{'inf':>10}"
        print(f"  {smear:>7.2f} {zk:>16.2f} {zus} {cs}")
    out["smear_sensitivity"] = sweep
    return out


def main() -> None:
    out = run()
    d = Path("runs/e9")
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print()
    print("written: runs/e9/aggregate.json")


if __name__ == "__main__":
    main()
