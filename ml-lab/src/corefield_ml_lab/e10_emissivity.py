"""E10 -- do emissivity variation and reflections kill the thermal-camera route?

THE RISK, STATED PRECISELY
--------------------------
E9 proved that a UNIFORM absolute error and a UNIFORM gain error are absorbed
exactly, because the fit already carries an offset and a scale. That is why a
camera's +/-2 K absolute accuracy does not matter.

The documented hard problem with infrared on transformers is different and the
literature is explicit about it: "variable emissivity and multiple reflections in
fully metallic environments". Those are not uniform. A tank whose paint is worn
at the top, or which reflects a hot bushing across part of its face, presents an
error that VARIES WITH HEIGHT -- and height is exactly the axis carrying the
location signal.

THE QUESTION THAT DECIDES IT
----------------------------
The location signal is a smooth function of height. So is a smooth emissivity
drift. If the nuisance basis is rich enough to imitate the location signal, the
two become confounded and location dies no matter how good the camera is.

So the decisive question is not "is there emissivity error" -- there is -- but:

    HOW MANY HEIGHT-VARYING NUISANCE DEGREES OF FREEDOM CAN BE ABSORBED
    BEFORE LOCATION BECOMES UNIDENTIFIABLE?

Two families are swept, because emissivity and reflection enter differently:

  additive        z**k          reflections, stray radiance, ambient gradients
  multiplicative  z**k * f(z)   emissivity varying with height

and one localised case, because a reflection of a hot object is not smooth at
all: a Gaussian spike at an unknown height with unknown amplitude, which is the
worst case because it can imitate the hot spot itself.

**(b)** Engineering analysis on the E8/E9 model. No camera has been pointed at a
transformer. The polynomial families are a stand-in for real emissivity maps,
which are measurable and are not modelled here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corefield.observability import AxialWindingModel

TRUE_Z = 0.90
_SZ = 1e-4


def _f(model: AxialWindingModel, z_loc: float, heights: np.ndarray) -> np.ndarray:
    """Normalised profile shape f(z; z_loc)."""
    prof = model.oil_profile(z_loc)
    return np.interp(heights,
                     model.height,
                     (prof - model.bottom_oil_C) / (model.top_oil_C - model.bottom_oil_C))


def location_bound(model: AxialWindingModel, heights: np.ndarray, scale_K: float,
                   *, additive_order: int = 0, multiplicative_order: int = 0,
                   localised_spike: bool = False, spike_z: float = 0.80,
                   spike_width: float = 0.08) -> float:
    """Location std [%H] with the stated nuisance families fitted alongside.

    additive_order P adds columns z**0 .. z**P. Order 0 is the plain offset E9
    already carried, so P = 0 reproduces E9 exactly.

    multiplicative_order Q adds columns z**1..z**Q times f(z), which is the
    linearisation of an emissivity that varies with height. Order 0 is the plain
    scale, again already carried.

    localised_spike adds a Gaussian in height with unknown amplitude AND unknown
    position -- two columns -- which is the reflection of a hot object.
    """
    f0 = _f(model, TRUE_Z, heights)
    d_loc = scale_K * (_f(model, TRUE_Z + _SZ, heights)
                       - _f(model, TRUE_Z - _SZ, heights)) / (2 * _SZ)

    cols = [d_loc]
    cols += [heights ** k for k in range(additive_order + 1)]       # offset upward
    cols += [f0]                                                     # the scale
    cols += [(heights ** k) * f0 for k in range(1, multiplicative_order + 1)]
    if localised_spike:
        g = np.exp(-0.5 * ((heights - spike_z) / spike_width) ** 2)
        dg = g * (heights - spike_z) / spike_width**2                # d/d(spike_z)
        cols += [g, dg]

    J = np.column_stack(cols)
    fim = (J.T @ J) / model.sensor_noise_K**2
    if J.shape[0] < J.shape[1] or not np.all(np.isfinite(fim)) \
            or np.linalg.cond(fim) > 1e12:
        return float("inf")
    cov = np.linalg.inv(fim)
    return float(np.sqrt(cov[0, 0]) * 100.0) if cov[0, 0] > 0 else float("inf")


def run() -> dict:
    base = AxialWindingModel(sensor_noise_K=0.05)     # camera NETD
    span = base.top_oil_C - base.bottom_oil_C
    out: dict = {"label": "(b) polynomial stand-in for real emissivity maps"}

    print("=" * 86)
    print("E10  EMISSIVITY VARIATION AND REFLECTIONS")
    print("=" * 86)
    print("  Camera NETD 0.05 K. Location signal is smooth in height; so is a smooth")
    print("  emissivity drift. The question is where they become indistinguishable.")
    print()

    print("-" * 86)
    print("1. ADDITIVE nuisances -- reflections, stray radiance, ambient gradients")
    print("-" * 86)
    print(f"  {'rows':>5} " + " ".join(f"P={p:<6}" for p in range(0, 7)))
    add = []
    for n in (16, 32, 64, 128, 256):
        hs = np.linspace(0.02, 0.98, n)
        cells = []
        for P in range(0, 7):
            v = location_bound(base, hs, span, additive_order=P)
            add.append({"rows": n, "order": P, "percent": v})
            cells.append(f"{v:7.2f}" if np.isfinite(v) and v < 1e4 else "    inf")
        print(f"  {n:>5} " + " ".join(cells))
    out["additive"] = add
    print("  P=0 is the plain offset, i.e. the E9 result. Higher P is a richer")
    print("  height-varying additive error the fit must absorb.")

    print()
    print("-" * 86)
    print("2. MULTIPLICATIVE nuisances -- emissivity varying with height")
    print("-" * 86)
    print(f"  {'rows':>5} " + " ".join(f"Q={q:<6}" for q in range(0, 7)))
    mul = []
    for n in (16, 32, 64, 128, 256):
        hs = np.linspace(0.02, 0.98, n)
        cells = []
        for Q in range(0, 7):
            v = location_bound(base, hs, span, multiplicative_order=Q)
            mul.append({"rows": n, "order": Q, "percent": v})
            cells.append(f"{v:7.2f}" if np.isfinite(v) and v < 1e4 else "    inf")
        print(f"  {n:>5} " + " ".join(cells))
    out["multiplicative"] = mul

    print()
    print("-" * 86)
    print("3. BOTH AT ONCE -- a realistic camera on a weathered tank")
    print("-" * 86)
    print(f"  {'rows':>5} " + " ".join(f"P=Q={p:<4}" for p in range(0, 5)))
    both = []
    for n in (32, 64, 128, 256):
        hs = np.linspace(0.02, 0.98, n)
        cells = []
        for P in range(0, 5):
            v = location_bound(base, hs, span, additive_order=P, multiplicative_order=P)
            both.append({"rows": n, "order": P, "percent": v})
            cells.append(f"{v:7.2f}" if np.isfinite(v) and v < 1e4 else "    inf")
        print(f"  {n:>5} " + " ".join(cells))
    out["both"] = both

    print()
    print("-" * 86)
    print("4. LOCALISED REFLECTION -- a hot object mirrored on the tank")
    print("-" * 86)
    print("  Gaussian in height, amplitude AND position unknown. The worst case is")
    print("  a reflection sitting where the hot spot is, because then it imitates it.")
    print(f"  {'spike z':>8} {'rows=32':>10} {'rows=64':>10} {'rows=128':>10} {'rows=256':>10}")
    loc = []
    for sz in (0.20, 0.50, 0.70, 0.85, 0.90, 0.95):
        cells = []
        for n in (32, 64, 128, 256):
            hs = np.linspace(0.02, 0.98, n)
            v = location_bound(base, hs, span, additive_order=1,
                               multiplicative_order=1, localised_spike=True, spike_z=sz)
            loc.append({"spike_z": sz, "rows": n, "percent": v})
            cells.append(f"{v:10.2f}" if np.isfinite(v) and v < 1e4 else "       inf")
        print(f"  {sz:>8.2f} " + " ".join(cells))
    out["localised"] = loc
    print(f"  (hot spot is at z = {TRUE_Z:.2f}; a smooth tilt and gain drift are also fitted)")

    return out


def main() -> None:
    out = run()
    d = Path("runs/e10")
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print()
    print("written: runs/e10/aggregate.json")


if __name__ == "__main__":
    main()
