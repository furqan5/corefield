"""E11 -- does eddy-current tank heating kill the camera concept?

THE OBJECTION, AND IT IS A GOOD ONE
-----------------------------------
E8 to E10 all assume the tank wall is a readout of the oil column:

    wall(z) = coupling * oil(z) + (1 - coupling) * ambient

An adversarial review pointed out that this is false whenever leakage flux
induces eddy currents directly in the tank plate. That heat is generated IN THE
WALL, not conducted from the oil, so it bypasses the entire mechanism the
inversion relies on.

**(a) The objection is well founded.** The stray-loss literature reports that in
large power transformers more than 20 per cent of total load loss is stray loss
in structural components, with the largest share in the tank itself, and that
flux linking the tank near high-current bushings overheats the wall locally.
Shielding papers exist precisely because tank hot spots are a real design
problem.

WHAT DECIDES WHETHER IT IS FATAL
--------------------------------
Eddy heating enters as an additive term with its own spatial shape:

    wall(z) = A + B*f(z; z_hot) + E(z)*K**2

Three questions, in increasing order of what the product actually needs:

  1. Is location identifiable at ONE load, with E(z) unknown? If the eddy shape
     can imitate the profile shape, no.
  2. Does measuring at SEVERAL loads help? The eddy term scales as K**2 while
     the oil term scales as the oil rise, roughly K**(2x) with x = 0.8, so the
     two have different load signatures even though both grow with load.
  3. Does DIFFERENCING over time help? For a given transformer the eddy pattern
     is fixed by geometry. If it is constant while the hot spot moves, the
     difference between two surveys cancels it exactly.

Question 3 matters most, because hot-spot MIGRATION is the condition-monitoring
signal, and a migration detector does not need to know absolute position.

**(b)** The eddy spatial shape is represented by a polynomial family, as in E10.
Real leakage-flux heating is concentrated near winding ends and bushings and is
not a low-order polynomial; this is a stand-in chosen to answer the
identifiability question, not a flux model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corefield.observability import AxialWindingModel

TRUE_Z = 0.90
_SZ = 1e-4
#: IEC oil exponent: the oil rise grows as K**(2x) while eddy loss grows as K**2.
OIL_X = 0.8


def _f(model, z_loc, heights):
    prof = model.oil_profile(z_loc)
    return np.interp(heights, model.height,
                     (prof - model.bottom_oil_C) / (model.top_oil_C - model.bottom_oil_C))


def single_load_bound(model, heights, scale_K, eddy_order: int) -> float:
    """Location std [%H] at one load, with an unknown eddy shape of given order."""
    f0 = _f(model, TRUE_Z, heights)
    d_loc = scale_K * (_f(model, TRUE_Z + _SZ, heights)
                       - _f(model, TRUE_Z - _SZ, heights)) / (2 * _SZ)
    cols = [d_loc, np.ones_like(heights), f0]
    cols += [heights ** k for k in range(1, eddy_order + 1)]
    J = np.column_stack(cols)
    fim = (J.T @ J) / model.sensor_noise_K**2
    if J.shape[0] < J.shape[1] or np.linalg.cond(fim) > 1e12:
        return float("inf")
    return float(np.sqrt(np.linalg.inv(fim)[0, 0]) * 100.0)


def multi_load_bound(model, heights, loads, span_K, eddy_order: int) -> float:
    """Location std [%H] from several loads sharing one eddy shape.

    Stacks the observations from each load. The eddy columns are shared across
    loads and weighted by K**2; the profile columns are weighted by the oil
    rise, K**(2x). Different exponents are what makes separation possible.
    """
    f0 = _f(model, TRUE_Z, heights)
    df = (_f(model, TRUE_Z + _SZ, heights) - _f(model, TRUE_Z - _SZ, heights)) / (2 * _SZ)
    n, L = heights.size, len(loads)
    n_eddy = eddy_order
    J = np.zeros((n * L, 3 + n_eddy))
    for i, K in enumerate(loads):
        oil_w = K ** (2 * OIL_X)
        eddy_w = K ** 2
        s = slice(i * n, (i + 1) * n)
        J[s, 0] = span_K * oil_w * df          # location
        J[s, 1] = 1.0                          # offset (per-survey, shared here)
        J[s, 2] = oil_w * f0                   # scale
        for k in range(1, n_eddy + 1):
            J[s, 2 + k] = eddy_w * heights ** k
    fim = (J.T @ J) / model.sensor_noise_K**2
    if J.shape[0] < J.shape[1] or np.linalg.cond(fim) > 1e12:
        return float("inf")
    return float(np.sqrt(np.linalg.inv(fim)[0, 0]) * 100.0)


def migration_bound(model, heights, scale_K, eddy_order: int) -> float:
    """Std [%H] on a CHANGE in location between two surveys at equal load.

    The eddy pattern is a property of the geometry, so at equal load it is
    identical in both surveys and cancels in the difference. What survives is
    the profile change, plus a doubled noise variance from differencing two
    independent measurements.
    """
    d_loc = scale_K * (_f(model, TRUE_Z + _SZ, heights)
                       - _f(model, TRUE_Z - _SZ, heights)) / (2 * _SZ)
    # Differencing removes anything constant between surveys: the eddy shape,
    # the offset and the scale all cancel. Only a CHANGE in them survives, and
    # a change in gain is retained as a nuisance because emissivity can drift.
    cols = [d_loc, np.ones_like(heights), _f(model, TRUE_Z, heights)]
    J = np.column_stack(cols)
    fim = (J.T @ J) / (2.0 * model.sensor_noise_K**2)   # two noisy surveys
    if J.shape[0] < J.shape[1] or np.linalg.cond(fim) > 1e12:
        return float("inf")
    return float(np.sqrt(np.linalg.inv(fim)[0, 0]) * 100.0)


def run() -> dict:
    base = AxialWindingModel(sensor_noise_K=0.05)
    span = base.top_oil_C - base.bottom_oil_C
    out: dict = {"label": "(b) polynomial stand-in for leakage-flux tank heating"}

    print("=" * 84)
    print("E11  EDDY-CURRENT TANK HEATING")
    print("=" * 84)
    print("  (a) Stray loss is over 20 % of load loss in large units, mostly in the tank.")
    print("      Heat generated IN the wall bypasses the oil-column mechanism entirely.")
    print()

    print("-" * 84)
    print("1. ONE SURVEY, eddy shape unknown")
    print("-" * 84)
    print(f"  {'rows':>5} " + " ".join(f"order {p:<3}" for p in range(0, 6)))
    one = []
    for n in (32, 64, 128, 256):
        hs = np.linspace(0.02, 0.98, n)
        cells = []
        for P in range(0, 6):
            v = single_load_bound(base, hs, span, P)
            one.append({"rows": n, "eddy_order": P, "percent": v})
            cells.append(f"{v:8.2f}" if np.isfinite(v) and v < 1e4 else "     inf")
        print(f"  {n:>5} " + " ".join(cells))
    out["single_survey"] = one
    print("  Order 0 is E9. Each extra order is a richer unknown eddy pattern.")

    print()
    print("-" * 84)
    print("2. SEVERAL LOADS, one shared eddy shape")
    print("-" * 84)
    print("  Eddy scales as K^2, oil rise as K^1.6. Different exponents separate them.")
    print(f"  {'loads':>28} " + " ".join(f"order {p:<3}" for p in range(1, 6)))
    multi = []
    hs = np.linspace(0.02, 0.98, 64)
    for loads in ([0.5, 1.0], [0.4, 0.7, 1.0], [0.3, 0.5, 0.7, 0.9, 1.1]):
        cells = []
        for P in range(1, 6):
            v = multi_load_bound(base, hs, loads, span, P)
            multi.append({"loads": loads, "eddy_order": P, "percent": v})
            cells.append(f"{v:8.2f}" if np.isfinite(v) and v < 1e4 else "     inf")
        print(f"  {str(loads):>28} " + " ".join(cells))
    out["multi_load"] = multi

    print()
    print("-" * 84)
    print("3. MIGRATION -- change in location between two surveys at equal load")
    print("-" * 84)
    print("  The eddy pattern is geometry, so at equal load it cancels in the")
    print("  difference no matter how complicated it is.")
    print(f"  {'rows':>5} {'migration std %H':>18}")
    mig = []
    for n in (32, 64, 128, 256):
        hs = np.linspace(0.02, 0.98, n)
        v = migration_bound(base, hs, span, 0)
        mig.append({"rows": n, "percent": v})
        print(f"  {n:>5} {v:>18.2f}")
    out["migration"] = mig
    print("  Noise variance is doubled for two independent surveys, and nothing")
    print("  else is assumed about the eddy shape at all.")

    return out


def main() -> None:
    out = run()
    d = Path("runs/e11")
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print()
    print("written: runs/e11/aggregate.json")


if __name__ == "__main__":
    main()
