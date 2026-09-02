"""Benchmark and mesh-convergence verification for the Boussinesq solver.

Nothing about a transformer is computed until this passes. The withdrawn
geometry-PINN branch failed exactly here -- its reference solver gave 161, 513
and 1960 degC on successive refinements and the divergence went unnoticed.

The test is the de Vahl Davis differentially heated square cavity: air at
Pr = 0.71, hot wall at T = 1, cold wall at T = 0, adiabatic top and bottom,
no-slip everywhere, and the reported quantity is the average Nusselt number on
the hot wall.

**(c) On the reference values.** The de Vahl Davis 1983 paper is paywalled and
its table could not be retrieved. The figures widely quoted from it are about
1.118, 2.243, 4.519 and 8.800 for Ra = 1e3 to 1e6. Those are recorded here as
UNVERIFIED recollection, not as a source. What this script establishes on its
own authority is mesh convergence -- that the solver settles to a value as the
mesh is refined instead of running away. Agreement with the quoted figures is
reported for interest and is not the pass criterion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from skfem import MeshTri

from corefield_ml_lab.boussinesq import solve_boussinesq, wall_nusselt

HOT = lambda x: x[0] < 1e-9          # noqa: E731
COLD = lambda x: x[0] > 1 - 1e-9     # noqa: E731

#: (c) UNVERIFIED recollection of the commonly quoted benchmark values.
QUOTED_UNVERIFIED = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}

#: Under-relaxation and iteration budget rise with Rayleigh number; Picard
#: stalls at 1e5 without them, which is a solver property and not a physics one.
SETTINGS = {1e3: (0.80, 200), 1e4: (0.60, 400), 1e5: (0.20, 2000)}


def main(refinements=(3, 4, 5, 6)) -> None:
    out = {"note": "mesh convergence is the pass criterion; quoted values are unverified",
           "quoted_unverified": {str(k): v for k, v in QUOTED_UNVERIFIED.items()},
           "runs": []}
    print("=" * 78)
    print("MESH CONVERGENCE -- de Vahl Davis cavity, Pr = 0.71")
    print("=" * 78)
    for Ra in (1e3, 1e4, 1e5):
        relax, mx = SETTINGS[Ra]
        print(f"\nRa = {Ra:.0e}   (relaxation {relax}, max {mx} Picard iterations)")
        print(f"  {'refine':>6} {'cells':>7} {'dofs':>8} {'iters':>6} {'conv':>6} "
              f"{'Nu':>9} {'change':>9}")
        prev = None
        for ref in refinements:
            m = MeshTri().refined(ref)
            s = solve_boussinesq(m, rayleigh=Ra, prandtl=0.71, hot_wall=HOT,
                                 cold_wall=COLD, max_iterations=mx,
                                 tolerance=1e-9, relaxation=relax)
            nu = abs(wall_nusselt(s, HOT))
            ch = "" if prev is None else f"{abs(nu - prev) / prev * 100:+8.2f}%"
            dofs = s.vel_basis.N + s.pre_basis.N + s.tem_basis.N
            print(f"  {ref:>6} {m.t.shape[1]:>7} {dofs:>8} {s.iterations:>6} "
                  f"{str(s.converged):>6} {nu:>9.4f} {ch:>9}")
            out["runs"].append({"rayleigh": Ra, "refine": ref, "cells": int(m.t.shape[1]),
                                "dofs": int(dofs), "iterations": s.iterations,
                                "converged": bool(s.converged), "nusselt": nu,
                                "change_percent": None if prev is None
                                else abs(nu - prev) / prev * 100})
            prev = nu
            sys.stdout.flush()
        q = QUOTED_UNVERIFIED[Ra]
        print(f"  finest {prev:.4f} against the quoted (UNVERIFIED) {q:.3f}: "
              f"{abs(prev - q) / q * 100:+.1f}%")

    d = Path("runs/verification")
    d.mkdir(parents=True, exist_ok=True)
    (d / "boussinesq_convergence.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwritten: runs/verification/boussinesq_convergence.json")


if __name__ == "__main__":
    main()
