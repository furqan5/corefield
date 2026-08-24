# WITHDRAWN — retracted results

Retractions are kept here permanently, with the numbers that were wrong and the reason they were
wrong. A withdrawn result that disappears from the record is indistinguishable from one that was
never made.

---

## v1.4 — geometry PINN for spatial hot-spot reconstruction

**Status: WITHDRAWN. Do not resurrect.**

### What was claimed

A physics-informed neural network trained against a finite-element reference solution would
reconstruct the transformer's internal temperature *field* and identify the spatial location of
the winding hot spot, extending the 0-D parameter identification to 2-D and eventually 3-D
geometry.

### Why it was withdrawn

**The finite-element reference solver failed mesh convergence.** A converging solver produces
answers that stabilise as the mesh is refined. This one diverged, badly:

| Mesh | Reported hot-spot temperature |
|---|---|
| 61 × 49 | 161.45 °C |
| 121 × 97 | 513.18 °C |
| 241 × 193 | **1960.20 °C** |

Roughly a factor of 3–4 per refinement, with no sign of settling. The final figure is above the
melting point of copper — the solution was not merely inaccurate, it was unphysical.

**Cause: element area cancelled in the stiffness assembly.** The element stiffness contributions
were assembled without their area weighting carried correctly, so the assembled system had no
consistent mesh-size scaling. Refining the mesh then changed the effective conductivity of the
discretised problem rather than converging toward the continuum solution. Every temperature the
solver produced was a function of the mesh, not of the physics.

### Why this invalidated the whole branch, not just one number

The PINN was to be trained and validated **against that solver**. A neural network trained on a
reference that does not converge learns the reference's error, and no amount of network
architecture, loss weighting or collocation strategy detects that — the training loss goes down
regardless. There was no ground truth, so there was nothing the branch could have been shown to
be right about.

### What was salvaged

Nothing from the field-reconstruction work. The 0-D parameter-identification track was
unaffected, because it never depended on the finite-element solver — its ground truth is an
ODE integration verified three independent ways (closed-form agreement to 1.10×10⁻⁷ K, the 63.2 %
oil low-pass point, and the 47.2 % gradient overshoot).

### A second, independent reason not to resurrect it

Even with a correct finite-element solver, the branch's stated goal — locating the hot spot from
external or sparsely-instrumented data — is **information-theoretically impossible**. This was
established in August 2026 and is implemented in `corefield.observability`:

- Top-oil temperature is **exactly invariant** to hot-spot location. Moving the hot spot from
  10 % to 90 % of winding height changes the reading by 0.0000000 K.
- The mechanism is conservation of energy, not numerical smallness: every external measurement is
  a function of *total* winding loss, and relocating the hot spot changes only its distribution.
  Location lies in the exact null space of the external observation map.
- The Cramér–Rao bound on location from external sensors is **± 40 % of winding height** — no
  information at all about a hot spot occupying the top 10 %.
- With two internal fibre-optic probes the bound is ± 0.33 % — but internal probes can only be
  fitted during manufacture, and existing practice already places them near 90 % of winding
  height by design analysis.

So the fix for the mesh bug would have bought a working solver for a problem with no recoverable
answer. See [ASSESSMENT.md](ASSESSMENT.md) for the full analysis and the prior-art review.

### Standing rule

Per `CLAUDE.md`: **no neural networks, PINNs, or deep learning in this repository.** The
production engine is classical nonlinear least squares, and that is a measured result — under
structural mismatch the alternatives read the hot spot several kelvin high at overload, which
triggers derating exactly when capacity is worth the most.

---

## Withdrawn claim — "all four parameters recovered at 0.99–1.02× CRLB"

**Status: WITHDRAWN and replaced with a stronger, supported claim.**

### What was claimed

That the classical estimator recovered all four IEC thermal parameters at 0.99–1.02× the
Cramér–Rao bound.

### Why it was withdrawn

No four-parameter efficiency table existed in any source document. The 0.99–1.02× figure came
from a **two-parameter** table (Δθ_hr and τ_w only) computed on the **older single-exponential**
truth model — not the production engine — and it covered **6 of that table's 8 cells**. The other
two were 1.13× and 0.68×. The phrase "across the full grid" contradicted the very table it was
drawn from.

The claim then strengthened at each restatement across four documents until it reached
"all four at 0.99–1.02×", which no measurement supported.

### What replaced it

The four-parameter efficiency computed properly, on the production model, at a seed count that
can actually resolve it:

> On the IEC two-exponential structure, the four-parameter classical estimator is unbiased to
> better than 0.12 % and sits on the Cramér–Rao bound — **0.97 / 1.01 / 0.95 / 0.97×** the
> folded-Gaussian expectation at 400 seeds.

A related methodological error was also corrected: a ±5 % band on this ratio is **unachievable at
10 seeds by any estimator**, because the ratio is a sample statistic whose own standard error is
`0.7555/√n` = ±0.24 at n = 10. The original 10-seed scatter (0.755 to 1.299) was entirely
sampling noise.

See [REPRODUCTION.md](REPRODUCTION.md) for the full derivation, and `tests/test_crlb.py` for the
regression that pins it.
