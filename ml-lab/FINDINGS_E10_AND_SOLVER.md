# The 2-D solver is verified, and emissivity has a hard limit

**2 September 2026.** Branch `ml-lab`. **Label (b)** for all transformer-facing numbers.

---

## Part 1 — the Boussinesq solver passes the check the withdrawn one failed

`WITHDRAWN.md` records a geometry-PINN branch lost because its finite-element reference solver gave
a hot spot of **161 → 513 → 1960 °C** under mesh refinement and nobody checked. So this solver was
built with a benchmark attached and nothing about a transformer was computed until it passed.

**Tooling:** `scikit-fem 12.0.2` — pure Python, pip-installable, no compiler, fits the 2 GB budget.
FEniCSx needs conda or Docker on Windows; FiPy is finite-volume and weak for saddle-point flow.

**Formulation:** steady Boussinesq, Taylor-Hood **P2 velocity / P1 pressure / P2 temperature**,
coupled momentum–continuity–energy, Picard iteration with under-relaxation. The axisymmetric form
carries the radial weight `r` on every integral **and the hoop term `u_r/r²`** in the radial
momentum equation — omitting that term is a standard silent error that leaves a solver looking
convergent.

**Benchmark:** de Vahl Davis differentially heated square cavity, Pr = 0.71.

| Ra | refine 3 | 4 | 5 | **6** | successive change |
|---|---|---|---|---|---|
| 10³ | 1.1221 | 1.1185 | 1.1179 | **1.1178** | 0.32 → 0.05 → **0.01 %** |
| 10⁴ | 2.3710 | 2.2932 | 2.2775 | **2.2752** | 3.28 → 0.69 → **0.10 %** |
| 10⁵ | 5.9021 | 4.9277 | 4.7663 | **4.7311** | 16.5 → 3.27 → **0.74 %** |

Finest mesh: 8 192 cells, 54 148 degrees of freedom.

**It converges monotonically.** That is the pass criterion and it is the thing the withdrawn solver
could not do.

**(c) On the reference values.** The de Vahl Davis 1983 table is paywalled and could not be
retrieved. The figures widely quoted from it — 1.118, 2.243, 4.519 — are recorded as **unverified
recollection**, not as a source. Against them this solver sits **+0.0 %, +1.4 %, +4.7 %**. The
growing gap with Rayleigh number is consistent with under-resolving the thermal boundary layer on a
uniform triangulation with no grading, which is a resolution limitation rather than an error, and it
would close with a graded mesh.

**The limit that matters for transformers.** Real oil convection runs at Ra ≈ 10⁸–10¹⁰, where the
flow is neither steady nor laminar and the CFD literature uses turbulence models or transient
solves. **This solver is valid in the laminar regime and must not be quoted outside it.**

## Part 2 — emissivity and reflections: survivable, until they are not

E9 proved a **uniform** absolute error and a **uniform** gain are absorbed exactly, which is why a
camera's ±2 K absolute accuracy is irrelevant. The documented hard problem is different: *"variable
emissivity and multiple reflections in fully metallic environments"*. Those vary **with height**,
and height is the axis carrying the location signal.

The decisive question is therefore not whether such error exists — it does — but how many
height-varying nuisance degrees of freedom the fit can absorb before location becomes
indistinguishable from them.

Camera NETD 0.05 K throughout. Location std, % of winding height.

### Additive nuisances — reflections, stray radiance, ambient gradients

| rows | P=0 | P=2 | P=4 | P=6 |
|---|---|---|---|---|
| 32 | 0.25 | 0.39 | 0.87 | 2.54 |
| 128 | 0.13 | 0.21 | 0.44 | 1.39 |
| 256 | 0.09 | 0.15 | 0.32 | 1.01 |

Graceful. Even a sixth-order additive error leaves a usable answer.

### Multiplicative nuisances — emissivity varying with height

| rows | Q=0 | Q=2 | Q=4 | Q=6 |
|---|---|---|---|---|
| 32 | 0.25 | 0.36 | 1.60 | 2.59 |
| 128 | 0.13 | 0.19 | 0.83 | **inf** |
| 256 | 0.09 | 0.14 | 0.59 | **inf** |

Also graceful, until it collapses at sixth order.

### Both at once — the realistic case, and the hard limit

| rows | P=Q=0 | P=Q=1 | P=Q=2 | **P=Q=3** |
|---|---|---|---|---|
| 32 | 0.25 | 0.36 | 5.84 | **inf** |
| 64 | 0.18 | 0.27 | 4.39 | **inf** |
| 128 | 0.13 | 0.19 | 3.23 | **inf** |
| **256** | 0.09 | 0.14 | 2.34 | **inf** |

**This is the finding.** With additive *and* multiplicative height variation at cubic order,
location is **unidentifiable at any number of camera rows**. Going from 32 to 256 rows does not
help at all — the nuisance basis spans the location signal, so it is structural confounding, not a
noise problem, and no amount of pixels or averaging touches it.

**Engineering consequence, and it is concrete:** the emissivity map must be **characterised, not
fitted**. Paint the tank uniformly, or measure emissivity once against a reference target, so the
fit only has to absorb first- or second-order residual variation. Beyond quadratic in both families
the instrument stops working regardless of camera quality.

### Localised reflection — a hot object mirrored on the tank

Gaussian in height with **both amplitude and position unknown**, on top of a fitted tilt and gain
drift. Hot spot at z = 0.90.

| reflection at z | 32 rows | 128 rows | 256 rows |
|---|---|---|---|
| 0.20 | 0.37 | 0.20 | 0.15 |
| 0.50 | 0.42 | 0.23 | 0.16 |
| 0.70 | 0.53 | 0.29 | 0.21 |
| **0.85** | **2.05** | **1.11** | **0.80** |
| **0.90** | **1.62** | **0.86** | **0.62** |

A reflection **away** from the hot spot is nearly free. A reflection **landing on** it costs 3–6×,
because a bright patch at the hot-spot height is exactly what a hot spot looks like. It degrades the
answer rather than destroying it, and unlike the polynomial case **more rows do help here**, because
a localised feature has a different shape from the broad profile signal.

**Operational consequence:** the reflection geometry matters more than its magnitude. Shoot the tank
from an angle that puts specular reflections of hot plant away from the expected hot-spot height, or
take two views and reject what moves between them.

## Part 3 — what this leaves

**Survives:** uniform absolute error, uniform gain, uniform emissivity, smooth height variation up
to about quadratic in both families, and localised reflections that miss the hot-spot height.

**Kills it:** simultaneous additive and multiplicative height variation at cubic order or worse.
**No camera specification rescues that** — it is a confounding, not a precision problem.

**Not tested and still open:** real emissivity maps rather than polynomial stand-ins; specular
reflection that is neither smooth nor a single Gaussian; wind and solar transients across a tank
face; and whether a clear vertical run of tank wall exists at all on a radiator-covered unit.

**Not tested and cheap to test:** point a thermal camera at an energised transformer and look. One
image settles the geometry questions that no amount of this modelling can.
