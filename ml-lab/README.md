EXPERIMENTAL. Not production. Nothing here is validated for loading decisions.

# CoreField ML lab — a falsification harness

A **quarantined** experimental repository on the orphan branch `ml-lab`. It shares no history with
`field-validation` or `main`, so it cannot be merged or fast-forwarded into a branch that ships. The
production repository forbids neural networks in the engine; this lab exists to test whether that
ban is still correct, which requires the work to be structurally incapable of leaking into it.

**Negative results are the primary deliverable.** No method is tuned after a loss. Where an analysis
error produced a confident wrong answer, the error is recorded rather than quietly fixed — three are
documented below, and each had first said "impossible".

---

## What was asked, and what came back

**The question:** can machine learning beat classical nonlinear least squares at transformer
hot-spot estimation, and can it infer *where* the hot spot is?

**The answer:** no to both, decisively — and the second "no" opened a more interesting question that
now occupies most of this repository.

## Experiments

| | question | verdict |
|---|---|---|
| **E1** | does the harness reproduce the classical baseline? | **gate passed**, 0.00 K on all eight checks |
| **E3** | do neural methods survive extrapolation above the training hull? | **no**, catastrophically |
| **E4** | does a hull-gated learned residual help? | collapses to classical, as predicted |
| **E6** | can any model infer hot-spot location from external channels? | **no**, R² = −0.018 |
| **E7** | what internal instrument would make location identifiable? | 2 probes **bracketing** the hot spot |
| **E8** | can an external array on the tank wall do it? | **yes**, ~16 sensors at 0.25 K |
| **E9** | does absolute accuracy matter? does radial structure kill it? | **no** and **no** — specify by NETD |
| **E10** | do emissivity variation and reflections kill it? | not until cubic order, then absolutely |
| **E11** | does eddy-current tank heating kill it? | no — and migration detection is immune |
| **E12** | is the cited prior art actually prior art? | no; and the tank question is a segmentation |
| **verification** | does the 2-D Boussinesq solver converge? | **yes**, monotonically |

E2 (scarce-reference identification) is **deferred with reason** — E3 settled the ranking among
already-rejected methods. E5 (conformal prediction) is **not yet run** and is the remaining item
with commercial value.

## The results that matter

### The neural ban is confirmed, with numbers

Trained on 0.60–0.95 pu, evaluated outside it. Mean signed peak error:

| load | classical NLS | PINN | plain NN |
|---|---|---|---|
| 1.00 pu | +0.66 K | **−14.42 K** | −48.63 K |
| 1.30 pu | −0.98 K | **−45.12 K** | −79.23 K |
| 1.60 pu | −8.48 K | **−85.89 K** | −119.50 K |

Every neural failure is **unsafe-low**, and it begins 0.05 pu outside the training data. The PINN
also never recovered physical parameters — Δθ_hr 43.7 K against a true 21.8 K — so it was neither
accurate nor interpretable. **Quote these whenever anyone recommends a PINN.**

### Location is not inferable from load, ambient and top-oil

E6 builds record pairs whose external channels are **bit-identical** (max difference 0.0) and which
differ only in the hidden hot-spot location. Best probe over ten seeds: **R² = −0.018**, worse than
predicting the mean. A null space, not a modelling failure.

### But the tank wall carries the signal, and nobody had looked

The four-channel "external" bundle collapses the tank to one mixed-oil **mean**, destroying shape.
The oil profile's endpoints are exactly invariant to location; its middle moves **0.875 K per 10 %
shift**. An array or a camera samples that.

- **E8** — 16 sensors at 0.25 K reach 1.78 % of winding height. A portable instrument pays **no
  penalty** for unknown thermal coupling: coupling and oil endpoints collapse into an offset and a
  scale the fit already removes.
- **E9** — **absolute accuracy is irrelevant.** A +2 K or −5 K bias moves the bound by *nothing*, so
  specify a camera by **NETD, not absolute accuracy**. At 0.05 K NETD and 32 rows, axial location
  reaches 0.71 %H *even with radial position unknown*, beating two internal probes.
- **E10** — the hard limit. Additive **and** multiplicative height variation at **cubic order makes
  location unidentifiable at any camera resolution**; 32 rows and 256 rows both give infinity. The
  emissivity map must be **characterised, not fitted**.
- **E11** — eddy-current tank heating is real (stray loss exceeds 20 % of load loss in large units,
  mostly in the tank) and survivable. The eddy pattern is geometry, so at equal load it **cancels in
  the difference between two surveys**: a *change* in location is recoverable to **0.13 %H** with
  nothing assumed about the eddy shape at all.

**That last result reframes the product: a migration detector, not a locator.** Operators dispatch
on temperature and bound migration by assuming the manufacturer's worst case. The one case where
that bound fails is a hot spot that has moved away from the probe measuring it.

### The 2-D solver is verified

`src/corefield_ml_lab/boussinesq.py` — full steady Boussinesq, Taylor-Hood P2/P1/P2, Picard
iteration, axisymmetric form carrying the `r` weight and the hoop term `u_r/r²`. Built on
**scikit-fem**: pure Python, no compiler, inside the 2 GB budget.

Verified on the de Vahl Davis cavity **with mesh convergence** — the check the withdrawn
geometry-PINN solver failed, having gone 161 → 513 → 1960 °C under refinement:

| Ra | refine 3 | 4 | 5 | 6 | successive change |
|---|---|---|---|---|---|
| 10³ | 1.1221 | 1.1185 | 1.1179 | 1.1178 | 0.32 → 0.05 → **0.01 %** |
| 10⁴ | 2.3710 | 2.2932 | 2.2775 | 2.2752 | 3.28 → 0.69 → **0.10 %** |
| 10⁵ | 5.9021 | 4.9277 | 4.7663 | 4.7311 | 16.5 → 3.3 → **0.74 %** |

**(c)** The published table is paywalled; the commonly quoted values are recorded as unverified
recollection and **mesh convergence is the criterion actually claimed**. Laminar and steady — real
oil convection is turbulent, and this solver must not be quoted outside the laminar regime.

## Three analysis errors, recorded rather than hidden

Each produced a confident, wrong "impossible".

1. **E8 first pass** carried the rated gradient as the nuisance, copying E7 — but the oil profile
   does not depend on it at all, so that column was exactly zero and every configuration came back
   singular.
2. **E8 second pass** carried oil endpoints and wall coupling as three separate nuisances when they
   enter only through an offset and a scale. Rank-deficient by construction.
3. **E7 first pass** used a scalar location bound assuming the hot-spot magnitude known, producing a
   flattering design that put both probes *above* the hot spot instead of bracketing it.

The pattern in all three: **a degeneracy inside the nuisances was read as location being
unidentifiable.** Worth remembering.

## Running it

```bash
PYTHONPATH="src;vendor" python -m corefield_ml_lab.verify_boussinesq
```

Substitute `e7_instrument`, `e10_emissivity` or `e11_eddy` for the others. The original E1–E5 CLI is
described in `PREREGISTRATION.md`, which was frozen before any model was written and states the
protocol, seeds, gates and stopping rules.

`vendor/` holds a read-only copy of the production package as the baseline under test. Nothing here
is ever imported back into it.

**Git note:** this repository was created by a sandboxed agent account. If git reports "dubious
ownership":

```bash
git config --global --add safe.directory "C:/Users/Nouman/Desktop/Furqan's Docs/CoreField-ML-Lab"
```

## Constraints

CPU only, under 2 GB RAM, no paid compute. All stochastic runs record seeds; ten seeds minimum with
mean and spread, never a single run. SI units: °C for absolute temperature, K for differences, per
unit for load. The private field record is read in place only; raw rows and derived series are never
copied here, and the 1.55 K score has **no external reporting permission** and must not be published
from this repository.

## What is not established

Every transformer-facing number rests on a **simplified 1-D axial model** with an assumed coupling
law, or on a **laminar** 2-D solve. A Cramér–Rao bound is a precision floor for an unbiased
estimator, not a guarantee any estimator reaches it. **No instrument has been built, costed or
tested**, and "cheap" means "tolerates the stated noise", not a bill of materials.

The two questions that decide the concept are not answerable here: how much unobstructed vertical
tank wall a real power transformer has, and whether anyone would pay to learn their probe is not at
the hot spot.
