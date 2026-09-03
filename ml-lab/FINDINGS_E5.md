# E5 — conformal upper bounds, and the fourth defect

**3 September 2026.** Branch `ml-lab`. The last preregistered experiment with commercial value.

**Provenance.** The E5 primary run was executed in the sibling Codex working repository
(`CoreField-ML-Lab`, commit `98dd22c`) at 02:58:24 UTC on 3 September 2026, not in this tree. The
finalised artefacts were copied here read-only and re-verified: `aggregate.json` and
`manifest.start.json` both hash to the values recorded in `manifest.final.json`. Run id
`e5-5bf123e28160484a`, seed `61000`, wall time `6.96 s`, peak RSS `307 MB` (gate 2 GB, passed).

---

## 1. The headline, and it is the most commercially important result in this lab

Split conformal was calibrated on 200 in-range episodes and asked for a one-sided 95 % upper bound
on the episode's peak hot-spot temperature. The calibrated margin is **q = 0.374 K**. Each test
condition is 1,000 independently generated episodes.

| test condition | coverage | mean (true − predicted) | interval width |
|---|---|---|---|
| in range, 0.60–0.90 pu | 0.972 | +0.14 K | 0.374 K |
| 1.00 pu | **1.000** | −0.50 K | 0.374 K |
| 1.15 pu | **1.000** | −0.17 K | 0.374 K |
| 1.30 pu | **0.000** | **+1.20 K** | 0.374 K |
| 1.60 pu | **0.000** | **+8.75 K** | 0.374 K |

**(a) At 1.30 pu and 1.60 pu the bound covers 0 of 1,000 episodes.** Not degraded — zero. And the
failure is silent: **the interval width never changes.** It reports 0.374 K at 1.60 pu, where the
true peak exceeds the prediction by 8.75 K on average — a **23× understatement** delivered with no
indication that anything is wrong.

**(c) This is the direct commercial warning for the dynamic loading envelope.** The product's whole
proposition is telling an operator how much extra load a unit can carry. If the uncertainty band on
that recommendation is calibrated on normal operating data — which is the only data anyone has —
then the band is not merely optimistic above nameplate. It is wrong every single time, and it looks
tight while being wrong. An honest envelope must **refuse**, not interpolate its confidence.

The cliff sits between **1.15 and 1.30 pu**. Below it the estimator reads conservatively high
(negative mean error) and the bound holds trivially. Above it the estimator reads low and the bound
never holds. There is no gradual warning region.

**Boundary.** E5 truth uses the declared structural-mismatch stress test, `x(K) = 0.8 + 0.21(K−1)`
with the fitted model holding `x = 0.8`. The `0.21/pu` slope is a **(b)** engineering estimate, not
a measured universal value, so *where* the cliff sits depends on it. **The structure of the result
does not**: a fixed-width band calibrated inside a hull carries no information outside it, whatever
the mismatch. Only the location of the cliff is slope-dependent.

## 2. The two weighted cases behaved as preregistered

**Overlapping shift (Beta(5,2) mapped to 0.60–0.90 pu), KDE density ratio.** Coverage **0.961**,
exact binomial CI [0.947, 0.972] — contains 0.95. Effective sample size **110.6 of 200**, so
weighting costs almost half the calibration set. The estimated ratio tracked the known one closely
(correlation 0.989 at calibration loads, 0.982 at test loads; bias +0.086 and +0.127). Mean finite
width 0.371 K. Marked in the artefact as empirical, not exact: estimated density ratios do not
inherit the finite-sample theorem.

**Strict support mismatch.** Every query outside the calibration hull was assigned all mass at
infinity and returned an unbounded limit: **finite availability 0.000, formal containment 1.000**,
refusal reason `query_outside_calibration_hull` on all 1,000 episodes in every band. Exactly the
preregistered prediction — 100 % containment achieved only through useless intervals. The
implementation refuses rather than clipping weights into a finite pseudo-guarantee, which is the
honest behaviour and is the one thing here that could go straight into the product.

## 3. A fourth data-construction defect, not previously recorded

Codex's own post-run audit named three defects in E3 (see §4). **This is a fourth, in E5's episode
generator, found here and not documented in `RESULTS.md`, `VERDICT.md` or the run artefact.**

An episode is 4 h at 0.75 pu followed by 4 h at the sampled target, and its outcome is the maximum
hot-spot temperature. The record opens at the settled 0.75 pu equilibrium. **So whenever the target
is below the opening load, the step is downward, the trajectory decays monotonically, and the
maximum is the `t = 0` opening equilibrium — a constant that carries no information whatever about
the sampled target load.**

Measured directly from the frozen generator:

| target load | peak hot spot | argmax time |
|---|---|---|
| 0.60 pu | 62.561942 °C | **0.000 h** |
| 0.65 pu | 62.561942 °C | **0.000 h** |
| 0.70 pu | 62.561942 °C | **0.000 h** |
| 0.74 pu | 62.979102 °C | 8.000 h |
| 0.90 pu | 75.105133 °C | 8.000 h |

Identical to the last recorded digit. The crossover is at **0.734054 pu** (bisection, 40 steps).
Under the preregistered Uniform(0.60, 0.90) support that is **44.7 %** of episodes in theory, and in
the actual draws **89 of 200 calibration episodes (44.5 %)** and **464 of 1,000 exchangeable test
episodes (46.4 %)**.

### What it damages, and what it does not

**It explains the one preregistered prediction that failed.** §9 case 1 required the exact binomial
95 % CI to contain 0.95. Observed coverage was **0.972**, CI [0.960, 0.981], and the artefact
records `nominal_0_95_inside_exact_ci: false`. With 44.5 % of calibration scores collapsed onto a
degenerate cluster — 89 episodes sharing only 14 distinct values — the conformity distribution is
heavily tied and split conformal cannot deliver its usual near-exact coverage.

**The contamination inflates the margin rather than shrinking it.** Degenerate scores span
[+0.243, +0.423] K while genuinely informative ones span [−0.308, +0.351] K. Recomputing the
quantile from the 98 informative calibration episodes alone gives **0.334 K** against the **0.374 K**
actually used — about 12 % tighter.

**(c) So the headline in §1 survives, and is understated.** Every strict-shift band sits far above
the 0.734 pu crossover, so all of those episodes are genuinely informative. And the band that failed
to cover a single one of 2,000 episodes at 1.30 and 1.60 pu was the *inflated* one; the clean band
is tighter and would fail at least as badly. The defect damages the in-range calibration claim, not
the extrapolation finding.

**What must be fixed before E5 is quoted as confirmatory.** The episode design must make the outcome
depend on the sampled load — an opening segment below the support (`0.55 pu` or lower), or scoring
the target plateau rather than the whole record, as E3 already does with its `time_s >= 14400`
window. Either is a protocol change and needs its own frozen re-run.

## 4. E3 is superseded — the numbers this repository was quoting are descriptive only

Codex's audit found **three data-construction defects** in the E3 execution of 1 September, and
`VERDICT.md` now records that run as **descriptive only, not confirmatory**:

1. leading sensor-noise draws were reused across records, breaking split independence;
2. some three-minute references were attached to the previous two-minute feature row (max offset 60 s);
3. **linearly interpolated five-minute top-oil samples were used as dense two-minute PINN measurement
   targets** — the PINN was scored against an objective built from interpolation, not measurement.

**(c) The third defect matters most and it cuts against the conclusion this lab was quoting.** It
corrupted the PINN's own loss function, so the PINN's catastrophic failure is partly an artefact of
a defective objective rather than purely a property of physics-informed learning.

The earlier README presented the E3 table under "**Quote these whenever anyone recommends a PINN**".
**That instruction was wrong and has been corrected.** The honest statement is Codex's: *no ML method
has validly demonstrated a win over classical NLS anywhere in this harness* — which is "no
demonstrated win", **not** "proof that no win is possible". The margins (PINN RMSE 11.50 K at
1.00 pu, plain NN 44.56 K) are wide enough to justify **rejecting these candidates as an interim
engineering decision**, and that is all they justify.

The production ban on neural networks in the engine is a **CLAUDE.md policy decision** that predates
this lab and does not depend on it. This lab has not produced evidence to overturn it, and has not
produced confirmatory evidence to support it either.

## 5. Execution-order deviation, declared

E5 ran with only **E1 and E3** as prerequisites. E2 was **deferred on the decision owner's explicit
written instruction** (`private/codex_ml_lab_next_instruction.md` §3, "Do not run E2 yet"), and
standalone E4 was deferred because E3 already verified its outside-hull invariant on all 40
seed/load cells. Codex recorded both in the run's own configuration payload under
`execution_order_deviation`, so the deviation is inside the config hash rather than only in prose.
That is the right way to declare it.

**But one gate was loosened to make the run possible, and that should be visible.** The working-tree
`e5.py` copied into this repository on 2 September required completed E2 **and** E4 **and** an E3
whose configuration and seeds matched the current frozen values exactly. The version committed at
`98dd22c` requires only `require_completed_primary(repository_path, "e3")` with no configuration or
seed matching. Running the stricter version here is what surfaced the E3 harness drift in the first
place — it refused, correctly, with *"1 completed run(s) exist but none match"*.

**(c) The practical contamination is small**: E5 re-fits NLS itself and never reads the E3 aggregate,
so E3's three defects do not enter E5's numbers. E5 ran entirely under the corrected harness. But
the check that would have caught the drift was removed in the same commit as the run, and a reader
of the artefact alone would not know that.

## 6. Status

| case | preregistered prediction | outcome |
|---|---|---|
| exchangeable in-range | exact CI contains 0.95 | **failed** — 0.972, CI [0.960, 0.981]; cause identified in §3 |
| strict shift | under-covers, worsening with load | **confirmed, and worse than predicted** — 0/1000 at 1.30 and 1.60 pu |
| weighted overlapping | finite width, report ESS | **confirmed** — 0.961 coverage, ESS 110.6/200 |
| weighted strict support | unbounded only, 0 % finite | **confirmed exactly** — availability 0.000 |

**E5 verdict: adopt the refusal behaviour, reject the bound outside the hull.** The one genuinely
shippable finding is that a conformal band must return *unbounded* outside its calibration support
rather than a number. Everything else here is a warning about what not to sell.
