# PREDICTIONS — pre-registered ledger

Every prediction below was **registered before the experiment that scored it**. The misses are
kept. A ledger that quietly drops its failures is not evidence of anything.

**23 registered.** The verdict on each entry is as recorded at the time. See
[the tally note](#a-note-on-the-tally) for why this file reports the counts the way it does.

Labels per the project's truth discipline: **(a)** verified fact with source · **(b)** engineering
estimate with stated assumptions · **(c)** inference or judgement.

---

## The ledger

| # | Registered prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Physics loss floors at O(1) on step loads (kinks); field error is the real metric | Confirmed in forward runs | HIT |
| 2 | After the schedule fix, the objective minimum coincides with truth | Sweep showed systematic high bias ~20 % | **MISS** — single-draw overclaim |
| 3 | Importance-sampled collocation improves τ_w | Made it worse (6.9 → 13.1 %, seed-paired) | **MISS** |
| 4 | Load ramps eliminate the bias (τ_w 3–8 %, straddling 7.0) | 16–19 %, still high | **MISS** — contributor only |
| 5 | PINN τ_w mean sits 7.5–8.5 min across configs | 7.2–8.4 for n ≥ 9; n=5 → 3.44; σ=2 → 6.89 | HIT, with stated exceptions |
| 6 | CRLB on τ_w at (σ=0.5, n=17) ≈ 3–4 % | 3.88 % | HIT |
| 7 | CRLB on Δθ_hr ≈ 1–1.5 % | 0.62 % | **MISS** — 2× conservative |
| 8 | Classical estimator within 1.5× CRLB, straddling, linear scaling | At the bound (0.99–1.02× on 6 of 8 cells) | HIT |
| 9 | CRLB at n=5 ≈ 7 % | 13.5 % | **MISS** |
| 10 | Classical at n=5 ≈ 7–12 % | 12.16 % | edge HIT |
| 11 | 4-param: τ_w stays 3–5 %, \|ρ(τ_w,τ_o)\| < 0.5, CRLB 4–5 % | 2.83 %, ρ = −0.07, CRLB 3.90 % | HIT / HIT / graze |
| 12 | Model A: RMSE 2–4 K, peak 2–6 K, Δθ_hr,eff 20–26, FAIL — and τ_w,eff 15–40 min | 4 of 5 correct; τ_w,eff = **4.84 min**, the opposite direction | **partial (4/5)** |
| 13 | Model B: RMSE 0.5–1.5 K, y_eff 1.25–1.35, peak 1–3 K | 1.82 / **1.665** / 2.24 | **MISS** (1 of 3) |
| 14 | Model C: noiseless RMSE < 0.3 K; noisy params near the 4-param pattern | Exact (0.00 %); τ_w 2.40 %, Δθ_hr 0.43 % | HIT |
| 15 | Ranking preserved on day B; winner ≤ 1.5× its day-A RMSE | C < B < A on both days; 1.1× | HIT |
| 16 | Truth-model harness: closed-form deviation < 1e-3 with a step at t = 0⁺ | 0.11 K RK4 quadrature artifact | **MISS — ours, harness fault** |
| 17 | Oil drift: Δθ_or +1…+3 %, RMSE ≤ 0.6, peak +0.4…+1.3 K, PASS | +2.15 / 0.29 / **−0.02** / PASS — least squares redistributed the ramp into the parameters, not the peak | **partial (3/4)** |
| 18 | Spikes degrade plain least squares (τ spread ±4–8 %); robust loss rescues | **No degradation at all** — 289 dense samples drown symmetric zero-mean glitches; "rescue" vacuous | **MISS — over-called** |
| 19 | Integer-°C quantisation ≤ 1.4× baseline | ≈ 1.0× | HIT |
| 20 | CT gain +2 %: Δθ's −2…−3.5 %, trajectory compensated, PASS | −2.59 / −2.54 / 0.15 K RMSE / PASS | HIT |
| 21 | WTI +3 K: Δθ_hr +11…+18 %, τ_w ±6 %, peak +2.5…+5 K, FAIL | +14.52 / **+10.55** / +4.11 / FAIL — the bias reshapes the *dynamics*, not just the amplitude | **partial (3/4)** |
| 22 | Ambient measured → PASS ≤ 1.5× baseline; ignored → RMSE 2.5–4.5, peak −3…−6 K, FAIL | 0.08 K PASS; 3.98 / −3.09 (band edge) / FAIL; τ_w +68 % was an unregistered observation | HIT |
| 23 | Day C: C \|peak\| ≤ 0.7 K PASS; B +3…+5 FAIL; A +6…+8 FAIL | C worst-case 0.32; B **+2.72**, A **+5.76** — both below the registered bands; FAIL verdicts and growth direction correct | **partial** |

### The most instructive entries

**P18** is the one worth reading twice. The prediction was that telemetry spikes would degrade
plain least squares and that a robust loss would rescue it. Neither happened — a 289-sample dense
channel drowns symmetric zero-mean glitches, and the "rescue" had nothing to rescue. The robust
loss stayed default-on anyway, but the honest label changed from *rescue* to *insurance*: it costs
nothing (0.12 K vs 0.13 K) and covers heavier-tailed glitch distributions than the one tested.

**P12** and **P13** both failed on *effective* parameters under structural mismatch, and in the
direction nobody guessed. Model A's effective τ_w was predicted at 15–40 min and came out at
4.84 min — least squares crushed it to chase the two-exponential overshoot hump. The general
lesson, which cost two predictions to learn: **effective parameters under structural mismatch are
not physical parameters**, and intuitions about their magnitude do not transfer.

**P16** is a harness fault, logged as a miss because it was one. An early release of the
truth-model check asserted a closed-form solution across a load discontinuity at t = 0⁺; RK4's
first stage samples the pre-step load, producing a one-time ~0.11 K quadrature artifact that was
briefly misread as a physics failure. A suppressed harness bug is indistinguishable from a
suppressed result, so it stays.

---

## A note on the tally

The legacy methods reports state a cumulative tally of **10 hits, 6 partial, 7 misses**.

Counting the verdict column above literally gives **11 hits, 4 partial, 8 misses**. Both sum to
23, so nothing is missing — but the two splits disagree, and the same gap exists in the earlier
v3 report (table gives 8/1/7, text claims 7/3/6).

The cause is labelling ambiguity, not invented data. Entries like "HIT, with stated exceptions"
(P5), "edge HIT" (P10), "HIT / HIT / graze" (P11) and "MISS (1 of 3)" (P13) can each be read
either way. Reaching 10/6/7 exactly requires reclassifying one HIT and one MISS as partial, and
no report says which.

**This file therefore reports both counts and does not choose between them.** Guessing which two
entries to move would be exactly the kind of quiet number-fitting this ledger exists to prevent.
The per-entry verdicts are the primary record; the summary count is a derived convenience and is
labelled as ambiguous **(a)** — the ambiguity is a verified property of the source documents.

---

## Findings that were NOT pre-registered

These emerged during the packaging work in August 2026. They are recorded separately because an
unregistered finding is weaker evidence than a registered prediction, and mixing the two would
inflate the ledger.

| Finding | Status |
|---|---|
| Four-parameter efficiency on Model C: 0.97 / 1.01 / 0.95 / 0.97× the folded-Gaussian expectation at 400 seeds, bias ≤ 0.12 % | **(a)** — computed, reproducible via `pytest` |
| The circulated claim "all four parameters at 0.99–1.02× CRLB" was **not supported**: that figure came from a two-parameter table on the older single-exponential model, covering 6 of 8 cells | **(a)** — see REPRODUCTION.md |
| A ±5 % band on the efficiency ratio is unachievable at 10 seeds by any estimator; the ratio's own standard error is ±0.24 there | **(a)** — analytic, `0.7555/√n` |
| Hot-spot **location** is exactly unobservable from external measurements (top-oil sensitivity is 0.0000 K per unit winding height) | **(a)** on the model, **(b)** on the model's realism — see ASSESSMENT.md |
| Load sampling rate dominates oil sampling rate for τ_w: +2.1 % at 1-minute load vs +8.4 % at 5-minute | **(a)** — pinned in `tests/test_ingest.py` |
| Model C's published day-A worst-absolute-error of 0.21 K does not reproduce (0.26 K); the notebook's gate aggregation never computed that column | **(a)** — see REPRODUCTION.md |

## Predictions registered for future work

Registered now, unscored, so that they can be scored honestly later.

| # | Prediction | To be scored by |
|---|---|---|
| F1 | On real ODAF telemetry with a measured hot spot, Model C's trajectory RMSE will fall between 1 and 4 K — an order of magnitude worse than the 0.11 K synthetic figure, because the synthetic result is structure-matched and the real one will not be | First field-validation run |
| F2 | The identified τ_o on a real unit will differ from its nameplate/heat-run value by more than 10 %, because nameplate values describe a new transformer and installed units have aged and been re-cooled | First field-validation run |
| F3 | Ambient sourced from a weather station >10 km away will cost less than 0.5 K of trajectory RMSE relative to an on-site sensor, because ambient enters through a ~75-minute low-pass | A paired-sensor comparison |
| F4 | On a real record, the binding constraint on the loading envelope will be top-oil rather than hot-spot more often than not, because utility oil limits are set conservatively | First 20 envelope computations on real data |
