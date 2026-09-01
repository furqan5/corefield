# ASSESSMENT — is 2D/3D worth building, and is any of this novel?

**Date:** 24 Aug 2026 · **Requested:** viability of extending 0-D → 2D → 3D by inverse PINN;
novelty check; dataset and geometry sourcing.

Labels per CLAUDE.md: **(a)** verified · **(b)** engineering estimate with assumptions ·
**(c)** inference/judgement.

---

## Verdict in four lines

1. **The specific 2D/3D vision — locate the hot spot from external or sparse external data —
   is not merely hard, it is information-theoretically impossible.** Computed, not asserted;
   see §1. The measurement carries exactly zero information about location.
2. **The version that does work needs sensors inside the winding**, which can only be fitted
   during manufacture — and needs no machine learning, because two probes already resolve
   location to 0.33 % of winding height. **(a)**
3. **Novelty is narrower than assumed.** PINNs for transformer thermal modelling, sparse-sensor
   field reconstruction for windings, and IEC-parameter calibration from operating data are all
   published, some by active groups. **(a)** But a real methodological gap exists — §2.
4. **Do not accept the CAD diagrams.** §4.

---

## 1. Can 2D/3D locate the hot spot from external data? No — and here is the proof

Implemented in `corefield/observability.py`, pinned by `tests/test_observability.py`.

The question is not "can a PINN learn this?" but "is the information present in the
measurement?" A Fisher-information analysis answers that for **every possible method at once** —
classical, learned, or otherwise. No estimator can extract information a measurement does not
contain.

### The result — **(a)** on the model, **(b)** on the model's realism

A 1-D axial winding with a loss concentration at height `z*`, normalised so total loss is
independent of `z*` (moving a hot spot redistributes heat; it does not create any):

| external channel | sensitivity to hot-spot location |
|---|---|
| top-oil | **0.0000 K** per unit height |
| bottom-oil | **0.0000 K** per unit height |
| tank surface (mixed oil) | 0.883 K per unit height |
| WTI replica (winding mean) | 0.883 K per unit height |

Top-oil reads **78.000000000 °C** whether the hot spot sits at 10 %, 50 % or 90 % of winding
height. That is not a small number — it is an exact cancellation, to machine precision.

**Mechanism.** Every external measurement is a function of the *total* heat the winding delivers
to the oil. Location changes the *distribution*, not the total. So location lies in the exact
null space of the external observation map, and the Fisher information for it is identically
zero.

A second-order effect survives — oil heats cumulatively as it rises past the losses, so the oil
*profile shape* carries a weak signal. Quantified:

| route | CRLB on hot-spot location |
|---|---|
| external sensors only | **± 40 % of winding height** |
| 1 internal fibre probe at 0.90 | ± 1.9 % |
| **2 internal probes at 0.80 / 0.95** | **± 0.33 %** |
| 8 internal probes | ± 0.22 % |

± 40 % on a hot spot that occupies the top 10 % of the winding is indistinguishable from knowing
nothing. To reach ± 5 % from outside you would need **0.044 K** sensor noise — about **11× better
than the best practical oil instrumentation (b)**.

### Why this is robust despite a simple model

The model is 1-D axial with a prescribed oil-rise profile, not CFD. Real windings vary radially,
disc to disc, and suffer oil-flow maldistribution. **(b)**

The conclusion survives that simplicity because the leading term is an *exact cancellation from
conservation of energy*. Any model in which the oil circuit is the only thermal path from winding
to the outside inherits it. A finer model changes the size of the second-order term; it cannot
make the first-order term non-zero.

### This is consistent with your own 0-D result

Your v1 campaign already established the one-dimensional version of this: ∂θ_o/∂{Δθ_hr, τ_w} = 0,
so top-oil carries **zero** winding information and direct hot-spot reads are unavoidable. If a
scalar amplitude and rate are unobservable from top-oil, a spatial *profile* certainly is. The
observability analysis above is the same finding, one dimension up.

### And it matches what the field already assumes

The literature does not treat hot-spot location as an inference problem. It treats it as a
design-determined constant — probes go in at **~90 % of winding height**, decided by design-time
electromagnetic and CFD analysis. The Sensors 2024 inversion paper is explicit: it estimates
hot-spot *temperature* from external data (including tank IR) and **assumes the location**.
Nobody infers location from outside. Now you know why. **(a)**

### What this means commercially

- Hot-spot *location* from external data: **dead**. Not a research risk — a physics one.
- Location from internal probes: **solved, and not by ML.** Two probes, linear interpolation.
- Any product needing internal probes can only be sold **at manufacture**, not retrofitted to an
  installed fleet — which destroys the "no new instrumentation on units already in service"
  wedge that makes the 0-D product interesting. **(c)**

**Recommendation: do not build 2D/3D for hot-spot location.** If you want the field-reconstruction
capability for its own sake, the honest first step is still fixing the finite-element solver's
mesh-convergence failure (161 → 513 → 1960 °C), because nothing built on an unconverged reference
can be validated. That is a separate project with no commercial pull that I can find.

---

## 2. Novelty — people are doing this, and here is exactly who

### Already published — do not claim these as new **(a)**

| Claim | Prior art |
|---|---|
| PINN for transformer dynamic thermal behaviour | Bragatto et al., *Electric Power Systems Research*, 2022 |
| **Spatio-temporal** PINN for transformer winding ageing | Mondragon group, residual-based-attention PINN |
| Bayesian PINN + uncertainty for transformer prognostics | Ramirez, Alcibar, Pino, Sanz, Pardo, Aizpurua — arXiv:2509.15933, Sept 2025 |
| Sparse-sensor winding temperature **field reconstruction** | *Case Studies in Thermal Engineering*, 2023 (POD-based) |
| Gappy-POD sparse-sensor thermal field inverse reconstruction | *Sensors* 25(16):4984, 2025 |
| Hot-spot **temperature** inversion from external data + tank IR | *Sensors* 24(14):4734, 2024 — MAE 0.81 °C, R² 0.987 |
| Estimating IEC 60076-7 thermal constants | Susa & Nordman, *ITEES* 23:946-960, 2013 |
| Calibrating IEC 60076-7 from operating data | SP Energy Networks technical note (a DNO, in practice) |
| Open-source IEC 60076-7 forward model | Alliander `transformer-thermal-model`, MPL-2.0, ~27 stars |
| Dynamic transformer rating | Decades of literature; IEEE C57.91 revision due 2025-26 **adds open-source code** |

That last row matters: **IEEE is about to ship open-source loading-model code.** Anything whose
value is "we implement the standard" is about to be commoditised. **(c)**

### Prior art found on 25 Aug 2026 — narrows the claim further **(a)**

A deeper literature pass turned up direct prior art for parameter identification from in-service
data. This must be cited, not competed with:

| Work | What they did | Accuracy |
|---|---|---|
| Doolgindachbaporn et al. (Southampton) | Nonlinear optimisation of top-oil rise and oil time constant against multi-year in-service data across **nine transformers** | tuned physical model **RMSE 2.6 degC**; ANN 1.5 degC; SVM 1.6 degC |
| SP Energy Networks, *Flexible Networks* | Offline Matlab/Simulink tuning of dtheta_or and tau_o to minimise SSE against measured top-oil | Cut dtheta_or from a 52 degC default to 45.3 degC (-6.7 K), as low as 43.0 degC on some assets |
| WPD/ENW *OpenLV* (SDRC 4) | Per-asset calibration of a dynamic rating application | top-oil error 5.5 K -> 4.0 K -> **< 1.0 K** after calibration |

**The distinction that survives.** Every one of these tunes **oil** parameters only -- dtheta_or
and tau_o -- against **measured top-oil**. That is the easy half: top-oil is directly measured,
so it is a straightforward curve fit. None identifies the **winding pair** (dtheta_hr, tau_w),
because that needs hot-spot data the units do not have.

CoreField identifies all four. The winding pair is the hard half, it is what the loading envelope
depends on, and it is what the calibration schedule and the CRLB exist to make possible. Narrow
the claim to that and it holds.

**The uncomfortable one.** In the Southampton study an ANN (1.5 degC) beat the tuned physical
model (2.6 degC) on real data. Do not pretend otherwise. The honest defence is not accuracy:

- an ANN returns no parameters, so no loading envelope and no propagated uncertainty;
- an ANN cannot extrapolate beyond its training hull, and the entire commercial product is the
  emergency-overload case *outside* the observed range. This repository's own day-C result shows
  what happens to models pushed past their fitted hull, and a black box has no structure to fall
  back on when it gets there.

Accuracy on interpolation is the wrong metric for a tool whose job is extrapolation.

### What I could not find prior art for -- the genuine contribution **(a)**

Searched twice, independently, and came up empty both times:

1. **Cramer-Rao / Fisher-information analysis of transformer thermal parameter identifiability.**
   Nothing. The literature calibrates parameters by regression, neural networks and
   metaheuristics without ever reporting the statistical bound those methods work against. This
   is the strongest methodological card you hold, and it is now confirmed rather than assumed.

   **Checked against the closest comparable work, 1 Sept 2026 (a).** Paulhiac and Desquiens'
   cooling-stage ODAF model (IEEE Trans. Power Del. 37(5), 4135-4144, 2022) fits its parameters
   by particle swarm optimisation or by hand and reports accuracy as post-hoc RMSE and cumulative
   error distributions. It contains no confidence intervals, no identifiability analysis, no
   information bound and no refusal criterion. That is a directed check of the single nearest
   paper rather than another literature sweep, and it came up empty in the same way.
2. **The observability law as a calibration-*scheduling* result** — amplitudes from quasi-steady,
   rates only from transients, and each sampled transient must anchor its own asymptote.
3. **"Commission on at least two load events"** as a CRLB-derived commissioning specification
   (12.3 % → 4.0 % floor on τ_w).

   **Qualified 1 Sept 2026 (a).** "No one else is stating this" was too strong. Paulhiac and
   Desquiens close their 2022 paper by proposing a factory acceptance-test load profile that
   exercises every cooling stage in turn, precisely to make fitting tractable. The idea of a
   *designed* commissioning profile is therefore not new. What remains unstated elsewhere is the
   **criterion**: theirs is a qualitative recommendation to cover the stages, while the spec here
   derives the requirement from an information bound and puts a number on what the profile buys.
4. **The structural-mismatch result quantified in the dangerous direction** — single-exponential
   models read **+5.76 K HIGH at 1.30 pu**, causing false derating exactly when capacity is worth
   most. Model comparison is common; framing it by *direction of error at overload* is not.
5. **The WTI calibration trap, quantified** — a +3 K reference bias yields +14.5 % on Δθ_hr,
   +10.6 % on τ_w, +4.1 K at the true peak, and contaminates *dynamics* not just level, so a
   "relative trends only" positioning does not escape it.
6. **The location-unobservability result in §1.**

### Honest read on the moat **(c)**

The physics is in a public standard. The algorithm is textbook nonlinear least squares. The PINN
angle is occupied by at least two active academic groups. **There is no algorithmic moat.**

What you actually have is **rigour and deployability**: bounds where others give point estimates,
a commissioning procedure derived rather than guessed, and a documented set of failure modes that
a utility will hit and a competitor will not have thought about. That is a real product wedge and
a poor patent. Pitch it as such — an investor who probes will find the prior art in an afternoon,
and being the one who already found it is worth more than a novelty claim that does not survive.

Your strongest asset for YC is not novelty. It is the **discipline**: the pre-registered ledger
with its misses kept, the withdrawn PINN branch, and now a self-inflicted kill on your own 2D/3D
extension. Very few technical founders can show that. Lead with it.

---

## 3. Datasets — what exists

**The one that matters (a):** *Operational data of Oil Directed Air Forced transformers* —
IEEE DataPort, Luc Paulhiac & Rémi Desquiens (EDF), Nov 2021, updated May 2022. Three real power
transformers (64, 360, 570 MVA) with **load factor, ambient, top-oil, and measured hot-spot
temperature**, plus cooling stage and oil viscosity. Excel format.

This is exactly the shape `corefield.ingest` is being built for, and it contains the one thing
your project has never had: **a measured hot spot to validate against**. It would convert
"everything is synthetic, no field validation exists" into a real result.

Caveat: **subscription-only**, which breaches the free-tier constraint. IEEE DataPort individual
subscription is modest **(b — verify current price; I did not)**. If you buy one thing for this
project, buy this. Note the cooling class is ODAF, not ONAF, so the Table-4 column must be
swapped — which `CoolingConstants` already supports.

**Secondary:** an arXiv study used 190 days of 5-minute current and top-oil from an operating
transformer with ambient from public weather data — top-oil only, no hot spot, so useful for
ingestion and oil-parameter work but not for validating the winding claim.

**Weather:** DMI (Denmark) and most national met services publish free hourly ambient. Your own
analysis says an hourly feed is adequate because ambient enters through the 75-minute oil
low-pass — so the ambient channel is a solved problem at zero cost. **(a)**

---

## 4. The CAD diagrams — decline them

You wrote that the utility contact has "2D schematics, full CAD diagrams of transformers of
different companies."

**Do not accept those files.** **(c)**, and I would hold this position firmly:

- Manufacturer drawings held by a utility are near-always covered by purchase-contract
  confidentiality or supplier NDAs. They are the *manufacturers'* IP, not the utility's to
  redistribute.
- "Different companies" makes it worse, not better — that is several parties' IP at once.
- Passing them to an outside startup would likely put your contact in breach of his employment
  obligations. He may not have considered that. You would be the proximate cause.
- YC diligence asks the provenance of training data. "A contact at a utility gave me
  manufacturers' drawings" is not survivable, and it is discoverable later by an acquirer.

**And note you do not need them.** §1 shows geometry does not rescue the external-sensor route —
the obstacle is conservation of energy, not insufficient detail about the winding.

**Legitimate geometry sources if you ever do need them (a):**

- Published CFD papers give **fully dimensioned** disc-winding geometries — e.g. 16 discs in
  2 passes (9 upstream, 7 downstream), radial spacers 5 mm and 3 mm, axial spacers 6 mm. Citable,
  reproducible, and free.
- Manufacturer public catalogues and IEC/IEEE standard figures.
- A university or manufacturer collaboration with an actual written agreement.

If your contact wants to help, the valuable and *safe* thing he can offer is **his own utility's
operational telemetry for their own units**, with his employer's written permission — not other
companies' drawings.

**Confidentiality reminder:** the job title you used in your message is itself identifying when
combined with the employer and dates already in the legacy files. The repo says "the pilot host".
It is worth using that habit in chat too, so it never slips into a commit message.

---

## 5. Recommendation

**Do:**
1. Finish 0-D — Stages 4-7. It is the fundable product and the only one with a retrofit market.
2. Buy the EDF dataset and do **real field validation**. This is the single highest-value action
   available and it closes your worst limitation.
3. Lead the pitch on rigour and failure modes, not novelty.
4. Publish the CRLB/observability work. It is a genuine gap, it is defensible, and it is cheap
   credibility.

**Do not:**
5. Build 2D/3D hot-spot location. §1 kills it on physics.
6. Accept the CAD diagrams. §4.
7. Claim novelty for PINN transformer thermal modelling or sparse-sensor field reconstruction.
   Both are occupied. **(a)**

**Open question for you:** the honest positioning is now "rigorous, deployable dynamic loading
from data a utility already has" rather than "novel AI for transformers". Those attract different
investors and different first customers. Worth deciding before the pitch deck, not after.
