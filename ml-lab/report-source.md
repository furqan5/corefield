# Evidence report source

**(a) Scope and status.** This is the canonical internal evidence ledger for the ML-versus-NLS transformer hot-spot falsification harness. A checkpoint was committed before E5 primary access at `98dd22ca81f952bb2b595b759757e089d81a9c1e`; this version incorporates the single E5 run completed on 03 Sep 2026. E1's reproduction gate passed but strict process-tree resource compliance is unresolved; retained E3 is superseded for confirmatory inference; E2 and standalone E4 are deferred; E5 is complete. Detailed tables are in [`RESULTS.md`](RESULTS.md), and decisions are in [`VERDICT.md`](VERDICT.md).

**(a) Conventions.** Evidence labels are **(a)** verified fact with named source, **(b)** engineering estimate with explicit assumptions, and **(c)** inference/judgment.

## Research question and bounded answer

**(c) Question.** Under the single preregistered synthetic transformer, fixed data schedules, fixed neural architectures, fixed NLS implementation, and registered safety rule, does any tested ML method beat classical NLS, and at which extrapolation load?

**(c) Current answer.** No valid run demonstrates an ML win at any load. The retained E3 execution observed no win—plain NN and PINN lost at all four loads, while grey-box equalled NLS outside the hull—but that execution failed a subsequent data-integrity audit. The conclusion is therefore absence of demonstrated superiority, not a universal no-ML theorem and not a valid confirmatory rejection by itself.

## Evidence hierarchy

1. **(a) Frozen protocol:** [`PREREGISTRATION.md`](PREREGISTRATION.md), SHA-256 `4e50cb8ff5de827dfc18c0206c56baa0b127f31f294aaee2f5737636c1dac4c6`, commit `b9648a6d`.
2. **(a) Frozen classical implementation:** thirteen byte-preserved modules under [`vendor/corefield`](vendor/corefield), sourced from read-only sibling commit `8219c99088645b7df984752e099a3f873bae773b`; hashes are in [`vendor/manifest.json`](vendor/manifest.json).
3. **(a) Primary artefacts:** write-once start manifest, aggregate, final manifest, and access sentinel for E1, E3, and E5.
4. **(a) Integrity audit:** direct inspection of the run-commit code and artefacts identified three E3 construction defects; corrected implementation and regression tests are separate from, and cannot rewrite, the retained result.
5. **(c) Interpretation:** classifications and commercial relevance are judgments constrained by the registered rules and validity boundary.

## External technical sources

| Source | What it verifies | What it does not verify here |
|---|---|---|
| **(a)** [IEC 60076-7:2018 official scope](https://webstore.iec.ch/en/publication/34351) | The edition/date and public scope concerning mineral-oil-immersed transformer loading, temperatures, ageing, and above-nameplate operation. | Exact equations, constants, or clauses in the vendored implementation. A licensed-copy comparison is still required before any IEC-compliance claim. |
| **(a)** Lei et al., [*Distribution-Free Predictive Inference for Regression*](https://doi.org/10.1080/01621459.2017.1307116), JASA 113(523), 2018 | Split-conformal finite-sample marginal coverage under independent/exchangeable sampling and rank-based calibration. | Conditional coverage, extrapolation under disjoint support, or exact validity of an estimated-weight variant. The harness's one-sided signed-score rule is a harness-derived specialization, not a verbatim algorithm from this source. |
| **(a)** Tibshirani et al., [*Conformal Prediction Under Covariate Shift*](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html), NeurIPS 2019 | Weighted conformal under unchanged conditional response distribution, target covariate absolute continuity, and the true likelihood ratio; target-point mass remains at `+∞`. | An exact guarantee for KDE-estimated density ratios or for strict disjoint calibration/test support. The harness's unbounded refusal under strict support mismatch is a safety policy consistent with not inventing finite support. |
| **(a)** Nordman and Lahtinen, [*Thermal Overload Tests on a 400-MVA Power Transformer With a Special 2.5-p.u. Short Time Loading Capability*](https://doi.org/10.1109/TPWRD.2002.807747), IEEE TPWRD 18(1), 2003 | Bibliographic existence of the cited overload study. | The project-specific hidden-slope setting or the internal `−6.35 K` overload value, neither of which was independently recovered from accessible primary text. |

## Claim-to-evidence ledger

| ID | Label | Claim | Direct evidence | Status/qualification |
|---|---|---|---|---|
| C01 | **(a)** | E1 reproduced all six synthetic targets and both authorized private aggregate RMSE targets within frozen tolerances. | [`runs/e1/e1-53df574b011a4dc4/aggregate.json`](runs/e1/e1-53df574b011a4dc4/aggregate.json) | Verified; overall gate `pass`. Private rows and paths were not persisted. |
| C02 | **(a)** | E1 completed in `14.3273 s`; recorded parent peak RSS was `281,378,816 B`. | [E1 final manifest](runs/e1/e1-53df574b011a4dc4/manifest.final.json) | Verified as recorded; child-process peak was not captured, so process-tree `<2 GB` is unresolved. |
| C03 | **(a)** | In the retained E3 execution, mean RMSE / mean signed peak [K] for NLS at `1.00/1.15/1.30/1.60 pu` was `0.51/+0.66`, `0.44/+0.36`, `1.08/−0.98`, `6.99/−8.48`. | [`runs/e3/e3-55464d9db759b4e2/aggregate.json`](runs/e3/e3-55464d9db759b4e2/aggregate.json) | Verified description of the superseded run; not confirmatory evidence. |
| C04 | **(a)** | In that execution, PINN mean RMSE was `11.50/28.20/47.22/88.15 K` and plain-NN mean RMSE was `44.56/57.03/71.16/105.53 K`. | E3 aggregate, `resolved.cells` | Verified description; all values are ten-seed means. |
| C05 | **(a)** | Every PINN and plain-NN signed peak error was negative at every E3 load and seed. | E3 aggregate, `rows` and `unsafe_low_fraction` | Verified description; `80/80` neural/load/seed peak errors were unsafe-low. |
| C06 | **(a)** | NLS was unsafe-low for `1/10` seeds at `1.15 pu` and `10/10` seeds at both `1.30` and `1.60 pu`. | E3 aggregate, safety metrics | Verified description; positive mean at `1.15 pu` must not be called universally safe. |
| C07 | **(a)** | Grey-box output was bit-exact with NLS outside the hull and flagged extrapolation in all 40 embedded checks. | E3 aggregate, `greybox_outside_hull_invariants` | Verified implementation invariant in the superseded run; standalone E4 in-range utility remains untested. |
| C08 | **(a)** | Generic IEC had lower RMSE than NLS at `1.00 pu` but failed safety; it did not beat NLS at `1.60 pu`. | E3 aggregate, `paired_decisions` and cells | Verified description; generic IEC is a fixed comparator, not an ML model or a real-unit compliance value. |
| C09 | **(a)** | The user-cited PINN/NLS parameter numbers were seed-31000 values, not ten-seed means. | E3 aggregate, `training` | Corrected means: PINN `Δθ_hr=41.68±2.38 K`, `τ_o=77.81±6.34 min`; NLS `21.94±0.27 K`, `167.39±1.07 min`. |
| C10 | **(a)** | E3 reused leading sensor-noise sequences across records. | Audit of run commit `bccae508`, synthetic generator RNG construction | Confirmatory blocker; corrected to stable `(seed, sensor, record)` substreams after the run. |
| C11 | **(a)** | E3 attached some three-minute references to a preceding two-minute row, with maximum `60 s` mismatch. | E3 configuration/training artefact plus run-commit feature construction | Confirmatory blocker; corrected to exact reference-time feature evaluation after the run. |
| C12 | **(a)** | E3 treated interpolated five-minute top-oil samples as dense two-minute PINN measurement targets. | Audit of run-commit PINN data/loss construction | Confirmatory blocker; corrected to actual measurement-time loss after the run. |
| C13 | **(c)** | Plain NN and PINN should not receive more tuning under the frozen candidate definitions. | C04–C06 plus registered no-rescue rule | Interim engineering rejection; not a general ML claim and not a valid confirmatory classification from E3. |
| C14 | **(c)** | Hull gating is honest but adds no outside-hull information. | C07 | Reject as outside-hull decoration; E4 in-range question remains open. |
| C15 | **(c)** | E2 was deferred so E5 could be prioritized for commercial value; E5 is now complete and E2 remains deferred. | Decision-owner instruction dated 02 Sep 2026 plus E5 final manifest | Protocol-order deviation; reason is resource prioritization, not a completed E2 result. |
| C16 | **(a)** | No E3 method met the registered win rule; no reserved confirmation was triggered. | E3 aggregate, `paired_decisions`, `confirmation_required=[]` | Verified within the superseded execution only. |
| C17 | **(a)** | E5 ordinary exchangeable coverage was `972/1,000=97.2%`, with exact binomial interval `95.98–98.13%` and median upper width `0.3741 K`. | [`runs/e5/e5-5bf123e28160484a/aggregate.json`](runs/e5/e5-5bf123e28160484a/aggregate.json), `exchangeable_in_range` | Verified; registered `nominal_0_95_inside_exact_ci` is false through overcoverage. |
| C18 | **(a)** | Ordinary strict-shift coverage at band centres `1.00/1.15/1.30/1.60 pu` was `100/100/0/0%`, while width remained `0.3741 K`. | E5 aggregate, `strict_ordinary_shift` | Verified; the failure was abrupt rather than progressively smooth. The `1.00 pu` band is `0.975–1.025 pu`, wholly outside calibration support but not wholly above `1.00 pu`. |
| C19 | **(a)** | KDE-weighted overlap gave `96.1%` empirical coverage, `100%` finite availability, median width `0.3725 K`, and calibration ESS `110.64`. | E5 aggregate, `weighted_overlapping_shift` | Verified empirical result; no exact guarantee is attached to estimated ratios. |
| C20 | **(a)** | Weighted strict support mismatch returned `100%` formal containment solely through unbounded limits and `0%` finite availability in all four bands. | E5 aggregate, `weighted_strict_support_mismatch` | Verified; this is a fail-closed refusal, not a finite predictive interval. |
| C21 | **(c)** | The registered “95% inside a test-set binomial CI” check is not a valid implementation checksum for marginal conformal coverage. | Lei et al. marginal theorem plus order-statistic derivation below | Criterion retained as failed; rank/rows independently recomputed without discrepancy. |
| C22 | **(c)** | E5's literal registered classification is reject; finite in-range replication remains scientifically worth investigating, and a finite strict-extrapolation product is rejected. | C17–C21 and `PREREGISTRATION.md` §11 | The post-run scientific judgment does not waive or relabel the failed registered primary rule. |
| C23 | **(a)** | E5 used one frozen generator seed despite the universal ten-seed sentence. | E5 start configuration and `PREREGISTRATION.md` §§3.2/9 | The specific seed/count prescription (200/1,000 independent episodes, seed `61000`) was followed; calibration-seed spread is unmeasured. |

## Reproducibility and integrity record

| Item | E1 | E3 superseded execution | E5 |
|---|---|---|---|
| **(a) Run ID** | `e1-53df574b011a4dc4` | `e3-55464d9db759b4e2` | `e5-5bf123e28160484a` |
| **(a) Seeds** | `1000..1009` | `31000..31009` | `61000` |
| **(a) Code commit** | `415a07229883af0a88bbc0665eae94a36567146c` | `bccae5081fc8c329f6d7d0b58b061818a99d9172` | `98dd22ca81f952bb2b595b759757e089d81a9c1e` |
| **(a) Aggregate SHA-256** | `a14eb8e572672dc5e3fa1a3282a0c806887e6b8911fff390c503c7281ffa3e44` | `ef42547a939d2d1dea80f70258253ee37e8c909f477fbad9585e6dc0d0ea2f65` | `ddb977c2f836918786966394f2e58848affbe64c64ecc02c793a5c5062fc3347` |
| **(a) Wall time** | `14.3273 s` | `408.2047 s` | `6.9645 s` |
| **(a) Recorded peak / conservative bound** | `281,378,816 B` parent peak | `373,751,808 B` current-process peak | `306,696,192 B` conservative process-tree bound |
| **(a) Execution** | CPU guard present; private child peak not captured | torch `2.14.0+cpu`, device CPU, one intra-op thread, CUDA unavailable; old Git-child peaks not captured | CPU-only, recorded one-thread environment limits, CUDA hidden/unavailable; parent `298,504,192 B` plus maximum Git child `8,192,000 B`; integrity and `<2 GB` gates passed |

**(a) Environment.** All three manifests record CPython `3.14.6` 64-bit on Windows 11 build `26200`, NumPy `2.5.2`, SciPy `1.18.1`, pandas `3.0.5`, matplotlib `3.11.1`, pytest `9.1.1`, and torch `2.14.0+cpu`; E1's private adapter records openpyxl `3.1.5`. E5 additionally records `12` logical CPUs and one-thread environment limits for MKL, OpenBLAS, OpenMP, NumExpr, and VECLIB.

**(a) Repository boundary.** The sibling `CoreField Startup` repository remained outside the experimental write scope. The vendored files are hash-manifested copies; no private field row, excerpt, derived private time series, path, or raw subprocess output appears in this lab.

## Registered-versus-observed synthesis

- **(a) E1:** prediction confirmed at stored precision; caveat is incomplete process-tree memory observation and missing synthetic per-seed spread.
- **(a) E3:** directional observations only partly matched: NLS RMSE was non-monotone at first and then rose sharply (`0.51→0.44→1.08→6.99 K`), plain-NN error was much larger, PINN was intermediate but losing, and grey-box collapsed exactly to NLS. The prediction that generic IEC might beat NLS at `1.60 pu` was not observed. All E3 comparisons are superseded for confirmatory use by C10–C12.
- **(a) E2:** no observation; deferred.
- **(a) E4:** outside-hull invariant observed inside E3; no standalone in-range result.
- **(a) E5:** ordinary in-range coverage was `972/1,000=97.2%` with fixed width `0.3741 K`; the registered exact-binomial-interval Boolean failed through overcoverage. Ordinary strict-shift coverage was `100/100/0/0%` at `1.00/1.15/1.30/1.60 pu`. KDE-weighted overlap gave `96.1%` coverage and median width `0.3725 K`; strict support mismatch produced only unbounded limits with zero finite availability.
- **(a) E6:** not triggered because there was no unexpected neural win and no spatial-information claim.

## E5 boundary, observed result, and statistical audit

**(a) Protocol fact.** With `n=200` calibration episodes and `α=0.05`, the frozen finite-sample rank is

`ceil((n+1)(1−α)) = ceil(201×0.95) = 191`.

**(a) Units.** The rank is dimensionless; conformity scores and the resulting additive upper width are in kelvin.

**(a) Train-record resolution.** E3 persisted ten fit summaries rather than one reusable noisy training record. E5 uses the same frozen 48 h E3 schedule, noise-free structural-mismatch truth, reference locations, and NLS implementation, with independent seed `61000` and corrected record-specific sensor streams. No observed E3 seed/model is selected post hoc.

**(a) Recorded pre-primary diagnostic disclosure.** The committed E5 configuration records a training-only comparison between seed `61000` under the corrected stream and an in-memory reconstruction of the old stream rule. It records no E5 calibration/test loads or truth, primary claim, written artefact, or changed choice. The corrected-minus-old parameter changes were `+0.0223 K` (`Δθ_or`), `−0.4001 min` (`τ_o`), `−0.3356 K` (`Δθ_hr`), and `−0.1989 min` (`τ_w`). **(c)** The non-exposure/no-choice assertion is provenance testimony and cannot be reconstructed from primary rows. “Fit once” was enforced once inside the primary E5 command.

**(a) Recorded pre-primary covariate disclosure.** The committed configuration records that non-primary tests generated the frozen E5 target-load draws in memory and passed them through a fake algebraic fixture before a primary-draw guard was added. It records no thermal-truth/fitted-model evaluation, persisted exact draw, sentinel claim, or changed choice; code and tests now refuse every public seed-`61000` draw before the sentinel. **(c)** The historical non-exposure assertion is provenance testimony and is not independently recoverable from primary rows.

**(c) Reporting rule.** Ordinary split-conformal width is constant because it adds one fixed calibration quantile; load-dependent deterioration must appear as coverage loss, not invented adaptive widening. Weighted results must always pair coverage with finite-interval availability. Under strict support mismatch, an unbounded interval is counted as formal containment but has zero commercial usefulness.

**(a) Primary execution.** Run `e5-5bf123e28160484a` used `200` calibration episodes and `1,000` test episodes per condition. The single NLS fit returned `Δθ_or=46.0154 K`, `τ_o=165.6738 min`, `Δθ_hr=21.7548 K`, and `τ_w=6.0401 min`; combined training residual RMSE was `0.7436 K`. The calibration score quantile was `q=0.374063 K` at rank `191`.

| E5 condition | Empirical coverage | Exact 95% binomial interval | Median upper width | Finite availability |
|---|---:|---:|---:|---:|
| **(a)** Ordinary, exchangeable `0.60–0.90 pu` | `97.2%` | `95.98–98.13%` | `0.3741 K` | `100%` |
| **(a)** KDE weighted, overlapping shift | `96.1%` | `94.71–97.21%` | `0.3725 K` | `100%` |
| **(a)** Weighted, each strict band | `100%` formal | `99.63–100%` | `∞` | `0%` |

**(a) Ordinary strict shift.** Coverage was `100%` in the `0.975–1.025 pu` and `1.125–1.175 pu` bands, then `0%` in the `1.275–1.325 pu` and `1.575–1.625 pu` bands, despite unchanged width `0.3741 K`. Truth exceeded the upper limit by at least `0.5053 K` in every episode in the `1.30 pu`-centred band and by at least `7.4391 K` in every episode in the `1.60 pu`-centred band.

**(a) Weighted overlap.** The KDE used 2,000 unlabeled Beta-target loads; a deterministic frozen-seed-stream audit reconstructed their support as `0.637828–0.897845 pu` and reproduced persisted ratios bit-for-bit. Calibration effective sample size was `110.64`. Weighted widths had median `0.3725 K` and range `0.3511–0.3725 K`. Post-hoc equal-width load bins covered `100%` (`n=25`), `88.3%` (`n=334`), and `100%` (`n=641`) from low to high load; these bins have no registered conditional-coverage guarantee. Estimated density ratios, rather than the true synthetic ratio required by the cited theorem, make the `96.1%` result empirical rather than an exact guarantee.

**(a) Literal registered sanity result.** The exact binomial interval for `972/1,000` excludes `95%`, so `nominal_0_95_inside_exact_ci=false`; the criterion is retained as failed. Independent recomputation reproduced the rank, quantile, row count, coverage, support classification, hashes, and prerequisite lineage.

**(c) Why that Boolean is not an implementation checksum.** Under a continuous-iid-score idealization, rank `191` gives marginal coverage `191/201=95.025%`; the fixed-calibration probability follows `Beta(191,10)`, and the resulting 1,000-row beta-binomial central 95% range is `913–978` covered rows. The observed `972` is inside, with idealized upper-tail probability `8.56%`. The preregistered Clopper–Pearson rule accepts only `936–963` and passes only about `59.4%` of complete idealized repetitions. It therefore confuses a conditional interval for one calibration split with the repeated-calibration marginal target.

**(a) Continuity caveat.** This artefact has only `112/200` unique calibration scores; score `0.243396 K` repeats `89` times, while the corresponding exchangeable-test atom repeats `464/1,000` times. The selected rank-191 score is unique and above that atom, but the exact beta law still does not apply. The continuous calculation is a diagnostic benchmark, not an exact distributional claim for this run.

**(a) Sampling limitation.** The universal protocol sentence calls for at least ten seeds per stochastic cell, while the specific E5 design freezes only seed `61000` and independent episode counts. The specific E5 single-seed and episode-count prescription was followed, so across-calibration-seed variation is unmeasured. No post-access repeat is permitted under the test-once rule.

**(c) E5 decision.** The literal registered classification is **reject** because the primary sanity rule failed. Separately, investigate finite in-range bounds only through a newly preregistered multi-calibration-seed replication; reject any finite strict-extrapolation product and retain the unbounded support guard as an honest refusal. E5 neither repairs the invalid E3 comparison nor demonstrates an ML win.

## Independent verification handoff

Before relying on any conclusion in a paper, client report, or operational decision, independently verify:

1. **(a)** Recompute SHA-256 for the protocol, vendor manifest/files, and every run aggregate; verify start/final manifest linkage and access sentinels.
2. **(a)** Re-run the full non-primary test suite from a clean commit with pytest temporary output outside the repository.
3. **(a)** Audit the corrected E3 generator for record-specific RNG streams, exact timestamp feature construction, and measurement-time-only PINN loss; a corrected primary E3 execution is required for confirmatory comparison.
4. **(a)** Inspect a licensed IEC 60076-7:2018 copy before calling the vendored equations or constants standards-compliant.
5. **(a)** Independently reproduce E1 private aggregates under the data owner's controls and measure the complete process-tree memory peak; do not export row-level evidence.
6. **(a)** Treat all synthetic conclusions as conditional on one declared unit, one hidden exponent stress test, and the frozen schedules; external transformers and operational loading remain out of scope.
7. **(a)** Verify E5 run `e5-5bf123e28160484a`, aggregate SHA-256 `ddb977c2f836918786966394f2e58848affbe64c64ecc02c793a5c5062fc3347`, exact E1/E3 prerequisite identities, process-tree bound, and no-override sentinel before citing the result.
8. **(c)** Do not rerun or tune against the accessed E5 test set. Any replication across calibration seeds or external transformers requires a new preregistration and untouched data.
