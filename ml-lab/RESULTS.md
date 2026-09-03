# Experimental results

**(a) Status.** The pre-E5 checkpoint was committed at `98dd22ca81f952bb2b595b759757e089d81a9c1e` before primary access, then updated on 03 Sep 2026 from the single completed E5 artefact. E1's reproduction gate passed but strict process-tree resource compliance is unresolved; retained E3 is **descriptive only** because a post-run audit found three data-construction defects; E2 and standalone E4 remain deferred; E6 was not triggered.

**(a) Conventions.** Evidence labels used throughout are **(a)** verified fact from a named source or immutable run artefact, **(b)** engineering estimate with assumptions, and **(c)** inference or decision judgment. Under the stored signed-error definition, a negative value means the estimator is unsafe-low relative to synthetic truth.

## E3 lead result — superseded execution

**(a) Retained observation, not confirmatory evidence.** The completed E3 artefact used ten seeds (`31000..31009`), trained on `0.60–0.95 pu`, and scored the four-hour target plateau. Values are mean RMSE / mean signed peak error [K].

| Target load [pu] | NLS | Grey-box | Generic IEC | PINN | Plain NN |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.51 / +0.66 | 0.51 / +0.66 | 0.31 / −0.03 | 11.50 / −14.42 | 44.56 / −48.63 |
| 1.15 | 0.44 / +0.36 | 0.44 / +0.36 | 0.54 / −0.44 | 28.20 / −28.73 | 57.03 / −62.91 |
| 1.30 | 1.08 / −0.98 | 1.08 / −0.98 | 1.62 / −1.89 | 47.22 / −45.12 | 71.16 / −79.23 |
| 1.60 | 6.99 / −8.48 | 6.99 / −8.48 | 7.62 / −9.64 | 88.15 / −85.89 | 105.53 / −119.50 |

**(a) Validity boundary.** The artefact accurately records what that execution produced, but it cannot support a registered adopt/reject claim. The run reused leading sensor-noise draws across records, attached some three-minute references to the previous two-minute feature row (recorded maximum offset `60 s`), and used linearly interpolated five-minute top-oil samples as dense two-minute PINN measurement targets. These affect split independence, label timing, and the PINN objective. The implementation is corrected, but those corrections do not retroactively repair this run.

**(a) Descriptive finding.** Neural failure in this execution began at `1.00 pu`, only `0.05 pu` above the training hull: PINN mean RMSE was `11.50 K`, and plain-NN mean RMSE was `44.56 K`.

**(a) Descriptive finding.** Every PINN and plain-NN signed peak error was negative at every target load: unsafe-low fraction `1.00` in all eight neural/load cells. This is an observed direction of error in this synthetic execution, not evidence of a universal neural-model tendency.

**(a) Descriptive finding.** NLS was also not safe throughout the above-nameplate domain. At `1.15 pu`, one of ten seeds was unsafe-low despite the positive mean; at `1.30` and `1.60 pu`, all ten were unsafe-low. At `1.60 pu`, mean signed peak error was `−8.48 K` and the worst seed was `−9.27 K`.

**(a) Descriptive finding.** The generic comparator had lower mean RMSE than NLS at `1.00 pu` (`0.31` versus `0.51 K`) but was unsafe-low on every seed at every load. NLS had positive mean signed peak error at `1.00` and `1.15 pu`, although one NLS seed at `1.15 pu` was negative. Accuracy rank therefore does not carry the safety direction.

**(a) Descriptive invariant.** All `40/40` embedded E3 grey-box checks passed: outside the training hull its residual was positive zero, its hot-spot output was bit-exact with NLS, and the extrapolation flag was set. This confirms only the preregistered outside-hull part of E4; the standalone in-range E4 adoption test has not run.

## E3 complete metric tables

**(a) Artefact facts.** Each entry below is `mean ± sample SD; median [minimum, maximum]` over ten seeds, in kelvin. Generic IEC has no stochastic fit, hence its zero or floating-point-roundoff spread. Availability was `10/10` for every method/load cell.

### Mean absolute trajectory error

| Load [pu] | NLS | Grey-box | Generic IEC | PINN | Plain NN |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.49 ± 0.23; 0.42 [0.15, 0.91] | 0.49 ± 0.23; 0.42 [0.15, 0.91] | 0.23 ± 0.00; 0.23 [0.23, 0.23] | 11.21 ± 0.26; 11.20 [10.87, 11.67] | 44.35 ± 0.44; 44.45 [43.51, 45.00] |
| 1.15 | 0.36 ± 0.20; 0.25 [0.20, 0.77] | 0.36 ± 0.20; 0.25 [0.20, 0.77] | 0.53 ± 0.00; 0.53 [0.53, 0.53] | 26.53 ± 0.67; 26.54 [25.20, 27.55] | 56.59 ± 0.44; 56.68 [55.75, 57.24] |
| 1.30 | 1.02 ± 0.30; 1.10 [0.59, 1.53] | 1.02 ± 0.30; 1.10 [0.59, 1.53] | 1.59 ± 0.00; 1.59 [1.59, 1.59] | 43.90 ± 1.24; 43.75 [41.50, 45.74] | 70.39 ± 0.45; 70.49 [69.56, 71.05] |
| 1.60 | 6.49 ± 0.46; 6.62 [5.74, 7.25] | 6.49 ± 0.46; 6.62 [5.74, 7.25] | 7.23 ± 0.00; 7.23 [7.23, 7.23] | 85.43 ± 8.07; 81.17 [78.95, 99.61] | 103.82 ± 0.53; 103.84 [102.91, 104.66] |

### RMSE

| Load [pu] | NLS | Grey-box | Generic IEC | PINN | Plain NN |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.51 ± 0.23; 0.44 [0.18, 0.93] | 0.51 ± 0.23; 0.44 [0.18, 0.93] | 0.31 ± 0.00; 0.31 [0.31, 0.31] | 11.50 ± 0.25; 11.48 [11.19, 11.95] | 44.56 ± 0.44; 44.66 [43.73, 45.21] |
| 1.15 | 0.44 ± 0.17; 0.39 [0.28, 0.82] | 0.44 ± 0.17; 0.39 [0.28, 0.82] | 0.54 ± 0.00; 0.54 [0.54, 0.54] | 28.20 ± 1.19; 28.26 [25.69, 29.92] | 57.03 ± 0.44; 57.13 [56.20, 57.68] |
| 1.30 | 1.08 ± 0.29; 1.14 [0.64, 1.60] | 1.08 ± 0.29; 1.14 [0.64, 1.60] | 1.62 ± 0.00; 1.62 [1.62, 1.62] | 47.22 ± 2.02; 47.35 [42.59, 49.64] | 71.16 ± 0.44; 71.25 [70.33, 71.81] |
| 1.60 | 6.99 ± 0.47; 7.12 [6.20, 7.76] | 6.99 ± 0.47; 7.12 [6.20, 7.76] | 7.62 ± 0.00; 7.62 [7.62, 7.62] | 88.15 ± 8.85; 83.33 [81.05, 104.94] | 105.53 ± 0.53; 105.55 [104.63, 106.37] |

### Mean signed trajectory error

| Load [pu] | NLS | Grey-box | Generic IEC | PINN | Plain NN |
|---:|---:|---:|---:|---:|---:|
| 1.00 | +0.48 ± 0.25; +0.41 [+0.04, +0.91] | +0.48 ± 0.25; +0.41 [+0.04, +0.91] | −0.23 ± 0.00; −0.23 [−0.23, −0.23] | −11.21 ± 0.26; −11.20 [−11.67, −10.87] | −44.35 ± 0.44; −44.45 [−45.00, −43.51] |
| 1.15 | +0.23 ± 0.31; +0.15 [−0.30, +0.77] | +0.23 ± 0.31; +0.15 [−0.30, +0.77] | −0.53 ± 0.00; −0.53 [−0.53, −0.53] | −26.53 ± 0.67; −26.54 [−27.55, −25.20] | −56.59 ± 0.44; −56.68 [−57.24, −55.75] |
| 1.30 | −0.78 ± 0.36; −0.88 [−1.40, −0.14] | −0.78 ± 0.36; −0.88 [−1.40, −0.14] | −1.59 ± 0.00; −1.59 [−1.59, −1.59] | −43.90 ± 1.24; −43.75 [−45.74, −41.50] | −70.39 ± 0.45; −70.49 [−71.05, −69.56] |
| 1.60 | −6.35 ± 0.48; −6.49 [−7.16, −5.48] | −6.35 ± 0.48; −6.49 [−7.16, −5.48] | −7.23 ± 0.00; −7.23 [−7.23, −7.23] | −85.43 ± 8.07; −81.17 [−99.61, −78.95] | −103.82 ± 0.53; −103.84 [−104.66, −102.91] |

### Signed peak error

| Load [pu] | NLS | Grey-box | Generic IEC | PINN | Plain NN |
|---:|---:|---:|---:|---:|---:|
| 1.00 | +0.66 ± 0.25; +0.59 [+0.23, +1.08] | +0.66 ± 0.25; +0.59 [+0.23, +1.08] | −0.03 ± 0.00; −0.03 [−0.03, −0.03] | −14.42 ± 0.31; −14.32 [−14.98, −13.99] | −48.63 ± 0.44; −48.73 [−49.29, −47.80] |
| 1.15 | +0.36 ± 0.30; +0.28 [−0.15, +0.86] | +0.36 ± 0.30; +0.28 [−0.15, +0.86] | −0.44 ± 0.00; −0.44 [−0.44, −0.44] | −28.73 ± 0.32; −28.58 [−29.31, −28.37] | −62.91 ± 0.44; −63.01 [−63.56, −62.08] |
| 1.30 | −0.98 ± 0.36; −1.07 [−1.58, −0.39] | −0.98 ± 0.36; −1.07 [−1.58, −0.39] | −1.89 ± 0.00; −1.89 [−1.89, −1.89] | −45.12 ± 0.33; −44.97 [−45.76, −44.80] | −79.23 ± 0.44; −79.33 [−79.89, −78.40] |
| 1.60 | −8.48 ± 0.47; −8.61 [−9.27, −7.70] | −8.48 ± 0.47; −8.61 [−9.27, −7.70] | −9.64 ± 0.00; −9.64 [−9.64, −9.64] | −85.89 ± 0.78; −85.84 [−87.49, −85.04] | −119.50 ± 0.44; −119.60 [−120.15, −118.66] |

**(a) Safety summary.** Each entry is `most-negative signed peak error [K] / unsafe-low fraction`.

| Load [pu] | NLS | Grey-box | Generic IEC | PINN | Plain NN |
|---:|---:|---:|---:|---:|---:|
| 1.00 | +0.23 / 0.00 | +0.23 / 0.00 | −0.03 / 1.00 | −14.98 / 1.00 | −49.29 / 1.00 |
| 1.15 | −0.15 / 0.10 | −0.15 / 0.10 | −0.44 / 1.00 | −29.31 / 1.00 | −63.56 / 1.00 |
| 1.30 | −1.58 / 1.00 | −1.58 / 1.00 | −1.89 / 1.00 | −45.76 / 1.00 | −79.89 / 1.00 |
| 1.60 | −9.27 / 1.00 | −9.27 / 1.00 | −9.64 / 1.00 | −87.49 / 1.00 | −120.15 / 1.00 |

## E3 fitted parameters and in-range validation

**(a) Artefact facts.** Entries are `mean ± sample SD; median [minimum, maximum]` over ten seeds. Because E3 deliberately used structural-mismatch truth while all fitted physics models fixed the exponents, these are pseudo-physical fitted parameters; they do not establish general physical identifiability.

| Quantity | NLS | PINN |
|---|---:|---:|
| `Δθ_or` [K] | 46.01 ± 0.03; 46.00 [45.98, 46.06] | 51.50 ± 0.57; 51.56 [50.41, 52.35] |
| `τ_o` [min] | 167.39 ± 1.07; 167.12 [166.10, 169.21] | 77.81 ± 6.34; 76.51 [70.13, 90.05] |
| `Δθ_hr` [K] | 21.94 ± 0.27; 21.89 [21.45, 22.36] | 41.68 ± 2.38; 42.59 [37.98, 44.44] |
| `τ_w` [min] | 6.00 ± 0.19; 5.97 [5.82, 6.33] | 7.64 ± 0.59; 7.64 [6.75, 8.41] |
| Validation-reference RMSE [K] | N/A | 7.53 ± 0.14; 7.48 [7.35, 7.82] |

**(a) Comparison detail.** Grey-box validation-reference RMSE was `0.56 ± 0.10 K`, median `0.55 K`, range `0.43–0.75 K`; plain-NN validation-reference RMSE was `36.17 ± 1.53 K`, median `35.87 K`, range `34.86–39.90 K`. NLS validation-reference RMSE was not recorded, so it is not equated to the grey-box result.

**(a) Correction to the supplied interpretation.** `Δθ_hr≈43.7 K`, `τ_o≈75.9 min`, NLS `Δθ_hr≈21.8 K`, NLS `τ_o≈166.1 min`, PINN validation RMSE `≈7.4 K`, and grey-box validation RMSE `≈0.54 K` are seed-`31000` values, not ten-seed means. The across-seed values are in the table above.

**(c) Interpretation.** The PINN physics penalty did not keep its fitted pseudo-parameters near the nominal/NLS solution in this execution, and its in-range validation fit was materially worse than the grey-box fit. The stronger statement that it “never recovered physical parameters” is not identified by this misspecified synthetic experiment.

## Predicted versus observed

| Experiment | Preregistered prediction | Observed status through 03 Sep 2026 |
|---|---|---|
| E1 | **(a)** Reproduce frozen A/B/C and private aggregate targets within fixed tolerances. | **(a)** Passed every available gate exactly at the persisted precision. |
| E3 | **(c)** NLS degrades gradually; plain NN may read dangerously low; PINN lies between it and NLS but loses; grey-box equals NLS outside hull; generic IEC may beat NLS at `1.60 pu`. | **(a)** The superseded execution partly matched: NLS RMSE was non-monotone at first and then rose sharply (`0.51→0.44→1.08→6.99 K`), both neural candidates were unsafe-low and lost, and grey-box equalled NLS outside the hull. Generic IEC did **not** beat NLS at `1.60 pu` (`7.62` vs `6.99 K`). No method met the registered E3 win rule, but the audit defects prevent confirmatory use. |
| E2 | **(c)** NLS is expected to equal or beat PINN for `N≥4`; NLS refuses at `N=3`. | **(a)** Not run; deferred, not skipped. |
| E4 | **(c)** Hard gating collapses outside-hull grey-box output exactly to NLS; an in-range residual may help. | **(a)** The embedded outside-hull invariant passed `40/40` in the superseded E3 run. No standalone in-range E4 result exists. |
| E5 | **(c)** In-range coverage should be compatible with `95%`; ordinary strict-shift coverage should deteriorate; strict weighted support mismatch should become unbounded. | **(a)** In-range ordinary coverage was conservative at `97.2%`, but the registered Clopper–Pearson interval excluded `95%`, so the literal sanity rule failed. Ordinary coverage was `100/100/0/0%` across the four strict bands. KDE-weighted overlap gave `96.1%` empirical coverage; strict weighted limits were unbounded with `0%` finite availability, as predicted. |
| E6 | **(c)** A location probe should fail under the paired null-space test. | **(a)** Not triggered: no unexpected neural win or spatial-information claim occurred. |

## E5 conformal upper-bound result

**(a) Verified primary artefact.** Run `e5-5bf123e28160484a` fitted NLS once inside the primary command, used `200` independent calibration episodes and `1,000` independent test episodes per condition, and used one-sided score `S=true peak−predicted peak` [K]. The calibration correction was the registered rank-`191` score, `q=0.374063 K`.

| Case | Load support [pu] | Covered / n | Empirical coverage | Exact 95% binomial CI | Finite availability | Median finite upper width [K] |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary exchangeable | 0.60–0.90 | 972 / 1,000 | 97.2% | 95.98–98.13% | 100% | 0.3741 |
| Ordinary strict, centre 1.00 | 0.975–1.025 | 1,000 / 1,000 | 100% | 99.63–100% | 100% | 0.3741 |
| Ordinary strict, centre 1.15 | 1.125–1.175 | 1,000 / 1,000 | 100% | 99.63–100% | 100% | 0.3741 |
| Ordinary strict, centre 1.30 | 1.275–1.325 | 0 / 1,000 | 0% | 0–0.37% | 100% | 0.3741 |
| Ordinary strict, centre 1.60 | 1.575–1.625 | 0 / 1,000 | 0% | 0–0.37% | 100% | 0.3741 |
| KDE-weighted overlap | Beta(5,2) mapped to 0.60–0.90 | 961 / 1,000 | 96.1% | 94.71–97.21% | 100% | 0.3725 |
| Weighted strict, centre 1.00 | 0.975–1.025 | 1,000 / 1,000 formal | 100% formal | 99.63–100% | 0% | unbounded |
| Weighted strict, centre 1.15 | 1.125–1.175 | 1,000 / 1,000 formal | 100% formal | 99.63–100% | 0% | unbounded |
| Weighted strict, centre 1.30 | 1.275–1.325 | 1,000 / 1,000 formal | 100% formal | 99.63–100% | 0% | unbounded |
| Weighted strict, centre 1.60 | 1.575–1.625 | 1,000 / 1,000 formal | 100% formal | 99.63–100% | 0% | unbounded |

**(a) Ordinary strict-shift failure.** The fixed `0.374063 K` correction did not widen with load. Its outcome was abrupt rather than gradually deteriorating: complete coverage in the `1.00` and `1.15 pu` bands, then complete failure in the `1.30` and `1.60 pu` bands.

| Strict-band centre [pu] | Mean score `S` [K] | Sample SD [K] | Median [min, max] [K] | Mean exceedance `S−q` [K] |
|---:|---:|---:|---:|---:|
| 1.00 | −0.4962 | 0.0078 | −0.4996 [−0.5036, −0.4776] | −0.8703 |
| 1.15 | −0.1674 | 0.0773 | −0.1745 [−0.2898, −0.0284] | −0.5415 |
| 1.30 | +1.2034 | 0.1965 | +1.2031 [+0.8793, +1.5524] | +0.8293 |
| 1.60 | +8.7504 | 0.5525 | +8.7393 [+7.8132, +9.7746] | +8.3763 |

**(a) Interpretation of the table.** Positive `S−q` is the amount by which truth exceeds the reported upper limit. Every episode in the `1.30 pu`-centred band (`1.275–1.325 pu`) missed by at least `0.5053 K`, and every episode in the `1.60 pu`-centred band (`1.575–1.625 pu`) missed by at least `7.4391 K`; the mean misses were `0.8293 K` and `8.3763 K`. The observed score changed from negative to positive while the ordinary bound retained the same narrow correction. **(c)** Product interpretation: this is a silent finite-bound failure under structural extrapolation.

**(a) Weighted overlap.** Estimated KDE weights used 2,000 unlabeled Beta-target loads and gave effective calibration size `110.64` from 200 episodes. A deterministic audit of the frozen seed stream reconstructed unlabeled-target support `0.637828–0.897845 pu` and reproduced the persisted KDE ratios bit-for-bit. Against the known synthetic ratio, estimated/known correlation was `0.9891` at calibration loads and `0.9819` at test loads; test-load ratio RMSE was `0.1806`. Coverage `96.1%` is empirical only because the ratios were estimated, not oracle.

**(a) Width distribution.** Calibration scores were `0.1269±0.2017 K`, median `0.2434 K`, range `−0.3084–+0.4232 K`. Ordinary widths were constant at `0.374063 K` apart from floating-point roundoff. Weighted-overlap widths were `0.3706±0.0033 K`, median `0.3725 K`, range `0.3511–0.3725 K`.

**(c) Post-hoc load-dependence diagnostic requested for product interpretation.** Equal-width bins were not a registered decision rule and carry no conditional-coverage guarantee.

| Weighted-overlap load bin [pu] | Episodes | Coverage | Median width [K] | Width range [K] |
|---:|---:|---:|---:|---:|
| 0.60–0.70 | 25 | 100% | 0.3511 | 0.3511–0.3678 |
| 0.70–0.80 | 334 | 88.3% | 0.3690 | 0.3678–0.3725 |
| 0.80–0.90 | 641 | 100% | 0.3725 | 0.3678–0.3725 |

**(c) Product reading.** Within common support, weighted median width rose by only `0.0214 K` across the lowest-to-highest bins and did not supply conditional reliability: the middle bin covered only `88.3%`. At the support boundary, width did not grow smoothly; it jumped from about `0.37 K` to unbounded, with `0%` finite availability. This is an honest refusal and a useful gap detector, but not a finite extrapolation product.

### The registered in-range sanity rule

**(a) Literal outcome.** Ordinary exchangeable coverage was `97.2%`; its Clopper–Pearson interval `[95.98%, 98.13%]` excludes `95%`, so `nominal_0_95_inside_exact_ci=false`. The registered sanity rule therefore failed through overcoverage, not undercoverage. Recomputing the stored rows reproduced `972/1,000`, and the stored `q` is exactly the 191st ordered calibration score.

**(c) Statistical audit.** This failure does not establish a conformal implementation error. Under a continuous-iid-score idealization, the registered order statistic gives marginal coverage `191/201=95.025%`, while the fixed-calibration coverage probability `F(S_(191))` varies as `Beta(191,10)` with SD `1.53` percentage points and central 95% range `91.63–97.58%`. Including 1,000 test draws gives an idealized beta-binomial central 95% coverage-count range of `913–978`; `972` lies inside it, with idealized upper-tail probability `P(K≥972)=8.56%`. The preregistered Clopper–Pearson rule would accept only `936–963` covered rows and would pass only about `59.4%` of full idealized repetitions, so it is mis-specified as a checksum for marginal coverage.

**(a) Continuity caveat.** The exact beta law does not apply to this artefact because only `112/200` calibration scores are unique: score `0.243396 K` repeats `89` times, and the corresponding exchangeable-test atom repeats `464/1,000` times. The selected rank-191 score `q=0.374063 K` is itself unique and lies above that atom. The continuous-law calculation is therefore a diagnostic benchmark, not a substitute for the stored empirical result. A test-set binomial interval remains conditional on this one realized calibration set and need not contain the repeated-calibration marginal target.

**(c) Decision consequence.** The preregistered Boolean is retained as failed; it is not redefined after seeing the result. Under §11, the registered E5 classification is therefore **reject**, despite independent rank, support, row-count, and coverage recomputation finding no implementation discrepancy. The separate scientific/product judgment is to investigate finite in-range bounds only through a newly preregistered replication and to reject finite out-of-support bounds.

### E5 fit and sampling limitation

**(a) NLS fit.** The single primary fit returned `Δθ_or=46.0154 K`, `τ_o=165.6738 min`, `Δθ_hr=21.7548 K`, and `τ_w=6.0401 min`, with combined training residual RMSE `0.7436 K` and no warning.

**(a) Protocol ambiguity/deviation.** The universal rule says at least ten seeds per stochastic cell, whereas the specific E5 section freezes one episode-generator seed (`61000`) with 200/1,000 independent episode samples. The run followed that specific E5 single-seed and episode-count prescription and the decision owner's requested counts. It estimates this calibration split's target-episode coverage from 1,000 draws but supplies no across-calibration-seed spread; that missing variability is material to the sanity-rule interpretation above.

## E1 reproduction gate

**(a) Verified run facts.** Run `e1-53df574b011a4dc4` completed with overall gate `pass`; no execution failure was recorded.

| Check | Predicted [K] | Observed [K] | Tolerance [K] | Status |
|---|---:|---:|---:|---|
| A mean RMSE | 2.59 | 2.59 | 0.1295 | pass |
| A largest signed peak | +6.17 | +6.17 | 0.3085 | pass |
| B mean RMSE | 1.77 | 1.77 | 0.0885 | pass |
| B largest signed peak | +3.17 | +3.17 | 0.1585 | pass |
| C mean RMSE | 0.11 | 0.11 | 0.0100 | pass |
| C largest signed peak | +0.32 | +0.32 | 0.0160 | pass |
| Private hot-spot RMSE | 1.55 | 1.55 | 0.0200 | pass |
| Private top-oil RMSE | 1.34 | 1.34 | 0.0200 | pass |

**(a) Synthetic aggregates.** E1 persisted means, extrema, and unsafe fractions but not per-seed rows or sample standard deviations; missing spread is reported rather than reconstructed.

| Model | Mean max-absolute error [K] | Mean signed peak [K] | Most-negative peak [K] | Unsafe fraction |
|---|---:|---:|---:|---:|
| A | 5.76 | +5.76 | +5.47 | 0.0 |
| B | 4.82 | +2.72 | +2.48 | 0.0 |
| C | 0.25 | −0.02 | −0.29 | 0.6 |

**(a) Private aggregate-only evidence.** No private rows, derived series, input path, or private stdout was persisted.

| Signal | Observations | Bias [K] | RMSE [K] | P95 absolute [K] | Worst absolute [K] |
|---|---:|---:|---:|---:|---:|
| Hot spot | 5,029 | −0.68 | 1.55 | 2.98 | 19.96 |
| Top oil | 5,029 | −0.93 | 1.34 | 2.70 | 5.19 |

## Failures, deviations, and execution status

- **(a) E3 validity failure:** cross-record noise reuse, reference-time mismatch up to `60 s`, and dense interpolation used as PINN measurement targets invalidate confirmatory E3 inference.
- **(a) E3 runtime/method failures:** none recorded; all five methods were available on all ten seeds, and all `200/200` method/load/seed score rows exist.
- **(a) E5 runtime/integrity failures:** none recorded. All expected calibration/test rows exist, manifest and aggregate hashes match, exact E1/E3 prerequisite identities are embedded, repository-integrity and CPU gates passed, and no override was used.
- **(a) E5 registered-criterion failure:** `nominal_0_95_inside_exact_ci=false`; this is retained as a failed sanity criterion even though the independent statistical audit identifies the criterion—not the rank implementation—as the problem.
- **(a) E1 resource-verification limitation:** the private adapter ran in a child process, while the old recorder captured only parent RSS. E1 therefore does not fully verify the preregistered process-tree memory gate.
- **(a) E3 resource-verification limitation:** the old recorder captured only the current Python process. E3 launched no experiment worker, but its short read-only Git provenance subprocesses were not included; the strict process-tree peak is therefore unresolved even though the recorded current-process peak was far below `2 GB`.
- **(a) Manifest limitation:** the old E1/E3 manifests omitted logical CPU count and explicit BLAS-thread limits. E3 separately records PyTorch intra-op threads `1`, CPU-only execution, CUDA unavailable, and zero visible CUDA devices.
- **(a) Schedule-resolution disclosure:** exact E3 train/validation amplitudes were frozen in committed code and in the hashed primary configuration after `PREREGISTRATION.md` was frozen but before primary access.
- **(a) Owner-directed sequence deviation:** E2 and standalone E4 were deferred, and E5 was run directly after the mandatory pre-access report checkpoint. The original sequence was E3→E2→E4→E5. Standalone E4 remains deferred because its outside-hull invariant was embedded in E3; its in-range adoption question remains unanswered.
- **(a) Recorded E5 train-record resolution and diagnostic disclosure:** E3 did not persist or designate one noisy training realization for later reuse. The committed E5 configuration records use of the same frozen E3 schedule, noise-free truth, reference locations, and NLS implementation with independent seed `61000` and corrected record-specific sensor streams. It also records the pre-access assertion that a training-only old-versus-corrected-stream audit generated no calibration/test episode loads or truth, claimed no sentinel, wrote no artefact, and changed no choice. **(c)** That non-exposure assertion is provenance testimony and cannot be reconstructed from the primary rows. “Fit once” was enforced once inside the primary protocol.
- **(a) Recorded E5 covariate-access disclosure:** the committed configuration records that, before a primary-draw guard was added, non-primary tests generated frozen target-load covariates in memory and used fake algebraic outcomes. It records no thermal-truth/fitted-model evaluation, persisted exact draw, sentinel claim, or changed choice; the code and tests now refuse public seed-`61000` draws before the E5 sentinel. **(c)** The historical non-exposure assertion is provenance testimony and is not independently recoverable from the primary rows.
- **(a) Private-data boundary:** the sibling `CoreField Startup` repository was read only; no private rows or derived private time series were copied here.
- **(c) No rescue tuning:** implementation corrections address audit failures and run-store integrity; no model architecture, loss weight, seed set, or result threshold was changed to rescue E3.

## Environment, runtime, and memory

**(a) Recorded environment.** All three runs used CPython `3.14.6` 64-bit on Windows 11 build `26200`; NumPy `2.5.2`, SciPy `1.18.1`, pandas `3.0.5`, matplotlib `3.11.1`, pytest `9.1.1`, and torch `2.14.0+cpu`. E1's private adapter used openpyxl `3.1.5`. E5 records 12 logical CPUs and requests one-thread numerical execution through environment limits for MKL, OpenBLAS, OpenMP, NumExpr, and VECLIB; CUDA was unavailable and hidden.

| Run | Code commit | Wall time [s] | Recorded peak / conservative bound [B] | Decimal MB | `<2 GB` recorded gate |
|---|---|---:|---:|---:|---|
| E1 `e1-53df574b011a4dc4` | `415a07229883af0a88bbc0665eae94a36567146c` | 14.3273 | 281,378,816 | 281.38 | pass for recorded parent; process tree unverified |
| E3 `e3-55464d9db759b4e2` | `bccae5081fc8c329f6d7d0b58b061818a99d9172` | 408.2047 | 373,751,808 | 373.75 | pass for current process; Git-child peaks unverified |
| E5 `e5-5bf123e28160484a` | `98dd22ca81f952bb2b595b759757e089d81a9c1e` | 6.9645 | 306,696,192 conservative bound | 306.70 | pass; parent 298,504,192 B + max Git child 8,192,000 B |

**(a) Integrity identifiers.** Frozen protocol SHA-256 is `4e50cb8ff5de827dfc18c0206c56baa0b127f31f294aaee2f5737636c1dac4c6`. E1 aggregate SHA-256 is `a14eb8e572672dc5e3fa1a3282a0c806887e6b8911fff390c503c7281ffa3e44`; E3 aggregate SHA-256 is `ef42547a939d2d1dea80f70258253ee37e8c909f477fbad9585e6dc0d0ea2f65`; E5 aggregate SHA-256 is `ddb977c2f836918786966394f2e58848affbe64c64ecc02c793a5c5062fc3347`.

## What would have to be true for these conclusions to be wrong

- **(c) E1:** the reproduction conclusion would fail if the aggregate-only private adapter did not execute the authorized source identified by its stored script hash, if its unmeasured child-process peak exceeded `2 GB`, or if the persisted aggregate is not reproducible under an independently controlled rerun.
- **(c) E3:** the large observed neural losses and unsafe direction could change under a corrected primary execution with independent record-level RNG streams, exact reference-time features, and actual measurement-time PINN losses. Until such a run exists, E3 supports no confirmatory superiority or rejection claim.
- **(c) Grey-box:** the retained outside-hull equality would be wrong if any stored E3 output were non-bit-exact, any residual were not positive zero, or any extrapolation flag were absent. The retained artefact passed those implementation checks, but standalone E4 could still reveal a useful in-range residual.
- **(c) Direct comparison:** “no demonstrated ML win” would become false if a valid preregistered run met the paired RMSE, safety, bootstrap, confirmation, resource, and integrity gates. E2 could also identify a scarce-reference niche; it remains deferred.
- **(c) E5:** the finite in-range assessment could change across independent calibration seeds, which this single-seed design did not measure. A finite strict-shift product would require calibration support covering the target loads, stable conditional behaviour, and a newly preregistered external validation; no allowed analysis of this artefact can manufacture those conditions.

## Evidence files

- **(a)** E1: [`aggregate.json`](runs/e1/e1-53df574b011a4dc4/aggregate.json), [`manifest.start.json`](runs/e1/e1-53df574b011a4dc4/manifest.start.json), [`manifest.final.json`](runs/e1/e1-53df574b011a4dc4/manifest.final.json), [access sentinel](run_state/e1.primary-access.json).
- **(a)** Superseded E3: [`aggregate.json`](runs/e3/e3-55464d9db759b4e2/aggregate.json), [`manifest.start.json`](runs/e3/e3-55464d9db759b4e2/manifest.start.json), [`manifest.final.json`](runs/e3/e3-55464d9db759b4e2/manifest.final.json), [access sentinel](run_state/e3.primary-access.json).
- **(a)** E5: [`aggregate.json`](runs/e5/e5-5bf123e28160484a/aggregate.json), [`manifest.start.json`](runs/e5/e5-5bf123e28160484a/manifest.start.json), [`manifest.final.json`](runs/e5/e5-5bf123e28160484a/manifest.final.json), [access sentinel](run_state/e5.primary-access.json).
- **(a)** Registered methods, predictions, and decision rules: [`PREREGISTRATION.md`](PREREGISTRATION.md).
