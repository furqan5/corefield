# EVIDENCE — label index for every claim in the README

Each claim carries a truth-discipline label:

- **(a)** verified fact, with the source or the test that verifies it
- **(b)** engineering estimate, with the assumptions it rests on
- **(c)** inference or judgement

"Verified" here means **reproduced by this repository on this machine**, not "appears in an
earlier document". The legacy notebooks stored zero cell outputs across all 73 code cells, so
every number below was re-derived rather than transcribed. `python -m pytest` re-derives them
again on every run.

The **Where verified** column names the test that would fail if the claim stopped being true.

---

## Headline claims

| # | Claim | Label | Where verified |
|---|---|---|---|
| 1 | Model A: RMSE 2.59 K, worst-case peak error **+6.17 K** at 1.30 pu | **(a)** | `test_stage_c.py::test_day_c_extrapolation_table` |
| 2 | Model B: RMSE 1.77 K, worst-case peak error **+3.17 K** at 1.30 pu | **(a)** | same |
| 3 | Model C: RMSE 0.11 K, worst-case peak error **+0.32 K** at 1.30 pu | **(a)** | same |
| 4 | A and B fail in the **over-prediction** direction | **(a)** | `test_stage_c.py::test_day_c_rivals_read_high_at_overload` |
| 5 | Over-prediction causes derating when capacity is worth most | **(c)** | Operational inference; not measured |
| 6 | Models A and B were driven by the **noise-free** top-oil signal | **(a)** | `models_ab.py` uses the truth series; `SingleExponentialFit.drive_is_noise_free` records it. Legacy reports described this channel as "measured" — see AUDIT.md §4.5 |
| 7 | The comparison is therefore handicapped in A and B's favour | **(c)** | Follows from 6; direction is unambiguous, magnitude not quantified |
| 8 | Whole campaign reproduces in ~17 s | **(a)** | Measured, this machine. Machine-dependent |

**Note on claim 1.** The legacy reports quote **+6.18 K**. This repository reproduces **+6.17 K**
(raw 6.1744). The 0.01 K difference is attributed to SciPy 1.17.1 → 1.18.0 changing one seed's
optimiser stopping point; Model A's other day-C values reproduce exactly. See REPRODUCTION.md.
The README quotes the reproduced value, not the published one.

## The four parameters and the standard

| # | Claim | Label | Where verified |
|---|---|---|---|
| 9 | The four parameters are the ones IEC 60076-7 says require a heat-run test with fibre-optic sensors | **(a)** | Checked against the standard 25 Aug 2026. It states these constants can be determined in a prolonged heat-run test, and that k21, k22 and tau_w require fibre-optic sensors |
| 10 | ONAF constants x=0.8, y=1.3, k11=0.5, k21=2.0, k22=2.0 | **(a)** | Checked against IEC 60076-7:2018 Ed. 2.0, 25 Aug 2026 — all five match, as do tau_o=150 min and tau_w=7 min. `test_physics.py::test_settled_constants_are_unchanged` pins the values in use |
| 10b | For OD/ODAF, k21 = 1.0, so the overshoot vanishes and the A/B/C separation does not transfer | **(a)** | Same check. `OD_MEDIUM_LARGE_POWER` in `iec60076_7.py`; `verify_k_assignment` returns 0.00 % overshoot for it |
| 10c | tau_w differs between natural and forced air (10 vs 7 min), so it cannot be shared across cooling stages | **(a)** | Same check. Corrected `staged.SHARED_BY_DEFAULT`, which had wrongly shared it |
| 11 | The **k-assignment** (which time constant on which branch) is verified | **(a)** | `test_physics.py::test_k_assignment_verified` — three independent numerical checks |
| 12 | Closed-form vs RK4 agreement: 1.097×10⁻⁷ K | **(a)** | same |
| 13 | Oil reaches 63.2 % of its step at t = k11·τ_o | **(a)** | same |
| 14 | Gradient overshoot 47.19 % at 41.0 min | **(a)** | same |
| 15 | Overshoot vanishes when k21 = 1 (small ONAN degeneracy) | **(a)** | `test_physics.py::test_overshoot_is_absent_without_slow_branch` |

## Estimator efficiency

| # | Claim | Label | Where verified |
|---|---|---|---|
| 16 | Estimator sits on the CRLB: 0.97 / 1.01 / 0.95 / 0.97× at 400 seeds | **(a)** | `test_crlb.py::test_efficiency_converges_with_more_seeds` (slow marker) |
| 17 | Bias below 0.12 % on all four parameters | **(a)** | `test_crlb.py::test_estimator_is_unbiased` |
| 18 | "No estimator can do materially better on this data" | **(a)** | Follows from the CRLB being a bound on *all* unbiased estimators — a theorem, not a measurement. Conditional on the Gaussian-noise model and on unbiasedness (17) |
| 19 | Four-parameter CRLB: 0.079 / 0.757 / 0.631 / 2.971 % | **(a)** | `test_crlb.py::test_crlb_values_at_campaign_configuration`. **New** — the legacy 0.09/0.43/0.64/3.90 % was the older single-exponential model |
| 20 | ρ(τ_o, τ_w) = −0.078, so the problem is not degenerate | **(a)** | `test_crlb.py::test_four_parameter_problem_is_not_degenerate` |
| 21 | The circulated "all four at 0.99–1.02× CRLB" was unsupported | **(a)** | AUDIT.md §5.1, WITHDRAWN.md. That figure was a two-parameter result on a different model, covering 6 of 8 cells |
| 22 | A ±5 % efficiency band is unachievable at 10 seeds | **(a)** | `test_crlb.py::test_ten_seeds_cannot_resolve_the_ratio`. Analytic: SE = 0.7555/√n |

## Commissioning guidance

| # | Claim | Label | Where verified |
|---|---|---|---|
| 23 | One load event → ~12 % floor on τ_w; two events → ~4 % | **(a)** | `test_crlb.py::test_single_event_calibration_hits_an_information_floor` (12.30 % / 3.98 %) |
| 24 | "Commission over at least two load events" | **(c)** | Recommendation inferred from 23. The bound is a fact; the threshold is a judgement about acceptable precision |
| 25 | Reads at 3/8/18/48 min carry rate then amplitude | **(a)** for the fractions (1−e^(−3/7)=0.348, 0.681, 0.923; 48 min = 6.9 τ_w) | `synthetic.calibration_indices` docstring; arithmetic |
| 26 | Sampling a transient without anchoring its asymptote leaves rate and amplitude correlated | **(a)** | Legacy v1 observability study: 147 % and 36 % τ_w error for schedules that fail this |
| 27 | Load sampling dominates oil sampling: +2.1 % vs +8.4 % on τ_w | **(a)** | `test_ingest.py::test_load_sampling_rate_matters_more_than_oil_sampling_rate` |
| 28 | "Insist on 1-minute load; 5-minute top-oil is fine" | **(c)** | Recommendation inferred from 27 |
| 29 | 1-minute logging is "usually free" because historians sample faster than they store | **(b)** | Industry expectation, not measured. Verify per site |

## Failure modes

| # | Claim | Label | Where verified |
|---|---|---|---|
| 30 | Ignoring ambient under-predicts the peak by 3.09 K | **(a)** | `test_stage_c.py::test_ambient_ignored_fails_in_the_dangerous_direction` |
| 31 | Ambient measured → 0.08 K RMSE; ignored → 3.98 K | **(a)** | same, and `test_ambient_measured_passes_cleanly` |
| 32 | Ambient reaches the winding through a ~75-minute low-pass | **(a)** | k11·τ_o = 0.5 × 150 min. Arithmetic on the synthetic unit's parameters |
| 33 | "An hourly weather feed is sufficient" | **(b)** | Follows from 32 under the synthetic unit's τ_o. A unit with a much shorter τ_o would need faster ambient |
| 34 | Spikes and quantisation cost nothing | **(a)** | `test_stage_c.py::test_spikes_did_not_degrade_plain_least_squares`, `test_quantization_costs_about_one_times_baseline` |
| 35 | Drift passes the gate while poisoning parameters | **(a)** | `test_stage_c.py::test_drift_poisons_parameters_while_the_trajectory_passes` |
| 36 | WTI +3 K bias → +14.5 % Δθ_hr, +4.1 K at true peak | **(a)** | `test_stage_c.py::test_wti_calibration_bias_fails` |
| 37 | The bias distorts dynamics, not just level (τ_w +10.6 %) | **(a)** | same |
| 38 | "A relative-trends-only positioning does not escape it" | **(c)** | Commercial inference from 37 |
| 39 | "Commissioning requires ≥1 bias-audited reference per unit" | **(c)** | Recommendation inferred from 36–37 |
| 40 | CT gain +2 %: trajectory compensated (0.15 K), parameters carry −2.6 % | **(a)** | `test_stage_c.py::test_ct_gain_compensates_in_trajectory_but_not_parameters` |
| 41 | Corruption magnitudes are plausible instrument bounds | **(b)** | Engineering estimates. No measured field values exist |

## Hot-spot location

| # | Claim | Label | Where verified |
|---|---|---|---|
| 42 | Top-oil is exactly invariant to hot-spot location (0.0000000 K) | **(a)** on the model | `test_observability.py::test_top_oil_is_exactly_invariant_to_hot_spot_location` |
| 43 | The mechanism is conservation of energy | **(a)** | `test_observability.py::test_total_loss_is_conserved_under_relocation` |
| 44 | External CRLB on location: ±40 % of winding height | **(a)** on the model, **(b)** on realism | `test_observability.py::test_external_measurements_cannot_locate_the_hot_spot` |
| 45 | Two internal probes: ±0.33 % | **(a)** on the model | `test_observability.py::test_two_internal_probes_solve_the_problem` |
| 46 | Resolving location externally needs ~11× better instrumentation | **(b)** | `test_observability.py::test_external_route_would_need_implausible_instrumentation`. The 0.1–0.5 K comparison class is an engineering estimate |
| 47 | The 1-D axial model's leading-order conclusion is robust to its simplicity | **(c)** | Argued in `observability.py`: the leading term is an exact cancellation any oil-path model inherits. Not tested against CFD |
| 48 | Existing practice places probes near 90 % of winding height | **(a)** | Literature — see ASSESSMENT.md sources |

## Package behaviour

| # | Claim | Label | Where verified |
|---|---|---|---|
| 49 | Refuses to fit without ambient | **(a)** | `test_ingest.py::test_missing_ambient_is_refused_not_warned` |
| 50 | Ships no loading limits; `iec_loading_limits()` raises | **(a)** | `test_envelope.py::test_package_refuses_to_supply_iec_limits` |
| 51 | Placeholder provenance strings are rejected | **(a)** | `test_envelope.py::test_placeholder_sources_are_rejected` |
| 52 | Raises rather than returning a railed fit | **(a)** | `test_estimator.py::test_all_starts_failing_raises_rather_than_returning_the_least_bad` |
| 53 | Observations are never interpolated; inputs are | **(a)** | `test_ingest.py::test_observations_are_not_interpolated` |
| 54 | Kelvin input is detected and rejected or converted | **(a)** | `test_physics.py::test_kelvin_ambient_is_rejected`, `test_ingest.py::test_kelvin_temperatures_are_converted` |
| 55 | Contains no neural networks | **(a)** | No torch dependency in `pyproject.toml`; grep the tree |
| 56 | The fast integrator is an exact restatement of the RK4 loop | **(a)** | `test_physics.py::test_fast_integrator_matches_reference_loop` (agrees to <1e-10 K) |
| 57 | Runs CPU-only in well under 2 GB | **(a)** | Measured 24 Aug 2026: **136.6 MiB process RSS**, 52.3 MiB peak Python allocation, running the day-C comparison plus three corruption scenarios. Against the 2 GB budget in CLAUDE.md that is a factor of ~15 margin |

## First field validation

The underlying data was supplied privately, is not in this repository, and may not be
redistributed. Results are reportable. These claims are therefore **(a) but not
reader-reproducible** — a reader can check the reasoning and the code paths, not the numbers.
That is a weaker standard than everything else in this file and is marked as such.

| # | Claim | Label | Where verified |
|---|---|---|---|
| 58 | Out-of-sample hot-spot RMSE **1.54 K** over 35 unseen days, 5 029 observations, τ_w held at Table 4 | **(a)**, not reader-reproducible | private field record (`final_tr3.py`), withheld |
| 59 | The same fit scores **7.91 K** when the stuck-channel window is included | **(a)**, not reader-reproducible | private field record (`clean_validation.py`), withheld |
| 60 | The rejected window holds load at exactly one value (0.01 pu) for 7.07 days while ambient swings 12 K, top-oil swings 11 K and the fan stage keeps switching | **(a)**, not reader-reproducible | private field record (`outage_check.py`), withheld |
| 61 | A transformer genuinely at 0.01 pu would show a steady oil rise of Δθ_or/(1+R) ≈ 6 K, not the measured 22 K | **(b)** | Follows from the model with the identified Δθ_or and the assumed R = 6 |
| 62 | The out-of-sample error is not seasonal: a within-segment split, same season, still shows it | **(a)**, not reader-reproducible | private field record (`transfer_test.py`), withheld |
| 63 | The measured quasi-steady winding exponent is y ≈ 0.94, against Table 4's 2.0 for OD | **(a)**, not reader-reproducible | private field record (`gradient_check.py`), withheld. **Caveat:** 5 552 of 6 576 points sit in one load bin — direction solid, value not |
| 64 | Loss ratio R is not load-bearing: out-of-sample RMSE moves 1.43 → 1.47 K across R = 5…10 | **(a)**, not reader-reproducible | private field record (`r_and_tr2.py`), withheld |
| 65 | Nothing in the three-unit corpus exceeds 0.93 pu, so the overload extrapolation is unvalidated | **(a)**, not reader-reproducible | private field record (`recon.py`), withheld |
| 66 | A second unit refused identification — 0.10 pu of load variation across 17 days | **(a)**, not reader-reproducible | private field record (`r_and_tr2.py`), withheld |

## Above nameplate — the regime the loading envelope exists to inform

Source: Nordman and Lahtinen, *IEEE Trans. Power Del.* 18(1), 2003, pp. 107–112 — published
fibre-optic measurements on a 400/400/125 MVA ONAF unit at 0.65, 1.00, 1.29 and 1.60 pu. The
paper is copyrighted; its tables are not reproduced here.

| # | Claim | Label | Where verified |
|---|---|---|---|
| 72 | Identified below nameplate and extrapolated, the fixed-exponent model reads **6.35 K low** at 1.60 pu — the unsafe direction | **(a)** | private overload record (`nordman_overload.py`), withheld |
| 73 | It does **not** beat the generic table it aims to replace: 4.64 K against 4.21 K RMSE across the two held-out overload points | **(a)** | same |
| 74 | The measured oil exponent is not constant: 0.717, 0.766, 0.846 over successive load intervals | **(a)** | same |
| 75 | Freeing the load-slope on two sub-nameplate points is **under-determined** and the fit refuses | **(a)** | `overload_refit.py`, withheld |
| 76 | Fitting a load-slope across the load at which the hot spot changes winding costs **8 K**, worse than making no correction | **(a)** | same |
| 77 | With amplitudes measured at nameplate (f(1)=1 for any exponent) and both slopes free, the held-out 1.60 pu error is **−2.63 K** hot spot, **−0.97 K** top oil | **(a)** | `overload_anchored.py`, withheld |
| 78 | That is 59 % of the unsafe bias removed, with x(1.60)=0.790 and y(1.60)=1.369 against tabulated 0.8 and 1.3 | **(a)** | same |
| 79 | Claim 77 uses a 1.29 pu observation, so it is evidence for a commissioning excursion, **not** for identification from service data | **(a)** | Statement about the experiment's design, verifiable by inspection |
| 80 | Every fit in claims 72–78 is exactly determined and carries no residual or error bars | **(a)** | Follows from parameter and observation counts |
| 81 | ρ(x₀,x₁) = 0.995 over an in-service load band; sd(x₁) is ~4× the parameter. Over 0.60–1.30 pu it is 5 % | **(a)** | `exponent_identifiability.py` — reader-reproducible, no private data |
| 82 | The two exponent sensitivities differ only by the factor (K−1), so load range alone separates them | **(a)** | Analytic; derivable from the model |
| 83 | Load-dependent exponents are implemented and default to zero, reproducing the fixed form exactly | **(a)** | `test_physics.py::test_zero_slope_reproduces_the_fixed_exponent_exactly` |
| 84 | A positive slope raises the predicted rise on **both** sides of nameplate, because the loss factor's base is below one for K < 1 | **(a)** | `test_physics.py::test_a_positive_oil_slope_raises_the_predicted_overload_temperature` |

**Not claimed:** that CoreField can compute a safe loading envelope above nameplate. Four separate
tests say it cannot, today, from service data alone.

## Package behaviour found and fixed by the field data

| # | Claim | Label | Where verified |
|---|---|---|---|
| 67 | The staged estimator accepted solutions railed against the τ_w < τ_o constraint, reporting them as converged and interior | **(a)** | Fixed 26 Aug 2026; `test_staged.py::test_a_solution_pressed_against_the_tau_w_constraint_is_refused` |
| 68 | From some starts the same condition raised a bare `ValueError` out of `_unpack` instead of the designed refusal | **(a)** | Same fix; same test |
| 69 | `fixed=` holds a parameter, excludes it from the optimiser, and reports it as held rather than identified | **(a)** | `test_staged.py::test_a_fixed_parameter_is_held_exactly_and_declared` |
| 70 | A channel pinned on one exact value passes every other ingest check | **(a)** | `test_ingest.py::test_a_stuck_load_channel_is_reported` |
| 71 | `STUCK_CHANNEL_HOURS = 48` sits in an empty gap in the corpus: longest genuine constant-load run 34.8 h, longest defective run 169.8 h | **(b)** | Measured over three units, four segments. A unit genuinely held flat for three days would trip it wrongly |

## Limitations section

Every claim in the README's Limitations section is **(a)** — each is a statement that something
has *not* been done, verifiable by inspection of this repository. Specifically: one field
validation on one unit below nameplate and otherwise synthetic data, one synthetic unit, Model C
structure-matched to its own truth, IEC text unverified, corruption magnitudes estimated,
uncertainty band parameter-only, WTI bias tested only as a constant offset, and the observability
model simplified.

## What is NOT claimed anywhere

Recorded so that absence is deliberate rather than accidental:

- **No accuracy claim against a real transformer that a reader can reproduce.** One field result
  exists (claims 58–66) and its data may not be redistributed, so a reader can audit the method
  and the reasoning but not re-derive the number. It covers one unit, over 0.04–0.90 pu, with
  three of four parameters identified and the fourth tabulated.
- **No validated claim about loading above nameplate.** The loading envelope extrapolates past
  the load hull of every record this project has seen. Claim 65.
- **No comparison against commercial winding-temperature indicators.** A "±5–15 K" figure appears
  in the legacy v1 report with **no citation anywhere** (AUDIT.md §5.2). It is the denominator of
  an attractive pitch line and it is unsourced, so it does not appear in the README, the demo, or
  any other file here. If a source is found it becomes **(a)**; if it is a field judgement it can
  appear as an explicitly-labelled **(b)** belonging to its author.
- **No novelty claim.** PINNs for transformer thermal modelling, sparse-sensor field
  reconstruction, and IEC-parameter calibration from operating data are all published work. See
  ASSESSMENT.md §2 for who did what.
- **No aging or loss-of-life calculation.** Not implemented.
