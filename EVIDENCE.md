# EVIDENCE — label index for every claim in the README

Each claim carries a truth-discipline label:

- **(a)** verified fact, with the source or the test that verifies it
- **(b)** engineering estimate, with the assumptions it rests on
- **(c)** inference or judgement

Distinguish a source fact, a reproducible numerical result, and physical validation. A passing
synthetic test establishes implementation behaviour under its assumptions, not field accuracy.
Private case studies and published projections are labelled separately below; the public test
suite does not rerun private measurements or establish permission to publish them.

**28 Aug 2026 correction:** earlier descriptions of a measured 2.5-pu test, automatic diagnostic
refusals, universal exponent drift, and a same-fit 7.91→1.54 K improvement were too strong.
The corrected entries below supersede those descriptions. See `docs/validation_scope.md`.

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
| 9 | The model's winding parameters require information not present in top-oil alone | **(a), model structure** | Winding parameters do not enter the oil equation. Claims about precise IEC requirements remain UNVERIFIED against a licensed copy |
| 10 | The implementation retains x=0.8, y=1.3, k11=0.5, k21=2.0, k22=2.0 for its ONAF example | **(a), implementation** | `test_physics.py::test_settled_constants_are_unchanged`; mirror-sourced, not licensed-source verification |
| 10b | The implemented OD constants set k21=1, removing the slow gradient branch | **(a), model algebra** | `OD_MEDIUM_LARGE_POWER`; does not validate transfer between cooling classes |
| 10c | The staged estimator does not share τ_w by default | **(a), implementation choice** | `staged.SHARED_BY_DEFAULT`; the cited tabulated priors remain mirror-sourced, not measurements |
| 11 | The implemented branch assignment passes numerical consistency checks | **(a), implementation only** | `test_physics.py::test_k_assignment_verified`; self-consistency cannot verify the standard or a physical transformer |
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

The underlying data was supplied privately and is not redistributed. The following numbers
are recorded internal analyses, not rerun by the public suite. The validation exclusion and
model choices require review; permission for external reporting must be checked separately.

| # | Claim | Label | Where verified |
|---|---|---|---|
| 58 | Recorded conditional hot-spot RMSE **1.54 K** over 5,029 later-period observations, τ_w held fixed and a constant-load window excluded | **(b), internal reanalysis** | `final_tr3.py`, private; not an independent prospective validation or worst-case bound. **Superseded by claim 99:** this row used an assumed loss ratio R = 6 |
| 59 | The earlier **7.91 K** score used a different fitting configuration | **(b), internal reanalysis** | `clean_validation.py`, private; not the same fit with/without masking, so a detector-only improvement is not established |
| 60 | The rejected window holds load at exactly one value (0.01 pu) for 7.07 days while ambient swings 12 K, top-oil swings 11 K and the fan stage keeps switching | **(a)**, not reader-reproducible | private field record (`outage_check.py`), withheld |
| 61 | A transformer genuinely at 0.01 pu would show a steady oil rise of Δθ_or/(1+R) ≈ 6 K, not the measured 22 K | **(b)** | Follows from the model with the identified Δθ_or and R; R was assumed 6 here and is sourced at 6.8 in claim 99, which does not change the conclusion |
| 110 | **The data supplier states he believes the constant-load window was a monitoring issue, and has authorised excluding it** (correspondence, 1 Sept 2026) | **(a)** for what he said; **(b)** for the root cause, which he hedged as "I think" | Written reply from the engineer who supplied the records. This moves the exclusion from an analyst's stated-in-advance rejection rule to a supplier-authorised one. It does **not** make the fault a verified instrument diagnosis, and it changes no other condition on the score |
| 111 | Both EDF contacts have now given **written** consent to be named in the acknowledgement | **(a)** | Correspondence, 1 Sept 2026. Luc's covers the earlier general wording; the sentence crediting him for the ambient method is still awaiting his specific approval |
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
| 74 | Calculated interval oil power-law exponents are 0.717, 0.766, 0.846 for this ONAF case | **(b), derived quantities** | same; not direct measurements of x1, a proven mechanism, or an ODAF/ONAN slope |
| 75 | Freeing the load-slope on two sub-nameplate points is **under-determined** and the fit refuses | **(a)** | `overload_refit.py`, withheld |
| 76 | Fitting a load-slope across the load at which the hot spot changes winding costs **8 K**, worse than making no correction | **(a)** | same |
| 77 | An exploratory single-winding fit anchored at nameplate and using data through 1.29 pu gives **−2.63 K** hot-spot and **−0.97 K** oil error at 1.60 pu | **(b)** | `overload_anchored.py`; model choices followed inspection of the small dataset. Anchor measurements also have uncertainty |
| 78 | The errors in claims 72 and 77 concern different fitting specifications and winding-target choices | **(a), experiment design** | Comparing their magnitudes does not establish a matched 59% improvement or a general repair |
| 79 | Claim 77 uses a 1.29-pu observation | **(a), experiment design** | It does not demonstrate prediction from an exclusively below-nameplate record |
| 80 | Parameter counting must use independent scalar observations and Jacobian rank, not load-level count alone | **(a), algebra** | A level may supply both oil and winding observations; noise, covariance and degrees of freedom remain to be assessed |
| 81 | ρ(x₀,x₁) = 0.995 over an in-service load band; sd(x₁) is ~4× the parameter. Over 0.60–1.30 pu it is 5 % | **(a)** | `exponent_identifiability.py` — reader-reproducible, no private data |
| 82 | The exponent sensitivities differ by (K−1); diversity affects rank, while independent count and noise affect precision | **(a)** | Model Jacobian and the additional IID scaling tests in `test_crlb.py` |
| 83 | Load-dependent exponents are implemented and default to zero, reproducing the fixed form exactly | **(a)** | `test_physics.py::test_zero_slope_reproduces_the_fixed_exponent_exactly` |
| 84 | A positive slope raises the predicted rise on **both** sides of nameplate, because the loss factor's base is below one for K < 1 | **(a)** | `test_physics.py::test_a_positive_oil_slope_raises_the_predicted_overload_temperature` |

**Not established:** a safe operating envelope from these analyses. This is a limitation of
the evidence, not a theorem that every sufficiently informative operating archive must fail.

## Transient comparison: corrected provenance and scope, 28 Aug 2026

**(a, source)** The Nordman and Lahtinen unit is **ONAF**, not ODAF. Section V and Table VII
construct a 2.5-pu scenario using response ratios derived from the measured 1.6-pu curves.
The paper does not report a physically measured 2.5-pu test. Its design and cooling class do
not establish transferable performance or exponent slopes on other units.
[Source DOI](https://doi.org/10.1109/TPWRD.2002.807747).

| # | Corrected claim | Label | Evidence and boundary |
|---|---|---|---|
| 85 | Table VII projects 156 °C for the 120 kV winding at 2.5 pu after 20 min, using R₂₀ = 0.56 derived from the 1.6-pu curves; Tables V/VI give 79/89 °C using the guide methods compared in that paper | **(a), published calculation, not a 2.5-pu measurement** | Source pp. 111–112, Tables V–VII and Section V assumptions |
| 86 | The private simulation gives approximately 160 °C and R₂₀ = 0.519 under its stated parameter assumptions | **(b)** | `transient_overload.py`; agreement with a projection, not a measured overload validation. The former 4 K physical-test accuracy claim is withdrawn |
| 87 | The second-winding comparison gives approximately 0.487 versus the source-derived ratio 0.39, and about +13 K versus the source projection | **(b)** | Same script; positive error is not a demonstrated safety margin |
| 88 | The simulated ratio changes substantially with τ_w, k21 and k11 | **(b)** | Same sensitivity calculation; a normalized ratio does not isolate one parameter or remove dependence on relative oil/winding amplitudes |
| 89 | Holding the remaining parameters fixed, matching the source-derived ratios implies k21 values of approximately 2.29 and 1.25 | **(b), conditional inversion only** | Same script; not independent identification, a confidence interval, or a third route avoiding hot-spot observations |
| 90 | k21 cancels from the steady-state gradient | **(a), model algebra** | k21·g − (k21−1)·g = g |
| 91 | The one-parameter synthetic k21 bound is much tighter with transient observations than with late observations | **(a), conditional numerical test** | `test_crlb.py` overshoot tests; all other parameters are assumed known. Passing does not establish joint identifiability |
| 92 | The local-exponent heuristic flags the published max-of-windings series | **(a), implementation behaviour only** | `test_observability.py::test_a_winding_handover_is_detected`; not a validated physical detector, and not automatically a fit refusal |
| 93 | Earlier level-count simulations were exploratory, not an approved commissioning specification | **(b), historical simulation** | `excursion_levels.py`; its six-level cutoff is a code choice, not proof that six distinct loads are mathematically necessary |
| 94 | Wider, more informative designs can improve precision, but the reported range/count ratios are scenario-specific | **(b)** | Same script; sample count, noise, dwell, covariance and prediction target also matter. No requirement to deliberately overload a service transformer follows |

**Still open:** prospective measured-transient validation; reliable joint slope estimation;
joint k21/k22/time-constant identifiability; validated handling of winding identity and cooling
changes; replication. No physical mechanism or slope magnitude is established for OD/ODAF or
ONAN by this ONAF case.

## Cooling-specific precision diagnostics and safe reporting

| # | Claim | Label | Where checked |
|---|---|---|---|
| 95 | The illustrative oil-slope reference is not silently reused with OD/ONAN constants | **(a), implementation** | `test_crlb.py::test_other_cooling_classes_require_an_explicit_slope_reference` |
| 96 | SVD rejects fewer than three independent oil-information directions; repeating a full-rank IID design improves standard error as 1/√N | **(a), model algebra and tests** | `test_two_distinct_levels_cannot_determine_three_oil_parameters`, `test_repeating_identical_full_rank_design_halves_std_at_four_times_count` |
| 97 | A below-nameplate design can meet a conditional precision threshold without validating overload behaviour | **(a), synthetic test** | `test_informative_below_nameplate_design_does_not_require_overloading` |
| 98 | The envelope warning does not transfer a single ONAF case-study error into another unit's safety margin | **(a), implementation** | `test_envelope.py::test_extrapolation_beyond_the_fitted_hull_is_flagged` |

## Package behaviour found and fixed by the field data

| # | Claim | Label | Where verified |
|---|---|---|---|
| 67 | The staged estimator accepted solutions railed against the τ_w < τ_o constraint, reporting them as converged and interior | **(a)** | Fixed 26 Aug 2026; `test_staged.py::test_a_solution_pressed_against_the_tau_w_constraint_is_refused` |
| 68 | From some starts the same condition raised a bare `ValueError` out of `_unpack` instead of the designed refusal | **(a)** | Same fix; same test |
| 69 | `fixed=` holds a parameter, excludes it from the optimiser, and reports it as held rather than identified | **(a)** | `test_staged.py::test_a_fixed_parameter_is_held_exactly_and_declared` |
| 70 | A channel pinned on one exact value passes every other ingest check | **(a)** | `test_ingest.py::test_a_stuck_load_channel_is_reported` |
| 71 | `STUCK_CHANNEL_HOURS = 48` sits in an empty gap in the corpus: longest genuine constant-load run 34.8 h, longest defective run 169.8 h | **(b)** | Measured over three units, four segments. A unit genuinely held flat for three days would trip it wrongly |

## The published fleet model, read 1 Sept 2026

The article describing the source dataset was obtained from its author and read. It is
copyrighted and is not redistributed; the local copy stays out of version control.
Reference: L. Paulhiac and R. Desquiens, "Dynamic Thermal Model for Oil Directed Air Forced
Power Transformers With Cooling Stage Representation," *IEEE Trans. Power Del.*, vol. 37,
no. 5, pp. 4135–4144, Oct. 2022, doi: 10.1109/TPWRD.2022.3145003.

| # | Claim | Label | Where verified |
|---|---|---|---|
| 99 | With the loss ratio sourced at **R = 6.8** instead of the assumed 6.0, the same fit gives out-of-sample hot-spot RMSE **1.55 K** and top-oil **1.34 K**, against 1.54 K and 1.31 K previously | **(b), internal reanalysis** | `final_tr3.py`, private. The number moved by 0.01 K; what changed is that R is no longer an assumption |
| 100 | The loss ratio for the 360 MVA unit is **612 kW / 90 kW = 6.8** | **(a)** | The article's Table IV, p. 4140, which tabulates the three units of the dataset by name |
| 101 | That article reports **no confidence intervals, no identifiability analysis, no information bound and no refusal criterion**; it fits by particle swarm optimisation or by hand | **(a)** | Direct reading, §VII and §VIII. Supports ASSESSMENT.md's prior-art gap as a directed check rather than a literature sweep |
| 102 | Its stated applications include **cooling-efficiency monitoring and load-capability evaluation**, with a measured case in which a cleaned exchanger reduced the identified rated oil rise by about 20 K | **(a)** | Direct reading, §IX–X and Fig. 21. **This is prior art on two CoreField product claims; no novelty is asserted for either** |
| 103 | Its model treats the winding time constant **τ_w as a required input, not a fitted parameter** | **(a)** | Direct reading: τ_w appears in its Table I (minimum data) and not in its Table III (fitted parameters). The `HELD, NOT IDENTIFIED` handling here matches the closest comparable practice |
| 104 | The first cooling stage of the 360 MVA unit **falls outside that model's own assumptions**, and is handled there by an empirically fitted special case | **(a)** | Direct reading, p. 4142 and its Eq. (37). The stage-1 difficulty is a property of the unit, not of the method used here |
| 105 | **Oil viscosity does not explain the ONAF oil-exponent drift.** Substituting the published viscosity correction for the load-dependent exponent gives n_v = 0.55–0.70 against a published range of 0.25–0.45, roughly doubles held-out error (RMSE 4.41 → 7.92 K), and produces an apparent exponent that **falls** with load where the measurement **rises** | **(b), falsified hypothesis** | `viscosity_exponent.py`, private. Two falsifiers were stated before the run; both fired. The buoyancy explanation is unaffected and `x1` stays empirical |
| 106 | Inverting the steady-state oil model for ambient and differencing against the measured channel recovers an injected +3 K ambient probe offset to within 0.3 K, and reports it as the unsafe direction | **(a), synthetic** | `test_observability.py::test_a_probe_reading_warm_is_flagged_and_named_as_the_unsafe_direction` |
| 107 | A cooling-stage-dependent probe error whose **mean is near zero** is detected by the stage spread, which a mean-only test passes | **(a), synthetic** | `test_observability.py::test_a_stage_dependent_offset_points_at_the_probe_not_the_model`. Method due to L. Paulhiac, in correspondence |
| 108 | The check declines to report, rather than reporting noise, when the load never holds still for three oil time constants | **(a), synthetic** | `test_observability.py::test_a_record_that_never_settles_declines_to_report` |
| 109 | `scripts/reproduce_study.py` regenerates `paper/generated/study_results.json` with **all 137 fields identical** to the committed copy | **(a)** | Direct comparison, 1 Sept 2026. The script had gone missing from the working tree while the manuscript instructed reviewers to run it |


## Second unit: a public CC-BY record, 2 Sept 2026

Baerug, Madshaven & Espedal (SINTEF Energi AS), "Supporting data for thermomechanical modelling of
clamping force in power transformer in operation," Zenodo, 2025,
doi: 10.5281/zenodo.17223516, CC-BY-4.0. 40 MVA ONAN, southern Norway, hourly, calendar 2024.
**Redistributable and citable, unlike the ODAF records.** Companion paper forthcoming; citing it is
a licence obligation.

| # | Claim | Label | Where verified |
|---|---|---|---|
| 112 | The record's load hull is **0.048-0.429 pu** against a 175.0 A rated HV current; it never approaches nameplate | **(a)** | `private/sintef/characterise.py`. Reaching 1.60 pu from here is a 3.7x extrapolation, so this record cannot support any near- or above-nameplate claim |
| 113 | Sampling is **hourly**, giving 0.17 samples per ON winding time constant against Table 4's 10 min | **(a)** | Six times worse than the 10-minute ODAF record. tau_w is not identifiable here and must be held |
| 114 | The governing winding hot spot reads **below** top oil in **74.4 %** of 6,242 quasi-steady samples | **(a)** | The IEC form cannot produce a negative gradient at any positive parameter value, so the record is not describable by the model as it stands |
| 115 | That negative gradient is orderly, not noise: it rises monotonically with load and crosses zero near 0.22 pu | **(a)** | `gradient_and_fit.py`, seven load bands |
| 116 | A constant datum offset explains it: `Delta_hr*K^y - C` gives C = 11.3 K, y = 0.98 and residual **0.99 K**, against **4.64 K** for the IEC form with no offset, whose exponent rails | **(b)** | `offset_hypothesis.py`. Reading (c): the fibre probes and the top-oil probe do not share a reference point. Not confirmed with the data's authors |
| 117 | With `y` fixed at the tabulated value, as the package does, the rated gradient is driven to its **lower bound** and the existing railed-parameter check refuses the fit | **(a)** | Correct behaviour, and it reports the symptom without the cause -- which is why claim 118 exists |
| 118 | `observability.check_gradient_datum` reports the negative fraction, fits both forms, and names the offset without ever subtracting it | **(a)** | `test_observability.py`, five tests; recovers an injected 11 K offset to within 0.5 K |
| 119 | **WITHDRAWN.** `check_ambient_consistency` was first reported as not suspect at +1.00 K on this record. Delta_or had been fitted on the same record the check was applied to, so the number was not independent | **withdrawn** | `private/sintef/ambient_check2.py`. Superseded by claims 121-123 |
| 121 | Breaking the circularity **flips the verdict**: fit on the first half and check the second gives +1.39 K and SUSPECT; the reverse gives +0.48 K and not suspect. The halves disagree on the rated oil rise by 2.20 K | **(a)** | A test whose answer depends on an arbitrary split has not answered anything on this record |
| 122 | The reported offset **passes only at the loss ratio we assumed**. Sweeping R from 4 to 10 moves it from +3.64 K to -3.12 K, changing sign; R is not supplied with the dataset | **(a)** | On this record the mean offset measures our own assumption, not the ambient probe. The sharpest limitation yet demonstrated for that check |
| 123 | The residual carries a **2.99 K day-night swing** (day -1.50 K, night +1.48 K) against a mean of +0.01 K, with the sign a sun-exposed probe would produce | **(a)** for the structure; **(c)** for the cause | Confounded with oil thermal lag: the quasi-steady test constrains load stability, not ambient stability. Not attributable to solar gain without the dynamic model |
| 124 | An ONAN unit has no cooling stages, so the check's stage-split diagnostic -- its sharper axis -- is unavailable, leaving only the mean | **(a)** | Structural limitation of the check on stageless units |
| 120 | The suppliers' own top-oil model scores RMSE 1.30 K, bias -0.01 K against the measurement over 8,592 rows | **(a)** | `Barug2025_calculated.csv`. Not stated whether in-sample, so a reference point rather than a benchmark beaten or lost |

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
- **No validated operational claim about loading above nameplate.** The ODAF records and the
  exploratory ONAF reanalysis do not establish one. Claims 65 and 85–89.
- **No comparison against commercial winding-temperature indicators.** A "±5–15 K" figure appears
  in the legacy v1 report with **no citation anywhere** (AUDIT.md §5.2). It is the denominator of
  an attractive pitch line and it is unsourced, so it does not appear in the README, the demo, or
  any other file here. If a source is found it becomes **(a)**; if it is a field judgement it can
  appear as an explicitly-labelled **(b)** belonging to its author.
- **No novelty claim.** PINNs for transformer thermal modelling, sparse-sensor field
  reconstruction, and IEC-parameter calibration from operating data are all published work. See
  ASSESSMENT.md §2 for who did what.
- **No aging or loss-of-life calculation.** Not implemented.
