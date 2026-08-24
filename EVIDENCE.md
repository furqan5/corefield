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
| 9 | The four parameters are the ones IEC 60076-7 says require a heat-run test with fibre-optic sensors | **(a)** for the standard's statement, **mirror-sourced and UNVERIFIED** for the text itself | Legacy methods v4 §9.4. **Not** re-verified against a licensed copy |
| 10 | ONAF constants x=0.8, y=1.3, k11=0.5, k21=2.0, k22=2.0 | **UNVERIFIED** — mirror-sourced | `test_physics.py::test_settled_constants_are_unchanged` pins the *values in use*, not their correctness |
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

## Limitations section

Every claim in the README's Limitations section is **(a)** — each is a statement that something
has *not* been done, verifiable by inspection of this repository. Specifically: no field data,
one synthetic unit, Model C structure-matched to its own truth, IEC text unverified, corruption
magnitudes estimated, uncertainty band parameter-only, WTI bias tested only as a constant offset,
and the observability model simplified.

## What is NOT claimed anywhere

Recorded so that absence is deliberate rather than accidental:

- **No accuracy claim against a real transformer.** None exists.
- **No comparison against commercial winding-temperature indicators.** A "±5–15 K" figure appears
  in the legacy v1 report with **no citation anywhere** (AUDIT.md §5.2). It is the denominator of
  an attractive pitch line and it is unsourced, so it does not appear in the README, the demo, or
  any other file here. If a source is found it becomes **(a)**; if it is a field judgement it can
  appear as an explicitly-labelled **(b)** belonging to its author.
- **No novelty claim.** PINNs for transformer thermal modelling, sparse-sensor field
  reconstruction, and IEC-parameter calibration from operating data are all published work. See
  ASSESSMENT.md §2 for who did what.
- **No aging or loss-of-life calculation.** Not implemented.
