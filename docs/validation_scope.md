# Validation scope and claim corrections

Updated 28 August 2026. Labels: **(a)** sourced fact or explicitly scoped numerical fact;
**(b)** engineering estimate or assumption-dependent calculation; **(c)** judgement or plan.

## What the evidence actually supports

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Synthetic tests, with Model C matched to the truth structure | (a) Numerical behaviour under the tested assumptions | Superiority on real transformers or robust extrapolation under structural mismatch |
| Private archived ODAF operating records | (b) A conditional retrospective result for one analyzed unit | Fleet-wide accuracy, overload capability, confirmed sensor failure, or permission to publish the data |
| Nordman and Lahtinen's ONAF experiment | (a) Published temperatures and load histories for a specifically designed unit | ODAF behaviour, a typical fleet's overload limit, or a general exponent-slope value |
| Their 2.5-pu scenario | (a) A published calculation using ratios from 1.6-pu measurements | A measured 2.5-pu validation target |
| CoreField's comparison with that scenario | (b) An assumption-dependent model comparison | A demonstrated 4 K physical-test error bound, safe capacity, or jointly identified dynamic parameters |

The original source is H. Nordman and M. Lahtinen, *Thermal Overload Tests on a
400-MVA Power Transformer With a Special 2.5-p.u. Short Time Loading Capability*,
IEEE Transactions on Power Delivery 18(1), 107–112 (2003).
[Publisher DOI](https://doi.org/10.1109/TPWRD.2002.807747).
Cooling class appears on p. 107; the 2.5-pu construction and its assumptions are on
pp. 111–112. The guides compared in that paper are its cited historical editions.

## Corrections that supersede earlier summaries

1. **(a, source)** The Nordman unit is ONAF, not ODAF. Do not transfer fitted slopes,
   error magnitudes, or inferred mechanisms between classes or between units without evidence.
2. **(a, arithmetic)** 1.60 pu means 160% of rated current, or 60% above rated current.
   It is an exploratory stress-test point, not a proposed operating instruction.
3. **(a, provenance)** The paper's 156 °C value at 2.5 pu is projected, not measured there.
   Agreement with it does not close measured-transient validation.
4. **(b, reanalysis)** Interval power-law exponents computed from a few temperatures are
   not direct observations of the load-slope parameter in CoreField's loss-ratio model.
   The physical cause and cross-class magnitude are unresolved.
5. **(b, reanalysis)** The 1.54 K archived-data score excluded a constant-load window and
   held the winding time constant fixed. The earlier 7.91 K score used a different fit.
   Do not attribute the entire difference to a detector or call the suspected fault confirmed.
6. **(a, implementation)** The production estimator fits four thermal parameters.
   Caller-supplied exponent slopes and standalone information diagnostics are additional
   capabilities, not automatic six-parameter estimation or an end-to-end refusal system.
7. **(a, algebra)** Distinct load levels, independent sample count, noise and parameter
   correlations all matter. A narrow but full-rank IID design improves with repetition;
   a rank-deficient design does not gain missing parameter directions by repetition.
8. **(a, model structure)** A one-parameter k21 bound assumes all nuisance parameters known.
   It cannot establish joint identification of k21, k22 and the winding time constant.
9. **(c, interpretation)** The current winding-handover heuristic is an anomaly flag, not
   a verified physical classifier. A maximum of smooth winding curves need not have its
   particular local-exponent pattern.
10. **(a, project provenance)** IEC text remains mirror-sourced and UNVERIFIED against a
    licensed copy. Numerical consistency checks do not close that source-verification gate.

## What changed in the software in this review

- **(a)** The oil-slope diagnostic requires an explicit reference magnitude for constants
  other than the unchanged ONAF example. Even the example magnitude is not a measured universal prior.
- **(a)** SVD is used to check information rank and compute the conditional covariance.
  Invalid references, tolerances, array shapes and non-finite sensitivities are rejected.
- **(a)** Envelope warnings no longer prescribe carrying one ONAF case's error as a safety
  margin on other units. Adding a slope does not validate a loading calculation.
- **(a)** The settled ONAF constants and the underlying thermal integration equations were
  not changed. No neural network was added.

## Next technical work, before an operational loading claim

All items here are **(c), proposed validation work**, not completed capabilities.

1. **Measured transients:** obtain raw curves, or carefully digitise the measured varying-load
   trace. Record axis calibration, read-off uncertainty, actual load history, initial thermal
   state, cooling stage and winding identity. Pre-specify comparison models and the held-out
   interval. Score early-time under-prediction and peak error as well as average RMSE.
2. **Winding slope:** use informative, independent observations with stated noise and
   covariance. Report joint parameter intervals and held-out predictions. A generic requirement
   for six, eight or ten load levels is not a substitute for a design-specific information analysis.
3. **Moving hot spot:** retain separate winding/sensor identities. Do not fit a smooth
   single-winding relationship to an undocumented maximum-of-sensors channel. Validate any
   anomaly/refusal rule on cases it was not designed around.
4. **Dynamic parameters:** study joint information for k21, k22 and the time constants.
   In the implemented OD case with k21 fixed to 1, only the product k22·τ_w enters the
   remaining fast winding time scale; those two parameters cannot be separated from that
   response alone. An independent constraint is needed.
5. **Field chain:** check reference calibration, current scaling, clock alignment, cooling
   state and missing-data handling before interpreting thermal residuals as faults.

## Boundary for a pilot without existing telemetry

**(a, model structure)** Load, ambient and top-oil observations can support oil-model work.
They do not directly observe internal winding temperature. Tank-wall temperature is a different
measurement and must not be silently relabelled top-oil; a WTI replica is not an independent
fibre-optic hot-spot measurement.

**(c, safety policy)** Start with passive data collection and shadow analysis within the
operator's existing approved operating conditions. Installation is for qualified personnel
under the host's procedures. No protection bypass, deliberate overload or automatic control
is authorized by this repository.

**(c, commercial boundary)** A measurement-quality and thermal-assessment service is a
plausible first offer. Repeatable installation, independent validation, paying demand and
positive delivery economics must be demonstrated before calling it a scalable capacity product.
