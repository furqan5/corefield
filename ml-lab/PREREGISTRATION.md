# Preregistration: falsification harness for ML versus classical NLS

**Frozen:** 1 Sep 2026, Asia/Karachi.  
**Audience:** transformer thermal-modelling researchers and the CoreField decision owner.  
**Decision:** for E2–E5, classify each method as **adopt**, **investigate further**, or **reject**, then answer whether any method beats classical NLS and exactly where.  
**Scope:** one frozen synthetic/field baseline replication followed by synthetic mechanism tests. No operational loading recommendation is in scope.

This file is write-once. After the preregistration commit, corrections belong in `RESULTS.md` as deviations; this file is not edited to match observed results.

## 1. Integrity and prior-information disclosure

- (a) The lab directory was empty and not a Git repository when audited on 1 Sep 2026. No lab model code or prior lab result existed.
- (a) The hypotheses are not blind. The sibling project already records prior PINN attempts, the A/B/C synthetic comparison, the 1.55 K internal field result, and above-nameplate analyses. E1 is replication; E2 and the classical part of E3 are confirmatory tests informed by those records.
- (a) During the read-only baseline audit, an agent reran the existing sibling E1 synthetic and private field commands before this file was created. Those executions wrote no lab or production files and reproduced already-disclosed targets. This is a sequence deviation from the requested ideal and is disclosed rather than hidden. No E2–E5 lab model has been implemented, trained, validated, or tested before this freeze.
- (c) Consequently, later language will be “confirmed”, “failed to confirm”, or “unexpected”, not “prospectively discovered”.
- (a) The four frozen CoreField source hashes are:

| File | SHA-256 |
|---|---|
| `corefield/iec60076_7.py` | `00369EDF705C1B43948F6D3EA7206CE89F34BC758C7B91F77F7D4F6B214CA994` |
| `corefield/estimator.py` | `FCEF72071DE74675EB2722721C0C094CDC9872E2F4E16425BDFE0F4C431624A2` |
| `corefield/synthetic.py` | `6333B97375DE826E6DB93E8FBC69D76984946B2F2DB1009F480C4B68D38A4BDC` |
| `corefield/crlb.py` | `662F647841702D862F52B18955EE27DD1E30833AEC1E471299FC4188F528FAD6` |

## 2. Evidence boundary and frozen physics

- (a) The authoritative code reference is sibling commit `8219c99088645b7df984752e099a3f873bae773b`. The lab will vendor an unmodified copy plus hash manifest after this freeze, never reimplement the baseline from memory.
- (a) The implemented states are top-oil temperature `theta_o` [°C], fast winding branch `h1` [K], and slow branch `h2` [K], with hot spot `theta_h = theta_o + h1 - h2` [°C]. Public thermal parameters are `delta_theta_or_K` [K], `tau_o_min` [min], `delta_theta_hr_K` [K], and `tau_w_min` [min].
- (a) The official IEC webstore verifies that IEC 60076-7:2018 covers mineral-oil-immersed transformers, operating temperatures, ageing, and loading above nameplate, and that its thermal models were revised. The public page does not verify the exact equations/constants used here. A licensed-copy check remains required before any standards-compliance claim.
- (a) Split-conformal finite-sample marginal coverage requires independent/exchangeable calibration and target observations at the record/episode level. Weighted conformal under covariate shift additionally requires unchanged `P(Y|X)`, `Q_X` absolutely continuous with respect to `P_X`, and the true density ratio for the exact theorem. Strict calibration support `K <= 0.9 pu` and target support `K > 1.0 pu` violates absolute continuity.

Primary sources fixed for interpretation:

1. IEC, *IEC 60076-7:2018*, official scope page: https://webstore.iec.ch/en/publication/34351
2. Lei et al., *Distribution-Free Predictive Inference for Regression*, JASA 113(523), 2018: https://doi.org/10.1080/01621459.2017.1307116
3. Tibshirani et al., *Conformal Prediction Under Covariate Shift*, NeurIPS 2019: https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html
4. Nordman and Lahtinen, *Thermal Overload Tests on a 400-MVA Power Transformer With a Special 2.5-p.u. Short Time Loading Capability*, IEEE TPWRD 18(1), 2003: https://doi.org/10.1109/TPWRD.2002.807747

The project-derived `-6.35 K` overload error and the transcribed `0.65 pu` point were not independently recoverable from accessible primary text. They remain internal verification claims, not literature quotations.

## 3. Universal design rules

### 3.1 Test access and run order

1. Freeze and commit this file.
2. Implement harness and its unit tests without reading primary test results.
3. Run E1. If its gate fails, stop; do not run E2–E6.
4. If E1 passes, run E3 first, then E2, E4, E5. E6 runs only under its trigger in §9.
5. Each primary test command writes a hash-stamped access sentinel before loading test truth and refuses a second access. A rerun is permitted only for a documented infrastructure failure or the reserved positive-result confirmation, never to choose a model.

No hyperparameter, tolerance, seed, feature, loss weight, data split, or metric is changed after seeing validation or test outcomes. A losing method is not tuned. An apparent winner receives only the reserved confirmation specified here.

### 3.2 Compute and reproducibility

- (a) CPU only. `CUDA_VISIBLE_DEVICES=-1`; PyTorch device must equal `cpu`; one intra-op thread by default; float64 for physics/NLS and float32 for neural weights unless a numerical test fails before test access.
- (a) Peak process-tree resident memory must be `< 2,000,000,000` bytes. Runs record wall time, peak RSS, Python/package versions, CPU thread count, Git commit, CoreField hashes, and seeds.
- (a) At least 10 seeds per reported stochastic cell. Report mean, sample standard deviation, median, minimum, and maximum. Never report a single best seed.
- (a) Primary seeds: E1 `1000..1009` as frozen upstream; E3/E4 `31000..31009`; E2 `41000..41009`; E5 episode generator `61000`; E6 `71000..71009`. Reserved positive-result confirmation seeds are `51000..51009` and are never used otherwise.
- (c) PyTorch CPU is allowed as the single extra ML dependency because it supplies automatic differentiation for the ODE residual. `openpyxl` is allowed only for the private E1 adapter. No other new dependency is added unless execution is impossible; such a need stops the run and is reported as a deviation.

### 3.3 Common synthetic unit, signals, and noise

- (b) The synthetic mechanism-test unit uses the frozen CoreField nominal parameters `[45 K, 150 min, 22 K, 7 min]`, loss ratio `R=6`, ONAF constants `x=0.8`, `y=1.3`, `k11=0.5`, `k21=2`, `k22=2`, and 20 °C mean ambient. These are test settings, not a claim about an installed transformer.
- (a) Truth is generated on a 30 s RK4 grid. Model inputs use load [pu], ambient [°C], and top oil [°C]; top oil is sampled at 5 min with independent Gaussian noise `sigma=0.5 K`. Neural inputs use linear interpolation of those noisy 5 min samples; they do not receive dense truth top oil.
- (a) Sparse hot-spot references carry independent Gaussian noise `sigma=0.5 K`. Test scores use noise-free hidden truth, never noisy target observations.
- (b) For E3–E5 structural-mismatch truth only, the oil exponent is `x(K)=0.8+0.21(K-1)` per pu and `y1=0`. The `0.21/pu` setting is the project’s declared engineering stress-test estimate from the published 400 MVA case, not a directly measured universal parameter. Every fitted physics model retains fixed `x=0.8`; no method is told the hidden slope.
- (a) E2 uses structure-matched truth (`x1=y1=0`) to isolate scarce-reference identification from extrapolation mismatch.

### 3.4 Fixed train, validation, and test schedules

- (b) Train record: 48 h, loads confined to `0.60–0.95 pu`, with deterministic smoothed events every 6 h and alternating amplitudes; ambient is a ±6 K diurnal profile. All parameter fitting and neural gradient updates use this record only.
- (b) Validation record: a different 24 h event schedule confined to `0.65–0.92 pu`. It selects the neural early-stopping epoch only; no architecture or loss weight is selected.
- (b) In-range test record for E2/E4: a third 24 h schedule confined to `0.62–0.94 pu`, never used for fitting or early stopping.
- (b) E3 test records: four separate step episodes, each beginning with 4 h at `0.75 pu` followed by 4 h at exactly `1.00`, `1.15`, `1.30`, or `1.60 pu`, with continuous initial thermal state. These records are never used in training or validation.
- (a) The convex training-load hull is the scalar interval `[min(K_train), max(K_train)]`; no synthetic fleet is claimed. Ten seeds vary measurement noise and neural initialization on this one declared synthetic unit.

### 3.5 Reference placement

For E2 budgets `N in {3,4,6,10,20,50}`, a deterministic candidate list is constructed before noise: for each train-record event, offsets `{3,8,18,48} min`, followed by evenly spaced quasi-steady anchors. Candidates are ordered round-robin across events and offsets, then anchors; the first `N` unique indices are used. This keeps smaller budgets nested inside larger ones and prevents selecting convenient readings after seeing results. E3 uses `N=20`.

## 4. Frozen methods

### 4.1 Classical NLS

- (a) Use frozen `corefield.estimator.identify`, `loss="linear"`, analytic half-step arrays, the upstream campaign start for E1, and the upstream four-start default for E2–E5. Cooling constants and `R` are fixed.
- (a) A railed/non-converged fit or fewer than four hot-spot references is a refusal. It is reported as a refusal, never converted into a numeric score or silently rescued with generic winding parameters.

### 4.2 Plain neural network

- (b) Inputs at 2 min intervals: current load plus load lags at 6, 16, 60, and 180 min; current ambient; current interpolated top oil plus top-oil lags at 16 and 60 min. All features are standardized using train statistics only.
- (b) Architecture: fully connected `9 -> 16 -> 16 -> 1`, `tanh` activations, predicting hot-spot temperature [°C]. Adam, learning rate `1e-3`, weight decay `1e-6`, full-batch, maximum 2,000 epochs, early-stopping patience 150, minimum normalized validation improvement `1e-4`. Loss is squared error on available hot-spot references only.
- (a) The plain NN exposes no physical parameters; parameter recovery is `not applicable`, not inferred from its weights.

### 4.3 Physics-informed neural network (PINN)

- (b) The same nine inputs and two `tanh` hidden layers of width 16 feed three outputs: `theta_o`, `h1`, `h2`. Hot spot is `theta_o+h1-h2`.
- (b) The four thermal parameters are trainable through sigmoid maps into the frozen CoreField bounds. The loss is the unweighted mean of five dimensionless terms: measured top-oil error divided by `0.5 K`; sparse hot-spot error divided by `0.5 K`; and the three centered-finite-difference ODE residuals multiplied by their current candidate time constants and divided by the corresponding nominal drive scale. Opening equilibrium residuals are included. Equal weighting is fixed; it is not tuned.
- (b) Optimizer, epoch cap, stopping rule, feature standardization, and initialization seeds match the plain NN. The PINN trains on the train record only and uses labelled validation references only for early stopping.
- (c) This is a discrete physics-informed state network, not a claim of a universal neural operator. It must generalize to the frozen schedules without test-time retraining.

### 4.4 Grey-box/hull-aware residual

- (b) Start with the NLS trajectory. Fit a `9 -> 8 -> 1` `tanh` residual network to sparse training-reference residuals using the same optimizer/stopping rule.
- (a) The learned residual is multiplied by `1{K is inside the scalar training-load hull}` and is exactly `0.0 K` outside. Every outside-hull prediction sets an extrapolation flag.

### 4.5 Generic IEC comparator

- (b) Run the frozen nominal parameters and fixed ONAF constants with no identification. This is a generic synthetic comparator only; it is not presented as an IEC-compliance value for a real unit.

## 5. E1 — reproduction gate

### Prediction

- (a) Known replication target on day C over seeds 1000–1009: mean RMSE A/B/C `2.59/1.77/0.11 K`; largest signed peak `+6.17/+3.17/+0.32 K` in the current environment. The legacy Model-A target `+6.18 K` is an acknowledged SciPy-version discrepancy.
- (a) Known internal field target: hot-spot RMSE `1.55 K` and top-oil RMSE `1.34 K` over the authorized 42 d held-out segment after the existing exclusion. The field fit is staged OD, fixes `tau_w=7 min`, uses `R=6.8`, and is not the same estimator configuration as synthetic Model C.

### Gate

- (a) Synthetic cells must lie within `max(0.05*abs(target), 0.01 K)` of the targets. Field aggregate RMSE values must be within `0.02 K` when the private file and reader are available. Missing private access is `not run`, not a fabricated pass.
- (a) The field adapter emits aggregate metrics only and describes the exclusion as authorized for suspected monitoring trouble, not as a confirmed sensor fault.
- (a) Failure stops all later experiments.

## 6. E3 — above-nameplate extrapolation (run first after E1)

### Prediction

- (c) NLS degrades gradually as hidden exponent mismatch grows; the plain NN is unstable and may read dangerously low; the PINN lies between them but loses to NLS; the hull-aware grey-box equals NLS outside the training hull; the generic comparator may beat identification at `1.60 pu` despite worse fit elsewhere.

### Scores and ranking

At each target load and for every method report over 10 seeds:

- mean absolute trajectory error [K];
- RMSE [K];
- mean signed trajectory error `prediction-truth` [K];
- signed peak error `max(prediction)-max(truth)` [K];
- most-negative per-seed signed peak error [K] (the unsafe worst case);
- fraction of seeds with signed peak error `< 0 K`.

Accuracy rank is increasing RMSE. Safety rank is lexicographic: zero unsafe-low seeds first; then the larger (less negative) worst signed peak; then the larger mean signed peak; then lower RMSE. Both rankings are reported.

### Win rule and confirmation

A method beats NLS at a load only if its paired mean RMSE is lower, its most-negative signed peak is no worse than NLS by more than `0.10 K`, and a two-sided 95% bootstrap confidence interval (10,000 fixed resamples) for paired RMSE difference excludes zero in its favour. The method must meet this at `1.30` or `1.60 pu`; an in-range-only win does not answer the project question.

If a neural method satisfies the rule, do not tune it. Rerun the same protocol on reserved seeds 51000–51009 at hidden widths 8 and 32. The result is believable only if both sizes preserve the safety condition and at least one preserves the paired RMSE win. Otherwise classify it as `investigate further`, not adopt.

## 7. E2 — scarce-reference identification

### Prediction

- (c) NLS refuses at `N=3`; the PINN may return a trajectory but its parameters remain weakly identified. For every `N>=4`, NLS is expected to equal or beat the PINN in parameter recovery and in-range test RMSE. The plain NN is expected to overfit at small N.

### Scores

- (a) Availability/refusal rate.
- (a) For NLS and PINN: signed and absolute percent error for each of the four parameters, plus median absolute percent error across parameters.
- (a) For all methods with a prediction: held-out in-range RMSE [K] and signed peak error [K]. Plain-NN parameter recovery is `N/A`.

At `N=3`, a finite PINN output is not called an accuracy win over a refusing NLS; it is `investigate further` only if all 10 seeds are finite/interior, held-out RMSE is below `2.0 K`, and the reserved-size confirmation succeeds. At `N>=4`, a PINN win requires the E3 paired-RMSE/bootstrap rule and no worse worst signed peak. If PINN never wins at any `N>=4`, physics-informed identification is rejected for this problem under this protocol.

## 8. E4 — hull-aware learned residual

### Prediction and rule

- (c) Hard safety gating is expected to collapse out-of-range predictions exactly to NLS. It may still improve in-range residuals.
- (a) Outside the load hull, grey-box and NLS predictions must agree bit-for-bit (`0.0 K` residual) and the extrapolation flag must be true; any violation fails the method.
- (b) Adopt for in-range use only if paired mean in-range RMSE improves by at least `0.10 K` and the 95% paired bootstrap interval excludes zero, while the exact outside-hull invariant holds. A smaller/uncertain improvement is decoration and is rejected.

## 9. E5 — conformal upper bounds on NLS

### Episode-level split

- (b) Fit NLS once on the E3 train record. An independent episode is 4 h at `0.75 pu` followed by 4 h at a sampled target load; its scalar outcome is the maximum true hot-spot temperature [°C], and the fixed NLS predictor supplies the predicted maximum.
- (a) Calibration/test samples are independent generated episodes, not correlated time points. Calibration has 200 episodes; each test condition has 1,000 episodes. The primary output is the one-sided 95% upper bound `predicted peak + q`, where conformity score is `true peak-predicted peak` [K] and the finite-sample higher quantile uses the standard `ceil((n+1)*(1-alpha))` rank.

### Cases and predictions

1. **Exchangeable in-range:** calibration and test target loads are independently Uniform(`0.60,0.90 pu`). (c) Empirical coverage should be compatible with 95%; the exact binomial 95% confidence interval must contain 0.95.
2. **Strict shift:** reuse calibration below `0.90 pu`; test in four non-overlapping 0.05 pu bands centred on `1.00`, `1.15`, `1.30`, `1.60 pu`. (c) Ordinary conformal is expected to under-cover, increasingly with load. This is a demonstrated failure, not a bug.
3. **Weighted, overlapping shift diagnostic:** test loads follow a fixed Beta(5,2) distribution mapped to `0.60–0.90 pu`. Estimate the density ratio using Gaussian KDEs with Scott bandwidth from calibration loads and 2,000 unlabeled target loads; compare with the known synthetic ratio but use the estimate for intervals. Report effective sample size, coverage, and finite upper width.
4. **Weighted, strict support mismatch:** if a query load is outside the calibration hull or the estimated calibration density is numerically zero, assign the target point all mass at `+infinity` and return an unbounded upper limit. Do not clip weights into a finite pseudo-guarantee. (c) Expected result: 100% formal containment only through useless unbounded intervals and 0% finite-interval availability.

Exact coverage is claimed only for the exchangeable case and for oracle weighting under its assumptions. KDE-weighted results are empirical because estimated density ratios do not inherit the exact finite-sample theorem automatically.

## 10. E6 — self-deception/null-space detector

Trigger E6 for any model claimed to infer internal spatial information, or before calling an unexpected neural win physically meaningful. Generate paired examples with identical external load, ambient, top-oil, and total winding-loss histories but independently permuted synthetic location labels in `[0,1]` winding height. No external feature is permitted to encode location.

- (a) The primary check is equality of every paired external feature to machine precision.
- (c) A fixed location probe with the plain-NN architecture must have held-out `R^2 <= 0.02` over all 10 seeds and must not beat the constant-median MAE by more than 2%. Any apparent success fails the generator/model as leakage; it is never reported as physical inference.

E6 does not claim to reconstruct a field and is not used to tune E2–E5.

## 11. Decision rules and stopping

- **Adopt:** preregistered primary rule passes, safety does not degrade, reserved confirmation passes when required, resource gate passes, and no applicable E6 leakage appears.
- **Investigate further:** primary evidence is positive but confirmation, identifiability, support, or external validity remains insufficient.
- **Reject:** primary rule fails, unsafe bias worsens, required invariant fails, or the method only wins after an unregistered choice.

Stop gathering evidence when all required cells have at least 10 seeds, consequential claims have direct run artefacts, discrepancies are bounded, and another allowed run cannot change the decision. Do not run architecture search, extra seeds, alternative loss weights, or a synthetic fleet to rescue a loss.

## 12. Required outputs

- `RESULTS.md`: predicted versus observed outcome for every run, all failures, mean/spread tables, deviations, environment, wall time, peak RSS, and what would have to be true for each conclusion to be wrong.
- `VERDICT.md`: one page; E2–E5 adopt/investigate/reject and the direct NLS-versus-ML answer.
- `report-source.md`: canonical internal evidence-backed report with claim-to-source ledger.
- Reproducible single entry point, write-once run manifests, machine-readable aggregate tables, and pytest tests for units, split isolation, sign conventions, seeds, resource gate, hull zeroing, conformal quantiles, and E6 pair equality.

## 13. Independent-verification handoff

Before any client, publication, standards, or loading-decision use, independently verify: (1) exact IEC equations/constants against a licensed IEC 60076-7:2018 copy; (2) permission to report the private field aggregates; (3) the transcribed published overload points against the paper; (4) every generated aggregate from raw run manifests; (5) CPU/RAM measurements on the stated Dell Latitude 5591 rather than only the current execution host; and (6) conformal assumptions at the intended deployment distribution.
