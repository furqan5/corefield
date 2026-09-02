# CoreField

**Research software for transformer thermal-parameter identification and candidate loading
envelopes. Not an operational loading authority.**

Load current, ambient temperature, top-oil temperature. From those plus a handful of hot-spot
calibration reads, CoreField identifies the four IEC 60076-7 thermal parameters
(Δθ_or, τ_o, Δθ_hr, τ_w) for a specific unit, then runs the identified model in service to
explore a future product question: **how much load could a validated model support, and for
how long?** Three external signals alone do not identify the winding parameters. Existing
hot-spot references or an independently justified calibration are needed; a new external
logger does not supply internal winding measurements.

**(c) Intended role: complement instrumented testing, not replace it.** A unit-specific
assessment might help where existing characterization is incomplete or operating conditions
have changed. This repository does not establish how many installed units lack characterization,
and an external logger cannot replace the winding references needed for identification.

---

> ## ⚠ TWO THINGS TO READ BEFORE ANYTHING ELSE
>
> **1. IEC 60076-7 provenance: MIRROR-SOURCED, UNVERIFIED against a licensed copy.**
> The 25 Aug 2026 mirror check did not close the licensed-source verification gate.
> Numerical self-consistency checks do not establish standards compliance or physical validity.
> The settled ONAF constants remain unchanged.
>
> **This repository reproduces no text, table or figure from the standard.** IEC standards are
> copyrighted and sold. If you claim standards compliance, hold your own copy from an authorised
> distributor — the constants here are used as engineering facts, not as a substitute for it.
>
> **2. Evidence is limited.** Synthetic tests, a qualified archived ODAF-data evaluation, and an
> exploratory reanalysis of a published ONAF experiment are different evidence types. None
> establishes a deployable overload rating. See [Limitations](#limitations).

---

## The result that chose the production engine

**(a, synthetic implementation test only).** Model C matches the structure of the synthetic
truth. The following table does not establish superiority on physical transformers.

Three model structures were fitted on the same ordinary day (0.6–1.2 pu), then asked about a
2-hour emergency overload at **1.30 pu** — outside the load range they were fitted over.

| Model | Structure | RMSE | Worst-case peak error at 1.30 pu | Verdict |
|---|---|---|---|---|
| A | single-exponential, K² drive | 2.59 K | **+6.17 K** | FAIL |
| B | single-exponential, free exponent | 1.77 K | **+3.17 K** | FAIL |
| **C** | **IEC two-exponential** | **0.11 K** | **+0.32 K** | **PASS** |

This is one **ONAF-configured synthetic case**, with `k21 = 2.0`. In the package's OD/ODAF
configuration, `k21 = 1.0` removes the slow gradient branch algebraically. That does not make
this table transferable to ODAF or prove that all model structures perform alike there.

![Three models fitted on an ordinary day, then asked about a 1.30 pu emergency overload](docs/day_c_extrapolation.png)

Models A and B read the hot spot several kelvin **high** at high load — triggering derating
exactly when spare capacity is worth the most. That is the commercial argument, and it is why
the production engine is classical nonlinear least squares on the IEC two-exponential structure
rather than anything more fashionable.

Two things make this table stronger than it looks. Models A and B were driven by the
**noise-free** top-oil signal while Model C had to fit a noisy one — the comparison is
handicapped in their favour and they still fail. And every value is **re-derived on every test
run** rather than quoted: `python -m pytest` reproduces the whole campaign in about 17 seconds.

## Quickstart

```bash
pip install -e .
```

```bash
python -m pytest
```

Get a blank telemetry file to hand to a site engineer:

```bash
corefield --template site_A.csv
```

Check whether a filled-in record can support a fit — before fitting anything:

```bash
corefield validate site_A.csv
```

Run the demo:

```bash
pip install -e ".[app]"
```

```bash
streamlit run app/streamlit_app.py --server.address localhost
```

`--server.address localhost` is not optional housekeeping. Streamlit binds to all interfaces by
default and prints an external URL, which on a laptop sharing a conference or client network
means anyone on it can open your demo — including whatever telemetry you last uploaded. Bind to
localhost unless you have deliberately decided otherwise.

### In Python

```python
from corefield.ingest import load_telemetry
from corefield.estimator import identify
from corefield.envelope import LoadingLimits, loading_envelope
from corefield.iec60076_7 import InitialState

frame = load_telemetry("site_A.csv")
print(frame.report.report())        # the gate: sampling, coverage, load hull, load events
frame.require_fittable()            # refuses records that cannot determine four parameters

result = identify(
    frame.time_s, frame.load_pu, frame.ambient_C, frame.top_oil_C, frame.hotspot_refs
)
print(result.report())

limits = LoadingLimits(
    hotspot_limit_C=...,            # from YOUR licensed copy of the standard
    top_oil_limit_C=...,
    label="normal cyclic loading, medium power",
    source="IEC 60076-7:2018 Table N, licensed copy, checked 2026-08-24 by AB",
)
envelope = loading_envelope(
    result.params, limits, ambient_C=30.0, duration_h=2.0,
    initial_state=InitialState(top_oil_C=70.0, prior_load_pu=0.9),
    nameplate_MVA=63.0,
)
print(envelope.summary())
```

## What the package refuses to do

These are design decisions, not gaps.

**It will not fit without ambient.** In the synthetic campaign, ignoring varying ambient
under-predicts the hot-spot peak by **3.09 K**. `load_telemetry` raises
`AmbientMissingError` rather than warning. Ambient measurement quality and sampling must
be justified for the actual site; the example's ~75-minute oil response does not prove an
hourly remote weather feed is sufficient everywhere.

**It will not supply loading limits.** `LoadingLimits` has no defaults and
`iec_loading_limits()` exists only to raise `NotImplementedError` explaining why. This is
independent of the still-unverified constants: thermal characteristics do not supply permissible
temperature limits. The limits affect candidate loading calculations and vary with loading type,
transformer category and each utility's own policy. That number must be owned by the person
relying on it. You supply the limits and a provenance string, and that string travels into every
result.

**It will not return a railed fit.** If every optimiser start converges onto a bound,
`identify` raises rather than returning the best of a bad set. A solution pinned to a bound is an
optimiser artifact that a caller cannot distinguish from a measurement. During this project's
history an independent implementation railed its optimiser in 9 of 9 runs and reported the result
as evidence of an identifiability problem in the physics; it was an implementation failure.

**A bound is not only a box bound.** `ThermalParams` requires τ_w < τ_o, and the staged residual
returns a flat penalty where that is violated — which makes the constraint an invisible wall in
the cost surface. The optimiser can walk τ_w up to that wall and stop, and until 26 Aug 2026 such
a solution was reported as converged and interior. Field data caught it: a 360 MVA ODAF unit
returned τ_w = τ_o − 1e-6 min and was accepted. `identify_staged` now detects the constraint
boundary as well as the box bounds, names the stage, and refuses.

**It will not chase a parameter the data cannot inform.** `identify_staged(..., fixed={...})`
holds a parameter at a supplied value, removes it from the optimiser, and reports it as
`HELD, NOT IDENTIFIED` rather than as a measurement. This is the intended response to the case
above: the archived-data fit did not support a reliable free τ_w estimate. Holding it at a
declared assumption is different from identifying it; sampling interval alone is not a proof
that no estimator could recover it.

**It warns about long constant channels.** A load or ambient channel pinned on one
exact value for `STUCK_CHANNEL_HOURS` (48 h) raises a warning from `load_telemetry`. Such a
channel is invisible to every other check — the row count is right, the timestamps are regular,
the value is in range. This is a screening heuristic, not proof of a failed sensor; it must not
automatically justify excluding an inconvenient validation period.

**It will not interpolate observations.** Load and ambient are model *inputs* and are
interpolated onto the integration grid. Top-oil is an *observation* and is not — interpolating it
would invent measurements and correlate their noise, making the residual RMSE flatter than the
data earns.

**It contains no neural networks.** That is a result, not a preference: see the table above, and
[WITHDRAWN.md](WITHDRAWN.md) for the geometry-PINN branch that was retracted.

## What the analysis established

**Efficiency.** On the IEC two-exponential structure, the four-parameter estimator is unbiased to
better than 0.12 % and **is consistent with the Cramér–Rao bound** — 0.97 / 1.01 / 0.95 / 0.97×
the folded-Gaussian expectation at 400 seeds. This statement is conditional on the synthetic
model, Gaussian noise and the unbiased-estimator assumptions; it is not a field-accuracy limit.

**Commissioning.** The bound is a property of the *record*, not the method, so it yields an
actionable spec. One load event leaves a ~12 % floor under τ_w that nothing can beat; two events
drop it to ~4 %. **Commission over at least two load events.** Take calibration reads at 3, 8, 18
and 48 minutes after each load change, plus one after ~4 h of steady load — the first three carry
the *rate*, the 48-minute point anchors the *amplitude*, and sampling a transient without
anchoring its own asymptote leaves the two correlated.

**Sampling rate.** Load sampling dominates; oil sampling barely matters. τ_w error is +2.1 % with
1-minute load logging and **+8.4 % with 5-minute** — the ~90-second load ramps get aliased.
Insist on 1-minute load current; 5-minute top-oil is fine.

**The quiet failure modes are the dangerous ones.** Telemetry spikes and integer-degree
quantisation cost nothing — a dense oil channel drowns them. Slow sensor drift and systematic
calibration bias are invisible on any plot and poison the *parameters* while the trajectory still
passes its gate. A winding-temperature-indicator replica reading +3 K high produces an engine
that looks perfectly calibrated against that replica while carrying +14.5 % on Δθ_hr and +4.1 K
at the true peak — and it distorts the *dynamics*, not just the level, so a "relative trends
only" positioning does not escape it. **Commissioning requires at least one bias-audited hot-spot
reference per unit.**

**Hot-spot location is not identifiable from top-oil in the implemented simplified model.**
Relocating its prescribed internal loss distribution leaves modeled top-oil unchanged.
Other reported location bounds also depend on that simplified sensing model. They do not prove
a universal impossibility for every external measurement arrangement. No spatial reconstruction
capability is validated here; see [ASSESSMENT.md](ASSESSMENT.md).

![Hot-spot location is invariant to every external measurement](docs/hotspot_location_invariance.png)

## Archived operating-data evaluation

**(b, previous internal run; not independently reproduced here).** One 360 MVA ODAF record
was used for fitting and later-period evaluation. The reported conditional result was
**1.54 K hot-spot RMSE and 1.31 K top-oil RMSE over 5,029 samples**, after excluding a
7.07-day constant-load window and holding the winding time constant fixed.

Those qualifications are part of the result:

- The excluded window is a **suspected data-quality issue, not a confirmed sensor failure**.
  The data supplier is checking it. Constant recorded load alone does not prove a failed channel.
- The previously quoted **7.91 K** came from a different fitting configuration. It is not the
  same fit scored with and without the exclusion, so it must not be presented as a clean
  before/after improvement caused solely by a fault detector.
- The record is below nameplate. A low average RMSE is not a worst-case error bound, an
  overload validation, or a guarantee across cooling stages.
- The winding time constant was **held, not identified**. Coarse sampling and poor excitation
  make estimation difficult; sampling interval alone is not a proof of non-identifiability.
- Private records remain excluded from the repository. Permission to share data, report results,
  acknowledge a contributor, and claim endorsement are separate matters.

## Published ONAF overload case: exploratory, not a validated capacity product

**(a, source)** Nordman and Lahtinen studied a purpose-designed 400/400/125 MVA **ONAF**
transformer. Their paper contains measured load-test points up to **1.60 pu** and measured
varying-load curves. It is not an ODAF fleet study.
[Original paper, DOI 10.1109/TPWRD.2002.807747](https://doi.org/10.1109/TPWRD.2002.807747).

**(b, exploratory reanalysis)** A fixed-exponent fit based on lower-load points read the
published hot spot **6.35 K low at 1.60 pu**. A later, differently specified single-winding
fit using data through 1.29 pu gave **−2.63 K** at 1.60 pu. These are case-study calculations,
not a matched demonstration of a general repair. The later model choices followed inspection
of the same small dataset; 1.60 pu is not an untouched prospective test of the final method.
Neither error is a transferable safety margin.

The interval oil power-law exponents calculated from those points vary. That motivates an
experimental load-dependent exponent, but does not establish its physical cause, prove that
this parameterization is correct, or determine the slope on any other ONAF, ONAN or ODAF unit.
Anchoring amplitudes to a nameplate observation does not make them noise-free.

**(a, implementation)** `CoolingConstants` accepts caller-supplied `x1` and `y1`, defaulting
to zero. The production four-parameter estimator does **not** automatically identify them.
`crlb.load_slope_identifiability` is a conditional steady-state precision diagnostic. Its
illustrative reference is available by default only for the unchanged ONAF example; other
constants require an explicit reference magnitude. Rank, load distribution, independent
sample count and noise all matter. No mathematical rule requires deliberately overloading a
transformer to estimate a slope.

### Corrected transient provenance

**(a, source)** The paper's **2.5-pu, 20-minute temperatures are calculated projections**.
Section V transfers response ratios derived graphically from the measured **1.6-pu**
curves to that scenario. In particular, **156 °C is not a measured 2.5-pu hot spot**.
The guides compared there are the editions cited in the 2003 paper, not a test of every
current IEC/IEEE implementation.

**(b, model comparison)** The private script's approximately 160 °C result is a comparison
with that projection under assumed parameters. The former headline that CoreField was
validated to within 4 K in a physical 2.5-pu test is **withdrawn**. One normalized response
ratio also depends on relative oil/winding contributions and other time constants; matching
it does not isolate a unique physical model.

**(a, implementation; c, interpretation limits)** The one-parameter `k21` information
diagnostic holds other parameters known. Passing it is not joint identification of
`k21`, `k22` and the winding time constant. Likewise,
`observability.detect_winding_handover` flags a local-exponent pattern; it is not a
validated detector of physical hot-spot migration. These diagnostics are not automatically
enforced by the estimator.

**Next evidence gate (c):** compare the actual measured transient against pre-specified
models, with cooling class, winding identity, initial state and digitisation uncertainty
recorded. A future field pilot stays read-only. No deliberate overload, automatic control,
or capacity promise follows from the results above.

See [validation scope and claim corrections](docs/validation_scope.md) before reusing results.

## Limitations

**Read this section before quoting anything from this repository.**

1. **One qualified archived operating-data result, below nameplate.** Its exclusion rule and
   fitting assumptions need independent review. Published ONAF data provide a separate
   exploratory case study, not replication on another operating ODAF unit. No prospective
   operational overload validation has been completed.
2. **Model C is structure-matched to the synthetic truth.** The mismatch test killed models A and
   B; it did not test C. The day-C result certifies parameter-error propagation under
   extrapolation — it does not certify structural risk on a real unit whose physics may differ
   from the standard's.
3. **One synthetic unit.** A single illustrative ONAF-scale parameter set. No unit-to-unit
   spread, no cooling-class variation beyond a constants swap, no ageing.
4. **IEC text is mirror-sourced and unverified** (see the banner above).
5. **Corruption magnitudes are plausible instrument bounds, not measured values.** Drift, spike
   rate, CT gain error and calibration bias were chosen as engineering estimates.
6. **The uncertainty band covers parameter error only.** It excludes model structural error and
   ambient/load forecast error, and it is drawn from the Cramér–Rao bound, which is a *lower*
   bound on estimator covariance. True uncertainty is larger than reported.
7. **WTI bias was tested as a constant offset.** Gain-type or load-dependent replica error is
   untested.
8. **The observability analysis in `corefield.observability` uses a simplified 1-D axial model**
   with a prescribed oil-rise profile, not CFD. Its conclusions depend on the implemented
   assumptions; it is not a universal impossibility proof for all external sensing arrangements.
   No spatial winding-hot-spot product is validated here.

## Documentation

| File | What it holds |
|---|---|
| [REPRODUCTION.md](REPRODUCTION.md) | Which published numbers reproduce, which do not, and why |
| [ASSESSMENT.md](ASSESSMENT.md) | Prior-art review, and why 2D/3D field reconstruction was not built |
| [EVIDENCE.md](EVIDENCE.md) | (a)/(b)/(c) label index for every claim in this README |
| [PREDICTIONS.md](PREDICTIONS.md) | Pre-registered prediction ledger, misses kept |
| [WITHDRAWN.md](WITHDRAWN.md) | The retracted geometry-PINN branch and its mesh-convergence failure |
| `docs/` | Figures, regenerated from the package by `python scripts/make_readme_figures.py` |

## Requirements

Python 3.11+, NumPy, SciPy, pandas. CPU-only. A 24 Aug 2026 measurement recorded about 137 MiB
resident memory for the day-C comparison plus three corruption scenarios, not every possible
workload. The project budget is 2 GB. No GPU or paid runtime service is required.
The demo adds Streamlit and Matplotlib; dependency installation requires network access.

## Licence

**Code** — **PolyForm Noncommercial License 1.0.0**, from 2 September 2026. See
[LICENSE](LICENSE). Noncommercial use — research, education, personal study, public institutions —
is expressly permitted by the licence; commercial use requires a separate licence from the
copyright holder.

**This is source-available, not open source.** PolyForm Noncommercial restricts the field of
endeavour, so it does not meet the Open Source Definition and is not OSI-approved. The source stays
readable, runnable and auditable, which is what this project's inspectability claims actually rest
on — but calling it "open source" would be wrong.

**The previous licence is not revoked.** Code released before 2 September 2026 was Apache-2.0 and
remains available under those terms, commercial use included. That grant is irrevocable and nothing
here attempts to withdraw it. The historical text is kept at
[LICENSE-Apache-2.0-historical](LICENSE-Apache-2.0-historical).

**Documentation** — Creative Commons Attribution 4.0 International, **unchanged**. See
[LICENSE-docs](LICENSE-docs). The prose files carry the evidence and the retractions, and CC BY
lets them be quoted and built on with attribution intact. That is deliberate: the claims and their
limitations should stay checkable by anyone, whatever the code licence says.

See [NOTICE](NOTICE) for the attribution notice, the split between the two licences, and two
things the licences do not cover: **no IEC standard text is redistributed here**, and
**no operational overload capability has been validated**. Loading decisions on
electrical plant carry safety and asset consequences; this software must not be the sole basis
for one.
