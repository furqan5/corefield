EXPERIMENTAL. Not production. Nothing here is validated for loading decisions.

# CoreField ML falsification lab

This is a quarantined experimental repository. Its only decision question is whether a machine-learning method can credibly outperform the frozen classical nonlinear least-squares (NLS) estimator for transformer winding hot-spot estimation under the actual data, extrapolation, compute, and confidentiality constraints.

Negative results are the primary deliverable. A method is not tuned after a loss. A positive result is not accepted until it survives reserved seeds, fixed alternative network sizes, safety-sign checks, and the applicable null-space/self-deception test.

## Quarantine

- (a) This directory is a standalone Git repository and is not a branch of the production project.
- (a) The sibling `CoreField Startup` repository is read-only input. This lab never writes to it and nothing from this lab is imported back into it.
- (a) The private field workbook is read in place only when its owner-provided local path is available. Raw data, rows, excerpts, derived time series, and identifying metadata are never copied into this repository. Only aggregate internal verification metrics may be emitted.
- (c) Any later promotion is a human evidence decision. A successful experiment produces a report and reproducible lab code, never an automatic production patch.

## Hard runtime limits

- (a) Python 3.11 or later; CPU execution only; no paid service or remote compute.
- (a) Peak resident memory must remain below 2 GB. The runner records peak RSS and fails the resource gate at or above that limit. E2–E5 and confirmation are single-process; BLAS, NumExpr, OpenMP, and Torch intra-op execution are fixed to one thread. The already-run private E1 adapter used a child process, whose separate peak was not captured; that limitation is retained for the final report rather than inferred away.
- (a) All stochastic runs record their seeds. Primary test artefacts are write-once and a rerun requires an explicit, logged override.
- (a) SI conventions: time in seconds or labelled minutes, load in per unit, absolute temperature in degrees Celsius, and temperature differences in kelvin.

## Dependencies

The base harness uses NumPy, SciPy, pandas, Matplotlib, pytest, and the frozen CoreField reference. `openpyxl` is an optional, lab-only dependency required solely for the private E1 workbook adapter.

PyTorch `2.14.0+cpu` is the only ML addition in the isolated lab environment. (c) It is justified because automatic differentiation is needed to train the preregistered three-state physics-informed residual without hand-maintaining a second derivative engine; the same small implementation also supplies the plain neural-net and grey-box comparators. CUDA is disabled in code and no vision/audio packages are needed. Scikit-learn, JAX, probabilistic-programming libraries, architecture-search packages, and paid APIs are excluded.

## Reproduction entry point

Use the repository's isolated interpreter; every command is CPU-guarded:

```powershell
.\.venv\Scripts\python.exe -m corefield_ml_lab preflight
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m corefield_ml_lab e1 --private-script <path> --private-workbook <path>
.\.venv\Scripts\python.exe -m corefield_ml_lab e3
.\.venv\Scripts\python.exe -m corefield_ml_lab e2
.\.venv\Scripts\python.exe -m corefield_ml_lab e4
.\.venv\Scripts\python.exe -m corefield_ml_lab e5
```

Primary commands enforce the E1 → E3 → E2 → E4 → E5 evidence order, require a clean committed worktree, and create a configuration-hashed, write-once access sentinel before loading test truth. A second access is refused unless an infrastructure-only override and reason are explicitly recorded. A triggered E2/E3 positive result is confirmed only through the `confirmation` subcommand shown by `--help`, using reserved seeds and widths after verifying the exact originating aggregate. Private field paths and raw subprocess output are never written to the lab.

The exact protocol, prior-information disclosure, frozen seeds, model sizes, metrics, gates, and stopping rules are in [PREREGISTRATION.md](PREREGISTRATION.md). That file is frozen before lab implementation or E2–E5 execution.

## Evidence status

- (a) Frozen implementation baseline: sibling Git commit `8219c99088645b7df984752e099a3f873bae773b` on 1 Sep 2026.
- (a) The official IEC page verifies the scope of IEC 60076-7:2018 but does not expose the detailed equations or constants. CoreField itself marks its exact IEC provenance as mirror-sourced and unverified against a licensed copy. This lab therefore tests the frozen implementation; it does not claim standards compliance.
- (a) The private 1.55 K field score is an internal reproduction target and lacks external reporting permission. It must not be published from this repository.
