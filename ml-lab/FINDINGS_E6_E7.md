# E6 and E7 — can AI locate the hot spot, and what would let it?

**2 September 2026.** Run on the `ml-lab` orphan branch. Continues from where Codex stopped at E1
and E3.

**Label (b) throughout for E7:** engineering analysis on the simplified 1-D axial model in
`corefield.observability`, under its stated assumptions. Not a validated field result. No instrument
has been built or tested.

---

## 1. E6 — the direct test, and it is negative

The generator builds pairs of records whose **external channels are bit-identical** and which differ
only in the hidden hot-spot location label. Verified before use: maximum feature difference `0.0`,
maximum history difference `0.0`, labels distinct.

A probe model is then asked to predict location from those externals, across ten preregistered
seeds.

| metric | result |
|---|---|
| maximum R² over 10 seeds | **−0.018** |
| maximum relative MAE improvement | **0.16 %** |

**A negative R² means the model does worse than predicting the mean.** Not "poorly" — worse than
ignoring the input entirely.

This is not a statement about model capacity, architecture or training budget. Every external
measurement is a function of the **total** heat the winding delivers to the oil, so redistributing
that heat along the winding height changes where the hot spot is without changing how much heat
there is. Location lies in the exact null space of the external observation map. No estimator,
classical or learned, extracts information a measurement does not contain.

**So: hot-spot location cannot be a SaaS built on load, ambient and top oil.** That door is closed,
and it is closed by physics rather than by effort.

## 2. The distinction that matters commercially

Two things get conflated in "predict the hot spot":

| quantity | observable from outside? | status |
|---|---|---|
| hot-spot **temperature** | **yes** | already done — 1.55 K out-of-sample on a real unit, no new hardware |
| hot-spot **location** | **no**, exactly | needs a sensor inside the winding |

The temperature half is the existing product and it works. The location half needs an instrument.
That is the fork in the road, and the founder's own second option — *"if not SaaS then a cheap but
reliable instrument that can help AI"* — is the correct branch.

## 3. E7 — what instrument, and how cheap

### A methodological correction that changed the answer

`corefield.observability` bounds location as a **scalar** parameter: everything else, including how
hot the hot spot actually is, is assumed known exactly. The first run of E7 used it and produced a
flattering, misleading design — it placed both probes **above** the hot spot, at 0.97 and 1.00, and
claimed 0.05 % of winding height from a single probe.

A probe reading high cannot by itself distinguish **"the hot spot moved closer"** from **"the hot
spot got hotter"**. Those are confounded, and a bound that assumes the magnitude away cannot see the
confounding.

E7 therefore computes the **joint** Fisher information over `(location, delta_theta_hr)` and reports
the marginal location standard deviation with magnitude unknown — the situation any real deployment
is in. The scalar figure is kept alongside to show the size of the flattery.

### Results — joint bound, location std as % of winding height

| probes | positions | 0.1 K | 0.5 K | 2.0 K |
|---|---|---|---|---|
| 0 (external only) | — | 11.32 | **56.62** | 226.47 |
| 1 | 1.00 | 0.12 | 0.59 | 2.37 |
| **2** | **0.78, 1.00** | **0.05** | **0.27** | **1.07** |
| 3 | 0.78, 0.80, 1.00 | 0.05 | 0.23 | 0.92 |
| 4 | 0.78, 0.80, 0.97, 1.00 | 0.04 | 0.20 | 0.78 |

### Three findings

**(a) External measurement is hopeless at any practical sensor quality.** Even laboratory-grade
0.1 K external instrumentation gives 11.3 % of winding height, which does not distinguish the top
disc from the upper third. At the 0.5 K of good practical instrumentation it is 56.6 % — wider than
the winding. Buying a better external sensor does not rescue this; it only divides a vanishingly
small sensitivity into a slightly smaller noise.

**(b) Bracketing matters more than precision. This is the design result.** Two probes straddling the
hot spot at 0.78 and 1.00, with cheap 2.0 K sensing, reach **1.07 %** — better than a single
*excellent* 0.5 K probe placed above it. The joint-to-scalar ratio tells you why:

| probes | joint / scalar at 0.5 K |
|---|---|
| 1 | **1.9× worse** |
| 2 | 1.0× |
| 3, 4 | 1.0× |

One probe pays a real penalty for not knowing the magnitude. **Two bracketing probes eliminate the
confounding entirely** — they separate "moved" from "got hotter" because the two readings respond
differently. That is a specification, and it is cheap.

**(c) Beyond two probes, stop.** Three and four probes buy 0.27 → 0.23 → 0.20 % at 0.5 K. Nothing
worth the installation.

### The specification

> **Two temperature probes, one below and one above the expected hot spot, at roughly 0.78 and 1.00
> of winding height. Sensor noise up to 2 K is acceptable. That resolves hot-spot location to about
> 1 % of winding height with the magnitude treated as unknown.**

## 4. What this does and does not mean for the business

**It is a real product hypothesis.** Not "AI locates the hot spot" — that is falsified — but "two
cheap probes, placed by this rule, plus the existing estimator, deliver location *and* temperature".
The AI content is honest: the estimator does the thermal identification, and the instrument supplies
the one thing no algorithm can synthesise.

**The hard constraint is installation, not cost.** Probes inside a winding go in during manufacture.
That makes this **manufacturer-facing, not retrofit** — which is the same conclusion the loading-
envelope work reached from the opposite direction, and it points at the same counterparty. Synergy
Elektrik is a transformer manufacturer that approached this project unprompted.

**What is not established, and must not be claimed:**
- Everything here rests on a **simplified 1-D axial model**. A real winding has radial structure,
  oil ducts and non-uniform loss that this does not represent.
- The joint bound covers **two** parameters. Real deployment has more unknowns — bump width, oil
  profile, ambient — and adding them can only widen the bound.
- A CRLB is a **precision lower bound for an unbiased estimator**, not a guarantee any estimator
  achieves it, and not an upper limit on error.
- **No instrument has been built, costed or tested.** "Cheap" here means "tolerates 2 K noise",
  which is an argument that cheap parts *could* suffice — not a bill of materials.

## 5. Next, if this is pursued

1. **Widen the joint bound** to include bump width and the oil profile. If two probes stay
   identifiable there, the specification is robust; if not, it needs a third.
2. **Test the placement rule against a real two-winding record.** The SINTEF deposit
   (Zenodo 10.5281/zenodo.17223516, CC-BY) has HV and LV hot-spot probes on the same unit for a
   year, which is the closest available check on whether real probes behave as the model says.
3. **Price it.** Two probes, a datalogger and installation-during-manufacture. Until that exists,
   "cheap" is a hypothesis.
4. **Do not pitch location from external data.** E6 is unambiguous and a competent reviewer will
   reach the same result in an afternoon.
