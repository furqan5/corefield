# E11 — the eddy-current objection, and what the diligence report got right and wrong

**2 September 2026.** Branch `ml-lab`. Response to an adversarial diligence report on the
thermal-camera concept, plus the experiment its strongest technical objection demanded.

---

## 1. The report is the best of the three, and its novelty finding is favourable

Source discipline worked this time: it marks UNSOURCED honestly instead of inventing citations.

| claim | its verdict |
|---|---|
| A. IR → hot-spot **temperature** | **Occupied** — SA-GRU inversion of tank side/top temperatures |
| B. IR → hot-spot **location** | **Partly occupied** — streamline inversion tracks shifting hot spots under unbalanced load |
| C. Offset-and-scale invariance | **Appears open** |
| D. Identifiability / CRLB for location | **Appears open** |
| E. Two probes bracketing | **Appears open** |
| F. Emissivity order as identifiability limit | **Appears open** |
| G. Adjacent threats | **Occupied** — fibre, acoustic emission |

**(c) Treat "appears open" as weak evidence.** It is absence of evidence from a search that could
not retrieve much, and the report says so in its own final section. Four open claims is encouraging,
not established.

**Claim B is worse for us than "partly occupied" suggests.** Its source tracks hot-spot location
*shifting under three-phase unbalance* — which is migration detection, and migration is exactly
where §4 below lands. Read that source properly before claiming anything about migration.

## 2. It makes one factual error about the concept, and it drives its top kill

> "the instrument fundamentally provides an answer (location) to a question the utility is not
> asking, while intentionally discarding the one metric (absolute peak temperature) they require"

**The concept does not discard temperature.** The offset and scale are discarded *inside the
location inversion*, because they carry no location information. Hot-spot **temperature** comes from
the existing classical engine — load, ambient and top-oil, 1.55 K out-of-sample on a real unit — and
is unaffected. The product delivers temperature *and* location; the report has read a step of the
maths as a property of the product.

That error inflates its top kill. But the underlying question survives it, and is still the
strongest objection in the report. See §4.

## 3. Its second kill is real. I verified it independently and then tested it.

> Eddy-current heating of the tank plate bypasses the oil-column mechanism the inversion assumes.

Its own citation was a vendor blog. **The physics is nonetheless well founded**, confirmed against
the stray-loss literature: in large power transformers **more than 20 % of total load loss is stray
loss in structural components, the largest share in the tank**, and flux linking the tank near
high-current bushings overheats the wall locally. Shielding papers exist because tank hot spots are
a real design problem.

So the observation model in E8–E10 is incomplete. The wall carries `A + B·f(z) + E(z)·K²`, where
`E(z)` is generated *in the steel* and owes nothing to the oil column.

### It degrades the answer; it does not destroy it

Location std [%H], camera NETD 0.05 K, eddy shape unknown to the stated polynomial order:

| rows | order 0 | order 2 | order 3 | order 5 |
|---|---|---|---|---|
| 32 | 0.25 | 0.39 | 0.39 | 1.89 |
| 128 | 0.13 | 0.21 | 0.21 | 1.01 |
| 256 | 0.09 | 0.15 | 0.15 | 0.73 |

Even a fifth-order unknown eddy pattern leaves a usable answer.

### Measuring at several loads separates them

Eddy loss scales as **K²**; the oil rise scales as **K^1.6** with the IEC exponent. Different
exponents mean different load signatures, so surveys at several loads pull them apart:

| loads surveyed | order 3 | order 5 |
|---|---|---|
| single survey | 0.29 | 1.39 |
| 0.5, 1.0 | 0.27 | 0.80 |
| 0.3, 0.5, 0.7, 0.9, 1.1 | **0.18** | **0.51** |

## 4. The result that reframes the product

The eddy pattern is a property of the **geometry**. At equal load it is identical between two
surveys, so it **cancels exactly in the difference** — whatever its shape, however complicated,
with nothing assumed about it at all.

Standard deviation on a **change** in hot-spot location between two surveys:

| rows | migration std |
|---|---|
| 32 | 0.36 %H |
| 128 | 0.18 %H |
| 256 | **0.13 %H** |

Noise variance is doubled for two independent surveys and no eddy model is used.

**This answers the report's strongest objection on its own terms.** It is right that a control-room
operator cannot act on "the hot spot is at 88 % of winding height" — they dispatch on temperature,
and the Arrhenius ageing law does not care where the hot spot sits.

But **"the hot spot has moved 4 % since the last survey"** is a different statement. A hot spot that
migrates indicates something changed mechanically: a blocked duct, winding deformation, a cooling
path degraded. That is a maintenance signal, not a dispatch signal, and it is the class of finding
periodic thermography exists to produce.

**(c) So the honest positioning is a condition-monitoring instrument, not a control-room one.** It
belongs in the periodic survey the utility already performs, alongside the qualitative
"cooler at the bottom, warmer at the top" reading technicians already take — and it is
quantitative where that is not.

There is a second use the report did not consider: **validating that an installed fibre probe is
actually at the hot spot.** The literature's own complaint about fibre is that "because the hot spot
position cannot be determined, there is a deviation between the measured position and the hot spot
position". An external estimate of location is a check on an internal probe's placement.

## 5. Citation quality in the report — read with this in mind

- **Load-bearing claim on a vendor blog.** The eddy-current case (84 °C tank against 51 °C top oil)
  cites `industrialmonitordirect.com`. The physics is real, verified elsewhere, but that number is
  not sourced to anything checkable.
- **Camera pricing rests on shopping aggregators** — PicClick and Accio — not on manufacturer
  specifications. The 40 mK InfiRay figure needs a datasheet before use.
- **Reference 15 is irrelevant**: magnetic hot spots in superconducting Bi-2212 monoliths, cited in
  a transformer context.
- Several thermography-practice claims rest on vendor FAQ pages rather than standards.
- The $10k–20k fibre figure is explicitly unverified, and it is load-bearing for the market case.

## 6. Where this leaves the concept

**Strengthened:** eddy heating tested and survivable; migration detection is immune to it by
construction and is the strongest technical result in this line of work.

**Weakened:** absolute location probably has little operational value, which the report argues
convincingly and which no experiment here contradicts. The product is a migration detector.

**Still untested and now top of the list:**
1. **Is there a clear vertical run of tank wall?** Nobody has looked. Both the report and E9 flag it
   and neither can answer it. One photograph settles it.
2. **Do utilities value hot-spot migration?** This is now the central commercial question and it is
   an interview question, not a modelling one.
3. **Real emissivity maps and specular reflection**, rather than polynomial stand-ins.
4. **Read the streamline-inversion paper** (claim B). It tracks shifting hot-spot locations, which
   is uncomfortably close to migration detection.
