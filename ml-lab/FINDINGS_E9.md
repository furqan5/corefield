# E9 — the instrument is probably a thermal camera, and radial structure does not kill it

**2 September 2026.** Branch `ml-lab`. **Label (b)**: engineering analysis on a simplified model
with an assumed radial-smearing law. Not CFD, not a validated field result, no instrument built.

---

## 1. Absolute accuracy is irrelevant. This changes what to buy.

Published practice measures tank-surface temperature with thermal-infrared cameras at about
**±2 K absolute** (Chen et al., *IET Sci. Meas. Technol.*, 2024), and the standard criticism is that
the reading is susceptible to external conditions. Against E8's ≤0.25 K requirement that looks
disqualifying by a factor of eight.

**It isn't, and the reason is structural.** E8 fits an offset `A` and a scale `B` and discards them.
A constant absolute error *is* an offset. A gain or emissivity error *is* a scale. Neither survives
the fit. Verified rather than asserted:

| injected error | axial location bound |
|---|---|
| none (baseline) | 0.39 %H |
| **+2.0 K absolute bias** | **0.39 %H** |
| **−5.0 K absolute bias** | **0.39 %H** |
| gain ×1.15 | 0.34 %H |
| gain ×0.85, bias +2 K | 0.46 %H |

Bias changes the answer **not at all**. Gain moves it only insofar as it changes the real
temperature span being read.

**Consequence: specify the instrument by NETD, not by absolute accuracy.** An uncooled
microbolometer routinely achieves 0.05 K noise-equivalent temperature difference, an order of
magnitude inside what E8 needs — while the ±2 K absolute figure that vendors quote and critics cite
has no bearing on locating the hot spot.

That reframes the hardware entirely. Not an array of contact sensors needing mounting, coupling and
calibration: **a camera.** Non-contact, portable, hundreds of heights in one frame, no thermal
contact to worry about, and the coupling problem that dominated E8 disappears because there is no
contact to be uncertain about.

## 2. Radial structure costs a factor, not the result

A real winding is not a line. The CFD literature reports that duct geometry materially moves the
hot spot, and that 2-D axisymmetric results represent 3-D when Reynolds, Richardson and Prandtl
match. So: does the **radial** position of the loss create an axial signature that confounds with
its **axial** position?

Carrying radial position as a second unknown, offset and scale still fitted:

| NETD | heights | axial, radial known | **axial, radial unknown** | cost | radial |
|---|---|---|---|---|---|
| 0.05 K | 16 | 0.39 %H | **1.00 %H** | 2.5× | 27 %R |
| 0.05 K | 32 | 0.28 %H | **0.71 %H** | 2.5× | 19 %R |
| 0.05 K | 64 | 0.20 %H | **0.51 %H** | 2.5× | 14 %R |
| 0.10 K | 32 | 0.57 %H | 1.43 %H | 2.5× | 39 %R |
| 0.25 K | 32 | 1.42 %H | 3.56 %H | 2.5× | 97 %R |

**Not knowing the radial position costs a consistent 2.5×, and that is affordable.** At 0.05 K NETD
with 32 heights — trivial for a camera, which has hundreds of rows — axial location lands at
**0.71 %H even with radial position unknown**. That beats the two internal probes from E7, which
gave 1.07 %H.

Radial position itself comes out poorly, 14–27 % of radius. That is the right trade: you learn
**where up the winding** the hot spot is, which is what governs ageing, and you learn little about
how far out it sits, which matters less.

## 3. The conclusion does not rest on the radial assumption

The smearing law is mine, so the conclusion has to survive it being wrong. Sweeping it over a 20×
range at 32 heights and 0.05 K:

| smear | axial, radial known | axial, radial unknown | cost |
|---|---|---|---|
| 0.05 | 0.26 %H | 0.71 %H | 2.7× |
| 0.25 | 0.28 %H | 0.71 %H | 2.5× |
| 1.00 | 0.40 %H | 0.71 %H | 1.8× |

The radial-unknown bound is **0.71 %H across the whole sweep**. The conclusion is about the
geometry of the problem, not about the number I chose.

## 4. The specification, revised

> **An uncooled thermal camera, specified by NETD ≤0.1 K, pointed at a clear vertical run of tank
> wall. Absolute accuracy, emissivity calibration and ambient compensation do not matter — they are
> offset and scale, and the fit removes them. Sample 32 or more heights from the image. Expect
> hot-spot axial location to roughly 0.7–1.4 % of winding height with radial position unknown.**

Against the alternatives, all with nuisances unfitted:

| instrument | axial location | retrofit? |
|---|---|---|
| 2 internal fibre probes, 2.0 K | 1.07 %H | **no** — factory only |
| **thermal camera, 0.05 K NETD, 32 rows** | **0.71 %H** | **yes** |
| thermal camera, 0.1 K NETD, 32 rows | 1.43 %H | yes |
| 16 contact sensors, 0.25 K | 1.78 %H | yes, with mounting |
| the 4-channel external bundle, 0.5 K | 56.62 %H | — |

## 5. What is still not established

- **The radial model is a smearing assumption, not CFD.** It was built to answer an identifiability
  question. A real 2-D axisymmetric solve, which the literature shows is standard and adequate,
  would replace it. That is the next serious piece of work and it does not fit on this laptop.
- **Emissivity varying with height** — dirt, paint wear, rust — acts like E8's coupling tilt rather
  than a clean scale. E8 showed 8+ sensors absorb a linear tilt at 1.1×; a camera has hundreds of
  rows, so this is likely fine, but it has not been tested here.
- **Sun and shadow across the tank** is the same class of problem and the same argument, untested.
- **A clear vertical run of tank wall may not exist** on a radiator-covered unit. Survey question.
- **The oil-profile shape is assumed fixed.** The SINTEF check (see `private/sintef/`) could not test
  that directly — the record's intermediate stations are winding sensors, not oil — but it did find
  a residual ambient dependence in the winding profile after controlling for load, which the model
  does not explain. If the oil profile shape moves with oil temperature, that is another nuisance.
- **A CRLB is a precision floor for an unbiased estimator**, not a guarantee.
- **No instrument has been built, costed or tested.**

## 6. Next

1. **Buy or borrow a thermal camera and photograph an energised transformer.** One image tests more
   of this than another week of modelling: whether a clear vertical run exists, what the profile
   actually looks like, and whether pixel noise behaves like the NETD figure outdoors.
2. **Build the 2-D axisymmetric model properly** — the literature route is COMSOL or Fluent, neither
   of which runs here. A reduced-order 2-D solve might.
3. **Test emissivity tilt and shadowing** as nuisances, the same way E8 tested coupling tilt.
