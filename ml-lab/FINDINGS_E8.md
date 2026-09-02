# E8 — a cheap external array. The answer is yes, with conditions.

**2 September 2026.** Branch `ml-lab`. **Label (b)** throughout: simplified 1-D axial model plus an
assumed linear wall-coupling model. Not a validated field result. No instrument built or tested.

---

## 1. Why E7's "external is useless" was too strong

E7 reported the external channels at 56.6 % of winding height and I framed that as external sensing
being closed. That framing was wrong, and the reason is specific.

`AxialWindingModel.external_observations` collapses the entire tank to **one scalar** — the
mixed-oil mean, `trapezoid(oil, z)`. But the model's own docstring says exactly where the signal is:

> "Oil heats cumulatively as it rises past the losses, so the profile SHAPE depends on where those
> losses sit — while its endpoints do not."

**A mean destroys shape. An array samples it.** That is a different observable and it had never been
tested here.

Measured raw signal, for a 10 % shift in hot-spot location:

| | change |
|---|---|
| largest, at height 0.88 | **0.875 K** |
| at the top, z = 1.00 | **+0.0000 K** |
| at the bottom, z = 0.00 | **+0.0000 K** |

The endpoints are exactly invariant — that is the null space, and it is why top-oil and bottom-oil
carry nothing. **The middle of the tank moves by nearly a kelvin.** That is a large signal against
cheap sensing, and no instrument in this project has ever looked at it.

## 2. Two analysis errors, both of which said "impossible"

Worth recording because both produced a confident wrong answer.

**First:** I carried `delta_theta_hr` as the nuisance, copying E7. But `oil_profile` does not depend
on the rated gradient at all — only `winding_profile` does. That column was exactly zero, the
information matrix was singular, and every configuration came back `inf`. A parameter with no effect
on the observable is *absent*, not confounded.

**Second:** I then carried oil endpoints and wall coupling as three separate nuisances. Writing the
observation out shows the problem:

```
obs = c*[bot + (top-bot)*f(z;loc)] + (1-c)*amb
    = [c*bot + (1-c)*amb]  +  [c*(top-bot)] * f(z;loc)
    =         A            +        B       * f(z;loc)
```

Bottom oil, top oil and coupling enter **only** through the offset `A` and the scale `B`. Three
unknowns, two degrees of freedom, singular by construction — but that degeneracy is entirely *inside
the nuisances*. Location enters through `f(z;loc)`, a separate direction. Condemning location for a
nuisance-only degeneracy was an analysis error, not a physical result.

**Correct parameterisation: (location, offset, scale).** Full rank, and it is the honest description
of a portable instrument that knows neither the oil temperatures nor its own thermal contact.

## 3. The result

Location std as % of winding height, offset and scale both fitted:

| sensors | coupling | 0.1 K | 0.25 K | 0.5 K | 1.0 K | 2.0 K |
|---|---|---|---|---|---|---|
| 4 | 1.0 | 2.1 | 5.2 | 10.3 | 20.6 | 41.3 |
| **8** | **1.0** | **1.0** | **2.5** | **5.0** | 10.0 | 20.0 |
| **16** | **1.0** | 0.7 | **1.8** | **3.6** | 7.1 | 14.2 |
| 32 | 1.0 | 0.5 | 1.3 | 2.5 | 5.1 | 10.2 |
| 8 | 0.5 | 2.0 | 5.0 | 10.0 | 20.0 | 39.9 |
| 16 | 0.5 | 1.4 | 3.6 | 7.1 | 14.2 | 28.4 |
| 32 | 0.5 | 1.0 | 2.5 | 5.1 | 10.2 | 20.3 |

One and two sensors give `inf` — three parameters need at least three independent readings, and
four is the practical floor.

**Reference points, all with nuisances unknown:**

| configuration | result |
|---|---|
| 2 internal probes bracketing the hot spot, 2.0 K (E7) | **1.07 %H** |
| 16 external sensors, 0.25 K, perfect coupling | **1.78 %H** |
| 16 external sensors, 0.25 K, coupling 0.5 | 3.55 %H |
| the 4-channel external bundle, 0.5 K (E7) | 56.62 %H |

**So roughly 16 cheap external sensors land in the same league as 2 internal probes** — a few times
worse, but retrofittable, portable, and needing no manufacturer.

## 4. The portable case costs nothing extra

A clamp-on instrument on an unfamiliar tank does not know its own thermal coupling. **It does not
need to.** Coupling and the two oil endpoints collapse into the same offset-and-scale pair the array
already fits, so the bound with coupling unknown is *identical* to the bound with coupling known —
verified numerically, both tables match. The instrument self-calibrates from the profile shape.

That is the single most encouraging finding here, and it is exactly what a portable product needs.

## 5. The real risk, tested

The `A + B*f` form assumes **one** coupling for every sensor. Radiators, fins, paint and sun on one
side make contact vary with height, which eats directly into the shape signal. Adding a linear
height-tilt in coupling as a fourth unknown, at 0.25 K:

| sensors | uniform | with tilt | cost |
|---|---|---|---|
| 4 | 5.16 %H | 24.72 %H | **4.8× worse** |
| **8** | 2.50 %H | 2.71 %H | **1.1× worse** |
| 16 | 1.78 %H | 2.09 %H | 1.2× worse |
| 32 | 1.27 %H | 1.58 %H | 1.2× worse |

**Eight sensors or more and non-uniform coupling barely matters.** Four sensors collapse, because
four readings cannot support four parameters with anything left over. That is a clean design rule.

## 6. The specification

> **8–16 temperature sensors in a vertical line up the tank wall, effective noise ≤ 0.25 K,
> spanning as much of the tank height as possible. Fits offset, scale and location together, so it
> needs no knowledge of oil temperatures or its own thermal contact. Expect roughly 2–4 % of winding
> height, and stay at 8 or more sensors so non-uniform coupling stays harmless.**

Sensor quality is the binding requirement, not sensor count: 0.25 K external against 2.0 K for the
internal pair. Cheap NTC thermistors and class-A PT1000s are specified in that range, so it is not
obviously out of reach — but **effective** noise here includes mounting quality and coupling
variation, not just the datasheet figure, and will be worse than the part alone.

## 7. What is not established

- **The 1-D axial model has no radial structure, no tank geometry, no convection cells.** A real
  tank's outer wall temperature is not a clean readout of the internal oil column.
- **The wall-coupling model is mine**, a linear attenuation toward ambient. It is an assumption, not
  a measurement.
- **Common-mode error is not modelled and is favourable.** Sun, wind and ambient drift hit every
  sensor at once and largely cancel in a shape estimate. This is an argument *for* the array that
  the numbers above do not yet claim.
- **A clear vertical run of tank wall may not exist** on a unit covered in radiators and pipework.
  That is a survey question, not a modelling one.
- **A CRLB is a precision floor for an unbiased estimator**, not a guarantee that any estimator
  reaches it.
- **No instrument has been built, costed or tested.**

## 8. Next

1. **Verify against the SINTEF record.** It has top oil *and* bottom oil on a real unit for a year —
   two points on the profile. Not an array, but enough to test whether the modelled profile shape
   resembles a real one at all. This is free and it is the first real check available.
2. **Survey a real tank** for a usable vertical run and measure what wall-to-oil coupling actually
   is. Every number above scales with it.
3. **Bench-test the effective noise** of a candidate cheap sensor mounted on steel — the gap between
   datasheet accuracy and mounted, outdoor, effective noise is where this succeeds or fails.
4. **Model the radial dimension** before claiming anything about a real transformer.
