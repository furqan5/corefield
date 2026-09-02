# The streamline paper, the tank wall, and whether migration sells

**2 September 2026.** Branch `ml-lab`. Adjudicating two contradictory research answers and reading
the one paper the diligence report called prior art.

---

## 1. The streamline paper is NOT prior art for what we propose

**"Research on Transformer Hot-Spot Temperature Inversion Method Under Three-Phase Unbalanced
Conditions", *Energies* 2025, 18(16), 4422** (accepted 15 Aug 2025).

The diligence report cited this to mark claim B — infrared to hot-spot **location** — as "partly
occupied". Reading it, that is overstated on four counts:

| | the paper | our concept |
|---|---|---|
| **output** | hot-spot **temperature** (HST) | hot-spot **location**, and change in it |
| **inputs** | streamlines from a **finite-volume CFD** solve, reduced by genetic algorithm, fed to **support vector regression** | measured tank-wall temperature profile |
| **role of location** | a nuisance to accommodate — streamlines are "reasonably selected in response to variations in hot-spot locations" | the quantity being measured |
| **unit** | **S13-M-100 kVA / 10 kV** distribution transformer | power transformers |

**It inverts for temperature and treats moving location as something to survive, not to measure.**
And it does so from a simulated internal field, not from an external measurement.

**(c) It is better read as support than as prior art.** It establishes in the literature that
hot-spot location *does* move under unbalanced loading, which is the premise our migration concept
depends on. It does not measure that movement externally, which is what we would be claiming.

Claim B should be downgraded from "partly occupied" to **"appears open, with the premise
independently corroborated"**. That is a stronger position than the report gave us.

## 2. The tank-wall question has an answer, and it is a segmentation

Two research answers contradicted each other flatly. One said radiators block 40–60 % of the
surface, leaving a usable run. The other said tank walls are corrugated and there is no flat plate
at all. **Both are right, about different transformers**, and neither said so:

| construction | used on | flat wall? |
|---|---|---|
| **corrugated fin wall** — fins welded directly to the shell, the wall *is* the radiator | **distribution transformers** | **no. Concept dead.** |
| **pressed-steel radiator banks** — separate panels welded to inlet/outlet manifolds | **power transformers** | yes, obstructed but present |

**(a)** Confirmed in the tank-construction literature: corrugated fin radiators are "commonly used
with distribution transformers", while pressed-steel panel radiators welded to manifold pipes are
"commonly used in power transformers".

**Consequence, and it is clean:** the concept is **dead for distribution transformers** and
**viable-but-obstructed for power transformers**. That is the right segmentation and it happens to
match where this project's data already sits — a 360 MVA ODAF and a 40 MVA unit.

It also means the streamline paper's 100 kVA unit is almost certainly corrugated, which is a second
reason it is not comparable.

**Still unanswered:** how much clear vertical run remains on a real power transformer once
stiffeners, pipework, bushing turrets, control cabinets and conduit are counted. One photograph of
a real unit answers it and no amount of searching has.

## 3. Does migration sell? The sceptical answer is the better one

The optimistic answer argued utilities value migration via dynamic rating, insulation life and
bubble formation. **Read carefully, those three arguments are about hot-spot TEMPERATURE, not
location.** Ageing, dynamic rating and bubble inception all depend on how hot the hot spot is,
wherever it happens to be. They do not establish value for knowing *where*.

The sceptical answer made a sharper and more credible claim about actual practice: winding
temperature indicators are **programmed with the manufacturer's worst-case gradient and a fixed
hot-spot factor**, and where the hot spot may shift between windings as a tap changer moves,
monitors are simply **set for the higher rise of the worst-case combination rather than tracking
the movement**.

**(c) That is the real answer to "why doesn't anyone measure this": they bound it instead.**
Assuming the worst case is conservative, cheap, and safe. A measurement that refines a conservative
bound only has value where the bound itself fails.

**Where does it fail?** One case survives the scrutiny, and it is the one the optimistic answer
stumbled into: *if a migrating hot spot moves to an unmonitored area*, the fixed probe or the
worst-case WTI is no longer measuring the hot spot. The literature's own standing complaint about
fibre is exactly this — probe position deviates from true hot-spot position and there is no way to
tell. **An external estimate of location is a check on whether the internal instrument is still
pointed at the right place.**

That is a narrow proposition. It is not dynamic rating, it is not a control-room product, and it
should not be sold as either.

## 4. Honest position after all of this

**Stronger than before:** the one cited prior-art paper does not do what we propose, and it
corroborates the premise that hot spots move. The eddy objection was tested and migration detection
is immune to it. Four novelty claims remain unoccupied.

**Weaker than before:** the addressable market is **power transformers only** — corrugated-wall
distribution units are excluded by construction. And the value proposition has narrowed from
"locate the hot spot" to "check whether the installed probe is still measuring the hot spot", which
is a real need but a small one.

**The two questions that decide it remain unanswered by any amount of desk work:**

1. **Photograph a real power transformer.** How much unobstructed vertical tank wall is there?
2. **Ask one utility engineer:** when your fibre probe or WTI was commissioned, how do you know it
   is at the hot spot — and would you pay to find out that it is not?

The second question is now the whole commercial case. If the answer is "we assume worst case and
that is fine", the concept is an instrument without a buyer, and that is worth knowing before
anything is built.
