# Case study: PCBA cost, composing three models into a bill of materials

> **Status:** Template. Scripts and narrative are scaffolded, results need to be filled in once the case study has been run.

## The procurement question

> What's the total cost of a Printed Circuit Board Assembly (PCBA), broken down into the three things that actually drive it: the components, the bare PCB, and the assembly operation? And when an engineer proposes a change, *"swap this connector,"* *"go from 4-layer to 6-layer,"* *"add a coating step"*, what's the predicted impact on the total cost and on its likely range?

## Why this case study

This is **composability**. Every prior case study trains one model and predicts one number. This one trains three models, components, bare PCB, assembly, and composes them into a bill-of-materials cost prediction with per-stage attribution.

This is how serious hardware-procurement cost engineering actually works. The cost-estimating community calls it *hierarchical parametric estimating*; aerospace cost engineers build entire work-breakdown-structure cost models this way. Most hardware companies do it today with spreadsheets and one senior engineer's memory.

A tool that composes parametric models rigorously, predicting each level, aggregating the predictions, and attributing the total back to the inputs that drive it, is the natural extension of P2Predict into BOM-level cost engineering.

## What's honest to claim, and what isn't

The thin version (predict three models, sum the point estimates) is **straightforward and useful today**. You can run it with the v0.8 Python API and ~20 lines of glue code, `compose.py` in this folder is that glue code.

The principled version that handles the math correctly, propagating likely-range intervals through a sum, composing SHAP attribution across heterogeneous component models, and producing a true BOM-level what-if, is a real chunk of math. Specifically:

- **Intervals don't naively sum.** If component A has a 90% likely range of $0.40–$0.55 and component B has a 90% range of $0.10–$0.18, the 90% range of A+B is *not* $0.50–$0.73, adding the bounds overshoots and you end up reporting something closer to a 99% range. Doing it right requires variance propagation, conformal aggregation, or Monte Carlo from per-component residuals. `compose.py` ships with **naive addition** of the bounds and is loud about that limitation in its output. Treat the per-stage intervals as informative; treat the summed interval as an upper-bound estimate.

- **SHAP across heterogeneous models needs care.** Capacitors have voltage rating; ICs have package and node size; PCB models have layer count and area. You can't sum SHAP feature-by-feature across them. What `compose.py` does instead is a **two-level attribution**: at the top, *"this $42 PCBA breaks down into $18 components, $7 PCB, $17 assembly"*. Within each stage, the standard per-feature SHAP from the underlying model. The narrative is procurement-natural; the math is honest.

The principled version of both of these is a candidate Pro-tier or v1.x feature. This case study demonstrates the *pattern* and the *use case*, and is honest about the limitation that matters.

## Data

This case study composes **three datasets**, not one. The honesty up front:

| Stage | Data shape | Source | Difficulty |
|---|---|---|---|
| Components | Per-part technical specs + unit price | [Octopart / Mouser / DigiKey API](../battery-management-ics/), already covered by case study #1 | Easy |
| Bare PCB | Per-board specs (layer count, area, finish, copper weight, vias) + price | JLCPCB / OSH Park / Sierra Circuits published price calculators, or scraped quote responses | Medium |
| Assembly (SMT/THT placement, solder, test) | Per-board operation specs (component count, placement type, double-sided, test coverage) + price | EMS provider published rates; or [Murphy's parametric assembly model](https://www.sciencedirect.com/topics/engineering/printed-circuit-board-assembly) as a calibration starting point | Hard |

The component model can be reused directly from [case study #1](../battery-management-ics/), train it once, save the `.model`, use it here.

The PCB and assembly datasets are smaller (hundreds of records each is enough for a decent parametric model) and the work is mostly in collecting them. Suggested starting approach: scrape a few hundred quotes from a public PCB fab service across varying specs, and use a parametric assembly model calibrated against ~50 historical PCBA quotes you have access to.

## Reproducing this case study

```bash
# 1. Train the component model from the battery-management-ics case study.
#    (See case-studies/battery-management-ics/ for fetch + train.)
p2predict-train -i case-studies/battery-management-ics/data/components.csv \
                -t unit_price_1k --budget thorough
# Save the path of the resulting model, compose.py will need it.

# 2. Train the PCB cost model (data from JLCPCB / OSH Park).
p2predict-train -i case-studies/pcba-composition/data/pcb_quotes.csv \
                -t price_per_board --budget thorough

# 3. Train the assembly cost model.
p2predict-train -i case-studies/pcba-composition/data/assembly_quotes.csv \
                -t price_per_board --budget thorough

# 4. Run the composition demo. Predicts an example PCBA cost from a BOM CSV.
python case-studies/pcba-composition/compose.py \
    --component-model models/<component-model.model> \
    --pcb-model models/<pcb-model.model> \
    --assembly-model models/<assembly-model.model> \
    --bom case-studies/pcba-composition/example_bom.csv
```

## Results

> _To be filled in once the case study has been run._

Per-stage:
- Component model: ? components, R² = ?, MAPE = ?
- PCB model: ? boards, R² = ?, MAPE = ?
- Assembly model: ? boards, R² = ?, MAPE = ?

BOM-level (example PCBA):
- Total predicted cost: ?
- Naive summed 90% range: ?–? (with the honest caveat about over-coverage)
- Breakdown: components $?, PCB $?, assembly $?
- Top component contributors: ?

## The procurement insight

> _To be filled in. Suggested form:_
>
> *"For a 60-component IoT sensor PCBA on a 4-layer board with standard SMT assembly, P2Predict's composed model predicts a unit cost of $X.XX. The biggest single driver isn't any one component, it's the assembly stage ($Y, 40% of total), followed by the 12 most expensive components (combined $Z, 35% of total). Using `compose.py --whatif`, swapping the connector from $A to $B at the same spec drops total PCBA cost by $C (-D%), driven mostly by the component delta but with a small assembly-side effect from the reduced placement complexity."*

## What this case study deliberately does not do

- It does **not** produce a defensibly-tight likely range on the total. The naive sum of per-stage intervals overshoots, clearly labelled in the output.
- It does **not** handle correlated risk (a supply-chain disruption that pushes both components and PCB prices). Independence is assumed.
- It does **not** optimise the BOM (suggest cheaper alternatives). It estimates the cost of a given BOM; suggesting alternatives is a different (future) feature.

These limitations are the entry point for a future Pro-tier "BOM cost engineering" feature that handles them correctly.
