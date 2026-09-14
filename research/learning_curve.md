# Finding: P2Predict can tell a buyer whether more parts would help — but not how much

> **Status: open.** Per the project's working agreement this note **proposes and
> changes nothing in core**. Reproduce everything below with
> `.venv/bin/python research/learning_curve.py`.
>
> **The headline is a demotion.** The extrapolation gets the *direction* right
> in 100% of draws but the *magnitude* wrong in a way that is biased against
> the user: it under-promises where collecting more parts would help, and
> over-promises where it would not. This proposes shipping the curve's
> **shape**, not its numbers.

A category manager with 150 parts and a ±77% likely-range has exactly one
lever P2Predict never mentions: **collect more parts.** The tool says how good
the model is. It never says whether that is a ceiling or a starting point.

## Method

Two experiments, three case-study datasets subsampled into synthetic catalogs,
each graded on a **disjoint 3,000-row evaluation sample**. Full results:
[`learning_curve_results.json`](learning_curve_results.json).

1. **Ground truth.** Build catalogs at 50 → 900 parts, 20 resamples each, and
   record median APE and the conformal half-width.
2. **The honest test.** Using *only* an n-part catalog, trace an internal
   curve by subsampled cross-validation at sizes up to n, fit
   `err(m) = a·m^−b + c`, and extrapolate to 2n and 3n — then compare against
   experiment 1 at those sizes. Anchors 150 and 300; 15 draws each.

What is scored is the **predicted gain**, not the predicted level. "Doubling
gets you 6 points better" is the claim a buyer would act on.

Calibration uses K-fold cross-conformal throughout (see
[`small_n_conformal.md`](small_n_conformal.md)) — split conformal's
half-width at n=50 is too unstable to read a curve through.

## The curve exists, and it is steep where it matters

| catalog n | cars / ridge | heavy equip / ridge | heavy equip / xgboost | fasteners / ridge |
|---|---|---|---|---|
| 50 | 37.1% / ±154% | 29.9% / ±106% | 35.9% / ±135% | 93.0% / ±3599% |
| 100 | 33.3% / ±114% | 26.6% / ±81% | 30.3% / ±111% | 90.3% / ±2201% |
| **150** | **32.3% / ±107%** | **25.2% / ±77%** | **28.2% / ±91%** | **89.6% / ±1787%** |
| 300 | 30.0% / ±111% | 24.2% / ±70% | 25.7% / ±82% | 88.9% / ±1243% |
| 600 | 27.1% / ±88% | 23.2% / ±66% | 24.1% / ±72% | 88.4% / ±1123% |
| 900 | 25.6% / ±83% | 22.9% / ±65% | 23.8% / ±71% | 88.3% / ±1100% |

*(median APE / conformal half-width)*

Going from 50 to 150 parts — the range most catalogs sit in — is worth 4.8pp
of median APE on used cars and 4.7pp on heavy equipment, and cuts the quoted
half-width by a third. That is a real, actionable lever, and nothing in the
product currently mentions it exists.

Aerospace fasteners is the counter-case and the important one: **6× the data
buys 1.3pp** (89.6% → 88.3%). The features do not explain the price, and no
amount of collecting will change that. Telling that buyer to go find 150 more
parts costs them weeks for nothing.

## The extrapolation: right direction, wrong size, biased the wrong way

24 extrapolation checks (3 datasets × 2 algorithms × 2 anchors × 2 multipliers):

- **Direction correct in 100% of draws.** It never says "more data won't help"
  when it would.
- Mean gain error **+0.05pp**, median **+0.31pp**, max **2.9pp**.
- Within 3pp of the truth in **90%** of draws — but as low as **40%** in the
  worst cell.
- The power-law fit **fails outright in 16% of draws** (up to 40% in one
  cell), which at least fails loudly rather than inventing a number.

Those aggregates look shippable. They are hiding the problem, which only
appears when the error is split by dataset:

| dataset / model | total headroom (n=50 → 900) | mean gain error |
|---|---|---|
| used cars / ridge | 11.5pp | **−1.9pp** (under-promises) |
| used cars / xgboost | 12.2pp | −0.1pp |
| heavy equipment / ridge | 6.9pp | +0.2pp |
| heavy equipment / xgboost | 12.1pp | −0.0pp |
| aerospace fasteners / ridge | 4.7pp | **+0.8pp** (over-promises) |
| aerospace fasteners / xgboost | 4.7pp | **+1.4pp** (over-promises) |

**The bias runs backwards.** On the catalog with the most to gain the fit
understates the payoff by 1.9pp; on the catalog with the least to gain it
overstates it by up to 1.4pp — and on the fasteners/xgboost anchor of 150 it
promised **+3.7pp** where the truth was **+1.0pp**. A buyer told "another 300
parts should take you 3.7 points better" would be collecting data for a model
that is already at its noise floor.

### The half-width curve is noisier still

The number a buyer actually reads is the ±range, and it is less well behaved
than APE: used cars / ridge goes ±107% at 150 parts to **±111%** at 300 —
the wrong direction, inside resampling noise. Promising a *range* improvement
is measurably less safe than promising an *accuracy* improvement.

## Recommendation: ship the shape, not the number

The precise-gain feature is not supportable on this evidence. What **is**
supportable, and is arguably the more valuable half:

- **Report the fitted floor `c`, not the predicted gain.** The power law
  already estimates the irreducible error. On fasteners it lands near 88% and
  says the thing worth saying: *"your specs explain very little of this price
  — more parts will not fix it; better features might."* That connects
  directly to the existing `diagnose_noise.py` in the fasteners case study.
- **Classify the curve into three states** rather than quoting a number —
  `still_steep` / `flattening` / `flat` — from the fitted `b` and the residual
  headroom above `c`. Direction is the part that was right 100% of the time.
- **Say it conservatively when steep**: *"still improving at your current size
  — another 150 parts should help, likely by a few points of accuracy."* No
  decimal promise.
- **Stay silent when the fit fails** (16% of draws). A failed fit is a result,
  not an error to paper over.

All of it is additive: a new `data_sufficiency` block on the train path and in
`get_model_quality`. No prediction, interval, attribution or model-file change.

### Cost

The internal curve costs roughly `len(INTERNAL_FRACTIONS) × INTERNAL_REPEATS ×
K` extra fits — about 90 on the settings used here. On a 150-part catalog that
is seconds; it should be opt-in (`--data-sufficiency`) or run only below a row
threshold rather than on every train.

## What this does not cover

The curves here are traced by **random** subsampling, which assumes the next
150 parts look like the last 150. A buyer extending a catalog usually adds a
new supplier or a new family, which is a distribution shift — the estimate
would be optimistic. That interacts with the group-aware-splits gap (finding 4
in [`methodology_review.md`](methodology_review.md)) and is untested here.

Only ridge and XGBoost were run, tuning off, log-target forced on.
