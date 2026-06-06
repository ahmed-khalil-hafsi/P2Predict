# Case study: Used vehicle pricing

> **Status:** Template. Scripts and narrative are scaffolded — results need to be filled in once the case study has been run.

## The question

> Given a used vehicle's make, model, year, mileage, trim, transmission, and region, what's the expected listing price — and how much does each spec contribute to the answer?

## Why this case study

This is the warm-up. It's not procurement, but it's the strongest possible visual demo of P2Predict's log-target multiplicative SHAP attribution because used-car prices span orders of magnitude ($800 beaters to $80k specialty cars). Anyone reading the case study has either bought or thought about buying a used car, so the mental model is free.

Frame it explicitly in the README as the **tutorial** case study — *"if you want to try P2Predict on data you already understand, start here."*

## Data

**Recommended source:** [Craigslist Cars+Trucks dataset on Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) — 426k US listings, CC0 license, redistributable.

**Alternative:** [Car Auction Prices dataset](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices) — 550k+ auction records with cleaner structure but smaller feature set.

**License:** both are CC-licensed; the Craigslist dataset is CC0. We could check in a small sample for smoke tests if useful, but the full dataset is too large for git.

**Features the model should learn from:**
- `manufacturer` (categorical)
- `model` (categorical, high cardinality — perfect for the `OrdinalEncoder` path on tree models)
- `year` (numerical)
- `odometer` (numerical, often log-distributed)
- `condition` (categorical: excellent / good / fair / salvage / …)
- `transmission` (categorical)
- `fuel` (categorical)
- `state` (categorical)
- `drive` (categorical: 4wd / fwd / rwd)

**Target:** `price` (USD).

**Data quality flags** worth mentioning in the case study narrative:
- Strong right-skew (handful of $99k+ specialty listings) → log-target should activate automatically (`should_log_target` returns True).
- Outliers in `price` (placeholder $1, $123,456 prices) → `--outliers warn` will report them, `--outliers drop` cleans them.
- Outliers in `odometer` (e.g. someone typed 9999999) → `--feature-outliers drop` removes those rows.
- Some Craigslist listings have a `year` of `2025` for new cars listed used; legit.

This makes used cars an excellent dataset for **demonstrating outlier handling alongside the modeling story**.

## Reproducing this case study

```bash
# 1. Download the Craigslist dataset from Kaggle.
#    Either through the Kaggle CLI:
kaggle datasets download -d austinreese/craigslist-carstrucks-data \
  -p case-studies/used-cars/data/ --unzip

# 2. Train.
p2predict-train \
  --input case-studies/used-cars/data/vehicles.csv \
  --target price \
  --outliers drop \
  --feature-outliers drop \
  --budget thorough

# 3. Sample predictions.
python case-studies/used-cars/predict_examples.py
```

## Results

> _To be filled in once the case study has been run._

- Algorithm selected (auto-mode): ?
- Log-target activated automatically: ✅ (expected — heavy right skew)
- Outliers dropped (target): ? rows
- Outliers dropped (features): ? rows
- Holdout R² (after outlier handling): ?
- Holdout MAPE: ?
- Empirical 90% coverage: ?
- Top feature importances: ?

## The story

> _Fill in after running. Suggested form for the tutorial framing:_
>
> *"Trained on 320k US Craigslist listings (after outlier handling). The model picks XGBoost; the log-target transform activates automatically because prices span $800 to $80k. SHAP says mileage is the single biggest factor (negative — older cars cost less), then year, then make. Using `--whatif` on a 2019 Honda Civic EX with 45k miles: change `odometer:45000 → odometer:90000` and the prediction drops $X.XX (-Y%). That's the depreciation per mile, learned from the data."*

## Worked examples for the README

> _Once results are in, add 2–3 worked examples of `p2predict -m M -p "..." --explain --interval 90 --whatif "..."`._
