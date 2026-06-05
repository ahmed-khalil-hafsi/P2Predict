# Case study: Aerospace and defense contract pricing

> **Status:** Template. Scripts and narrative are scaffolded — results need to be filled in once the case study has been run.

## The question

> Given the technical and contractual features of a procurement contract (commodity class, agency, contractor, set-aside type, period of performance, place of performance), what's the expected obligated dollar value?

## Why this case study

This one is about **credibility with the cost-estimating community**. ICEAA's ~3,000 members, NASA's cost-estimating organisation, and the cost engineers at every major defense prime invented parametric estimating. Their public datasets are coarser than per-part EE data, but showing P2Predict on government contract data sends the strongest possible signal to that audience.

It's also genuinely a different case study from #1 (electronics) and #2 (used cars): government contracts are typically *contract-level* not part-level, so the modeling story is about predicting contract dollar values from contract attributes rather than unit prices from spec attributes. Honest framing: this is parametric estimating at the **bid / contract** level, the level the cost-estimating profession most often works at.

## Data

**Recommended source:** [USAspending.gov contract data](https://www.usaspending.gov/download_center). Public domain — fully redistributable. The "Award-level data" download for FPDS contracts is the right shape.

**Alternatives:**
- [NASA Procurement Data System](https://prod.nais.nasa.gov/cgibin/npas) — narrower scope, easier to filter to specific commodity codes.
- [SAM.gov contract opportunities](https://sam.gov/content/opportunities) — current and recent solicitations rather than awards.

**License:** USAspending data is U.S. Government public domain. We could check in a small filtered sample (e.g. one PSC code in one fiscal year) for smoke tests and quick reproducibility — that's the only one of the three case studies where checking in a sample is unambiguously fine.

**Pick one commodity (PSC code) for the case study.** Examples:
- **1560 (Airframe Structural Components)** — aerospace structural parts; many similar contracts; good fit.
- **5805 (Telephone and Telegraph Equipment)** — defence comms; clean shape.
- **6605 (Navigation Instruments)** — high-spec, parametric-friendly.

Mixing PSCs gives you a model that's averaging across very different categories — same problem as mixing capacitors with connectors in case study #1. One PSC, deep.

**Features to model on:**
- Awarding agency (categorical)
- Contractor (categorical, possibly very high cardinality — `OrdinalEncoder` works well)
- Set-aside type (categorical: small business, 8(a), unrestricted, …)
- Period of performance, in months (numerical)
- Place of performance state (categorical)
- Competition type (categorical)
- Major subcategory codes (categorical)
- Fiscal year (numerical or treated as time column for `--time-column`)

**Target:** `obligated_amount` (USD). Heavy right skew — `should_log_target` will fire.

**Honest note for the README's framing:** these contracts are *coarser* than per-part procurement. The case study should frame it as "given everything we know about how a contract is structured, what's its expected dollar value?" — useful for benchmarking new solicitations against historical ones in the same PSC, not for predicting unit prices of specific parts.

## Reproducing this case study

```bash
# 1. Download FPDS contract data for one PSC and one fiscal year.
python case-studies/aerospace-contracts/fetch_data.py \
  --psc 1560 --fiscal-year 2023

# 2. Train. Use --time-column award_date because government contracts
#    are time-ordered and chronological CV matters here.
p2predict-train \
  --input case-studies/aerospace-contracts/data/contracts.csv \
  --target obligated_amount \
  --time-column award_date \
  --outliers warn \
  --feature-outliers warn \
  --budget thorough

# 3. Sample predictions.
python case-studies/aerospace-contracts/predict_examples.py
```

## Results

> _To be filled in once the case study has been run._

- PSC selected: ?
- Fiscal year(s): ?
- Algorithm selected (auto-mode): ?
- Log-target activated: ✅ (expected — heavy right skew on dollar values)
- Holdout R²: ?
- Holdout MAPE: ?
- 90% empirical coverage: ?
- Time-aware split confirmed: ✅
- Top contributors to obligated amount: ?

## The story

> _Fill in once the case study has been run. Suggested form for the cost-estimating audience:_
>
> *"Trained on 4,200 FPDS contract records for PSC 1560 (Airframe Structural Components), FY 2018–2023. The model picks XGBoost; the log-target wrap activates because dollar values span four orders of magnitude. SHAP attribution shows the dominant predictors are the contractor's tier and the period of performance, more than the awarding agency or set-aside type. Used by a cost estimator: a new solicitation in PSC 1560 with a 36-month POP and a tier-1 prime, set aside as full-and-open, would be benchmarked against the model's 90% likely range — quotes outside that range merit a structured cost analysis review."*

## Worked examples for the README

> _Once results are in, paste 2–3 examples of `p2predict -m M -p "..." --explain --interval 90`._

## Caveat to declare clearly

This case study uses **publicly obligated dollar amounts on completed contracts**, not contractor cost breakdowns. The model predicts what similar contracts were awarded at, not what they "should" have cost in a bottom-up sense. That's the same parametric-vs-should-cost distinction the main README's "What it is (and isn't)" section makes, but it's worth restating in this case study because the cost-estimating audience will look for it.
