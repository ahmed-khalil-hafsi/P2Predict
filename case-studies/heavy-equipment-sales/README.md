# Case Study: Heavy-Equipment Resale (Sell-Side)

Every P2Predict case study so far asks the buyer's question: *what should we pay?* This one flips the chair. The exact same parametric engine answers the **seller's** question: *what will this used machine fetch at auction, and which of its attributes lift or sink that number?*

The audience here is an equipment remarketing desk, a dealer's used-equipment team, or a fleet manager disposing of end-of-life assets. They set reserve prices, decide which machines to recondition before the sale, and argue value with bidders. P2Predict gives them the resale levers in dollars and percent, from public auction data.

This study uses the **Blue Book for Bulldozers** dataset: 401,125 real heavy-equipment auction sales from 1989–2011 (excavators, dozers, loaders, backhoes, graders, skid steers). It's the sell-side counterpart to the used-vehicle study, on machines that move six-figure money.

## The Sales Question

A machine is going across the block. From its vintage, category, size, cab type, and where it's being sold, we trained a P2Predict model to answer:

1. What is the expected hammer price?
2. Which attributes are driving that number — and by how much?
3. If a lot has an enclosed AC cab instead of an open station, what is that worth at resale?

---

## Part 1: Business Insights

We trained a P2Predict model on 80,000 auction records. It landed a **"Good"** accuracy rating (holdout R² = 0.738, median error ~18%) — but with an important honesty flag we'll get to: the model is a **comparator, not an appraiser**. It's excellent at telling you what an attribute is *worth relative to* the rest of the machine, and it should not be used as a single-number valuation. That's exactly the sell-side job: ranking levers, not replacing the auctioneer.

Here is what the analysis revealed.

### The Executive Summary

If you run equipment disposal, here are the levers you can act on today:

1. **An AC cab is real money at resale.** Holding every other attribute equal, an enclosed cab with air conditioning (EROPS w AC) sells for **+15%** over a plain enclosed cab, and **+22%** over an open station (OROPS). On the premium wheel loader below, the cab alone is worth **$9,619 (+19%)** of the hammer price. That tells you which lots are worth photographing and describing around the cab, and gives you a floor when a bidder claims "the cab doesn't matter."

   ![Same machine, different cab — the resale lever you control](assets/enclosure_premium.png)

2. **Depreciation is a front-loaded curve, not a straight line.** A machine loses value fastest in its first five years, then flattens hard. Our baseline loader is worth +97% of the average at 0 years, but the drop from year 2 to year 5 alone erases ~$22,700, while the entire decade from year 20 to 30 only takes off ~$5,000. The selling lesson: the cost of holding an asset one more year is huge when it's young and trivial when it's old.

   ![Depreciation is a front-loaded curve](assets/depreciation_curve.png)

3. **Size and category dominate everything else.** Machine size (Mini → Large) and product group together drive **68%** of the model's decisions. A Mini machine fetches **-67%** versus a Medium; a skid steer fetches **-60%** versus a wheel loader. When you're triaging a mixed fleet for disposal, this is the first cut.

4. **Auction timing is a ~30% swing.** Holding the machine fixed, the same lot that sold at a +7% premium in 2006 sold at **-14%** in the 2009 downturn, then rebounded to **+16%** in 2011. The market cycle is a bigger lever than most sellers assume.

5. **Geography barely matters.** Moving the same machine across states shifts the price by only ±2%. Auction *location* is a rounding error next to *what* and *when* you sell. Don't truck a machine across the country chasing a regional premium that isn't there.

### Where to Trust the Model (and Where to Get a Comp)

P2Predict is designed to make its own limits visible. This model earned a blunt computed verdict:

> **Good accuracy, but the model runs systematically high or low — its single-number estimates aren't trustworthy. Use it only to compare options, not to set an absolute target.**

That's the honest read, and it shapes how to use it:

* **🟢 Trust the relative levers.** The cab premium, the depreciation shape, the size and category rankings — these are what the model is *for*. Rank lots, quantify what an attribute is worth, and defend a negotiating position with them.
* **🟢 Trust the mid-market bands.** Accuracy is strongest in the **$13,500–$30,000** range (median error ~15–16%), which is where the bulk of auction volume lives.
* **🔴 Verify the absolute number with a comp.** Because the model carries a systematic bias (a known side effect of modelling on a percentage scale, see Under the Hood), don't take the point estimate as a reserve price on its own. Use the 90% range as a sanity band and confirm the number against recent comparable sales.
* **🔴 Widen the band on cheap and premium extremes.** Error is worst on sub-$10k machines (27% median) and on the $67k+ top band (~23%). For those, lean harder on a real comp.

### Worked Example: How the Model Prices a Lot

P2Predict breaks down exactly why a lot will fetch what it fetches. Because prices are heavily right-skewed, the model works on a percentage scale, so each driver reads as a **% lift or cut** on the price.

![Honest ranges across three lots](assets/intervals_comparison.png)

Take the **3-year-old Large wheel loader with an AC cab** (sold 2008, Texas). The model predicts **$51,697** — a 2.0x multiple of its $25,389 average machine. Here is how it gets there:

![Per-driver resale attribution for the wheel loader](assets/wheel_loader_attribution.png)

```text
  Average machine (baseline):   $25,389
  Prediction:                   $51,697   (x2.04)

  Resale drivers (% lift/cut on the price):
    Age (3 yr, near-new)        + 28.0%   (+$9,137)
    Size (Large)                + 24.9%   (+$8,215)
    Cab (enclosed + AC)         + 19.9%   (+$6,699)
    Category (Wheel Loader)     +  5.3%   (+$1,919)
    Auction year (2008)         +  2.8%   (+$1,025)
    Auction state (Texas)       -  1.8%   (-$687)

  Product of factors = 2.036 = prediction / baseline  ✓
```

The attribution is exact: the product of the individual factors reconstructs the prediction to the dollar. P2Predict checks that axiom on every run.

### The What-If Scenario

You can ask the tool how a single change moves the price. Take that same wheel loader and strip the enclosed AC cab down to an open station:

```text
  As listed (AC cab):           $51,697
  Stripped to open station:     $42,078
  The cab is worth:              $9,619  (+18.6% of hammer)
```

That's the number you take into a lot walkthrough: *this cab is carrying nearly a fifth of the machine's resale value.* Whether it's worth reconditioning a damaged cab before the sale becomes an easy arithmetic decision, not a gut call.

---

## Part 2: Under the Hood

For the technical team, here is how P2Predict processes the data, builds the model, and generates the outputs above.

### Data

* **Source:** the Kaggle *Blue Book for Bulldozers* dataset (Fast Iron / Ritchie Bros. auction records). The original is a gated competition; `fetch_data.py` pulls an openly-downloadable re-upload of the identical `Train.csv`.
* **Size:** 401,125 raw auction records, 53 columns, 1989–2011.
* **Cleaning:** ~38k records use `1000` as an "unknown build year" sentinel. A resale model is anchored on machine age, so we drop those rather than impute a fake vintage, leaving **362,800** clean records. We then sample 80,000 for tractable hyperparameter search.
* **Features kept (6):** `age_at_sale` (derived from build year and sale date), `sale_year`, `product_group`, `product_size`, `enclosure` (cab type), and `state`.

### Pipeline & Methodology

* **Target transformation:** the hammer price is heavily right-skewed (skew ≈ 1.5), so P2Predict automatically wraps the target with a log transform. As a result the model is **multiplicative**: SHAP drivers come out as percentages, and conformal intervals stay strictly positive.
* **Algorithm selection:** P2Predict ran cross-validation across Ridge, Random Forest, and XGBoost. **XGBoost won** (CV R² = 0.79, vs 0.785 Random Forest, 0.729 Ridge). This is the mirror image of the 150-part Battery Management IC study, where Ridge won: with 80,000 rows there is plenty of data for gradient-boosted trees to shine.
* **Categorical encoding:** because a tree model won, P2Predict automatically switches categoricals to **target-encoding**, so the trees split on resale price rather than alphabetical category order. This is what lets `state` (53 values) and `product_group` participate cleanly.
* **Confidence intervals:** split-conformal prediction on a held-out set, calibrated in *bands* by predicted price, so a $9k skid steer gets a proportionally tighter range than a $50k loader instead of one global margin.
* **Feature attribution (SHAP):** the multiplicative factors reconstruct the prediction exactly (product of factors = prediction ÷ baseline). P2Predict enforces that axiom on every run.

### Model Performance

| Metric | Result | What it means |
|---|---|---|
| **Holdout R²** | **0.738** | Explains ~74% of hammer-price variation from six basic attributes. |
| **MAE** | **$7,690** | The typical miss on a machine whose median price is ~$25k. |
| **Median % error** | **18.4%** | Half of predictions land within ~18% of the actual sale price. |
| **Residual bias** | **flagged** | The model runs systematically high or low. This is why the verdict is "compare, don't appraise." |

Why the bias? Modelling on a log (percentage) scale and transforming back to dollars introduces a small, systematic level shift (a well-known property of log-target back-transforms). It barely affects the *relative* levers this study is built on, but it's exactly why P2Predict refuses to bless the single-number estimate — and says so, instead of quietly shipping an over-confident valuation.

### Visual Quality Report

P2Predict generates a procurement-ready PDF detailing model calibration and feature importance. *(Full PDF in [`assets/model_quality_report.pdf`](assets/model_quality_report.pdf))*

**1. Overall Accuracy:**
![Model quality report, page 1](assets/model_quality_report_page_1.png)

**2. Calibration by Price Band:**
Strongest in the mid-market ($13.5k–$30k), weakest on the cheapest and priciest machines — a clear map of where to trust the benchmark and where to pull a comp.
![Model quality report, page 2](assets/model_quality_report_page_2.png)

**3. Feature Importance:**
Size and category drive 68% of the decision; the enclosure lever holds a genuine 14%; auction geography is a rounding error at under 1%.
![Model quality report, page 3](assets/model_quality_report_page_3.png)

---

## Part 3: Reproducing the Results

You can reproduce this exact analysis from the command line.

Run these from the **repository root** — `p2predict-train` writes the model into `./models/`, which is where the case-study scripts look for it.

### Full Path (Requires a Free Kaggle API Token)

```bash
# 1. Install and set up a Kaggle token (https://www.kaggle.com/settings)
pip install -e . 'kagglehub>=0.4.1'
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
printf 'KGAT_...' > ~/.kaggle/api_token && chmod 600 ~/.kaggle/api_token

# 2. Fetch the auction data (~111 MB) and clean it (scripts resolve their own paths)
python case-studies/heavy-equipment-sales/fetch_data.py
python case-studies/heavy-equipment-sales/prepare_data.py

# 3. Train the model (XGBoost wins on this data volume). Model lands in ./models/
p2predict-train \
  -i case-studies/heavy-equipment-sales/data/bulldozers_training.csv \
  -t sale_price_usd \
  -tf "age_at_sale,sale_year,product_group,product_size,enclosure,state" \
  --outliers warn \
  --feature-outliers warn \
  --budget thorough

# 4. Generate insights, charts, and the PDF report
python case-studies/heavy-equipment-sales/predict_examples.py
python case-studies/heavy-equipment-sales/extract_insights.py
python case-studies/heavy-equipment-sales/generate_charts.py
python case-studies/heavy-equipment-sales/generate_quality_report.py
```

### Quick Path (No Kaggle Account Needed)

The Blue Book data is openly redistributable, so we committed a 5,000-row sample. The metrics will be rougher, but the pipeline runs end-to-end. From the repository root:

```bash
p2predict-train \
  -i case-studies/heavy-equipment-sales/data-sample/bulldozers_sample.csv \
  -t sale_price_usd \
  -tf "age_at_sale,sale_year,product_group,product_size,enclosure,state" \
  --outliers warn \
  --feature-outliers warn \
  --budget thorough
```

## Limitations & Next Steps

* **Comparator, not appraiser.** The systematic residual bias means this model ranks and quantifies levers well but should not set a reserve price on its own. Pair it with recent comps.
* **No machine hours.** The meter reading (`MachineHoursCurrentMeter`) is blank or zero on ~83% of records, so it can't carry a usage signal without masquerading missingness as "brand new." We let machine age carry the wear story instead. A dataset with reliable hours would add a second usage lever.
* **No make/model.** The dataset encodes manufacturer inside a high-cardinality model-description string rather than a clean column, so this study can't quantify a brand premium the way the Battery Management IC study quantifies a supplier premium. That's the natural next feature to engineer.
* **Random split, not chronological.** Auction data is time-ordered; a `--time-column` chronological split would prevent any look-ahead and let the model be used to forecast forward-looking resale, not just explain historical sales.
