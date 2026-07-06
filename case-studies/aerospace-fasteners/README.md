# Case Study: Aerospace Fasteners

Benchmarking aerospace fasteners is notoriously messy. This case study uses real, public-domain catalog data from the U.S. Defense Logistics Agency's **PUB LOG**—the master catalog the DoD, NASA, and their contractors use to order parts. 

Bolts seem like a perfect target for parametric pricing because they are defined by rigid physical specs. However, the reality of the catalog is different: the exact same physical bolt can be listed across a massive price band. 

This case study is about reality. Instead of chasing a fake "90% accuracy" metric, it shows you how to mathematically measure the noise in your data, extract the few directional signals you can actually trust, and know exactly when to stop tuning your model.

## The Sourcing Question

Given a bolt's physical specs—material, head style, thread diameter, length, thread class, strength grade, and finish—what should it cost? And more importantly, what is the exact price premium for an aerospace-grade material like titanium or nickel alloy compared to a commodity steel bolt?

---

## Part 1: Business Insights

We pulled ~188,000 bolt records from PUB LOG, cleaned the data, and trained a P2Predict model on the 36,668 valid bolts that remained. The model's accuracy is objectively low (a median error of ~80%). 

But that is the point of this study: **no model can predict a precise price when the underlying catalog data is inherently scattered.** Here is what the data actually tells us.

### The Executive Summary

If you are sourcing fasteners, here is how you use this data:

1. **The catalog is intrinsically noisy.** This is the headline. 80% of the bolts in the catalog have a completely unique spec signature. For the 20% that *do* share identical specs, the catalog prices them across a **4.5x price band**. Because identical parts have wildly different prices, no predictive model will ever give you a perfect point estimate. 
2. **Material grade is a directional hint, not a strict calculator.** Upgrading from commodity alloy steel to titanium (+174%) or nickel alloy (+213%) carries a clear, massive premium. However, the exact order of the middle tiers is messy (for example, A286 superalloy occasionally prices lower than basic CRES). Trust the overall direction, not the exact percentages.
3. **Length is your cleanest cost ruler.** A commodity hex bolt reliably scales in price as it gets longer—from $2.75 at 0.5 inches up to $7.01 at 3.0 inches (+155%). If a supplier quotes you a flat price across a large length jump, or falls wildly off this curve, push back.

### Where to Trust the Model (and Where to Ignore It)

This model provides guardrails, not exact appraisals. Here is your trust map:

* **🟢 Trust the length cost ruler.** It is a clean, physically sensible cost driver.
* **🟡 Use material premiums directionally.** You know titanium is roughly 2–3x more expensive than steel. Use that as an anchor, but don't quote a hyper-specific benchmark price.
* **🔴 Ignore exact point estimates.** The median error is nearly 80%. The catalog is too scattered to use the model's exact dollar output for a negotiation. 
* **🔴 Do not benchmark sub-$5 commodity bolts.** This is where the catalog is noisiest. A few cents of variance creates a massive percentage error. 

### Worked Examples: Honest Uncertainty

Because fastener prices span from pennies to thousands of dollars, we use a multiplicative model. This means our confidence intervals stay strictly positive and scale with the price of the part.

Here is how the model prices three different 1/4-28 × 1.0″ bolts:

| Bolt Archetype | Point Estimate | 90% Likely Range | Range Width |
|---|---|---|---|
| Commodity alloy-steel hex | **$2.64** | $0.14 – $50.79 | 369x |
| CRES hex | **$3.88** | $0.20 – $74.58 | 369x |
| Aerospace titanium 12-point | **$29.73** | $2.92 – $302.36 | 103x |

Read these massive ranges as a feature, not a bug. The model is telling you exactly how chaotic the catalog is. 

However, notice the **titanium bolt**. P2Predict uses banded calibration, meaning it calculates confidence intervals based on price tiers. The model is actually *more* consistent on high-dollar aerospace parts than it is on cheap commodity bolts, so the confidence band correctly tightens where the real procurement dollars are spent.

---

## Part 2: Diagnosing Noisy Data

When a model outputs a low R² score, the immediate reflex is to blame the algorithm and keep tuning. This case study demonstrates how to prove the data is the bottleneck using two heuristics built into P2Predict (`diagnose_noise.py`).

**1. Signature Uniqueness**
We grouped every row by its full physical spec. We found that **80% of bolts are one-offs**—meaning they are the only bolt in the catalog with that exact combination of features. A model cannot interpolate trends if almost every part is entirely unique.

**2. The Duplicate-Signature Noise Floor**
For the remaining 20% of bolts that *do* share identical specs, we measured their price variance. We found that identical specs are priced across a median **4.5x band**. 

![Why the ceiling is ~0.60: identical specs, very different prices](assets/noise_floor.png)

This variance is irreducible. Because the inputs are identical, no algorithm can predict the difference in their prices. Based on this math, the absolute maximum theoretical accuracy (R²) for this dataset is **~0.60**. Our model hit ~0.32 in log space. It captured the available signal, and the rest is simply catalog noise.

---

## Part 3: Under the Hood

For the technical team, here is how the pipeline processes the DLA PUB LOG data.

### Data
* **Source:** DLA PUB LOG (Public Domain). We isolated Federal Supply Class (FSC) 5306.
* **Streaming Extraction:** The raw catalog files are multi-gigabyte ZIPs. Our script reads them line-by-line without unzipping the whole file to disk, filtering only for bolts. 
* **Unit-of-Issue Normalization:** The catalog lists prices per-each, per-hundred, per-dozen, etc. We normalize everything to a true per-each price before modeling so the algorithm isn't confused by package quantities.
* **Domain Capping:** We hard-capped the dataset at $2,000 per bolt. This drops mis-cataloged kitted assemblies (e.g., an $80k "bolt") without deleting valid feature outliers.

### Pipeline & Methodology

* **Target Transformation:** Prices range from $0.01 to $2,000, creating a massive right skew (5.36). We explicitly turn on `--log-target on`. This ensures our SHAP attributions are multiplicative (percentages) and our confidence bounds never dip below zero.
* **Algorithm Selection:** P2Predict chose **XGBoost**. Because tree models require single numerical inputs, categorical features (like material and head style) are handled via Target Encoding. This maps each category to its smoothed mean price, giving the tree a meaningful hierarchy to split on.
* **Outliers:** We use `--feature-outliers warn`. If we dropped outliers, we would destroy near-constant columns entirely.

### Model Performance

| Metric | Result | What it means |
|---|---|---|
| **Log-price R²** | **0.322** | Objectively low, but sits at roughly 54% of the absolute theoretical ceiling (~0.60) allowed by the data. |
| **Median % Error** | **79.7%** | The typical prediction is off by roughly 80%. Do not use this for point-pricing. |
| **Algorithm** | **XGBoost** | Required log-target transformation due to high skew. |

### Visual Quality Report

P2Predict generates a procurement-ready PDF report detailing model calibration and feature importance. *(You can download the full PDF in [`assets/model_quality_report.pdf`](assets/model_quality_report.pdf))*

**1. Calibration by Price Band:**
Notice how the median error peaks wildly on sub-$5 bolts, but stabilizes slightly in the mid-tier where more capital is actually deployed.
![Model quality report, page 2](assets/model_quality_report_page_2.png)

**2. Feature Importance:**
Unlike a clean dataset where one or two variables dominate, importance here is smeared across thread series, diameter, strength, and material. This diffuse chart is the visual fingerprint of a weak, noisy dataset.
![Model quality report, page 3](assets/model_quality_report_page_3.png)

---

## Part 4: Reproducing the Results

You can reproduce this exact analysis from the command line using public data.

```bash
# 1. Download the Identification, Characteristics, and Management segment ZIPs
#    from the FLIS Electronic Reading Room (Free, no login required).
#    Save them in ~/Downloads.

cd case-studies/aerospace-fasteners

# 2. Extract and filter for FSC 5306 (Streaming extraction, disk-safe)
python fetch_data.py --src-dir ~/Downloads

# 3. Pivot characteristics, normalize pricing, and clean the dataset
python prepare_data.py

# 4. Measure the noise floor BEFORE training
python diagnose_noise.py

# 5. Train the model (Log-target on due to massive price skew)
p2predict-train \
  -i data/bolts_clean.csv \
  -t unit_price_each_usd \
  -tf "material,head_style,thread_diameter_in,length_in,thread_class,thread_series,finish,tensile_strength_psi,threads_per_inch,width_across_flats_in" \
  --log-target on \
  --outliers warn \
  --feature-outliers warn \
  --budget thorough

# 6. Generate predictions, ranges, SHAP, and what-if scenarios
python predict_examples.py

# 7. Extract dimensional sweeps and generate the PDF report
python extract_insights.py
python generate_charts.py
python generate_quality_report.py
```

## Limitations 

* **Catalog Pricing Only:** These are standard catalog unit prices, not negotiated volume contracts or bottom-up should-costs. 
* **Missing Variables:** PUB LOG provides a unit price, but does not provide the manufacturer or the volume-tier break. Adding either of these variables in a future dataset pull is the most viable path to improving the model's accuracy. 
* **The Goal is Not Accuracy:** R² 0.021 is not a model you deploy to an ERP system. This is a framework for diagnosing noisy data and stopping your engineering team from wasting weeks trying to tune a model that the data will never support. 
```
