# Case Study: Used Vehicle Pricing

Used car prices are the perfect sandbox to see how P2Predict handles real, noisy data. 

Before pointing the toolkit at a complex procurement BOM, it helps to look at a dataset where everyone already has domain expertise. You know intuitively what a 2019 Civic should cost versus a 2008 Ford F-150. You can easily judge whether the model's logic makes sense.

The math running under the hood here is exactly the same as what you would run on a parts catalog—only the vocabulary is friendlier. This case study demonstrates how P2Predict handles highly skewed prices, outputs percentage-based cost drivers, and automatically scales its confidence intervals based on what it knows.

## The Pricing Question

Given a vehicle's year, mileage, manufacturer, body type, drive, fuel, transmission, condition, state, and color, what is the expected market price? How confident are we in that number? And exactly how much does each feature contribute to the final price?

---

## Part 1: Business Insights

We trained a P2Predict model on roughly 78,000 used-vehicle listings from Craigslist. The model achieved a "Good" rating with an R² of 0.781 and a median error of ~14%. 

Here is what the analysis revealed, and how these findings directly translate to procurement.

### The Executive Summary

When we isolate one feature at a time, the model reveals the hidden rules of the market:

1. **Brand alone creates a 2.7x swing.** Holding every other spec equal, a baseline vehicle is worth $8,028 as a Kia and $21,837 as a Porsche.
   * *Procurement Parallel:* P2Predict can quantify exact supplier and brand premiums. It separates "the supplier's premium" from "the expensive parts that supplier happens to make."
2. **Quality ratings have cliffs, not gradients.** The market values "Excellent," "Like New," and "Good" conditions all within ±6% of each other. But dropping to "Fair" immediately cuts the price by 50%.
   * *Procurement Parallel:* Supplier quality ratings and certifications often behave as step-functions. P2Predict shows you where paying for the next tier up yields diminishing returns, and where dropping a tier creates a cliff.
3. **Depreciation is a percentage, not a flat dollar rate.** A common rule of thumb is that cars lose a flat "$X per mile." The model proves this is wrong. Depreciation scales with the car's value: losing $425 per 10k miles when new, but only $225 per 10k miles when old. 
   * *Procurement Parallel:* Continuous cost drivers (like weight or throughput) are rarely linear. Linear cost models routinely over-charge at the high end and under-charge at the low end.
4. **Geography is a minor lever.** Holding the car constant, moving from the cheapest state (CT) to the most expensive (ND) only shifts the price by 1.34x. 
   * *Procurement Parallel:* While regional pricing matters, it is often a much smaller lever than supplier choice or spec adjustments.
5. **Beware of hidden proxies.** If you switch a baseline sedan from "Gas" to "Diesel", the model spikes the price by +93%. Why? Because diesel engines are rare in sedans but common in expensive heavy-duty trucks. The model learned "Diesel = probably a truck."
   * *Procurement Parallel:* This is how spec-interaction artifacts sneak into pricing models. SHAP exposes this logic immediately so you don't blindly trust a black-box estimate.

### Worked Examples: Banded Confidence Intervals

The model doesn't just guess a price; it tells you how confident it is. Notice how the confidence bands change depending on the vehicle:

![Honest uncertainty across three listings](assets/intervals_comparison.png)

| Listing | Predicted | 90% Likely Range | Confidence Band |
|---|---:|---|---|
| 2021 Tesla sedan, 22k mi, like new | **$42,246** | $29,354 – $60,798 (2.1x) | **Tightest** |
| 2019 Honda sedan, 45k mi, excellent | **$15,671** | $8,841 – $27,776 (3.1x) | **Mid** |
| 2008 Ford pickup, 180k mi, good | **$9,319** | $4,555 – $19,062 (4.2x) | **Widest** |

Because P2Predict uses **banded conformal calibration**, it knows that late-model luxury EVs (Tesla) price very consistently. Conversely, high-mileage beaters (the 2008 Ford) are highly chaotic. Instead of slapping a generic global margin of error on every car, the model tightens the band where the data is clean and widens it where the data is noisy. 

### Why is the Civic $15,671? (Percentage-Based SHAP)

Because car prices span from $500 to $200,000, the data has a heavy "right skew." P2Predict automatically detects this and applies a **log-target transformation**. 

Because of this transformation, SHAP attributes cost drivers as **multiplicative percentages** rather than flat dollars. This is exactly how category managers naturally speak ("Low miles add about 10%").

![Per-feature attribution for the Civic](assets/civic_attribution.png)

```text
  Baseline Average Price:      $14,291  
  Prediction:                  $15,671
  Net factor:                  ×1.097 (Roughly +10% above average)

  Cost breakdown by feature:
    Year (2019 is newer than avg)       ×1.655 (+65.5%)
    Odometer (45k is below avg)         ×1.106 (+10.6%)
    Condition (Excellent)               ×0.908 ( -9.2%)
    Manufacturer (Honda)                ×0.889 (-11.1%)
    Type (Sedan is cheaper than truck)  ×0.875 (-12.5%)
    Drive (FWD is cheaper than 4WD)     ×0.863 (-13.7%)
    ...

  Axiom check: The product of these factors exactly matches the prediction.
```

### The What-If Scenario

What if we had that exact same Civic, but it had 90,000 miles instead of 45,000? 

![Mileage depreciation curve for the Civic](assets/mileage_curve.png)

```text
  Base prediction:        $15,671
  Counterfactual:         $15,217
  Expected Impact:       -$453 (-2.9%)
```

Because the Random Forest captures non-linear curves, it correctly identifies that doubling the mileage at this specific age only shaves off ~3% of the value. In procurement, you use this exact same workflow to ask: *"What if we moved from EU to APAC suppliers?"* or *"What if we relaxed the weight spec by 15%?"*

---

## Part 2: Under the Hood

For the technical team, here is how the pipeline processes the Craigslist dataset.

### Data
* **Source:** Craigslist Cars+Trucks dataset (Kaggle). CC0 Public Domain.
* **Size:** ~78k listings used for the training sample.
* **Target:** Vehicle Price.

### Pipeline & Methodology

* **Log-Target Trigger:** The target price has a heavy right skew (1.5). The system automatically wrapped the pipeline in a `TransformedTargetRegressor(log, exp)`. This is why our SHAP values are percentages and why the model handles the $100k luxury cars so well.
* **Outliers:** This dataset requires a split strategy. We set `--feature-outliers drop` to automatically delete data-entry typos (like a car with 9,999,999 miles). However, we set `--outliers warn` on the target price to *keep* the expensive luxury cars. If we dropped the expensive cars, we would destroy the right tail and the log-target wrap wouldn't trigger.
* **Algorithm Selection:** P2Predict chose **Random Forest** (CV R² = 0.749). It beat out XGBoost and Ridge. Why? Because P2Predict applies *Target Encoding* to categorical features. Instead of grouping "Tesla" and "Toyota" together alphabetically, it assigned them numerical values based on their average price, allowing the tree to instantly split premium brands from economy brands.
* **Confidence Intervals:** We used split-conformal prediction on a 20% holdout set. Because it was computed in log space, the interval gives a constant percentage width (e.g., ±20%) rather than a flat dollar amount (e.g., ±$2,000), which naturally fits price distributions.

### Model Performance

| Metric | Result | What it means |
|---|---|---|
| **Holdout R²** | **0.781** | The model explains roughly 78% of the price variation using just 10 basic features. Rated: "Good". |
| **MAE** | **$3,658** | The typical miss is roughly 22% of the $16,419 median price. |
| **Residual-bias** | **p < 0.05** | The residual bias test flagged a systematic lean. The Random Forest leaves some structured variance behind at the extreme tails (very cheap beaters and ultra-luxury cars). |

### Visual Quality Report

P2Predict generates a procurement-ready PDF report detailing model calibration and feature importance. *(You can download the full PDF in [`assets/model_quality_report.pdf`](assets/model_quality_report.pdf))*

**1. Overall Accuracy:**
The scatter plot shows the model predicting very reliably across the core $8k–$40k band, though it fans out at the extreme cheap tail.
![Model quality report, page 1](assets/model_quality_report_page_1.png)

**2. Calibration by Price Band:**
The bar chart proves the model is highly calibrated in the $30k–$38k range (6.4% error) but struggles heavily on the sub-$4k tier (44.8% error).
![Model quality report, page 2](assets/model_quality_report_page_2.png)

**3. Feature Importance:**
Year alone explains 46% of the model's decisions, followed by mileage and body type.
![Model quality report, page 3](assets/model_quality_report_page_3.png)

---

## Part 3: Reproducing the Results

You can reproduce this exact analysis from the command line using the Kaggle dataset.

### Full Path (Requires Kaggle API Token)

```bash
# 1. Install dependencies and set up the Kaggle Token
pip install -e .
pip install 'kagglehub>=0.4.1'
echo "YOUR_KAGGLE_TOKEN" > ~/.kaggle/api_token && chmod 600 ~/.kaggle/api_token

cd case-studies/used-cars

# 2. Fetch the dataset (~1.4 GB extracted)
python fetch_data.py

# 3. Clean and prepare data
python prepare_data.py

# 4. Train the model 
#    Note: We warn on target outliers to keep the right tail, but drop feature outliers (typos).
#    The --report flag generates the PDF automatically.
p2predict-train \
  -i data/vehicles_training.csv \
  -t price \
  -tf year,odometer,manufacturer,condition,fuel,transmission,drive,type,state,paint_color \
  --outliers warn \
  --feature-outliers drop \
  --budget thorough \
  --report assets/model_quality_report.pdf

# 5. Generate predictions, ranges, SHAP, and what-if scenarios
python predict_examples.py

# 6. Extract dimensional sweeps and generate charts
python extract_insights.py
python generate_charts.py
```

### Quick Path (No Kaggle Account Needed)

If you just want to test the workflow without a Kaggle account, we included a scrubbed 5,000-row sample dataset. The metrics will be rougher due to the small sample size, but the pipeline will run end-to-end.

```bash
cd case-studies/used-cars
p2predict-train \
  -i data-sample/vehicles_sample.csv \
  -t price \
  -tf year,odometer,manufacturer,condition,fuel,transmission,drive,type,state,paint_color \
  --outliers warn \
  --feature-outliers drop \
  --budget thorough
python predict_examples.py
```

## Limitations 

* **No Time-Series:** This model evaluates a single static snapshot of the market. It does not predict future resale value or account for macroeconomic inflation over time.
* **Basic Specs Only:** We treat each column as a flat feature. We did not ingest VIN-decoded trim levels or option packages, which account for a massive amount of price variation on modern vehicles. 
* **One Global Model:** We built a single model for all cars. For a production-grade pricing tool, you would split this into separate models based on body type (e.g., one model specifically tuned for pickup trucks, another for sedans).
```
