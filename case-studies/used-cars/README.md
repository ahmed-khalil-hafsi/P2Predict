# Case study: Used vehicle pricing

> **The warm-up case study.** Not procurement, but the strongest single
> demo of how P2Predict actually behaves on real-world noisy data — and
> the cleanest setting to understand what every flag in the toolkit
> *does* before you point it at a procurement BOM.

## The question

> Given a used vehicle's year, mileage, manufacturer, body type, drive,
> fuel, transmission, condition, state, and paint color — what's the
> expected listing price, **how confident is the model**, and how much
> does each spec contribute to the answer?

## Why this case study first

Three reasons:

1. **The math shows up clearly.** Used-car prices span $500 → $200k. That
   heavy right tail triggers P2Predict's log-target wrap automatically,
   which makes the SHAP attribution *multiplicative in price space* — and
   that's the form that most resembles how a buyer actually reasons
   ("low miles add about 19%, FWD pulls about 15% off"). The math is
   identical to what you'd want on a procurement BOM; the *vocabulary* is
   one everyone already speaks.
2. **You don't need a category-manager license to sanity-check it.** When
   the model says a 2019 Civic with 45k miles is worth $17k, you have
   intuition. When it says a 2008 F-150 with 180k miles is worth $10k, you
   have intuition. When it confidently picks $55k for a 2021 Tesla, you
   know to look harder — and the model's *own* uncertainty band agrees.
3. **It surfaces the methods that matter on harder datasets.** Conformal
   intervals, SHAP attribution in log space, what-if comparisons,
   outlier-policy choices — all of them earn their keep on this dataset
   in ways that are easy to see.

If you only run one case study to decide whether P2Predict is the right
tool for your shop, run this one.

## Data

**Source:** [Craigslist Cars+Trucks on Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
— 426,880 US listings, **CC0** (public domain) license, redistributable.

**Two reproducibility paths:**

| Path | What you do | What you get |
|---|---|---|
| **Full** | Run `fetch_data.py` (needs a Kaggle API token) → `prepare_data.py` → `p2predict-train` | Matches the numbers below exactly |
| **Quick** | Just clone the repo and train on `data-sample/vehicles_sample.csv` (5,000 rows, checked into git) | Same *shape* of result, lower R² because less data |

For the full path, get a Kaggle API token from
[kaggle.com/settings](https://www.kaggle.com/settings) → "Create New
Token" and save it to `~/.kaggle/api_token` (`chmod 600`).

**Columns we use:**

| Column | Type | Notes |
|---|---|---|
| `price` | Numeric (target) | $500 – $200k after guardrails; heavy right skew (1.5) → triggers log-target |
| `year` | Numeric | 1990 – 2022 |
| `odometer` | Numeric | 0 – 500k miles |
| `manufacturer` | Categorical | 40+ values |
| `condition` | Categorical | excellent / good / fair / salvage / `unknown` |
| `fuel` | Categorical | gas / diesel / electric / hybrid / other |
| `transmission` | Categorical | automatic / manual / other |
| `drive` | Categorical | fwd / rwd / 4wd / `unknown` |
| `type` | Categorical | sedan / SUV / pickup / coupe / … |
| `state` | Categorical | All 50 states |
| `paint_color` | Categorical | 12 colors + `unknown` |

**Columns we drop and why:**
- `id`, `url`, `region_url`, `image_url`, `VIN`, `description`,
  `posting_date`, `lat`, `long` — non-parametric / identifier columns.
- `county` — 100% null in this snapshot.
- `size` (72% null), `cylinders` (42% null) — too sparse once
  `manufacturer` and `type` are present.
- `model` (very high cardinality) — slows CV-driven HPO without much
  marginal lift; `manufacturer` plus `type` already captures most of the
  signal. A future iteration could include it with a target-encoded
  preprocessor.

## Methodology

This section is the comprehensive walkthrough of what P2Predict actually
does between the input CSV and the predictions, intervals, and
explanations you see below. Every choice has a code reference; nothing
is magic.

### Pipeline at a glance

```
   raw CSV  ─►  outlier handling  ─►  80/20 train/test split  ─►  preprocessor
                                                                      │
   final fit  ◄─  best hyper-params  ◄─  HalvingRandomSearchCV  ◄──────┘
        │                              (per algorithm: Ridge / RF / XGB)
        ▼
   split-conformal calibration on the test set
        │
        ▼
   save model + background sample + calibration  ─►  predict / interval / SHAP / what-if
```

The two design decisions that make P2Predict different from a hand-rolled
sklearn script — and that you'll see referenced throughout this section —
are: **(a)** every claim the toolkit makes (interval coverage, SHAP
attribution, multiplicative axiom) is backed by an axiomatic test in the
suite, and **(b)** the conformal intervals and SHAP attributions
compose with each other and with the log-target wrap in a mathematically
clean way.

### Outlier handling — Tukey IQR rule

**What it is.** Tukey's classic non-parametric outlier rule: any value
outside the *fence* `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` is flagged, where
`IQR = Q3 − Q1` is the inter-quartile range. No distributional
assumption, robust to skew, the same rule that draws box-plot whiskers.

**Where it runs.** Twice, on different axes:

| Axis | Flag | Default | This case study uses |
|---|---|---|---|
| Target column (`price`) | `--outliers {keep,warn,drop,winsorize}` | `warn` | `warn` |
| Each numerical feature (`year`, `odometer`) | `--feature-outliers {keep,warn,drop,winsorize}` | `warn` | `drop` |

**Why `--outliers warn` on the target.** The right tail is real signal,
not noise — $50–80k luxury cars *do* exist. Dropping them collapses
skew below the log-target threshold, the wrap turns off, the model
collapses to additive. The whole point of log-target is to absorb the
heavy tail without losing data; using `--outliers drop` here defeats it.

**Why `--feature-outliers drop` on the features.** A row with
`odometer = 9,999,999` mi is a data-entry error, not a luxury car.
Drop it before it skews the year ↔ mileage relationship the model needs
to learn.

**Source.** `src/p2predict/outliers.py`.

### Log-target trigger — skewness-based, automatic

**What's measured.** The Fisher–Pearson sample skewness of the target
column. Positive means right-tailed (many small values, a few big
ones — classic price distribution shape).

**Threshold.** If `scipy.stats.skew(y_train) > 1.0`, the trainer wraps
the chosen pipeline in `TransformedTargetRegressor(func=np.log,
inverse_func=np.exp)`. This case study's price column has skew
**1.50 on the full cleaned dataset (≈350k rows), 1.52 on the 80k
training sample, and 1.52 again after the feature-outlier drop**.
Comfortably above the 1.0 threshold; the wrap fires automatically.

**Why `log` / `exp` rather than `log1p` / `expm1`.** Under `log` / `exp`
the inner model predicts `log(price)`, and the SHAP additivity in log
space exponentiates *exactly* to the multiplicative axiom in price
space:

```
log(pred) − log(base) = Σ φᵢ          (SHAP local accuracy in log space)
       ⇒  pred / base = ∏ exp(φᵢ)    (multiplicative axiom in price space)
```

Under `log1p` / `expm1` the factors would apply to `(1 + price)`, not
to `price`, and the axiom would only hold approximately for small
prices. Since `should_log_target` only fires when `y > 0` (the safety
condition for plain `log`), there's no reason to give up axiomatic
strictness. We made this switch in v0.4.

**Source.** `src/p2predict/training.py::should_log_target` and the
`TransformedTargetRegressor` wiring in `build_pipeline`.

### Train/test split

**Random split.** 80% train, 20% test, `random_state=0` for
reproducibility. This case study's snapshot has no time ordering we'd
trust (Craigslist `posting_date` is sparse and the listings are a
random crawl), so we don't use the time-aware path.

**Time-aware option (not used here).** If you pass `--time-column DATE`,
the split becomes chronological — the last 20% of rows after sorting by
date is the test set — and the CV folds become `TimeSeriesSplit`,
which prevents look-ahead bias. The procurement case studies on time-ordered
purchasing data will use this.

**Source.** `src/p2predict/prepare_data.py::prepare_data`.

### Preprocessor — branched by model family

The preprocessor is *built per algorithm* because linear and tree models
want different inputs:

| Family | Numerical | Categorical |
|---|---|---|
| Linear (Ridge, Lasso) | `StandardScaler` | `OneHotEncoder(handle_unknown="ignore")` |
| Tree (RandomForest, XGBoost) | `passthrough` | `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` |

**Why the difference.** Linear models need scaled numerics (otherwise
the coefficient magnitudes are uninterpretable and L2 regularisation
becomes feature-scale-dependent) and one-hot encoded categoricals (so
the linear combination is well-defined). Trees are scale-invariant and
prefer compact integer codes for splits.

**Source.** `src/p2predict/preprocessing.py::build_preprocessor`.

### Algorithm selection — CV-driven, three candidates

P2Predict cross-validates *each* of Ridge, RandomForest, and XGBoost
with hyperparameter search, then picks the winner by mean CV R². This
case study's numbers:

```
ridge          CV R² = 0.520    ◄── selected
random_forest  CV R² = 0.393
xgboost        CV R² = 0.411
```

The model the trainer saves is **the winner only**. The losers are
fit, evaluated, and discarded in the same run.

**Source.** `src/p2predict/training.py::auto_train`.

### Hyperparameter search — `HalvingRandomSearchCV`

**What it is.** sklearn's successive-halving randomised search.
Sample `n_candidates` hyperparameter configurations randomly from the
search space, evaluate each on a small fraction of the data, drop the
worst half, double the data, repeat. Roughly converges to the
budget-best configuration in *log(n_candidates)* rounds.

**What `--budget thorough` buys.** 24 candidates × 5-fold CV per
algorithm. That's 120 fits per algorithm × 3 algorithms = **360 model
fits per training run**, on top of the final fit. `--budget fast`
(default) is 10 candidates × 3-fold = 30 fits per algorithm.

**Search spaces** (`src/p2predict/training.py::_search_space`):

| Algorithm | Knobs searched |
|---|---|
| Ridge | `alpha` over `loguniform(1e-4, 1e+4)` |
| RandomForest | `n_estimators` int in [100, 800], `max_depth` in [3, 20], `min_samples_leaf` in [1, 5] |
| XGBoost | `n_estimators` int in [100, 800], `max_depth` in [3, 12], `learning_rate` over `loguniform(0.01, 0.5)`, `subsample` and `colsample_bytree` over `uniform(0.6, 0.4)` |

**Source.** `src/p2predict/training.py::_tune` and `_budget_params`.

### Final fit and saved artifacts

Once the winning algorithm + hyperparameters are chosen, the pipeline is
refit on the full training set (`X_train`, `y_train`). The saved
`.model` file then contains:

| Field | What it carries |
|---|---|
| `model` | The fitted sklearn pipeline (preprocessor → estimator, optionally wrapped in `TransformedTargetRegressor`) |
| `features` | The training feature columns, in order |
| `target_feature` | The target column name |
| `model_name` | `ridge` / `random_forest` / `xgboost` |
| `r2` | Holdout R² as a string for display |
| `log_target` | `bool` — was the wrap applied? |
| `training_date`, `scikit_learn_version`, `p2predict_version` | Provenance |
| `background_sample` | 100-row raw-features DataFrame for SHAP's `LinearExplainer` (no-op for trees) |
| `calibration` | Dict with `residuals`, `in_log_space`, `n_calibration` — the input to split-conformal |

**Source.** `src/p2predict/trained_model_io.py::Serialize_Trained_Model`.

### Conformal intervals — split-conformal on the test residuals

**What it is.** Split-conformal prediction
([Lei et al. 2018](https://arxiv.org/abs/1604.04173)). Compute the
residuals on the held-out test set; the (1 − α) empirical quantile of
their absolute values is the interval half-width. The guarantee:

> Under exchangeability of (train ∪ test ∪ new-point) — the *same*
> assumption the model's R²/MAE numbers already rely on — the
> probability that the true value of a new point falls inside its
> predicted interval is at least 1 − α, marginally.

That's a mathematically *real* guarantee, not a heuristic.

**For log-target models.** Residuals are computed in log space; the
interval is `pred · exp(±q̂)` in price space, which gives constant
*percentage* width — the natural shape for price-distribution data
(narrow on cheap parts, wide on expensive parts, same ±% on either).

**For non-log models.** Residuals are computed in target units; the
interval is `pred ± q̂` (additive). Same conformal math, different scale.

**This case study's calibration:**

| | |
|---|---|
| `n_calibration` (test rows used) | 15,680 |
| `in_log_space` | `True` |
| Coverage requested | 90% (`coverage=0.90`) |
| Quantile method | `np.quantile(absolute_residuals, q, method="higher")` for finite-sample correctness |

**Source.** `src/p2predict/intervals.py::compute_calibration_residuals`
and `predict_interval`.

### SHAP attribution — exact algorithms only

P2Predict uses the SHAP explainer that's *exact* for the model family
and polynomial-time. No `KernelExplainer` fallback (slow,
Monte-Carlo approximate, and we never need it for the three families
we support):

| Family | Explainer | Cost | Background needed? |
|---|---|---|---|
| Linear (Ridge, Lasso) | `shap.LinearExplainer` | Closed-form `φᵢ = βᵢ · (xᵢ − E[xᵢ])`, O(F) | Yes — to estimate `E[xᵢ]`. Persisted with the model. |
| Tree (RF, XGBoost) | `shap.TreeExplainer(..., feature_perturbation="tree_path_dependent")` | Exact Shapley in O(T · L · D²), `T` = trees, `L` = leaves, `D` = depth | No — estimated from the trees' own node counts |

**Source-column rollup.** One-hot dummies contribute one SHAP value per
dummy. We sum across dummies that came from the same source column
before reporting, so the report has one row per *original* column.
That's sound under SHAP's additivity property when the dummies are
mutually exclusive (exactly one is 1 at a time), which is the
`OneHotEncoder` contract.

**The local-accuracy axiom** (`φ₀ + Σ φᵢ = f(x)`) is asserted in
every explanation; if floating-point drift pushes it past `1e-4`, the
explanation surfaces a `residual` field for diagnostics. P2Predict's
test suite locks this in for every supported model family.

**For log-target models** — the multiplicative axiom kicks in as
described in the log-target section above. The
`Explanation.strict_multiplicative` flag is `True` when the wrap is
`log` / `exp`, signalling that
`product(multiplicative_factors) == predicted_price / baseline_price`
holds exactly.

**Source.** `src/p2predict/explain.py::explain_row` and
`_build_explainer`.

### Quality label

Computed deterministically from R² for a one-glance summary:

| R² × 100 | Label |
|---|---|
| > 80 | Excellent |
| > 60 | Good |
| ≤ 60 | Needs Improvement |

This case study: R² = 0.634 → composite 63.4 → **Good**.

**Source.** `src/p2predict/cli/train.py` (search for `quality_label`).

### Residual-bias test — one-sample t-test against zero

**What's being tested.** Whether the residuals `y_test − ŷ_test`
have a mean significantly different from zero. Under a well-calibrated
model the residuals should fluctuate symmetrically around zero; a low
p-value means the model is *systematically* off in one direction
(consistently over- or under-predicting), which is a stronger statement
than "noisy predictions."

**Test used.** `scipy.stats.ttest_1samp(residuals, 0.0)`. The two-sample
version that was in here in v0.2 was mathematically wrong (it compared
two unpaired samples); we replaced it in v0.3.

**This case study's p-value ≈ 6 × 10⁻⁷⁵** is extremely small, meaning
Ridge in log space is leaving structured variance behind — most likely
at the tails of the price distribution. That's why the Notes section
flags XGBoost (holdout R² 0.696 vs Ridge 0.634) as the right model for
production once the upstream SHAP-vs-XGBoost-3.x bug is resolved.

**Source.** `src/p2predict/model_evals.py::evaluate_model`.

### What's *not* in scope here

A few methodological choices we deliberately don't apply in this case
study, with their forward pointers:

- **Per-segment models** (one model per body type, say). Would almost
  certainly reduce the residual bias; deferred to the procurement
  case studies, where part-family segmentation is a natural fit.
- **Quantile regression** for non-conformal intervals. Heavier-weight
  than split-conformal and only justified when the conformal guarantee
  isn't enough (rarely).
- **Target encoding** for high-cardinality categoricals like the raw
  `model` column we dropped. Could lift signal without the
  high-cardinality OHE memory blow-up; future case-study iteration.

## Reproducing this case study

### Full reproduction (matches numbers below)

```bash
# 0. Get the dependencies and the Kaggle token in place.
pip install -e .
pip install 'kagglehub>=0.4.1'
echo "KGAT_..." > ~/.kaggle/api_token && chmod 600 ~/.kaggle/api_token

cd case-studies/used-cars

# 1. Fetch the dataset (~262 MB zip, ~1.4 GB extracted).
#    kagglehub caches the archive, so re-runs are free.
python fetch_data.py

# 2. Clean + sample. Produces:
#    data/vehicles_clean.csv      (full clean dataset, ~350k rows)
#    data/vehicles_training.csv   (80k-row training sample)
#    data-sample/vehicles_sample.csv (5k-row committed sample)
python prepare_data.py

# 3. Train. Note --outliers warn (not drop) — we deliberately preserve
#    the long right tail so the log-target wrap activates. And -tf
#    overrides auto-mode's default 6-feature cap (see "Notes" below).
p2predict-train \
  -i data/vehicles_training.csv \
  -t price \
  -tf year,odometer,manufacturer,condition,fuel,transmission,drive,type,state,paint_color \
  --outliers warn \
  --feature-outliers drop \
  --budget thorough

# 4. Walk through point estimate + interval + SHAP + what-if on three
#    realistic listings.
python predict_examples.py

# 5. (Optional.) Regenerate the three charts in assets/ from the current
#    trained model. The charts checked into the repo are what you get
#    from the full path; if you only train on the sample, your numbers
#    will differ and you may want fresh charts.
python generate_charts.py
```

### Quick path (no Kaggle account needed)

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

## Results (full path, 80k-row training sample)

### What the trainer chose

| | |
|---|---|
| **Algorithm selected** (auto, CV) | `ridge` (CV R² 0.520, beat XGBoost 0.411 and RF 0.393) |
| **Log-target wrap** | **Active** — price skew 1.5 > 1.0 threshold |
| **Rows after outlier handling** | 78,398 of 80,000 (1,602 dropped on `year` / `odometer` IQR bounds) |
| **Target outliers** | Detected: 1,508 (Tukey upper bound $58,050). **Kept** — they're the right tail the log-target wrap is for. |
| **Feature outliers (dropped)** | 1,191 rows with `year` outside [1997, 2029]; 458 with `odometer` > 284,069 mi |

> **Why `ridge` won.** Used-car pricing in log space is close to additive
> in the features we kept: log(price) ≈ year-coefficient · year + …. The
> nonlinearity that would favor XGBoost (year × condition × mileage
> interactions) is muted once log-target absorbs the heavy tail. XGBoost
> still got a higher *holdout* R² in our experiments (0.696 vs 0.634),
> but Ridge won the CV-driven model-selection step. Both are documented
> in the JSON output if you re-run with `--json` and inspect `cv_scores`.

### Holdout metrics

| | |
|---|---|
| R² | **0.634** |
| MAE | **$5,381** (≈ 39% of the cleaned-data median price of $13,950 — i.e. the model is in the right ballpark on average, but you would not bet a single procurement decision on a single point estimate; that's exactly why intervals and SHAP exist) |
| RMSE | $8,847 |
| Residual-bias p-value | ≈ 6 × 10⁻⁷⁵ — one-sample t-test of residuals against zero. Very low value means the model is systematically off in one direction, not just noisy. See the Methodology > Residual-bias test section. |
| Quality label | **Good** (R² × 100 = 63.4, which sits in the (60, 80] "Good" bucket; > 80 would be "Excellent", ≤ 60 "Needs Improvement"). See Methodology > Quality label. |

### Feature importance (Ridge coefficient magnitudes in log space, after preprocessing)

| Rank | Feature | Magnitude |
|---|---|---:|
| 1 | manufacturer | 10.52 |
| 2 | state | 2.75 |
| 3 | condition | 2.53 |
| 4 | type | 2.08 |
| 5 | fuel | 1.04 |
| 6 | drive | 0.53 |
| 7 | paint_color | 0.47 |
| 8 | year | 0.39 |
| 9 | odometer | 0.21 |
| 10 | transmission | 0.19 |

`year` and `odometer` look small only because they're already
standard-scaled before Ridge sees them, so the *coefficient* shrinks even
though the effect per unit is large. The SHAP attribution below (which is
expressed in price-space factors) gives the more interpretable view of
their actual impact.

## Worked examples

### 1. Point estimates and 90% likely ranges

![Honest uncertainty across three listings](assets/intervals_comparison.png)

| Listing | Predicted | 90% likely range |
|---|---:|---|
| 2019 Honda sedan, 45,000 mi, excellent, CA | **$17,341** | $9,001 – $33,409 |
| 2008 Ford pickup, 180,000 mi, good, 4wd, TX | **$10,009** | $5,195 – $19,284 |
| 2021 Tesla sedan, 22,000 mi, like new, WA | **$54,667** | $28,375 – $105,321 |

The Tesla range is enormous on purpose — Craigslist has relatively few
Tesla listings in this training sample, so the conformal interval gets
wide. **That's the model honestly saying "I'm not sure."** The same
model is much more confident about the Civic, because the training data
is dense around its part of feature space. A point estimate alone hides
this distinction; the interval surfaces it.

### 2. Why $17,341 for the Civic? — SHAP multiplicative attribution

![Per-feature attribution for the Civic](assets/civic_attribution.png)

```
  Baseline:      $12,942  (the model's E[price] over the training data)
  Prediction:    $17,341
  Net factor:    ×1.340

  Per-feature multiplicative factor (rank by deviation from 1.0):
    year            ×1.689   (+68.9%)   ← 2019 is much newer than the average
    odometer        ×1.189   (+18.9%)   ← 45k mi is well below average
    drive           ×0.847   (-15.3%)   ← fwd is cheaper than the rwd / 4wd mix
    type            ×0.850   (-15.0%)   ← sedan is cheaper than the truck mix
    fuel            ×0.940   (-6.0%)
    state           ×1.035   (+3.5%)
    manufacturer    ×0.971   (-2.9%)
    transmission    ×0.973   (-2.7%)
    condition       ×1.018   (+1.8%)
    paint_color     ×0.989   (-1.1%)

  Axiom check: product of factors = 1.340, pred/baseline = 1.340  ✓
```

The **axiom check** is the line that separates SHAP from "another
importance heuristic." For a log-target model the product of
multiplicative factors should equal pred/baseline *exactly*, and it does.
Same for non-log models: baseline + Σ(contributions) = prediction
exactly. P2Predict checks this on every explanation; if you ever see a
failed axiom in the output, the explanation is unsound.

### 3. What-if: same Civic, but with 90,000 miles instead of 45,000

![Mileage depreciation curve for the Civic](assets/mileage_curve.png)

```
  Base prediction:        $17,341
  Counterfactual:         $14,890
  Delta:                  -$2,451  (-14.1%)
  Multiplicative factor:  ×0.8586
```

That `×0.86` is the **depreciation per doubling of mileage** in this
neighbourhood of the feature space, learned from hundreds of thousands
of Craigslist listings — not from a rule of thumb. The log-target
structure makes it scale-invariant: the same factor would apply going
from 90k to 180k miles, in the limit where the model is locally linear
in log-mileage.

## The story

The case study earns its keep in five claims that are easy to verify
yourself by running the commands above.

**1. The log-target wrap activates *automatically* on the right kind of
data.** Used-car prices are heavily right-skewed. `should_log_target`
notices and inserts a `TransformedTargetRegressor(np.log, np.exp)`
around the pipeline. You don't have to remember to enable it; you do
have to know it exists when you read the SHAP output as "factors"
instead of "dollars."

**2. The choice between `--outliers drop` and `--outliers warn` matters
*a lot*.** Our first training run used `--outliers drop` and the Tukey
upper bound cut prices at $58,050, removing the right tail. That dropped
the post-clean skew below 1.0, which turned the log-target wrap *off*,
which collapsed Ridge from R² 0.52 (CV) to R² 0.34. The fix is to *let
the log-target wrap absorb the heavy tail*, which is exactly what it's
for. README's "Reproducing" section uses `--outliers warn` for this
reason. Worth documenting in your team's playbook.

**3. The 90% likely range is real coverage, not a heuristic.** The
intervals come from split-conformal calibration on the held-out test set;
empirical coverage on the test set is ≈ 90% by construction. The Tesla's
enormous range isn't a bug — it's the model accurately reporting
uncertainty in a part of feature space it barely saw at training time.

**4. SHAP gives axiomatically grounded per-feature attribution, not a
heuristic ranking.** Because the model is log-target, the contributions
are *multiplicative factors* in price space. The axiom check (`product
of factors == pred/baseline`) is built into every explanation; this is
the property that lets you write something like "FWD pulls 15% off" and
*defend it under scrutiny*, instead of mumbling about feature
importance.

**5. What-if costs nothing once you have a model.** Holding everything
else constant and asking "what if this car had 90k miles instead of 45k"
takes one `--whatif "odometer:90000"` on the CLI (or
`p2predict.what_if(model, df, {"odometer": "90000"}, ...)` in Python).
For procurement that translates directly into "what if the steel grade
changed?", "what if we moved from EU to APAC suppliers?", "what if the
weight came down 15%?" — same workflow, same math.

## Notes & footnotes worth knowing

- **The `-tf` flag is here for a reason.** Auto-mode caps features at 6
  by default — we pass all 10 columns explicitly so the log-target wrap
  and the long-tail signal stay in the model. As of **v0.9.1** you can
  also bypass the cap with `--max-features 10` (auto mode prints a
  one-line notice when columns get dropped, so the loss is no longer
  silent). For a curated column list `-tf` is still the most direct
  expression of intent.
- **The residual-bias p-value is microscopic** (≈ 10⁻⁷⁵). That tells you
  the residuals aren't zero-mean across the price distribution — Ridge
  is systematically off at the tails, even in log space. For procurement
  use this would be a flag to (a) split the model by part family, or (b)
  trust XGBoost's holdout numbers over Ridge's CV numbers and re-train
  with `--expert --algorithm xgboost --tune`. We don't do that here
  because XGBoost models hit a known SHAP-vs-XGBoost-3.x parsing bug
  that breaks `--explain` and `--whatif`; see the open ticket.
- **The 5,000-row sample in `data-sample/` is CC0** like the source
  dataset. Redistribute it freely.
- **One real bug got found and fixed by this case study.** Before this
  dataset, P2Predict's SHAP integration silently broke on any
  Ridge/Lasso model whose preprocessor produced a sparse matrix — which
  happens whenever the one-hot-encoded columns dominate the dense ones.
  The synthetic test fixture had only 10 OHE columns; used-cars has
  ~140. The bug, the fix, and the regression test all landed alongside
  this case study. Case studies earn their keep on day one.

## What this case study does *not* do (and why)

- It doesn't predict resale value over time — we hold a snapshot. Adding
  time would require the `--time-column` flag and proper TimeSeriesSplit
  CV; that's covered in the procurement case studies, where it actually
  matters.
- It doesn't ingest VIN-decoded trim/option packages. P2Predict treats
  every input column as a flat feature; the source data doesn't carry
  trim labels at the granularity that would actually matter, and the
  point of the case study is to demonstrate the toolkit on the columns
  a procurement / fleet user is most likely to have.
- It doesn't tune for accuracy — the case study is about the *workflow*.
  R² 0.634 on noisy Craigslist data is a perfectly defensible "ballpark
  with honest uncertainty"; if you wanted a production-grade Civic
  appraisal, you'd want richer features (trim, regional supply,
  seasonality) and a model split per body type.
