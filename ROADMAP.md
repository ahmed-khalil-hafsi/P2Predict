# Roadmap — getting to 8/10 robustness

The current release is a confident 7/10: CV-based model selection, log-target handling, outlier policies, time-aware CV, a 50-test suite running in CI. The items below are what flip the should-cost positioning from credible to compelling. None are huge — they're roughly half a day each.

Ordered by impact for the "engineering ↔ procurement design-review" use case.

---

## 1. SHAP — per-prediction explanations

**Why it matters more than anything else here.** The use case is "should we keep this feature given the cost?" Without per-prediction attribution, the answer is a black-box number that gets dismissed. With it: *"this part is predicted at $1.32 because Weight contributes +$0.40, Region=EU contributes +$0.32, and Supplier choice contributes +$0.18."* That's what unlocks the cross-functional conversation.

**Scope**
- Add `shap` to dependencies (works with all three model families: linear via `LinearExplainer`, trees via `TreeExplainer`).
- Compute explanations in `p2predict.py` and add a `--explain` flag that prints the per-feature contribution table alongside the prediction.
- Save a small background-data sample with the model (for KernelExplainer fallback if needed).

**Acceptance**
- `p2predict.py -m M -p "Weight:15,..." --explain` prints a table of (feature, contribution_to_price) summing to the predicted value minus the baseline.
- Works for ridge, RF, and XGBoost models.
- Works for log-target models (explanation in price space, not log space).

---

## 2. Prediction intervals

**Why it matters.** "$1.32" reads as a guess; "$1.32 (90% CI: $1.10–$1.55)" reads as analysis. Engineering reviews dismiss the former and engage with the latter.

**Scope**
- For tree models: quantile regression via XGBoost's `quantile` objective, or bootstrap intervals from the RF.
- For Ridge: residual-based intervals from the training distribution.
- Add a `--interval` flag (default `90`) to `p2predict.py` and surface low/high alongside the point estimate.

**Acceptance**
- Predictions include `prediction_low` and `prediction_high` columns in batch mode.
- Inline mode prints both with the point estimate.
- Width of intervals is sensible (a quick sanity test: ~90% of held-out test points fall inside the 90% interval).

---

## 3. What-if / counterfactual mode

**Why it matters.** "What if we change the region from CN to EU?" is the question design reviews want to ask. Today users have to re-run the CLI twice and eyeball the difference. A dedicated mode that takes a base set of features and a list of changes, and prints a delta table, is the killer feature for the should-cost discussion.

**Scope**
- New CLI subcommand or flag: `p2predict.py whatif -m M --base "Weight:15,Region:CN,..." --change "Region=EU,Supplier=B"`.
- Returns: base prediction, counterfactual prediction, delta, and (if SHAP is implemented) the per-feature contribution of each change.

**Acceptance**
- Works with multiple simultaneous changes.
- Output is a clean table, copy-pasteable into review meeting notes.

---

## 4. Feature-side outlier detection

**Why it matters.** v0.3 catches outliers in the target column. But a misrecorded `Weight=100000` in a feature column silently distorts training and there's no warning. Procurement data has both kinds.

**Scope**
- Extend `modules/outliers.py` with `detect_feature_outliers(df, numerical_cols)` using the same IQR rule per column.
- Report counts per column during training; reuse the existing `--outliers` policy semantics.
- Add a `--feature-outliers {keep,warn,drop}` flag (separate from target outliers — we don't want to winsorize features blindly).

**Acceptance**
- Training output shows a per-column outlier summary for numerical features.
- New tests in `tests/test_outliers.py` covering the multi-column case.

---

## 5. `pyproject.toml` and pip-installable

**Why it matters.** Not robustness per se, but adoption friction. "Clone the repo and run `python3 p2predict_train.py`" is a steeper ask than `pip install p2predict && p2predict-train ...`. Also unlocks an MCP server (see below) being a normal dependency.

**Scope**
- Add `pyproject.toml` with project metadata.
- Define console_scripts entry points: `p2predict-train = p2predict_train:train`, `p2predict = p2predict:main`.
- Move modules under a `p2predict/` package so it's importable.
- Add a release workflow that publishes to PyPI on tag.

**Acceptance**
- `pip install -e .` works locally.
- `p2predict-train --help` and `p2predict --help` work from anywhere in the venv.
- CI publishes on `v*` tags.

---

## What I deliberately left out

- **LightGBM / CatBoost** — adds maintenance for a marginal accuracy gain. RF + XGBoost already cover the tree-ensemble space well.
- **Drift detection / model monitoring** — that's 9/10 territory. Production-grade.
- **Multi-format input (Parquet, Excel, DB)** — UX, not robustness. CSV covers 95% of procurement use today.
- **Quantile regression for Ridge** — too involved relative to the bootstrap fallback.
