# Roadmap — getting to 8/10 robustness

The current release is a confident 7/10: CV-based model selection, log-target handling, outlier policies, time-aware CV, a 50-test suite running in CI. The items below are what flip the should-cost positioning from credible to compelling. None are huge — they're roughly half a day each.

Ordered by impact for the "engineering ↔ procurement design-review" use case.

---

## ~~1. SHAP — per-prediction explanations~~ ✅ Shipped in v0.4

Per-prediction Shapley attributions via `--explain`. Exact algorithms only: `LinearExplainer` (closed-form) for Ridge/Lasso, `TreeExplainer` with `tree_path_dependent` (exact in O(TLD²)) for Random Forest and XGBoost. No KernelExplainer fallback.

The log-target wrap switched from `log1p/expm1` to `log/exp` so SHAP's multiplicative axiom holds strictly in price space: `pred / base = ∏ exp(φᵢ)`. Per-feature multiplicative factors are the axiomatically-clean attribution; an approximate dollar attribution is also surfaced for procurement readability, clearly labelled as approximate.

Local-accuracy axiom (`φ₀ + Σ φᵢ = f(x)`) is asserted in the test suite for every supported model family.

---

## ~~2. Prediction intervals~~ ✅ Shipped in v0.5

Likely-range intervals via `--interval N` (default 90). Split-conformal calibration on the test holdout — coverage is mathematically guaranteed under exchangeability, the same assumption the model's R²/MAE/RMSE already rely on. Empirical coverage at 80% / 90% / 95% is asserted in the test suite within ±5pp on synthetic data.

For log-target models the calibration runs in log space, yielding multiplicative intervals in price space (constant percentage width, scale-natural for procurement data). For non-log targets we use additive intervals in target units.

Deliberately model-agnostic — one code path serves Ridge, Random Forest, and XGBoost. Language is procurement-facing throughout: "likely range", "9 in 10 similar parts", no "confidence interval" or "alpha".

---

## ~~3. What-if / counterfactual mode~~ ✅ Shipped in v0.6

`p2predict.py -m M -p "..." --whatif "Region:EU,Supplier:B"`. Side-by-side base/counterfactual predictions with deltas (dollars and percent), composed with the v0.5 likely-range intervals and the v0.4 SHAP attributions. The SHAP decomposition of the delta is locked in by the test suite: per-feature contributions sum to the total delta (and for log-target models, multiplicative factors multiply to the total change ratio).

Features the user *didn't* change can still pick up SHAP attribution due to real interactions in the model. Surfaced as a single "other interaction effects" row only when material (>5% of total delta).

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
