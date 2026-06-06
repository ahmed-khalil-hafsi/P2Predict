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

## ~~4. Feature-side outlier detection~~ ✅ Shipped in v0.7

`--feature-outliers {keep,warn,drop,winsorize}` flag in `p2predict_train.py`. Per-column Tukey IQR detection on the numerical features; categorical features are ignored. `drop` is row-level (any column outlier removes the row); `winsorize` is per-column independent. Default is `warn`.

Also fixed a latent bug in `detect_outliers()` while we were in there — near-constant columns like `[10]*20 + [10_000]` used to slip through silently because IQR was zero. The new behaviour treats anything not equal to the central point as an outlier in that degenerate case.

---

## ~~5. `pyproject.toml` and pip-installable~~ ✅ Shipped in v0.8

`pip install -e .` (development) or `pip install p2predict` (once published) installs the package and the `p2predict` / `p2predict-train` console scripts. `from p2predict import auto_train, explain, predict_interval, what_if, ...` is now the supported Python API surface — used by embedded apps, notebooks, and the MCP server that lands next.

Package layout moved to `src/p2predict/` (PEP 517 / 518 src-layout). CI installs via `pip install -e ".[dev]"` and runs an install-time smoke check, so the install path is validated on every push.

---

## Next: agent-first deployment surface (v1.0 — MCP server)

With v0.8 the Python API is stable. The MCP server wraps it as typed tools so AI agents (Claude, Cursor, custom procurement agents) can call P2Predict natively — `predict`, `explain`, `predict_interval`, `what_if`, `train`, `list_models` — without shelling out to the CLI. This is the surface that makes P2Predict appear as a first-class tool in agent platforms and procurement workflows where the human user never sees a terminal.

---

## What I deliberately left out

- **LightGBM / CatBoost** — adds maintenance for a marginal accuracy gain. RF + XGBoost already cover the tree-ensemble space well.
- **Drift detection / model monitoring** — that's 9/10 territory. Production-grade.
- **Multi-format input (Parquet, Excel, DB)** — UX, not robustness. CSV covers 95% of procurement use today.
- **Quantile regression for Ridge** — too involved relative to the bootstrap fallback.
