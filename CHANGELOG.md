# Changelog

All notable changes to P2Predict are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [v0.9.2] — 2026-06

### Added
- **`--report PATH` flag in `p2predict-train`.** Writes the procurement-style 3-page PDF model-quality report (provenance, performance metrics, predicted-vs-actual scatter, error-by-target-band chart, feature-importance bar chart) to PATH after training. Works in both auto and expert mode, and with or without `--interactive` — the report was previously reachable only via the expert + interactive prompt, which hid it from auto-mode users and from non-interactive callers (CI, agents, scripted runs). Surfaced by the used-cars case study, which was shelling out to `plot_results_pdf` via a separate `generate_quality_report.py` script because the CLI couldn't produce the PDF itself.
- **`report_path` field in the train `--json` response.** String when `--report` wrote a PDF, `null` otherwise — so agents calling `p2predict-train --json --report ...` know exactly where the PDF landed without parsing terminal output.
- **Integration test `test_train_auto_writes_pdf_report_when_requested`** in `tests/test_cli.py` asserts that `--report` produces a non-trivially-sized PDF on disk in auto mode and that the JSON payload's `report_path` matches.

### Changed
- The expert + interactive "Generate the model quality PDF report?" prompt still works as a fallback when `--report` is not passed. When `--report PATH` is passed in interactive mode, the prompt is skipped to avoid asking the same question twice.
- Case-study `case-studies/used-cars/README.md` reproduction recipe now uses `--report assets/model_quality_report.pdf` in the train step. The separate `generate_quality_report.py` script becomes optional (PNG-preview generation only).

### Compatibility
- No model-format change. `p2predict_version` unchanged from v0.9.

## [v0.9.1] — 2026-06

### Added
- **First end-to-end case study: used vehicle pricing.** `case-studies/used-cars/` now ships a full reproducible build on the 426k-row Craigslist Cars+Trucks dataset (CC0). `fetch_data.py` uses the new-style Kaggle API token + `kagglehub`; `prepare_data.py` cleans and samples; `predict_examples.py` walks through point estimates, 90% likely ranges, SHAP multiplicative attribution, and a what-if counterfactual on three realistic listings. The README documents both reproducibility paths (full Kaggle + 5k-row checked-in sample) and the results: Ridge wins auto-selection at CV R² 0.520 with log-target active, holdout R² 0.634, MAE $5,381.
- **`--max-features N` flag in `p2predict-train`** for auto mode. Defaults to 6 (prior behaviour). Raises the cap on how many top-ranked features auto-selection keeps — the silent 6-cap was masking real signal on wider datasets. The used-cars case study exposed this: passing the 10 curated columns lifted CV R² from 0.34 → 0.52 (and unlocked the log-target transform that the 6-cap was hiding). Expert mode is unaffected (features are picked interactively or via `-tf`).
- **Drop-notice in auto mode.** When the ranker returns more features than the cap, the CLI now prints `Auto-selected N of M features (use --max-features to override or pass -tf).` so the dropped columns aren't invisible.

### Fixed
- **SHAP `LinearExplainer` crashed on Ridge/Lasso models with high-cardinality categoricals.** When the OneHotEncoder columns dominate, `ColumnTransformer` returns a scipy sparse matrix; `np.asarray` on that matrix wraps it in a 0-d object array, breaking every downstream `len()` and indexing call inside SHAP. Surfaced by the used-cars case study (10 categoricals → ~140 OHE columns → sparse output). Fixed via a `_to_dense_2d` helper at the SHAP boundary in `explain.py`; regression test `test_ridge_explain_works_with_high_cardinality_categoricals` deliberately trips the sparse path so this can't regress.
- **SHAP + XGBoost 3.x `base_score` parse error.** XGBoost ≥ 3.0 serialises `base_score` as a stringified one-element list (e.g. `'[9.567467E0]'`); SHAP 0.49.x's `XGBTreeModelLoader` calls `float(...)` on it and raises `ValueError` ([shap/shap#4184](https://github.com/shap/shap/issues/4184)). Worked around with an idempotent monkeypatch in `_build_explainer` that coerces the field inside the UBJ payload before the loader sees it. Patch goes away once the upstream fix ships. Regression test `test_xgboost_local_accuracy` locks in the SHAP local-accuracy axiom on XGBoost.

### Compatibility
- Default behaviour unchanged: auto mode still caps at 6 unless `--max-features` is passed.
- No model-format change. `p2predict_version` unchanged from v0.9.

## [v0.9] — 2026-06

### Added
- **`--json` flag on both CLIs** for machine-readable structured output, replacing Rich-formatted tables. Designed for AI agents, scripts, and downstream tooling that need to ingest P2Predict's output without regexing terminal text. Closes ROADMAP item #1 (the agent-readiness prerequisite for the upcoming MCP server).
- **`p2predict.json_output` module** documents the stable response schema for both `predict` and `train`. Every response carries `schema_version: "1.0"` so consumers can evolve safely as fields are added.
- **JSON error path** — when `--json` is set, abort cases emit `{"error": {"code": "...", "message": "..."}}` on stdout with exit 1 instead of Rich abort messages, so an agent piping output gets a parseable failure document.
- **Composability** — `--json` works alongside `--interval`, `--explain`, and `--whatif`; each adds its block to the response without affecting the others. Interactive mode is rejected with a clear error under `--json` (it would require prompts no agent can answer).
- **15 new tests in `tests/test_json_output.py`** assert the schema shape for every documented field, on both CLIs, on success and on failure. The schema is now a tested contract, not a docstring.

### Changed
- Train CLI no longer starts the import-time Halo spinner under `--json` (sniffs `sys.argv` early so the spinner doesn't write to stdout before Click parses the flag).
- The Rich `Console` used by both CLIs is redirected to `/dev/null` under `--json` as a belt-and-suspenders measure — guards on individual `console.print` calls are the primary defence; the redirect catches anything that escapes.

### Compatibility
- `p2predict_version` bumped to `v0.9`. No persisted metadata schema change; older models load and predict unchanged.
- Default (non-JSON) output is unchanged. Existing scripts and workflows continue to work without modification.

## [v0.8] — 2026-06

### Added
- **Pip-installable as the `p2predict` package.** `pyproject.toml` with `[project.scripts]` entries registers `p2predict` and `p2predict-train` as console scripts. Closes ROADMAP item #5. Install with `pip install -e .` (development) or `pip install p2predict` (once published).
- **Public Python API.** `from p2predict import auto_train, explain, predict_interval, what_if, save_model, load_model, ...` — the same functionality the CLI exposes, callable from scripts / notebooks / agent code without shelling out. This is the surface AI agents will call once the MCP server lands in v1.0.
- **`python -m p2predict`** invokes the predict CLI without needing the console script entry point (useful for sandboxed environments or one-off use).
- **`examples/python_api.py`** — end-to-end walkthrough loading `examples/example.csv`, training, persisting, reloading, and demonstrating all four programmatic entry points.

### Changed
- **Package layout: `modules/` → `src/p2predict/`.** Standard PEP 517 / 518 src-layout. The internal module rename `modules.p2predict_feature_selection` → `p2predict.feature_selection` drops a redundant package-name prefix now that everything lives under the `p2predict` namespace. Git history is preserved via `git mv`.
- **CI workflow now installs via `pip install -e ".[dev]"`** and runs an install-time smoke check (`p2predict --help` / `p2predict-train --help`), so the install path is validated on every push, not just the test path.
- **Root scripts:** `p2predict_train.py` stays as a thin shim that delegates to `p2predict.cli.train`. The old `p2predict.py` is gone — it collided with the package name. Use the `p2predict` console script or `python -m p2predict`.

### Compatibility
- `p2predict_version` bumped to `v0.8`. No model-format change. v0.7 and earlier models load and predict unchanged.
- **Breaking for direct CLI invocation:** `python3 p2predict.py ...` no longer works (the package name collided with the script). Use `p2predict ...` (after install) or `python -m p2predict ...`. `python3 p2predict_train.py ...` still works.

## [v0.7] — 2026-06

### Added
- **`--feature-outliers {keep,warn,drop,winsorize}` flag in `p2predict_train.py`** for catching outliers in the numerical *feature* columns (Tukey IQR per column). Closes the silent-data-bug gap from v0.3 where target outliers were handled but a misrecorded `Weight = 100000` would still pull the model around. Categorical features are ignored — "outlier" doesn't have a clean meaning for a discrete code.
- **`modules.outliers.apply_feature_outlier_policy()`** — public API matching `apply_outlier_policy()`. Reports per-column counts and bounds. `drop` is row-level: any row with an outlier in any numerical feature gets removed. `winsorize` is per-column: each column gets capped at its own IQR bounds independently.

### Fixed
- **`detect_outliers()` no longer misses outliers in near-constant columns.** Previously a column like `[10]*20 + [10_000]` would return zero outliers because IQR was zero. The Tukey rule degenerates in that case; the new behaviour treats anything not equal to the central point as an outlier (since by definition it sits outside the central 50%). Affects both target-side and feature-side detection.

### Compatibility
- `p2predict_version` bumped to `v0.7`. No model-format change. CLI changes are additive — existing scripts run unchanged because the new `--feature-outliers` defaults to `warn`.

## [v0.6] — 2026-06

### Added
- **`--whatif "Feature:NewValue,..." flag in `p2predict.py`** for design-review cost trade-off comparisons. Takes the base scenario from `-p` and overrides one or more features; renders a side-by-side comparison of base vs counterfactual predictions, the delta in dollars and percent (and for log-target models, the multiplicative factor), and a SHAP-attributed decomposition of where the change came from feature by feature. Composes with `--interval` (shifts in the likely range) and `--explain` (full attribution of the base prediction). Inline-only — not supported with `-i` batch mode.
- **`modules/whatif.py`** — comparison + SHAP-delta decomposition. Uses the local-accuracy axiom on the difference of two predictions: `f(x') - f(x) = Σ (φᵢ(x') - φᵢ(x))`. Features the user *didn't* change can still show non-zero contributions when there are real interactions in the model — surfaced as a single "other interaction effects" row only when material (>5% of total delta) to avoid floating-point noise.

### Notes
- For log-target models the decomposition is multiplicative in price space: `cf_price / base_price = ∏ exp(Δφᵢ)`. Per-feature factors and the interaction factor multiply to the total change ratio — locked in by the test suite.
- No model-format change; v0.5 models work with `--whatif` immediately.

## [v0.5] — 2026-06

### Added
- **`--interval N` flag in `p2predict.py`** for likely-range prediction intervals (default coverage 90). Inline mode prints a labelled low/predicted/high table with plain-English framing ("the price for about 9 in 10 similar parts falls in this range"). Batch mode adds `<target>_low` and `<target>_high` columns to the output CSV.
- **`modules/intervals.py`** — split-conformal prediction intervals calibrated on the training holdout. Coverage is mathematically guaranteed under the same exchangeability assumption the rest of the metrics already rely on; no extra calibration split required. For log-target models the calibration happens in log space, yielding multiplicative intervals in price space (constant percentage width, natural for procurement data spanning multiple orders of magnitude).
- Empirical-coverage tests at 80% / 90% / 95% target rates assert the intervals actually cover the right fraction of held-out points to within ±5pp.

### Changed
- Model metadata gains a `calibration` field carrying the holdout residuals (and an `in_log_space` flag) so the user can ask for any coverage level at predict time without retraining. v0.4 and earlier models still load and predict; `--interval` refuses to run on them with a helpful message.

### Compatibility
- `p2predict_version` bumped to `v0.5`. Pipeline / preprocessor schema unchanged.

## [v0.4] — 2026-06

### Added
- **`--explain` flag in `p2predict.py`** for per-prediction SHAP attribution. Inline mode prints a full breakdown (`baseline + Σ contributions = prediction`); batch mode adds `top1_driver` / `top2_driver` / `top3_driver` columns to the output CSV.
- **`modules/explain.py`** — exact Shapley value computation. `LinearExplainer` (closed-form) for Ridge / Lasso; `TreeExplainer` with `tree_path_dependent` (exact in $O(TLD^2)$) for Random Forest / XGBoost. No KernelExplainer fallback.
- Model files now persist a **100-row background sample of `X_train`** so `LinearExplainer` can compute $E[x_i]$ at explain time without access to the original CSV.
- Redesigned **procurement-style PDF model-quality report** (`plot_results_pdf`). Three pages: provenance + headline metrics + Predicted-vs-Actual scatter / absolute %-error distribution + median error by target-value band / top-N feature importance. Restrained navy/amber palette, consistent page chrome, one-line glossary for non-ML readers.

### Changed
- **Log-target wrap switched from `log1p / expm1` to `log / exp`.** This makes SHAP's multiplicative axiom in price space hold strictly — a per-feature multiplicative factor of 1.18 now means "this feature multiplies the predicted price by 1.18". Safe because `should_log_target()` only fires when targets are strictly positive.
- PDF report titles and axis labels now follow the user's actual `target_name` instead of being hardcoded to "Procurement Price".
- The model-quality PDF now uses the true holdout (`X_test` / `y_test`) instead of `data[target]` concatenated with predictions on the full dataset — previously the "performance" plots silently leaked training data.

### Fixed
- `plot_histograms` no longer crashes on a single-column DataFrame.
- PDF `/Info` metadata is now strictly ASCII so the target name (e.g. "Revenue") appears as plain text in file properties instead of UTF-16BE hex.

### Removed
- `plot_results_html` (was dead code; nothing called it).
- `mpld3` runtime dependency (was only used by `plot_results_html`).
- `documentation/getting_started.md` (had wrong flag names, wrong feature-separator, missing every CLI option added in v0.2–v0.4; the README has accurate equivalents).

### Compatibility
- `p2predict_version` bumped to `v0.4`. v0.2 and v0.3 models still load and predict — the new `background_sample` field is optional and absent on older models. `--explain` on an older model works for tree-family models (no background needed) and errors helpfully on linear-family models.

## [v0.3] — 2026-05

### Added
- **`pytest` suite (50 tests)** covering preprocessor, log-target detection, save/load round-trip, auto-train, outlier handling, time-aware CV, CSV sanity check, evaluation metrics, feature-selection ranking, predict CLI handling of unseen categories. CI runs them on every push.
- **CLI integration tests** via `click.testing.CliRunner` covering both `p2predict_train.py` and `p2predict.py` end-to-end.
- **Outlier handling** on the target column (Tukey IQR rule) with four policies: `warn`, `drop`, `winsorize`, `keep`. New `--outliers` CLI flag.
- **Time-aware cross-validation** via `--time-column`. Chronological train/test split + `TimeSeriesSplit` for HPO. Prevents look-ahead bias on time-ordered data. Opt-in; random k-fold remains the default.
- `ROADMAP.md` laying out the path to 8/10 robustness.

### Changed
- Abort paths now exit with non-zero status code (was `raise SystemExit` with no argument → exit 0). Makes the CLIs properly scriptable.

### Fixed
- (None directly; the v0.3 cycle focused on closing test-coverage and data-honesty gaps rather than bug fixing.)

## [v0.2] — 2026-05

### Added
- **CV-based model selection** in auto-mode. `HalvingRandomSearchCV` across Ridge, Random Forest, and XGBoost picks the best model *and* hyperparameters. Auto-mode no longer hardcoded to a default Random Forest.
- **Hyperparameter tuning in expert mode actually replaces the saved model** (previously it printed scores and threw the tuned model away).
- **Automatic log-target transform** via `TransformedTargetRegressor` when the target is strictly positive and skewed (`skew(y) > 1`).
- Tree models use `OrdinalEncoder` (fast, handles high-cardinality categoricals); linear models continue to use `OneHotEncoder + StandardScaler`.
- `--budget {fast,thorough}` flag for HPO search size.
- Evaluation now reports R², MAE, RMSE, and a residual-bias check.

### Changed
- Saved-model filename in non-interactive mode now includes a timestamp (`models/<algo>_<target>_<timestamp>.model`) instead of a random-int suffix that occasionally collided.
- README repositioned as **parametric price benchmarking for engineering and procurement design trade-offs** — explicitly disambiguated from bottom-up should-cost tools. "What it is (and isn't)" section added.

### Fixed
- Feature-importance grouping no longer mis-splits names containing underscores (e.g., `weight_g` was incorrectly grouped with `weight`).
- `evaluate_model` previously ran an independent-samples t-test on `y_test` vs `predictions`. Replaced with a residual-bias check (`ttest_1samp(residuals, 0)`).
- `find_high_variation_features` no longer divides by raw `mean()` (which is undefined for symmetric / near-zero distributions); now uses `abs(mean)` with a zero-guard.
- `check_csv_sanity` no longer reads the file three times. Vectorized NA check; warns and drops NA rows instead of aborting.
- Non-interactive auto-mode saves no longer produce `None_*.model` filenames.

### Breaking
- Pipeline structure changed (new shared preprocessor, optional `TransformedTargetRegressor` wrap). **v0.1 models will not load — retrain them.** Metadata format gains a `log_target` field.

## [v0.1] — 2024

Initial public release.
