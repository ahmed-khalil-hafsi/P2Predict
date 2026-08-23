# Changelog

All notable changes to P2Predict are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [v1.0.0] — 2026-08

### Added
- **MCP server — 10 typed tools for AI agent integration.** `pip install p2predict[mcp]` adds the `p2predict-mcp` command, a local stdio-transport MCP server that lets Claude, Cursor, and custom agents call P2Predict as typed tools. The procurement user talks to their agent; the agent calls P2Predict; the answer flows back in plain English. Tools: `list_models`, `get_model_info`, `predict`, `predict_batch`, `explain`, `predict_interval`, `what_if`, `predict_from_csv`, `train`, `generate_report`. Trained models are also exposed as MCP resources (`model://{model_id}`). All data stays on the user's machine — nothing leaves disk.
  - **`ModelRegistry`** (`p2predict.mcp.registry`) scans a models directory, lazy-loads `.model` files with an LRU cache (max 5), and exposes `ModelInfo` metadata (algorithm, target, features, feature types, categories, calibration status).
  - **`model_utils.py`** — shared helpers extracted from `cli/predict.py` so both CLI and MCP call the same code: `inner_pipeline()`, `extract_feature_info()` (enhanced to handle `TargetEncoder` pipelines alongside `OneHotEncoder`/`OrdinalEncoder`), `coerce_features()`, `interval_to_dicts()`, `explanation_to_dict()`, `whatif_to_dict()`.
  - **`train` tool** stores `holdout_y_test` and `holdout_y_pred` in the model metadata so `generate_report` can produce the PDF without re-deriving the test set.
  - **`generate_report` tool** uses `matplotlib.use("agg")` for thread-safe PDF generation via `asyncio.to_thread()`.
  - **34 integration tests** in `tests/test_mcp.py` covering all tools and error paths. Guarded by `pytest.importorskip("mcp")` for graceful skip when the extra isn't installed.
  - **CI updated** to install `.[dev,mcp]` + `pytest-asyncio` and run a `p2predict-mcp --help` smoke check.
  - **`pyproject.toml`**: `mcp = ["mcp>=1.0"]` optional dependency; `p2predict-mcp` console script entry point.
- **Banded (Mondrian) conformal calibration for likely-range intervals.** A single global conformal quantile gives every prediction the same width, letting the noisiest price segment set the width for everyone (on the fasteners catalog: one ×225 band for all parts, inflated by near-random sub-$5 bolts). When the calibration set has ≥150 points, `predict_interval` now partitions it into three bands by *predicted* value and computes a separate quantile per band, so the range width tracks where the model is actually good — and the ~90% coverage guarantee holds *within each band*, not just on average (the banding rule depends only on the model's prediction, never on calibration labels). `compute_calibration_residuals` additionally stores the calibration predictions; `IntervalResult` gains an optional `band` description; the predict `--json` interval rows gain a `band` field. Fallbacks reproduce the old behaviour bit-for-bit: model files saved before this version (no stored predictions), calibration sets under 150 points, and degenerate prediction distributions all use the single global quantile with `band: null`. Tests in `tests/test_intervals.py` (banded width tracking, per-band empirical coverage, positivity, all three fallbacks).

### Changed
- **`predict_batch` and `predict_from_csv` now reach the full breadth of the single-part tools.** Both batch tools were point-estimate only at the inline-dict level (`predict_batch`) and silently lenient at the file level (`predict_from_csv` swallowed explanation errors and dropped intervals when a model lacked calibration). They now share one contract: an opt-in `coverage` (1–99) adds a per-row likely-range (the same conformal interval, `reliability` and `say_to_user` read as `predict_interval`), and `with_explanation` adds per-row price drivers (the same `explanation` shape as `explain`). Defaults are plain point predictions on both (`coverage=None`); an explicit `coverage` on an uncalibrated model returns a clear `no_calibration` error rather than silently dropping the range, and explanation failures surface as `explain_error` instead of being swallowed. This lets an agent get ranges and drivers across many parts without a per-part fan-out or a CSV round-trip. Tests in `tests/test_mcp.py` cover both enrichments, the default point-only path, and the `bad_coverage`/`no_calibration` error paths for each tool.
- **Tree models now target-encode categoricals instead of ordinal integer codes.** `build_preprocessor` fed `RandomForest`/`XGBoost` an `OrdinalEncoder`, which assigns each category an *arbitrary alphabetical integer*. A tree can only threshold-split that code, so it lumps alphabetically-adjacent categories into the same leaf — which destroys the signal for high-cardinality *nominal* features (supplier, manufacturer, brand: the most common procurement categorical). The concrete failure, found in the used-cars case study: XGBoost priced a 2021 like-new **Tesla at ~$6k** (≈ the prediction for a Toyota — the alphabetically adjacent code) where the realistic figure is ~$40–55k. Categoricals are now passed through `TargetEncoder` (cross-fitted, smoothed mean target per category), so the code *orders by price* and a single split separates premium from commodity. Measured effect: used-cars Tesla **$6k → $42,246** (with the tightest conformal band of the three test vehicles) and holdout R² 0.634 → **0.781**; BMIC (150 parts) R² 0.512 → **0.691** because `smooth="auto"` shrinks sparse categories toward the global mean (an empirical-Bayes prior) instead of overfitting them. One numeric column per categorical, so SHAP/`--explain`, feature importances, and conformal intervals are unaffected. A most-frequent imputer precedes the encoder (TargetEncoder rejects a NaN category unseen at fit), and `_AdaptiveTargetEncoder` shrinks the cross-fitting fold count to the sample size so small datasets and HalvingRandomSearchCV's early rungs don't crash. Linear models (Ridge/Lasso) keep one-hot encoding. Regression test `test_tree_prices_sparse_premium_category_via_target_encoding` asserts a sparse premium category is priced >2.5× its commodity neighbour (ordinal collapses it toward 1×). (Root-caused from the used-cars Tesla anomaly.)

### Fixed
- **Install no longer fails on Python 3.14 (now the default download on python.org).** The `shap>=0.44,<0.50` ceiling resolved to shap 0.49.1, which ships no cp314 wheel, so `pip install "p2predict[mcp]"` on Python 3.14 fell back to building shap from source and died with `error: Microsoft Visual C++ 14.0 or greater is required` on machines without a C++ toolchain (reported on a locked-down corporate Windows box). The shap ceiling is raised to `<0.53`: pip now resolves per-interpreter — Python 3.10/3.11 keep shap 0.49.1 (unchanged), while 3.12–3.14 get shap 0.52.0's `abi3` wheel, so 3.14 installs from wheels with no compiler. Verified: full test suite (199 passed, 1 skipped) against shap 0.52.0, and the cross-platform install smoke test now spans Linux/macOS/Windows × Python 3.10/3.13/**3.14**. No code changes — the SHAP explainer path (including the XGBoost 3.x `base_score` patch) is unaffected.
- **`requires-python` capped to `>=3.10,<3.15`.** With no upper bound, an unsupported-newer Python (e.g. a future 3.15) would attempt a source build and fail cryptically; the cap makes pip refuse up front with a clear "requires a different Python" message. Classifiers now list 3.13 and 3.14.
- **HPO and algorithm selection no longer decided on a 90-row resource floor.** `_tune()` in `p2predict.training` ran `HalvingRandomSearchCV` with the default `min_resources='smallest'`, which for cv=5 regression scheduled candidate resources `10 → 30 → 90` samples *regardless of dataset size*. On the 15,197-row aerospace-fasteners training set the winner-deciding rung saw only 90 rows (`search.n_resources_ == [10, 30, 90]`, best CV score −1.18), so CV scores were meaningless and the selected algorithm flipped between identical runs (XGBoost vs random_forest, ~0.06 log-R² apart). Now passes `min_resources='exhaust'` so the final rung uses the full training set and selection is reproducible. Regression test `test_tune_decides_on_full_training_set_not_resource_floor` asserts the largest rung equals the full training size. (Found by the independent verification in PR #14.)
- **CV is now scored in log space when the log-target wrap is active.** `_tune()` hard-coded `scoring="r2"` on a `TransformedTargetRegressor`, so candidate selection happened in raw price space even when the model was trained on `log(price)`. On a 5.36-skew target this selected a model scoring log-R² −0.25 / 265% median error, where scoring in log space yields log-R² 0.337 / 80% median error. A new `log_r2_scorer` (`make_scorer` of R² on `log(y_true)` vs `log(clip(y_pred))`) is used whenever `log_target` is true, plain `"r2"` otherwise, selected via `_scoring_for(log_target)`. Regression tests `test_log_space_r2_rewards_model_good_in_log_space` and `test_scoring_for_uses_log_scorer_only_under_log_target`. (PR #14.)
- **All-NA-row dropping at CSV load no longer discards data or corrupts `--json` output.** `check_csv_sanity()` ran `df.dropna()` over *all* columns at load, silently discarding rows with NAs in columns that were not even selected as features — 48% of the fasteners catalogue (36,668 → 18,997 rows), and the dropped half became scoreable at predict time. NA handling is now surgical:
  - `check_csv_sanity()` keeps every row and only *reports* NA counts (routed to **stderr** so `p2predict-train --json` stdout stays pure JSON — the warning previously printed to stdout before the `{`, making the document unparseable).
  - The train CLI drops only rows with NA in the **target** column (after feature selection), and aborts cleanly (`all_target_na`) if that empties the data.
  - `build_preprocessor` now handles feature NAs per family so auto mode (which compares all three algorithms on the same data) works end to end: XGBoost receives NaNs natively (passthrough), while random_forest and ridge/lasso get `SimpleImputer` (median for numerics, most-frequent for categoricals) ahead of the encoder/scaler.
  - The train `--json` `input` block gains `rows_dropped_target_na` and `rows_used` fields (additive; `rows_loaded` and `rows_after_outlier_handling` keep their meaning).
  - Regression tests in `tests/test_cli.py` (`test_train_keeps_rows_with_feature_only_nas`, `test_train_drops_only_target_na_rows`, `test_train_json_stdout_is_pure_json_with_nas`, `test_train_auto_mode_handles_feature_nas_across_all_algorithms`), `tests/test_input_checks.py`, and `tests/test_preprocessing.py`. (PR #14.)

- **`TargetEncoder` cross-fitting no longer crashes on scikit-learn 1.5–1.8.** The initial `_AdaptiveTargetEncoder` passed a `KFold` splitter object as `cv`, which works on sklearn ≥ 1.9 but raises `TypeError` on 1.5–1.8 (where `cv` must be a plain `int`). Now sets `cv` as an int and seeds the fold shuffle via `random_state` instead. CI runs Python 3.10 with older sklearn, which caught this. (PR #19.)

### Compatibility
- No model-format change; `p2predict_version` unchanged. The `--json` train schema only gains fields (`rows_dropped_target_na`, `rows_used`); existing fields are unchanged. Python-API default behaviour is preserved except that the log-target CV scorer and full-resource HPO now produce better, reproducible selections.
- Supported Python is now **3.10–3.14** (`requires-python` capped to `>=3.10,<3.15`), so a newer interpreter fails fast with a clear pip message instead of a source build. No API or behaviour change; models trained on any supported Python are interchangeable.

## [v0.9.6] — 2026-08

### Added
- **`p2predict-mcp --print-config` writes your MCP client config for you.** Hand-writing the Claude Desktop block was the most error-prone step of setup, and it fails silently: the client needs an *absolute* command path (MCP clients don't inherit the shell's `PATH`, so a bare `p2predict-mcp` resolves in the terminal but not in the client), and on Windows every backslash has to be doubled because the config is JSON. The flag prints the whole block with the running install's real paths already filled in, escaped by `json.dumps` rather than by the user. It resolves the command by preferring the `p2predict-mcp` console script next to the running interpreter, falling back to `<python> -m p2predict.mcp` where the console script was never placed on disk or never landed on `PATH`. Also flags a models directory that doesn't exist yet (harmless — it's created at first train). Six tests in `tests/test_mcp.py`.

### Changed
- **Package license metadata now reflects the internal-use exception.** `LICENSE` grants any organization — explicitly including for-profit corporations — the right to use P2Predict internally for its own operations, procurement, and benchmarking at no cost, but `pyproject.toml` still declared the bare SPDX id `PolyForm-Noncommercial-1.0.0`. That is the string PyPI displays and that corporate license scanners read, so the package advertised "Noncommercial" to exactly the enterprise users the exception is meant to welcome — a policy auto-block in many compliance workflows. Changed to `LicenseRef-PolyForm-Noncommercial-1.0.0-with-Internal-Use-Exception`, a valid PEP 639 custom identifier that routes scanners to read the bundled `LICENSE` instead of matching a restrictive id. No change to the license terms themselves.
- **README and INSTALL state the internal-use grant plainly.** The license badge read "PolyForm Noncommercial"; the licensing section and FAQ said "free for internal use" without saying that for-profit companies are covered. Both now say so explicitly, and INSTALL gains a "Do I need to pay for this?" section for the non-technical reader.
- **INSTALL's client-setup step now leads with `--print-config`**, replacing the manual "copy these two paths, then substitute YOURNAME into this JSON" flow. The three hand-editing rules it required (absolute path, doubled backslashes, merge don't replace) collapse to one instruction plus a note about merging into an existing file.

### Compatibility
- No model-format change, no API change, no change to any existing flag. `--print-config` prints and exits without starting the server or touching the registry. `p2predict_version` unchanged.

## [v0.9.5] — 2026-08

### Fixed
- **Unbounded `mcp>=1.0` floor let pip install `mcp` 2.x, breaking `p2predict-mcp` on every fresh install.** The `mcp` package's 2.0 release moved `mcp.server.fastmcp`, so `from mcp.server.fastmcp import FastMCP` at `p2predict/mcp/server.py:14` raised `ModuleNotFoundError: No module named 'mcp.server.fastmcp'` and the server died at startup. Since the MCP server is the primary interface, this blocked new users at setup — `pip install "p2predict[mcp]"` produced a package that could not start. Existing installs with an already-resolved `mcp` 1.x were unaffected, and no prediction path was involved, so no numbers were ever wrong. Extra pinned to `mcp>=1.0,<2` (verified against `mcp` 1.29.0).

### Added
- **`--version` on all three entry points.** `p2predict --version`, `p2predict-train --version`, and `p2predict-mcp --version` now report the installed release. Previously there was no way to tell which version you were on without `pip show`, which made version-dependent troubleshooting advice circular.

### Changed
- **[INSTALL.md](INSTALL.md) rewritten for non-technical users**, with separate step-by-step Mac and Windows walkthroughs, an admin-rights summary, and troubleshooting organized by literal error message. The install now uses a self-contained folder (`~/P2Predict`) and absolute paths, which sidesteps three failure modes the old one-line `pip install` hit: zsh globbing unquoted `p2predict[mcp]`, PEP 668 `externally-managed-environment` on Homebrew Python, and MCP clients not inheriting the shell `PATH`.
- Quoted the `"p2predict[mcp]"` extra in [TECHNICAL.md](TECHNICAL.md) for the same zsh globbing reason.

### Compatibility
- No model-format change, no behavior change, no API change. `p2predict_version` unchanged. Anyone whose environment already resolved `mcp` 1.x sees no difference.

## [v0.9.4] — 2026-07

### Fixed
- **`matplotlib>=3.7` floor let pip resolve a pre-NumPy-2.0 wheel alongside NumPy 2.x, breaking `p2predict-train` on fresh installs.** matplotlib's compiled `_path` extension needs 3.9+ for NumPy 2.x C-API compatibility; older wheels crash on import with `AttributeError: _ARRAY_API not found`, surfacing as `ImportError: numpy.core.multiarray failed to import`. Only `p2predict-train` was affected — it imports `p2predict.plotting` → `matplotlib.pyplot`; the `p2predict` predict-side CLI never touches matplotlib. Floor raised to `matplotlib>=3.9`.

### Compatibility
- No model-format change, no behavior change for anyone already on matplotlib ≥3.9 (the common case). `p2predict_version` unchanged.

## [v0.9.3] — 2026-06

### Added
- **`--log-target {auto,on,off}` flag in `p2predict-train`.** Overrides the automatic skew-based decision in `should_log_target` (`scipy.stats.skew(y_train) > 1.0`). `auto` (default) preserves existing behaviour; `on` always wraps the target with `TransformedTargetRegressor(np.log, np.exp)` regardless of sample skew (with the same safety check — aborts cleanly if any `y_train <= 0`); `off` never wraps. The right default for any multiplicative positive-quantity target (prices, costs, weights, lead times) is `on`: SHAP attribution composes multiplicatively, conformal intervals stay strictly positive, and a 10% move means the same thing at $1 and $1000. The Battery Management ICs case study surfaced this — a 150-part dataset with skew 0.12 left auto-mode log-target off, producing a 90% conformal interval of `-$1.40` to `$4.95` on a $1.77 prediction (the negative bound is meaningless for a part price). `--log-target on` fixes that without touching the auto rule for everyone else.
- **`log_target_decision` field in the train `--json` response.** String taking one of `"auto:skew=<value>"`, `"manual:on"`, or `"manual:off"` so consumers (agents, CI dashboards, the case studies) can see *why* the wrap was applied (or not) without re-running `scipy.stats.skew` themselves. Sits alongside the existing `log_target` bool.
- **`resolve_log_target(y, mode)` helper in `p2predict.training`.** Returns `(log_target: bool, decision: str)`. Threaded through `auto_train`, `start_training`, and `hyper_parameter_tuning` via a new optional `log_target=` override parameter; the legacy callers that don't pass it fall back to the original `should_log_target` rule, so the Python API is backwards compatible.
- **Three integration tests in `tests/test_cli.py`** asserting (a) `--log-target on` activates the wrap on low-skew clean data, (b) `--log-target off` disables it on heavily-skewed log-normal data, and (c) `auto` reports the numeric skew in the JSON decision string.

### Changed
- Case-study `case-studies/battery-management-ics/README.md` reproduction recipe now uses `--log-target on` and documents the negative-interval failure mode the flag fixes.

### Compatibility
- Default behaviour unchanged: without `--log-target`, the skew rule still decides. No model-format change. `p2predict_version` unchanged from v0.9.

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
- **`--json` flag on both CLIs** for machine-readable structured output, replacing Rich-formatted tables. Designed for AI agents, scripts, and downstream tooling that need to ingest P2Predict's output without regexing terminal text. Closes ROADMAP item #1 (the agent-readiness prerequisite for the MCP server shipped in v1.0).
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
- **Public Python API.** `from p2predict import auto_train, explain, predict_interval, what_if, save_model, load_model, ...` — the same functionality the CLI exposes, callable from scripts / notebooks / agent code without shelling out. This is the surface the MCP server (v1.0) and AI agents call.
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
