# Roadmap

The first arc — getting from 6/10 to 8/10 robustness — shipped across v0.4 to v0.8. SHAP attributions, conformal likely-range intervals, what-if comparisons, feature-side outlier handling, and pip-install + public Python API. All five with axiomatic tests locking in the property each feature claims.

The next arc is **distribution and agent-first deployment**: putting the rigorous math in front of the procurement workflows and AI agents that will actually use it. Four items, ordered.

## Shipped — v0.4 → v0.8 (8/10 robustness)

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

## ~~6. First case study — used vehicle pricing~~ ✅ Shipped in v0.9.1

`case-studies/used-cars/` is the tutorial case study and the warm-up before the procurement-specific ones. End-to-end reproducible build on the 426k-row Craigslist Cars+Trucks dataset (CC0): `fetch_data.py` (kagglehub + new-style Kaggle API token), `prepare_data.py` (clean + sample), `p2predict-train`, and `predict_examples.py` (point estimate + 90% likely range + SHAP multiplicative attribution + what-if).

Two reproducibility paths in the README: full Kaggle (matches the numbers exactly — Ridge wins auto-selection at CV R² 0.520 with log-target active, holdout R² 0.634, MAE $5,381) and a 5,000-row sample checked into git for readers without a Kaggle account.

Case studies earn their keep on day one: this one surfaced two real bugs (SHAP sparse-matrix breakage on Ridge/Lasso with high-cardinality categoricals; SHAP + XGBoost 3.x `base_score` parse error) and one UX wart (auto-mode's silent 6-feature cap). All three landed alongside the case study as fixes + regression tests + the `--max-features` flag.

Battery management ICs (Octopart / DigiKey) and aerospace fasteners (DLA PUB LOG, public domain) are now built out in `case-studies/`. PCBA composition (compose three trained models into BOM-level cost) remains scaffolded and queued for v0.9.2+.

---

## Next — distribution + agent-first

### ~~0b. `--log-target {auto,on,off}` to override the skew rule~~ ✅ Shipped in v0.9.3

The auto rule (`scipy.stats.skew(y_train) > 1.0`) only flips on log-target when the sample happens to be skewed, but every multiplicative positive-quantity target (prices, costs, weights, lead times) wants the wrap regardless of whether *this* sample looks skewed — multiplicative attribution and strictly-positive conformal intervals are the *property* the wrap defends, not a fix for a particular distribution shape. The Battery Management ICs case study made this concrete: a 150-part dataset with skew 0.12 left log-target off and produced a 90% conformal interval of `-$1.40` to `$4.95` on a $1.77 prediction. `--log-target on` overrides the auto rule, with the same `y_train > 0` safety check. `off` is the symmetric override for additive-scale targets. Default stays `auto` so existing scripts and case studies are untouched, and `log_target_decision` lands in the `--json` payload so consumers can see which path was taken.

### ~~0a. `--report PATH` for the PDF model-quality report~~ ✅ Shipped in v0.9.2

The procurement-style 3-page model-quality PDF was previously reachable only via the expert + interactive prompt — invisible to auto-mode users and to non-interactive callers (CI, agents, scripted runs). `--report PATH` works in both auto and expert mode, with or without `--interactive`, and adds `report_path` to the `--json` payload so agents know where the PDF landed. The used-cars case study surfaced this gap by routing around it with a separate `generate_quality_report.py` script that called `plot_results_pdf` directly; the case-study reproduction now uses the flag.

### ~~0. `--max-features` for auto-mode feature selection~~ ✅ Shipped in v0.9.1

Small but load-bearing. Auto-mode previously capped at 6 features with no override — the used-cars case study showed this leaves real signal on the table (CV R² 0.34 → 0.52 on a 10-feature dataset, and the cap was also masking the log-target transform). `--max-features N` lifts the cap; default stays at 6 so existing workflows are untouched. When the ranker returns more columns than the cap, a one-line notice now reports it so the dropped features aren't invisible.

### ~~1. JSON output mode~~ ✅ Shipped in v0.9

`--json` on both `p2predict` and `p2predict-train`. Stable schema with `schema_version: "1.0"` on every response. Composes with `--interval`, `--explain`, and `--whatif`; each adds its block. Errors emit JSON too (`{"error": {"code": "...", "message": "..."}}` on stdout, exit 1) so agents piping the output get a parseable failure document.

15 new tests in `tests/test_json_output.py` assert the documented top-level keys on both CLIs, for both success and failure paths. The schema is now a tested contract.

Schema documented in [`src/p2predict/json_output.py`](src/p2predict/json_output.py) and the [README](README.md#machine-readable-json-output).

### 2. MCP server (v1.0 — the agentic-first headliner)

**Why it matters.** This is the agent-first deployment surface. Claude, Cursor, Zed, and custom procurement agents call P2Predict as typed MCP tools instead of shelling out to a CLI. The procurement user never sees a terminal — they talk to their existing agent, the agent calls P2Predict, the answer flows back in plain English. This is what makes P2Predict a first-class citizen in modern procurement workflows.

**Scope**
- New package `p2predict-mcp` (or an extra: `pip install p2predict[mcp]`).
- Typed MCP tools wrapping the v0.8 Python API: `predict`, `predict_batch`, `explain`, `predict_interval`, `what_if`, `train`, `list_models`, `get_model_info`.
- Trained models exposed as MCP resources with their metadata (target, features, training date, calibration size, log-target flag).
- Local-by-default deployment — runs in the user's environment, calls the user's models, no data leaves the network.
- Documentation for both self-hosted and (optional) hosted deployment.

**Acceptance**
- A Claude or Cursor agent can list, inspect, and invoke P2Predict tools without shelling out — the procurement user sees an answer, not a CLI.
- An RFQ-shaped batch prediction through MCP returns structured results an agent can summarise back to the user.
- Listed on the Anthropic MCP directory (or equivalent in other agent platforms).

### 3. PyPI publish (v1.0.x)

**Why it matters.** `pip install p2predict` works only against a local clone right now (`pip install -e .`). Publishing to PyPI removes the last install-friction step and makes the agent-platform listing requirements trivial — most MCP marketplaces want a published package, not a Git URL.

**Scope**
- A release workflow on tag push (`v1.0.0` and later).
- `python -m build` + `twine upload` with [PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/) so there's no long-lived token to manage.
- A post-publish smoke test: pull the published package into a fresh venv, run `p2predict --help`, run one prediction.

**Acceptance**
- `pip install p2predict` works from anywhere on PyPI.
- New releases publish automatically on tagged commits.
- The README install snippet stops saying "or once published".

### 4. Procurement-facing landing page (v1.1)

**Why it matters.** The README sells to developers and to procurement readers who already found the repo. A landing page (probably `p2predict.com` or `p2predict.dev`) sells to procurement leaders who don't know what GitHub is. Different audience, different language, different decision criteria — the README is the product spec, the landing page is the elevator pitch. Without it, you're invisible to the actual buyer.

**Scope**
- Domain registered (one of `p2predict.com` / `.dev` / `.ai`).
- Single-page static site focused on Part 1 of the README (the value scenarios) in a procurement-CFO voice — concrete numbers, named meetings, no ML jargon.
- Clear path to "try it" (developer track → GitHub) and "talk to us about deployment" (enterprise track → contact page).
- Built with a minimal static-site setup. The site is not the focus of the project; the math is.

**Acceptance**
- Landing page live at the registered domain.
- Three meetings booked with procurement leaders from the inbound link.

---

## Still deliberately left out

- **LightGBM / CatBoost** — adds maintenance for a marginal accuracy gain. RF + XGBoost already cover the tree-ensemble space well.
- **Quantile regression for Ridge** — too involved relative to the conformal interval already shipping.
- **Multi-format input (Parquet, Excel, DB)** — UX, not robustness or distribution. CSV covers 95% of procurement use today.
- **Web dashboard / JS UI** — wrong abstraction for a math-first tool. The agent-first thesis says presentation belongs in the user's existing agent / chat / spreadsheet, not in another dashboard.
- **Pro-tier features** (multi-user model registry, audit log, SSO, drift monitoring, ERP connectors) — these are the *paid* layer in the dual-licensing model. Don't build until 10+ companies are asking the same questions. Listen first.

## Anti-goals

These are things P2Predict deliberately is *not* trying to be, so contributions in these directions will be politely declined:

- A replacement for bottom-up should-cost tooling (aPriori, Siemens Teamcenter PCM). P2Predict is the *parametric* counterpart, not a replacement.
- A general-purpose AutoML library. The training pipeline is tuned for procurement-shaped data (tens of features, hundreds to low thousands of rows, mixed numerical and high-cardinality categorical).
- A black-box "trust us" model. Every answer is auditable — explanation, interval, what-if decomposition — by design.
