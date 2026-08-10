# Methodology review — what's missing, scientifically

A statistics/ML audit of the modeling core: training and model selection
(`training.py`), preprocessing (`preprocessing.py`, `outliers.py`,
`feature_selection.py`), conformal intervals (`intervals.py`), explanations
(`explain.py`), and quality verdicts (`quality.py`). Findings are ranked by
how much they can mislead the numbers we hand a category manager. The Tier-1
items are **quantified on the case-study datasets** by
`case-studies/benchmark_methodology_gaps.py`; measured results live in
`case-studies/methodology_gaps_benchmark_results.txt` and are summarized
inline below.

Per the project's working agreement, this PR changes nothing in core — it
proposes; the follow-up PRs dispose.

## What's already solid

Worth stating first, because these are the traps most tools in this space
fall into and P2Predict doesn't:

- **Target encoding is cross-fitted.** The `TargetEncoder` sits inside the
  pipeline handed to CV, uses internal 5-fold cross-fitting with
  empirical-Bayes smoothing (`smooth="auto"`), and refits per fold — the
  classic target-encoding leakage bug is absent (`preprocessing.py`).
- **Conformal intervals are done right.** Split conformal with the exact
  finite-sample quantile `ceil((n+1)(1−α))/n`, conservatively rounded;
  Mondrian tercile banding when calibration n ≥ 150; multiplicative bounds
  for log-target models so ranges never go negative (`intervals.py`).
- **Explanations are exact.** `LinearExplainer`/`TreeExplainer` only — no
  KernelExplainer approximations (`explain.py`).
- **Verdicts are honesty-gated** on holdout size and residual bias
  (`quality.py`).

## Tier 1 — the math is biased or leaks (fix)

### 1. Retransformation bias: log-target models back-transform with plain `exp()`

`build_pipeline` wraps the estimator in
`TransformedTargetRegressor(func=np.log, inverse_func=np.exp)`
(`training.py:152-154`). Because `E[exp(ŷ+ε)] ≠ exp(ŷ)`, the naive
back-transform is biased low in price space by roughly `E[exp(ε)]` — and it
fires precisely on the heavily-skewed datasets where the log wrap
auto-triggers. The textbook fix is Duan's smearing estimator: multiply the
naive prediction by `S = mean(exp(ε̂ᵢ))` over **out-of-sample** residuals
(a boosted tree's training residuals are near zero, so S must come from the
holdout or CV out-of-fold residuals, not the training fit).

**Measured (Experiment 1):**

- **Used cars** (48k train / 16k eval, R² 0.74): naive predictions run
  **7.9% low in mean terms** (S = 1.079). Mean holdout residual −$1,419 →
  **+$2** after smearing; price-space R² 0.741 → 0.754; the bias t-test goes
  from p < 10⁻⁴ to p = 0.97. Median APE essentially unchanged
  (18.1% → 18.7%). Textbook case: real bias, cleanly corrected.
- **Fasteners** (intrinsically noisy catalog, log residuals heavy-tailed):
  S = **4.05**. Smearing fixes the mean (−$49 → +$15) but multiplies every
  quote by ~4×, so median APE explodes 81% → 315%. The correction is
  mathematically right for the mean and commercially wrong for a per-part
  quote on this data.

**Nuance the numbers confirm:** naive `exp()` is approximately
*median*-unbiased (fasteners median residual $0.24), while smearing targets
*mean*-unbiasedness — and on heavy-tailed log residuals S is dominated by a
few extreme ratios. Recommendation for the core fix: apply smearing where the
mean matters (aggregate-spend estimates, batch totals) and keep the per-part
point estimate median-anchored, or expose both — but the choice should be
explicit rather than an accident of `inverse_func=np.exp`.

### 2. Pre-split leakage: outlier bounds and feature selection see the holdout

The CLI computes Tukey outlier bounds (`cli/train.py`, target and feature
policies) and RF-importance feature selection
(`feature_selection.get_most_predictable_features`, auto mode keeps top 6)
on the **full dataset before the train/test split**. Two consequences:

- the selection/bounds decisions have seen the holdout rows (information
  leakage into the reported R², band errors, and conformal calibration);
- with `drop` policies, the holdout itself is cleaned of the extremes
  production quotes will contain, so the reported error understates
  deployment error.

**Measured (Experiment 2) — smaller than principle suggests:**

- The *information leakage itself* is *negligible* at every scale tested:
  bounds fitted pre- vs post-split move eval R² by ≤ ~0.01; leaky feature
  selection is noisy at n=400 (±0.34 across seeds on fasteners) but not
  systematically optimistic.
- The *material* piece is **reporting on a cleaned holdout** under `drop`
  policies: at full scale the reported R² beats the honest raw-holdout R² by
  **+0.06 on cars (0.792 vs 0.733)** and **+0.10 on fasteners**; at small n
  the direction is erratic (fasteners n=400: −0.30) because dropping extremes
  also shrinks target variance. The default `warn` policy is unaffected.

**Fix (hygiene, demoted below #3 in urgency):** split first; fit outlier
policy and feature selection on the training rows only; report holdout
metrics on raw holdout rows so the quoted error reflects the quotes
production will actually see.

### 3. The trustworthy/unreliable gate runs a t-test in the wrong space

`quality.residual_bias_p` runs `ttest_1samp` on **raw price residuals**
(`quality.py:333-340`) even for log-target models, and `assess_model` gates
`trustworthy` vs `unreliable` on that single p-value at 0.05. Price residuals
on skewed targets are heavily skewed and dominated by the expensive tail —
the t-test's assumptions are weakest exactly where it decides the verdict.
Compounding: the retransformation bias of finding #1 shows up as a price-space
mean shift, so a model that is well-behaved in log space gets branded
"unreliable" for a defect the back-transform introduced.

**Measured (Experiment 3) — both case studies flip verdict:**

| dataset | t-test, price space (today) | t-test, log space |
|---|---|---|
| cars (R² 0.74) | p < 10⁻⁴ → **unreliable** | p = 0.50 → **trustworthy** |
| fasteners | p < 10⁻⁴ → **unreliable** | p = 0.96 → **usable** |

Both models are well-behaved in log space; the price-space test is flunking
them for the retransformation bias of finding #1. A price-space Wilcoxon
also trips (p < 10⁻³) — which exposes a second defect: the gate is **pure
statistical significance with no materiality threshold**. On a 16k-row
holdout, a −$349 median residual on ~$18k cars (−2%) is "significant" and
costs the model its verdict.

**Fix:** for log-target models test residuals in log space; add an
effect-size floor (e.g. |mean or median residual| above some % of the mean
price) alongside the p-value; re-run after the smearing correction of
finding #1.

### 4. No group-aware validation

Random KFold inside `HalvingRandomSearchCV` and a random 80/20
`train_test_split` (`prepare_data.py`) mean near-duplicate parts, part-family
variants, and same-supplier rows straddle the train/test line. Holdout scores
are optimistic on real catalogs, which are full of such near-duplicates. Not
benchmarked here (the case-study CSVs carry no part-family key), but the
mechanism is well-established.

**Fix:** optional `--group-column` → `GroupShuffleSplit` for the holdout and
`GroupKFold` for CV; recommend it in `propose_training_plan` when a column
looks like a part/family/supplier identifier.

## Tier 2 — missing methodology that undermines the product's claims

5. **No out-of-distribution guard at predict time.** An unseen supplier
   silently gets the global target mean (`TargetEncoder`) or an all-zero
   one-hot vector; numeric inputs outside the training range are not flagged
   anywhere in `predict`/`what_if`/`predict_batch`. The tool hands a
   category manager a confident benchmark for a part unlike anything it
   trained on. Fix: flag unseen category levels and out-of-range numerics in
   the predict payload with a `say_to_user` caveat.
6. **Intervals are not input-adaptive.** Width is constant (or
   piecewise-constant across 3 predicted-price terciles); it never widens
   for genuinely unusual parts. Later: normalized conformal scores or CQR.
7. **Model selection rides a single CV point estimate** — chosen on
   log-space R² but graded on raw-space R², no 1-SE rule, no selection-aware
   inference. The family choice is noisy on small data and the winning CV
   score is optimistically biased.
8. **Log-model dollar attribution divides by the sum of log contributions**
   (`explain.py`, `dollar_attribution`): when drivers nearly cancel
   (`Σφ ≈ 0`) the rescale blows up. Needs a guard. Related: the user-facing
   "feature signal" uses native importances (coefficients / tree gain) while
   explanations use SHAP — two different importance notions surfaced to the
   same user.
9. **One holdout does everything** — R², the bias test, per-band APE, and
   conformal calibration all ride the same ≤20% sample. Valid individually,
   but there is no independent check on any of them.

## Tier 3 — worth noting

10. No minimum-sample guard to train — only post-hoc verdict caveats; no
    learning-curve signal to tell a user "50 more rows would help".
11. XGBoost regularization (`reg_alpha`/`reg_lambda`/`min_child_weight`) and
    early stopping are untuned; no ElasticNet/LightGBM in the zoo.
12. The leakage screen `find_leaky_features` is wired into the MCP path only;
    the CLI `train` never calls it.
13. Impurity-based importances (biased toward high-cardinality
    target-encoded columns) drive both auto-selection and the user-facing
    "price drivers", while an unbiased permutation-importance helper
    (`model_evals.calculate_feature_importance`) sits unused.
14. No model-staleness signal: `training_date` is persisted but never
    checked, and there is no drift monitoring story.

## Proposed follow-up PRs (re-ordered by the evidence)

1. **Bias test in the right space, with a materiality floor** (`quality.py`):
   log-space residuals for log-target models plus an effect-size threshold.
   Highest urgency — today it mislabels both flagship case-study models as
   "unreliable"-grade, and the failure direction (under-trust) is quietly
   costing good models their verdict.
2. **Smearing made explicit** for the log-target back-transform
   (`training.py`): correct the mean where the mean matters (batch totals,
   aggregate spend), keep per-part quotes median-anchored; document the
   choice. On cars this is a real 8% mean underestimate.
3. **Split-first hygiene** (`cli/train.py`, `mcp/server.py`): fit outlier
   policy and feature selection on training rows only and report holdout
   metrics on raw holdout rows (closes the +0.06–0.10 reported-vs-honest R²
   gap under drop policies); wire `find_leaky_features` into the CLI on the
   way through.
4. **Optional group column** (`prepare_data.py`, `training.py`):
   `GroupShuffleSplit`/`GroupKFold`, surfaced in `propose_training_plan`.

Tier-2 items are follow-on candidates once the above land; #5 (OOD flags) is
the highest-value of them for the agentic surface.
