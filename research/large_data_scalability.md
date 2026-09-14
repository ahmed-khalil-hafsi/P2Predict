# Does P2Predict work at 100k+ rows?

**Status:** findings, no core changes made. 2026-09-15.

## Why this was asked

P2Predict's documentation, thresholds and internal reasoning are written around
the *typical* dataset — 50–300 parts. That is an accurate description of the
common case, but it has leaked into places where it reads as a *limit*: the
ROADMAP anti-goal says the pipeline is tuned for "hundreds to low thousands of
rows", and several code paths are sized as if the data could never be larger.

The intent is that P2Predict works equally well at hundreds of thousands of
data points. This document measures where that is already true, where it is
merely slow, and where it is actually broken.

## How it was measured

Two datasets:

* **Synthetic**, 100,000 rows × 7 specs (4 numeric, 3 categorical, 40
  suppliers), multiplicative lognormal noise. Used for timing and payload size.
* **Real**, the heavy-equipment (Blue Book for Bulldozers) case-study data —
  400k rows available, the shipped 80k-row model with its 16,000-point
  calibration set. Used for anything where the answer depends on real
  heteroscedastic structure.

Machine: the dev Mac, `.venv` Python 3.14. Reproduce with
`research/large_data_scalability.py`; captured output in
`large_data_scalability_results.txt`.

**On the timings.** Run-to-run variance on this machine is large — two
independent runs of the identical probe gave 18.0 s and 55.7 s for the same
feature ranking, and 273 s and 414 s for the same random_forest tuning. Every
timing below is therefore quoted as a range across both runs and is
order-of-magnitude evidence, not a benchmark. Payload and artifact **sizes**
are deterministic and matched exactly across runs; those figures are exact.

## Summary

| # | Area | At 100k rows | Verdict |
|---|------|--------------|---------|
| 1 | `predict_from_csv` response | ~37 MB of JSON | **Broken** |
| 2 | `get_model_quality(include_holdout=True)` | 40,000 floats inline | **Broken** |
| 3 | Stored holdout + calibration in the model file | 1.21 MB vs 0.49 MB lean, grows forever | Wasteful |
| 4 | Feature ranking (`get_most_predictable_features`) | 18–56 s, ~600 MB, on data a 25k subsample ranks identically | Wasteful |
| 5 | `auto_train` | 6–8 min, 60–90% of it random_forest | Slow but correct |
| 6 | Interval banding pinned at 3 bands | extra calibration data buys nothing | **Capability left on the table** |
| 7 | Quality verdicts / bias equivalence test | behave correctly and sharpen with n | Fine |
| 8 | Conformal coverage | 90.3–90.7% realised at every band count | Fine |
| 9 | Preprocessing, log-target, SHAP, domain check | no size assumptions | Fine |
| 10 | Positioning in ROADMAP / SKILL.md | states a small-data limit | Wrong message |

The headline: **the maths scales; the agent-facing plumbing does not.** Nothing
in the modelling is small-data-only. The failures are all in how much data the
MCP tools try to hand back, how much the model artifact carries, and how much
work feature ranking does for a result that does not need it.

---

## 1. `predict_from_csv` returns an unbounded payload — broken

`src/p2predict/mcp/server.py` builds one dict per CSV row (`input`,
`prediction`, `in_domain`) and returns the whole list through `_ok`, which is a
bare `json.dumps` with no cap.

Measured: **0.75 MB per 2,000 rows**, so **~37 MB for 100k rows** and **~149 MB**
for the 400k-row bulldozers file — plus 23–93 s just to assemble it. That goes
straight into the agent's context window. This is not a slow path, it is a path
that cannot complete usefully.

This matters more than any other item here, because the MCP server is the
primary interface in v1.0 and `predict_from_csv` is precisely the tool a user
hits when they drop a large file.

`with_explanation=True` and `coverage` multiply the per-row payload further.

**Options** (a design decision, not an obvious fix):

* **A — cap and summarise.** Return the first N rows (say 500) in full plus
  aggregate statistics over all rows (count, price distribution, how many
  out-of-domain), with an explicit `truncated` flag naming the total. Cheapest,
  keeps the tool honest, but the agent cannot see every row.
* **B — write results to a CSV and return the path** plus the same summary
  block. Matches how a category manager actually uses a 100k-row file (they
  open it in Excel), and it is the only option that scales without limit.
* **C — both**: write the file always, inline the rows only when under the cap.

Recommend **C**. Whichever is chosen, `predict_batch` should get the same cap
for consistency even though its input is agent-supplied and therefore
self-limiting.

## 2. `include_holdout=True` inlines the whole holdout — broken

`quality.build_quality_report(..., include_holdout=True)` attaches
`y_actual` / `y_predicted` in full. At 100k rows that is 40,000 floats
(~0.5 MB); at 1M rows, ~5 MB. The stated purpose is "so an agent with a
plotting tool can draw its own charts", which a bounded sample serves just as
well.

**Recommendation:** downsample to a fixed budget (e.g. 2,000 points, uniformly
at random, seeded) and report `n_holdout_total` alongside so the agent knows it
is looking at a sample. Charts are visually identical; the verdict numbers are
unaffected because they are computed on the full holdout regardless.

## 3. The model artifact carries the full holdout and full calibration set

`compute_calibration_residuals` stores every test residual *and* every test
prediction as Python lists, and the MCP `train` tool adds `holdout_y_test` /
`holdout_y_pred` on top. At 100k rows the saved model is **1.21 MB** against
**0.49 MB** with those fields stripped; at 1M rows the lists dominate the file
entirely.

The conformal quantile does not need 20,000 residuals — the order statistic is
stable well below that, and a uniformly random subsample preserves
exchangeability, so the finite-sample coverage guarantee survives intact.

**Recommendation:** cap both at a budget (e.g. 20,000 calibration points,
2,000 stored holdout points) by seeded random subsample, recording the true
`n_calibration` and the fact that a subsample was taken. Needs a decision on
the budget and a coverage check before shipping — this is the one item here
that touches the interval guarantee, so it should not be changed casually.

Note the interaction with item 6: if band count is allowed to grow with the
calibration set, the calibration cap sets the ceiling on how many bands are
ever possible. Decide them together.

## 4. Feature ranking fits an unbounded forest on the full dataset

`get_most_predictable_features` fits a 100-tree `RandomForestRegressor` with
`max_depth=None` on **every row**, purely to rank columns. It is called from
the train CLI's auto path, from MCP `train` when `features` is not given, and
from `propose_training_plan`.

Measured at 100k × 7 (times from the two runs):

| Rows used | Time | Ranking produced |
|-----------|------|------------------|
| 100,000 | 18.0 s / 55.7 s | `length_mm, weight_g, supplier, qty, tolerance_um, material, finish` |
| 50,000 | 7.4 s / 22.7 s | identical |
| 25,000 | 3.7 s / 7.3 s | identical |
| 10,000 | 1.6 s / 2.3 s | one adjacent swap (`qty` ↔ `tolerance_um`, both minor) |

Peak RSS reached ~600 MB. At 300k × 30 this extrapolates to several minutes and
multiple GB — paid *before* any model is trained, and paid again by
`propose_training_plan`, which the MCP flow calls first by design. So an
agent-led session on a large file pays it twice.

The ranking is stable down to 25k rows and only reorders two near-tied minor
specs at 10k, which is the whole point: the full-data fit buys nothing a
subsample doesn't already give.

**Recommendation:** subsample to a cap (25k–50k rows) for ranking only. The
ranking is a screening step, not a fitted artifact; the measured cost of the
subsample is zero and the saving is an order of magnitude. Low risk, clear
gain.

## 5. `auto_train` is slow but correct

Tuning cost at 100k × 7, `budget="fast"` (two runs):

| Algorithm | Time |
|-----------|------|
| ridge | 17.7 s / 26.0 s |
| xgboost | 52.1 s / 19.5 s |
| random_forest | 272.9 s / 414.1 s |
| **total (auto)** | **5.7 min / 7.7 min** |

Nothing is wrong here — `min_resources="exhaust"` already does the right thing
at scale (the winner-deciding rung uses the full training set). But
random_forest is 60–90% of the bill, and its search space includes
`max_depth=None` with up to 400 (fast) / 800 (thorough) trees, which is the
worst possible shape on large data. At 400k rows this becomes tens of minutes.

Worth noting but **not** worth changing blind: dropping or bounding
random_forest at scale would change which algorithm wins on some datasets, and
the heavy-equipment case study exists precisely because trees + target encoding
win at scale. If this is pursued, the right move is a depth/`max_samples` bound
in the large-n branch of `_search_space`, measured against the case studies for
accuracy loss — not removing the candidate.

Also worth a look: both the estimator (`n_jobs=-1`) and `HalvingRandomSearchCV`
(`n_jobs=-1`) claim all cores, which oversubscribes threads. Harmless on 200
rows, possibly a real tax at 100k. Unmeasured.

## 6. Interval banding is pinned at 3 bands — more data buys nothing

`intervals.N_BANDS = 3`, always, from `MIN_CALIBRATION_FOR_BANDING = 150`
upward. A 300k-row dataset gets exactly the same three coarse terciles as a
750-row one.

Tested on the real 80k heavy-equipment model, splitting its 16,000 calibration
points in half — bands built on one half, coverage and width evaluated on the
other:

| Bands | Overall coverage | Median width | Cheapest band width | Per-band coverage spread |
|-------|------------------|--------------|---------------------|--------------------------|
| global | 90.7% | ±107.6% | — | — |
| 3 (today) | 90.3% | ±116.1% | ±90.0% | 90.0–90.6% |
| 5 | 90.4% | ±112.3% | ±84.3% | 90.0–91.2% |
| 10 | 90.4% | ±101.7% | ±74.2% | 87.5–92.5% |
| 20 | 90.7% | ±109.1% | ±73.7% | 83.6–94.9% |

At 10 bands the cheapest decile's likely-range tightens from **±90% to ±74%**
while per-band coverage stays near target. At 20 bands it falls apart (one band
at 83.6%, another at 94.9%) — the per-band sample is too thin.

So the gain is real but bounded, and it appears only on heteroscedastic data:
the same experiment on the homoscedastic synthetic set showed no improvement at
any band count, as expected by construction.

**Recommendation:** make band count a function of calibration size with a floor
on points-per-band (~800 on this evidence), capped somewhere around 10. This is
a genuine capability gain — it is the one place where "we have 100k rows"
should visibly buy the user a better answer — but it changes a shipped
guarantee's shape, so it wants measurement across all four case studies before
implementation, not just this one.

## 7–9. What already scales correctly

* **Bias / verdict layer.** The equivalence test (`bias_assessment`) is the
  right design at every n: `resolution_pct` shrinks as the holdout grows, so a
  large dataset earns a crisp `immaterial` / `material` instead of `unknown`.
  The materiality band (±5%) is a commercial constant and correctly does *not*
  move with n — the old p-value gate would have flagged every model at 100k
  rows as biased, which is exactly why it was replaced. Nothing to change.
* **Conformal coverage.** Realised 90.3–90.7% at every configuration tested on
  16k calibration points.
* **Preprocessing.** `_AdaptiveTargetEncoder` clamps folds *downward* for small
  data only; at scale it is plain 5-fold `TargetEncoder`. Target encoding is
  the right choice for high-cardinality categoricals at any size.
* **Log-target, SHAP, what-if, domain check.** No size-dependent logic.
  `explain_batch` already builds the explainer once; `background_sample` is a
  fixed 100 rows by design and correctly stays constant.
* **`find_leaky_features`.** 0.34 s at 100k × 8. Fine.

## 10. Positioning says small

Two places state a limit rather than a typical case:

* `ROADMAP.md` anti-goals: *"tuned for procurement-shaped data (tens of
  features, hundreds to low thousands of rows...)"*.
* `.claude/skills/p2predict/SKILL.md`: *"A real BOM-benchmarking dataset is
  small (tens to low-hundreds of parts)"* — correct as guidance for the
  `--feature-outliers` rule it introduces, but it reads as a global claim.

Docstrings in `quality.py` and `preprocessing.py` also describe the 100–300
part case. Those are accurate and load-bearing (they justify specific threshold
choices) and should stay — but they should say "typical", not imply "maximum".

## Recommended order of work

1. **Cap the agent-facing payloads** (items 1 and 2). This is the only thing
   that is outright broken, and it is on the primary interface.
2. **Subsample feature ranking** (item 4). Low risk, order-of-magnitude saving,
   no design trade-off.
3. **Fix the positioning** (item 10).
4. **Cap what the artifact stores** (item 3) — needs a coverage check.
5. **Scale the band count** (item 6) — needs measurement across the case
   studies. The real capability win, and the one that deserves its own PR.
6. **Bound random_forest at large n** (item 5) — only if the case studies show
   the accuracy cost is nil.

Items 1–3 have no trade-off worth debating. Items 4–6 change numbers users
already see and should be decided deliberately.
