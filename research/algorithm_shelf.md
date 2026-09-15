# Finding: what the benchmark does — and does not — say about the algorithm shelf

**Status: open, and deliberately scoped down.** Two of the three changes this
started as do **not** clear the bar that prior decisions already set, and are
re-filed as *measurements to run before proposing anything*. One survives:
cross-fitting the conformal calibration. **No core code changed.**

Evidence: the TDC ADMET validation in
[`../evals/tdc_admet_validation.md`](../evals/tdc_admet_validation.md), raw
numbers in `evals/results/1.1.0/*.json`. Prior decisions this must answer to:
`ROADMAP.md` ("Still deliberately left out"),
[`large_data_scalability.md`](large_data_scalability.md) (item 5, declined
2026-09-15) and [`methodology_review.md`](methodology_review.md) (Tier-2 items 7
and 9, both still open).

## TL;DR

The benchmark produced one striking number — **XGBoost won model selection in
29 of 30 runs; Ridge never won** — and three tempting conclusions from it: add
CatBoost/LightGBM, blend the runners-up instead of crowning one winner, and
stop discarding the calibration split. Held against what has already been
decided:

| Idea | Verdict |
|---|---|
| **A. Add CatBoost / LightGBM** | **Still a no.** The ROADMAP already declined this on maintenance-vs-marginal-gain grounds, and this run measured neither library. It also does not show what it looks like it shows. Re-filed as a harness experiment. |
| **B. Blend instead of crowning a winner** | **Not supported by this evidence, and cheaper to test than to argue about.** On descriptor data the runner-up is 0.015–0.059 CV R² behind in every sweep, which is the regime where blending helps least. Re-filed as a harness experiment. |
| **C. Cross-fit the conformal calibration** | **Survives — on different evidence than it started with.** Not because it would move the leaderboard (it would barely) but because Finding 3 measured exactly the failure it fixes, in the regime our users occupy. |

The genuinely new asset is not any of the three ideas. It is that **`auto_train`
now has an objective external measuring stick**: a change can be scored on a
public leaderboard before and after, instead of on self-scored case studies.
That is worth more than the proposals it generated, and it is what makes
"measure first" a real option rather than a polite refusal.

## What was already decided, and by whom

Nothing below is new ground. Three prior records bear on it:

1. **`ROADMAP.md` → "Still deliberately left out":**
   *"LightGBM / CatBoost — adds maintenance for a marginal accuracy gain. RF +
   XGBoost already cover the tree-ensemble space well."* A standing decline on
   exactly item A.
2. **[`large_data_scalability.md`](large_data_scalability.md) item 5, declined
   2026-09-15.** Often summarised as "random_forest dominating `auto_train`" —
   but read it carefully: it is about **runtime**, not selection. At 100k rows
   random forest is 60–90% of the tuning bill. The decline said bounding or
   dropping it "would change which algorithm wins on some datasets, and the
   heavy-equipment case study exists precisely because trees + target encoding
   win at scale", and ranked it the **riskiest** of the five declined items.
3. **[`methodology_review.md`](methodology_review.md), still open:** item 7 —
   *"model selection rides a single CV point estimate … the family choice is
   noisy on small data and the winning CV score is optimistically biased"*; item
   9 — *"one holdout does everything: R², the bias test, per-band APE, and
   conformal calibration all ride the same ≤20% sample."* Those are the existing
   statements of ideas B and C respectively, written before this benchmark
   existed.

## A. CatBoost / LightGBM — the evidence does not reach the decision

The number is real: across 3 endpoints × 2 budgets × 5 seeds, XGBoost was
crowned 29 times, random forest once (PPBR `fast`, seed 2), Ridge never. Four
reasons that does not reopen the ROADMAP decline.

**1. The run measured neither library.** Nothing here compares XGBoost to
CatBoost or LightGBM. The gap to the top of the Caco-2 board (0.289 → 0.276, a
4.5% improvement needed for top 3) is attributed to *entries that use Morgan/ECFP
fingerprints*, i.e. better features, not a better booster. The confound is total,
and the ROADMAP's claim — the gain is marginal — is untouched by this run.

**2. On P2Predict's own data the shelf is not one algorithm deep.** The
29-of-30 result describes 210-column descriptor data at 637–2,940 rows. The
case studies say something different:

| Case study | Rows | Winner |
|---|---:|---|
| Battery-management ICs | 150 | **Ridge** |
| Used cars | 80,000 | **Random forest** |
| Heavy equipment | 80,000 | XGBoost |
| Aerospace fasteners | 15,197 (train) | XGBoost |

All three shelf members win something, and the one that wins at the *typical*
dataset size is the one that never won on TDC.

**3. "Ridge never won" is partly a featurisation artifact.** Ridge's CV R²
reaches −2.4e32 on Lipophilicity (`cv_scores` in the result files). RDKit's
`Ipc` descriptor reaches 3.6e31 in that training set; a scaled linear model
extrapolating past that range produces astronomical predictions. Ridge did not
lose a fair fight on those boards — it blew up. Worth knowing on its own terms
(P2Predict's own preprocessing does not guard against an extreme-magnitude
numeric column on the linear path), but it is not evidence about linear models
on parts data.

**4. The install cost is not hypothetical here.** `ROADMAP.md` item 5 documents
that XGBoost's macOS wheel needs Homebrew `libomp`, which needs an admin
password, which the target user often does not have — the reason INSTALL.md
carries a manual prerequisite step. LightGBM's macOS wheel has the same
`libomp` dependency; CatBoost ships self-contained but is a substantially
larger wheel. *(Both recalled, not verified — check before relying on either.)*
Adding a fourth estimator adds a fourth way for a category manager's laptop to
fail on install, which is precisely the "maintenance" the ROADMAP priced in.

**Scoped down to:** a harness experiment, not a core change. Fit LightGBM and
CatBoost inside `evals/` (same preprocessing, same protocol, same 5 seeds) and
see whether either beats XGBoost by more than the across-seed spread. That
touches nothing under `src/` and produces the number the ROADMAP decline was
missing. If neither wins there *and* neither wins on the four case studies, the
decline stands and this section can be deleted.

**Adoption gate, pre-registered** (so the result cannot be read generously
after the fact). A new shelf member ships only if it:

* beats the incumbent's mean MAE by more than the baseline across-seed std on
  **≥ 2 of 3** endpoints, and regresses none of them by more than that std;
* wins or ties on **≥ 1 of the four case studies** at its own dataset size,
  with no verdict downgrade anywhere;
* does not increase `auto_train` wall clock at case-study sizes by more than
  ~2×; and
* installs on a Mac with no admin rights, or is skipped gracefully the way
  ROADMAP item 5 describes for XGBoost.

### The 100k-row convergence: a note, not a justification

LightGBM is substantially faster than XGBoost at scale, so it is tempting to
say the algorithm-shelf question and the large-data question are one piece of
work. Two reasons not to lean on that:

* The declined item 5 is about **random forest's** runtime, and the fix it
  contemplated was bounding RF's depth in the large-n branch of `_search_space`
  — not substituting a different library. Adding LightGBM does not remove RF
  from the loop, so it does not fix what item 5 measured unless RF is *also*
  dropped, which item 5 explicitly called the riskiest option.
* The reason item 5 was declined applies here unchanged: it is a no-op for the
  50–300 part catalogue. Nothing in this benchmark (max 2,940 rows) speaks to
  100k rows at all.

So: worth a sentence in whoever's notes picks up large-data work, not a reason
to act now.

## B. Blending instead of crowning a single winner

`auto_train` fits all three candidates with `refit=True`, compares one CV score
each, keeps the winner and **discards two fitted models**. Averaging them is
the cheapest known win in tabular ML — in general. On this evidence,
specifically, it is not supported:

Mean CV R² margin of the winner (XGBoost) over the runner-up (random forest):

| Endpoint | `fast` | `thorough` |
|---|---|---|
| Caco-2 | +0.042 | +0.059 |
| PPBR | +0.015 | +0.027 |
| Lipophilicity | +0.045 | +0.049 |

Blending pays when candidates are close and their errors are decorrelated. A
runner-up 0.04–0.06 R² behind, with Ridge diverging entirely, is the regime
where a naive average *hurts*. The one place the eval hints otherwise is the
selection-noise story, not the blending story: the single seed where CV crowned
random forest (PPBR `fast`, seed 2, RF 0.399 vs XGB 0.383) produced the **worst
test MAE of that sweep** (8.32) — a one-seed illustration of
`methodology_review.md` item 7, that the CV point estimate is noisy enough to
pick the wrong family.

Where blending plausibly earns its keep is the opposite regime — 150 parts,
three near-tied candidates, which is what the battery-IC study looks like — and
that is **unmeasured**.

The cost side is larger than it first appears, and is the real reason not to
reach for this yet. A blend is not a `Pipeline`, and the rest of the product
assumes one:

* `explain.py` picks TreeExplainer vs LinearExplainer from the fitted
  estimator's family; a mixed-family blend has no single answer.
* `extract_feature_importances` reads `feature_importances_` or `coef_` off one
  estimator.
* The log-target wrap (`TransformedTargetRegressor`) wraps one pipeline.
* `feature_domain` / the shipped `in_domain` block, the calibration dict and
  `model_name` in the artifact all describe one model.

So a blend is a change to the *artifact contract*, not to one function. That
work needs a measured gain in front of it.

**Scoped down to:** measure a two-model average (winner + runner-up, weights
1:1 and CV-score-weighted) in `evals/` and on the four case studies. If the
gain at case-study sizes is smaller than the verdict layer's own resolution,
delete this section.

## C. Cross-fit the conformal calibration — this one survives

This started as "we trained on `train` while others used `train_val`, worth
~14% more data". That framing is about the leaderboard and is the weakest
version of the argument. The product version is stronger, and Finding 3
measured it.

### What happens today

`prepare_data` splits 80/20 (`test_size=0.2`, `random_state=0`). `auto_train`
fits on the 80%. The 20% then does **four jobs at once**
(`methodology_review.md` item 9): holdout R², the ±5% median-relative
equivalence bias test shipped in #39, per-band APE, and conformal calibration
(`compute_calibration_residuals(model, X_test, y_test)`). The shipped model is
never refit on the full dataset.

At the typical 50–300 parts that means:

| Catalogue | Trained on | Calibration points | Banding? |
|---:|---:|---:|---|
| 60 parts | 48 | 12 | no |
| 150 parts | 120 | 30 | no |
| 300 parts | 240 | 60 | no |
| 750 parts | 600 | 150 | just barely |

### Why the benchmark makes this concrete

Finding 3 measured coverage against calibration-set size directly: **94.5–100%
observed at 91 calibration rows**, converging to 86.3–90.1% at 420. Intervals
built on a small calibration set are conservative and noisy — not wrong, but
wider than the model's actual error justifies, and unstable seed to seed. Our
users sit at **10–60** calibration rows, i.e. below the worst point measured.
That is the failure this fixes, and the benchmark is the first place it has
been quantified rather than asserted.

Cross-fitting (CV+ / jackknife+) computes residuals out-of-fold across all
rows, so the same 300-part catalogue yields **300 calibration residuals instead
of 60**, and the shipped model can be fit on all 300 rows instead of 240
(+25% training data at any dataset size — the proportion is fixed by the split,
what changes with size is how much each row is worth).

It also reaches something that was declined from the other direction:
`intervals.MIN_CALIBRATION_FOR_BANDING = 150` means banding never engages below
~750 rows today. With cross-fit residuals a 150–300 part catalogue crosses that
threshold. Note this is a **different case** from the banding item declined in
`large_data_scalability.md` — that one was declined as "only helps above ~12k
rows"; this is the small-n side of the same constant, and it is the side where
the typical user lives.

### What it costs, and what must be decided first

* **K model fits instead of one.** Trivial at 300 rows; prohibitive at 100k,
  where `auto_train` already takes 6–8 minutes. Any implementation needs a
  size-dependent policy: cross-fit below some n, keep split-conformal above.
  That also contains the interaction with the declined artifact-size item —
  cross-fitting makes the stored residual list O(n), which is exactly what that
  item flagged at scale.
* **A weaker worst-case guarantee.** Split conformal gives exactly 1−α;
  CV+/jackknife+ guarantee 1−2α in the worst case, and land near nominal in
  practice. P2Predict advertises a coverage promise to a category manager, so
  this is a change to what is being promised and must be measured, not assumed.
* **The verdict layer moves with it.** If the holdout dissolves into folds,
  R², the #39 equivalence test and per-band APE must be computed on out-of-fold
  predictions — arguably an improvement (the equivalence test sharpens with n,
  as `large_data_scalability.md` found), but it changes `holdout_y_test` /
  `holdout_y_pred` in the artifact and the meaning of every quality number.
  `background_sample` and `feature_domain` would also then describe the full
  dataset rather than the training split.

None of that is a reason not to do it. It is a reason it needs its own PR,
its own measurement, and a decision about the guarantee before code.

## The experiment, spelled out

The reason to take any of this seriously is that the measurement now exists.
Core changes to `auto_train` have previously been judged on case studies we
built and scored ourselves. This one can be judged on a public board.

1. **Baseline is already recorded:** `evals/results/1.1.0/` — three endpoints,
   two budgets, five seeds, official `group.evaluate_many()` output, plus
   per-seed `interval_90_coverage` and `mean_interval_width`.
2. **Change core on a branch.**
3. **Re-run** `evals/run_all.sh` (or the `fast` sweep alone for a cheaper
   signal: ~21 minutes of fitting for all three endpoints). Results land in
   `evals/results/<new version>/` beside the baseline, not on top of it.
4. **Compare like for like** — same endpoints, same seeds, same budget — and
   apply the pre-registered gate above.
5. **Re-run the four case studies** for R², verdict and training-time
   regressions. The leaderboard scores accuracy only; the verdict layer, the
   intervals and the dollar translation are scored nowhere on TDC.

For item C specifically, MAE is the wrong instrument — cross-fitting is an
*interval* change and will barely move the mean error. The right measurement is
already instrumented: observed coverage versus the 90% nominal, and mean
interval width, per endpoint, at calibration sizes of 91 / 279 / 420. Cross-fit
intervals should land closer to nominal and narrower at the small end without
dropping below it. If they do not, C fails on its own evidence.

**Caution on the rank itself.** Improving a pharma leaderboard is not a product
goal, and "we went from 7th to 5th" is a vanity metric if the change did not
also help a category manager with 200 parts. The gate above deliberately
requires both.
