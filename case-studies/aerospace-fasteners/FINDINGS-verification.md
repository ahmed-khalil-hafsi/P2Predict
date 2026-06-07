# Verification: is the fastener ceiling really the data?

**Purpose.** Independently re-run, benchmark, and stress-test every
load-bearing claim in the case-study README, and answer one question: *is the
modest model quality truly a property of the catalog data, or partly a property
of how P2Predict trains?* All numbers come from `verify_findings.py`
(committed alongside this document; full output in
`assets/verify_findings_output.txt`). **No core code was changed** — this PR
proposes findings for discussion only.

## TL;DR

The README's central diagnosis — the catalog is intrinsically noisy — is
**confirmed, and in one respect even strengthened**. But "the model is near
the ceiling, not under-tuned" is **wrong in detail**: four core P2Predict
behaviours each leave measurable accuracy or trust on the table, and three of
them affect *every* dataset, not just this one.

| # | Finding | Scope | Measured effect here |
|---|---------|-------|----------------------|
| 1 | HPO **and algorithm selection** are decided after training candidates on **≤ 90 rows** of a 15,197-row training set — `HalvingRandomSearchCV` default `min_resources='smallest'` schedules resources 10 → 30 → 90 | **core, every dataset** | CV scores are noise (best = −1.18); the winning algorithm flips between identical runs: README run picked XGBoost (log R² 0.343), today's rerun picked random forest (log R² 0.282) |
| 2 | HPO is scored by **raw-space R²** even when log-target is on | **core, every skewed target** | the raw-scored search selected a model with log R² **−0.25** and 265% median error; the identical search scored in log space: 0.337 / 80% |
| 3 | `check_csv_sanity` silently **drops every row containing any NA** — here 48% of the dataset (36,668 → 18,997) | **core, every dataset with missing values** | training on all rows with XGBoost's native NaN handling: log R² 0.310 → 0.342 on the identical test set — and the model can then score the 48% of catalog parts the current pipeline refuses to touch |
| 4 | The conformal likely-range is **one global multiplicative width** for every part (×142 for the published model; ×225 for the refit) | **core** | banded calibration: parts > $155 get an honest ×58 band instead of ×225, the $5–$155 mid-tier ×200, sub-$5 parts an honestly *wider* ×435 — per-band coverage stays ≈ 90% |
| 5 | The study models 10 of ~20 well-covered catalog characteristics | case study only | +8 specs + ITEM_NAME: log R² +0.033 on the same split, and the measured noise ceiling rises **0.60 → 0.71** (band 4.5× → 3.3×) — part of the "irreducible" noise was just unmeasured features |

Combined best effort (all rows + native NaN + extended features + log-space
tuning): **log R² 0.375, median error 76.8%** on the harder all-rows holdout —
versus 0.319 for the study's configuration on the same split, and 0.282 for
what `p2predict-train` actually produces on a rerun today.

## What the README gets right (verified ✓)

- **`diagnose_noise.py` reproduces exactly**: 80% singleton signatures, 4.5×
  median within-signature band, log-R² ceiling ≈ 0.60 on the duplicate
  subset. The methodology — including the "compute the ceiling on the
  duplicate subset only" gotcha — is sound.
- **The published model's headline metrics reproduce to the digit** on the
  reconstructed split: raw R² 0.043, median error 79.8%, MAE $53.32.
- **The worked examples reproduce exactly**: material ladder
  +49/+59/+135/+164%, length curve $1.94 → $6.75, all three archetype point
  estimates and 90% ranges.
- **Part 2b (aggregation doesn't help) replicates**: per-signature medians
  give log R² 0.270 (README: 0.305) — same conclusion.
- **New, stronger evidence for the noise claim (E6):** among
  duplicate-signature rows, predicting each bolt from the *other* bolts with
  the identical spec (leave-one-out lookup) scores log R² **−0.17** — worse
  than predicting the subset mean. A bolt's spec-twin carries almost no
  information about its price. The 0.60 ceiling is a generous *in-sample*
  bound; the practical one is lower. The catalog noise is unambiguously real.
- The directional findings survive every configuration we trained: the
  material-premium ordering and the length ruler are robust.

## What does not hold up

1. **"log-price R² ≈ 0.43" is not the published model.** The saved model
   behind the README's raw-R²/MAE/median-error numbers
   (`xgboost_…_110411`) scores **log R² 0.343**. A different model saved 10
   minutes earlier (`xgboost_…_105450`) scores log R² 0.407 with raw R²
   0.139. The README pairs one model's raw R² with the other's log R².
   At 0.343, the published model sits at ~57% of the 0.60 ceiling — not
   "~70%, basically done".
2. **"Not under-tuned" is exactly backwards.** The tuner compares candidates
   (and algorithms) on at most 90 training rows, scored in the wrong space,
   on half the data. The case study did everything right and was still
   betrayed by the harness: re-running the README's own training command
   today selects random forest and lands at log R² 0.282.
3. **Reproducibility:** the README command does not reproduce the README
   model. Same data, same flags, same seed-controlled code path — different
   algorithm, −0.06 log R². The instability is finding #1, not the data.
4. **The trust table calls the diameter ruler "smooth, monotonic" — it
   isn't.** The published model's diameter sweep is U-shaped: $5.39 @ 0.164″
   → $2.39 @ 0.25″ → $7.81 @ 0.5″. Length is clean; diameter is not.
5. **"Median imputation for missing numerics" (Methodology §3) doesn't
   exist.** No imputation happens anywhere in core; rows with any NA are
   dropped wholesale at CSV load.
6. **The ceiling itself moves when you add features** (0.60 → 0.71 with 8
   more specs + ITEM_NAME). "Irreducible within-signature variance" was
   partly "two different bolts that look identical in 10 columns".
7. *(minor, agent-surface)* Under `--json`, the NA-drop warning prints to
   **stdout** before the JSON document, corrupting the machine-readable
   output that the MCP/agent surface depends on.

## Experiment log

| Exp | Configuration | log R² | raw R² | med %err |
|---|---|---|---|---|
| E1 | published model `xgboost_…110411`, reconstructed cc holdout | 0.343 | 0.043 | 79.8 |
| E1 | `xgboost_…105450` (earlier same-day model), same holdout | 0.407 | 0.139 | 76.8 |
| E1 | fresh `p2predict-train --budget thorough` rerun (picks RF) | 0.282 | 0.014 | 80.6 |
| E2 | XGB trained on complete cases → cc test | 0.310 | 0.030 | 79.4 |
| E2 | XGB trained on ALL rows, native NaN → same cc test | **0.342** | 0.045 | 78.3 |
| E2 | XGB all rows → NA-only test rows (unscoreable today) | 0.292 | 0.076 | 81.3 |
| E3 | HPO scored in raw price space (core behaviour) | **−0.254** | 0.122 | 265.0 |
| E3 | identical HPO scored in log space | 0.337 | 0.041 | 80.0 |
| E4 | + 8 specs + ITEM_NAME, all rows (all-rows test) | 0.352 | 0.065 | 77.4 |
| E4 | base features, same all-rows split (control) | 0.319 | 0.054 | 79.3 |
| E5 | per-signature median aggregation (Part 2b re-run) | 0.270 | 0.043 | 79.5 |
| E6 | leave-one-out spec-twin lookup, duplicate rows | **−0.171** | — | — |
| E7 | combined: all rows + extended features + log-space tuning | **0.375** | 0.055 | 76.8 |
| E8 | banded conformal: ×58 (> $155) / ×200 ($5–155) / ×435 (< $5) vs global ×225, coverage ≈ 90% per band | | | |

Training benchmark: the README's full `--budget thorough` command runs in
**51.5 s wall / 4.7 min CPU** on this machine (3 algorithms × HPO × 5-fold).

## Verdict: data problem or tool problem?

**The business conclusion stands; the engineering conclusion doesn't.**

The data really is the binding constraint: no configuration we tried —
doubled training data, ten extra features, honest tuning — moved median
error below ~77% or log R² above 0.375 against an extended-feature ceiling
of ~0.71. E6 shows spec-twins barely predict each other. Point estimates on
this catalog will never be deployable, exactly as the README says, and the
noise-floor methodology is worth keeping as the centerpiece.

But the claim "the model is near the ceiling, so the tool is fine" is not
supported: the tool currently loses ~0.09 log R² (0.282 vs 0.375) to fixable
core behaviours, can't reproduce its own case study, silently refuses half
the catalog, selected its model in a 90-row tournament, and reports one
global ×225 likely-range where a banded ×58 is honest for the parts
procurement actually cares about. None of these change *this* study's
conclusion — they change how much we should trust the tool's verdicts on the
*next* dataset, where the ceiling might be 0.9 and the tuner's noise the
difference between shipping and not.

## Proposed core changes (for discussion — none implemented)

1. **Fix the HPO resource floor** — `min_resources='exhaust'` on
   `HalvingRandomSearchCV`, or switch to `RandomizedSearchCV` outright at
   P2Predict's typical data sizes. Smallest diff, biggest stability win;
   makes algorithm selection meaningful again.
2. **Score CV in log space when log-target is active.** Selection should
   happen in the space the model fits in; raw-space R² under heavy skew
   selected a negative-log-R² model here.
3. **Stop dropping NA rows for tree models** — XGBoost handles NaN natively;
   for ridge, impute + missing-indicator. Never drop rows based on NAs in
   columns that aren't even selected as features. Fix the `--json` stdout
   corruption from the NA warning while there.
4. **Banded (Mondrian) conformal calibration** — calibrate the likely-range
   per predicted-price band instead of one global quantile. The README's own
   page-2 chart (error by price band) is the motivation.
5. *(case study)* Correct the log-R² claim (0.343), the diameter-ruler trust
   row, and the imputation sentence; optionally re-pull with the extra
   characteristics and report the 0.71 ceiling alongside 0.60.

Items 1–3 are small, surgical, and benefit every user. Item 4 is a feature
with visible procurement value. We'd suggest landing 1–2 first: they are the
difference between P2Predict's verdicts being reproducible or not.
