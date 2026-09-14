# Finding: the likely-range is calibrated on 10–60 rows, and below 90 parts it is just the largest residual

> **Status: open.** Per the project's working agreement this note **proposes and
> changes nothing in core**. It quantifies the gap and sketches an *additive*,
> opt-in fix for a follow-up PR. Reproduce everything below with
> `.venv/bin/python research/small_n_conformal.py`.

P2Predict calibrates its likely-range with **split conformal**: fit on 80% of
the rows, take the absolute residuals on the 20% holdout, read a quantile off
them ([`intervals.py:174`](../src/p2predict/intervals.py)). The guarantee is
exact and finite-sample, and on the case-study CSVs — 16k to 400k rows — the
holdout is thousands of parts and the quantile is rock solid.

That is not the size P2Predict is built for. A procurement catalog is
**50–300 parts**, so the calibration set is **10–60 rows**. The guarantee
still holds. What stops holding is any claim that the resulting number is
*stable* or *useful*.

## The mechanism: below 90 parts the quantile degenerates to the maximum

The conformal quantile level is

```python
q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)   # intervals.py:186
```

For a 90% likely-range (`alpha = 0.1`), `ceil((n+1)·0.9) ≥ n` for **every
n ≤ 18**. The `min(1.0, …)` clamp then makes `q_level` exactly 1.0, and the
"90th percentile of the residuals" is **the single largest residual observed**:

| calibration rows | catalog size (20% holdout) | q level | what it reads |
|---|---|---|---|
| ≤ 18 | **≤ 90 parts** | 1.000 | the maximum residual |
| 20 | 100 parts | 0.950 | 2nd largest of 20 |
| 30 | 150 parts | 0.933 | 3rd largest of 30 |
| 60 | 300 parts | 0.967 | 3rd largest of 60 |

This is correct conformal inference — the clamp is what *preserves* the
coverage guarantee at small n, and rounding up is the conservative choice.
But it means that for any catalog of 90 parts or fewer, **one unlucky part
sets the quoted range for every part in the catalog**, and at 150–300 parts
the width is still a 2nd- or 3rd-order statistic of a few dozen numbers.

The aerospace-fasteners run shows what that looks like when the catalog has a
heavy tail. At n=75 with ridge, one draw's maximum log residual produced a
half-width of **10¹²⁰ %**. Not a bug — the arithmetic working exactly as
specified, on the largest of fifteen residuals. The same draw calibrated on
all 75 rows gives 2,349%.

## Measured

Three case-study datasets, subsampled into synthetic catalogs of 75 / 150 /
300 parts, 30 resamples each, graded on a **disjoint 3,000-row evaluation
sample**. The log-target wrap is forced on throughout so half-widths read as
a percentage of the price and are comparable across draws. Target coverage
90%. Full results: [`small_n_conformal_results.json`](small_n_conformal_results.json).

Three calibrations compared:

- **`split`** — what ships today.
- **`cv_cross`** — K-fold cross-conformal. Out-of-fold residuals over all n
  rows give the quantile; the shipped model is trained on all n.
- **`cv_plus`** — CV+ proper (Barber, Candès, Ramdas & Tibshirani 2021,
  *Ann. Statist.* 49(1), Sec. 3). Keeps the K fold-models and builds the
  interval from the ensemble.

At the 150-part anchor:

| dataset / model | method | cal rows | coverage | half-width | **half-width SD** | median APE |
|---|---|---|---|---|---|---|
| used cars / ridge | split | 30 | 93.4% | 213.2% | **195.5** | 35.2% |
| | cv_cross | 150 | 91.1% | 122.1% | **24.3** | 34.1% |
| | cv_plus | 150 | 92.1% | 122.6% | **23.8** | 33.8% |
| used cars / xgboost | split | 30 | 93.1% | 224.0% | **243.7** | 33.9% |
| | cv_cross | 150 | 91.5% | 133.1% | **32.5** | 32.1% |
| | cv_plus | 150 | 94.6% | 133.2% | **25.3** | 29.2% |
| heavy equipment / ridge | split | 30 | 92.8% | 88.0% | **19.7** | 26.1% |
| | cv_cross | 150 | 91.0% | 75.4% | **7.7** | 25.5% |
| | cv_plus | 150 | 90.9% | 74.5% | **7.0** | 25.5% |
| heavy equipment / xgboost | split | 30 | 93.2% | 110.6% | **27.6** | 29.6% |
| | cv_cross | 150 | 91.7% | 94.3% | **12.4** | 28.2% |
| | cv_plus | 150 | 94.6% | 98.0% | **11.5** | 26.6% |
| aerospace fasteners / ridge | split | 30 | 93.3% | 8443.8% | **26582.8** | 90.2% |
| | cv_cross | 150 | 90.7% | 1508.8% | **409.0** | 89.9% |
| | cv_plus | 150 | 91.2% | 1454.0% | **353.0** | 89.8% |

Across all 18 cells (excluding the degenerate fasteners/ridge/n=75 draw
above, which has no finite summary):

- **Split conformal over-covers everywhere**: 91.4%–95.2% actual against a
  90% target, mean **+3.2pp**. Cross-conformal: 90.3%–94.0%, mean +1.5pp.
- **Cross-conformal is narrower in every cell** — median **27%** narrower,
  up to 82%.
- **Width stability improves by a median 4.5×** (SD of the half-width across
  resamples) — up to 65× on the heavy-tailed catalog.
- **Point accuracy improves too**, because the shipped model trains on all n
  instead of 0.8n: median APE better in 17 of 18 cells, by up to 2.9pp.

### Over-coverage is not the safe direction here

A 90% range that actually covers 95% is not cautious, it is uninformative. The
buyer asked what the part should cost; a wider range answers less. Split
conformal at 150 parts quotes ±213% on used cars where ±122% would have held
the promised coverage — the extra width is pure noise in the quantile, not
information about the part.

## The fix: calibrate out-of-fold, keep everything else

**`cv_cross` is the one to ship**, and it is almost entirely additive:

- `compute_calibration_residuals()` gains an out-of-fold path. The dict it
  returns keeps its exact current shape — `residuals`, `predictions`,
  `in_log_space`, `n_calibration` — so `predict_interval()`,
  `coverage_health()`, the banding logic and the persisted model format are
  **untouched**.
- The residual list goes from 0.2n entries to n entries. Mondrian banding,
  which needs 150 calibration points
  ([`intervals.py:105`](../src/p2predict/intervals.py)), becomes reachable at
  a 150-part catalog instead of the ~750-row catalog it needs today — so the
  tercile bands stop being dead code for the typical user.
- Cost: K extra fits at train time. On a 150-part catalog that is
  milliseconds to a few seconds.

**CV+ proper is not worth it here.** It matches `cv_cross` on coverage and
width (the table above), and its only clear edge is a better point estimate
from the fold-model ensemble — but it requires persisting K models instead of
one, which changes the model file format, the registry, and the MCP resource
surface. Its theoretical advantage is a worst-case 1−2α bound that the
measured coverage never comes close to needing. Recorded here so the choice
is on the record, not re-derived later.

### Suggested shape, if this is taken up

- `--calibration {split,cv}` on `p2predict-train`, **default `split`**. Every
  existing model, script and case study behaves identically.
- `propose_training_plan` recommends `cv` when the catalog is under ~500
  rows, in plain language: *"your catalog is small enough that the likely-range
  would be set by a handful of parts; calibrating across the whole catalog
  instead makes it about a third tighter."*
- The persisted `calibration` dict records which path produced it, so
  `get_model_quality` can say so.
- Axiomatic test, matching the existing interval suite: empirical coverage
  within ±5pp at 80/90/95% on synthetic data, calibrated both ways.

## What this does not fix

The width is still **constant across parts** — cross-conformal makes it
tighter and far more stable, but an unusual part still gets the same range as
a routine one. That is Tier-2 #6 in
[`methodology_review.md`](methodology_review.md) and wants normalized
conformal scores or CQR, which is a separate study.

It also does not change the point prediction's mean-vs-median question
(see [`log_retransformation_bias.md`](log_retransformation_bias.md)), and it
does not touch the out-of-domain flag, which is the guard for parts the model
never saw.
