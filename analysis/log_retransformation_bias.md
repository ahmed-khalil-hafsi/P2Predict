# Findings: log-target models are flagged "unreliable" for a mean-vs-median gap, not a modelling failure

**Purpose.** The heavy-equipment resale case study trained a log-target model that scored a healthy holdout R² of 0.738 ("Good") yet earned the computed verdict **"unreliable — its single-number estimates aren't trustworthy."** This document asks whether that verdict reflects a real modelling failure or an artifact of how P2Predict back-transforms log predictions and how the honesty layer defines "bias." All numbers come from `log_retransformation_bias.py` (committed alongside; results in `log_retransformation_bias_results.json`). It trains a throwaway model on the committed 5k heavy-equipment sample so the finding reproduces with no Kaggle account, and additionally reports the on-disk 80k case-study model. **No core code was changed — this proposes findings for discussion only.**

## TL;DR

The model is fine. The verdict is triggered by a **~5% systematic under-shoot of the dollar *mean*** that comes entirely from P2Predict's own `exp()` back-transform — while the same predictions are **essentially unbiased for the *median***. P2Predict's `residual_bias_p` test only checks the mean, so it stamps a good median-unbiased model "unreliable," which needlessly withdraws the single-part appraisal use case on the most common procurement data shape (skewed prices).

| | 5k sample (reproducible) | 80k case-study model |
|---|---|---|
| Holdout R² | 0.733 | 0.738 |
| **Mean** residual (raw `exp()`) | **+4.7%** | **+4.8%** |
| **Median** residual (raw `exp()`) | **+0.8%** | **+1.3%** |
| `residual_bias_p` (mean test) | 6.0e-5 → **flagged** | 1.2e-60 → **flagged** |
| Duan smearing factor | ×1.043 | ×1.046 |
| Mean residual after smearing | +0.6% | +0.4% |
| `residual_bias_p` after smearing | 0.62 → **passes** | 0.177 → **passes** |
| R² after smearing | 0.737 | 0.744 |

The Duan (nonparametric) factor and the log-normal factor `exp(σ²/2)` agree to within 0.01 on both models, which says the log-scale errors are close to homoscedastic here and the constant correction is well-behaved.

## The mechanism

P2Predict wraps a skewed target and predicts on the log scale, `m(x) ≈ E[log Y | x]`, then reports dollars as `exp(m(x))`. Two facts about that step:

1. **It under-shoots the arithmetic mean.** With `log Y = m(x) + ε` and `E[ε]=0`,
   `E[Y|x] = exp(m(x))·E[exp(ε)|x]`, and `E[exp(ε)] ≥ exp(0) = 1` by Jensen. So `exp(m(x))` is systematically **below** `E[Y|x]`. This is textbook retransformation bias (Duan, *JASA* 1983).
2. **But it hits the median.** `exp(E[log Y])` is the **geometric mean**, which for a right-skewed price sits at roughly the **median**. So the identical prediction that is mean-biased is close to median-*un*biased — confirmed empirically above (median residual ≈ +1%).

The honesty layer's `residual_bias_p` runs a one-sample test on the **mean** residual. A median-unbiased-but-mean-biased predictor fails it by construction, regardless of how well the model actually fits — which is exactly what happened here (R² 0.74, but flagged).

Why it matters beyond this case study: **every** log-target model has this property, and log-target is the default for the skewed price/cost data P2Predict is built for (the used-vehicle study is also log-target). The flag isn't specific to bulldozers; the mechanism is general.

## Two ways to close the gap

**Option A — correct the back-transform toward the mean (Duan smearing / log-normal).**
Multiply dollar predictions by `S = mean(exp(ε_train))`. Measured effect: mean residual → ~0, the bias flag clears, R² even ticks up slightly.
- *Pro:* the point estimate becomes a true expected-value ($) benchmark; the flag disappears for the right reason.
- *Con — non-trivial blast radius:* it silently redefines every existing prediction from ~median to mean, so users comparing against last quarter's numbers would see a step change. It also interacts with the **conformal intervals**, whose coverage is calibrated on the *current* (uncorrected) predictions — shifting the point estimate without recalibrating could distort coverage. (SHAP multiplicative factors are safe: a global scalar cancels in the ratios.)

**Option B — fix the verdict logic, offer the correction opt-in (recommended).**
Teach `assess_model` that a log-target model which is median-unbiased is *usable*, and report which sense of "typical price" the number represents (median/geometric-mean) instead of stamping "unreliable." Expose the smearing/log-normal correction as an explicit choice (e.g. a `--price-estimate median|mean` flag) rather than changing everyone's numbers by default.
- *Pro:* restores the appraisal use case on skewed data with almost no blast radius; keeps intervals valid; is honest about mean vs median rather than hiding the choice.
- *Con:* doesn't make the default number an expected-value benchmark — users who want the mean have to opt in.

## Recommendation

Do **not** ship a blind global rescale (Option A by default). The measured win is real but the semantic change to every prediction and the interval-coverage interaction are exactly the kind of quiet correctness risk P2Predict's honesty layer exists to avoid. Prefer **Option B**: stop mislabelling median-unbiased log models, name the mean-vs-median choice explicitly, and make the mean correction opt-in with its own interval recalibration. That is a genuine capability gain — it returns trustworthy single-part appraisal on the most common data shape — at low risk.

## What this finding does *not* claim

- It does **not** claim the model is accurate enough to appraise to the dollar. Separate data ceilings cap that: no make/model column, machine-hours blank on ~83% of records, coarse features, and irreducible auction noise. Those cause **scatter**, not the bias flag, and are not addressed here.
- It does **not** claim smearing is R²-optimal. The R²-minimising scalar is the least-squares scale `Σyŷ/Σŷ²`, not the smearing factor; smearing targets *unbiasedness of the mean*, and the small R² improvement is incidental.
- The constant-factor correction assumes roughly homoscedastic log-errors. That holds on these two models (Duan ≈ log-normal factor); on a model with strongly price-dependent log-variance a band-wise correction would be needed, and the verdict-logic fix (Option B) is unaffected either way.
