# Finding: the "unreliable" gate measures holdout size, not bias

**Status: open.** Sharpens finding #2 of [`methodology_review.md`](methodology_review.md)
and shares a root cause with [`log_retransformation_bias.md`](log_retransformation_bias.md).
All numbers from `bias_gate_materiality.py` (results in
`bias_gate_materiality_results.json`). **No core code changed.**

## TL;DR

`assess_model()` calls a model biased when `residual_bias_p <= 0.05`, where
`residual_bias_p` (`quality.py:333`) is a one-sample t-test of price-space
residuals against zero. A p-value answers *"could this bias be noise?"* — a
question whose answer depends on sample size. It does not answer *"is this
bias big enough to matter?"*, which is what the verdict claims to report.

Consequence, measured: **hold the model and its bias exactly fixed, vary only
the holdout size, and the verdict flips.** Same predictions, same error, from
`trustworthy` to `unreliable` — because n grew.

| n holdout | median mean-residual | median bias p | % of draws flagged `unreliable` |
|---|---|---|---|
| 50 | −5.4% | 3.2e-01 | **10%** |
| 100 | −5.4% | 1.9e-01 | 23% |
| 250 | −6.1% | 6.4e-02 | 47% |
| 500 | −5.7% | 6.6e-03 | 80% |
| 1,000 | −5.7% | 9.2e-05 | 96% |
| 2,500 | −5.8% | 1.1e-09 | 100% |
| 16,000 | −5.8% | 9.8e-55 | **100%** |

Heavy-equipment model, holdout R² 0.737; 200 random draws per row. The bias
column is flat. The verdict column goes from 10% to 100%.

## The flaw in one line

`unbiased = residual_bias_p > UNBIASED_P` (`quality.py:200`) treats **failure
to reject** as **proof of no bias**. Those are not the same claim, and the gap
between them is exactly sample size. So the gate fails in *both* directions:

- **Large holdout → guaranteed `unreliable`.** Any systematic offset, however
  commercially irrelevant, reaches p < 0.05 once n is big enough. The user with
  the best price history is the most likely to be told their model can't be
  trusted.
- **Small holdout → free pass.** The battery-management-IC model carries a
  **−7.5% mean residual** on 30 holdout parts and is stamped `trustworthy`
  (p = 0.59), because 30 points can't reject anything. This is the dangerous
  direction, and it is the one the July review explicitly ruled out when it
  said "failure direction is under-trust, so live users see over-caution, not
  wrong prices." Over-caution is the large-n direction. Small-n is over-trust.

## What the verdict costs

`unreliable` is not a footnote. The MCP server instructions tell the agent to
lead with the verdict, and its headline says the model's "single-number
estimates aren't trustworthy. Use it only to compare options, not to set an
absolute target" (`quality.py:218`) — it withdraws the single-part appraisal,
which is the product's main use. Both flagship case studies currently earn it:

| Dataset | R² | mean resid | median resid | p (price space) | verdict today |
|---|---|---|---|---|---|
| used cars (16,000 holdout) | 0.735 | −27.2% | **+1.7%** | 4.4e-71 | `unreliable` |
| heavy equipment (16,000 holdout) | 0.737 | −5.8% | **+1.7%** | 9.8e-55 | `unreliable` |
| battery mgmt ICs (30 holdout) | 0.606 | −7.5% | −0.8% | 5.9e-01 | `trustworthy` |

Note the median column: both large models are within ±2% on the *typical*
part. What fails them is the mean, and for a log-target model the mean is the
wrong statistic — `exp(E[log Y])` targets the median by construction
(mechanism in [`log_retransformation_bias.md`](log_retransformation_bias.md)).

## Correction to the July review

The review reported that testing in log space flips both case-study models to
`trustworthy`/`usable`. Independently refit here (xgboost, seed 11, 80/20):

- **heavy equipment reproduces** — p goes 9.8e-55 (price) → **0.585** (log),
  verdict `unreliable` → `trustworthy`.
- **used cars does not** — p goes 4.4e-71 → **0.034**, still under 0.05, still
  `unreliable`. Its mean residual is −27.2%, far larger than a retransformation
  artifact; the heavy right tail of the vehicle catalog leaves real mean bias
  that a log-space test should not excuse.

Different fit, different numbers — this doesn't make the review wrong about
the mechanism. But it does mean **testing in log space is not on its own a
sufficient fix**, and a p-value at 0.034 vs 0.585 deciding a headline verdict
is itself the argument for a materiality floor.

## Proposed fix

Two changes in `quality.py`, no model-format change, no retrain:

1. **Add an effect-size floor.** Flag bias only when it is *both* statistically
   detectable *and* commercially material — e.g. `|median relative residual| >
   5%`. Judge the number a buyer would feel, not the one n inflates.
2. **Test where the model is unbiased.** For log-target models run the test on
   log residuals, so a median-unbiased model isn't failed for a mean it never
   claimed to estimate.
3. **Stop reading small-n silence as a pass.** Below the size where a material
   bias would be detectable, the honest verdict is `unknown`, which already
   exists — not `trustworthy`.

Item 3 matters most for safety and is the one nobody has proposed yet.

## Limits

- Models here are refit (xgboost, no tuning, seed 11) rather than the on-disk
  case-study models, which don't store signed holdout residuals. R² lands
  within ~0.04 of the published figures; the effect is a property of the test,
  not of a particular fit.
- A 5% floor is a starting point, not a derived constant. Picking it properly
  wants a look across more models than three.
- Fixing the gate does not make a mean-biased model mean-unbiased. It stops
  *mislabelling*; whether to also correct the back-transform is the separate
  open question in `log_retransformation_bias.md`.
