# Finding: the "unreliable" gate measures holdout size, not bias

**Status: open.** Sharpens finding #2 of [`methodology_review.md`](methodology_review.md)
and shares a root cause with [`log_retransformation_bias.md`](log_retransformation_bias.md).
All numbers from `bias_gate_materiality.py` (results in
`bias_gate_materiality_results.json`). **No core code changed.**

> **Read the [addendum](#addendum-2026-09-11-the-floor-is-the-wrong-question--test-the-interval) at the end first if you are implementing this.**
> It supersedes the proposed fix below: the materiality floor is replaced by an
> equivalence test on the median relative residual, which subsumes the log-space
> and small-n items and comes with measured operating characteristics. The
> analysis of the *flaw* in this document stands unchanged.

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

---

# Addendum (2026-09-11): the floor is the wrong question — test the interval

The proposed fix above needs a materiality floor, and the obvious way to pick
one is to calibrate it on our data. That doesn't work: we have three case
studies, and users price everything from castings to cloud contracts. A
constant fitted to bulldozers, used cars and battery ICs is a constant fitted
to three categories and shipped to all of them.

So this addendum asks a different question. Instead of *"how big is bias in the
wild"* (unknowable from here), it measures the **operating characteristics of
the test itself** on a synthetic grid where the true bias is known by
construction — false-flag rate, miss rate, and whether the verdict tracks bias
or sample size. Those are properties of the gate, not of anyone's dataset, so
they generalise. The case studies then serve as spot-checks, which is the right
job for three datasets.

All numbers from `bias_gate_equivalence.py` (results in
`bias_gate_equivalence_results.json`). **No core code changed.**

## The design: stop testing the point, test the interval

The root flaw at `quality.py:200` is that *failure to reject* is read as *proof
of no bias*. Adding a floor to a point-null test patches the symptom. The
textbook fix for that exact error is an **equivalence test**: compute a
confidence interval on the bias and ask where it sits relative to a materiality
band `±m`.

| Where the median CI sits | Verdict | Meaning |
|---|---|---|
| Entirely inside ±m | pass | Confidently immaterial |
| Entirely outside ±m | `unreliable` | Confidently material |
| Straddles an edge | `unknown` | We genuinely can't tell |

Three things fall out of this that are worth more than the floor itself.

**1. It measures the median relative residual, which dissolves proposal item 2.**
The median of `y/ŷ − 1` is a monotone transform of the median log residual, so
testing it *is* the log-space test — for log-target and additive models alike,
with no branch on `log_target`. One rule, no special case.

**2. The interval is deterministic.** A bootstrap CI would make a user-facing
*verdict* depend on an RNG seed — the same model judged differently on two
runs. The distribution-free **order-statistic** CI for the median (the
`(k_lo, k_hi)` order statistics whose binomial tail mass sits below `α/2`)
needs no resampling, assumes no distribution, and is exact. It is also O(n log n),
which is why sweeping 891 grid cells was affordable.

**3. Proposal item 3 stops being a separate rule.** The small-n free pass isn't
patched by a special case; it disappears, because at small n the interval is
wide, straddles the band, and returns `unknown` on its own.

## Measured: operating characteristics

891 cells — holdout size × true bias × noise × price spread — 400 simulated
holdouts each. "False-flag" is calling a truly immaterial bias (≤1%)
`unreliable`; "miss" is passing a clearly material one (≥10%) as trustworthy.
"Inversion" is the original complaint in one number: hold the model fixed, vary
only `n`, and score whether the gate both *passes* it at one size and calls it
*unreliable* at another. A gate that measures bias scores ~0. A gate that
measures sample size scores ~1.

| gate | false-flag | miss | inversion | unknown (immaterial) | unknown (material) |
|---|---|---|---|---|---|
| **today** | **37.3%** | **42.0%** | **0.99** | — | — |
| m = 2% | 0.9% | 0.0% | 0.04 | 77.4% | 26.9% |
| m = 3% | 0.6% | 0.0% | 0.04 | 67.5% | 29.6% |
| **m = 5%** | **0.3%** | **0.0%** | **0.04** | 55.6% | 36.2% |
| m = 7% | 0.2% | 0.0% | 0.02 | 47.0% | 45.7% |
| m = 10% | 0.1% | **7.2%** | 0.04 | 37.1% | 64.8% |

Today's gate flags a truly unbiased model 37% of the time and waves through a
materially biased one 42% of the time, and its inversion score is 0.99 — it is
very nearly a pure measurement of `n`. Every equivalence variant scores under
0.05.

The two `unknown` columns are the honest cost. They are not errors — they are
the gate declining to answer — but they are the usability price, and they are
what the floor actually trades against.

## Choosing `m`: anchor it normatively, then verify

The sweep cannot tell us what is *material to a buyer* — that is not a
statistical question. Anchor it in the work instead: a category manager
benchmarks in order to negotiate, and typical negotiated movement is **2–5%**.
A systematic offset as large as the saving being chased corrupts the decision;
one well under it does not. That gives a band in the 3–5% region on grounds
that hold for any category, because what's constant is the negotiation, not the
part.

The sweep then picks within that range, by a stated rule: **take the loosest
band that never green-lights a materially biased model**, since widening buys
usability (less `unknown`) and tightening buys nothing once the miss rate is
already zero. That is **m = 5%** — 7% also misses nothing but sits outside the
normative range, and 10% starts missing real bias (7.2%).

Note the circularity this does *not* have: "miss rate" is scored against the
≥10% definition of clearly-material, which comes from the normative anchor, not
from the data. The sweep chooses *how to implement* the anchor safely; it does
not choose the anchor.

## What a holdout of size n can actually resolve

The median CI half-width is a hard limit: no band below it is resolvable at
that holdout size, so a tighter floor buys `unknown`, not safety.

| n holdout | noise 15% | noise 30% | noise 50% |
|---|---|---|---|
| 30 | ±6.9% | ±14.0% | ±23.4% |
| 50 | ±5.7% | ±11.2% | ±19.2% |
| 100 | ±4.0% | ±7.9% | ±13.2% |
| 250 | ±2.4% | ±4.6% | ±7.7% |
| 500 | ±1.7% | ±3.4% | ±5.6% |
| 2,500 | ±0.7% | ±1.5% | ±2.5% |

**At typical noise, a 5% band needs roughly n ≥ 250.** Below that the honest
answer is `unknown`. That is a real product change — today most small models
read `trustworthy` — and the two tables below are why it is the right one.

A **perfectly unbiased** model, and what each gate says:

| n | today | proposed: pass / unknown / unreliable |
|---|---|---|
| 30 | trustworthy (6% flagged) | 1% / 98% / 1% |
| 50 | trustworthy (9%) | 2% / 98% / 0% |
| 250 | trustworthy (34%) | 39% / 61% / 0% |
| 1,000 | **unreliable (66%)** | 79% / 21% / 0% |
| 16,000 | **unreliable (99%)** | **100% / 0% / 0%** |

A model reading **10% high** — one a buyer would feel — and what each gate says:

| n | today | proposed: pass / unknown / unreliable |
|---|---|---|
| 30 | **trustworthy (24% flagged)** | 0% / 88% / 12% |
| 50 | **trustworthy (31%)** | 0% / 85% / 15% |
| 250 | trustworthy (49%) | 0% / 45% / 55% |
| 2,500 | unreliable (78%) | 0% / 2% / 98% |

The second table is the safety case. Today's gate calls a 10%-biased model
`trustworthy` about three quarters of the time at n = 30, and the proposed gate
**never passes it at any holdout size** — 0% in every row. It says `unknown`
instead, which is true.

## Make `unknown` carry its bound

`unknown` as it reads today is a shrug: *"couldn't tell whether this model runs
systematically high or low."* But the CI half-width is exactly the information
the user needs, and we have already computed it. Report it — call it
`bias_resolution_pct` — and `unknown` becomes a quantified statement:

> With 30 parts kept back, a systematic offset up to about ±11% wouldn't show
> up. Treat the number as indicative and sanity-check it against a quote.

That is strictly more useful than today's false confidence, and it turns the
usability cost of the `unknown` region into a feature rather than a downgrade.
**Recommend adding this to the proposal.**

## Spot-checks: the three case studies

Refit throwaway (xgboost, seed 11, 80/20), so these compare to the main
document's numbers rather than to the on-disk models.

| dataset | n | R² | median residual | median CI | today | proposed (m=5%) |
|---|---|---|---|---|---|---|
| used cars | 16,000 | 0.735 | +1.7% | [+1.2%, +2.3%] | `unreliable` | **pass** |
| heavy equipment | 16,000 | 0.737 | +1.7% | [+1.3%, +2.3%] | `unreliable` | **pass** |
| battery mgmt ICs | 30 | 0.606 | −0.8% | [−11.4%, +10.7%] | `trustworthy` | **`unknown`** |

Both flagship models clear the band with room to spare, and the n-ladder
confirms the inversion is gone on real data: the proposed gate flags used cars
in **0%** of draws at every holdout size from 30 to 16,000, where today's gate
goes 12% → 100%. The battery model moves the other way, to `unknown`, with a
resolution bound of about ±11% — which is the honest read on 30 parts.

## Reconciling a number with the main document

This run reports a used-cars **mean** residual of **+8.8%**; the main document
reports **−27.2%**. Both are correct, and the gap is the point. The main
document computes `(y − ŷ)/y`, this one `y/ŷ − 1`. For a right-skewed price
catalogue those two denominators disagree violently — cheap cars dominate the
first, and the sign flips.

The **median is identical in both parametrisations: +1.7%.** It is invariant to
a choice the mean is hostage to. A headline verdict that swings from −27% to
+9% on an arbitrary denominator convention is not measuring a property of the
model.

## What this changes in the proposal above

| Original item | Status |
|---|---|
| 1. Effect-size floor | **Keep, reframed.** Not a floor bolted onto the p-test — a materiality band in an equivalence test. m = 5%. |
| 2. Test in log space for log-target models | **Dissolved.** The median relative residual is the log-space test, with no branch. |
| 3. `unknown` instead of a small-n free pass | **Keep — now free.** Falls out of the interval rather than needing its own rule. |
| — | **New: report `bias_resolution_pct`** so `unknown` states the bound it can rule out. |

## Limits

- The synthetic grid models residuals directly (lognormal prices, multiplicative
  noise, a known median offset). It deliberately does not model feature-dependent
  or heteroscedastic bias, where a single median understates the harm to a
  particular price band. The existing per-band calibration block is the
  safeguard there, and should keep its independent veto.
- 400 draws per cell puts roughly ±2pp of Monte-Carlo error on the rates in the
  headline table. The gaps being argued from are tens of points wide.
- The 2–5% negotiation anchor is a judgement about procurement practice, stated
  so it can be argued with. It is the one number here that is not measured.
- `noise` in the grid is log-scale residual spread, which maps to typical % error
  only approximately at the 50% end.

---

# Revision (2026-09-11, same day): three states was too blunt for the users we have

The addendum above collapses the equivalence test into three outcomes and
reports strong numbers for it (miss 0.0%, inversion 0.04). Shipping it that way
would have been a mistake, for a reason the synthetic grid hid: **it averages
over holdout sizes P2Predict users don't have.**

Most users train on **100–300 parts**, which after the 80/20 split is a **20–60
part holdout**. (At 50 parts the holdout is 10, already below
`MIN_HOLDOUT_FOR_JUDGMENT`, so those read `insufficient_data` today regardless.)
At that size a 5% band is never resolvable — so under the three-state design
**essentially every real user reads `unknown`**, whatever the quality of their
model. A verdict that returns the same answer for a good model and a bad one
carries no information.

Worse, `unknown` is a rhetorical overclaim. "We cannot certify the lean is
under 5%" is true; "we know nothing" is not. Measured — of the models whose
interval straddles the band, where does the point estimate actually land?

| holdout | model noise | truth | estimate lands on the correct side |
|---|---|---|---|
| 60 | 15% | unbiased | **95%** |
| 60 | 15% | 10% biased | **95%** |
| 30 | 15% | unbiased | 85% |
| 30 | 30% | unbiased | 54% |
| 20 | 30% | unbiased | 46% |

On a clean 300-part model the estimate is right 95% of the time. Discarding it
throws away the most useful thing we have about exactly the users who need it
most. At 100 parts with noisy data it genuinely is a coin flip — so how much
nuance is warranted *depends on the model*, which one flat label cannot express.

## Four states

| median CI vs ±band | status | verdict |
|---|---|---|
| entirely inside | `immaterial` | `trustworthy` / `usable` |
| straddles, estimate inside | `likely_immaterial` | `trustworthy` / `usable`, **blind spot stated** |
| straddles, estimate outside | `likely_material` | `unknown`, **with a directional warning** |
| entirely outside | `material` | `unreliable` |
| no interval possible | `unmeasured` | `unknown` |

`likely_material` deliberately does **not** promote to `unreliable`. At a 20–60
part holdout the estimate alone is far too noisy to condemn a model on — doing
so would rebuild the old gate in reverse, flagging good models for having a
small holdout. It gets a warning, not a verdict.

## What it costs, measured

Re-running the same 891-cell grid with the four-state mapping:

| gate | false-flag | miss | inversion |
|---|---|---|---|
| today | 37.3% | 42.0% | 0.99 |
| three states (addendum above) | 0.3% | **0.0%** | **0.04** |
| **four states (shipped)** | **0.3%** | **5.7%** | **0.36** |

Both regressions are real and both are concentrated at small holdouts, which is
the point — that is where a pass is now granted on an estimate instead of a
proof. The honest read, at a 30-part holdout:

| | good model earns a verdict | 10%-biased model green-lit |
|---|---|---|
| today's gate | 94% (but meaningless — it can't fail) | **76%** |
| three states | 1% | 0% |
| **four states** | **58%** | **20%** |

So the trade is: a genuinely biased small model slips through 20% of the time
instead of 0% — against 76% today — and in exchange a good small model gets a
usable answer 58% of the time instead of 1%. Given the user base is almost
entirely in this regime, the three-state version optimised a number
(`miss = 0`) by making the feature useless for nearly everyone.

`inversion` rising to 0.36 is the one genuinely unattractive number: near the
band edge a model can pass at a small holdout and be flagged at a large one.
That is inherent to granting a pass on an estimate, and it is bounded by the
fact that the small-holdout pass always ships with its blind spot attached.

## Where the bound actually bites

The caveat on a `likely_immaterial` pass names the right risk rather than
hedging generally:

> Only 30 part(s) were kept back to check it, so an error up to about 11% could
> still be hiding — fine for comparing options and pricing single parts against
> the likely-range, but sanity-check a whole-basket target against a quote.

The distinction is real. For a **single part**, the conformal likely-range
already carries the full uncertainty honestly, so the bias verdict barely
matters. Across a **basket or an annual target**, random error cancels and a
systematic lean does not — which is exactly where an unquantified blind spot
would hurt. Same number, two different exposures, and only one of them needs
the warning.

## Spot-check, through shipped core

| dataset | holdout | before | after |
|---|---|---|---|
| used cars | 16,000 | `unreliable` | `trustworthy` (lean +1.7%, certified) |
| heavy equipment | 16,000 | `unreliable` | `trustworthy` (lean +1.7%, certified) |
| battery mgmt ICs | 30 | `trustworthy` | `trustworthy` **+ "an error up to about 11% could still be hiding"** |

The battery model is the case that moved twice. Today it reads `trustworthy`
with no caveat at all; under three states it read `unknown`; it now reads
`trustworthy` **with its blind spot quantified** — which is the only one of the
three that is both usable and true.
