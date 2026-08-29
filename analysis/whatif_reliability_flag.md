# Finding: `what_if` hands over unreliable directions with no caution flag

`what_if` is the one action a category manager acts on directly — "dropping to
multi-chemistry saves 1%, let's do it" — and it is the one action in the MCP
surface that ships its result **without a built-in reliability flag**. When a
change lands on a thinly-sampled spec, or when the reported direction is driven
by interactions rather than the change itself, the payload still reports a
confident `"direction": "saves", "effect_pct": -8.0` with no caveat attached.

Compare the rest of the surface, which already self-flags in the data:

| Action | Reliability flag in the payload? | Field |
|---|---|---|
| `predict_interval` | yes | `interval.reliability` + `interval.say_to_user` (`quality.py:114–152`) |
| `get_model_quality` | yes, per feature | `feature_importance[].signal` + `say_to_user` (`quality.py:84–90, 155–170`) |
| `what_if` | **no** | — just `direction` / `effect_pct` / `interaction_is_material` |

The safeguard today lives entirely in the **guidance layer**: the MCP server
instructions tell the agent to sign-check drivers and flag counterintuitive,
weakly-sampled ones as noise. A careful agent obeys. A weaker or less obedient
model reads "going to 125°C saves you 12%" off the payload and relays it as a
negotiation lever. That is a trust gap that shouldn't depend on which model is
driving — the other two actions don't.

Per the project's working agreement, this note **proposes and changes nothing
in core**. It quantifies the gap and sketches the fix for a follow-up PR.

## Measured — the battery-management-IC model

Model `xgboost_unit_price_at_1_usd_20260613_161828` (verdict *usable*,
unbiased, ~13% typical error, 30 holdout parts). Base part: multi-cell Li-Ion,
I²C, 8-pin, 2 cells, −25 °C…85 °C, one supplier held fixed → base $3.26. Each
row is a single-spec `what_if`, with the changed feature's importance signal
from the same model's quality report:

| Change | Reported | Feature signal | Direction a buyer expects | Silent? |
|---|---|---|---|---|
| pins 8 → 16 | **+44.6%** | strong (15.0%) | costs more | ok — correct |
| temp floor −25 → −40 °C | 0.0% | moderate (7.4%) | ~free | ok — genuinely flat |
| cells 2 → 4 | −8.0% | moderate (9.8%) | costs **more** | ⚠️ wrong sign |
| max temp 85 → 125 °C | −11.9% | moderate (9.2%) | costs **more** | ⚠️ wrong sign |
| add SPI to I²C | −7.3% | strong (10.8%) | costs **more** | ⚠️ wrong sign |
| Li-Ion → multi-chem | −1.1% | strong (12.4%) | costs **more** | ⚠️ wrong sign |

Four of six single-spec what-ifs came back with a commercially wrong sign, each
reported flatly as "saves". On this run the driving agent (an Opus-class model)
caught all four and captioned them as noise — exactly because the server
guidance told it to. Nothing in the `what_if` payload would have stopped a
model that didn't.

## Two flags the payload can carry — both already computable

The tool cannot know the *domain-expected* sign (it has no prior that "more
cells should cost more"). But it can flag the two internal conditions that
produce most of these silent misses, and both inputs already exist:

**(a) Thin-data flag — reuse `feature_signal`.** If a changed feature's
importance signal is `moderate` or `weak`, its direction rests on few parts and
should not be quoted. `feature_signal()` / `feature_say_to_user()` already exist
(`quality.py:84, 155`); the `what_if` handler already holds the model and
`background_X`, and `extract_feature_importances(model, background_X)` is already
used elsewhere in the server (`server.py:994`). Catches: cells, max temp.

**(b) Interaction-driven / sign-flip flag — already in `WhatIfResult`.** The
decomposition splits the delta into `changed_contributions` (the effect of the
feature you actually changed) and `interaction_contribution` (everything else
shifting). Two tells, computed from fields that already ship:

  - **Interaction dominates:** `|interaction_contribution| ≥ |Σ changed_contributions|`
    — the reported move isn't coming from the change you made. On *cells 2→4*
    the direct effect is −0.039 but the interaction term is −0.044, larger than
    the change itself.
  - **Sign flip:** the changed feature's own contribution and the net delta
    disagree in sign. On *Li-Ion → multi-chem* the direct contribution is
    **+0.014 (adds)**, but interactions (−0.025) flip the headline to
    "saves 1%." The number contradicts its own driver.

Flag (b) needs no new computation at all — just a comparison of two numbers
already on the result.

## Honest limits of the proposal

- **It will not catch every wrong sign.** *Add SPI to I²C* (−7.3%) has a
  strong-signal, direct-dominated, non-sign-flipped delta — neither flag fires,
  yet the direction is still commercially backwards. No purely-internal signal
  catches that; it needs a domain prior the model doesn't have. So the
  server-side guidance sign-check should **stay** even after the payload flag
  lands — belt and suspenders, not a replacement.
- **It flags reliability, not truth.** A moderate-signal feature can still have
  the right sign; the flag says "don't quote this direction," not "this is
  wrong."
- **Thresholds are shared with quality.** Reusing `FEATURE_STRONG_MIN_PCT` /
  `FEATURE_MODERATE_MIN_PCT` keeps one definition of "strong" across the
  surface; no new tunables.

## Sketch of the core change (for the follow-up PR)

1. In `compute_whatif` (or the MCP `whatif_to_dict` conversion), compute the
   changed features' signals via `extract_feature_importances` + `feature_signal`.
2. Derive a `reliability` ∈ {`trust`, `caution`, `quote`} from: weakest changed
   signal, interaction-dominance, and sign-flip.
3. Attach `reliability` + a plain `say_to_user` line to the `whatif` summary,
   mirroring `interval_say_to_user` phrasing ("directional only — don't hold a
   supplier to this move").
4. Tests: the four battery cases above as regression fixtures — assert
   `caution`/`quote` on cells, max-temp, and multi-chem; assert `trust` on pins.

## Why it's worth doing

`what_if` is the highest-leverage, most-acted-on tool in the surface, and it is
the only one whose safety currently depends on the agent rather than the data.
The fix reuses machinery that already exists, adds no new tunables, and closes
the gap for the weaker models that production deployments may put in front of
it. Recommend scheduling the core change as a small, self-contained follow-up.
