---
name: p2predict
description: >-
  Drive P2Predict — parametric price/cost benchmarking for procurement — to
  train a model, predict a price, explain why a part costs what it does
  (SHAP), show a likely-price range (conformal intervals), and run supplier
  what-if swaps. Use this skill whenever the user wants to benchmark or
  estimate the price/cost of parts, components, materials, or a BOM; build or
  train a pricing/cost model from a spec-and-price CSV; sanity-check a supplier
  quote against a defensible target; quantify a brand/supplier premium; find
  cost-down or switch-cost opportunities; or interpret a P2Predict model's
  output. Trigger it even when the user doesn't name P2Predict — e.g. "is this
  quote fair", "what should this connector cost", "how much does going to a
  16-cell pack add", "which supplier premium can we negotiate away", or they
  drop a CSV of parts-with-prices and ask what drives cost. Also use it to read
  and present P2Predict results to a procurement/category-manager audience.
---

# P2Predict operator

P2Predict turns a CSV of *specs + historical price* into a model that
**predicts a fair price**, **explains** each prediction feature-by-feature,
puts an **honest likely-range** around it, and runs **what-if** supplier or
spec swaps. It is built for procurement: BOM benchmarking, quote sanity-checks,
should-cost models, supplier-premium quantification.

The hard part of using P2Predict well is **not** running the commands — it's
**interpreting the output and knowing which numbers to trust.** This skill
front-loads that judgment. Read "The interpretation rules" carefully; that's
where the value is.

---

## The shape of every job

P2Predict has two surfaces over one core. Prefer the **CLI** — it's what the
case studies use, it's the most stable surface, and `--json` makes it
agent-friendly. Reach for the **Python API** only when embedding in a script
or notebook.

A typical job is a pipeline:

```
CSV (specs + price)  ─►  train  ─►  .model file
                                       │
            predict ─ explain ─ interval ─ whatif  (on new parts)
```

1. **Train** a model from historical data (`p2predict-train`).
2. **Predict** a fair price for a new part (`p2predict`).
3. **Explain** the prediction — per-feature attribution (`--explain`).
4. **Interval** — the likely range, a per-part confidence signal (`--interval`).
5. **What-if** — change supplier/spec, get the delta (`--whatif`).

You rarely need all five. Match the tool to the question — see "Map the
question to the tool" below.

---

## CLI reference (the primary surface)

### Train — `p2predict-train`

```bash
p2predict-train -i parts.csv -t unit_price \
    -tf "manufacturer,package_pins,op_temp_max_C,interface" \
    --budget thorough \
    --log-target on \
    --outliers warn --feature-outliers warn \
    --report quality_report.pdf \
    --json
```

| Flag | Meaning | Guidance |
|---|---|---|
| `-i, --input` | Training CSV | Each row = one part: spec columns + one price column. |
| `-t, --target` | Price column name | The thing to predict. |
| `-tf, --training_features` | Comma-separated feature columns | Pin these explicitly for reproducibility. Omit to let it auto-select (capped by `--max-features`, default 6). |
| `-b, --budget` | `fast` \| `thorough` | `thorough` does real hyperparameter search; use it for anything you'll quote from. |
| `--log-target` | `auto` \| `on` \| `off` | **Read the log-target rule below — this is the highest-leverage flag.** |
| `--outliers` | `keep`/`warn`/`drop`/`winsorize` (target) | Default `warn`. |
| `--feature-outliers` | same choices (numeric features) | Default `warn`. **`drop` is dangerous on small data — see the rule.** |
| `-a, --algorithm` | force `ridge`/`random_forest`/`xgboost` | Omit to let CV pick the best. |
| `--report PATH` | Write the multi-page PDF quality report | The procurement-facing artifact: metrics, calibration, feature importance. |
| `--json` | Machine-readable result to stdout | Use this when an agent runs the command — parse the JSON instead of scraping tables. |

### Predict / explain / interval / what-if — `p2predict`

```bash
# Point estimate
p2predict -m model.model -p "manufacturer:Texas Instruments,package_pins:8,op_temp_max_C:85,interface:I2C"

# + per-feature attribution and a 90% likely range
p2predict -m model.model -p "<spec>" --explain --interval 90

# What-if: same spec, different supplier — returns the delta
p2predict -m model.model -p "<base spec>" --whatif "manufacturer:Microchip Technology"

# Batch: a CSV of parts, one prediction per row
p2predict -m model.model -i new_parts.csv

# Agent-friendly: structured output
p2predict -m model.model -p "<spec>" --explain --interval 90 --json
```

| Flag | Meaning |
|---|---|
| `-m, --model` | Path to the `.model` file. |
| `-p, --predict_using` | One part as `"key:value,key:value"`. |
| `-i, --predict_file` | CSV for batch prediction (no `--whatif` in batch mode). |
| `--explain` | SHAP per-feature attribution. |
| `--interval N` | Likely range at N% coverage (e.g. `90` = 9-in-10). |
| `--whatif "k:v,..."` | Counterfactual vs. the `-p` base; one or more changed features. |
| `--json` | Structured output (schema in `p2predict.json_output`). |

---

## Python API (for embedding)

Training programmatically is lower-level (it wants pre-split arrays), so for
training prefer the CLI. The prediction-side API is clean and is what the case
study chart scripts use:

```python
from p2predict import load_model, predict_interval, explain, what_if
import pandas as pd

m = load_model("model.model")          # dict: model, calibration,
                                        # background_sample, features,
                                        # target_feature, feature_types, ...
part = pd.DataFrame([{ "manufacturer": "Texas Instruments",
                       "package_pins": 8, "op_temp_max_C": 85,
                       "interface": "I2C" }])

price      = m["model"].predict(part)[0]
intervals  = predict_interval(m["model"], part, m["calibration"], coverage=0.90)
expl       = explain(m["model"], part, background_X=m["background_sample"])
whatif     = what_if(m["model"], part, {"manufacturer": "Microchip Technology"},
                     m["feature_types"], background_X=m["background_sample"],
                     calibration=m["calibration"])
```

`expl` is an `Explanation`: `.baseline`, `.prediction`, `.contributions`
(dollar dict for additive models); for log-target models also
`.baseline_price`, `.predicted_price`, `.multiplicative_factors`.

---

## Map the question to the tool

| The user asks… | Reach for |
|---|---|
| "What should this part cost?" / "Is this quote fair?" | `predict` (+ `--interval` for the defensible range) |
| "Why is it priced this way?" / "What's driving the cost?" | `--explain` |
| "How sure are you?" / "What's the range?" | `--interval 90` |
| "What if we switched supplier / changed a spec?" | `--whatif` |
| "Which supplier premium can we negotiate away?" | `--whatif` swapping `manufacturer`, read the delta |
| "Build me a should-cost model from this data" | `p2predict-train` (then `--report` for the writeup) |
| "How good is the model?" | `--report PATH` → the PDF (metrics + calibration + importance) |

---

## The interpretation rules (the actual expertise)

These are the lessons that separate a usable answer from a confident-wrong one.
Apply them every time; explain the relevant ones to the user so they trust the
result for the right reasons.

### 1. Set `--log-target on` for any price or cost model

Prices, costs, weights, lead times are **multiplicative** quantities — a 10%
move means the same thing at \$1 and at \$1000. The log-target wrap
(`TransformedTargetRegressor(log, exp)`) is what makes the model behave that
way, and it has two concrete payoffs:

- **SHAP attribution composes as multipliers** ("ADI × 1.18") instead of flat
  dollars — the natural language for "this supplier adds 18%."
- **Conformal intervals stay strictly positive.** An additive interval is
  `prediction ± a fixed dollar half-width`; on a cheap part that half-width can
  exceed the price and the lower bound goes **negative** — meaningless for a
  price.

The default is `auto`, which only flips the wrap on when the *sample* skew
exceeds 1.0. That under-fires: a perfectly multiplicative price target whose
particular sample happens to look symmetric (skew 0.12, say) is left additive,
and you get negative interval bounds on the cheap parts. **Don't trust auto to
catch this — pass `--log-target on` explicitly for price/cost targets.** Use
`off` only when the target is genuinely additive-scale. (Caveat: `on` requires
all training targets strictly positive; it aborts cleanly otherwise.)

### 2. On small data, use `--feature-outliers warn`, not `drop`

A real BOM-benchmarking dataset is small (tens to low-hundreds of parts).
`drop` removes any row with a numeric feature outside the Tukey fence — fine on
a large dataset where a few data-entry errors deserve removal and losing rows
costs nothing. On ~100 parts it is **destructive**: when a column is
near-constant (e.g. mostly −40/85 °C industrial parts), its fence collapses and
`drop` deletes a large fraction of rows; the no-variation pruner then removes
the column entirely, which can break prediction with an `unknown_features`
error. Keep every part: `warn` reports outliers without dropping them.

### 3. Read SHAP by its scale, and sign-check it

- **Additive model** → contributions are **dollars** and obey
  `baseline + Σ contributions = prediction` exactly.
- **Log-target model** → **multiplicative factors** with
  `baseline × ∏ factors = prediction`.

Either way there's an **axiom check** printed; if it ever fails, the
explanation is unsound — don't use it. Then sanity-check the **signs** against
engineering intuition. A counterintuitive sign (e.g. "more cells → cheaper")
on a **low-importance** feature is the model telling you that feature is
**under-sampled**, not that the world is upside-down. Lean on the
high-importance, correctly-signed drivers; flag the rest as noise.

### 4. Interval width is a per-part trust signal

The `--interval` band width *is* the model's confidence on **that specific
part**. A tight band = predict with confidence. A very wide band — or one whose
lower bound hits/passes \$0 on an additive model — means "I'm genuinely unsure
here; get a quote, don't benchmark." Always surface the interval, not just the
point estimate, when the user is going to act on the number.

On larger datasets the width is **calibrated per price band** (Mondrian
conformal), so the noisy cheap end of the catalog no longer inflates the range
on the expensive parts — the width genuinely tracks model quality in *that*
part's price range, and the "9 in 10" coverage holds within each band. Each
interval row in `--json` carries a `band` string (the price range its width was
calibrated on) or `null` when one global width was used (small calibration set,
or a pre-banding model file). When present, quote it: *"calibrated on
similar-priced parts ($5–$155)"* is a stronger defense than a bare range.

### 5. Judge the model by bias, not just R²

R² alone can mislead. The honesty check is the **residual-bias t-test** (on PDF
page 2 of `--report`): a high p-value (say > 0.05) means the model is
**statistically unbiased** — not systematically over- or under-pricing — which
is what makes its intervals and attributions usable even when R² is modest. A
model with higher R² but biased residuals is *less* trustworthy for procurement
than a modest-R², unbiased one.

### 6. Feature importance tells you which findings to quote

Before repeating any finding to a stakeholder, check its feature's importance
(PDF page 3). A finding resting on a high-importance, well-sampled feature
(e.g. supplier at 40%) is quotable. One resting on a 1–2% feature is a weak
signal — present it as a hypothesis, not a number to negotiate against.

---

## Presenting results to a category manager

When the user wants insights for a procurement/business audience, structure the
output like a consulting case — and **point to where each finding is read off
the tool**, so nothing is asserted on faith:

1. **The case** — one line: what question, what data, what model (and its
   headline quality: R², unbiased-or-not).
2. **The findings** — lead with results, each tied to its source: which chart,
   which `--report` PDF page, or which command produced it.
3. **The so-what** — the sourcing action for each finding, *then* an honest
   "where's the value vs. where's the rubbish" map (🟢 trust / 🟡 use with care
   / 🔴 don't trust yet), grounded in rules 3–6 above.

Keep it value-first and honest about limits — surfacing where the model is thin
(and *why*) builds more credibility than a uniformly confident deck. The
`case-studies/battery-management-ics/README.md` in this repo is the reference
example of this structure.

---

## Data hygiene & guardrails

- **Don't commit raw vendor data or credentials.** Catalog pulls (DigiKey,
  etc.) are often not redistributable in bulk; keep raw data gitignored and
  credentials outside the repo (`~/.digikey/credentials`, `chmod 600`).
- **Respect data-source terms of service** when fetching.
- **Never present a single high-value part's point estimate as a final
  appraisal.** Use the model to set the target and find the lever; get a real
  quote for the decision.
