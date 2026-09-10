# Case studies

Four real-world stories that show P2Predict on data anyone can download and verify. Each one targets a different audience and demonstrates a different feature.

| Case study | Audience | Data source | Demonstrates |
|---|---|---|---|
| [Battery Management ICs](battery-management-ics/) | EE procurement engineers, electronics category managers, hardware BOM owners | Octopart / Mouser / DigiKey API | The classic parametric procurement use case, `--explain` and `--whatif` for spec-driven cost trade-offs |
| [Used vehicles](used-cars/) | Anyone, the universally-relatable tutorial | Kaggle (CC license) | Log-target SHAP + multiplicative explanations on prices that span orders of magnitude |
| [Aerospace fasteners](aerospace-fasteners/) | Fastener / MRO category managers, federal procurement, cost engineers | DLA PUB LOG / FLIS catalog (public domain) | **Detecting noisy data**, measuring a model's R² ceiling before tuning, plus log-target multiplicative SHAP on a heavily-skewed price |
| [Heavy-equipment resale (sell-side)](heavy-equipment-sales/) | Equipment remarketing desks, dealer used-equipment teams, fleet asset-disposal managers | Kaggle *Blue Book for Bulldozers* (open re-upload) | **The sell-side flip**: same engine, seller's question. XGBoost on 80k rows (trees + target-encoding win at scale), a resale what-if lever, and the honest "comparator, not appraiser" verdict on a biased-but-accurate model |

## Why these, in this order

- **Battery Management ICs** is the closest-to-target audience. Procurement engineers in EE will recognise the methodology immediately and pay attention.
- **Used vehicles** is the warm-up tutorial. Everyone gets the problem, the data is clean, and it's the strongest visual demo of P2Predict's log-target multiplicative SHAP attribution.
- **Aerospace fasteners** is the *honesty* story. It's the one case study where the model is deliberately weak, and the value is the diagnostic that proves it's the data, not the model. It teaches how to measure a noise floor and know when to stop tuning, the skill that matters most on messy real-world data. The dataset (DLA PUB LOG, public domain) is also a credibility signal to the federal-procurement and cost-estimating audience.
- **Heavy-equipment resale** is the *sell-side* story. Same parametric engine, opposite chair: instead of "what should we pay", it answers "what will this used machine fetch, and which attributes carry the resale value". It proves P2Predict isn't only a buyer's tool, shows trees + target-encoding winning at 80k rows (the mirror image of the tiny-data BMIC study), and demonstrates the honest "comparator, not appraiser" verdict on a model that's accurate but biased.

The four different doors into the same product matter: an EE buyer finds #1, a curious developer finds #2, a fastener category manager finds #3, and an equipment *seller* finds #4. None overlap.

## The shape each case study follows

Each subfolder has the same four files:

```
case-studies/<name>/
  README.md               ← the procurement story, results, takeaways (5-minute read)
  fetch_data.py           ← pulls data from the source, credentials in env vars
  train.py                ← trains a P2Predict model (or just documents the CLI command)
  predict_examples.py     ← runs sample predictions with --explain and --interval
```

The pattern is deliberate: anyone reading the case study's `README.md` should be able to reproduce the result end-to-end in under 15 minutes if they have the right API credentials (for #1) or just an internet connection (for the public-data studies).

## Reproducibility, what's checked in and what isn't

| Type | Checked in? | Why |
|---|---|---|
| Scripts (`fetch_data.py`, `train.py`, `predict_examples.py`) | ✅ Yes | They're code |
| Case study narrative (`README.md`) | ✅ Yes | It's documentation |
| Trained model file (`.model`) | ❌ No | Re-trainable from the scripts |
| Raw downloaded data | ❌ No | Repo bloat + licensing |
| A tiny anonymised sample for smoke-testing | ✅ Maybe | Useful for CI |

Each case study's `fetch_data.py` writes its raw data to a `data/` subdirectory inside that case study, which is git-ignored.

## Adding a new case study

Copy any case study folder, rename it, replace the source-specific code, and add a row to the table at the top of this README. New case studies should target an audience the existing three don't reach.
