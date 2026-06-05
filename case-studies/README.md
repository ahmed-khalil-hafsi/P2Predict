# Case studies

Three real-world stories that show P2Predict on data anyone can download and verify. Each one targets a different audience and demonstrates a different feature.

| Case study | Audience | Data source | Demonstrates |
|---|---|---|---|
| [Electronic components](electronic-components/) | EE procurement engineers, electronics buyers, hardware BOM owners | Octopart / Mouser / DigiKey API | The classic parametric procurement use case — `--explain` and `--whatif` for spec-driven cost trade-offs |
| [Used vehicles](used-cars/) | Anyone — the universally-relatable tutorial | Kaggle (CC license) | Log-target SHAP + multiplicative explanations on prices that span orders of magnitude |
| [Aerospace contracts](aerospace-contracts/) | Cost-estimating community (ICEAA, NASA, DoD primes) | USAspending.gov (public domain) | Parametric estimating on government / large-procurement data |
| [PCBA composition](pcba-composition/) | Hardware BOM owners, EMS-cost analysts, cost engineers doing WBS estimating | Three composed sources: Octopart + PCB fab quotes + assembly rates | **Composability** — three trained models (components + PCB + assembly) summed into BOM-level cost with per-stage attribution and a what-if mode |

## Why these, in this order

- **Electronic components** is the closest-to-target audience. Procurement engineers in EE will recognise the methodology immediately and pay attention.
- **Used vehicles** is the warm-up tutorial. Everyone gets the problem, the data is clean, and it's the strongest visual demo of P2Predict's log-target multiplicative SHAP attribution.
- **Aerospace contracts** is the credibility move. The cost-estimating community (ICEAA's ~3,000 members, plus NASA / DoD cost engineers) invented parametric estimating. Showing P2Predict on their public data is the strongest credibility signal you can send to that audience.
- **PCBA composition** is the *composability* story. It demonstrates that P2Predict scales naturally to BOM-level cost engineering — three independently-trained models, composed at the assembly level. This is how serious hardware-procurement cost engineering actually works.

The four different doors into the same product matter: an EE engineer finds #1, a curious developer finds #2, an ICEAA member finds #3, a hardware BOM owner finds #4. None overlap.

## The shape each case study follows

Each subfolder has the same four files:

```
case-studies/<name>/
  README.md               ← the procurement story, results, takeaways (5-minute read)
  fetch_data.py           ← pulls data from the source — credentials in env vars
  train.py                ← trains a P2Predict model (or just documents the CLI command)
  predict_examples.py     ← runs sample predictions with --explain and --interval
```

The pattern is deliberate: anyone reading the case study's `README.md` should be able to reproduce the result end-to-end in under 15 minutes if they have the right API credentials (for #1) or an internet connection (for #2 and #3).

The PCBA composition case study (#4) is a *variation* on the pattern — it composes three models rather than training one. Its top-level orchestration script is `compose.py`, and it reuses the component model trained in case study #1.

## Reproducibility — what's checked in and what isn't

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
