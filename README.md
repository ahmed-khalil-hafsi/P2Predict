# P2Predict

[![CI](https://github.com/ahmed-khalil-hafsi/P2Predict/actions/workflows/p2predict_train.yml/badge.svg)](https://github.com/ahmed-khalil-hafsi/P2Predict/actions/workflows/p2predict_train.yml)

**Parametric price benchmarking for procurement. Talk to your AI agent, get data-grounded answers. Runs on your data, stays on your machine.**

![P2Predict MCP Demo](./documentation/p2predict_mcp_demo.gif)

- **Ask what a part should cost** — point prediction grounded in your historical purchases
- **Know how confident the model is** — calibrated "9-in-10" likely ranges (conformal intervals)
- **See exactly why** — per-feature SHAP attribution: which spec drives the price, by how much
- **Test design trade-offs in seconds** — *"what if we switch supplier?"* with dollar deltas and confidence ranges
- **Batch-benchmark an entire RFQ** — drop a CSV, flag the lines worth a phone call
- **Your data never leaves your machine** — local models, local inference, stdio transport

---

## Quick start

```bash
pip install p2predict[mcp]
```

Configure your agent (Claude Desktop, Cursor, or any MCP client):

```json
{
  "mcpServers": {
    "p2predict": {
      "command": "p2predict-mcp",
      "args": ["--models-dir", "/path/to/your/models"]
    }
  }
}
```

Then just talk:

> *"Train a model on my purchasing data at ~/data/parts.csv, predicting Price"*
>
> *"What would a 25kg EU part from Supplier A cost?"*
>
> *"What happens to the price if we switch to Supplier B?"*
>
> *"Benchmark these 200 RFQ line items against the model"*

The agent discovers models and tools automatically. 10 MCP tools cover the full surface:

| Tool | What it does |
|---|---|
| `list_models` | Discover trained models |
| `get_model_info` | Features, types, categories, calibration status |
| `predict` | Point prediction for a single part |
| `predict_batch` | Predict multiple parts in one call |
| `explain` | SHAP attribution — which features drive this price |
| `predict_interval` | Conformal "likely range" (e.g. 9-in-10 coverage) |
| `what_if` | What changes if I switch supplier / material / spec? |
| `predict_from_csv` | Batch-predict a CSV file |
| `train` | Train a new model from a CSV |
| `generate_report` | Model-quality PDF report |

Also available as a **CLI** (`p2predict`, `p2predict-train`) and a **Python API** (`from p2predict import auto_train, explain, predict_interval, what_if`). All three surfaces call the same math. See the [technical reference](TECHNICAL.md) for full docs.

---

## What it is

P2Predict is a **parametric cost-prediction tool** — the kind of model NASA, ICEAA, and cost-estimating bodies call *parametric estimating*. It learns `features → price` from your historical purchasing data, then answers questions about new or proposed parts.

It is **not bottom-up should-costing** (aPriori, Siemens Teamcenter PCM). Those tools decompose a part into material + labor + machine time. P2Predict answers the complementary question: *"what has the market actually charged us for parts like this?"*

### How it works

1. **Bring your history.** A CSV — one row per part, with specs (weight, material, region, supplier, …) and the price you paid.
2. **Train a model.** P2Predict fits Ridge, Random Forest, and XGBoost, cross-validates them, and keeps the best one.
3. **Ask questions.** Point estimates, likely ranges, per-feature explanations, what-if comparisons — through your agent or the CLI.

The model learns from *your* data — so the benchmark reflects your supply base, not a vendor's catalog.

---

## Built for these conversations

### Design reviews: "is this feature worth it?"

Engineer proposes tighter tolerances — ±0.05mm instead of ±0.1mm. Ask the agent: *"what happens to cost?"*

Answer: *"+$0.42/unit (+18%), 9-in-10 range $0.30–$0.55."* Now the conversation is *"is +18% worth it for this requirement?"* — not a debate based on who speaks loudest.

### RFQ triage: focus on what matters

200 line items, one afternoon. Drop the CSV on the agent: *"benchmark these."* Every line gets a prediction and a likely range. The 8–15 lines outside the range are the ones worth a call. The rest are routine.

### Negotiation prep: argue the components, not the total

Supplier quotes $14.20, model says $12.40 (90% range $10.80–$13.90). Ask for the breakdown: supplier choice +$0.85, rush delivery +$1.20, size +$0.40. *"Why is rush delivery on this line? We agreed to standard lead time."*

### Supplier and material trade-offs

*"What if we switch from ADI/Maxim to Microchip on this 16-cell BMS?"* → **−$2.07/unit (−37.7%)**, with per-feature SHAP showing what's driving the delta. Same answer in 30 seconds that would take a week of RFQs.

### Audit defense

Finance asks about PO #4521 six months later. The model's explanation is on file: predicted $12.40 ± $1.50, drivers were tolerance and supplier, consistent with the engineering spec. That's an auditable paper trail.

### Buyer onboarding

Senior buyer retires — 20 years of benchmarking intuition walks out. A P2Predict model trained on historical buys captures what features drive cost, what a normal range looks like, and what looks off. The new buyer inherits a baseline, not just a spreadsheet.

---

## Case studies

Three reproducible builds on public datasets:

- **[Battery Management ICs](case-studies/battery-management-ics/)** — 150 parts from DigiKey. The procurement-shaped case study: thin data, additive target, supplier-swap what-if.
- **[Used vehicles](case-studies/used-cars/)** — 426k rows, prices spanning orders of magnitude. The tutorial.
- **[Aerospace fasteners](case-studies/aerospace-fasteners/)** — public-domain DLA catalog. Measuring a model's R² ceiling on noisy data.

Each includes a `fetch_data.py`, training command, and worked predictions with explanations, intervals, and what-if.

---

## Under the hood

| | |
|---|---|
| **Algorithms** | Ridge, Random Forest, XGBoost — auto-selected via cross-validated `HalvingRandomSearchCV` |
| **Explanations** | Exact SHAP — `TreeExplainer` for tree models, `LinearExplainer` for linear |
| **Intervals** | Split-conformal prediction intervals, banded by price range (Mondrian) on larger datasets |
| **Categoricals** | `TargetEncoder` for trees (orders by price, not alphabet), `OneHotEncoder` for linear |
| **Outliers** | Tukey IQR on target and features, four policies: `warn`, `drop`, `winsorize`, `keep` |
| **Log-target** | Auto-detected (skew > 1.0) or manual override — multiplicative SHAP, always-positive intervals |
| **Time-aware** | Chronological split + `TimeSeriesSplit` CV via `--time-column` |

Full CLI reference, Python API, JSON schema, and data format docs: **[TECHNICAL.md](TECHNICAL.md)**

![Model performance plot](./documentation/model_perf_plot.png)

---

## Contributing

Bug reports, feature requests, and dataset suggestions — [open an issue](https://github.com/ahmed-khalil-hafsi/P2Predict/issues).

I'm particularly keen on **open procurement datasets**: ICs, passive components, plastic parts, mechanical parts. If you know of one or your organization would share, please reach out.

**Code contributions** require a CLA (P2Predict is dual-licensed). [Reach out](https://ahmedhafsi.com/contact/) before investing time in a large patch.

## Licensing

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE).

- **Free for internal use** — use P2Predict within your own organization at no cost.
- **Commercial use requires a license** — deploying for clients, embedding in a paid service, or consulting engagements.

**Ahmed K. Hafsi** — [ahmedhafsi.com/contact](https://ahmedhafsi.com/contact/)
