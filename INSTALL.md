# Install & set up P2Predict

P2Predict runs locally and connects to your AI agent over the Model Context Protocol (MCP). Below is the fast path to a working setup; for the full CLI, Python API, and data-format reference see **[TECHNICAL.md](TECHNICAL.md)**.

## 1. Install

```bash
pip install p2predict[mcp]
```

This installs the engine, the command-line tools (`p2predict`, `p2predict-train`), and the MCP server (`p2predict-mcp`).

## 2. Connect your AI agent

Add P2Predict to your MCP client (Claude Desktop, Claude Code, Cursor, or any MCP-capable assistant):

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

Point `--models-dir` at the folder where your trained models live (or will live). The agent discovers models and tools from there automatically.

## 3. Talk to it

Once connected, just ask:

> *"Train a model on my purchasing data at ~/data/parts.csv, predicting Price."*
>
> *"What would a 25 kg EU part from Supplier A cost?"*
>
> *"What happens to the price if we switch to Supplier B?"*
>
> *"Benchmark these 200 RFQ line items against the model."*

## What the agent can do (MCP tools)

The agent has the full surface available and picks the right tool for the question:

| Tool | What it does |
|---|---|
| `list_models` | Discover trained models |
| `get_model_info` | Features, types, categories, calibration status |
| `get_model_quality` | Trust verdict: is the model reliable, where, and by how much |
| `predict` | Point estimate for a single part |
| `predict_batch` | Predict multiple parts in one call |
| `predict_from_csv` | Batch-predict a CSV file |
| `predict_interval` | The likely range (e.g. 9-in-10 coverage) |
| `explain` | Per-feature attribution — what drives this price |
| `what_if` | What changes if I switch supplier / material / spec? |
| `propose_training_plan` | Review the data and propose how to train, before training |
| `train` | Train a new model from a CSV |
| `generate_report` | Model-quality PDF report |

## Other surfaces

The same engine is available three ways, all calling the same math:

- **AI agent (MCP)** — the primary interface, described above.
- **Command line** — `p2predict` and `p2predict-train` for scripted or interactive use.
- **Python API** — `from p2predict import auto_train, explain, predict_interval, what_if` for embedding in a notebook or pipeline.

Full reference for the CLI, Python API, JSON output schema, and CSV data format: **[TECHNICAL.md](TECHNICAL.md)**.

## Keeping your data safe

- P2Predict reads your CSV, trains, and predicts entirely on your machine over local (stdio) transport. Nothing is uploaded.
- Don't commit raw vendor data or credentials to a shared repo — keep raw pulls and API keys outside version control.
- Respect the terms of service of any catalog or data source you pull from.

## Contributing

Bug reports, feature requests, and dataset suggestions — [open an issue](https://github.com/ahmed-khalil-hafsi/P2Predict/issues).

Open procurement datasets are especially welcome: ICs, passive components, plastic parts, mechanical parts. If you know of one, or your organization would share, [reach out](https://ahmedhafsi.com/contact/).

Code contributions require a CLA (P2Predict is dual-licensed). [Reach out](https://ahmedhafsi.com/contact/) before investing time in a large patch.
