# Technical Reference

This is the developer reference for P2Predict — CLI flags, Python API, JSON schema, data format, and internals. For an overview of what P2Predict does and why, see the [README](README.md).

---

## Data format

One row per part. One column is the target (price); the rest are the features the model learns from.

```csv
CPN,Weight,Region,Supplier,Size,Price
CP17-17921595,17,EU,supplier A,Standard,1.41
CP2-5580430,2,CN,supplier A,Small,0.18
CP30-19674030,30,SG,supplier A,Large,2.15
```

- **Column types are auto-detected.** Numeric → numerical features; text/boolean → categorical.
- **Target column** is whatever you pass with `--target` (CLI) or tell the agent. Doesn't have to be "Price" — `Cost`, `Revenue`, anything numeric works.
- **Identifier columns** (part numbers, SKUs) are auto-detected as high-variation and flagged.
- **Missing values:** target NAs → row dropped. Feature NAs → XGBoost handles natively; RF/Ridge impute (median for numeric, most-frequent for categorical).
- **Outliers:** Tukey IQR on target (`--outliers`) and features (`--feature-outliers`). Policies: `warn` (default), `drop`, `winsorize`, `keep`. Feature `drop` is row-level; `winsorize` is per-column.
- **Time-ordered data:** `--time-column DATE` enables chronological train/test split and `TimeSeriesSplit` CV.

See [`examples/example.csv`](examples/example.csv) for a working dataset.

---

## Install

```bash
pip install p2predict          # core: CLI + Python API
pip install p2predict[mcp]     # adds the MCP server for AI agents
```

From a clone:

```bash
pip install -e ".[dev,mcp]"    # development with all extras
```

---

## MCP server

The primary interface. See the [README quick start](README.md#quick-start) for setup.

Entry point: `p2predict-mcp --models-dir /path/to/models`

All 10 tools are documented in the README. The server runs over stdio — no network, no data leaves disk. CPU-bound calls (predict, explain, train) run via `asyncio.to_thread()` to keep the event loop responsive.

Source: [`src/p2predict/mcp/server.py`](src/p2predict/mcp/server.py)

---

## CLI reference

### Train

```bash
p2predict-train --input examples/example.csv --target Price
```

Auto-mode cross-validates Ridge, Random Forest, and XGBoost, picks the best, and saves the model.

| Flag | Description |
|---|---|
| `--input`, `-i` | Path to CSV |
| `--target`, `-t` | Column to predict |
| `--expert`, `-x` | Expert mode — control algorithm and HPO |
| `--algorithm`, `-a` | Algorithm in expert mode: `ridge`, `xgboost`, `random_forest` |
| `--interactive`, `-c` | Guided interactive mode |
| `--training_features`, `-tf` | Comma-separated features to use |
| `--max-features` | Max features in auto mode (default 6) |
| `--budget`, `-b` | HPO budget: `fast` (default) or `thorough` |
| `--tune / --no-tune` | Expert mode: run hyperparameter tuning |
| `--outliers` | Target outlier policy: `warn`, `drop`, `winsorize`, `keep` |
| `--feature-outliers` | Feature outlier policy (same options) |
| `--time-column` | Date column → chronological split + TimeSeriesSplit CV |
| `--log-target` | `auto` (default), `on`, or `off` — override the skew-based log-target rule |
| `--report PATH` | Write the model-quality PDF report |
| `--json` | Structured JSON output |

Examples:

```bash
# Interactive guided mode
p2predict-train --interactive

# Thorough HPO search
p2predict-train --input data/parts.csv --target Price --budget thorough

# Expert mode: tuned XGBoost on specific features
p2predict-train --expert --input examples/example.csv \
  --algorithm xgboost --target Price \
  --training_features Weight,Size,Region --tune --budget fast

# With PDF report and JSON output
p2predict-train --input data/parts.csv --target Price \
  --report model_report.pdf --json
```

### Predict

```bash
p2predict -m MODEL_PATH [-p PREDICT_USING] [-i PREDICT_FILE]
```

| Flag | Description |
|---|---|
| `-m, --model` | Path to `.model` file |
| `-p, --predict_using` | Inline features: `"Weight:15,Region:EU"` |
| `-i, --predict_file` | CSV of parts to batch-predict |
| `--explain` | Per-feature SHAP attribution |
| `--interval N` | Likely range at N% coverage |
| `--whatif "F:V,..."` | Counterfactual comparison |
| `--json` | Structured JSON output |

Examples:

```bash
# Point prediction
p2predict -m models/my_model.model -p "Weight:25,Region:EU,Supplier:A,Size:Standard"

# With likely range and SHAP explanation
p2predict -m models/my_model.model \
  -p "Weight:25,Region:EU,Supplier:A,Size:Standard" \
  --interval 90 --explain

# What-if: switching supplier
p2predict -m models/my_model.model \
  -p "Weight:25,Region:EU,Supplier:A,Size:Standard" \
  --whatif "Supplier:B" --interval 90

# Batch from CSV
p2predict -m models/my_model.model -i rfq_lines.csv --interval 90
```

---

## Python API

```python
import pandas as pd
from p2predict import (
    auto_train,           # CV-based model selection (Ridge/RF/XGBoost)
    explain,              # exact SHAP attribution per prediction
    predict_interval,     # conformal "likely range" with guaranteed coverage
    what_if,              # base vs counterfactual comparison
    load_model, save_model,
)
from p2predict.prepare_data import prepare_data
from p2predict.intervals import compute_calibration_residuals

data = pd.read_csv("purchases.csv")
features = ["Weight", "Region", "Supplier", "Size"]

# Train (cross-validated model selection across three algorithms).
X_train, X_test, y_train, y_test, num, cat = prepare_data(data, features, "Price")
model, algorithm, scores, log_target = auto_train(X_train, y_train, num, cat)

# Inference + per-feature attribution.
new_part = pd.DataFrame([{"Weight": 15, "Region": "EU",
                          "Supplier": "A", "Size": "Standard"}])
calibration = compute_calibration_residuals(model, X_test, y_test)

[interval] = predict_interval(model, new_part, calibration, coverage=0.90)
explanation = explain(model, new_part, background_X=X_train.sample(100))

print(f"Predicted: {interval.prediction:.2f}")
print(f"Likely range (90%): {interval.low:.2f}–{interval.high:.2f}")
for feature, contribution in explanation.contributions.items():
    print(f"  {feature}: {contribution:+.2f}")
```

See [`examples/python_api.py`](examples/python_api.py) for an end-to-end walkthrough.

The public API surface is `from p2predict import ...`. Submodule paths (`p2predict.training`, `p2predict.intervals`) are stable but lower-level.

---

## JSON output

Both CLIs accept `--json` — suppresses all Rich output and emits a single JSON document on stdout. Schema documented in [`src/p2predict/json_output.py`](src/p2predict/json_output.py), versioned via `schema_version`.

```bash
p2predict -m my_model.model \
  -p "Weight:15,Region:EU,Supplier:A,Size:Standard" \
  --interval 90 --explain --whatif "Region:CN" \
  --json | jq '.'
```

```json
{
  "schema_version": "1.0",
  "command": "predict",
  "model": {
    "path": "my_model.model",
    "algorithm": "random_forest",
    "target": "Price",
    "version": "v0.9",
    "log_target": false,
    "features": ["Weight", "Region", "Supplier", "Size"]
  },
  "mode": "inline",
  "predictions": [
    {"input": {"Weight": "15", "Region": "EU", "Supplier": "A", "Size": "Standard"},
     "prediction": 1.318}
  ],
  "interval": {
    "coverage": 0.90,
    "per_row": [{"low": 1.04, "prediction": 1.318, "high": 1.59,
                 "band": "predicted 0.80 to 2.10"}],
    "soft_warning": null
  },
  "explanation": [
    {
      "baseline": 1.35,
      "prediction": 1.318,
      "log_target": false,
      "contributions": [
        {"feature": "Weight", "value": -0.059},
        {"feature": "Region", "value": 0.030}
      ],
      "multiplicative_factors": null,
      "dollar_attribution": null,
      "residual": 0.0
    }
  ],
  "whatif": {
    "changes": {"Region": {"from": "EU", "to": "CN"}},
    "base_prediction": 1.318,
    "counterfactual_prediction": 1.250,
    "delta": -0.068,
    "delta_pct": -5.2,
    "changed_contributions": [{"feature": "Region", "value": -0.068}],
    "interaction_contribution": -0.012,
    "interaction_is_material": false,
    "base_interval": {"low": 1.04, "high": 1.59},
    "cf_interval": {"low": 0.97, "high": 1.52}
  }
}
```

Key points:

- **Stable, versioned schema.** Tests in [`tests/test_json_output.py`](tests/test_json_output.py) lock in top-level keys. New fields don't bump the version; renames/removals do.
- **stdout is exclusively JSON.** No banners, spinners, or Rich formatting.
- **Errors are JSON too.** `{"error": {"code": "...", "message": "..."}}` on stdout, exit code 1.
- **`--json` composes** with `--interval`, `--explain`, `--whatif`. Each adds its block.
- **Each interval row carries a `band`** — the price range it was calibrated on, or `null` for global width.
- **Interactive mode is incompatible** with `--json` — errors cleanly instead of prompting.

Train CLI JSON shape (`cv_scores`, `evaluation`, `feature_importances`, `model_path`, etc.) documented at the top of [`src/p2predict/json_output.py`](src/p2predict/json_output.py).

---

## Features (detailed)

### Benchmarking / prediction
- Point predictions for single parts or entire CSVs
- **Likely-range intervals** — conformal, calibrated on holdout. On larger datasets, **banded by predicted price** (Mondrian conformal): widths track where the model is good, and the 90% guarantee holds within each band, not just on average
- **Per-prediction explanations** — exact SHAP (TreeExplainer / LinearExplainer). Additive decomposition `baseline + Σ contributions = prediction`, or multiplicative factors for log-target models
- **What-if analysis** — base vs counterfactual with dollar/percent delta and per-feature SHAP decomposition. Composes with intervals
- Robust to unseen categorical values at prediction time

### Model training
- **Cross-validated model selection** across Ridge, Random Forest, XGBoost via `HalvingRandomSearchCV`
- **Automatic log-target transform** for skewed positive targets (overridable with `--log-target`)
- **Outlier handling** on target and features (Tukey IQR), four policies each
- **Time-aware CV** via `--time-column`
- Auto-detection of predictive features (RF baseline) and low-information features
- `TargetEncoder` for tree categoricals; `OneHotEncoder + StandardScaler` for linear
- Expert mode with algorithm selection and HPO control
- Model-quality PDF report (`--report`)
- Evaluation: R², MAE, RMSE, residual-bias check

---

## Dependencies

Core: click, joblib, numpy, pandas, scipy, scikit-learn (≥ 1.5), xgboost, shap, rich, halo, questionary, matplotlib, seaborn.

MCP extra: [Anthropic MCP SDK](https://modelcontextprotocol.io) (`mcp >= 1.0`).

## Running the tests

```bash
pip install -e ".[dev,mcp]"
pytest tests/
```

CI runs on every push to `main` and on PRs. MCP tests skip gracefully without the `mcp` extra.
