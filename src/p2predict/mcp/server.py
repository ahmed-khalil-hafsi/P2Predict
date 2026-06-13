"""P2Predict MCP server — typed tools for AI agents.

Start with:  p2predict-mcp --models-dir /path/to/models
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from p2predict.mcp.registry import ModelRegistry

mcp = FastMCP(
    "P2Predict",
    instructions=(
        "P2Predict is a parametric price/cost benchmarking toolkit for "
        "procurement. Use list_models to discover trained models, then "
        "predict, explain, predict_interval, or what_if to analyse parts. "
        "Use train to build a new model from a CSV of specs + prices."
    ),
)

_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialized — server not started correctly.")
    return _registry


def _error(code: str, message: str) -> str:
    return json.dumps({"error": {"code": code, "message": message}})


def _ok(data: dict) -> str:
    return json.dumps(data, default=_json_default)


def _json_default(obj: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_models() -> str:
    """List all trained P2Predict models in the configured models directory.

    Call this first to discover which models are available. Returns each
    model's ID, algorithm, target, features, and quality metrics.
    """
    registry = _get_registry()
    infos = await asyncio.to_thread(registry.scan)
    return _ok({
        "models_dir": str(registry.models_dir),
        "models": [info.to_dict() for info in infos],
    })


@mcp.tool()
async def get_model_info(model_id: str) -> str:
    """Get detailed information about a specific model.

    Returns the model's features, their types (Numerical/Categorical),
    allowed categories for each categorical feature, calibration status,
    and training metadata. Use this to understand what inputs a model
    expects before calling predict or explain.
    """
    registry = _get_registry()
    try:
        info = await asyncio.to_thread(registry.get_info, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))
    return _ok(info.to_dict())


@mcp.tool()
async def predict(model_id: str, features: dict) -> str:
    """Predict the target value (e.g. price) for a single part.

    Pass the model_id from list_models and a dictionary of feature values
    matching the model's expected features. Example:
    {"Weight": 15, "Region": "EU", "Supplier": "A"}
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    from p2predict.mcp.conversions import features_to_dataframe
    from p2predict.model_utils import extract_feature_info, inner_pipeline

    pipeline = inner_pipeline(loaded["model"])
    feature_types, _ = extract_feature_info(pipeline)
    try:
        df = features_to_dataframe(features, loaded["features"], feature_types)
    except ValueError as e:
        return _error("missing_feature", str(e))

    preds = await asyncio.to_thread(loaded["model"].predict, df)
    return _ok({
        "model_id": model_id,
        "target": loaded.get("target_feature"),
        "prediction": float(preds[0]),
        "input": features,
    })


@mcp.tool()
async def predict_batch(model_id: str, rows: list[dict]) -> str:
    """Predict the target value for multiple parts at once.

    More efficient than calling predict repeatedly. Pass a list of
    feature dictionaries, one per part. Returns one prediction per row.
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    from p2predict.mcp.conversions import rows_to_dataframe
    from p2predict.model_utils import extract_feature_info, inner_pipeline

    pipeline = inner_pipeline(loaded["model"])
    feature_types, _ = extract_feature_info(pipeline)
    try:
        df = rows_to_dataframe(rows, loaded["features"], feature_types)
    except ValueError as e:
        return _error("missing_feature", str(e))

    preds = await asyncio.to_thread(loaded["model"].predict, df)
    return _ok({
        "model_id": model_id,
        "target": loaded.get("target_feature"),
        "predictions": [
            {"input": row, "prediction": float(p)}
            for row, p in zip(rows, preds)
        ],
    })


@mcp.tool()
async def explain(model_id: str, features: dict, top_n: int = 3) -> str:
    """Explain why a prediction has its value using SHAP attribution.

    Shows each feature's contribution to moving the prediction from the
    baseline. For log-target models, also shows multiplicative factors.
    top_n controls how many top drivers are highlighted (default 3).
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    from p2predict import explain as explain_fn, top_drivers
    from p2predict.mcp.conversions import features_to_dataframe
    from p2predict.model_utils import (
        explanation_to_dict,
        extract_feature_info,
        inner_pipeline,
    )

    pipeline = inner_pipeline(loaded["model"])
    feature_types, _ = extract_feature_info(pipeline)
    try:
        df = features_to_dataframe(features, loaded["features"], feature_types)
    except ValueError as e:
        return _error("missing_feature", str(e))

    background = loaded.get("background_sample")
    try:
        expl = await asyncio.to_thread(
            explain_fn, loaded["model"], df, background_X=background
        )
    except ValueError as e:
        return _error("explain_error", str(e))

    drivers = top_drivers(expl, n=top_n)
    return _ok({
        "model_id": model_id,
        "target": loaded.get("target_feature"),
        "prediction": float(expl.prediction),
        "explanation": explanation_to_dict(expl),
        "top_drivers": [{"feature": f, "value": float(v)} for f, v in drivers],
    })


@mcp.tool()
async def predict_interval(
    model_id: str, features: dict, coverage: int = 90
) -> str:
    """Predict with a likely range (conformal prediction interval).

    For a 90% interval, about 9 in 10 similar parts fall within the range.
    coverage is an integer 1-99 (default 90). Requires a model trained with
    P2Predict v0.5+ (which stores calibration data).
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    calibration = loaded.get("calibration")
    if not calibration or not calibration.get("residuals"):
        return _error(
            "no_calibration",
            "This model has no calibration data. Retrain with P2Predict v0.5+ "
            "to enable prediction intervals.",
        )

    from p2predict import predict_interval as pi_fn
    from p2predict.mcp.conversions import features_to_dataframe
    from p2predict.model_utils import (
        extract_feature_info,
        inner_pipeline,
        interval_to_dicts,
    )

    pipeline = inner_pipeline(loaded["model"])
    feature_types, _ = extract_feature_info(pipeline)
    try:
        df = features_to_dataframe(features, loaded["features"], feature_types)
    except ValueError as e:
        return _error("missing_feature", str(e))

    if not (1 <= coverage <= 99):
        return _error("bad_coverage", "coverage must be between 1 and 99")

    try:
        intervals = await asyncio.to_thread(
            pi_fn, loaded["model"], df, calibration, coverage=coverage / 100.0
        )
    except ValueError as e:
        return _error("interval_error", str(e))

    ir = intervals[0]
    return _ok({
        "model_id": model_id,
        "target": loaded.get("target_feature"),
        "prediction": float(ir.prediction),
        "interval": interval_to_dicts(intervals)[0],
        "coverage_pct": coverage,
    })


@mcp.tool()
async def what_if(
    model_id: str,
    features: dict,
    changes: dict,
    coverage: int | None = 90,
) -> str:
    """Compare a base scenario with a counterfactual where features change.

    Shows both predictions, the delta (dollars and percent), and SHAP
    attribution of each change. Answers questions like "what if we switch
    from supplier A to B?" Set coverage to null to skip intervals.
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    from p2predict import what_if as whatif_fn
    from p2predict.mcp.conversions import features_to_dataframe
    from p2predict.model_utils import (
        extract_feature_info,
        inner_pipeline,
        whatif_to_dict,
    )

    pipeline = inner_pipeline(loaded["model"])
    feature_types, _ = extract_feature_info(pipeline)
    try:
        df = features_to_dataframe(features, loaded["features"], feature_types)
    except ValueError as e:
        return _error("missing_feature", str(e))

    for key in changes:
        if key not in feature_types:
            return _error(
                "bad_whatif",
                f"Cannot change '{key}': not a training feature. "
                f"Valid features: {list(feature_types.keys())}",
            )

    calibration = loaded.get("calibration") if coverage else None
    background = loaded.get("background_sample")
    cov = (coverage / 100.0) if coverage else 0.90

    try:
        result = await asyncio.to_thread(
            whatif_fn,
            loaded["model"],
            df,
            changes,
            feature_types,
            background_X=background,
            calibration=calibration,
            coverage=cov,
        )
    except ValueError as e:
        return _error("whatif_error", str(e))

    return _ok({
        "model_id": model_id,
        "target": loaded.get("target_feature"),
        "whatif": whatif_to_dict(result),
    })


@mcp.tool()
async def predict_from_csv(
    model_id: str,
    csv_path: str,
    with_explanation: bool = False,
    coverage: int | None = 90,
) -> str:
    """Batch-predict from a CSV file on the local filesystem.

    Equivalent to: p2predict -m model.model -i parts.csv
    Optionally includes SHAP explanations and/or likely-range intervals
    for every row. Use this when the user drops a spreadsheet of parts.
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    import pandas as pd

    from p2predict.model_utils import (
        coerce_features,
        explanation_to_dict,
        extract_feature_info,
        inner_pipeline,
        interval_to_dicts,
    )

    path = Path(csv_path)
    if not path.exists():
        return _error("file_not_found", f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return _error("csv_read_error", str(e))

    model_features = loaded["features"]
    missing = [f for f in model_features if f not in df.columns]
    if missing:
        return _error(
            "missing_feature",
            f"CSV is missing columns: {missing}. Expected: {model_features}",
        )

    pipeline = inner_pipeline(loaded["model"])
    feature_types, _ = extract_feature_info(pipeline)
    X = coerce_features(df[model_features].copy(), feature_types)

    preds = await asyncio.to_thread(loaded["model"].predict, X)

    rows_out: list[dict] = []
    for i in range(len(X)):
        row_data: dict[str, Any] = {
            "input": {f: df[f].iloc[i] for f in model_features},
            "prediction": float(preds[i]),
        }
        rows_out.append(row_data)

    result: dict[str, Any] = {
        "model_id": model_id,
        "target": loaded.get("target_feature"),
        "csv_path": csv_path,
        "n_rows": len(X),
        "predictions": rows_out,
    }

    if coverage and loaded.get("calibration"):
        from p2predict import predict_interval as pi_fn

        calibration = loaded["calibration"]
        if 1 <= coverage <= 99:
            intervals = await asyncio.to_thread(
                pi_fn, loaded["model"], X, calibration, coverage=coverage / 100.0
            )
            interval_dicts = interval_to_dicts(intervals)
            for i, iv in enumerate(interval_dicts):
                rows_out[i]["interval"] = iv
            result["coverage_pct"] = coverage

    if with_explanation:
        from p2predict import explain_batch

        background = loaded.get("background_sample")
        try:
            explanations = await asyncio.to_thread(
                explain_batch, loaded["model"], X, background_X=background
            )
            for i, expl in enumerate(explanations):
                rows_out[i]["explanation"] = explanation_to_dict(expl)
        except Exception:
            pass

    return _ok(result)


@mcp.tool()
async def train(
    csv_path: str,
    target: str,
    features: list[str] | None = None,
    algorithm: str = "auto",
    budget: str = "fast",
    log_target: str = "auto",
    outlier_policy: str = "warn",
    feature_outlier_policy: str = "warn",
    max_features: int = 6,
) -> str:
    """Train a new P2Predict model from a local CSV file.

    The CSV must have spec columns and a price/cost target column.
    Training runs locally — no data leaves the machine. The trained model
    is saved to the models directory and immediately available for
    prediction.

    algorithm: "auto" (default), "ridge", "random_forest", or "xgboost".
    budget: "fast" (default) or "thorough".
    log_target: "auto" (default), "on", or "off". Use "on" for prices.
    """
    registry = _get_registry()

    def _do_train() -> dict:
        import pandas as pd

        from p2predict import auto_train, Serialize_Trained_Model, save_model
        from p2predict.feature_selection import (
            find_no_variation_features,
            get_most_predictable_features,
        )
        from p2predict.intervals import compute_calibration_residuals
        from p2predict.model_evals import evaluate_model
        from p2predict.outliers import (
            apply_feature_outlier_policy,
            apply_outlier_policy,
        )
        from p2predict.prepare_data import prepare_data
        from p2predict.trained_model_io import load_csv_file
        from p2predict.training import (
            extract_feature_importances,
            resolve_log_target,
            start_training,
        )

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        data = load_csv_file(csv_path)
        rows_loaded = len(data)

        if target not in data.columns:
            raise ValueError(
                f"Target '{target}' not in CSV columns: {list(data.columns)}"
            )

        data = data[data[target].notna()]
        if data.empty:
            raise ValueError(f"All rows have missing values in target '{target}'.")

        data, _ = apply_outlier_policy(data, target, policy=outlier_policy)

        num_candidates = [
            c for c in data.columns if c != target and pd.api.types.is_numeric_dtype(data[c])
        ]
        data, _ = apply_feature_outlier_policy(
            data, num_candidates, policy=feature_outlier_policy
        )

        low_vars = find_no_variation_features(data)
        if low_vars:
            data = data.drop(low_vars, axis=1)

        if features:
            missing = [f for f in features if f not in data.columns]
            if missing:
                raise ValueError(f"Requested features not in CSV: {missing}")
            selected = list(features)
        else:
            ranked = get_most_predictable_features(data, target, output_only_headers=True)
            n_ranked = len(ranked)
            cap = max(2, min(n_ranked, max_features))
            selected = ranked.head(cap).tolist()

        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(
            data, selected, target
        )

        log_target_override, log_target_decision = resolve_log_target(
            y_train, mode=log_target
        )

        scores: dict = {}
        if algorithm == "auto":
            model, algo, scores, log_t = auto_train(
                X_train, y_train, num_cols, cat_cols,
                budget=budget, log_target=log_target_override,
            )
        else:
            model, _, log_t = start_training(
                X_train, y_train, num_cols, cat_cols, algorithm,
                budget=budget, tune=(budget == "thorough"),
                log_target=log_target_override,
            )
            algo = algorithm

        mae, r2, p_value, rmse = evaluate_model(X_test, y_test, model)

        background_n = min(100, len(X_train))
        background_sample = (
            X_train.sample(n=background_n, random_state=0).reset_index(drop=True)
            if background_n > 0
            else None
        )
        calibration = compute_calibration_residuals(model, X_test, y_test)

        y_pred_test = model.predict(X_test)

        model_metadata = Serialize_Trained_Model(
            algo, selected, target, model, r2,
            log_target=log_t,
            background_sample=background_sample,
            calibration=calibration,
        )
        model_metadata["holdout_y_test"] = y_test.tolist()
        model_metadata["holdout_y_pred"] = y_pred_test.tolist()

        try:
            importances = extract_feature_importances(model, X_train)
            importances_block = [
                {"feature": k, "importance": float(v)} for k, v in importances
            ]
        except Exception:
            importances = None
            importances_block = []

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{algo}_{target}_{timestamp}.model"
        model_path = registry.models_dir / model_filename
        registry.models_dir.mkdir(parents=True, exist_ok=True)
        save_model(model_metadata, str(model_path))

        model_id = model_path.stem
        registry.register(model_id, model_path, model_metadata)

        r2_clamped = min(max(r2, 0.0), 1.0)
        composite = r2_clamped * 100
        if composite > 80:
            quality_label = "Excellent"
        elif composite > 60:
            quality_label = "Good"
        else:
            quality_label = "Needs Improvement"

        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "algorithm": algo,
            "target": target,
            "features": selected,
            "log_target": bool(log_t),
            "log_target_decision": log_target_decision,
            "evaluation": {
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
                "residual_bias_p_value": float(p_value),
                "quality_label": quality_label,
            },
            "cv_scores": {k: float(v) for k, v in scores.items()} if scores else {},
            "feature_importances": importances_block,
            "rows_loaded": rows_loaded,
            "rows_used": len(data),
            "calibration_size": calibration.get("n_calibration"),
        }

    try:
        result = await asyncio.to_thread(_do_train)
    except FileNotFoundError as e:
        return _error("file_not_found", str(e))
    except ValueError as e:
        return _error("train_error", str(e))
    except Exception as e:
        return _error("internal_error", str(e))

    return _ok(result)


@mcp.tool()
async def generate_report(
    model_id: str,
    output_path: str | None = None,
) -> str:
    """Generate a procurement-style model-quality PDF report (3 pages).

    Page 1: summary metrics + predicted vs actual scatter.
    Page 2: error distribution + median % error by price band.
    Page 3: top-N feature importance.

    Works best with models trained via the MCP train tool (which stores
    holdout data). For older models, the report may be unavailable.
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    import numpy as np

    y_test = loaded.get("holdout_y_test")
    y_pred = loaded.get("holdout_y_pred")
    if y_test is None or y_pred is None:
        return _error(
            "no_holdout_data",
            "This model doesn't have stored holdout data (trained before MCP "
            "support). Retrain via the MCP train tool to enable report generation.",
        )

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    if output_path is None:
        output_path = str(registry.models_dir / f"{model_id}_report.pdf")

    def _generate() -> str:
        import matplotlib
        matplotlib.use("agg")
        from p2predict import plotting
        from p2predict.model_utils import inner_pipeline
        from p2predict.training import extract_feature_importances

        try:
            importances = extract_feature_importances(
                loaded["model"], loaded.get("background_sample")
            )
        except Exception:
            importances = None

        plotting.plot_results_pdf(
            y_test_arr,
            y_pred_arr,
            output_path,
            target_name=loaded.get("target_feature", "Price"),
            model_name=loaded.get("model_name"),
            n_train=None,
            training_date=loaded.get("training_date"),
            feature_importances=importances,
        )
        return output_path

    try:
        path = await asyncio.to_thread(_generate)
    except Exception as e:
        return _error("report_error", str(e))

    return _ok({
        "model_id": model_id,
        "report_path": path,
    })


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("model://{model_id}")
async def model_resource(model_id: str) -> str:
    """Model metadata as a resource."""
    registry = _get_registry()
    try:
        info = await asyncio.to_thread(registry.get_info, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))
    return _ok(info.to_dict())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="P2Predict MCP server — parametric price benchmarking for AI agents"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing .model files (default: models)",
    )
    args = parser.parse_args()

    global _registry
    _registry = ModelRegistry(Path(args.models_dir).resolve())

    mcp.run(transport="stdio")
