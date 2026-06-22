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

# NOTE: the "INTERPRETING OUTPUT" rules below are a condensed, self-contained
# mirror of "The interpretation rules" in .claude/skills/p2predict/SKILL.md.
# The SKILL.md version is the canonical, fuller teaching (it also covers the
# CLI/Python surface and feature engineering); this copy exists because
# non-Claude clients never see the skill and the server must stand alone.
# If you change a rule here, update SKILL.md too (and vice versa).
mcp = FastMCP(
    "P2Predict",
    instructions=(
        "P2Predict is a parametric price/cost benchmarking toolkit for "
        "procurement. The person you are helping is almost always a "
        "category manager or buyer, NOT a data scientist — they do not know "
        "what a 'feature', 'target', 'leakage', or 'log-target' is. Your job "
        "is to do that thinking for them and explain results in plain "
        "procurement language ('this supplier adds $0.72', not 'feature "
        "importance 0.4').\n"
        "\n"
        "SPEAK BUSINESS, NOT STATISTICS. Never say these words to the user — "
        "translate every one:\n"
        "  • 'SHAP' / 'attribution' / 'contribution' -> 'what's driving the "
        "price' (the JSON gives you `price_drivers` and `starting_point` "
        "already in dollars and percent — quote those).\n"
        "  • 'log-target' -> 'I'm modelling on a percentage scale so the likely-"
        "range never goes negative on cheap parts'.\n"
        "  • 'R²' / 'p-value' / 'residual bias' / 'feature importance' -> use the "
        "computed verdicts instead: every quality block carries a `headline` and "
        "per-band / per-feature `say_to_user` sentence written in plain words. "
        "Quote those; do not read raw metrics aloud.\n"
        "  • 'feature' / 'target' -> 'spec' / 'the price'.\n"
        "The raw statistical keys stay in the JSON for your reasoning — just "
        "don't surface their names to a category manager.\n"
        "\n"
        "DISCOVER & ANALYSE: use list_models to find trained models, then "
        "predict, explain, predict_interval, or what_if on parts. Use "
        "get_model_quality to judge whether a model is trustworthy before "
        "quoting its numbers. Lead with its computed `verdict`: 'trustworthy' "
        "(benchmark against it), 'usable' (unbiased but modest — relative "
        "comparisons and benchmarks only, not a single-part appraisal), "
        "'unreliable' (biased residuals — relative comparisons only, never an "
        "absolute target), or 'insufficient_data'/'unknown' (too little data "
        "to judge — treat metrics as indicative). It also returns per-price-"
        "band reliability and per-feature signal strength as computed flags.\n"
        "\n"
        "BUILD A MODEL: do NOT call `train` directly on a user's CSV first. "
        "Call `propose_training_plan` first, relay its plain_summary and "
        "questions_for_the_user to the user, and only call `train` after they "
        "confirm. If you do call `train` directly, it applies safe defaults "
        "(it screens out target-leakage columns and recommends a log-target "
        "for price/cost targets) and reports them in `warnings` — surface "
        "those warnings to the user.\n"
        "\n"
        "INTERPRETING OUTPUT (this is where the value is — apply every time):\n"
        "1. log-target: prices/costs are multiplicative; a log-target keeps "
        "intervals positive and makes SHAP read as percentages. Recommend it "
        "for any price/cost target — don't trust 'auto' to catch it.\n"
        "2. SHAP has an axiom check (baseline +/x contributions = prediction); "
        "if it fails the explanation is unsound. Sign-check contributions: a "
        "counterintuitive sign on a LOW-importance feature means that feature "
        "is under-sampled, not that the world is upside-down — lean on the "
        "high-importance, correctly-signed drivers.\n"
        "3. The interval width is the per-part trust signal. A wide band — or "
        "a lower bound at/below $0 on an additive model — means 'I'm unsure "
        "here; get a quote, don't benchmark.' Always show the interval, not "
        "just the point estimate, when the user will act on the number.\n"
        "4. Judge a model by residual-bias (unbiasedness), not R2 alone: a "
        "modest-R2 but unbiased model is more trustworthy for procurement than "
        "a higher-R2 biased one.\n"
        "5. Before quoting a finding to a stakeholder, check its feature's "
        "importance — a finding resting on a 1-2% feature is a hypothesis, not "
        "a number to negotiate against.\n"
        "\n"
        "Never present a single high-value part's point estimate as a final "
        "appraisal — use the model to set the target and find the lever, then "
        "get a real quote for the decision."
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
    """Explain what is driving a part's predicted price, spec by spec.

    Returns a business-ready view to quote directly — `starting_point` (the
    baseline price every part starts from) and `price_drivers` (each spec /
    supplier's effect in BOTH dollars and percent, biggest mover first) — plus
    the underlying technical attribution for your own reasoning. top_n controls
    how many top drivers are highlighted (default 3).

    Reading it for the user: the explanation carries an axiom check
    (baseline +/x contributions = prediction); if it fails the explanation is
    unsound. SIGN-CHECK the drivers against intuition — a counterintuitive
    sign (e.g. "more cells -> cheaper") on a LOW-importance driver means that
    spec is under-sampled, not that the world is upside-down. Quote the
    high-importance, correctly-signed drivers; flag the rest as noise. State
    effects in the user's terms ("this supplier adds $0.72" / "+18%") — never
    say "SHAP", "contribution", or "baseline" to a category manager.
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

    Reading it for the user: the band WIDTH is the per-part trust signal. A
    tight band = predict with confidence; a very wide band — or a lower bound
    at/below $0 on an additive (non-log) model — means "I'm genuinely unsure
    on this part; get a quote, don't benchmark." Always surface the range, not
    just the point estimate, when the user will act on the number. A negative
    lower bound is a sign the model should have used a log-target.
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

    Returns a plain `summary` to quote directly (does the change add or save,
    how many dollars, what percent, old vs. new price), plus both predictions,
    the delta, and per-driver attribution of each change for your reasoning.
    Answers "what if we switch from supplier A to B?" Set coverage to null to
    skip intervals.
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
async def propose_training_plan(
    csv_path: str,
    target: str,
    max_features: int = 6,
) -> str:
    """Inspect a training CSV and return a plain-language plan BEFORE training.

    Call this first whenever a user wants to build a should-cost / pricing
    model. It reads the CSV, decides what it would predict, which columns it
    would use as specs, which it would leave out (and why — target leakage,
    ID-like columns), and whether the target should use a log-target. It
    trains nothing and writes nothing.

    Relay `plain_summary` and `questions_for_the_user` to the user in their
    own language, get confirmation, then call `train` (passing the agreed
    `features` and `log_target`).
    """
    registry = _get_registry()  # noqa: F841 — validates server is initialised

    def _do_plan() -> dict:
        import pandas as pd

        from p2predict.feature_selection import (
            find_high_variation_features,
            find_leaky_features,
            find_no_variation_features,
            get_most_predictable_features,
        )
        from p2predict.trained_model_io import load_csv_file
        from p2predict.training import resolve_log_target

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

        y = pd.to_numeric(data[target], errors="coerce").dropna()

        # Columns to leave out, with reasons the user can understand.
        leaky = find_leaky_features(data, target)
        leaky_names = {d["feature"] for d in leaky}

        no_var = [c for c in find_no_variation_features(data) if c != target]
        high_var = find_high_variation_features(data)
        id_like = [
            c for c in high_var
            if c != target
            and c not in leaky_names
            and not pd.api.types.is_numeric_dtype(data[c])
        ]

        excluded = []
        for d in leaky:
            excluded.append({"column": d["feature"], "reason": d["reason"]})
        for c in id_like:
            excluded.append({
                "column": c,
                "reason": "Looks like an ID / free-text column (almost every "
                          "row is unique), not a spec the model can learn from.",
            })
        for c in no_var:
            excluded.append({
                "column": c,
                "reason": "Same value in every row — carries no information.",
            })

        drop_all = leaky_names | set(id_like) | set(no_var)
        ranked = get_most_predictable_features(data, target, output_only_headers=True)
        candidate_specs = [c for c in ranked.tolist() if c not in drop_all]
        cap = max(2, min(len(candidate_specs), max_features))
        selected = candidate_specs[:cap]

        # Log-target recommendation.
        _, auto_decision = resolve_log_target(y, mode="auto")
        positive = bool((y > 0).all()) and len(y) > 0
        recommend_log_target = "on" if positive else "off"

        questions = [
            f"I'll predict '{target}'. Is that the price/cost you actually pay "
            "per part? If not, tell me which column is.",
        ]
        if leaky:
            cols = ", ".join(f"'{d['feature']}'" for d in leaky)
            questions.append(
                f"I'm leaving out {cols} because it's almost the same number as "
                f"'{target}' — it would make the model 'cheat'. OK to exclude?"
            )
        if positive and recommend_log_target == "on":
            questions.append(
                "I'll model this on a percentage scale (log-target) so the "
                "likely-range never goes negative on cheap parts. Sound good?"
            )

        plain_summary = (
            f"I found {rows_loaded} rows. I can build a model that estimates "
            f"'{target}' from {len(selected)} spec column(s): "
            f"{', '.join(selected)}."
        )
        if excluded:
            plain_summary += (
                f" I'd leave out {len(excluded)} column(s) "
                f"({', '.join(e['column'] for e in excluded)}) — see "
                "i_am_leaving_out for why."
            )

        return {
            "status": "needs_confirmation",
            "plain_summary": plain_summary,
            "i_will_predict": target,
            "i_will_use_these_specs": selected,
            "i_am_leaving_out": excluded,
            "recommended_log_target": recommend_log_target,
            "log_target_auto_decision": auto_decision,
            "rows_available": rows_loaded,
            "questions_for_the_user": questions,
            "to_proceed": (
                "After the user confirms, call train(csv_path, target, "
                "features=i_will_use_these_specs, "
                "log_target=recommended_log_target)."
            ),
        }

    try:
        result = await asyncio.to_thread(_do_plan)
    except FileNotFoundError as e:
        return _error("file_not_found", str(e))
    except ValueError as e:
        return _error("plan_error", str(e))
    except Exception as e:
        return _error("internal_error", str(e))

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
    allow_leaky_features: bool = False,
) -> str:
    """Train a new P2Predict model from a local CSV file.

    Prefer calling `propose_training_plan` first and confirming with the user
    — this tool is the execution step. The CSV must have spec columns and a
    price/cost target column. Training runs locally; no data leaves the
    machine. The trained model is saved and immediately available.

    Safe defaults (always surfaced in the returned `warnings` list):
      - When features are auto-selected (features=None), columns that look
        like target leakage — a near-duplicate of the price being predicted —
        are excluded automatically.
      - For a strictly-positive (price/cost) target where the automatic skew
        test leaves the log-target off, the result recommends log_target="on".

    algorithm: "auto" (default), "ridge", "random_forest", or "xgboost".
    budget: "fast" (default) or "thorough".
    log_target: "auto" (default), "on", or "off". Use "on" for prices.
    allow_leaky_features: set True only to override the leakage guard and
        train on an explicitly-requested feature that looks like leakage.
    """
    registry = _get_registry()

    def _do_train() -> dict:
        import pandas as pd

        from p2predict import auto_train, Serialize_Trained_Model, save_model
        from p2predict.feature_selection import (
            find_leaky_features,
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

        warnings: list[str] = []
        leaky = find_leaky_features(data, target)
        leaky_names = {d["feature"] for d in leaky}

        if features:
            missing = [f for f in features if f not in data.columns]
            if missing:
                raise ValueError(f"Requested features not in CSV: {missing}")

            explicit_leaky = [d for d in leaky if d["feature"] in set(features)]
            if explicit_leaky and not allow_leaky_features:
                # Stop and ask rather than train a confidently-wrong model.
                return {
                    "status": "needs_confirmation",
                    "reason": "target_leakage",
                    "message": (
                        "Some requested features look like target leakage — a "
                        "near-duplicate of the value you're predicting, not a "
                        "real spec. Training on them produces a model that looks "
                        "near-perfect but is useless on real parts."
                    ),
                    "leaky_features": explicit_leaky,
                    "to_proceed": (
                        "Re-call train without these features (recommended), or "
                        "pass allow_leaky_features=true to override deliberately."
                    ),
                }
            selected = list(features)
        else:
            ranked = get_most_predictable_features(data, target, output_only_headers=True)
            # Safe default: never auto-select a leakage column.
            ranked = [c for c in ranked.tolist() if c not in leaky_names]
            n_ranked = len(ranked)
            cap = max(2, min(n_ranked, max_features))
            selected = ranked[:cap]
            if leaky_names:
                warnings.append(
                    "Auto-excluded likely target-leakage column(s) from feature "
                    f"selection: {sorted(leaky_names)}. "
                    + "; ".join(d["reason"] for d in leaky)
                )

        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(
            data, selected, target
        )

        log_target_override, log_target_decision = resolve_log_target(
            y_train, mode=log_target
        )

        # Prices/costs are multiplicative: a log-target keeps intervals strictly
        # positive and makes SHAP read as percentages. The skew-based "auto"
        # test under-fires on samples that happen to look symmetric, so for a
        # strictly-positive target left additive by auto, recommend "on".
        if (
            log_target == "auto"
            and not log_target_override
            and bool((y_train > 0).all())
        ):
            warnings.append(
                "Target is strictly positive (price/cost-like) but the automatic "
                "skew test left the log-target OFF, so intervals are additive and "
                "can go negative on cheap parts. Consider re-training with "
                "log_target=\"on\" for percentage-based, always-positive intervals."
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

        from p2predict.quality import r2_quality_label
        quality_label = r2_quality_label(r2)

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
            "excluded_leaky_features": leaky,
            "warnings": warnings,
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


def _quality_report_for(loaded: dict, include_holdout: bool = False) -> dict:
    """Build the structured quality report for a loaded model (shared by
    get_model_quality and generate_report). Raises ValueError('no_holdout_data')."""
    from p2predict.quality import build_quality_report
    from p2predict.training import extract_feature_importances

    try:
        importances = extract_feature_importances(
            loaded["model"], loaded.get("background_sample")
        )
    except Exception:
        importances = None
    return build_quality_report(loaded, importances, include_holdout=include_holdout)


@mcp.tool()
async def get_model_quality(model_id: str, include_holdout: bool = False) -> str:
    """Structured, agent-readable model-quality report — the JSON form of the PDF.

    Use this (not just generate_report, which only writes a PDF) when you need
    to *reason about or relay* model quality. Every judgment is computed so you
    don't eyeball thresholds:

      - `assessment.verdict` — LEAD WITH THIS. One of: 'trustworthy' | 'usable'
        | 'unreliable' (biased residuals) | 'unknown' (bias not measurable) |
        'insufficient_data' (too few holdout points to judge). It folds bias and
        sample size into the headline — e.g. a modest-R² but unbiased model is
        'usable', not just 'Needs Improvement'. `assessment.confidence` is
        'high' | 'limited' | 'insufficient'.
      - `calibration_by_price_band[].reliability` — 'trust' | 'caution' |
        'quote' per price range (with `low_confidence` when a band is thin).
        Each band carries a `say_to_user` sentence in plain words — quote it to
        tell the user which prices to benchmark vs. get a quote on.
      - `feature_importance[].signal` — 'strong' | 'moderate' | 'weak', each
        with its own `say_to_user` sentence. Only quote findings resting on
        'strong' drivers to a stakeholder.

    Lead with the plain-language `headline` / `say_to_user` strings; the raw
    `metrics` (R², p-values) are for your reasoning, not for the user's ears.

    Set include_holdout=true to also get the raw actual/predicted arrays, so an
    agent with a code/plotting tool can draw its own charts (predicted-vs-actual,
    residuals, error-by-band). `metrics` and `provenance` round out the report.

    Requires a model trained via the MCP train tool (which stores holdout data).
    """
    registry = _get_registry()
    try:
        loaded = await asyncio.to_thread(registry.load, model_id)
    except FileNotFoundError as e:
        return _error("model_not_found", str(e))

    try:
        report = await asyncio.to_thread(_quality_report_for, loaded, include_holdout)
    except ValueError:
        return _error(
            "no_holdout_data",
            "This model has no stored holdout data (trained before MCP support). "
            "Retrain via the MCP train tool to enable the quality report.",
        )
    except Exception as e:
        return _error("quality_error", str(e))

    return _ok({"model_id": model_id, **report})


@mcp.tool()
async def generate_report(
    model_id: str,
    output_path: str | None = None,
) -> str:
    """Generate a procurement-style model-quality PDF report (3 pages).

    Page 1: summary metrics + predicted vs actual scatter.
    Page 2: error distribution + median % error by price band.
    Page 3: top-N feature importance.

    The PDF is the human deliverable; the return value also echoes the same
    numbers as a structured `quality` block (identical to get_model_quality)
    so you can both hand the user the file AND reason over the metrics.

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

    # Echo the same numbers as structured data so the agent can reason over the
    # report, not just hand the user a PDF path.
    try:
        quality = await asyncio.to_thread(_quality_report_for, loaded)
    except Exception:
        quality = None

    return _ok({
        "model_id": model_id,
        "report_path": path,
        "quality": quality,
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
