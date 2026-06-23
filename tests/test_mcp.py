"""Integration tests for the P2Predict MCP server tools."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp", reason="MCP SDK not installed (pip install p2predict[mcp])")

from p2predict import auto_train, save_model, Serialize_Trained_Model
from p2predict.intervals import compute_calibration_residuals
from p2predict.prepare_data import prepare_data
from p2predict.mcp.registry import ModelRegistry
from p2predict.mcp import server as mcp_server


@pytest.fixture
def mcp_registry(tmp_path, synthetic_parts):
    """Train a model, save it, wire up a ModelRegistry."""
    features = ["Weight", "Region", "Supplier", "Size"]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        synthetic_parts, features, "Price"
    )
    model, algo, scores, log_t = auto_train(
        X_train, y_train, num, cat, budget="fast"
    )
    bg = X_train.sample(min(100, len(X_train)), random_state=0).reset_index(drop=True)
    cal = compute_calibration_residuals(model, X_test, y_test)
    y_pred = model.predict(X_test)

    meta = Serialize_Trained_Model(
        algo, features, "Price", model, scores[algo],
        log_target=log_t, background_sample=bg, calibration=cal,
    )
    meta["holdout_y_test"] = y_test.tolist()
    meta["holdout_y_pred"] = y_pred.tolist()

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / f"{algo}_Price_test.model"
    save_model(meta, str(model_path))

    registry = ModelRegistry(models_dir)
    mcp_server._registry = registry
    yield registry, model_path.stem
    mcp_server._registry = None


@pytest.fixture
def model_id(mcp_registry):
    return mcp_registry[1]


@pytest.fixture
def registry(mcp_registry):
    return mcp_registry[0]


SAMPLE_FEATURES = {
    "Weight": 15.0,
    "Region": "EU",
    "Supplier": "A",
    "Size": "Standard",
}


def _parse(result: str) -> dict:
    return json.loads(result)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models(registry, model_id):
    result = _parse(await mcp_server.list_models())
    assert "models" in result
    assert len(result["models"]) == 1
    # Build stamp so a caller can confirm which server build is running.
    assert "server" in result
    assert result["server"]["version"]
    assert "git_sha" in result["server"] and "source" in result["server"]
    m = result["models"][0]
    assert m["model_id"] == model_id
    assert m["target"] == "Price"
    assert len(m["features"]) == 4
    # Discovery leads with a plain line and hides the raw stats by default.
    assert m["say_to_user"]
    assert "algorithm" not in m and "r2" not in m and "log_target" not in m
    # Opt-in exposes the internals for the agent's own reasoning.
    full = _parse(await mcp_server.list_models(include_internal=True))["models"][0]
    assert "algorithm" in full and "r2" in full


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_info(model_id):
    result = _parse(await mcp_server.get_model_info(model_id))
    assert "error" not in result
    assert result["model_id"] == model_id
    assert "feature_types" in result
    assert result["feature_types"]["Weight"] == "Numerical"
    assert result["feature_types"]["Region"] == "Categorical"
    assert "categories" in result
    # Plain line present; raw stats gated out by default.
    assert result["say_to_user"]
    assert "algorithm" not in result and "log_target" not in result


@pytest.mark.asyncio
async def test_get_model_info_not_found(registry):
    result = _parse(await mcp_server.get_model_info("nonexistent"))
    assert result["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict(model_id):
    result = _parse(await mcp_server.predict(model_id, SAMPLE_FEATURES))
    assert "error" not in result
    assert isinstance(result["prediction"], float)
    assert result["target"] == "Price"


@pytest.mark.asyncio
async def test_predict_missing_feature(registry, model_id):
    result = _parse(await mcp_server.predict(model_id, {"Weight": 15}))
    assert result["error"]["code"] == "missing_feature"


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_batch(model_id):
    rows = [SAMPLE_FEATURES, {**SAMPLE_FEATURES, "Region": "CN"}]
    result = _parse(await mcp_server.predict_batch(model_id, rows))
    assert "error" not in result
    assert len(result["predictions"]) == 2
    for p in result["predictions"]:
        assert isinstance(p["prediction"], float)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain(model_id):
    result = _parse(await mcp_server.explain(model_id, SAMPLE_FEATURES))
    assert "error" not in result
    assert "explanation" in result
    expl = result["explanation"]
    assert "contributions" in expl
    assert len(expl["contributions"]) > 0
    total = sum(c["value"] for c in expl["contributions"])
    assert abs(total - (expl["prediction"] - expl["baseline"])) < 0.5
    assert len(result["top_drivers"]) <= 3
    # Business-facing view the agent quotes to a category manager.
    assert "starting_point" in expl
    assert expl["price_drivers"], "price_drivers must be populated"
    top = expl["price_drivers"][0]
    assert {"driver", "effect_dollars", "effect_pct"} <= set(top)
    # Sorted biggest-mover-first by absolute dollar effect.
    effects = [abs(d["effect_dollars"] or 0.0) for d in expl["price_drivers"]]
    assert effects == sorted(effects, reverse=True)


# ---------------------------------------------------------------------------
# predict_interval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_interval(model_id):
    result = _parse(await mcp_server.predict_interval(model_id, SAMPLE_FEATURES, 90))
    assert "error" not in result
    iv = result["interval"]
    assert iv["low"] <= result["prediction"] <= iv["high"]
    # Per-part trust read the agent quotes instead of bare bounds.
    assert iv["reliability"] in {"trust", "caution", "quote"}
    assert iv["say_to_user"]


@pytest.mark.asyncio
async def test_predict_interval_bad_coverage(registry, model_id):
    result = _parse(await mcp_server.predict_interval(model_id, SAMPLE_FEATURES, 0))
    assert result["error"]["code"] == "bad_coverage"


# ---------------------------------------------------------------------------
# what_if
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_what_if(model_id):
    result = _parse(await mcp_server.what_if(
        model_id, SAMPLE_FEATURES, {"Region": "CN"}, 90
    ))
    assert "error" not in result
    wi = result["whatif"]
    assert abs(wi["delta"] - (wi["counterfactual_prediction"] - wi["base_prediction"])) < 0.01
    # Plain-language summary the agent quotes directly.
    summary = wi["summary"]
    assert summary["direction"] in {"adds", "saves", "no change"}
    assert summary["effect_dollars"] >= 0
    assert abs(summary["effect_dollars"] - abs(wi["delta"])) < 0.01


@pytest.mark.asyncio
async def test_what_if_bad_feature(registry, model_id):
    result = _parse(await mcp_server.what_if(
        model_id, SAMPLE_FEATURES, {"Nonexistent": "X"}, 90
    ))
    assert result["error"]["code"] == "bad_whatif"


# ---------------------------------------------------------------------------
# predict_from_csv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_from_csv(model_id, tmp_path, synthetic_parts):
    csv = tmp_path / "batch.csv"
    synthetic_parts.head(5).to_csv(csv, index=False)
    result = _parse(await mcp_server.predict_from_csv(
        model_id, str(csv), with_explanation=False, coverage=90
    ))
    assert "error" not in result
    assert result["n_rows"] == 5
    assert len(result["predictions"]) == 5
    assert "interval" in result["predictions"][0]


@pytest.mark.asyncio
async def test_predict_from_csv_not_found(registry, model_id):
    result = _parse(await mcp_server.predict_from_csv(
        model_id, "/nonexistent.csv"
    ))
    assert result["error"]["code"] == "file_not_found"


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train(registry, tmp_path, synthetic_parts):
    csv = tmp_path / "train_data.csv"
    synthetic_parts.to_csv(csv, index=False)
    result = _parse(await mcp_server.train(
        csv_path=str(csv),
        target="Price",
        features=["Weight", "Region", "Supplier", "Size"],
        algorithm="ridge",
        budget="fast",
    ))
    assert "error" not in result
    assert "model_id" in result
    assert result["algorithm"] == "ridge"
    assert result["evaluation"]["r2"] > 0

    listing = _parse(await mcp_server.list_models())
    ids = [m["model_id"] for m in listing["models"]]
    assert result["model_id"] in ids


@pytest.mark.asyncio
async def test_train_bad_target(registry, tmp_path, synthetic_parts):
    csv = tmp_path / "train_data.csv"
    synthetic_parts.to_csv(csv, index=False)
    result = _parse(await mcp_server.train(
        csv_path=str(csv), target="NonexistentColumn"
    ))
    assert result["error"]["code"] == "train_error"


# ---------------------------------------------------------------------------
# train / propose_training_plan guardrails (leakage + log-target)
# ---------------------------------------------------------------------------


def _leaky_csv(tmp_path, synthetic_parts):
    """A copy of the synthetic data with an added target-leakage column."""
    df = synthetic_parts.copy()
    # Price_at_1k is ~the same number as the target Price -> leakage.
    df["Price_at_1k"] = df["Price"] * 0.5 + np.random.default_rng(0).normal(
        0, 0.005, len(df)
    )
    csv = tmp_path / "leaky.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.asyncio
async def test_train_auto_excludes_leaky_feature(registry, tmp_path, synthetic_parts):
    csv = _leaky_csv(tmp_path, synthetic_parts)
    result = _parse(await mcp_server.train(
        csv_path=str(csv), target="Price", algorithm="ridge", budget="fast",
    ))
    assert "error" not in result
    # The leaky column must not be among the auto-selected features...
    assert "Price_at_1k" not in result["features"]
    # ...and the user must be told it was excluded.
    assert any("Price_at_1k" in w for w in result["warnings"])
    assert any(d["feature"] == "Price_at_1k" for d in result["excluded_leaky_features"])


@pytest.mark.asyncio
async def test_train_blocks_explicit_leaky_feature(registry, tmp_path, synthetic_parts):
    csv = _leaky_csv(tmp_path, synthetic_parts)
    result = _parse(await mcp_server.train(
        csv_path=str(csv), target="Price",
        features=["Weight", "Price_at_1k"], algorithm="ridge", budget="fast",
    ))
    # Should refuse and ask rather than train a leaking model.
    assert result.get("status") == "needs_confirmation"
    assert result["reason"] == "target_leakage"
    assert any(d["feature"] == "Price_at_1k" for d in result["leaky_features"])


@pytest.mark.asyncio
async def test_train_override_allows_leaky_feature(registry, tmp_path, synthetic_parts):
    csv = _leaky_csv(tmp_path, synthetic_parts)
    result = _parse(await mcp_server.train(
        csv_path=str(csv), target="Price",
        features=["Weight", "Price_at_1k"], algorithm="ridge", budget="fast",
        allow_leaky_features=True,
    ))
    assert "error" not in result
    assert "Price_at_1k" in result["features"]


@pytest.mark.asyncio
async def test_train_recommends_log_target_for_positive_price(
    registry, tmp_path, synthetic_parts
):
    csv = tmp_path / "clean.csv"
    synthetic_parts.to_csv(csv, index=False)
    result = _parse(await mcp_server.train(
        csv_path=str(csv), target="Price",
        features=["Weight", "Region", "Supplier", "Size"],
        algorithm="ridge", budget="fast", log_target="auto",
    ))
    assert "error" not in result
    # synthetic_parts price is near-symmetric -> auto leaves log off -> recommend on.
    if not result["log_target"]:
        assert any("log_target" in w for w in result["warnings"])


@pytest.mark.asyncio
async def test_propose_training_plan_flags_leakage(registry, tmp_path, synthetic_parts):
    csv = _leaky_csv(tmp_path, synthetic_parts)
    result = _parse(await mcp_server.propose_training_plan(
        csv_path=str(csv), target="Price",
    ))
    assert result["status"] == "needs_confirmation"
    assert "Price_at_1k" not in result["i_will_use_these_specs"]
    assert any(e["column"] == "Price_at_1k" for e in result["i_am_leaving_out"])
    assert result["recommended_log_target"] == "on"
    assert result["questions_for_the_user"]


@pytest.mark.asyncio
async def test_propose_training_plan_bad_target(registry, tmp_path, synthetic_parts):
    csv = tmp_path / "clean.csv"
    synthetic_parts.to_csv(csv, index=False)
    result = _parse(await mcp_server.propose_training_plan(
        csv_path=str(csv), target="Nope",
    ))
    assert result["error"]["code"] == "plan_error"


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_report(model_id, tmp_path):
    out = str(tmp_path / "report.pdf")
    result = _parse(await mcp_server.generate_report(model_id, out))
    assert "error" not in result
    assert result["report_path"] == out
    assert Path(out).exists()
    # The PDF call also echoes the structured quality block.
    assert result["quality"] is not None
    assert "assessment" in result["quality"]


# ---------------------------------------------------------------------------
# get_model_quality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_quality(model_id):
    result = _parse(await mcp_server.get_model_quality(model_id))
    assert "error" not in result
    assert result["model_id"] == model_id
    # Structured, agent-readable quality with computed verdicts.
    assert "metrics" in result and "quality_label" in result["metrics"]
    a = result["assessment"]
    assert set(a) >= {"verdict", "quality_label", "accuracy", "unbiased",
                      "confidence", "headline"}
    assert a["verdict"] in {"trustworthy", "usable", "unreliable", "unknown",
                            "insufficient_data"}
    assert a["confidence"] in {"high", "limited", "insufficient"}
    assert a["unbiased"] in (True, False, None)
    for band in result["calibration_by_price_band"]:
        assert band["reliability"] in {"trust", "caution", "quote"}
        # Plain sentence the agent quotes — no "median % error" leaks through.
        assert band["say_to_user"]
        assert "% error" not in band["say_to_user"]
    for feat in result["feature_importance"]:
        assert feat["signal"] in {"strong", "moderate", "weak"}
        assert feat["say_to_user"]
    # Default payload is business-only: raw stats gated out, every quotable
    # string free of the jargon a category manager has never heard.
    assert "r2" not in result["metrics"]
    assert "algorithm" not in result["provenance"]
    assert "log_target" not in result["provenance"]
    banned = ("shap", "r²", "p-value", "holdout", "residual", "log-target")
    quotables = [result["assessment"]["headline"]]
    quotables += [b["say_to_user"] for b in result["calibration_by_price_band"]]
    quotables += [f["say_to_user"] for f in result["feature_importance"]]
    for text in quotables:
        low = text.lower()
        for term in banned:
            assert term not in low, f"{term!r} leaked: {text!r}"


@pytest.mark.asyncio
async def test_get_model_quality_include_metrics_restores_raw(model_id):
    result = _parse(await mcp_server.get_model_quality(model_id, include_metrics=True))
    assert "error" not in result
    assert "r2" in result["metrics"]
    assert "log_target" in result["provenance"]


@pytest.mark.asyncio
async def test_get_model_quality_include_holdout(model_id):
    result = _parse(await mcp_server.get_model_quality(model_id, include_holdout=True))
    assert "error" not in result
    assert "holdout" in result
    assert len(result["holdout"]["y_actual"]) == len(result["holdout"]["y_predicted"])


@pytest.mark.asyncio
async def test_get_model_quality_not_found(registry):
    result = _parse(await mcp_server.get_model_quality("nonexistent"))
    assert result["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------


def test_registry_scan(registry, model_id):
    infos = registry.scan()
    assert len(infos) == 1
    assert infos[0].model_id == model_id
    assert infos[0].target == "Price"
    assert infos[0].has_calibration is True
    assert infos[0].has_background is True


def test_registry_model_not_found(registry):
    with pytest.raises(FileNotFoundError):
        registry.load("nonexistent_model")
