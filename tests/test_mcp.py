"""Integration tests for the P2Predict MCP server tools."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
    m = result["models"][0]
    assert m["model_id"] == model_id
    assert m["target"] == "Price"
    assert len(m["features"]) == 4


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


# ---------------------------------------------------------------------------
# predict_interval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_interval(model_id):
    result = _parse(await mcp_server.predict_interval(model_id, SAMPLE_FEATURES, 90))
    assert "error" not in result
    iv = result["interval"]
    assert iv["low"] <= result["prediction"] <= iv["high"]


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
# generate_report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_report(model_id, tmp_path):
    out = str(tmp_path / "report.pdf")
    result = _parse(await mcp_server.generate_report(model_id, out))
    assert "error" not in result
    assert result["report_path"] == out
    assert Path(out).exists()


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
