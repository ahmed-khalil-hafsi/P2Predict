"""Integration tests for the P2Predict MCP server tools."""
from __future__ import annotations

import json
import os
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
def log_target_model_id(tmp_path, synthetic_parts_skewed):
    """Train and wire a LOG-TARGET model (skewed prices trigger the log wrap).

    The explain/predict price-space agreement regression test needs a model
    whose inner estimator predicts log(price); the default mcp_registry model
    is additive and would not exercise that path.
    """
    features = ["Weight", "Region", "Supplier", "Size"]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        synthetic_parts_skewed, features, "Price"
    )
    model, algo, scores, log_t = auto_train(
        X_train, y_train, num, cat, budget="fast"
    )
    assert log_t, "skewed fixture expected to yield a log-target model"
    bg = X_train.sample(min(100, len(X_train)), random_state=0).reset_index(drop=True)
    cal = compute_calibration_residuals(model, X_test, y_test)

    meta = Serialize_Trained_Model(
        algo, features, "Price", model, scores[algo],
        log_target=log_t, background_sample=bg, calibration=cal,
    )
    models_dir = tmp_path / "log_models"
    models_dir.mkdir()
    model_path = models_dir / f"{algo}_Price_log_test.model"
    save_model(meta, str(model_path))

    registry = ModelRegistry(models_dir)
    mcp_server._registry = registry
    yield model_path.stem
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
    # Plain point predictions by default — no enrichment unless asked.
    assert all("interval" not in p for p in result["predictions"])
    assert all("explanation" not in p for p in result["predictions"])


@pytest.mark.asyncio
async def test_predict_batch_with_interval(model_id):
    rows = [SAMPLE_FEATURES, {**SAMPLE_FEATURES, "Region": "CN"}]
    result = _parse(await mcp_server.predict_batch(model_id, rows, coverage=90))
    assert "error" not in result
    assert result["coverage_pct"] == 90
    for p in result["predictions"]:
        iv = p["interval"]
        assert iv["low"] <= p["prediction"] <= iv["high"]
        assert iv["reliability"] in {"trust", "caution", "quote"}
        assert iv["say_to_user"]


@pytest.mark.asyncio
async def test_predict_batch_with_explanation(model_id):
    rows = [SAMPLE_FEATURES, {**SAMPLE_FEATURES, "Region": "CN"}]
    result = _parse(
        await mcp_server.predict_batch(model_id, rows, with_explanation=True)
    )
    assert "error" not in result
    for p in result["predictions"]:
        assert p["explanation"]["contributions"]


@pytest.mark.asyncio
async def test_predict_batch_bad_coverage(model_id):
    result = _parse(
        await mcp_server.predict_batch(model_id, [SAMPLE_FEATURES], coverage=0)
    )
    assert result["error"]["code"] == "bad_coverage"


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


@pytest.mark.asyncio
async def test_explain_prediction_matches_predict_log_target(log_target_model_id):
    """Regression: for a log-target model, explain's top-level `prediction`
    must be in price space and equal predict()/predict_batch(), not the inner
    model's log-space output. (Previously explain surfaced log(price).)"""
    predict_res = _parse(await mcp_server.predict(log_target_model_id, SAMPLE_FEATURES))
    explain_res = _parse(await mcp_server.explain(log_target_model_id, SAMPLE_FEATURES))
    batch_res = _parse(
        await mcp_server.predict_batch(log_target_model_id, [SAMPLE_FEATURES])
    )

    assert "error" not in predict_res and "error" not in explain_res
    assert explain_res["explanation"]["log_target"] is True

    predict_price = predict_res["prediction"]
    batch_price = batch_res["predictions"][0]["prediction"]
    explain_price = explain_res["prediction"]

    assert explain_price == pytest.approx(predict_price, rel=1e-6)
    assert explain_price == pytest.approx(batch_price, rel=1e-6)
    # The price-space prediction must differ from the raw log-space inner value
    # that the technical view still carries (guards against a no-op "fix").
    assert explain_price == pytest.approx(
        np.exp(explain_res["explanation"]["prediction"]), rel=1e-6
    )


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
async def test_predict_from_csv_default_is_point_only(model_id, tmp_path, synthetic_parts):
    # Default coverage is None: plain point predictions, no intervals, parity
    # with predict_batch's default.
    csv = tmp_path / "batch.csv"
    synthetic_parts.head(3).to_csv(csv, index=False)
    result = _parse(await mcp_server.predict_from_csv(model_id, str(csv)))
    assert "error" not in result
    assert "coverage_pct" not in result
    assert all("interval" not in p for p in result["predictions"])


@pytest.mark.asyncio
async def test_predict_from_csv_bad_coverage(model_id, tmp_path, synthetic_parts):
    csv = tmp_path / "batch.csv"
    synthetic_parts.head(3).to_csv(csv, index=False)
    result = _parse(await mcp_server.predict_from_csv(model_id, str(csv), coverage=0))
    assert result["error"]["code"] == "bad_coverage"


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


# ---------------------------------------------------------------------------
# --print-config
# ---------------------------------------------------------------------------


def _emitted_config(capsys, models_dir: Path) -> dict:
    """Run _print_config and parse the JSON block out of what it printed."""
    mcp_server._print_config(models_dir)
    text = capsys.readouterr().out
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def test_print_config_emits_valid_pasteable_json(capsys, tmp_path):
    """The whole point: the user never hand-escapes anything."""
    models = tmp_path / "models"
    models.mkdir()
    blob = _emitted_config(capsys, models)

    entry = blob["mcpServers"]["p2predict"]
    assert Path(entry["command"]).is_absolute()
    assert entry["args"][-2] == "--models-dir"
    assert entry["args"][-1] == str(models)


def test_print_config_uses_absolute_command_not_bare_name(capsys, tmp_path):
    """MCP clients don't inherit PATH, so a bare command name would fail."""
    blob = _emitted_config(capsys, tmp_path / "models")
    command = blob["mcpServers"]["p2predict"]["command"]
    assert command != "p2predict-mcp"
    assert Path(command).is_absolute()


def test_print_config_flags_missing_models_dir(capsys, tmp_path):
    mcp_server._print_config(tmp_path / "not_there")
    assert "doesn't exist yet" in capsys.readouterr().out


def test_print_config_stays_quiet_when_models_dir_exists(capsys, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    mcp_server._print_config(models)
    assert "doesn't exist yet" not in capsys.readouterr().out


def test_launch_command_prefers_console_script(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    name = "p2predict-mcp.exe" if os.name == "nt" else "p2predict-mcp"
    script = fake_bin / name
    script.touch()
    monkeypatch.setattr(mcp_server.sys, "executable", str(fake_bin / "python"))

    assert mcp_server._launch_command() == [str(script)]


def test_launch_command_falls_back_to_unresolved_interpreter(tmp_path, monkeypatch):
    """Regression: resolve() followed a venv's python symlink out to the base
    interpreter, which has no p2predict on its path — the emitted config then
    died with ModuleNotFoundError. sys.executable must be used as-is."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    real_python = tmp_path / "real_python"
    real_python.touch()
    venv_python = venv_bin / "python"
    venv_python.symlink_to(real_python)
    monkeypatch.setattr(mcp_server.sys, "executable", str(venv_python))

    command = mcp_server._launch_command()
    assert command == [str(venv_python), "-m", "p2predict.mcp"]
    assert command[0] != str(real_python)
