"""Tests for ``--json`` on both CLIs.

The headline assertions are *schema-shape* checks: every JSON response
carries ``schema_version: "1.0"`` and the top-level keys the schema
docstring in ``p2predict.json_output`` promises. These exist so a future
edit can't quietly rename a field — agents that ingest the output break
in ways that are hard to discover.

We also check that:

- The Rich-formatted output is fully suppressed (stdout is exclusively
  the JSON document).
- ``--json`` composes correctly with ``--interval``, ``--explain``,
  ``--whatif``.
- Errors emit a JSON error document on stdout (exit code 1) rather
  than a Rich-styled abort message.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from click.testing import CliRunner

from p2predict.cli.predict import main as predict_cli
from p2predict.cli.train import train as train_cli
from p2predict.json_output import JSON_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_args(csv_path, **overrides):
    args = [
        "-i", str(csv_path),
        "-t", "Price",
        "-x",
        "-a", "ridge",
        "-tf", "Weight,Region,Supplier,Size",
        "-b", "fast",
    ]
    args.extend(overrides.pop("extra", []))
    return args


def _parse_json(output: str) -> dict:
    """Locate the JSON object in stdout. Click captures everything;
    when --json is set the response is the only thing emitted, but we
    keep this resilient in case a stray line slips in during dev so
    the failure mode is a clear assertion rather than a JSONDecodeError
    with no context."""
    output = output.strip()
    assert output, "stdout was empty"
    # Trim anything before the first '{' and after the last '}' (no-op
    # when output is purely JSON, which is the contract).
    start = output.find("{")
    end = output.rfind("}")
    assert start != -1 and end != -1, f"no JSON object found in stdout: {output[:200]!r}"
    return json.loads(output[start:end + 1])


def _saved_model(tmp_path, prefix):
    matches = list((tmp_path / "models").glob(f"{prefix}_*.model"))
    assert len(matches) == 1, (
        f"expected exactly one {prefix}_*.model, found {matches}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# Train CLI — JSON success path
# ---------------------------------------------------------------------------


def test_train_json_returns_valid_document(tmp_path, monkeypatch, csv_path_clean):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(train_cli, _train_args(csv_path_clean, extra=["--json"]))

    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)

    assert doc["schema_version"] == JSON_SCHEMA_VERSION
    assert doc["command"] == "train"
    # Top-level shape promised by the schema docstring.
    for key in (
        "input", "mode", "outliers", "features_selected",
        "algorithm_selected", "log_target", "feature_importances",
        "evaluation", "model_path",
    ):
        assert key in doc, f"missing top-level key: {key}"


def test_train_json_evaluation_block_has_metrics(tmp_path, monkeypatch, csv_path_clean):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(train_cli, _train_args(csv_path_clean, extra=["--json"]))
    doc = _parse_json(result.output)

    ev = doc["evaluation"]
    for metric in ("r2", "mae", "rmse", "residual_bias_p_value", "quality_label"):
        assert metric in ev
    assert isinstance(ev["r2"], (int, float))
    assert ev["quality_label"] in ("Excellent", "Good", "Needs Improvement")


def test_train_json_saved_model_path_round_trips(tmp_path, monkeypatch, csv_path_clean):
    """The model_path returned in the JSON should point at a file that
    actually exists on disk and loads cleanly via joblib."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(train_cli, _train_args(csv_path_clean, extra=["--json"]))
    doc = _parse_json(result.output)

    saved = Path(doc["model_path"])
    assert saved.exists()
    meta = joblib.load(saved)
    assert meta["target_feature"] == "Price"


# ---------------------------------------------------------------------------
# Train CLI — JSON error paths
# ---------------------------------------------------------------------------


def test_train_json_emits_error_on_missing_input():
    runner = CliRunner()
    result = runner.invoke(train_cli, ["--json"])
    assert result.exit_code != 0
    doc = _parse_json(result.output)
    assert doc["command"] == "train"
    assert "error" in doc
    assert doc["error"]["code"] == "missing_input"


def test_train_json_emits_error_on_missing_target(tmp_path, csv_path_clean):
    runner = CliRunner()
    result = runner.invoke(train_cli, ["-i", csv_path_clean, "--json"])
    assert result.exit_code != 0
    doc = _parse_json(result.output)
    assert doc["error"]["code"] == "missing_target"


def test_train_json_emits_error_on_bad_time_column(
    tmp_path, monkeypatch, csv_path_clean
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        _train_args(csv_path_clean, extra=["--time-column", "NotAColumn", "--json"]),
    )
    assert result.exit_code != 0
    doc = _parse_json(result.output)
    assert doc["error"]["code"] == "bad_time_column"


# ---------------------------------------------------------------------------
# Predict CLI — JSON success paths
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_model_path(tmp_path, monkeypatch, csv_path_clean):
    """Train once via the JSON path so the fixture also exercises that
    code path before the predict tests run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    train_result = runner.invoke(
        train_cli, _train_args(csv_path_clean, extra=["--json"])
    )
    assert train_result.exit_code == 0, train_result.output
    return _parse_json(train_result.output)["model_path"]


def test_predict_inline_json_minimal(tmp_path, monkeypatch, trained_model_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard",
         "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)

    assert doc["schema_version"] == JSON_SCHEMA_VERSION
    assert doc["command"] == "predict"
    assert doc["mode"] == "inline"
    assert doc["model"]["target"] == "Price"
    assert doc["model"]["features"] == ["Weight", "Region", "Supplier", "Size"]
    assert len(doc["predictions"]) == 1
    assert "prediction" in doc["predictions"][0]
    assert isinstance(doc["predictions"][0]["prediction"], (int, float))


def test_predict_inline_json_with_interval(tmp_path, monkeypatch, trained_model_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard",
         "--interval", "90", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)

    assert "interval" in doc
    assert doc["interval"]["coverage"] == 0.90
    assert len(doc["interval"]["per_row"]) == 1
    row = doc["interval"]["per_row"][0]
    assert row["low"] <= row["prediction"] <= row["high"]


def test_predict_inline_json_with_explain(tmp_path, monkeypatch, trained_model_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard",
         "--explain", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)

    assert "explanation" in doc
    assert len(doc["explanation"]) == 1
    ex = doc["explanation"][0]
    for key in ("baseline", "prediction", "log_target", "contributions",
                "multiplicative_factors", "dollar_attribution", "residual"):
        assert key in ex
    # Local accuracy axiom — surfaced through the JSON.
    assert isinstance(ex["contributions"], list)
    assert all("feature" in c and "value" in c for c in ex["contributions"])


def test_predict_inline_json_with_whatif(tmp_path, monkeypatch, trained_model_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard",
         "--whatif", "Region:CN", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)

    assert "whatif" in doc
    wf = doc["whatif"]
    assert "Region" in wf["changes"]
    assert wf["changes"]["Region"]["to"] == "CN"
    assert isinstance(wf["delta"], (int, float))
    assert isinstance(wf["delta_pct"], (int, float))
    assert "changed_contributions" in wf


def test_predict_inline_json_composes_all_three_extras(
    tmp_path, monkeypatch, trained_model_path
):
    """--interval, --explain, --whatif should all coexist in one response."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard",
         "--interval", "90", "--explain", "--whatif", "Region:CN",
         "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert "interval" in doc
    assert "explanation" in doc
    assert "whatif" in doc


def test_predict_batch_json_includes_per_row_predictions(
    tmp_path, monkeypatch, trained_model_path, synthetic_parts
):
    monkeypatch.chdir(tmp_path)
    batch = synthetic_parts.head(3).drop(columns=["Price"])
    batch_path = tmp_path / "batch.csv"
    batch.to_csv(batch_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path), "-i", str(batch_path),
         "--interval", "90", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)

    assert doc["mode"] == "batch"
    assert len(doc["predictions"]) == 3
    assert doc["batch"]["n_rows"] == 3
    # CSV side effect still happens.
    written = pd.read_csv(batch_path)
    assert "Price" in written.columns
    assert "Price_low" in written.columns
    assert "Price_high" in written.columns


# ---------------------------------------------------------------------------
# Predict CLI — JSON error paths
# ---------------------------------------------------------------------------


def test_predict_json_emits_error_when_whatif_in_batch_mode(
    tmp_path, monkeypatch, trained_model_path, synthetic_parts
):
    monkeypatch.chdir(tmp_path)
    batch = synthetic_parts.head(2).drop(columns=["Price"])
    batch_path = tmp_path / "batch.csv"
    batch.to_csv(batch_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path), "-i", str(batch_path),
         "--whatif", "Region:CN", "--json"],
    )
    assert result.exit_code != 0
    doc = _parse_json(result.output)
    assert doc["error"]["code"] == "whatif_in_batch"


def test_predict_json_emits_error_when_no_input_with_json(
    tmp_path, monkeypatch, trained_model_path
):
    """Interactive prompts under --json are nonsensical (the agent
    can't answer prompts). We expect a clean JSON error, not a hang."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path), "--json"],
    )
    assert result.exit_code != 0
    doc = _parse_json(result.output)
    assert doc["error"]["code"] == "missing_input"


# ---------------------------------------------------------------------------
# Output cleanliness — stdout is exclusively the JSON document
# ---------------------------------------------------------------------------


def test_predict_json_stdout_is_only_json(
    tmp_path, monkeypatch, trained_model_path
):
    """No Rich-formatted banner, logo, or table should leak into stdout
    under --json. A naive parser must succeed on the raw output."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        predict_cli,
        ["-m", str(trained_model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard",
         "--json"],
    )
    assert result.exit_code == 0, result.output
    # Strict parse — no `_parse_json` flexibility. If stuff leaked,
    # json.loads raises and the test fails loudly.
    doc = json.loads(result.output.strip())
    assert doc["schema_version"] == JSON_SCHEMA_VERSION
