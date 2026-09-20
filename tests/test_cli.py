"""End-to-end CLI tests using Click's CliRunner.

These complement the per-module unit tests by exercising the actual entry
points users invoke (`p2predict_train.py` and `p2predict.py`) — wiring of
flags, flow control, and the save/load contract between the two scripts.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from click.testing import CliRunner

from p2predict.cli.predict import main as predict_cli
from p2predict.cli.train import train as train_cli


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


def _saved_model(tmp_path, prefix):
    matches = list((tmp_path / "models").glob(f"{prefix}_*.model"))
    assert len(matches) == 1, f"expected exactly one {prefix}_*.model, found {matches}"
    return matches[0]


def test_train_aborts_without_input():
    runner = CliRunner()
    result = runner.invoke(train_cli, [])
    assert result.exit_code != 0
    assert "Aborted" in result.output


def test_train_aborts_without_target(tmp_path, csv_path_clean):
    runner = CliRunner()
    result = runner.invoke(train_cli, ["-i", csv_path_clean])
    assert result.exit_code != 0
    assert "Aborted" in result.output


def test_train_aborts_on_missing_time_column(tmp_path, monkeypatch, csv_path_clean):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli, _train_args(csv_path_clean, extra=["--time-column", "NotARealCol"])
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "aborted" in result.output.lower()


def test_train_expert_ridge_writes_model_with_v03_metadata(
    tmp_path, monkeypatch, csv_path_clean
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(train_cli, _train_args(csv_path_clean))

    assert result.exit_code == 0, result.output
    model_path = _saved_model(tmp_path, "ridge_Price")
    meta = joblib.load(model_path)
    # Read the version from the module rather than hardcoding so future
    # bumps don't break this test the way the v0.3 -> v0.4 bump did.
    from p2predict.trained_model_io import P2PREDICT_VERSION
    assert meta["p2predict_version"] == P2PREDICT_VERSION
    assert meta["target_feature"] == "Price"
    assert set(meta["features"]) == {"Weight", "Region", "Supplier", "Size"}
    assert "log_target" in meta


def test_train_with_outliers_drop_reports_and_proceeds(tmp_path, monkeypatch, synthetic_parts):
    # Inject one extreme outlier into the target.
    df = synthetic_parts.copy()
    df.loc[len(df)] = {
        "Weight": 25.0, "Region": "EU", "Supplier": "A", "Size": "Standard",
        "Price": 99_999.0,
    }
    csv = tmp_path / "with_outlier.csv"
    df.to_csv(csv, index=False)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli, _train_args(csv, extra=["--outliers", "drop"])
    )

    assert result.exit_code == 0, result.output
    assert "Outliers in 'Price'" in result.output
    assert "dropped" in result.output


def test_train_time_aware_excludes_date_from_features(
    tmp_path, monkeypatch, synthetic_parts_with_date
):
    csv = tmp_path / "timed.csv"
    synthetic_parts_with_date.to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli, _train_args(csv, extra=["--time-column", "Date"])
    )
    assert result.exit_code == 0, result.output
    assert "Time-aware" in result.output or "chronological" in result.output.lower()

    model_path = _saved_model(tmp_path, "ridge_Price")
    meta = joblib.load(model_path)
    assert "Date" not in meta["features"]


def test_predict_inline_round_trip(tmp_path, monkeypatch, csv_path_clean):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    train_result = runner.invoke(train_cli, _train_args(csv_path_clean))
    assert train_result.exit_code == 0, train_result.output

    model_path = _saved_model(tmp_path, "ridge_Price")
    predict_result = runner.invoke(
        predict_cli,
        ["-m", str(model_path),
         "-p", "Weight:15,Region:EU,Supplier:A,Size:Standard"],
    )
    assert predict_result.exit_code == 0, predict_result.output
    assert "Prediction" in predict_result.output


def test_predict_batch_writes_predictions_back_to_csv(
    tmp_path, monkeypatch, csv_path_clean, synthetic_parts
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    train_result = runner.invoke(train_cli, _train_args(csv_path_clean))
    assert train_result.exit_code == 0, train_result.output

    model_path = _saved_model(tmp_path, "ridge_Price")

    # Build a batch CSV without the target column.
    batch = synthetic_parts.head(5).drop(columns=["Price"])
    batch_path = tmp_path / "batch.csv"
    batch.to_csv(batch_path, index=False)

    predict_result = runner.invoke(
        predict_cli, ["-m", str(model_path), "-i", str(batch_path)]
    )
    assert predict_result.exit_code == 0, predict_result.output

    written = pd.read_csv(batch_path)
    assert "Price" in written.columns
    assert written["Price"].notna().all()


def _csv_with_ten_features(tmp_path):
    """CSV with 10 numerical features, all carrying signal so none get
    pruned as low-information before auto feature selection runs."""
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({f"f{i}": rng.uniform(0.5, 5.0, n) for i in range(10)})
    coefs = np.linspace(0.3, 1.2, 10)
    df["Price"] = df.values @ coefs + rng.normal(0, 0.1, n)
    p = tmp_path / "ten_features.csv"
    df.to_csv(p, index=False)
    return str(p)


def _parse_json(output):
    start = output.find("{")
    end = output.rfind("}")
    return json.loads(output[start:end + 1])


def test_auto_mode_default_max_features_caps_at_six(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    csv_path = _csv_with_ten_features(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", csv_path, "-t", "Price", "-b", "fast", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert len(doc["features_selected"]) == 6


def test_train_auto_writes_pdf_report_when_requested(tmp_path, monkeypatch, csv_path_clean):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    pdf_path = tmp_path / "report.pdf"
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path_clean), "-t", "Price", "-b", "fast",
         "--report", str(pdf_path), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 5_000  # non-trivial multi-page PDF
    doc = _parse_json(result.output)
    assert doc["report_path"] == str(pdf_path)


def test_auto_mode_respects_max_features_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    csv_path = _csv_with_ten_features(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", csv_path, "-t", "Price", "-b", "fast",
         "--max-features", "10", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert len(doc["features_selected"]) == 10


def test_log_target_on_activates_wrap_on_low_skew_data(
    tmp_path, monkeypatch, csv_path_clean
):
    """`--log-target on` must wrap the regressor even when skew is well below
    the 1.0 auto threshold. This is the BMIC case-study scenario: a
    multiplicative positive-quantity target (price) with a small,
    near-symmetric sample. Auto would skip the wrap; manual override
    forces it on so conformal intervals stay multiplicative."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path_clean), "-t", "Price", "-b", "fast",
         "--log-target", "on", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert doc["log_target"] is True
    assert doc["log_target_decision"] == "manual:on"

    model_path = _saved_model(tmp_path, f"{doc['algorithm_selected']}_Price")
    meta = joblib.load(model_path)
    assert meta["log_target"] is True


def test_log_target_off_disables_wrap_on_high_skew_data(
    tmp_path, monkeypatch, synthetic_parts_skewed
):
    """`--log-target off` must skip the wrap even on data the auto rule
    would have flagged (heavily skewed log-normal target). Keeps the
    flag honest as an override in both directions."""
    csv = tmp_path / "skewed.csv"
    synthetic_parts_skewed.to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv), "-t", "Price", "-b", "fast",
         "--log-target", "off", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert doc["log_target"] is False
    assert doc["log_target_decision"] == "manual:off"

    model_path = _saved_model(tmp_path, f"{doc['algorithm_selected']}_Price")
    meta = joblib.load(model_path)
    assert meta["log_target"] is False


def test_log_target_auto_records_skew_in_decision(
    tmp_path, monkeypatch, csv_path_clean
):
    """The default `auto` mode should record the numeric skew it observed
    so consumers can see *why* the wrap was (or wasn't) applied."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path_clean), "-t", "Price", "-b", "fast", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert doc["log_target_decision"].startswith("auto:skew=")


def _csv_with_feature_nas(tmp_path, seed=0):
    """50 rows: clean target, NAs scattered across feature columns (including
    one column that is NOT selected for training)."""
    rng = np.random.default_rng(seed)
    n = 50
    df = pd.DataFrame({
        "Weight": rng.uniform(1, 50, n),
        "Region": rng.choice(["EU", "CN", "US"], n),
        "Supplier": rng.choice(["A", "B", "C"], n),
        "Size": rng.choice(["Small", "Standard", "Large"], n),
        "Unused": rng.uniform(0, 1, n),  # never selected for training
        "Price": rng.uniform(1, 100, n),
    })
    # NA in a *selected* feature column...
    df.loc[0, "Weight"] = np.nan
    df.loc[1, "Region"] = np.nan
    # ...and NA in a column that is not a training feature at all.
    df.loc[2, "Unused"] = np.nan
    p = tmp_path / "feature_nas.csv"
    df.to_csv(p, index=False)
    return p


def test_train_keeps_rows_with_feature_only_nas(tmp_path, monkeypatch):
    """Rows whose only NA is in a feature column (selected or not) must be
    kept — XGBoost handles NaN natively, ridge/RF impute. The old blanket
    df.dropna() at load discarded these rows (and any with NA in unselected
    columns), which is the data-loss bug this fix targets."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    csv_path = _csv_with_feature_nas(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path), "-t", "Price",
         "-tf", "Weight,Region,Supplier,Size", "-b", "fast", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    # All 50 rows are used: no target NAs, feature NAs are not dropped.
    assert doc["input"]["rows_loaded"] == 50
    assert doc["input"]["rows_dropped_target_na"] == 0
    assert doc["input"]["rows_used"] == 50


def test_train_drops_only_target_na_rows(tmp_path, monkeypatch):
    """NAs in the target column can't supervise training, so those rows (and
    only those) are dropped — feature NAs are retained."""
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "Weight": rng.uniform(1, 50, n),
        "Region": rng.choice(["EU", "CN", "US"], n),
        "Supplier": rng.choice(["A", "B", "C"], n),
        "Size": rng.choice(["Small", "Standard", "Large"], n),
        "Price": rng.uniform(1, 100, n),
    })
    df.loc[[0, 1, 2], "Price"] = np.nan       # 3 target NAs -> dropped
    df.loc[3, "Weight"] = np.nan              # feature NA -> kept
    csv_path = tmp_path / "target_nas.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path), "-t", "Price",
         "-tf", "Weight,Region,Supplier,Size", "-b", "fast", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert doc["input"]["rows_loaded"] == 40
    assert doc["input"]["rows_dropped_target_na"] == 3
    assert doc["input"]["rows_used"] == 37


def test_train_json_stdout_is_pure_json_with_nas(tmp_path, monkeypatch):
    """The NA warning must go to stderr so `--json` stdout parses cleanly:
    the document's first non-whitespace char on stdout is '{'."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    csv_path = _csv_with_feature_nas(tmp_path)
    # Click 8.2+ CliRunner separates stdout/stderr by default (the old
    # mix_stderr kwarg is gone); result.stdout is stdout only.
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path), "-t", "Price",
         "-tf", "Weight,Region,Supplier,Size", "-b", "fast", "--json"],
    )
    assert result.exit_code == 0, result.output
    # Pure JSON: stdout parses as a single document with no leading warning.
    doc = json.loads(result.stdout.strip())
    assert doc["command"] == "train"


def test_train_auto_mode_handles_feature_nas_across_all_algorithms(
    tmp_path, monkeypatch
):
    """Auto mode compares ridge/random_forest/xgboost on the SAME NaN-bearing
    data. All three preprocessors must cope (impute or pass-through) so the
    comparison completes without a NaN crash."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    csv_path = _csv_with_feature_nas(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", str(csv_path), "-t", "Price",
         "-tf", "Weight,Region,Supplier,Size", "-b", "fast", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert set(doc["cv_scores"].keys()) == {"ridge", "random_forest", "xgboost"}


def _heavy_tailed_parts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(1, 10, n)
    weight[: n // 10] = rng.uniform(200, 2000, n // 10)
    region = rng.choice(["EU", "CN"], n)
    price = 5 + 2 * np.minimum(weight, 10) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"Weight": weight, "Region": region, "Price": price})


def test_feature_winsorize_caps_are_applied_at_prediction(tmp_path, monkeypatch):
    # Training capped Weight but the saved model used to predict on raw values,
    # so a linear model extrapolated without limit past the training caps.
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "heavy.csv"
    _heavy_tailed_parts().to_csv(csv, index=False)
    result = CliRunner().invoke(train_cli, [
        "-i", str(csv), "-t", "Price", "-x", "-a", "ridge", "-tf", "Weight,Region",
        "--feature-outliers", "winsorize", "--log-target", "off", "--json",
    ])
    assert result.exit_code == 0, result.output

    model = joblib.load(_saved_model(tmp_path, "ridge"))["model"]
    far = pd.DataFrame({"Weight": [5_000.0, 500_000.0], "Region": ["EU", "EU"]})
    preds = model.predict(far)
    assert preds[0] == preds[1], "prediction kept moving past the training cap"
    assert preds[1] < 100, f"linear model extrapolated past the cap: {preds[1]}"


def test_feature_winsorize_caps_come_from_training_rows_only():
    from p2predict.preprocessing import FeatureClipper

    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
    clipper = FeatureClipper().fit(train)
    out = clipper.transform(pd.DataFrame({"x": [-100.0, 3.0, 100.0, np.nan]}))
    assert out[0, 0] == clipper.lower_[0] and out[2, 0] == clipper.upper_[0]
    assert out[1, 0] == 3.0 and np.isnan(out[3, 0])
    assert list(clipper.get_feature_names_out(["x"])) == ["x"]


def test_time_column_train_warns_about_interval_coverage(
    tmp_path, monkeypatch, synthetic_parts_with_date, csv_path_clean
):
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "timed.csv"
    synthetic_parts_with_date.to_csv(csv, index=False)
    timed = CliRunner().invoke(
        train_cli, _train_args(csv, extra=["--time-column", "Date", "--json"])
    )
    assert timed.exit_code == 0, timed.output
    assert any("Time-ordered" in w for w in json.loads(timed.stdout)["warnings"])

    plain = CliRunner().invoke(train_cli, _train_args(csv_path_clean, extra=["--json"]))
    assert plain.exit_code == 0, plain.output
    assert json.loads(plain.stdout)["warnings"] == []


def test_predict_cli_points_train_subcommand_to_p2predict_train():
    result = CliRunner().invoke(predict_cli, ["train", "-i", "missing.csv"])
    assert result.exit_code == 2
    assert "p2predict-train" in result.output


def _csv_with_part_numbers(tmp_path, synthetic_parts):
    """Mirrors examples/example.csv: a unique part-number column and a column
    with one value in every row, next to fewer real specs than --max-features."""
    df = synthetic_parts.copy()
    df.insert(0, "CPN", [f"CP{i}-{i * 7919}" for i in range(len(df))])
    df["Supplier"] = "supplier A"
    p = tmp_path / "with_cpn.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_auto_mode_headless_never_selects_id_like_column(
    tmp_path, monkeypatch, synthetic_parts
):
    """Regression: headless auto-mode used to train on the part number because
    only constant columns were dropped and the feature cap never bit. The saved
    model then demanded a CPN at predict time and crashed."""
    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_part_numbers(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli, ["-i", csv_path, "-t", "Price", "-b", "fast", "--json"]
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert "CPN" not in doc["features_selected"]
    assert "Supplier" not in doc["features_selected"]
    assert any("CPN" in w for w in doc["warnings"])

    predict_result = runner.invoke(
        predict_cli,
        ["-m", doc["model_path"], "-p", "Weight:25,Region:EU,Size:Standard", "--json"],
    )
    assert predict_result.exit_code == 0, predict_result.output


def test_explicit_features_keep_id_like_column(tmp_path, monkeypatch, synthetic_parts):
    """-tf is the user's decision: an ID-like column they name is used."""
    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_part_numbers(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", csv_path, "-t", "Price", "-b", "fast",
         "-tf", "Weight,Region,CPN", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert "CPN" in _parse_json(result.output)["features_selected"]


def test_explicit_constant_feature_is_left_out_with_a_note(
    tmp_path, monkeypatch, synthetic_parts
):
    """A requested column with one value everywhere used to abort with the
    false message 'not in CSV'; now it is left out and training proceeds."""
    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_part_numbers(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", csv_path, "-t", "Price", "-b", "fast",
         "-tf", "Weight,Region,Supplier,Size", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = _parse_json(result.output)
    assert doc["features_selected"] == ["Weight", "Region", "Size"]
    assert any("Supplier" in w and "same value" in w for w in doc["warnings"])


def _csv_with_leaky_column(tmp_path, synthetic_parts):
    df = synthetic_parts.copy()
    df["Price_at_1k"] = df["Price"] * 0.5 + np.random.default_rng(0).normal(0, 0.005, len(df))
    p = tmp_path / "leaky.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_explicit_leaky_feature_aborts_like_mcp(tmp_path, monkeypatch, synthetic_parts):
    """-tf with a leakage column stops, as MCP `train` does, instead of quietly
    training a model that looks near-perfect and is useless on real parts."""
    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_leaky_column(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", csv_path, "-t", "Price", "-b", "fast",
         "-tf", "Weight,Price_at_1k", "--json"],
    )
    assert result.exit_code == 1
    doc = _parse_json(result.output)
    assert doc["error"]["code"] == "target_leakage"
    assert "Price_at_1k" in doc["error"]["message"]
    assert not (tmp_path / "models").exists()


def test_allow_leaky_features_overrides_the_guard(tmp_path, monkeypatch, synthetic_parts):
    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_leaky_column(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli,
        ["-i", csv_path, "-t", "Price", "-b", "fast",
         "-tf", "Weight,Price_at_1k", "--allow-leaky-features", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert "Price_at_1k" in _parse_json(result.output)["features_selected"]


def test_auto_mode_reports_excluded_features_in_json(tmp_path, monkeypatch, synthetic_parts):
    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_part_numbers(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli, ["-i", csv_path, "-t", "Price", "-b", "fast", "--json"]
    )
    assert result.exit_code == 0, result.output
    kinds = {e["column"]: e["kind"] for e in _parse_json(result.output)["excluded_features"]}
    assert kinds == {"CPN": "id_like", "Supplier": "constant"}


class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


def test_interactive_mode_pre_ticks_the_same_exclusions(
    tmp_path, monkeypatch, synthetic_parts
):
    """Interactive mode offers exactly what auto-mode would leave out,
    pre-ticked, so accepting the defaults gives the headless result."""
    import p2predict.cli.train as train_mod

    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_part_numbers(tmp_path, synthetic_parts)
    offered = {}

    def fake_checkbox(message, choices):
        offered.update({c.value: c.checked for c in choices})
        return _Answer([c.value for c in choices if c.checked])

    monkeypatch.setattr(train_mod.questionary, "checkbox", fake_checkbox)
    monkeypatch.setattr(train_mod.questionary, "confirm", lambda *a, **k: _Answer(False))
    runner = CliRunner()
    result = runner.invoke(
        train_cli, ["-i", csv_path, "-t", "Price", "-b", "fast", "-c"]
    )
    assert result.exit_code == 0, result.output
    assert offered == {"CPN": True, "Supplier": True}
    selected_line = next(
        line for line in result.output.splitlines()
        if line.startswith("Auto-selected features for training:")
    )
    assert "CPN" not in selected_line and "Supplier" not in selected_line


def test_cli_and_mcp_auto_pick_the_same_specs(tmp_path, monkeypatch, synthetic_parts):
    """The harmonisation contract: one CSV, same specs on every surface."""
    import asyncio

    from p2predict.mcp import server as mcp_server
    from p2predict.mcp.registry import ModelRegistry

    monkeypatch.chdir(tmp_path)
    csv_path = _csv_with_part_numbers(tmp_path, synthetic_parts)
    runner = CliRunner()
    result = runner.invoke(
        train_cli, ["-i", csv_path, "-t", "Price", "-b", "fast", "--json"]
    )
    assert result.exit_code == 0, result.output
    cli_features = _parse_json(result.output)["features_selected"]

    (tmp_path / "mcp_models").mkdir()
    monkeypatch.setattr(mcp_server, "_registry", ModelRegistry(tmp_path / "mcp_models"))
    plan = json.loads(asyncio.run(
        mcp_server.propose_training_plan(csv_path=csv_path, target="Price")
    ))
    assert sorted(plan["i_will_use_these_specs"]) == sorted(cli_features)
