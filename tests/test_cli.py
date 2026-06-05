"""End-to-end CLI tests using Click's CliRunner.

These complement the per-module unit tests by exercising the actual entry
points users invoke (`p2predict_train.py` and `p2predict.py`) — wiring of
flags, flow control, and the save/load contract between the two scripts.
"""

from pathlib import Path

import joblib
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

    import pandas as pd
    written = pd.read_csv(batch_path)
    assert "Price" in written.columns
    assert written["Price"].notna().all()
