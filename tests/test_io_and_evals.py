import numpy as np
import pandas as pd

from modules.model_evals import evaluate_model
from modules.prepare_data import prepare_data
from modules.trained_model_io import (
    LoadModel,
    P2PREDICT_VERSION,
    SaveModel,
    Serialize_Trained_Model,
)
from modules.training import start_training


def test_serialize_trained_model_includes_log_target_field():
    meta = Serialize_Trained_Model(
        "ridge", ["Weight"], "Price", model=None, r2=0.5, log_target=True
    )
    assert meta["log_target"] is True
    assert meta["p2predict_version"] == P2PREDICT_VERSION
    assert meta["features"] == ["Weight"]


def test_save_load_round_trip(tmp_path, synthetic_parts):
    features = ["Weight", "Region", "Supplier", "Size"]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        synthetic_parts, features, "Price"
    )
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )

    meta = Serialize_Trained_Model(
        "random_forest", features, "Price", model, r2=0.9, log_target=log_target
    )
    path = tmp_path / "round_trip.model"
    SaveModel(meta, str(path))
    reloaded = LoadModel(str(path))

    assert reloaded["features"] == features
    assert reloaded["target_feature"] == "Price"
    assert reloaded["log_target"] == log_target
    # The pipeline still predicts after load.
    preds = reloaded["model"].predict(X_test)
    assert preds.shape == y_test.shape


def test_evaluate_model_returns_four_values(synthetic_parts):
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        synthetic_parts, ["Weight", "Region", "Supplier", "Size"], "Price"
    )
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    result = evaluate_model(X_test, y_test, model)
    assert len(result) == 4
    mae, r2, p_value, rmse = result
    assert mae >= 0
    assert -1 <= r2 <= 1
    assert 0 <= p_value <= 1
    assert rmse >= 0
    # rmse is always >= mae (Jensen's inequality on a non-negative variable).
    assert rmse >= mae - 1e-9


def test_evaluate_model_handles_perfect_predictions():
    # When predictions exactly equal targets, residuals are zero — the t-test
    # branch must not raise.
    class IdentityModel:
        def predict(self, X):
            return np.asarray(X["y"])

    X = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    y = X["y"]
    mae, r2, p_value, rmse = evaluate_model(X, y, IdentityModel())
    assert mae == 0
    assert r2 == 1
    assert rmse == 0
    assert 0 <= p_value <= 1
