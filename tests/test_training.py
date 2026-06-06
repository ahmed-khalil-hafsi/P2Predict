import numpy as np
import pandas as pd

from p2predict.prepare_data import prepare_data
from p2predict.training import (
    ALGORITHMS,
    auto_train,
    build_pipeline,
    extract_feature_importances,
    should_log_target,
    start_training,
)


def _split(df, target="Price", time_column=None):
    features = [c for c in df.columns if c not in {target, time_column}]
    return prepare_data(df, features, target, time_column=time_column)


def test_should_log_target_detects_skew(synthetic_parts_skewed):
    assert should_log_target(synthetic_parts_skewed["Price"]) is True


def test_should_log_target_negative_values_returns_false():
    assert should_log_target([-1, 1, 2, 3]) is False


def test_should_log_target_uniform_returns_false():
    # No skew → no transform.
    assert should_log_target(np.linspace(1, 100, 200)) is False


def test_should_log_target_empty_returns_false():
    assert should_log_target([]) is False


def test_start_training_ridge_fits_and_predicts(synthetic_parts):
    X_train, X_test, y_train, y_test, num, cat = _split(synthetic_parts)
    model, importances, _ = start_training(
        X_train, y_train, num, cat, algorithm="ridge", tune=False
    )
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    assert importances and importances[0][1] >= importances[-1][1]


def test_start_training_random_forest_fits_and_predicts(synthetic_parts):
    X_train, X_test, y_train, y_test, num, cat = _split(synthetic_parts)
    model, importances, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    # Weight is the strongest signal in synthetic data — it should rank top.
    assert importances[0][0] == "Weight"


def test_start_training_log_target_when_skewed(synthetic_parts_skewed):
    X_train, _, y_train, _, num, cat = _split(synthetic_parts_skewed)
    _, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    assert log_target is True


def test_auto_train_picks_best_and_returns_scores(tiny_parts):
    X_train, _, y_train, _, num, cat = _split(tiny_parts)
    model, algo, scores, log_target = auto_train(
        X_train, y_train, num, cat, budget="fast"
    )
    assert algo in ALGORITHMS
    assert set(scores.keys()) == set(ALGORITHMS)
    assert model is not None
    assert isinstance(log_target, bool)


def test_auto_train_time_aware_runs(synthetic_parts_with_date):
    X_train, X_test, y_train, y_test, num, cat = _split(
        synthetic_parts_with_date, time_column="Date"
    )
    # Time column must be excluded from the feature set used for training.
    assert "Date" not in X_train.columns
    # Chronological split → test indices come after train indices.
    assert X_test.index.min() > X_train.index.max()

    model, algo, scores, _ = auto_train(
        X_train, y_train, num, cat, budget="fast", time_aware=True
    )
    assert algo in ALGORITHMS


def test_extract_feature_importances_groups_high_cardinality_correctly(synthetic_parts):
    # 'Weight' and 'weight_extra' both contain the substring 'Weight'/'weight' —
    # the importance grouper must not collapse them.
    df = synthetic_parts.copy()
    df["weight_extra"] = df["Weight"] * 0.1
    X_train, _, y_train, _, num, cat = _split(df)
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    importances = extract_feature_importances(model, X_train)
    names = [name for name, _ in importances]
    assert "Weight" in names
    assert "weight_extra" in names
    assert len(names) == len(set(names))


def test_pipeline_handles_unseen_categories_at_predict_time(synthetic_parts):
    X_train, _, y_train, _, num, cat = _split(synthetic_parts)
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    new_row = pd.DataFrame([{
        "Weight": 10.0,
        "Region": "MARS",      # unseen
        "Supplier": "Z",       # unseen
        "Size": "Standard",
    }])
    pred = model.predict(new_row)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_build_pipeline_log_target_wrap_inverts_correctly(synthetic_parts_skewed):
    X_train, _, y_train, _, num, cat = _split(synthetic_parts_skewed)
    model = build_pipeline("random_forest", num, cat, log_target=True)
    model.fit(X_train, y_train)
    # Predictions should be in original price scale, not log scale.
    preds = model.predict(X_train.head(5))
    assert (preds > 0).all()
    assert preds.max() > 1  # sanity: skewed price ** 2 means values aren't all tiny
