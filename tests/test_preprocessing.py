import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from p2predict.preprocessing import build_preprocessor, model_family_for


def _cat_encoder(transformer):
    """Return the OrdinalEncoder whether or not it's wrapped in an imputing
    Pipeline."""
    if isinstance(transformer, Pipeline):
        return transformer.named_steps.get("ordinal")
    return transformer


def test_xgboost_passes_nans_through_for_native_handling():
    # XGBoost handles NaN natively, so its preprocessor must NOT impute:
    # numerics passthrough, categoricals a bare OrdinalEncoder.
    pp = build_preprocessor(["Weight"], ["Region"], model_family="xgboost")
    assert isinstance(pp, ColumnTransformer)
    transformer_map = {name: t for name, t, _ in pp.transformers}
    assert transformer_map["num"] == "passthrough"
    assert isinstance(transformer_map["cat"], OrdinalEncoder)


def test_random_forest_imputes_nans():
    # RandomForest rejects NaN, so its preprocessor imputes both numerics and
    # categoricals, then ordinal-encodes the categoricals.
    pp = build_preprocessor(["Weight"], ["Region"], model_family="random_forest")
    transformer_map = {name: t for name, t, _ in pp.transformers}
    num = transformer_map["num"]
    # The numeric branch is a bare median SimpleImputer (no scaling needed
    # for trees); the categorical branch wraps its imputer in a Pipeline.
    assert isinstance(num, SimpleImputer)
    assert num.strategy == "median"
    cat = transformer_map["cat"]
    assert isinstance(cat, Pipeline)
    assert "impute" in cat.named_steps
    assert isinstance(_cat_encoder(cat), OrdinalEncoder)


def test_linear_family_uses_onehot_and_scaler():
    pp = build_preprocessor(["Weight"], ["Region"], model_family="ridge")
    transformer_map = {name: t for name, t, _ in pp.transformers}
    assert transformer_map["num"] != "passthrough"
    # The cat transformer is a Pipeline containing a OneHotEncoder.
    cat = transformer_map["cat"]
    assert isinstance(cat.named_steps["onehot"], OneHotEncoder)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        build_preprocessor(["Weight"], ["Region"], model_family="quantum")


def test_model_family_for_known_algorithms():
    assert model_family_for("ridge") == "ridge"
    assert model_family_for("random_forest") == "random_forest"
    assert model_family_for("xgboost") == "xgboost"


def test_model_family_for_unknown_raises():
    with pytest.raises(ValueError):
        model_family_for("svm")


def test_ordinal_encoder_handles_unseen_categories(synthetic_parts):
    pp = build_preprocessor(["Weight"], ["Region"], model_family="xgboost")
    pp.fit(synthetic_parts[["Weight", "Region"]])
    # Apply to a row with a region the encoder has never seen.
    unseen = synthetic_parts.head(1).copy()
    unseen["Region"] = "MARS"
    transformed = pp.transform(unseen[["Weight", "Region"]])
    # unknown_value=-1 means the unseen region is encoded as -1, not a crash.
    assert transformed.shape == (1, 2)
    assert transformed[0, 1] == -1


def test_random_forest_preprocessor_imputes_feature_nans():
    # End-to-end: a NaN in a numeric and a categorical feature column must be
    # filled (not crash, not propagate) for the imputing families.
    pp = build_preprocessor(["Weight"], ["Region"], model_family="random_forest")
    train = pd.DataFrame({
        "Weight": [1.0, 2.0, 3.0, 4.0],
        "Region": ["EU", "CN", "EU", "US"],
    })
    pp.fit(train)
    with_nan = pd.DataFrame({"Weight": [np.nan], "Region": [np.nan]})
    out = pp.transform(with_nan)
    assert np.isfinite(out).all()


def test_xgboost_preprocessor_preserves_feature_nans():
    # XGBoost relies on NaN reaching the model, so the preprocessor must keep
    # the numeric NaN intact (categorical NaN encodes via OrdinalEncoder).
    pp = build_preprocessor(["Weight"], ["Region"], model_family="xgboost")
    train = pd.DataFrame({
        "Weight": [1.0, 2.0, 3.0, 4.0],
        "Region": ["EU", "CN", "EU", "US"],
    })
    pp.fit(train)
    with_nan = pd.DataFrame({"Weight": [np.nan], "Region": ["EU"]})
    out = pp.transform(with_nan)
    assert np.isnan(out[0, 0])
