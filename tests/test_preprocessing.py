import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from p2predict.preprocessing import build_preprocessor, model_family_for


def _target_encoder(cat_transformer):
    """The TargetEncoder inside the tree categorical branch (impute -> target)."""
    return cat_transformer.named_steps["target"]


def test_xgboost_target_encodes_categoricals_and_passes_numeric_nans():
    # XGBoost handles NaN natively, so numerics pass through untouched;
    # categoricals are target-encoded (price-ordered codes, not arbitrary
    # alphabetical ordinals).
    pp = build_preprocessor(["Weight"], ["Region"], model_family="xgboost")
    assert isinstance(pp, ColumnTransformer)
    transformer_map = {name: t for name, t, _ in pp.transformers}
    assert transformer_map["num"] == "passthrough"
    assert isinstance(_target_encoder(transformer_map["cat"]), TargetEncoder)


def test_random_forest_imputes_numeric_and_target_encodes():
    # RandomForest rejects NaN, so numerics get a median imputer; categoricals
    # are imputed (most-frequent) then target-encoded.
    pp = build_preprocessor(["Weight"], ["Region"], model_family="random_forest")
    transformer_map = {name: t for name, t, _ in pp.transformers}
    num = transformer_map["num"]
    assert isinstance(num, SimpleImputer)
    assert num.strategy == "median"
    cat = transformer_map["cat"]
    assert isinstance(cat, Pipeline)
    assert "impute" in cat.named_steps
    assert isinstance(_target_encoder(cat), TargetEncoder)


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


def test_target_encoder_orders_categories_by_target(synthetic_parts):
    # The whole point of target encoding: the encoded value reflects the
    # category's mean target, so a tree can split premium from commodity in
    # one cut. Build a frame where one region is systematically pricier and
    # assert its encoded value comes out highest.
    df = pd.DataFrame({
        "Region": ["EU"] * 30 + ["CN"] * 30 + ["US"] * 30,
        "Price": [100.0] * 30 + [10.0] * 30 + [55.0] * 30,
    })
    pp = build_preprocessor([], ["Region"], model_family="xgboost")
    enc = pp.fit_transform(df[["Region"]], df["Price"])
    codes = {r: enc[df["Region"].to_numpy() == r, 0][0]
             for r in ("EU", "CN", "US")}
    assert codes["EU"] > codes["US"] > codes["CN"]


def test_target_encoder_handles_unseen_categories(synthetic_parts):
    pp = build_preprocessor(["Weight"], ["Region"], model_family="xgboost")
    pp.fit(synthetic_parts[["Weight", "Region"]], synthetic_parts["Price"])
    unseen = synthetic_parts.head(1).copy()
    unseen["Region"] = "MARS"
    transformed = pp.transform(unseen[["Weight", "Region"]])
    # Unseen category -> encoded to the global target mean, finite, no crash.
    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed[0, 1])
    assert transformed[0, 1] == pytest.approx(synthetic_parts["Price"].mean(), rel=0.5)


def test_random_forest_preprocessor_imputes_feature_nans():
    # End-to-end: a NaN in a numeric and a categorical feature column must be
    # filled (not crash, not propagate) for the imputing families.
    pp = build_preprocessor(["Weight"], ["Region"], model_family="random_forest")
    train = pd.DataFrame({
        "Weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Region": ["EU", "CN", "EU", "US", "CN", "EU"],
    })
    y = pd.Series([10.0, 20.0, 12.0, 30.0, 22.0, 11.0])
    pp.fit(train, y)
    with_nan = pd.DataFrame({"Weight": [np.nan], "Region": [np.nan]})
    out = pp.transform(with_nan)
    assert np.isfinite(out).all()


def test_xgboost_preprocessor_preserves_numeric_nans():
    # XGBoost relies on numeric NaN reaching the model, so the preprocessor
    # must keep it intact; the categorical NaN is absorbed by the encoder.
    pp = build_preprocessor(["Weight"], ["Region"], model_family="xgboost")
    train = pd.DataFrame({
        "Weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Region": ["EU", "CN", "EU", "US", "CN", "EU"],
    })
    y = pd.Series([10.0, 20.0, 12.0, 30.0, 22.0, 11.0])
    pp.fit(train, y)
    with_nan = pd.DataFrame({"Weight": [np.nan], "Region": ["EU"]})
    out = pp.transform(with_nan)
    assert np.isnan(out[0, 0])  # numeric NaN preserved
    assert np.isfinite(out[0, 1])  # categorical encoded to a real value
