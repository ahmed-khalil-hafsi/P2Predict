import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from modules.preprocessing import build_preprocessor, model_family_for


def test_tree_family_uses_ordinal_encoder():
    pp = build_preprocessor(["Weight"], ["Region"], model_family="random_forest")
    assert isinstance(pp, ColumnTransformer)
    transformer_map = {name: t for name, t, _ in pp.transformers}
    assert transformer_map["num"] == "passthrough"
    assert isinstance(transformer_map["cat"], OrdinalEncoder)


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
    pp = build_preprocessor(["Weight"], ["Region"], model_family="random_forest")
    pp.fit(synthetic_parts[["Weight", "Region"]])
    # Apply to a row with a region the encoder has never seen.
    unseen = synthetic_parts.head(1).copy()
    unseen["Region"] = "MARS"
    transformed = pp.transform(unseen[["Weight", "Region"]])
    # unknown_value=-1 means the unseen region is encoded as -1, not a crash.
    assert transformed.shape == (1, 2)
    assert transformed[0, 1] == -1
