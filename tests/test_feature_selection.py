import pandas as pd
import pytest

from modules.p2predict_feature_selection import (
    find_high_variation_features,
    find_no_variation_features,
    get_most_predictable_features,
)


def test_find_no_variation_flags_constant_columns():
    df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]})
    assert find_no_variation_features(df) == ["a"]


def test_find_high_variation_handles_zero_mean_without_crashing():
    # Symmetric around zero → mean ≈ 0. Earlier versions divided by mean and
    # produced nonsense; we expect a stable result and no exception.
    df = pd.DataFrame({"a": [-5, -1, 1, 5]})
    result = find_high_variation_features(df)
    assert isinstance(result, list)


def test_find_high_variation_flags_unique_id_column():
    df = pd.DataFrame({
        "id": [f"P{i}" for i in range(20)],
        "size": ["S"] * 20,
    })
    assert "id" in find_high_variation_features(df)


def test_get_most_predictable_features_ranks_signal_first(synthetic_parts):
    ranked = get_most_predictable_features(synthetic_parts, "Price")
    assert ranked.iloc[0]["Feature"] == "Weight"
    assert ranked["Importance (%)"].sum() == pytest.approx(100, abs=0.5)


def test_get_most_predictable_features_headers_only(synthetic_parts):
    ranked = get_most_predictable_features(
        synthetic_parts, "Price", output_only_headers=True
    )
    assert "Weight" in ranked.tolist()
    assert "Price" not in ranked.tolist()
