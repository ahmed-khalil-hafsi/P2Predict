import pandas as pd
import pytest

from p2predict.feature_selection import (
    find_auto_exclusions,
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


def test_find_auto_exclusions_classifies_id_constant_and_leakage(synthetic_parts):
    df = synthetic_parts.copy()
    df["CPN"] = [f"CP{i}-{i * 7919}" for i in range(len(df))]
    df["Plant"] = "SG01"
    df["Price_at_1k"] = df["Price"] * 0.5
    kinds = {e["column"]: e["kind"] for e in find_auto_exclusions(df, "Price")}
    assert kinds == {"Price_at_1k": "leakage", "CPN": "id_like", "Plant": "constant"}


def test_find_auto_exclusions_keeps_wide_ranging_numeric_specs():
    # A numeric spec spanning orders of magnitude is flagged as high-variation
    # but is a real spec, not an ID — it must stay selectable.
    df = pd.DataFrame({
        "Capacitance_uF": [0.1, 1, 10, 100, 1000, 0.47, 4.7, 47, 470, 2200],
        "Price": [0.02, 0.03, 0.05, 0.4, 0.9, 0.02, 0.04, 0.2, 1.1, 1.7],
    })
    assert "Capacitance_uF" in find_high_variation_features(df)
    assert find_auto_exclusions(df, "Price") == []


def test_find_date_like_features_flags_common_date_formats():
    from p2predict.feature_selection import find_date_like_features

    n = 30
    df = pd.DataFrame({
        "iso": [f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        "german": [f"{(i % 28) + 1:02d}.{(i % 12) + 1:02d}.2024" for i in range(n)],
        "iso_time": [f"2024-03-{(i % 28) + 1:02d} 10:{i % 60:02d}" for i in range(n)],
    })
    assert sorted(find_date_like_features(df)) == ["german", "iso", "iso_time"]


def test_find_date_like_features_ignores_spec_text_that_looks_numeric():
    from p2predict.feature_selection import find_date_like_features

    n = 30
    df = pd.DataFrame({
        "thread": [["1/4", "3/8", "1/2"][i % 3] for i in range(n)],
        "ip": [["IP54", "IP67"][i % 2] for i in range(n)],
        "metric": [["M8x1.25", "M10x1.5"][i % 2] for i in range(n)],
        "one_date": ["2024-01-01"] * n,  # constant: reported as constant, not date
    })
    assert find_date_like_features(df) == []


def test_find_auto_exclusions_leaves_out_a_quote_date(synthetic_parts):
    df = synthetic_parts.copy()
    df["Quote_Date"] = pd.date_range("2023-01-01", periods=len(df)).strftime("%Y-%m-%d")
    kinds = {e["column"]: e["kind"] for e in find_auto_exclusions(df, "Price")}
    assert kinds.get("Quote_Date") == "date"


def test_requested_date_column_is_kept_with_a_note(synthetic_parts):
    from p2predict.feature_selection import choose_training_features

    df = synthetic_parts.copy()
    df["Quote_Date"] = pd.date_range("2023-01-01", periods=len(df)).strftime("%Y-%m-%d")
    choice = choose_training_features(df, "Price", ["Weight", "Quote_Date"])
    assert choice.selected == ["Weight", "Quote_Date"]
    assert any("Quote_Date" in n and "date" in n for n in choice.notes)
