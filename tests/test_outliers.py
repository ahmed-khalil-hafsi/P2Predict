import numpy as np
import pandas as pd
import pytest

from modules.outliers import (
    POLICIES,
    apply_feature_outlier_policy,
    apply_outlier_policy,
    detect_outliers,
)


def test_detect_outliers_flags_extreme_values():
    values = list(range(1, 21)) + [10_000]  # clear outlier at the end
    mask, lower, upper = detect_outliers(values)
    assert mask.iloc[-1]
    assert mask.iloc[:-1].sum() == 0
    assert upper < 10_000


def test_detect_outliers_handles_degenerate_distribution():
    mask, _, _ = detect_outliers([5, 5, 5, 5])
    assert mask.sum() == 0


def test_detect_outliers_handles_empty_series():
    mask, lower, upper = detect_outliers([])
    assert mask.empty
    assert np.isnan(lower) and np.isnan(upper)


def test_drop_policy_removes_outliers():
    df = pd.DataFrame({"price": list(range(1, 21)) + [10_000]})
    out, summary = apply_outlier_policy(df, "price", policy="drop")
    assert summary["applied"] == "drop"
    assert summary["n_outliers"] == 1
    assert len(out) == len(df) - 1
    assert 10_000 not in out["price"].values


def test_winsorize_policy_clips_outliers():
    df = pd.DataFrame({"price": list(range(1, 21)) + [10_000]})
    out, summary = apply_outlier_policy(df, "price", policy="winsorize")
    assert summary["applied"] == "winsorize"
    assert len(out) == len(df)
    assert out["price"].max() <= summary["upper"]


def test_warn_policy_does_not_modify_data():
    df = pd.DataFrame({"price": list(range(1, 21)) + [10_000]})
    out, summary = apply_outlier_policy(df, "price", policy="warn")
    assert summary["applied"] == "none"
    pd.testing.assert_frame_equal(out, df)


def test_unknown_policy_raises():
    df = pd.DataFrame({"price": [1, 2, 3]})
    with pytest.raises(ValueError):
        apply_outlier_policy(df, "price", policy="explode")


def test_policies_constant_lists_canonical_set():
    assert set(POLICIES) == {"keep", "warn", "drop", "winsorize"}


# ---------------------------------------------------------------------------
# Feature-side outlier policy (v0.7).
# ---------------------------------------------------------------------------


def _typical_features_with_outliers():
    """Twenty good rows plus one row with an outlier in Weight only and
    one row with outliers in both Weight and Length. Region is categorical
    and should be ignored entirely."""
    df = pd.DataFrame({
        "Weight": list(range(1, 21)) + [10_000, 9_000],
        "Length": [10.0] * 20 + [12.0, 50_000.0],
        "Region": ["EU"] * 22,
    })
    return df


def test_feature_outliers_drop_removes_rows_with_any_column_outlier():
    """The v0.7 drop policy is row-level: a row with an outlier in any
    feature column gets removed. Verify both kinds of bad rows go."""
    df = _typical_features_with_outliers()
    out, summary = apply_feature_outlier_policy(
        df, ["Weight", "Length", "Region"], policy="drop"
    )
    assert summary["applied"] == "drop"
    # Both extreme rows removed; the 20 clean rows survive.
    assert len(out) == 20
    assert summary["n_outliers_total"] == 2
    # Region is categorical and gets ignored — per-column entry should not
    # appear for it.
    assert "Region" not in summary["per_column"]


def test_feature_outliers_winsorize_caps_per_column_without_dropping_rows():
    """Winsorize is per-column and preserves row count. Each column gets
    capped at its own IQR bounds independently."""
    df = _typical_features_with_outliers()
    out, summary = apply_feature_outlier_policy(
        df, ["Weight", "Length"], policy="winsorize"
    )
    assert summary["applied"] == "winsorize"
    assert len(out) == len(df)
    # Both outlier values are capped at the upper IQR bound.
    weight_upper = summary["per_column"]["Weight"]["upper"]
    length_upper = summary["per_column"]["Length"]["upper"]
    assert out["Weight"].max() == pytest.approx(weight_upper)
    assert out["Length"].max() == pytest.approx(length_upper)


def test_feature_outliers_warn_does_not_change_data():
    df = _typical_features_with_outliers()
    out, summary = apply_feature_outlier_policy(
        df, ["Weight", "Length"], policy="warn"
    )
    assert summary["applied"] == "none"
    pd.testing.assert_frame_equal(out, df)
    # But the per-column counts are still reported.
    assert summary["per_column"]["Weight"]["n_outliers"] >= 1
    assert summary["per_column"]["Length"]["n_outliers"] >= 1


def test_feature_outliers_keep_is_silent_like_warn_but_reports_no_action():
    df = _typical_features_with_outliers()
    out, summary = apply_feature_outlier_policy(
        df, ["Weight", "Length"], policy="keep"
    )
    assert summary["applied"] == "none"
    pd.testing.assert_frame_equal(out, df)


def test_feature_outliers_categorical_columns_silently_ignored():
    """Passing a categorical column should not produce a per-column entry
    and should not affect drop/winsorize behaviour."""
    df = pd.DataFrame({
        "Weight": list(range(1, 21)) + [10_000],
        "Region": ["EU"] * 21,
    })
    out, summary = apply_feature_outlier_policy(
        df, ["Weight", "Region"], policy="drop"
    )
    assert "Region" not in summary["per_column"]
    assert "Weight" in summary["per_column"]
    assert len(out) == 20  # one row dropped on Weight


def test_feature_outliers_no_outliers_means_no_change():
    """Clean numeric data should pass through untouched, summary should
    report zeros, and the policy should be a no-op."""
    df = pd.DataFrame({"Weight": list(range(1, 21))})
    out, summary = apply_feature_outlier_policy(df, ["Weight"], policy="drop")
    assert summary["applied"] == "none"
    assert summary["n_outliers_total"] == 0
    pd.testing.assert_frame_equal(out, df)


def test_feature_outliers_drop_with_no_numeric_features_is_a_noop():
    """If the user passes only categorical columns, there's nothing to
    detect and the data passes through untouched."""
    df = pd.DataFrame({"Region": ["EU"] * 10, "Size": ["S"] * 10})
    out, summary = apply_feature_outlier_policy(
        df, ["Region", "Size"], policy="drop"
    )
    assert summary["per_column"] == {}
    assert summary["n_outliers_total"] == 0
    pd.testing.assert_frame_equal(out, df)


def test_feature_outliers_unknown_policy_raises():
    df = pd.DataFrame({"Weight": [1, 2, 3]})
    with pytest.raises(ValueError):
        apply_feature_outlier_policy(df, ["Weight"], policy="implode")


def test_feature_outliers_per_column_counts_reflect_overlap():
    """When a row has outliers in multiple columns, each column gets
    credited individually in per_column counts, but n_outliers_total
    counts each *row* once."""
    df = pd.DataFrame({
        "Weight": list(range(1, 21)) + [10_000],
        "Length": [10.0] * 20 + [50_000.0],   # same row is extreme in both
    })
    out, summary = apply_feature_outlier_policy(
        df, ["Weight", "Length"], policy="warn"
    )
    assert summary["per_column"]["Weight"]["n_outliers"] == 1
    assert summary["per_column"]["Length"]["n_outliers"] == 1
    # The bad row is the *same* row in both columns, so the rowwise total is 1.
    assert summary["n_outliers_total"] == 1


def test_feature_outliers_drop_handles_unaligned_indices():
    """detect_outliers' mask uses the column's index; the function must
    reindex against the DataFrame's index so non-default indices don't
    silently produce wrong results."""
    df = pd.DataFrame(
        {"Weight": list(range(1, 21)) + [10_000]},
        index=[i * 7 for i in range(21)],
    )
    out, summary = apply_feature_outlier_policy(df, ["Weight"], policy="drop")
    assert len(out) == 20
    assert summary["n_outliers_total"] == 1
    assert 10_000 not in out["Weight"].values
