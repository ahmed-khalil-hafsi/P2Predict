import numpy as np
import pandas as pd
import pytest

from modules.outliers import POLICIES, apply_outlier_policy, detect_outliers


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
