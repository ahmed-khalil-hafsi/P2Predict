"""Tests for target-leakage detection in feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from p2predict.feature_selection import find_leaky_features


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    n = 120
    price = rng.uniform(0.5, 7.0, n)
    return pd.DataFrame({
        "manufacturer": rng.choice(["TI", "ADI", "Microchip"], n),
        "package_pins": rng.integers(6, 48, n).astype(float),
        # An alternate form of the target: the same price at the 1k break.
        "price_at_1k_usd": price * 0.54 + rng.normal(0, 0.01, n),
        "unit_price_at_1_usd": price,
    })


def test_flags_alternate_form_of_target():
    leaks = find_leaky_features(_frame(), "unit_price_at_1_usd")
    names = [d["feature"] for d in leaks]
    assert "price_at_1k_usd" in names
    leak = next(d for d in leaks if d["feature"] == "price_at_1k_usd")
    assert leak["correlation"] >= 0.97
    assert "alternate form" in leak["reason"]


def test_does_not_flag_genuine_specs():
    leaks = find_leaky_features(_frame(), "unit_price_at_1_usd")
    names = [d["feature"] for d in leaks]
    assert "package_pins" not in names
    assert "manufacturer" not in names  # categorical, never screened


def test_target_never_returned_and_missing_target_is_safe():
    df = _frame()
    leaks = find_leaky_features(df, "unit_price_at_1_usd")
    assert "unit_price_at_1_usd" not in [d["feature"] for d in leaks]
    assert find_leaky_features(df, "no_such_column") == []


def test_threshold_is_respected():
    # A merely-correlated spec (below threshold) is not leakage.
    rng = np.random.default_rng(1)
    n = 200
    price = rng.uniform(1, 5, n)
    df = pd.DataFrame({
        "loosely_related": price + rng.normal(0, 2.0, n),  # weak corr
        "price": price,
    })
    assert find_leaky_features(df, "price", threshold=0.97) == []
