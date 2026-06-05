"""Tests for the what-if comparison module.

Headline assertions are axiomatic:

  Total delta decomposition (the SHAP local-accuracy axiom applied to
  the difference of two predictions): for both non-log and log-target
  models, the per-feature deltas must sum to the inner-model delta
  exactly (modulo floating point). Without this property, the
  "drivers of the change" attribution doesn't mean anything.

  Log-target multiplicativity: the product of per-feature
  multiplicative factors equals cf_price / base_price exactly.

Without these tests the rendering looks fine but the numbers in a
design-review meeting could be wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.prepare_data import prepare_data
from modules.training import start_training
from modules.intervals import compute_calibration_residuals
from modules.whatif import (
    INTERACTION_MATERIALITY_THRESHOLD,
    WhatIfResult,
    compute_whatif,
    interaction_is_material,
    parse_changes,
)


# ---------------------------------------------------------------------------
# Synthetic-data fixtures large enough that SHAP has signal to attribute.
# ---------------------------------------------------------------------------


def _generate(n=600, seed=0, skewed=False):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(1, 50, n)
    region = rng.choice(["EU", "CN", "SG", "US"], n)
    size = rng.choice(["Small", "Standard", "Large"], n)
    base = (
        0.08 * weight
        + np.where(region == "EU", 0.5, 0.0)
        + np.where(size == "Large", 0.7, 0.0)
    )
    noise = rng.normal(0, 0.15, n)
    price = np.clip(base + noise, 0.05, None)
    if skewed:
        price = np.exp(price)
    return pd.DataFrame({
        "Weight": weight, "Region": region, "Size": size, "Price": price,
    })


def _train(df, algorithm, log_target_in_data=False):
    features = [c for c in df.columns if c != "Price"]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(df, features, "Price")
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm=algorithm, tune=False
    )
    if log_target_in_data:
        assert log_target
    calibration = compute_calibration_residuals(model, X_test, y_test)
    return model, X_train, X_test, calibration, log_target


@pytest.fixture
def rf_setup():
    df = _generate()
    model, X_train, X_test, calibration, log_target = _train(df, "random_forest")
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "calibration": calibration,
        "log_target": log_target,
        "feature_types": {
            "Weight": "Numerical", "Region": "Categorical", "Size": "Categorical"
        },
    }


@pytest.fixture
def ridge_setup():
    df = _generate()
    model, X_train, X_test, calibration, log_target = _train(df, "ridge")
    return {
        "model": model,
        "X_train": X_train,
        "background": X_train.sample(100, random_state=0),
        "X_test": X_test,
        "calibration": calibration,
        "log_target": log_target,
        "feature_types": {
            "Weight": "Numerical", "Region": "Categorical", "Size": "Categorical"
        },
    }


@pytest.fixture
def log_target_setup():
    df = _generate(skewed=True)
    model, X_train, X_test, calibration, log_target = _train(
        df, "random_forest", log_target_in_data=True
    )
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "calibration": calibration,
        "log_target": log_target,
        "feature_types": {
            "Weight": "Numerical", "Region": "Categorical", "Size": "Categorical"
        },
    }


# ---------------------------------------------------------------------------
# parse_changes — the CLI parser.
# ---------------------------------------------------------------------------


def test_parse_changes_single():
    assert parse_changes("Region:EU") == {"Region": "EU"}


def test_parse_changes_multiple():
    assert parse_changes("Region:EU,Supplier:B,Weight:20") == {
        "Region": "EU", "Supplier": "B", "Weight": "20",
    }


def test_parse_changes_strips_whitespace():
    assert parse_changes(" Region : EU , Supplier : B ") == {
        "Region": "EU", "Supplier": "B",
    }


def test_parse_changes_rejects_malformed():
    with pytest.raises(ValueError):
        parse_changes("RegionEU")
    with pytest.raises(ValueError):
        parse_changes("Region:")  # empty value
    with pytest.raises(ValueError):
        parse_changes(":EU")  # empty key


# ---------------------------------------------------------------------------
# Decomposition axiom — the property the feature claims.
# ---------------------------------------------------------------------------


def test_decomposition_sums_to_total_delta_rf(rf_setup):
    """Sum of changed contributions + interaction contribution should
    equal the inner-model delta within floating-point tolerance."""
    base = rf_setup["X_test"].iloc[[0]]
    # Change one categorical feature.
    cf_value = "EU" if base["Region"].iloc[0] != "EU" else "CN"
    result = compute_whatif(
        rf_setup["model"], base, {"Region": cf_value},
        rf_setup["feature_types"],
    )
    decomposed = (
        sum(result.changed_contributions.values())
        + result.interaction_contribution
    )
    inner_delta = result.counterfactual_prediction - result.base_prediction
    assert decomposed == pytest.approx(inner_delta, abs=1e-6, rel=1e-4)
    assert abs(result.decomposition_residual) < 1e-5


def test_decomposition_sums_to_total_delta_ridge(ridge_setup):
    base = ridge_setup["X_test"].iloc[[0]]
    cf_value = "EU" if base["Region"].iloc[0] != "EU" else "CN"
    # For ridge, explainer needs background and we wire it via the same
    # background that lives in the model metadata at predict time. Pass
    # explicitly here.
    result = compute_whatif(
        ridge_setup["model"], base, {"Region": cf_value},
        ridge_setup["feature_types"],
        background_X=ridge_setup["background"],
    )
    decomposed = (
        sum(result.changed_contributions.values())
        + result.interaction_contribution
    )
    inner_delta = result.counterfactual_prediction - result.base_prediction
    assert decomposed == pytest.approx(inner_delta, abs=1e-6, rel=1e-4)


def test_multi_feature_change_decomposition(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    # Change two features at once.
    cf_changes = {
        "Region": "EU" if base["Region"].iloc[0] != "EU" else "CN",
        "Weight": str(float(base["Weight"].iloc[0]) + 5.0),
    }
    result = compute_whatif(
        rf_setup["model"], base, cf_changes, rf_setup["feature_types"]
    )
    decomposed = (
        sum(result.changed_contributions.values())
        + result.interaction_contribution
    )
    inner_delta = result.counterfactual_prediction - result.base_prediction
    assert decomposed == pytest.approx(inner_delta, abs=1e-4, rel=1e-3)
    # Each of the two changed features should appear in the contributions.
    assert set(result.changed_contributions.keys()) == set(cf_changes.keys())


# ---------------------------------------------------------------------------
# Log-target multiplicativity — the equivalent contract in price space.
# ---------------------------------------------------------------------------


def test_log_target_multiplicative_decomposition(log_target_setup):
    """For a log-target model the per-feature multiplicative factors
    times the interaction factor should equal cf_price / base_price."""
    base = log_target_setup["X_test"].iloc[[0]]
    cf_value = "EU" if base["Region"].iloc[0] != "EU" else "CN"
    result = compute_whatif(
        log_target_setup["model"], base, {"Region": cf_value},
        log_target_setup["feature_types"],
    )
    assert result.log_target
    assert result.multiplicative_factor is not None
    assert result.changed_multiplicative_factors is not None
    assert result.interaction_multiplicative_factor is not None

    product = float(
        np.prod(list(result.changed_multiplicative_factors.values()))
        * result.interaction_multiplicative_factor
    )
    assert product == pytest.approx(result.multiplicative_factor, abs=1e-4, rel=1e-4)


def test_log_target_factors_are_strictly_positive(log_target_setup):
    base = log_target_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        log_target_setup["model"], base, {"Region": "EU"},
        log_target_setup["feature_types"],
    )
    for factor in result.changed_multiplicative_factors.values():
        assert factor > 0
    assert result.interaction_multiplicative_factor > 0
    assert result.multiplicative_factor > 0


# ---------------------------------------------------------------------------
# Counterfactual sanity — predict on the actual modified row.
# ---------------------------------------------------------------------------


def test_counterfactual_prediction_matches_independent_predict(rf_setup):
    """compute_whatif must call predict on the *modified* row, not on the
    base row. Verify by independently building the modified row and
    predicting on it."""
    model = rf_setup["model"]
    base = rf_setup["X_test"].iloc[[0]]
    target_region = "EU" if base["Region"].iloc[0] != "EU" else "CN"

    result = compute_whatif(
        model, base, {"Region": target_region},
        rf_setup["feature_types"],
    )

    modified = base.copy()
    modified["Region"] = target_region
    independent_pred = float(model.predict(modified)[0])
    assert result.counterfactual_prediction == pytest.approx(independent_pred)


def test_base_prediction_matches_independent_predict(rf_setup):
    model = rf_setup["model"]
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        model, base, {"Region": "EU"}, rf_setup["feature_types"]
    )
    independent_pred = float(model.predict(base)[0])
    assert result.base_prediction == pytest.approx(independent_pred)


# ---------------------------------------------------------------------------
# Interval composition — both base and cf carry intervals when calibration
# is available.
# ---------------------------------------------------------------------------


def test_intervals_attached_when_calibration_present(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        rf_setup["model"], base, {"Region": "EU"},
        rf_setup["feature_types"],
        calibration=rf_setup["calibration"],
        coverage=0.90,
    )
    assert result.base_interval is not None
    assert result.cf_interval is not None
    assert result.base_interval.low < result.base_interval.high
    assert result.cf_interval.low < result.cf_interval.high


def test_intervals_skipped_when_calibration_missing(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    result = compute_whatif(
        rf_setup["model"], base, {"Region": "EU"},
        rf_setup["feature_types"],
        calibration=None,
    )
    assert result.base_interval is None
    assert result.cf_interval is None


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------


def test_unknown_feature_raises_helpful(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    with pytest.raises(ValueError, match="not a training feature"):
        compute_whatif(
            rf_setup["model"], base, {"NotAColumn": "X"},
            rf_setup["feature_types"],
        )


def test_non_numeric_value_for_numeric_feature_raises(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    with pytest.raises(ValueError, match="numeric"):
        compute_whatif(
            rf_setup["model"], base, {"Weight": "heavy"},
            rf_setup["feature_types"],
        )


def test_empty_changes_raises(rf_setup):
    base = rf_setup["X_test"].iloc[[0]]
    with pytest.raises(ValueError):
        compute_whatif(rf_setup["model"], base, {}, rf_setup["feature_types"])


def test_multi_row_base_raises(rf_setup):
    base = rf_setup["X_test"].iloc[:2]  # two rows
    with pytest.raises(ValueError, match="single-row"):
        compute_whatif(
            rf_setup["model"], base, {"Region": "EU"},
            rf_setup["feature_types"],
        )


# ---------------------------------------------------------------------------
# Interaction materiality helper.
# ---------------------------------------------------------------------------


def test_interaction_is_material_threshold():
    big = WhatIfResult(
        base_prediction=10.0, counterfactual_prediction=12.0,
        delta=2.0, delta_pct=20.0, changes={},
        changed_contributions={}, interaction_contribution=0.5,  # 25% of delta
    )
    assert interaction_is_material(big)

    tiny = WhatIfResult(
        base_prediction=10.0, counterfactual_prediction=12.0,
        delta=2.0, delta_pct=20.0, changes={},
        changed_contributions={}, interaction_contribution=0.02,  # 1% of delta
    )
    assert not interaction_is_material(tiny)
