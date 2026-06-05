"""Tests for the SHAP-based explanation module.

The most important assertions here are the *axiomatic* ones: SHAP's local
accuracy (baseline + sum(contributions) == prediction) and, for log-target
models, multiplicative additivity in price space (product(factors) ==
prediction / baseline). If either of those fails, the explanation is not
SHAP — it's something else with a SHAP label, which is the failure mode
this module exists to prevent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.explain import (
    Explanation,
    _detect_family,
    explain_row,
    top_drivers,
)
from modules.prepare_data import prepare_data
from modules.training import start_training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _train(df, algorithm, target="Price", log_target_in_data=False):
    features = [c for c in df.columns if c != target]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(df, features, target)
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm=algorithm, tune=False
    )
    if log_target_in_data:
        assert log_target, (
            "Test was meant to exercise the log-target branch but the fitted "
            "pipeline did not wrap a TransformedTargetRegressor."
        )
    return model, X_train, X_test, log_target


@pytest.fixture
def ridge_model(synthetic_parts):
    return _train(synthetic_parts, "ridge")


@pytest.fixture
def rf_model(synthetic_parts):
    return _train(synthetic_parts, "random_forest")


@pytest.fixture
def log_target_rf_model(synthetic_parts_skewed):
    return _train(synthetic_parts_skewed, "random_forest", log_target_in_data=True)


# ---------------------------------------------------------------------------
# _detect_family — the routing logic that picks TreeExplainer vs LinearExplainer.
# ---------------------------------------------------------------------------


def test_detect_family_routes_each_supported_estimator():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from xgboost import XGBRegressor

    assert _detect_family(Ridge()) == "linear"
    assert _detect_family(Lasso()) == "linear"
    assert _detect_family(RandomForestRegressor()) == "tree"
    assert _detect_family(XGBRegressor()) == "tree"


def test_detect_family_unknown_estimator():
    class Mystery:
        pass

    assert _detect_family(Mystery()) == "unknown"


# ---------------------------------------------------------------------------
# Local accuracy axiom — the strongest contract this module promises.
# ---------------------------------------------------------------------------


def test_ridge_local_accuracy(ridge_model):
    """phi_0 + sum(phi_i) ~= f(x) for a linear model.

    LinearExplainer is closed-form so this should be exact to machine
    precision. We allow 1e-6 to absorb the additivity-via-source-rollup step.
    """
    model, X_train, X_test, _ = ridge_model
    bg = X_train.sample(50, random_state=0)
    row = X_test.iloc[[0]]
    ex = explain_row(model, row, background_X=bg)

    reconstructed = ex.baseline + sum(ex.contributions.values())
    assert reconstructed == pytest.approx(ex.prediction, abs=1e-6)
    assert abs(ex.residual) < 1e-6


def test_random_forest_local_accuracy(rf_model):
    """phi_0 + sum(phi_i) ~= f(x) for a tree ensemble.

    TreeExplainer with tree_path_dependent is exact in theory; in practice
    sklearn RandomForest can leave a 1e-5 floor from how leaf weights are
    averaged. The point is that the residual is *tiny* relative to the
    prediction — not literally zero.
    """
    model, X_train, X_test, _ = rf_model
    row = X_test.iloc[[0]]
    ex = explain_row(model, row, background_X=None)

    reconstructed = ex.baseline + sum(ex.contributions.values())
    assert reconstructed == pytest.approx(ex.prediction, rel=1e-4, abs=1e-4)


def test_random_forest_local_accuracy_across_many_rows(rf_model):
    """The local-accuracy property should hold for every row, not just one."""
    model, _, X_test, _ = rf_model
    for i in range(min(10, len(X_test))):
        row = X_test.iloc[[i]]
        ex = explain_row(model, row, background_X=None)
        reconstructed = ex.baseline + sum(ex.contributions.values())
        assert reconstructed == pytest.approx(ex.prediction, rel=1e-4, abs=1e-4)


# ---------------------------------------------------------------------------
# Source-feature rollup — the bit that combines one-hot dummies.
# ---------------------------------------------------------------------------


def test_ridge_rollup_keeps_all_source_columns(ridge_model):
    """Every source column should appear in `contributions`, even ones with
    near-zero attribution (so callers see the full picture and can decide
    what to drop on display)."""
    model, X_train, X_test, _ = ridge_model
    bg = X_train.sample(50, random_state=0)
    ex = explain_row(model, X_test.iloc[[0]], background_X=bg)
    assert set(ex.contributions.keys()) == set(X_test.columns)


def test_rollup_does_not_collapse_prefix_collisions(synthetic_parts):
    """A source column whose name is a prefix of another (e.g. 'Weight' and
    'weight_extra') must not have its contribution leaked into the other.
    """
    df = synthetic_parts.copy()
    df["weight_extra"] = df["Weight"] * 0.1
    model, X_train, X_test, _ = _train(df, "random_forest")
    row = X_test.iloc[[0]]
    ex = explain_row(model, row, background_X=None)
    # Both columns are present and tracked independently.
    assert "Weight" in ex.contributions
    assert "weight_extra" in ex.contributions
    assert len(set(ex.contributions.keys())) == len(X_test.columns)


# ---------------------------------------------------------------------------
# Log-target wrap — multiplicative additivity in price space.
# ---------------------------------------------------------------------------


def test_log_target_multiplicative_additivity(log_target_rf_model):
    """For a log-target model, the product of per-feature multiplicative
    factors times the baseline price must equal the predicted price.

    This is the axiomatically clean statement of "SHAP in price space" for
    a multiplicative model.
    """
    model, _, X_test, log_target = log_target_rf_model
    assert log_target

    row = X_test.iloc[[0]]
    ex = explain_row(model, row, background_X=None)

    assert ex.multiplicative_factors is not None
    product = float(np.prod(list(ex.multiplicative_factors.values())))
    ratio = ex.predicted_price / ex.baseline_price
    assert product == pytest.approx(ratio, rel=1e-4, abs=1e-4)


def test_log_target_dollar_attribution_sums_to_delta(log_target_rf_model):
    """The (approximate) dollar attribution is the proportional rescaling
    that *forces* additivity in price space — the residual we trade away the
    SHAP axioms for. Verify the rescaling actually adds up."""
    model, _, X_test, _ = log_target_rf_model
    row = X_test.iloc[[0]]
    ex = explain_row(model, row, background_X=None)

    assert ex.dollar_attribution is not None
    delta = ex.predicted_price - ex.baseline_price
    assert sum(ex.dollar_attribution.values()) == pytest.approx(delta, rel=1e-6, abs=1e-6)


def test_log_target_explanation_marks_log_target(log_target_rf_model):
    model, _, X_test, _ = log_target_rf_model
    ex = explain_row(model, X_test.iloc[[0]], background_X=None)
    assert ex.log_target is True
    assert ex.baseline_price > 0
    assert ex.predicted_price > 0


# ---------------------------------------------------------------------------
# top_drivers
# ---------------------------------------------------------------------------


def test_top_drivers_returns_n_features_in_decreasing_magnitude(ridge_model):
    model, X_train, X_test, _ = ridge_model
    bg = X_train.sample(50, random_state=0)
    ex = explain_row(model, X_test.iloc[[0]], background_X=bg)
    drivers = top_drivers(ex, n=2)
    assert len(drivers) == 2
    # The first driver has at least as much absolute contribution as the second.
    contrib = ex.contributions
    a = abs(contrib[drivers[0][0]])
    b = abs(contrib[drivers[1][0]])
    assert a >= b


def test_top_drivers_log_target_returns_multiplicative_factors(log_target_rf_model):
    model, _, X_test, _ = log_target_rf_model
    ex = explain_row(model, X_test.iloc[[0]], background_X=None)
    drivers = top_drivers(ex, n=2)
    assert len(drivers) == 2
    # In log-target mode the second element of each tuple is a multiplicative
    # factor — strictly positive, and the strongest driver has the largest
    # |log(factor)|.
    for _, value in drivers:
        assert value > 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_linear_model_without_background_raises_helpful_error(ridge_model):
    model, _, X_test, _ = ridge_model
    with pytest.raises(ValueError, match="background"):
        explain_row(model, X_test.iloc[[0]], background_X=None)


def test_explain_row_rejects_multi_row_input(ridge_model):
    model, X_train, X_test, _ = ridge_model
    bg = X_train.sample(50, random_state=0)
    with pytest.raises(ValueError, match="single-row"):
        explain_row(model, X_test.iloc[:2], background_X=bg)
