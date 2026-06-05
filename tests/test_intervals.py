"""Tests for split-conformal prediction intervals.

The headline assertions are the *empirical coverage* tests: a 90% likely
range should actually cover ~90% of held-out points. That's the property
the feature claims to provide; without it, the intervals are pretty
graphics with no procurement value. Conformal prediction's coverage
guarantee holds for any n >= 1 under exchangeability, but the finite-
sample width can be noisy on small calibration sets — so we use a large
synthetic dataset to give the empirical coverage room to converge on
the target rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2predict.intervals import (
    IntervalResult,
    _conformal_quantile,
    compute_calibration_residuals,
    coverage_health,
    predict_interval,
)
from p2predict.prepare_data import prepare_data
from p2predict.training import start_training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _generate_synthetic(n=2000, seed=0, skewed=False):
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
        # Log-normal target so should_log_target fires and we exercise the
        # multiplicative-interval path.
        price = np.exp(price)
    return pd.DataFrame({
        "Weight": weight, "Region": region, "Size": size, "Price": price,
    })


@pytest.fixture
def big_holdout_setup():
    """Train on 1600 rows, calibrate + evaluate coverage on 400 rows."""
    df = _generate_synthetic(n=2000)
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        df, ["Weight", "Region", "Size"], "Price"
    )
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    calibration = compute_calibration_residuals(model, X_test, y_test)
    return model, X_test, y_test, calibration


@pytest.fixture
def log_target_setup():
    df = _generate_synthetic(n=2000, skewed=True)
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        df, ["Weight", "Region", "Size"], "Price"
    )
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False
    )
    assert log_target, "Expected the log-target branch to engage."
    calibration = compute_calibration_residuals(model, X_test, y_test)
    return model, X_test, y_test, calibration


# ---------------------------------------------------------------------------
# Coverage axiom — the feature's headline contract.
# ---------------------------------------------------------------------------


def test_90_percent_interval_covers_about_90_percent(big_holdout_setup):
    """A 90% likely range should actually cover ~90% of held-out points.

    With 400 holdout points and a target rate of 90%, the empirical
    coverage will sit somewhere around 88-95% (binomial noise). We assert
    +/- 5pp which is well within statistical tolerance and still tight
    enough to catch a miscalibrated implementation.
    """
    model, X_test, y_test, calibration = big_holdout_setup
    intervals = predict_interval(model, X_test, calibration, coverage=0.90)
    actuals = y_test.to_numpy()
    inside = sum(
        ir.low <= a <= ir.high for ir, a in zip(intervals, actuals)
    )
    empirical = inside / len(actuals)
    assert empirical == pytest.approx(0.90, abs=0.05)


def test_80_percent_interval_covers_about_80_percent(big_holdout_setup):
    model, X_test, y_test, calibration = big_holdout_setup
    intervals = predict_interval(model, X_test, calibration, coverage=0.80)
    actuals = y_test.to_numpy()
    inside = sum(ir.low <= a <= ir.high for ir, a in zip(intervals, actuals))
    assert inside / len(actuals) == pytest.approx(0.80, abs=0.05)


def test_95_percent_interval_covers_about_95_percent(big_holdout_setup):
    model, X_test, y_test, calibration = big_holdout_setup
    intervals = predict_interval(model, X_test, calibration, coverage=0.95)
    actuals = y_test.to_numpy()
    inside = sum(ir.low <= a <= ir.high for ir, a in zip(intervals, actuals))
    assert inside / len(actuals) == pytest.approx(0.95, abs=0.05)


def test_coverage_is_monotone_in_coverage_level(big_holdout_setup):
    """Higher coverage targets must produce wider intervals."""
    model, X_test, _, calibration = big_holdout_setup
    sample = X_test.head(20)
    intervals_80 = predict_interval(model, sample, calibration, coverage=0.80)
    intervals_95 = predict_interval(model, sample, calibration, coverage=0.95)
    for i80, i95 in zip(intervals_80, intervals_95):
        width_80 = i80.high - i80.low
        width_95 = i95.high - i95.low
        assert width_95 >= width_80


# ---------------------------------------------------------------------------
# Log-target path — multiplicative intervals.
# ---------------------------------------------------------------------------


def test_log_target_intervals_are_multiplicative(log_target_setup):
    """When the model uses log/exp, the bounds in price space should be
    multiplicative — i.e. the ratio high / low is the same constant for
    every prediction. That constant is exp(2 * q_hat), which is what
    makes the interval scale-natural for procurement prices."""
    model, X_test, _, calibration = log_target_setup
    intervals = predict_interval(model, X_test.head(50), calibration, coverage=0.90)
    ratios = [ir.high / ir.low for ir in intervals]
    # All ratios should be equal (multiplicative interval) to within
    # floating-point tolerance.
    assert max(ratios) - min(ratios) < 1e-9


def test_log_target_intervals_are_strictly_positive(log_target_setup):
    """A multiplicative bound on a positive prediction must stay positive —
    something a constant-width additive interval cannot guarantee."""
    model, X_test, _, calibration = log_target_setup
    intervals = predict_interval(model, X_test.head(50), calibration, coverage=0.99)
    for ir in intervals:
        assert ir.low > 0
        assert ir.prediction > 0
        assert ir.high > ir.low


def test_log_target_empirical_coverage(log_target_setup):
    """Same coverage assertion as the non-log case, exercised through
    the log-space calibration path."""
    model, X_test, y_test, calibration = log_target_setup
    intervals = predict_interval(model, X_test, calibration, coverage=0.90)
    actuals = y_test.to_numpy()
    inside = sum(ir.low <= a <= ir.high for ir, a in zip(intervals, actuals))
    assert inside / len(actuals) == pytest.approx(0.90, abs=0.05)


# ---------------------------------------------------------------------------
# Quantile arithmetic — guard against off-by-one in the conformal recipe.
# ---------------------------------------------------------------------------


def test_conformal_quantile_uses_method_higher():
    """k = ceil((n+1)(1-alpha)). With n=10 and alpha=0.1 that's k=10,
    so the q_hat is the *largest* residual."""
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    q = _conformal_quantile(residuals, alpha=0.1)
    assert q == 10.0


def test_conformal_quantile_with_one_residual():
    """Degenerate case — a single calibration point gives a wide
    interval but the algorithm must not crash."""
    assert _conformal_quantile(np.array([3.0]), alpha=0.5) == 3.0


def test_conformal_quantile_zero_residuals_raises():
    with pytest.raises(ValueError):
        _conformal_quantile(np.array([]), alpha=0.5)


# ---------------------------------------------------------------------------
# Health / soft-warning surface
# ---------------------------------------------------------------------------


def test_coverage_health_returns_none_for_healthy_calibration():
    cal = {"residuals": [0.1] * 50, "in_log_space": False, "n_calibration": 50}
    assert coverage_health(cal) is None


def test_coverage_health_warns_about_tiny_calibration():
    cal = {"residuals": [0.1] * 8, "in_log_space": False, "n_calibration": 8}
    assert "small" in coverage_health(cal)


def test_coverage_health_signals_missing_calibration():
    msg = coverage_health(None)
    assert msg is not None
    assert "re-train" in msg


# ---------------------------------------------------------------------------
# Compute side — residuals match what split conformal expects.
# ---------------------------------------------------------------------------


def test_compute_calibration_residuals_non_log_target(big_holdout_setup):
    model, X_test, y_test, _ = big_holdout_setup
    cal = compute_calibration_residuals(model, X_test, y_test)
    assert cal["in_log_space"] is False
    assert cal["n_calibration"] == len(X_test)
    # Residuals are non-negative absolute residuals.
    assert all(r >= 0 for r in cal["residuals"])


def test_compute_calibration_residuals_log_target(log_target_setup):
    model, X_test, y_test, _ = log_target_setup
    cal = compute_calibration_residuals(model, X_test, y_test)
    assert cal["in_log_space"] is True
    assert all(r >= 0 for r in cal["residuals"])


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_predict_interval_rejects_bad_coverage(big_holdout_setup):
    model, X_test, _, calibration = big_holdout_setup
    with pytest.raises(ValueError):
        predict_interval(model, X_test.head(5), calibration, coverage=0.0)
    with pytest.raises(ValueError):
        predict_interval(model, X_test.head(5), calibration, coverage=1.0)
    with pytest.raises(ValueError):
        predict_interval(model, X_test.head(5), calibration, coverage=1.5)
