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
    multiplicative — i.e. the ratio high / low is the constant exp(2 * q_hat)
    for every prediction *within the same calibration band* (and globally
    when banding is inactive). That's what makes the interval scale-natural
    for procurement prices."""
    model, X_test, _, calibration = log_target_setup
    intervals = predict_interval(model, X_test.head(50), calibration, coverage=0.90)
    by_band: dict = {}
    for ir in intervals:
        by_band.setdefault(ir.band, []).append(ir.high / ir.low)
    for ratios in by_band.values():
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
# Banded (Mondrian) calibration
# ---------------------------------------------------------------------------


def _generate_heteroscedastic(n=3000, seed=7):
    """Multiplicative price with noise that *shrinks* as the price grows —
    the fastener-catalog pattern (cheap commodity parts are near-random,
    expensive parts are priced more consistently)."""
    rng = np.random.default_rng(seed)
    weight = rng.uniform(1, 50, n)
    region = rng.choice(["EU", "CN", "SG", "US"], n)
    base = weight ** 1.5
    sigma = np.where(weight < 15, 0.9, np.where(weight < 30, 0.4, 0.1))
    price = base * np.exp(rng.normal(0.0, sigma))
    return pd.DataFrame({"Weight": weight, "Region": region, "Price": price})


@pytest.fixture
def banded_setup():
    df = _generate_heteroscedastic()
    X_train, X_test, y_train, y_test, num, cat = prepare_data(
        df, ["Weight", "Region"], "Price"
    )
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="random_forest", tune=False,
        log_target=True,  # force the multiplicative path (skew here is ~0.9)
    )
    assert log_target
    # Split the holdout into calibration and evaluation halves so the
    # coverage assertions are computed on points the calibration never saw.
    half = len(X_test) // 2
    X_cal, y_cal = X_test.iloc[:half], y_test.iloc[:half]
    X_eval, y_eval = X_test.iloc[half:], y_test.iloc[half:]
    calibration = compute_calibration_residuals(model, X_cal, y_cal)
    return model, X_eval, y_eval, calibration


def test_calibration_stores_predictions(big_holdout_setup):
    model, X_test, y_test, cal = big_holdout_setup
    assert "predictions" in cal
    assert len(cal["predictions"]) == len(cal["residuals"]) == cal["n_calibration"]


def test_banded_widths_track_heteroscedastic_noise(banded_setup):
    """On data whose noise shrinks with price, the high-price band must get
    a materially narrower multiplicative interval than the low-price band —
    the business point of banding: the noisiest segment no longer sets the
    width for everyone."""
    model, X_eval, _, calibration = banded_setup
    intervals = predict_interval(model, X_eval, calibration, coverage=0.90)
    assert any(ir.band is not None for ir in intervals)
    preds = np.array([ir.prediction for ir in intervals])
    ratios = np.array([ir.high / ir.low for ir in intervals])
    lo_band = ratios[preds <= np.quantile(preds, 0.2)].mean()
    hi_band = ratios[preds >= np.quantile(preds, 0.8)].mean()
    assert hi_band < lo_band / 2, (
        f"expected the expensive band to be at least 2x narrower; "
        f"got low-band ratio {lo_band:.1f} vs high-band {hi_band:.1f}"
    )


def test_banded_per_band_empirical_coverage(banded_setup):
    """The Mondrian guarantee: ~90% coverage *within each band*, not just on
    average. This is exactly what a single global quantile does NOT provide
    on heteroscedastic data (it over-covers the quiet band and under-covers
    the noisy one)."""
    model, X_eval, y_eval, calibration = banded_setup
    intervals = predict_interval(model, X_eval, calibration, coverage=0.90)
    actuals = y_eval.to_numpy()
    by_band: dict = {}
    for ir, a in zip(intervals, actuals):
        by_band.setdefault(ir.band, []).append(ir.low <= a <= ir.high)
    for band, hits in by_band.items():
        if len(hits) < 50:
            continue  # too small for a stable empirical rate
        assert np.mean(hits) == pytest.approx(0.90, abs=0.07), band


def test_banded_intervals_stay_positive_under_log_target(banded_setup):
    model, X_eval, _, calibration = banded_setup
    for ir in predict_interval(model, X_eval, calibration, coverage=0.99):
        assert 0 < ir.low < ir.high


def test_old_calibration_dict_falls_back_to_global(banded_setup):
    """Calibration dicts persisted by older model files carry no
    'predictions' key. They must reproduce the pre-banding behaviour
    exactly: one global q_hat, identical width everywhere, band=None."""
    model, X_eval, _, calibration = banded_setup
    legacy = {k: v for k, v in calibration.items() if k != "predictions"}
    intervals = predict_interval(model, X_eval, legacy, coverage=0.90)
    assert all(ir.band is None for ir in intervals)
    ratios = [ir.high / ir.low for ir in intervals]
    assert max(ratios) - min(ratios) < 1e-9
    expected = float(
        np.exp(2 * _conformal_quantile(np.asarray(legacy["residuals"]), 0.1))
    )
    assert ratios[0] == pytest.approx(expected)


def test_small_calibration_falls_back_to_global(banded_setup):
    """Below MIN_CALIBRATION_FOR_BANDING the per-band quantiles would be
    too noisy to trust — stay global (BMIC-sized datasets hit this)."""
    model, X_eval, _, calibration = banded_setup
    small = dict(calibration)
    small["residuals"] = calibration["residuals"][:100]
    small["predictions"] = calibration["predictions"][:100]
    small["n_calibration"] = 100
    intervals = predict_interval(model, X_eval.head(10), small, coverage=0.90)
    assert all(ir.band is None for ir in intervals)


def test_degenerate_predictions_fall_back_to_global(banded_setup):
    """If the calibration predictions are (near-)constant the tercile edges
    collapse; banding must bow out instead of building empty bands."""
    model, X_eval, _, calibration = banded_setup
    degenerate = dict(calibration)
    degenerate["predictions"] = [5.0] * len(calibration["residuals"])
    intervals = predict_interval(model, X_eval.head(10), degenerate, coverage=0.90)
    assert all(ir.band is None for ir in intervals)


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
