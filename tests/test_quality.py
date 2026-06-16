"""Tests for the centralised model-quality judgment layer."""
from __future__ import annotations

import numpy as np

from p2predict import quality


def test_r2_quality_label_thresholds():
    assert quality.r2_quality_label(0.95) == "Excellent"
    assert quality.r2_quality_label(0.70) == "Good"
    assert quality.r2_quality_label(0.512) == "Needs Improvement"
    # Clamps out-of-range R².
    assert quality.r2_quality_label(-3.0) == "Needs Improvement"


def test_band_reliability_thresholds():
    assert quality.band_reliability(5.0) == "trust"
    assert quality.band_reliability(quality.BAND_TRUST_MAX_PCT) == "trust"
    assert quality.band_reliability(25.0) == "caution"
    assert quality.band_reliability(80.0) == "quote"


def test_feature_signal_thresholds():
    assert quality.feature_signal(40.0) == "strong"
    assert quality.feature_signal(5.0) == "moderate"
    assert quality.feature_signal(1.2) == "weak"


def test_assess_model_modest_but_unbiased_is_usable():
    a = quality.assess_model(r2=0.512, residual_bias_p=0.09)
    assert a["accuracy"] == "modest"
    assert a["unbiased"] is True
    assert "unbiased" in a["headline"].lower()


def test_assess_model_flags_bias():
    a = quality.assess_model(r2=0.85, residual_bias_p=1e-6)
    assert a["unbiased"] is False
    assert "bias" in a["headline"].lower()


def test_build_quality_report_shape():
    rng = np.random.default_rng(0)
    y_test = rng.uniform(0.5, 7.0, 40)
    y_pred = y_test + rng.normal(0, 0.4, 40)
    loaded = {
        "holdout_y_test": y_test.tolist(),
        "holdout_y_pred": y_pred.tolist(),
        "target_feature": "unit_price_at_1_usd",
        "model_name": "ridge",
        "log_target": False,
        "features": ["manufacturer", "package_pins"],
        "training_date": "20260616",
    }
    importances = [("manufacturer", 40.0), ("package_pins", 0.5)]
    rep = quality.build_quality_report(loaded, importances)

    assert set(rep) >= {
        "provenance", "metrics", "assessment",
        "calibration_by_price_band", "feature_importance",
    }
    assert rep["provenance"]["n_features"] == 2
    assert "quality_label" in rep["metrics"]
    assert rep["calibration_by_price_band"]  # 40 points -> bins
    for band in rep["calibration_by_price_band"]:
        assert band["reliability"] in {"trust", "caution", "quote"}
    sig = {f["feature"]: f["signal"] for f in rep["feature_importance"]}
    assert sig["manufacturer"] == "strong"
    assert sig["package_pins"] == "weak"


def test_build_quality_report_requires_holdout():
    import pytest
    with pytest.raises(ValueError):
        quality.build_quality_report({"features": []})


def test_plotting_reuses_quality_stats():
    # The PDF must compute identical numbers to the JSON report.
    from p2predict import plotting
    assert plotting._summary_metrics is quality.summary_metrics
    assert plotting._error_by_price_band is quality.error_by_price_band
