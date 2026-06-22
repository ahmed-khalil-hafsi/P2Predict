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


def test_interval_reliability_thresholds():
    # Tight band (±10% of prediction) → trust.
    assert quality.interval_reliability(90.0, 100.0, 110.0) == "trust"
    # Moderate band (±30%) → caution.
    assert quality.interval_reliability(70.0, 100.0, 130.0) == "caution"
    # Wide band (>80% of prediction) → quote.
    assert quality.interval_reliability(40.0, 100.0, 150.0) == "quote"
    # Lower bound underwater → always quote, regardless of width.
    assert quality.interval_reliability(-5.0, 10.0, 30.0) == "quote"


def test_interval_say_to_user_flags_negative_floor():
    msg = quality.interval_say_to_user(-5.0, 10.0, 30.0)
    assert "$0" in msg and "quote" in msg.lower()
    # No statistical jargon leaks into the plain sentence.
    for jargon in ("conformal", "coverage", "residual"):
        assert jargon not in msg.lower()


def test_assess_model_modest_but_unbiased_is_usable():
    a = quality.assess_model(r2=0.512, residual_bias_p=0.09, n_holdout=30)
    assert a["accuracy"] == "modest"
    assert a["unbiased"] is True
    assert a["verdict"] == "usable"
    assert a["confidence"] == "limited"
    assert "even-handed" in a["headline"].lower()


def test_assess_model_flags_bias():
    a = quality.assess_model(r2=0.85, residual_bias_p=1e-6, n_holdout=60)
    assert a["unbiased"] is False
    assert a["verdict"] == "unreliable"
    assert "systematically" in a["headline"].lower()


def test_assess_model_trustworthy_high_confidence():
    a = quality.assess_model(r2=0.85, residual_bias_p=0.5, n_holdout=60)
    assert a["verdict"] == "trustworthy"
    assert a["confidence"] == "high"


def test_assess_model_insufficient_data_overrides_everything():
    # Even a great-looking model is 'insufficient_data' with too few points.
    a = quality.assess_model(r2=0.95, residual_bias_p=0.9, n_holdout=8)
    assert a["verdict"] == "insufficient_data"
    assert a["confidence"] == "insufficient"
    assert "too few" in a["headline"].lower()


def test_assess_model_unknown_when_bias_unmeasurable():
    a = quality.assess_model(r2=0.6, residual_bias_p=float("nan"), n_holdout=30)
    assert a["verdict"] == "unknown"
    assert a["unbiased"] is None


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
    assert "verdict" in rep["assessment"]
    assert rep["calibration_by_price_band"]  # 40 points -> bins
    for band in rep["calibration_by_price_band"]:
        assert band["reliability"] in {"trust", "caution", "quote"}
    # 40 points / 10 bins -> ~4 per band -> flagged low_confidence.
    assert any(b.get("low_confidence") for b in rep["calibration_by_price_band"])
    sig = {f["feature"]: f["signal"] for f in rep["feature_importance"]}
    assert sig["manufacturer"] == "strong"
    assert sig["package_pins"] == "weak"
    # No raw holdout unless asked.
    assert "holdout" not in rep


def test_build_quality_report_include_holdout():
    rng = np.random.default_rng(0)
    y_test = rng.uniform(0.5, 7.0, 40)
    y_pred = y_test + rng.normal(0, 0.4, 40)
    loaded = {"holdout_y_test": y_test.tolist(), "holdout_y_pred": y_pred.tolist(),
              "features": ["a"]}
    rep = quality.build_quality_report(loaded, include_holdout=True)
    assert len(rep["holdout"]["y_actual"]) == 40
    assert len(rep["holdout"]["y_predicted"]) == 40


# Terms a category manager has never heard — must never appear in any string
# the payload hands the agent to quote. Matched case-insensitively as substrings.
BANNED_USER_TERMS = (
    "shap", "r²", "r2", "p-value", "p_value", "holdout",
    "residual", "log-target", "log target",
)


def _assert_clean(text: str):
    low = text.lower()
    for term in BANNED_USER_TERMS:
        assert term not in low, f"jargon {term!r} leaked into a user string: {text!r}"


def test_assess_model_headlines_are_jargon_free():
    # Every verdict's headline is quoted to the user — keep them all clean.
    cases = [
        (0.512, 0.09, 30),   # usable
        (0.85, 1e-6, 60),    # unreliable
        (0.85, 0.5, 60),     # trustworthy
        (0.95, 0.9, 8),      # insufficient_data
        (0.6, float("nan"), 30),  # unknown
    ]
    for r2, p, n in cases:
        _assert_clean(quality.assess_model(r2, p, n)["headline"])


def test_quality_report_default_is_business_only():
    rng = np.random.default_rng(0)
    y_test = rng.uniform(0.5, 7.0, 60)
    y_pred = y_test + rng.normal(0, 0.4, 60)
    loaded = {
        "holdout_y_test": y_test.tolist(), "holdout_y_pred": y_pred.tolist(),
        "target_feature": "price", "model_name": "xgboost", "log_target": True,
        "features": ["manufacturer", "pins"],
    }
    importances = [("manufacturer", 40.0), ("pins", 8.0)]

    # Default: raw stats gated out, every emitted string clean.
    rep = quality.build_quality_report(loaded, importances)
    assert "r2" not in rep["metrics"]
    assert "residual_bias_p_value" not in rep["metrics"]
    assert "algorithm" not in rep["provenance"]
    assert "log_target" not in rep["provenance"]
    assert "typical_pct_error" in rep["metrics"]
    _assert_clean(rep["assessment"]["headline"])
    for band in rep["calibration_by_price_band"]:
        _assert_clean(band["say_to_user"])
    for feat in rep["feature_importance"]:
        _assert_clean(feat["say_to_user"])

    # Opt-in restores the raw statistics for developer use.
    rep_full = quality.build_quality_report(loaded, importances, include_metrics=True)
    assert "r2" in rep_full["metrics"]
    assert "log_target" in rep_full["provenance"]


def test_build_quality_report_thin_holdout_says_so():
    # Few points: report still builds, but the verdict is honest about it.
    rng = np.random.default_rng(0)
    y_test = rng.uniform(0.5, 7.0, 8)
    y_pred = y_test + rng.normal(0, 0.3, 8)
    loaded = {"holdout_y_test": y_test.tolist(), "holdout_y_pred": y_pred.tolist(),
              "features": ["a"]}
    rep = quality.build_quality_report(loaded)
    assert rep["assessment"]["verdict"] == "insufficient_data"
    # Too few to band -> explained, not silently empty.
    assert rep["calibration_by_price_band"] == []
    assert rep["calibration_note"]


def test_build_quality_report_requires_holdout():
    import pytest
    with pytest.raises(ValueError):
        quality.build_quality_report({"features": []})


def test_plotting_reuses_quality_stats():
    # The PDF must compute identical numbers to the JSON report.
    from p2predict import plotting
    assert plotting._summary_metrics is quality.summary_metrics
    assert plotting._error_by_price_band is quality.error_by_price_band
