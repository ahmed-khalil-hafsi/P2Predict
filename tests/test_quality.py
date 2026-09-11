"""Tests for the centralised model-quality judgment layer."""
from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Bias materiality — the equivalence test.
# Rationale and measured operating characteristics:
# research/bias_gate_materiality.md (2026-09-11 addendum).
# ---------------------------------------------------------------------------

def _holdout(n, bias=0.0, noise=0.30, spread=1.0, seed=11):
    """Holdout with a known TRUE median bias: the model reads `bias` high."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, spread, n)
    y = np.exp(mu + rng.normal(0.0, noise, n))
    y_pred = np.exp(mu) * (1.0 + bias)
    return y, y_pred


def _holdout_exact(n, offset_pct=0.0, spread_pct=25.0):
    """Deterministic holdout: residuals evenly spread around a known median.

    Seeded random draws make a status assertion hostage to one lucky sample,
    so the state tests use an exact construction instead — the median is
    `offset_pct` by design and the interval width follows from `spread_pct`.
    """
    ratios = np.linspace(-spread_pct, spread_pct, n) / 100.0
    y_pred = np.full(n, 100.0)
    y = y_pred * (1.0 + offset_pct / 100.0) * (1.0 + ratios)
    return y, y_pred


def _rate(n, bias, noise, verdict, draws=300, r2=0.85):
    """How often a model of this shape earns `verdict`."""
    hits = 0
    for seed in range(draws):
        b = quality.bias_assessment(*_holdout(n, bias=bias, noise=noise, seed=seed))
        hits += quality.assess_model(r2, None, n, bias=b)["verdict"] == verdict
    return hits / draws


# --- the four states -------------------------------------------------------

def test_bias_assessment_certifies_a_big_bias_on_a_big_holdout():
    b = quality.bias_assessment(*_holdout(4000, bias=0.20))
    assert b["status"] == "material"
    assert b["median_pct"] < -10          # reads 20% high -> actuals land above


def test_bias_assessment_certifies_an_unbiased_model_on_a_big_holdout():
    b = quality.bias_assessment(*_holdout(4000, bias=0.0))
    assert b["status"] == "immaterial"
    assert abs(b["median_pct"]) < quality.MATERIAL_BIAS_PCT


def test_thin_holdout_with_a_small_estimate_reads_likely_immaterial():
    # The common case for P2Predict users: 100-300 parts, so a 20-60 part
    # holdout. Can't certify, but the best estimate says the lean is small.
    b = quality.bias_assessment(*_holdout_exact(30, offset_pct=0.0))
    assert b["status"] == "likely_immaterial"
    assert abs(b["median_pct"]) < quality.MATERIAL_BIAS_PCT
    # can't certify: the blind spot is wider than the band
    assert b["resolution_pct"] > quality.MATERIAL_BIAS_PCT


def test_thin_holdout_with_a_big_estimate_reads_likely_material():
    b = quality.bias_assessment(*_holdout_exact(20, offset_pct=15.0))
    assert b["status"] == "likely_material"
    assert abs(b["median_pct"]) >= quality.MATERIAL_BIAS_PCT


def test_holdout_too_small_for_an_interval_is_unmeasured():
    b = quality.bias_assessment(*_holdout(4))
    assert b["status"] == "unmeasured"
    assert b["ci_pct"] is None and b["resolution_pct"] is None


def test_bias_assessment_handles_empty_and_degenerate_input():
    assert quality.bias_assessment([], [])["status"] == "unmeasured"
    # a zero prediction can't produce a relative residual; it must not divide
    assert quality.bias_assessment([1.0, 2.0], [0.0, 2.0])["status"] == "unmeasured"


def test_bias_verdict_is_deterministic():
    # A user-facing verdict must not depend on an RNG seed, which is why the
    # interval is built from order statistics rather than a bootstrap.
    y, p = _holdout(500, bias=0.08)
    assert quality.bias_assessment(y, p) == quality.bias_assessment(y, p)


def test_median_bias_is_invariant_to_the_denominator_convention():
    # The mean residual swings from -27% to +9% on the used-car model depending
    # on whether you divide by actual or predicted. The median must not move.
    y, p = _holdout(4000, bias=0.06)
    med_pred_denom = quality.median_relative_residual_ci(y, p)[0]
    med_actual_denom = float(np.median((y - p) / y * 100))
    assert med_pred_denom == pytest.approx(
        100 * (1 / (1 - med_actual_denom / 100) - 1), rel=1e-6)


# --- the properties the change exists to guarantee -------------------------

def test_a_resolvable_material_bias_is_never_green_lit():
    """Once the holdout is big enough to resolve the band, a materially biased
    model must never read 'trustworthy'."""
    for n in (250, 500, 1000, 4000):
        b = quality.bias_assessment(*_holdout(n, bias=0.10))
        v = quality.assess_model(0.85, None, n, bias=b)["verdict"]
        assert v != "trustworthy", f"n={n} green-lit a 10%-biased model"


def test_small_holdout_green_light_rate_is_bounded_and_beats_the_old_gate():
    """The accepted trade-off, guarded.

    Letting the point estimate grant a pass means some genuinely biased small
    models slip through — that is the price of not telling 90% of users
    'unknown'. It must stay far below the legacy gate, which green-lights a
    10%-biased 30-part holdout roughly three quarters of the time.
    """
    new = _rate(30, bias=0.10, noise=0.30, verdict="trustworthy")
    legacy = 0
    for seed in range(300):
        y, p = _holdout(30, bias=0.10, noise=0.30, seed=seed)
        legacy += quality.assess_model(
            0.85, quality.residual_bias_p(y, p), 30)["verdict"] == "trustworthy"
    legacy /= 300
    assert new < 0.35, f"too many biased small models green-lit: {new:.0%}"
    assert new < legacy / 2, f"no better than the old gate ({new:.0%} vs {legacy:.0%})"


def test_a_good_small_model_usually_passes():
    """The reason the four states exist. A clean 30-part holdout on an unbiased
    model must usually earn a verdict, not a flat 'unknown' — under the strict
    three-state gate this was ~1%."""
    assert _rate(30, bias=0.0, noise=0.15, verdict="trustworthy") > 0.60


def test_verdict_does_not_invert_as_the_holdout_grows():
    """The original complaint. Hold the model fixed, vary only n: an unbiased
    model must never be called unreliable at any holdout size."""
    seen = set()
    for n in (50, 100, 250, 500, 1000, 2500, 8000):
        b = quality.bias_assessment(*_holdout(n, bias=0.0))
        seen.add(quality.assess_model(0.85, None, n, bias=b)["verdict"])
    assert "unreliable" not in seen, f"unbiased model flagged somewhere: {seen}"
    assert "trustworthy" in seen, f"never earned a pass at any size: {seen}"


# --- what the user is actually told ---------------------------------------

def test_a_pass_on_a_thin_holdout_states_the_blind_spot():
    b = quality.bias_assessment(*_holdout_exact(30, offset_pct=0.0))
    a = quality.assess_model(0.85, None, 30, bias=b)
    assert a["verdict"] == "trustworthy"
    assert f"{abs(b['resolution_pct']):.0f}%" in a["headline"]
    # the bound bites on aggregates, where a systematic lean doesn't cancel
    assert "basket" in a["headline"]
    _assert_clean(a["headline"])


def test_likely_material_warns_with_a_direction_but_does_not_condemn():
    b = quality.bias_assessment(*_holdout_exact(20, offset_pct=15.0))
    a = quality.assess_model(0.85, None, 20, bias=b)
    assert a["verdict"] == "unknown"        # warned, not condemned
    assert "looks like it reads" in a["headline"]
    assert "not enough to be sure" in a["headline"]
    _assert_clean(a["headline"])


def test_unreliable_headline_says_which_way_and_how_far():
    b = quality.bias_assessment(*_holdout(4000, bias=0.20))
    a = quality.assess_model(0.85, None, 4000, bias=b)
    assert a["verdict"] == "unreliable"
    assert "low" in a["headline"]
    _assert_clean(a["headline"])


def test_new_headlines_are_jargon_free():
    for n, bs, nz in [(30, 0.0, 0.15), (20, 0.18, 0.30), (4000, 0.0, 0.30),
                      (4000, 0.20, 0.30), (8, 0.0, 0.30), (500, 0.02, 0.30)]:
        b = quality.bias_assessment(*_holdout(n, bias=bs, noise=nz))
        for r2 in (0.95, 0.70, 0.40):
            _assert_clean(quality.assess_model(r2, None, n, bias=b)["headline"])


# --- plumbing --------------------------------------------------------------

def test_assess_model_without_bias_keeps_the_legacy_behaviour():
    legacy = quality.assess_model(r2=0.85, residual_bias_p=1e-6, n_holdout=60)
    assert legacy["verdict"] == "unreliable"
    assert quality.assess_model(0.85, 0.5, 60)["verdict"] == "trustworthy"
    assert "bias_resolution_pct" not in legacy


def test_quality_report_uses_the_equivalence_gate():
    y_test, y_pred = _holdout(400, bias=0.0)
    loaded = {"holdout_y_test": y_test, "holdout_y_pred": y_pred,
              "features": ["a"], "target_feature": "price"}
    rep = quality.build_quality_report(loaded, include_metrics=True)
    assert rep["assessment"]["verdict"] in ("trustworthy", "usable", "unknown")
    assert rep["metrics"]["bias_status"] in (
        "immaterial", "likely_immaterial", "likely_material", "unmeasured")
    # the business-safe default must not leak the raw interval
    assert "bias_status" not in quality.build_quality_report(loaded)["metrics"]
