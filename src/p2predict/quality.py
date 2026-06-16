"""Centralised model-quality judgment — the single source of truth for how
P2Predict decides whether a trained model is trustworthy, and at what
granularity.

Why this module exists: the quality verdict used to be an R²-only label
duplicated in the trainer and unavailable to agents in any structured form.
The interpretation that actually matters for procurement (is it unbiased?
which price bands can I trust? which features are quotable?) lived only as
prose. This module makes that judgment *computed and shared*: the trainer,
the CLI, and the MCP ``get_model_quality`` / ``generate_report`` tools all
read their thresholds from here, so the verdict never drifts between
surfaces.

All thresholds are in one place below so they can be audited and tuned.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Thresholds — the one auditable place
# ---------------------------------------------------------------------------

# Overall R²-based accuracy label. Kept identical to the historical trainer
# label for continuity (composite = r2 * 100).
R2_EXCELLENT = 80.0
R2_GOOD = 60.0

# Residual-bias one-sample t-test: a p-value ABOVE this means the model is not
# flagged as systematically high or low. For procurement this matters more
# than R²: an unbiased modest model is usable; a biased accurate one is not.
UNBIASED_P = 0.05

# Per-price-band calibration, judged on median absolute % error in the band.
BAND_TRUST_MAX_PCT = 15.0      # ≤ this  → benchmark with confidence
BAND_CAUTION_MAX_PCT = 40.0    # ≤ this  → usable, sanity-check the number
#                                > this  → get a quote, don't benchmark

# Per-feature signal strength, judged on share of total importance (%).
FEATURE_STRONG_MIN_PCT = 10.0   # ≥ this → quotable to a stakeholder
FEATURE_MODERATE_MIN_PCT = 3.0  # ≥ this → directional
#                                 < this → weak / likely under-sampled


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def r2_quality_label(r2: float) -> str:
    """The headline accuracy label from R² (Excellent / Good / Needs Improvement)."""
    composite = min(max(r2, 0.0), 1.0) * 100
    if composite > R2_EXCELLENT:
        return "Excellent"
    if composite > R2_GOOD:
        return "Good"
    return "Needs Improvement"


def band_reliability(median_ape: float) -> str:
    """Per-band verdict: 'trust' | 'caution' | 'quote' from median % error."""
    if median_ape <= BAND_TRUST_MAX_PCT:
        return "trust"
    if median_ape <= BAND_CAUTION_MAX_PCT:
        return "caution"
    return "quote"


def feature_signal(importance_pct: float) -> str:
    """Per-feature verdict: 'strong' | 'moderate' | 'weak' from importance share."""
    if importance_pct >= FEATURE_STRONG_MIN_PCT:
        return "strong"
    if importance_pct >= FEATURE_MODERATE_MIN_PCT:
        return "moderate"
    return "weak"


def assess_model(r2: float, residual_bias_p: float | None) -> dict:
    """Multi-factor honest read combining accuracy AND bias.

    This is the judgment the R²-only label can't make: it's why a modest-R²
    but unbiased model is correctly described as usable rather than just
    'Needs Improvement'.
    """
    label = r2_quality_label(r2)
    accuracy = {
        "Excellent": "excellent", "Good": "good", "Needs Improvement": "modest",
    }[label]
    unbiased = residual_bias_p is not None and not np.isnan(residual_bias_p) \
        and residual_bias_p > UNBIASED_P

    if unbiased and accuracy == "modest":
        headline = (
            "Modest accuracy but statistically unbiased — usable as a benchmark "
            "and for relative comparisons (supplier premiums, what-ifs), not as a "
            "single-part appraisal. Lean on the interval and SHAP, not the bare "
            "point estimate."
        )
    elif unbiased:
        headline = (
            f"{accuracy.capitalize()} accuracy and statistically unbiased — "
            "trustworthy for benchmarking."
        )
    else:
        headline = (
            f"{accuracy.capitalize()} accuracy but residuals look biased "
            "(systematically high or low) — treat point estimates with caution "
            "and prefer relative comparisons."
        )
    return {
        "quality_label": label,
        "accuracy": accuracy,
        "unbiased": bool(unbiased),
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# Pure stats — canonical home (plotting.py imports these so the PDF and the
# JSON report compute identical numbers)
# ---------------------------------------------------------------------------

def abs_pct_errors(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_test != 0
    if not mask.any():
        return np.array([])
    return np.abs(y_test[mask] - y_pred[mask]) / np.abs(y_test[mask]) * 100.0


def summary_metrics(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_test - y_pred
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(r2_score(y_test, y_pred))
    ape = abs_pct_errors(y_test, y_pred)
    if ape.size:
        mape = float(np.mean(ape))
        median_ape = float(np.median(ape))
        p90_ape = float(np.quantile(ape, 0.9))
    else:
        mape = median_ape = p90_ape = float("nan")
    return {
        "n_test": int(len(y_test)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ape": median_ape,
        "p90_ape": p90_ape,
    }


def error_by_price_band(y_test, y_pred, n_bins=10):
    """Bucket holdout points by actual-price quantile; return median APE per bucket.

    Returns (labels, median_apes, counts) or None if data is too thin to bin.
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_test) < n_bins:
        return None
    edges = np.unique(np.quantile(y_test, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    bin_idx = np.clip(np.digitize(y_test, edges[1:-1]), 0, len(edges) - 2)
    labels, medians, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ape = abs_pct_errors(y_test[mask], y_pred[mask])
        if ape.size == 0:
            continue
        labels.append(f"{edges[b]:,.0f}–{edges[b + 1]:,.0f}")
        medians.append(float(np.median(ape)))
        counts.append(n)
    if not labels:
        return None
    return labels, medians, counts


# ---------------------------------------------------------------------------
# Full structured report — the agent-readable form of the PDF
# ---------------------------------------------------------------------------

def residual_bias_p(y_test, y_pred) -> float:
    """One-sample t-test of residuals against zero. NaN if too few points."""
    from scipy.stats import ttest_1samp

    resid = np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float)
    if resid.size < 2:
        return float("nan")
    return float(ttest_1samp(resid, 0.0).pvalue)


def build_quality_report(loaded: dict, importances=None) -> dict:
    """Assemble the structured quality report from a loaded model dict.

    ``loaded`` is the dict returned by ``load_model`` (needs
    ``holdout_y_test`` / ``holdout_y_pred``). ``importances`` is an optional
    list of ``(feature, value)`` pairs. Raises ``ValueError("no_holdout_data")``
    when the holdout isn't stored (older models).
    """
    y_test = loaded.get("holdout_y_test")
    y_pred = loaded.get("holdout_y_pred")
    if y_test is None or y_pred is None:
        raise ValueError("no_holdout_data")

    metrics = summary_metrics(y_test, y_pred)
    bias_p = residual_bias_p(y_test, y_pred)
    assessment = assess_model(metrics["r2"], bias_p)

    bands = error_by_price_band(y_test, y_pred)
    band_block = []
    if bands:
        for label, med, n in zip(*bands):
            band_block.append({
                "band": label,
                "median_pct_error": round(med, 1),
                "n": n,
                "reliability": band_reliability(med),
            })

    fi_block = []
    if importances:
        total = sum(abs(float(v)) for _, v in importances) or 1.0
        for name, value in importances:
            pct = abs(float(value)) / total * 100
            fi_block.append({
                "feature": name,
                "importance_pct": round(pct, 1),
                "signal": feature_signal(pct),
            })

    return {
        "provenance": {
            "target": loaded.get("target_feature"),
            "algorithm": loaded.get("model_name"),
            "log_target": loaded.get("log_target"),
            "features": loaded.get("features"),
            "n_features": len(loaded.get("features") or []),
            "training_date": loaded.get("training_date"),
        },
        "metrics": {
            "r2": round(metrics["r2"], 4),
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "median_pct_error": round(metrics["median_ape"], 1),
            "mape": round(metrics["mape"], 1),
            "p90_pct_error": round(metrics["p90_ape"], 1),
            "residual_bias_p_value": round(bias_p, 4),
            "n_holdout": metrics["n_test"],
            "quality_label": assessment["quality_label"],
        },
        "assessment": assessment,
        "calibration_by_price_band": band_block,
        "feature_importance": fi_block,
    }
