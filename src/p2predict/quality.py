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
#
# RETAINED FOR THE LEGACY PATH ONLY. A p-value answers "could this bias be
# noise?", whose answer depends on sample size — not "is this bias big enough
# to matter?", which is what the verdict claims to report. Measured, the old
# gate flagged a truly unbiased model 37% of the time and waved through a
# materially biased one 42% of the time. The equivalence test below replaces
# it; see research/bias_gate_materiality.md (2026-09-11 addendum).
UNBIASED_P = 0.05

# --- bias materiality (the equivalence test) -------------------------------
# A category manager benchmarks in order to negotiate, and typical negotiated
# movement is 2–5%. A systematic offset as large as the saving being chased
# corrupts the decision; one well under it does not. Within that normative
# range the sweep picks the loosest band that never green-lights a materially
# biased model, since widening buys usability (less 'unknown') and tightening
# buys nothing once the miss rate is already zero.
MATERIAL_BIAS_PCT = 5.0

# Confidence level for the bias interval. The interval is built from ORDER
# STATISTICS, not a bootstrap: a bootstrap would make a user-facing verdict
# depend on an RNG seed — the same model judged differently on two runs.
BIAS_CI_LEVEL = 0.95

# Per-price-band calibration, judged on median absolute % error in the band.
BAND_TRUST_MAX_PCT = 15.0      # ≤ this  → benchmark with confidence
BAND_CAUTION_MAX_PCT = 40.0    # ≤ this  → usable, sanity-check the number
#                                > this  → get a quote, don't benchmark

# Per-feature signal strength, judged on share of total importance (%).
FEATURE_STRONG_MIN_PCT = 10.0   # ≥ this → quotable to a stakeholder
FEATURE_MODERATE_MIN_PCT = 3.0  # ≥ this → directional
#                                 < this → weak / likely under-sampled

# How many holdout points we need before a quality verdict means anything.
MIN_HOLDOUT_FOR_JUDGMENT = 15   # below → "insufficient_data" verdict
HIGH_CONFIDENCE_MIN_N = 50      # at/above → "high" confidence; between → "limited"

# Per-band: fewer than this many points → the band's median % error is too
# noisy to act on, so we flag it rather than dress it up as a verdict.
MIN_BAND_N = 5

# Per-PART interval trust, judged on the band width relative to the prediction
# (full width / prediction). This is the model's confidence on THIS specific
# part, separate from the model-wide verdict.
INTERVAL_TIGHT_MAX_FRAC = 0.30   # ≤ this  → benchmark with confidence
INTERVAL_WIDE_MAX_FRAC = 0.80    # ≤ this  → usable, sanity-check the number
#                                  > this  → get a quote, don't benchmark


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


def band_say_to_user(band: str, reliability: str, low_confidence: bool = False) -> str:
    """Plain sentence a category manager understands — no 'median % error'."""
    phrase = {
        "trust": (
            f"For parts priced around {band}, you can benchmark against this "
            "model with confidence."
        ),
        "caution": (
            f"For parts priced around {band}, use the estimate as a guide and "
            "sanity-check it before you negotiate on it."
        ),
        "quote": (
            f"For parts priced around {band}, the model is shaky — get a real "
            "quote rather than benchmarking off it."
        ),
    }[reliability]
    if low_confidence:
        phrase += " (Very few parts in this price range, so even this is rough.)"
    return phrase


def interval_reliability(low: float, prediction: float, high: float) -> str:
    """Per-part verdict from the likely-range: 'trust' | 'caution' | 'quote'.

    A lower bound at/below $0 (an additive model underwater on a cheap part) is
    always 'quote' — a price can't be negative, so the floor is meaningless and
    the part needs a real quote.
    """
    if prediction <= 0 or low <= 0:
        return "quote"
    frac = (high - low) / prediction
    if frac <= INTERVAL_TIGHT_MAX_FRAC:
        return "trust"
    if frac <= INTERVAL_WIDE_MAX_FRAC:
        return "caution"
    return "quote"


def interval_say_to_user(low: float, prediction: float, high: float) -> str:
    """Plain sentence on how far to trust THIS part's estimate — no 'conformal'."""
    if prediction > 0 and low <= 0:
        return (
            "The likely-range dips to or below $0, which can't be a real price — "
            "get a quote for this part rather than benchmarking, and the model "
            "should be rebuilt on a percentage scale so cheap parts stay positive."
        )
    reliability = interval_reliability(low, prediction, high)
    return {
        "trust": (
            "Tight range — you can benchmark against this number with confidence."
        ),
        "caution": (
            "Moderate range — usable as a guide, but sanity-check it before you "
            "hold a supplier to it."
        ),
        "quote": (
            "Wide range — the model is genuinely unsure on this part; get a real "
            "quote rather than benchmarking off this number."
        ),
    }[reliability]


def feature_say_to_user(feature: str, signal: str) -> str:
    """Plain sentence on how much weight to put on a driver — no 'importance share'."""
    return {
        "strong": (
            f"'{feature}' is a major price driver here — solid enough to quote "
            "in a negotiation."
        ),
        "moderate": (
            f"'{feature}' moves the price somewhat — treat it as directional, "
            "not a hard number."
        ),
        "weak": (
            f"'{feature}' barely moves the price in this data — treat any "
            "finding about it as a hypothesis, not a number to negotiate against."
        ),
    }[signal]


def confidence_for(n_holdout: int) -> str:
    """How much to trust the quality verdict, from holdout size."""
    if n_holdout < MIN_HOLDOUT_FOR_JUDGMENT:
        return "insufficient"
    if n_holdout < HIGH_CONFIDENCE_MIN_N:
        return "limited"
    return "high"


def median_relative_residual_ci(y_test, y_pred, level: float = BIAS_CI_LEVEL):
    """Median relative residual and a distribution-free CI for it, in percent.

    The relative residual is ``y/ŷ - 1`` — what the model is off by on a part,
    as a fraction of what it predicted. Three reasons this is the quantity to
    judge rather than the mean dollar residual the t-test uses:

    * **It is the log-space test, without a branch.** The median of ``y/ŷ - 1``
      is a monotone transform of the median log residual, so one rule covers
      log-target and additive models alike.
    * **It doesn't move when the convention does.** The mean residual on the
      used-car model reads −27.2% or +8.8% depending on whether you divide by
      ``y`` or ``ŷ``. The median is +1.7% either way.
    * **It is what a log-target model actually estimates.** ``exp(E[log Y])``
      is the geometric mean, which sits near the median — so a median-unbiased
      model was being failed for a mean it never claimed to estimate.

    The interval is the ``(k_lo, k_hi)`` order statistics whose binomial tail
    mass falls below ``(1-level)/2`` either side of the median: exact,
    assumption-free, deterministic, and O(n log n).

    Returns ``(median_pct, lo_pct, hi_pct)``. The bounds are NaN when the
    holdout is too small for any interval to exist at this level (n < 6 at
    95%), in which case the caller must not claim to have measured bias.
    """
    from scipy.stats import binom

    yt = np.asarray(y_test, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp) & (yp != 0)
    rel = (yt[ok] / yp[ok] - 1.0) * 100.0
    n = rel.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    median = float(np.median(rel))
    alpha = 1.0 - level
    k_lo = int(binom.ppf(alpha / 2.0, n, 0.5)) - 1   # 0-indexed
    k_hi = int(binom.isf(alpha / 2.0, n, 0.5))
    if k_lo < 0 or k_hi > n - 1:
        # Too few points for the interval to exist; the median still reports.
        return median, float("nan"), float("nan")
    srt = np.sort(rel)
    return median, float(srt[k_lo]), float(srt[k_hi])


def bias_assessment(y_test, y_pred, band_pct: float = MATERIAL_BIAS_PCT) -> dict:
    """Is this model's bias materially large? An equivalence test.

    The old gate asked "could the bias be zero?" and read *failure to reject*
    as *proof of no bias*. Those are different claims and the gap between them
    is exactly sample size, so the verdict tracked the holdout rather than the
    model. This asks the question the verdict claims to answer — "is the bias
    big enough to matter?" — by placing the interval against a band:

    ============================  ====================  ======================
    Median CI vs ±band            ``status``            Meaning
    ============================  ====================  ======================
    entirely inside               ``immaterial``        confidently small
    straddles, estimate inside    ``likely_immaterial`` probably small
    straddles, estimate outside   ``likely_material``   probably large
    entirely outside              ``material``          confidently large
    no interval possible          ``unmeasured``        nothing to say
    ============================  ====================  ======================

    The two ``likely_`` states exist because "we cannot certify" is not the
    same claim as "we know nothing", and collapsing them loses a signal that is
    usually right. Measured on a 60-part holdout at 15% typical error, when the
    interval straddles the band the point estimate still lands on the correct
    side **95%** of the time. Most P2Predict users have 100–300 parts (a 20–60
    part holdout), so this distinction is the common case, not an edge one —
    reporting it all as one flat ``unknown`` would hand a good model and a bad
    one the identical verdict.

    ``resolution_pct`` is the interval's half-width: the smallest offset this
    holdout could have detected, i.e. the size of the blind spot. It is what
    keeps the ``likely_`` states honest — a best estimate is reported *with*
    the precision behind it.
    """
    median, lo, hi = median_relative_residual_ci(y_test, y_pred)
    if np.isnan(median) or np.isnan(lo) or np.isnan(hi):
        return {
            "status": "unmeasured",
            "median_pct": None if np.isnan(median) else round(median, 1),
            "ci_pct": None,
            "resolution_pct": None,
            "band_pct": band_pct,
        }
    if lo > band_pct or hi < -band_pct:
        status = "material"
    elif lo > -band_pct and hi < band_pct:
        status = "immaterial"
    elif abs(median) < band_pct:
        status = "likely_immaterial"
    else:
        status = "likely_material"
    return {
        "status": status,
        "median_pct": round(median, 1),
        "ci_pct": [round(lo, 1), round(hi, 1)],
        "resolution_pct": round((hi - lo) / 2.0, 1),
        "band_pct": band_pct,
    }


def assess_model(
    r2: float,
    residual_bias_p: float | None,
    n_holdout: int,
    *,
    bias: dict | None = None,
) -> dict:
    """Bias- AND confidence-aware overall verdict.

    The headline `verdict` is the thing to lead with — it folds in three
    questions the R²-only label can't answer:
      1. Is there enough data to judge at all?  ('insufficient_data')
      2. Could we even measure bias?            ('unknown')
      3. Is the bias big enough to matter?      ('unreliable')
    Only a model whose bias is measurably small earns 'trustworthy'
    (good/excellent accuracy) or 'usable' (modest accuracy — fine for
    benchmarking, not for a single-part appraisal).

    ``bias`` is the dict from :func:`bias_assessment` and is how question 3
    should be answered. When it is omitted the legacy ``residual_bias_p`` gate
    is used instead, which measures the holdout size as much as the bias — it
    remains only so older callers keep working, and question 3 above then reads
    "are the residuals distinguishable from zero?", which is not the same
    question. Prefer passing ``bias``.
    """
    label = r2_quality_label(r2)
    accuracy = {
        "Excellent": "excellent", "Good": "good", "Needs Improvement": "modest",
    }[label]
    confidence = confidence_for(n_holdout)
    if bias is not None:
        status = bias.get("status")
        # 'likely_immaterial' counts as even-handed: the best estimate says the
        # lean is small, and the blind spot is reported alongside it rather
        # than swallowing the verdict. 'likely_material' does NOT get promoted
        # to 'unreliable' -- at a 20-60 part holdout the estimate alone is too
        # noisy to condemn a model on, and doing so would rebuild the old gate
        # in reverse. It lands on 'unknown' carrying a directional warning.
        bias_known = status in ("material", "immaterial", "likely_immaterial")
        unbiased = status in ("immaterial", "likely_immaterial")
    else:
        bias_known = residual_bias_p is not None and not np.isnan(residual_bias_p)
        unbiased = bias_known and residual_bias_p > UNBIASED_P

    # NOTE: `headline` is quoted to the user (often verbatim by a weaker agent),
    # so it must stay in plain procurement language — no 'SHAP', 'R²', 'holdout',
    # 'residual', 'p-value', 'log-target'. The banned-term test in
    # tests/test_quality.py enforces this; keep it green.
    if confidence == "insufficient":
        verdict = "insufficient_data"
        headline = (
            f"Only {n_holdout} part(s) were kept back to check this model — too "
            "few to judge it reliably. Treat the numbers below as rough, and "
            "gather more price history before benchmarking against it."
        )
    elif not bias_known:
        verdict = "unknown"
        resolution = (bias or {}).get("resolution_pct")
        if (bias or {}).get("status") == "likely_material":
            median_pct = bias.get("median_pct") or 0.0
            way = "low" if median_pct > 0 else "high"
            headline = (
                f"This model looks like it reads about {abs(median_pct):.0f}% {way} "
                f"on a typical part, but only {n_holdout} part(s) were kept back "
                "to check it — not enough to be sure either way. Use it to compare "
                "options, and get a quote before you set an absolute target."
            )
        elif resolution is not None:
            # Quantified, not a shrug: we know how small an offset would have
            # escaped notice, so say it. This is the whole reason the honest
            # 'unknown' beats the old gate's false confidence.
            headline = (
                f"Only {n_holdout} part(s) were kept back to check this model — "
                f"enough to rule out a systematic error bigger than about "
                f"{abs(resolution):.0f}%, but not enough to prove it runs even. "
                "Treat the number as indicative, and sanity-check it against a "
                "quote before you set a target on it."
            )
        else:
            headline = (
                "Couldn't tell whether this model runs systematically high or "
                "low, so how far to trust it is unknown — judge with caution "
                "and stick to relative comparisons."
            )
    elif not unbiased:
        verdict = "unreliable"
        median_pct = (bias or {}).get("median_pct")
        direction = ""
        if median_pct:
            # Signed on the RELATIVE residual: a positive median means the
            # actual price came in above what the model said, i.e. it reads low.
            way = "low" if median_pct > 0 else "high"
            direction = f" — it reads about {abs(median_pct):.0f}% {way} on a typical part"
        headline = (
            f"{accuracy.capitalize()} accuracy, but the model runs systematically "
            f"high or low{direction}. Its single-number estimates aren't "
            "trustworthy; use it only to compare options, not to set an absolute "
            "target."
        )
    elif accuracy in ("excellent", "good"):
        verdict = "trustworthy"
        headline = (
            f"{accuracy.capitalize()} accuracy and even-handed (it doesn't run "
            "systematically high or low) — trustworthy to benchmark against."
        )
    else:
        verdict = "usable"
        headline = (
            "Modest accuracy but even-handed (it doesn't run systematically high "
            "or low) — usable as a benchmark and for comparisons like supplier "
            "premiums and what-ifs, but not to appraise a single part. Lean on the "
            "likely-range and the price-driver breakdown, not the bare single "
            "number."
        )

    # A pass earned on a best estimate rather than a proof must say how big an
    # error could still be hiding. Random error washes out across a basket;
    # a systematic lean does not, which is where this bound actually bites.
    if verdict in ("trustworthy", "usable") and (bias or {}).get(
            "status") == "likely_immaterial" and bias.get("resolution_pct"):
        headline += (
            f" Only {n_holdout} part(s) were kept back to check it, so an error "
            f"up to about {abs(bias['resolution_pct']):.0f}% could still be hiding "
            "— fine for comparing options and pricing single parts against the "
            "likely-range, but sanity-check a whole-basket target against a quote."
        )
    elif verdict in ("trustworthy", "usable") and confidence == "limited":
        headline += (
            f" (Based on only {n_holdout} parts kept back for checking — treat as "
            "indicative.)"
        )

    out = {
        "verdict": verdict,
        "quality_label": label,
        "accuracy": accuracy,
        "unbiased": bool(unbiased) if bias_known else None,
        "confidence": confidence,
        "headline": headline,
    }
    if bias is not None:
        # Business-safe numbers: a percentage a buyer feels, not a test statistic.
        out["typical_bias_pct"] = bias.get("median_pct")
        out["bias_resolution_pct"] = bias.get("resolution_pct")
        out["material_bias_threshold_pct"] = bias.get("band_pct")
    return out


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


def build_quality_report(
    loaded: dict, importances=None, include_holdout=False, include_metrics=False
) -> dict:
    """Assemble the structured quality report from a loaded model dict.

    ``loaded`` is the dict returned by ``load_model`` (needs
    ``holdout_y_test`` / ``holdout_y_pred``). ``importances`` is an optional
    list of ``(feature, value)`` pairs. Set ``include_holdout=True`` to attach
    the raw actual/predicted arrays so a caller can draw its own charts.
    Raises ``ValueError("no_holdout_data")`` when the holdout isn't stored.

    ``include_metrics`` (default False) gates the raw statistics — R²,
    p-value, MAE/RMSE, algorithm name, log-target flag. They stay OUT of the
    default payload on purpose: a weaker agent surfaces whatever fields it
    sees, so the default carries only business-safe numbers (typical % error,
    how many parts were checked) plus the plain-language assessment. Pass
    ``include_metrics=True`` for the developer/debug view.
    """
    y_test = loaded.get("holdout_y_test")
    y_pred = loaded.get("holdout_y_pred")
    if y_test is None or y_pred is None:
        raise ValueError("no_holdout_data")

    metrics = summary_metrics(y_test, y_pred)
    n_holdout = metrics["n_test"]
    bias_p = residual_bias_p(y_test, y_pred)
    bias = bias_assessment(y_test, y_pred)
    assessment = assess_model(metrics["r2"], bias_p, n_holdout, bias=bias)

    bands = error_by_price_band(y_test, y_pred)
    band_block = []
    calibration_note = None
    if bands:
        for label, med, n in zip(*bands):
            reliability = band_reliability(med)
            low_conf = n < MIN_BAND_N
            entry = {
                "band": label,
                "median_pct_error": round(med, 1),
                "n": n,
                "reliability": reliability,
                "say_to_user": band_say_to_user(label, reliability, low_conf),
            }
            if low_conf:
                # Honest about thin bands: the verdict is computed but shaky.
                entry["low_confidence"] = True
                entry["note"] = (
                    f"only {n} part(s) in this price range — this read is noisy"
                )
            band_block.append(entry)
    else:
        calibration_note = (
            "Too few parts were kept back to break reliability down by price "
            "range — no per-range read available."
        )

    fi_block = []
    if importances:
        total = sum(abs(float(v)) for _, v in importances) or 1.0
        for name, value in importances:
            pct = abs(float(value)) / total * 100
            signal = feature_signal(pct)
            fi_block.append({
                "feature": name,
                "importance_pct": round(pct, 1),
                "signal": signal,
                "say_to_user": feature_say_to_user(name, signal),
            })
    feature_note = None if fi_block else (
        "Feature importance unavailable for this model."
    )

    # Business-safe by default. provenance/metrics carry only what a category
    # manager can hear; the raw statistics are added only when include_metrics.
    provenance = {
        "target": loaded.get("target_feature"),
        "features": loaded.get("features"),
        "n_features": len(loaded.get("features") or []),
        "training_date": loaded.get("training_date"),
    }
    metrics_block = {
        "typical_pct_error": round(metrics["median_ape"], 1),
        "parts_checked_against": n_holdout,
        "quality_label": assessment["quality_label"],
    }
    if include_metrics:
        provenance["algorithm"] = loaded.get("model_name")
        provenance["log_target"] = loaded.get("log_target")
        metrics_block.update({
            "r2": round(metrics["r2"], 4),
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "mape": round(metrics["mape"], 1),
            "p90_pct_error": round(metrics["p90_ape"], 1),
            "residual_bias_p_value": (
                round(bias_p, 4) if not np.isnan(bias_p) else None
            ),
            "median_relative_residual_pct": bias["median_pct"],
            "median_relative_residual_ci_pct": bias["ci_pct"],
            "bias_status": bias["status"],
        })

    report = {
        "provenance": provenance,
        "metrics": metrics_block,
        "assessment": assessment,
        "calibration_by_price_band": band_block,
        "calibration_note": calibration_note,
        "feature_importance": fi_block,
        "feature_note": feature_note,
    }
    if include_holdout:
        # Raw points so an agent with a plotting/code tool can draw its own
        # charts (predicted-vs-actual, residuals, error-by-band, ...).
        report["holdout"] = {
            "y_actual": [float(v) for v in y_test],
            "y_predicted": [float(v) for v in y_pred],
        }
    return report
