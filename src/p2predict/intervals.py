"""Conformal prediction intervals for P2Predict models.

What this module computes
-------------------------
For each prediction, a "likely range" [low, high] that is mathematically
guaranteed to contain the true value with probability >= 1 - alpha (the
target coverage rate), under the assumption that future inputs come from
the same distribution as the training data.

This guarantee is what split conformal prediction provides — the interval
isn't just a heuristic ±2σ, it has a finite-sample coverage proof. The
proof rests on exchangeability of (X_test, y_test) with (X_future,
y_future): a much weaker assumption than the parametric normality
assumptions that classical prediction intervals rely on.

Algorithm: split conformal with the test set as calibration set
---------------------------------------------------------------
1. Train the model on the training split (already done by the time we
   get here).
2. Compute absolute residuals on the held-out test split:
       r_i = |y_test_i - model.predict(x_test_i)|         (or in log space for log-target)
3. For coverage 1 - alpha, the conformal threshold is the
       k-th smallest residual, where k = ceil((n + 1) * (1 - alpha))
   In numpy: ``np.quantile(residuals, q, method="higher")`` with q
   chosen to match.
4. At predict time:
       low, high = pred - q_hat, pred + q_hat            (additive intervals)
   For log-target models the calibration is done in log space, so the
   bounds transform via exp() to multiplicative intervals in price space:
       low, high = pred * exp(-q_hat), pred * exp(+q_hat)

Why use the test set for calibration instead of a separate split
----------------------------------------------------------------
The natural worry is double-dipping: "if we report R² on the test set
and then use the same test set residuals to calibrate intervals, are
the metrics still valid?" Yes. R² on the test set remains an unbiased
estimate of generalization on the underlying distribution; computing a
downstream statistic (the conformal quantile) from those same residuals
doesn't change that. The exchangeability assumption holds for (X_test,
y_test) ~ (X_future, y_future) regardless of what else we use the test
residuals for, as long as we don't select the model based on them.

The data-efficiency win is real: a separate calibration split would
shrink the training set by 16% in the standard 80/20 setup, slightly
degrading the model. Using test residuals avoids that.

Why we calibrate in log space when log-target is active
-------------------------------------------------------
Procurement prices vary by orders of magnitude. A $1 part and a $1,000
part should not get the same ± dollar interval — that would be useless
on the small end and reckless on the large. Calibrating absolute log-
residuals gives constant-width intervals in log space, which transform
to *multiplicative* intervals in price space:

    [pred * exp(-q_hat), pred * exp(+q_hat)]

Same percentage-width regardless of prediction magnitude. Procurement-
natural.

For non-log-target models we use absolute residuals in the target's
native units, giving constant-width additive intervals. That's the
right behaviour when the target is something like profit margin
(which can be negative and isn't bounded multiplicatively).

Banded (Mondrian) calibration
-----------------------------
A single global q_hat gives every prediction the same width, which lets the
noisiest segment of the data set the width for everyone: on a catalog whose
sub-$5 parts are near-random, the $200 parts inherit that noise in their
likely range. When the calibration set is large enough we therefore
partition it into bands by *predicted* value (terciles of the calibration
predictions) and compute a separate conformal quantile per band — Mondrian
conformal prediction. The coverage guarantee then holds *within each band*,
not just on average, because the banding rule depends only on the model's
prediction (a function of X), never on the calibration labels.

Fallbacks keep the old behaviour bit-for-bit:
  * calibration dicts saved by older versions (no "predictions" key),
  * calibration sets smaller than MIN_CALIBRATION_FOR_BANDING,
both produce the single global quantile exactly as before.

User-facing language
--------------------
The CLI and README deliberately avoid "confidence interval" (technically
wrong for prediction intervals anyway), "alpha", "conformal", and
"coverage". We use "likely range" and natural-frequency framing
("9 in 10 similar parts fall in this range"). Bands surface to users as
"calibrated on similar-priced parts". This module's docstrings keep the
technical names because the audience here is developers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.compose import TransformedTargetRegressor


# Banding thresholds. Three bands of >= 50 calibration points each keeps the
# per-band conformal quantile stable; below 150 total we stay global. Chosen
# so a standard 80/20 split bands from ~750 training rows upward.
N_BANDS = 3
MIN_CALIBRATION_FOR_BANDING = 150


@dataclass
class IntervalResult:
    """One prediction with its likely range.

    Attributes are named for user-facing rendering: ``low`` and ``high``
    are in the same units as ``prediction``, regardless of whether the
    underlying model used a log-target transform.

    ``band`` is a human-readable description of the calibration band the
    width came from (e.g. ``"predicted 5.20 to 155.00"``), or ``None`` when
    the global quantile was used (old calibration data, or a calibration
    set too small to band).
    """

    low: float
    prediction: float
    high: float
    coverage: float  # the realised target coverage, e.g. 0.90 for 90%
    band: Optional[str] = None


def compute_calibration_residuals(model, X_test, y_test) -> dict:
    """Return the residuals to stash in the saved model.

    The dict shape is what gets persisted in model metadata under
    ``calibration``. We store the raw residuals (not a precomputed q_hat)
    so the user can pick any coverage level at predict time without
    retraining.

    For log-target models the residuals are in log space; the
    ``in_log_space`` flag tells the predict-time code to inverse-transform
    them multiplicatively.
    """
    y_test = np.asarray(y_test, dtype=float)
    is_log_target = isinstance(model, TransformedTargetRegressor)

    # We need the model's prediction in the user's target units regardless;
    # for log-target models we then take the log of both sides to get
    # log-space residuals.
    preds = np.asarray(model.predict(X_test), dtype=float)

    if is_log_target:
        # Guard against numerical zero/negative preds (shouldn't happen if
        # should_log_target gated correctly, but defend the math).
        with np.errstate(invalid="ignore", divide="ignore"):
            valid = (preds > 0) & (y_test > 0)
            residuals = np.abs(np.log(y_test[valid]) - np.log(preds[valid]))
        cal_preds = preds[valid]
        in_log_space = True
    else:
        residuals = np.abs(y_test - preds)
        cal_preds = preds
        in_log_space = False

    return {
        "residuals": residuals.tolist(),
        # Target-space predictions aligned 1:1 with `residuals`. New in the
        # banded-calibration version; lets predict_interval() partition the
        # calibration set by predicted value (Mondrian bands). Older models
        # without this key keep the global-quantile behaviour.
        "predictions": cal_preds.tolist(),
        "in_log_space": in_log_space,
        "n_calibration": int(len(residuals)),
    }


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """The k-th smallest residual that gives finite-sample coverage 1-alpha.

    Standard split-conformal quantile rule. Uses ``method="higher"`` so we
    err on the safe side (slightly wider interval) rather than the
    optimistic side when the order statistic falls between samples.
    """
    n = len(residuals)
    if n == 0:
        raise ValueError("Cannot calibrate with zero residuals.")
    # k / n quantile, where k = ceil((n + 1) * (1 - alpha)).
    # Clip to (0, 1] so np.quantile is well-defined for tiny n.
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def _build_bands(cal_preds: np.ndarray, residuals: np.ndarray, alpha: float):
    """Partition the calibration set into N_BANDS by predicted value and
    return ``(edges, band_q_hats, band_labels)``, or ``None`` when banding
    isn't justified (too few calibration points, or degenerate predictions
    that collapse the band edges).

    The banding rule uses only the model's predictions — a function of X —
    so the split-conformal coverage guarantee holds within each band.
    """
    n = len(cal_preds)
    if n < MIN_CALIBRATION_FOR_BANDING or len(residuals) != n:
        return None

    quantiles = np.linspace(0, 1, N_BANDS + 1)[1:-1]
    edges = np.quantile(cal_preds, quantiles)
    if len(np.unique(edges)) != len(edges):
        # Predictions so concentrated that the terciles coincide — banding
        # would create empty/degenerate bands. Stay global.
        return None

    band_of = np.searchsorted(edges, cal_preds, side="right")
    q_hats = []
    labels = []
    bounds = np.concatenate(([-np.inf], edges, [np.inf]))
    for b in range(N_BANDS):
        r = residuals[band_of == b]
        if len(r) == 0:
            return None
        q_hats.append(_conformal_quantile(r, alpha))
        lo, hi = bounds[b], bounds[b + 1]
        if np.isinf(lo):
            labels.append(f"predicted under {hi:,.2f}")
        elif np.isinf(hi):
            labels.append(f"predicted over {lo:,.2f}")
        else:
            labels.append(f"predicted {lo:,.2f} to {hi:,.2f}")
    return edges, q_hats, labels


def predict_interval(
    model,
    x,
    calibration: dict,
    coverage: float = 0.90,
) -> list[IntervalResult]:
    """Predict a likely-range interval for each row of ``x``.

    Parameters
    ----------
    model
        A fitted P2Predict pipeline (with or without a
        ``TransformedTargetRegressor`` wrap).
    x
        DataFrame of inputs. Each row gets its own interval.
    calibration
        The dict returned by ``compute_calibration_residuals`` and
        persisted with the model in v0.5+.
    coverage
        Target coverage rate in (0, 1). 0.90 means a "9-in-10" interval.

    Returns
    -------
    A list of IntervalResult — one per input row.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"Coverage must be strictly between 0 and 1, got {coverage}.")

    residuals = np.asarray(calibration["residuals"], dtype=float)
    in_log_space = bool(calibration.get("in_log_space", False))
    alpha = 1.0 - coverage
    q_global = _conformal_quantile(residuals, alpha)

    # Banded (Mondrian) calibration: a per-band quantile, keyed by predicted
    # value, so the width tracks where the model is actually good instead of
    # the noisiest segment setting one width for everyone. Falls back to the
    # global quantile for old calibration dicts or small calibration sets.
    bands = None
    cal_preds = calibration.get("predictions")
    if cal_preds is not None:
        bands = _build_bands(
            np.asarray(cal_preds, dtype=float), residuals, alpha
        )

    preds = np.asarray(model.predict(x), dtype=float)
    if bands is None:
        q_hat = np.full(preds.shape, q_global)
        band_labels = [None] * len(preds)
    else:
        edges, band_q_hats, labels = bands
        band_of = np.searchsorted(edges, preds, side="right")
        q_hat = np.asarray(band_q_hats, dtype=float)[band_of]
        band_labels = [labels[b] for b in band_of]

    if in_log_space:
        # Multiplicative bounds in price space.
        low = preds * np.exp(-q_hat)
        high = preds * np.exp(+q_hat)
    else:
        low = preds - q_hat
        high = preds + q_hat

    return [
        IntervalResult(
            low=float(lo), prediction=float(p), high=float(hi),
            coverage=coverage, band=band,
        )
        for lo, p, hi, band in zip(low, preds, high, band_labels)
    ]


def coverage_health(calibration: Optional[dict]) -> Optional[str]:
    """Return a short caveat string if the calibration set is too small
    to give reliable intervals. None means "intervals are fine."

    Split conformal's coverage guarantee is technically valid for any
    n >= 1, but the *interval width* becomes very sensitive to individual
    residuals when n is small. Below ~20 calibration points we surface a
    warning at the CLI so users know to take the range with a grain of
    salt.
    """
    if calibration is None:
        return "no calibration data stored with this model — re-train on v0.5+ for likely-range support"
    n = int(calibration.get("n_calibration", 0))
    if n == 0:
        return "calibration set is empty — likely range is undefined"
    if n < 20:
        return f"calibration set is small (n={n}) — likely range may be noisy"
    return None
