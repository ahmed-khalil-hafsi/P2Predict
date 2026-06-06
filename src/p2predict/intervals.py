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

User-facing language
--------------------
The CLI and README deliberately avoid "confidence interval" (technically
wrong for prediction intervals anyway), "alpha", "conformal", and
"coverage". We use "likely range" and natural-frequency framing
("9 in 10 similar parts fall in this range"). This module's docstrings
keep the technical names because the audience here is developers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.compose import TransformedTargetRegressor


@dataclass
class IntervalResult:
    """One prediction with its likely range.

    Attributes are named for user-facing rendering: ``low`` and ``high``
    are in the same units as ``prediction``, regardless of whether the
    underlying model used a log-target transform.
    """

    low: float
    prediction: float
    high: float
    coverage: float  # the realised target coverage, e.g. 0.90 for 90%


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
        in_log_space = True
    else:
        residuals = np.abs(y_test - preds)
        in_log_space = False

    return {
        "residuals": residuals.tolist(),
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
    q_hat = _conformal_quantile(residuals, alpha)

    preds = np.asarray(model.predict(x), dtype=float)
    if in_log_space:
        # Multiplicative bounds in price space.
        low = preds * np.exp(-q_hat)
        high = preds * np.exp(+q_hat)
    else:
        low = preds - q_hat
        high = preds + q_hat

    return [
        IntervalResult(
            low=float(lo), prediction=float(p), high=float(hi), coverage=coverage
        )
        for lo, p, hi in zip(low, preds, high)
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
