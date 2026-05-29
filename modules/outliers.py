"""Outlier detection and handling for the training target.

Procurement data routinely contains rush orders, one-off spot buys, and data-
entry errors that distort a learned cost model. This module flags them via
the Tukey IQR rule and applies one of four policies.

Policies
--------
keep        — flag only, change nothing.
warn        — flag and warn (default). Same as keep but with a console message.
drop        — remove flagged rows before training.
winsorize   — cap flagged values at the IQR bounds (preserves row count).
"""

import numpy as np
import pandas as pd

POLICIES = ("keep", "warn", "drop", "winsorize")
IQR_MULTIPLIER = 1.5


def detect_outliers(values, multiplier=IQR_MULTIPLIER):
    """Return a (mask, lower, upper) tuple flagging Tukey-IQR outliers.

    `values` may be any 1-D numeric iterable. Non-numeric / NaN entries are
    treated as non-outliers (the mask is False for them) so callers don't
    need to clean inputs first.
    """
    series = pd.Series(values).astype(float)
    finite = series.dropna()
    if finite.empty:
        return pd.Series([False] * len(series), index=series.index), float("nan"), float("nan")

    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        # Degenerate distribution — every non-NaN value is identical.
        mask = pd.Series([False] * len(series), index=series.index)
        return mask, float(q1), float(q3)

    lower = float(q1 - multiplier * iqr)
    upper = float(q3 + multiplier * iqr)
    mask = (series < lower) | (series > upper)
    mask = mask.fillna(False)
    return mask, lower, upper


def apply_outlier_policy(data, target_column, policy="warn", multiplier=IQR_MULTIPLIER):
    """Apply `policy` to outliers in `data[target_column]`. Returns (df, summary).

    The summary dict is suitable for caller-side logging:
        {n_outliers, n_total, lower, upper, policy, applied}
    `applied` is the action that actually changed the data ("drop", "winsorize",
    or "none").
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown outlier policy: {policy}. Choose from {POLICIES}.")

    mask, lower, upper = detect_outliers(data[target_column], multiplier=multiplier)
    n_outliers = int(mask.sum())
    summary = {
        "n_outliers": n_outliers,
        "n_total": len(data),
        "lower": lower,
        "upper": upper,
        "policy": policy,
        "applied": "none",
    }

    if n_outliers == 0:
        return data, summary

    if policy == "drop":
        summary["applied"] = "drop"
        return data.loc[~mask].reset_index(drop=True), summary

    if policy == "winsorize":
        summary["applied"] = "winsorize"
        new_data = data.copy()
        new_data[target_column] = new_data[target_column].clip(lower=lower, upper=upper)
        return new_data, summary

    # keep / warn: don't change the data
    return data, summary
