"""Outlier detection and handling for both the target and the feature columns.

Procurement data routinely contains rush orders, one-off spot buys, and
data-entry errors (a `Weight` of `100000` when someone meant `100`, a unit
mix-up between kg and g). Both kinds quietly distort a learned cost model:
target outliers warp R² and inflate intervals, feature outliers pull the
model's response surface around the bad rows.

This module flags both via the same Tukey IQR rule and applies one of four
policies.

Policies
--------
keep        Flag only, change nothing.
warn        Flag and warn (default). Same as keep, but with a console
            message so the user actually finds out.
drop        Remove flagged rows before training.
winsorize   Cap flagged values at the IQR bounds (preserves row count).

Two surfaces
------------
1. ``apply_outlier_policy(data, target_column, policy, multiplier)``
   Target-side. Inspects ``data[target_column]`` only. Has been around
   since v0.3.

2. ``apply_feature_outlier_policy(data, feature_columns, policy, multiplier)``
   Feature-side. Inspects every *numerical* column in ``feature_columns``
   independently. ``drop`` removes any row that has an outlier in any
   feature column; ``winsorize`` caps each column at its own IQR bounds
   independently. Categorical columns are silently ignored — "outlier"
   doesn't have a clean meaning there (a rare category isn't necessarily
   wrong, just rare). Added in v0.7.

drop semantics for the feature-side path
----------------------------------------
We chose to drop the *whole row* when any one feature column flags an
outlier, rather than null out the offending cell. Reasoning: in
procurement data an outlier in one feature almost always signals
data-entry corruption that correlates with quality issues elsewhere in
the row (a transcription mistake, a unit confusion). Throwing the row
is the conservative move. Users who want per-cell handling should
pre-clean the CSV before feeding it in.

Detection rule (both surfaces)
------------------------------
Standard Tukey IQR: a value is an outlier when it is below Q1 − 1.5·IQR
or above Q3 + 1.5·IQR, where IQR = Q3 − Q1. The multiplier is exposed
on the API but defaults to 1.5; raising it (e.g. 3.0) catches only
extreme outliers, lowering it (e.g. 1.0) is more aggressive.
"""

import numpy as np
import pandas as pd

POLICIES = ("keep", "warn", "drop", "winsorize")
IQR_MULTIPLIER = 1.5


def detect_outliers(values, multiplier=IQR_MULTIPLIER):
    """Return a ``(mask, lower, upper)`` tuple flagging Tukey-IQR outliers.

    ``values`` may be any 1-D numeric iterable. Non-numeric / NaN entries
    are treated as non-outliers (the mask is False for them) so callers
    don't need to clean inputs first.
    """
    series = pd.Series(values).astype(float)
    finite = series.dropna()
    if finite.empty:
        return pd.Series([False] * len(series), index=series.index), float("nan"), float("nan")

    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        # Central 50% collapses to a single point. The Tukey rule
        # degenerates and the old behaviour (mask all False) silently
        # missed obvious outliers in near-constant columns — e.g. a
        # Weight column of [10]*20 + [10_000] would slip through. Anything
        # not equal to the central point is, by definition, outside the
        # central 50%, so we flag it; bounds collapse to that point.
        point = float(q1)
        mask = (series != point).fillna(False)
        return mask, point, point

    lower = float(q1 - multiplier * iqr)
    upper = float(q3 + multiplier * iqr)
    mask = (series < lower) | (series > upper)
    mask = mask.fillna(False)
    return mask, lower, upper


def apply_outlier_policy(data, target_column, policy="warn", multiplier=IQR_MULTIPLIER):
    """Apply ``policy`` to outliers in ``data[target_column]``.

    Returns ``(df, summary)``. The summary dict is suitable for
    caller-side logging:
        ``{n_outliers, n_total, lower, upper, policy, applied}``

    ``applied`` is the action that actually changed the data (``drop``,
    ``winsorize``, or ``none``).
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


def _is_numeric_series(series: pd.Series) -> bool:
    """True iff the column is numeric. Categorical columns are silently
    skipped by the feature-outlier path because "outlier" doesn't have a
    clean meaning for a discrete code — a rare category isn't necessarily
    wrong, just rare. Use ``find_high_variation_features`` instead for
    that case."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def apply_feature_outlier_policy(
    data, feature_columns, policy="warn", multiplier=IQR_MULTIPLIER
):
    """Apply ``policy`` to outliers across one or more *feature* columns.

    Parameters
    ----------
    data : pd.DataFrame
        Training data. Must contain every column in ``feature_columns``.
    feature_columns : Iterable[str]
        Columns to inspect. Non-numeric columns are silently skipped (see
        module docstring for the rationale). Pass the model's feature
        list — exclude the target and any time column upstream.
    policy : str
        One of POLICIES. ``drop`` removes rows that have an outlier in
        *any* numeric feature column; ``winsorize`` caps each column at
        its own IQR bounds independently; ``keep`` / ``warn`` change
        nothing (warn surfaces a message at the caller).
    multiplier : float
        Tukey IQR multiplier. 1.5 is the textbook default.

    Returns
    -------
    (df, summary) where summary has the shape::

        {
            "policy": policy,
            "applied": "drop" | "winsorize" | "none",
            "n_total": int,
            "n_outliers_total": int,        # rows touched at all
            "per_column": {
                column_name: {
                    "n_outliers": int,
                    "lower": float,
                    "upper": float,
                },
                ...
            },
        }

    ``n_outliers_total`` counts rows containing at least one outlier
    (the relevant figure for ``drop``); per-column counts can sum to
    more than this when a row has outliers in multiple columns.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown outlier policy: {policy}. Choose from {POLICIES}.")

    numeric_cols = [
        c for c in feature_columns
        if c in data.columns and _is_numeric_series(data[c])
    ]

    per_column = {}
    any_outlier_mask = pd.Series(False, index=data.index)

    for col in numeric_cols:
        mask, lower, upper = detect_outliers(data[col], multiplier=multiplier)
        n = int(mask.sum())
        per_column[col] = {"n_outliers": n, "lower": lower, "upper": upper}
        if n > 0:
            # Align the column-specific mask onto the full-frame mask.
            any_outlier_mask = any_outlier_mask | mask.reindex(data.index, fill_value=False)

    n_outliers_total = int(any_outlier_mask.sum())
    summary = {
        "policy": policy,
        "applied": "none",
        "n_total": len(data),
        "n_outliers_total": n_outliers_total,
        "per_column": per_column,
    }

    if n_outliers_total == 0:
        return data, summary

    if policy == "drop":
        summary["applied"] = "drop"
        return data.loc[~any_outlier_mask].reset_index(drop=True), summary

    if policy == "winsorize":
        # Per-column winsorisation: each column is capped at its own
        # IQR bounds independently. Row count preserved.
        summary["applied"] = "winsorize"
        new_data = data.copy()
        for col, stats in per_column.items():
            if stats["n_outliers"] == 0:
                continue
            new_data[col] = new_data[col].clip(
                lower=stats["lower"], upper=stats["upper"]
            )
        return new_data, summary

    # keep / warn: don't change the data.
    return data, summary


# A column whose largest magnitude exceeds a typical magnitude by this factor
# median will destabilise the *linear* model path badly enough to remove Ridge
# from `auto_train`'s selection entirely.
#
# The linear preprocessor scales correctly, but `StandardScaler` fits on the
# training fold: a validation-fold row carrying a value orders of magnitude
# larger gets an astronomical z-score, the linear model extrapolates from it,
# and the resulting cross-validated R² is unrecoverable. Selection is by CV R²,
# so Ridge then loses on every such dataset no matter how well it fits the rest.
#
# The threshold is set from measurement, not taste, and deliberately far above
# anything ordinary data produces. Across the checked-in case-study samples the
# highest observed ratio is 168 (`threads_per_inch`, which contains a genuine
# 4040 where the bulk is 10-32, and which does *not* destabilise selection).
# The RDKit `Ipc` descriptor that surfaced this reaches ~7e29. A threshold of
# 1e6 sits four orders of magnitude above any real column measured and well
# inside the range where the collapse is certain rather than borderline.
# See research/feature_outlier_visibility.md.
EXTREME_MAGNITUDE_RATIO = 1e6


def _magnitude_ratio(series: pd.Series) -> float | None:
    """How many times larger the biggest magnitude is than a typical one.

    Deliberately a ratio of magnitudes rather than an IQR-based z-score. An
    IQR statistic reports infinity for a near-constant column (zero IQR, one
    different value), which is a low-variance column and not the failure this
    is looking for -- `op_temp_min_C` in the battery-IC sample is exactly that
    shape and must not be flagged.

    Returns None when the column is too small or wholly zero to judge.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().abs().to_numpy(dtype=float)
    if values.size < 4:
        return None
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return None
    typical = float(np.median(nonzero))
    if typical <= 0:
        return None
    return float(np.max(values) / typical)


def describe_feature_outliers(data, feature_columns, multiplier=IQR_MULTIPLIER):
    """Inspect feature columns and report, changing nothing.

    `apply_feature_outlier_policy` already returns this information, but only
    as a side effect of applying a policy. Callers that want to *tell someone*
    before deciding anything need it without the mutation, which is what this
    provides.

    Returns::

        {
            "n_total": int,
            "n_outliers_total": int,
            "per_column": {col: {"n_outliers": int, "lower": float, "upper": float}},
            "extreme_magnitude": [
                {"column": str, "max_abs": float, "magnitude_ratio": float,
                 "say_to_user": str},
                ...
            ],
        }

    `per_column` is ordinary Tukey IQR flagging. On small datasets it routinely
    touches 20-25% of rows, so treat it as context rather than as an alarm.
    `extreme_magnitude` is the narrow, high-confidence subset: columns that will
    actually break model selection. It is empty for ordinary data.
    """
    _, summary = apply_feature_outlier_policy(
        data, feature_columns, policy="keep", multiplier=multiplier
    )

    extreme = []
    for column in feature_columns:
        if column not in data.columns or not _is_numeric_series(data[column]):
            continue
        ratio = _magnitude_ratio(data[column])
        if ratio is None or ratio < EXTREME_MAGNITUDE_RATIO:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        max_abs = float(values.abs().max())
        extreme.append({
            "column": column,
            "max_abs": max_abs,
            "magnitude_ratio": ratio,
            "say_to_user": (
                f"The '{column}' column contains at least one value far outside "
                f"everything else in it (largest is {max_abs:.3g}). That is "
                "usually a unit mix-up or a typo, and left alone it will quietly "
                "push the model toward a worse fit. I can cap those values or "
                "drop those rows before training. Which would you prefer?"
            ),
        })

    return {
        "n_total": summary["n_total"],
        "n_outliers_total": summary["n_outliers_total"],
        "per_column": summary["per_column"],
        "extreme_magnitude": extreme,
    }
