from __future__ import annotations

import pandas as pd

from p2predict.model_utils import coerce_features


def features_to_dataframe(
    features: dict,
    expected_features: list[str],
    feature_types: dict[str, str],
) -> pd.DataFrame:
    """Convert a features dict from an MCP tool call to a single-row DataFrame.

    Validates that all expected features are present and coerces numeric types.
    """
    missing = [f for f in expected_features if f not in features]
    if missing:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"Expected: {expected_features}"
        )
    row = {f: features[f] for f in expected_features}
    df = pd.DataFrame([row])
    return coerce_features(df, feature_types)


def rows_to_dataframe(
    rows: list[dict],
    expected_features: list[str],
    feature_types: dict[str, str],
) -> pd.DataFrame:
    """Convert a list of feature dicts to a multi-row DataFrame."""
    if not rows:
        raise ValueError("rows list is empty")
    for i, row in enumerate(rows):
        missing = [f for f in expected_features if f not in row]
        if missing:
            raise ValueError(
                f"Row {i}: missing required features: {missing}. "
                f"Expected: {expected_features}"
            )
    df = pd.DataFrame([{f: r[f] for f in expected_features} for r in rows])
    return coerce_features(df, feature_types)
