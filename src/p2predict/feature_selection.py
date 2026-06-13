import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from p2predict.preprocessing import build_preprocessor


def find_high_variation_features(df):
    high_variation = []

    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])
    if not numeric_df.empty:
        means = numeric_df.mean()
        stds = numeric_df.std()
        # Use |mean| and guard against ~0 to keep CV well-defined.
        safe_means = means.abs().where(means.abs() > 1e-9)
        cv = (stds / safe_means).dropna()
        high_variation.extend(cv[cv > 1].index.tolist())

    categorical_df = df.select_dtypes(include=["object", "bool", "category"])
    if not categorical_df.empty:
        unique_ratio = categorical_df.apply(lambda x: x.nunique() / max(len(x), 1))
        high_variation.extend(unique_ratio[unique_ratio > 0.9].index.tolist())

    return high_variation


def find_no_variation_features(df):
    unique_counts = df.nunique(dropna=False)
    return unique_counts[unique_counts <= 1].index.tolist()


def find_leaky_features(data, target_column, threshold=0.97):
    """Flag features that look like an alternate form of the target (leakage).

    A numeric feature whose absolute Pearson correlation with the target
    exceeds ``threshold`` is almost certainly **target leakage** — a
    near-duplicate of the answer (e.g. the same price at a different quantity
    break, or a pre-rounded copy) rather than a genuine spec. Training on it
    inflates every metric while producing a model that is useless on real
    parts, because at prediction time you wouldn't have the leaked column
    (or you'd already know the price).

    Only numeric columns are screened — a categorical can't be a linear
    duplicate of a numeric target — and the target column itself is never
    returned.

    Returns a list of ``{"feature", "correlation", "reason"}`` dicts sorted
    by absolute correlation, descending. Empty when nothing looks leaky.
    """
    if target_column not in data.columns:
        return []

    y = pd.to_numeric(data[target_column], errors="coerce")
    leaks = []
    for col in data.columns:
        if col == target_column:
            continue
        x = pd.to_numeric(data[col], errors="coerce")
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if corr is not None and not pd.isna(corr) and abs(corr) >= threshold:
            leaks.append({
                "feature": col,
                "correlation": round(float(corr), 4),
                "reason": (
                    f"correlates {corr:.2f} with the target '{target_column}' — "
                    "almost certainly an alternate form of the value being "
                    "predicted (e.g. a different quantity break), not a spec. "
                    "Training on it makes the model look near-perfect but useless "
                    "on real parts."
                ),
            })

    leaks.sort(key=lambda d: abs(d["correlation"]), reverse=True)
    return leaks


def _column_types(X):
    numerical_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns
    return numerical_cols, categorical_cols


def get_most_predictable_features(data, target_column, output_only_headers=False):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    numerical_cols, categorical_cols = _column_types(X)

    preprocessor = build_preprocessor(numerical_cols, categorical_cols, model_family="tree")
    model = RandomForestRegressor(random_state=0, n_jobs=-1)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )
    pipeline.fit(X, y)

    # With OrdinalEncoder, each source column maps to a single transformed
    # column — no expansion, no underscore-grouping needed.
    raw_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = np.asarray(model.feature_importances_, dtype=float)

    source_cols = list(X.columns)
    by_source = {}
    for name, imp in zip(raw_names, importances):
        rest = name.split("__", 1)[1] if "__" in name else name
        match = None
        for col in source_cols:
            if rest == col or rest.startswith(f"{col}_"):
                if match is None or len(col) > len(match):
                    match = col
        source = match if match is not None else rest
        by_source[source] = by_source.get(source, 0.0) + float(imp)

    feature_importances = pd.DataFrame(
        sorted(by_source.items(), key=lambda kv: kv[1], reverse=True),
        columns=["Feature", "Importance"],
    )

    if output_only_headers:
        return feature_importances["Feature"]

    total = feature_importances["Importance"].sum()
    if total > 0:
        feature_importances["Importance"] = (
            feature_importances["Importance"] / total * 100
        ).round(2)
    feature_importances.rename(columns={"Importance": "Importance (%)"}, inplace=True)
    return feature_importances


# Kept as a thin alias for backwards compatibility with any external callers.
def get_most_predictable_features_RFE(data, target_column, n_features_to_select=10):
    return get_most_predictable_features(data, target_column, output_only_headers=True).head(
        n_features_to_select
    ).tolist()
