import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from modules.preprocessing import build_preprocessor


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

    categorical_df = df.select_dtypes(include=["object", "str", "bool", "category"])
    if not categorical_df.empty:
        unique_ratio = categorical_df.apply(lambda x: x.nunique() / max(len(x), 1))
        high_variation.extend(unique_ratio[unique_ratio > 0.9].index.tolist())

    return high_variation


def find_no_variation_features(df):
    unique_counts = df.nunique(dropna=False)
    return unique_counts[unique_counts <= 1].index.tolist()


def _column_types(X):
    numerical_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_cols = X.select_dtypes(include=["object", "str", "bool", "category"]).columns
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
