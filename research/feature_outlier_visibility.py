"""Measures what one extreme-magnitude feature cell does to `auto_train`'s
model selection, and whether the existing feature-outlier policies fix it.

Run:  python research/feature_outlier_visibility.py
Needs only the project environment (no eval extras).

The dataset is deliberately procurement-shaped and deliberately linear: price
is generated as a linear function of two specs plus noise, so Ridge is the
*correct* model and any run that does not select it has been misled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from p2predict.model_evals import evaluate_model
from p2predict.outliers import apply_feature_outlier_policy
from p2predict.prepare_data import Get_Column_Types
from p2predict.training import auto_train

FEATURES = ["Weight", "Size", "Wide_Range_Spec"]
SEED = 0


def make_data(n: int = 200, pathological: bool = True) -> pd.DataFrame:
    """200 parts, price linear in Weight and Size. One spec column has a huge
    spread; with `pathological`, a single row in it is absurd, which is the
    shape a unit mix-up or a fat-fingered entry produces in real spend data.
    """
    rng = np.random.default_rng(SEED)
    weight = rng.uniform(1, 40, n)
    size = rng.uniform(5, 60, n)
    price = 3.1 * weight + 0.8 * size + rng.normal(0, 2, n) + 20
    spec = rng.uniform(1, 100, n)
    if pathological:
        spec[7] = 3.6e31
    return pd.DataFrame(
        {"Weight": weight, "Size": size, "Wide_Range_Spec": spec, "Price": price}
    )


def fit_and_score(data: pd.DataFrame, label: str) -> dict:
    X, y = data[FEATURES], data["Price"]
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    num, cat = Get_Column_Types(X_train)
    model, algorithm, scores, _ = auto_train(X_train, y_train, num, cat, budget="fast")
    mae, r2, _, _ = evaluate_model(X_test, y_test, model)
    print(f"  {label:<28} selected={algorithm:<14} "
          f"ridge_cv={scores['ridge']:>11.4g}  holdout_mae={mae:6.2f}  holdout_r2={r2:.4f}")
    return {"label": label, "algorithm": algorithm, "ridge_cv": float(scores["ridge"]),
            "mae": float(mae), "r2": float(r2)}


def main() -> None:
    print("Baseline: no pathological cell (Ridge is the correct model)")
    fit_and_score(make_data(pathological=False), "clean")

    print("\nOne pathological cell, each feature-outlier policy:")
    data = make_data()
    fit_and_score(data, "warn (current default)")

    for policy in ("drop", "winsorize"):
        treated, summary = apply_feature_outlier_policy(data.copy(), FEATURES, policy=policy)
        fit_and_score(treated, policy)
        per_col = summary["per_column"].get("Wide_Range_Spec", {})
        print(f"     summary computed and currently discarded by the MCP path: "
              f"n_outliers_total={summary['n_outliers_total']}, "
              f"Wide_Range_Spec n_outliers={per_col.get('n_outliers')}")


if __name__ == "__main__":
    main()
