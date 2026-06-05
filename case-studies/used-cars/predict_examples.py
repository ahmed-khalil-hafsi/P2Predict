"""Sample predictions on the trained used-car model.

Demonstrates point estimate + likely range + per-feature SHAP +
what-if comparison. The log-target wrap will be active (used car
prices are log-normal), so the SHAP attribution is *multiplicative*
in price space — see `p2predict.explain` and the README for the math.

Run after fetch_data.py + p2predict-train.

Status: TEMPLATE — fill in realistic example vehicles once a model
has been trained.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if


# TODO: replace with the actual trained-model filename.
MODEL_PATH = Path("models/EDIT_ME_xgboost_price_XXXXX.model")


def _example_vehicles() -> list[dict]:
    """Three realistic example listings a procurement engineer for a
    fleet (or a curious developer) might ask about.

    Replace these once the actual feature names from the trained model
    are known.
    """
    return [
        {
            "manufacturer": "honda",
            "model": "civic ex",
            "year": 2019,
            "odometer": 45_000,
            "condition": "excellent",
            "transmission": "automatic",
            "fuel": "gas",
            "state": "ca",
        },
        {
            "manufacturer": "toyota",
            "model": "tacoma",
            "year": 2017,
            "odometer": 85_000,
            "condition": "good",
            "transmission": "automatic",
            "fuel": "gas",
            "state": "tx",
        },
        {
            "manufacturer": "tesla",
            "model": "model 3",
            "year": 2021,
            "odometer": 22_000,
            "condition": "like new",
            "transmission": "automatic",
            "fuel": "electric",
            "state": "wa",
        },
    ]


def main() -> None:
    loaded = load_model(MODEL_PATH)
    model = loaded["model"]
    background = loaded.get("background_sample")
    calibration = loaded.get("calibration")
    target = loaded["target_feature"]
    is_log_target = loaded.get("log_target", False)

    print(f"Model: {loaded['model_name']}  target: {target}  "
          f"log-target active: {is_log_target}\n")

    examples = _example_vehicles()
    df = pd.DataFrame(examples)

    # 1. Point predictions + likely range.
    intervals = predict_interval(model, df, calibration, coverage=0.90)
    for vehicle, interval in zip(examples, intervals):
        desc = f"{vehicle['year']} {vehicle['manufacturer']} {vehicle['model']}"
        print(f"  {desc}  ({vehicle['odometer']:,} mi, {vehicle['condition']})")
        print(f"    predicted: ${interval.prediction:,.0f}")
        print(f"    likely range (90%): ${interval.low:,.0f} – ${interval.high:,.0f}\n")

    # 2. Per-feature attribution on the first vehicle.
    #    Because log-target is active, the contributions in
    #    `explanation.contributions` are in log space — but the
    #    Explanation also carries the multiplicative_factors and the
    #    approximate dollar attribution in price space.
    print("\nAttribution for the first vehicle:\n")
    explanation = explain(model, df.head(1), background_X=background)
    if explanation.log_target and explanation.multiplicative_factors:
        print("(Multiplicative factors in price space — product ≈ predicted/baseline)")
        for feature, factor in sorted(
            explanation.multiplicative_factors.items(),
            key=lambda kv: abs(1 - kv[1]),
            reverse=True,
        ):
            pct = (factor - 1.0) * 100.0
            print(f"  {feature:<14} × {factor:.3f}  ({pct:+.1f}%)")
        print(f"\n  Baseline price: ${explanation.baseline_price:,.0f}")
        print(f"  Predicted:      ${explanation.predicted_price:,.0f}")
    else:
        for feature, contribution in sorted(
            explanation.contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        ):
            print(f"  {feature:<14} {contribution:+.4f}")

    # 3. What-if: double the mileage on the first vehicle.
    feature_types = {
        # TODO: pull from the saved preprocessor if you want this to be
        # exact rather than hand-maintained.
        "manufacturer": "Categorical", "model": "Categorical",
        "year": "Numerical", "odometer": "Numerical",
        "condition": "Categorical", "transmission": "Categorical",
        "fuel": "Categorical", "state": "Categorical",
    }
    original_miles = examples[0]["odometer"]
    print(f"\nWhat-if: bump mileage from {original_miles:,} → "
          f"{original_miles * 2:,} on the first vehicle:\n")
    comparison = what_if(
        model,
        df.head(1),
        {"odometer": str(original_miles * 2)},
        feature_types,
        background_X=background,
        calibration=calibration,
        coverage=0.90,
    )
    print(f"  Base:           ${comparison.base_prediction:,.0f}")
    print(f"  Counterfactual: ${comparison.counterfactual_prediction:,.0f}")
    print(f"  Delta:          ${comparison.delta:+,.0f} "
          f"({comparison.delta_pct:+.1f}%)")
    if comparison.multiplicative_factor:
        print(f"  ×factor:        {comparison.multiplicative_factor:.3f}")
    print("  → This is the depreciation-per-extra-mile the model learned, "
          "across hundreds of thousands of listings.")


if __name__ == "__main__":
    main()
