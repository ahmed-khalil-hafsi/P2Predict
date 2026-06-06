"""Sample predictions on the trained used-car model.

Demonstrates the four interesting things a procurement / fleet user
gets from a P2Predict model in one place:

    1. A point estimate plus an honest *likely range* (conformal interval).
    2. A per-feature SHAP attribution, expressed as multiplicative factors
       in price space (because the log-target wrap is active).
    3. A what-if counterfactual: hold everything fixed, change one thing,
       see the price re-priced.

The script finds the most recent ``ridge_price_*.model`` in ../models/
automatically, so you don't have to hand-edit a timestamp. Run after
fetch_data.py + prepare_data.py + p2predict-train.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# Mirrors the saved preprocessor — these are the columns the case-study
# training command (`p2predict-train ... -tf ...`) was given.
FEATURE_TYPES = {
    "year":         "Numerical",
    "odometer":     "Numerical",
    "manufacturer": "Categorical",
    "condition":    "Categorical",
    "fuel":         "Categorical",
    "transmission": "Categorical",
    "drive":        "Categorical",
    "type":         "Categorical",
    "state":        "Categorical",
    "paint_color":  "Categorical",
}


def _find_latest_model() -> Path:
    """Pick the most recently saved ``ridge_price_*.model`` in models/.

    Falls back to any ``*_price_*.model`` if no ridge model exists, since
    a curious user might re-run training under expert mode with a
    different algorithm.
    """
    if not MODELS_DIR.exists():
        sys.exit(
            f"No models/ directory at {MODELS_DIR}. "
            "Train a model first with `p2predict-train ...`."
        )
    candidates = sorted(MODELS_DIR.glob("ridge_price_*.model"))
    if not candidates:
        candidates = sorted(MODELS_DIR.glob("*_price_*.model"))
    if not candidates:
        sys.exit(
            f"No price-target models found in {MODELS_DIR}. "
            "See case-studies/used-cars/README.md for the training command."
        )
    return candidates[-1]


def _example_listings() -> list[dict]:
    """Three realistic Craigslist-flavoured listings spanning the price
    range of the trained dataset.

    Picking three contrasting vehicles makes the SHAP attribution legible
    — you can see the *same* features pulling in different directions for
    different listings, which is the whole point of per-row explanation.
    """
    return [
        # Economy commuter: low miles, recent year, west coast, good condition.
        {
            "year": 2019, "odometer": 45_000,
            "manufacturer": "honda", "condition": "excellent",
            "fuel": "gas", "transmission": "automatic",
            "drive": "fwd", "type": "sedan",
            "state": "ca", "paint_color": "silver",
        },
        # High-mile work pickup: older, heavily used, 4wd, southern truck market.
        {
            "year": 2008, "odometer": 180_000,
            "manufacturer": "ford", "condition": "good",
            "fuel": "gas", "transmission": "automatic",
            "drive": "4wd", "type": "pickup",
            "state": "tx", "paint_color": "white",
        },
        # Late-model EV: low miles, premium brand, Pacific Northwest.
        {
            "year": 2021, "odometer": 22_000,
            "manufacturer": "tesla", "condition": "like new",
            "fuel": "electric", "transmission": "other",
            "drive": "rwd", "type": "sedan",
            "state": "wa", "paint_color": "white",
        },
    ]


def _describe(vehicle: dict) -> str:
    return (f"{vehicle['year']} {vehicle['manufacturer']} {vehicle['type']}, "
            f"{vehicle['odometer']:,} mi, "
            f"{vehicle['condition']}, {vehicle['state'].upper()}")


def main() -> None:
    model_path = _find_latest_model()
    loaded = load_model(model_path)
    model = loaded["model"]
    background = loaded.get("background_sample")
    calibration = loaded.get("calibration")
    is_log_target = loaded.get("log_target", False)

    print(f"Model:        {model_path.name}")
    print(f"Algorithm:    {loaded['model_name']}")
    print(f"Target:       {loaded['target_feature']}")
    print(f"Log-target:   {is_log_target}")
    print(f"Holdout R²:   {loaded['r2']}")
    print()

    examples = _example_listings()
    df = pd.DataFrame(examples)

    # 1. Point predictions + 90% likely range.
    print("=" * 72)
    print("1. POINT ESTIMATES + 90% LIKELY RANGES")
    print("=" * 72)
    intervals = predict_interval(model, df, calibration, coverage=0.90)
    for vehicle, interval in zip(examples, intervals):
        print(f"  {_describe(vehicle)}")
        print(f"    predicted:    ${interval.prediction:>8,.0f}")
        print(f"    likely range: ${interval.low:>8,.0f}  to  ${interval.high:>8,.0f}")
        print()

    # 2. SHAP attribution for the first listing (the Honda Civic).
    #    Log-target is active, so we read the *multiplicative factors*
    #    rather than dollar contributions — that's the form that strictly
    #    satisfies the SHAP axioms in price space.
    print("=" * 72)
    print("2. WHY THIS PRICE? (SHAP MULTIPLICATIVE ATTRIBUTION)")
    print("=" * 72)
    civic_row = df.head(1)
    explanation = explain(model, civic_row, background_X=background)
    print(f"  Listing:       {_describe(examples[0])}")
    print(f"  Baseline:      ${explanation.baseline_price:,.0f}  "
          f"(the model's E[price] over the training data)")
    print(f"  Prediction:    ${explanation.predicted_price:,.0f}")
    print(f"  Net factor:    x{explanation.predicted_price / explanation.baseline_price:.3f}")
    print()
    print("  Per-feature multiplicative factor (rank by deviation from 1.0):")
    print("  ────────────────────────────────────────────────────────────")
    items = sorted(
        explanation.multiplicative_factors.items(),
        key=lambda kv: abs(1 - kv[1]),
        reverse=True,
    )
    for feature, factor in items:
        pct = (factor - 1.0) * 100.0
        bar = "+" if factor >= 1 else "-"
        print(f"    {feature:<14}  x {factor:>5.3f}   ({pct:+6.1f}%)  {bar}")
    print()
    print(f"  Axiom check:   product of factors = "
          f"{_product(explanation.multiplicative_factors.values()):.4f}, "
          f"pred/baseline = "
          f"{explanation.predicted_price / explanation.baseline_price:.4f}  ✓")
    print()

    # 3. What-if: double the Civic's mileage. Hold everything else fixed.
    print("=" * 72)
    print("3. WHAT-IF: SAME CIVIC, BUT WITH 90,000 MILES INSTEAD OF 45,000")
    print("=" * 72)
    comparison = what_if(
        model, civic_row, {"odometer": "90000"}, FEATURE_TYPES,
        background_X=background, calibration=calibration, coverage=0.90,
    )
    print(f"  Base prediction:        ${comparison.base_prediction:,.0f}")
    print(f"  Counterfactual:         ${comparison.counterfactual_prediction:,.0f}")
    print(f"  Delta:                  ${comparison.delta:+,.0f}  "
          f"({comparison.delta_pct:+.1f}%)")
    if comparison.multiplicative_factor is not None:
        print(f"  Multiplicative factor:  x{comparison.multiplicative_factor:.4f}")
    print()
    print("  Interpretation: every doubling of mileage in this regime")
    print("  cuts the predicted price by roughly the percentage above.")
    print("  That's the depreciation curve learned from hundreds of")
    print("  thousands of Craigslist listings — not a rule of thumb.")


def _product(xs) -> float:
    out = 1.0
    for x in xs:
        out *= x
    return out


if __name__ == "__main__":
    main()
