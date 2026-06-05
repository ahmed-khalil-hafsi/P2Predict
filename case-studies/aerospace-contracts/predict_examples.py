"""Sample predictions on the trained aerospace-contracts model.

Demonstrates how a cost estimator at a defense prime (or a procurement
officer at an agency) could benchmark a new solicitation against
historical awards in the same PSC.

Status: TEMPLATE — fill in realistic example contracts once a model
has been trained on the chosen PSC.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if


# TODO: replace with the actual trained-model filename.
MODEL_PATH = Path("models/EDIT_ME_xgboost_obligated_amount_XXXXX.model")


def _example_solicitations() -> list[dict]:
    """Three hypothetical solicitations a cost estimator might benchmark.

    Replace these with the actual feature shape from the trained model
    (in particular: the PSC-specific contractor and place-of-performance
    values that the model has seen).
    """
    return [
        {
            "awarding_agency": "Department of the Navy",
            "contractor": "Boeing",
            "set_aside": "unrestricted",
            "period_of_performance_months": 36,
            "place_of_performance_state": "WA",
            "competition_type": "full_and_open",
            "fiscal_year": 2024,
        },
        {
            "awarding_agency": "Department of the Air Force",
            "contractor": "Lockheed Martin",
            "set_aside": "unrestricted",
            "period_of_performance_months": 24,
            "place_of_performance_state": "TX",
            "competition_type": "limited_sources",
            "fiscal_year": 2024,
        },
        {
            "awarding_agency": "Defense Logistics Agency",
            "contractor": "SmallBusinessInc",
            "set_aside": "small_business",
            "period_of_performance_months": 12,
            "place_of_performance_state": "PA",
            "competition_type": "full_and_open",
            "fiscal_year": 2024,
        },
    ]


def main() -> None:
    loaded = load_model(MODEL_PATH)
    model = loaded["model"]
    background = loaded.get("background_sample")
    calibration = loaded.get("calibration")
    target = loaded["target_feature"]

    examples = _example_solicitations()
    df = pd.DataFrame(examples)

    print(f"Model: {loaded['model_name']}  target: {target}  "
          f"log-target active: {loaded.get('log_target', False)}\n")

    intervals = predict_interval(model, df, calibration, coverage=0.90)
    for solicitation, interval in zip(examples, intervals):
        desc = (f"{solicitation['awarding_agency']} → "
                f"{solicitation['contractor']} "
                f"({solicitation['period_of_performance_months']} mo)")
        print(f"  {desc}")
        print(f"    predicted: ${interval.prediction:,.0f}")
        print(f"    likely range (90%): "
              f"${interval.low:,.0f} – ${interval.high:,.0f}\n")

    # Per-feature attribution on the first solicitation.
    print("\nAttribution for the first solicitation:\n")
    explanation = explain(model, df.head(1), background_X=background)
    if explanation.log_target and explanation.multiplicative_factors:
        print("(Multiplicative factors in dollar space)")
        for feature, factor in sorted(
            explanation.multiplicative_factors.items(),
            key=lambda kv: abs(1 - kv[1]),
            reverse=True,
        ):
            pct = (factor - 1.0) * 100.0
            print(f"  {feature:<35} × {factor:.3f}  ({pct:+.1f}%)")
        print(f"\n  Baseline: ${explanation.baseline_price:,.0f}")
        print(f"  Predicted: ${explanation.predicted_price:,.0f}")
    else:
        for feature, contribution in sorted(
            explanation.contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        ):
            print(f"  {feature:<35} {contribution:+,.0f}")

    # What-if: change the set-aside type.
    feature_types = {
        # TODO: pull from the saved preprocessor for precision.
        "awarding_agency": "Categorical", "contractor": "Categorical",
        "set_aside": "Categorical",
        "period_of_performance_months": "Numerical",
        "place_of_performance_state": "Categorical",
        "competition_type": "Categorical",
        "fiscal_year": "Numerical",
    }
    print("\nWhat-if: change set-aside from 'unrestricted' to "
          "'small_business' on the first solicitation:\n")
    comparison = what_if(
        model,
        df.head(1),
        {"set_aside": "small_business"},
        feature_types,
        background_X=background,
        calibration=calibration,
        coverage=0.90,
    )
    print(f"  Base:           ${comparison.base_prediction:,.0f}")
    print(f"  Counterfactual: ${comparison.counterfactual_prediction:,.0f}")
    print(f"  Delta:          ${comparison.delta:+,.0f} "
          f"({comparison.delta_pct:+.1f}%)")


if __name__ == "__main__":
    main()
