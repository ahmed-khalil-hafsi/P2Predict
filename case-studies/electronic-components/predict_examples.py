"""Sample predictions on the trained electronic-component model.

Demonstrates the full P2Predict inference surface — point estimate,
likely range (--interval), per-feature attribution (--explain), and a
what-if comparison (--whatif). The output of this script is what the
case study's README pastes in.

Run after `fetch_data.py` and `p2predict-train` have produced a model
under `models/`.

Status: TEMPLATE — fill in realistic example parts once you have a
trained model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if
from p2predict.intervals import compute_calibration_residuals  # noqa: F401 — example import


# TODO: point at the trained model produced by `p2predict-train`.
# The default filename `p2predict-train` writes is
# `models/<algo>_<target>_<timestamp>.model` — replace this with the
# actual one once trained.
MODEL_PATH = Path("models/EDIT_ME_random_forest_unit_price_1k_XXXXX.model")


def _example_parts() -> list[dict]:
    """Three example components a procurement engineer might ask about.

    Replace these with realistic ones once you know what's in the
    dataset.
    """
    return [
        # TODO: fill in real example components matching the columns the
        # model was trained on (manufacturer / package / voltage /
        # capacitance / tolerance / lead_time).
        {
            "manufacturer": "Murata",
            "package": "0603",
            "voltage": 25.0,
            "capacitance": 100e-9,    # 100 nF
            "tolerance": 10.0,
            "lead_time": 8,
        },
        {
            "manufacturer": "KEMET",
            "package": "1206",
            "voltage": 50.0,
            "capacitance": 10e-6,     # 10 µF
            "tolerance": 20.0,
            "lead_time": 12,
        },
        {
            "manufacturer": "Nichicon",
            "package": "Radial",
            "voltage": 100.0,
            "capacitance": 470e-6,    # 470 µF — electrolytic
            "tolerance": 20.0,
            "lead_time": 16,
        },
    ]


def main() -> None:
    loaded = load_model(MODEL_PATH)
    model = loaded["model"]
    background = loaded.get("background_sample")
    calibration = loaded.get("calibration")
    target = loaded["target_feature"]

    parts = pd.DataFrame(_example_parts())

    print(f"Model: {loaded['model_name']}  target: {target}\n")

    # 1. Point predictions + likely range.
    intervals = predict_interval(model, parts, calibration, coverage=0.90)
    for part, interval in zip(_example_parts(), intervals):
        spec = ", ".join(f"{k}={v}" for k, v in part.items())
        print(f"  {spec}")
        print(f"    predicted: {interval.prediction:.4f}")
        print(f"    likely range (90%): {interval.low:.4f} – {interval.high:.4f}\n")

    # 2. Per-feature SHAP attribution on one part. The contributions sum
    #    to the prediction by SHAP's local-accuracy axiom — see
    #    p2predict/explain.py for the proof sketch.
    print("\nAttribution for the first part:\n")
    explanation = explain(model, parts.head(1), background_X=background)
    print(f"  Baseline: {explanation.baseline:+.4f}")
    for feature, contribution in sorted(
        explanation.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
    ):
        print(f"  {feature:<14} {contribution:+.4f}")
    print(f"  Prediction: {explanation.prediction:+.4f}  "
          "(should ≈ baseline + sum of contributions)\n")

    # 3. What-if: relax tolerance from 10% to 20% and see the cost delta.
    feature_types = {
        # TODO: this should match what the saved model's preprocessor
        # learned. Pull from loaded['model'] if you want to be precise.
        "manufacturer": "Categorical",
        "package": "Categorical",
        "voltage": "Numerical",
        "capacitance": "Numerical",
        "tolerance": "Numerical",
        "lead_time": "Numerical",
    }
    print("\nWhat-if: relax tolerance from 10% to 20% on the first part:\n")
    comparison = what_if(
        model,
        parts.head(1),
        {"tolerance": "20.0"},
        feature_types,
        background_X=background,
        calibration=calibration,
        coverage=0.90,
    )
    print(f"  Base:           {comparison.base_prediction:.4f}")
    print(f"  Counterfactual: {comparison.counterfactual_prediction:.4f}")
    print(f"  Delta:          {comparison.delta:+.4f} ({comparison.delta_pct:+.1f}%)")
    print("  Procurement read: this is the cost penalty a buyer pays for a "
          "tighter tolerance spec.")


if __name__ == "__main__":
    main()
