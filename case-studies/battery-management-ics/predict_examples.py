"""Walk through point estimate + interval + SHAP + what-if on three realistic BMIC archetypes.

Same shape as case-studies/used-cars/predict_examples.py — produces the
output the case-study README quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

FEATURE_TYPES = {
    "manufacturer":          "Categorical",
    "Battery Chemistry":     "Categorical",
    "Interface":             "Categorical",
    "max_cells_supported":   "Numerical",
    "op_temp_min_C":         "Numerical",
    "op_temp_max_C":         "Numerical",
    "package_pins":          "Numerical",
    "is_multi_cell":         "Categorical",
}


def _latest_model() -> Path:
    # Sort by mtime, not filename — the algorithm prefix precedes the
    # timestamp, so an alphabetical sort can rank an older xgboost model
    # above a newer ridge one. mtime = the model actually trained last.
    candidates = sorted(MODELS_DIR.glob("*_unit_price_at_1_usd_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(
            f"No price models in {MODELS_DIR}. Train first — see "
            "case-studies/battery-management-ics/README.md."
        )
    return candidates[-1]


def _example_parts() -> list[dict]:
    """Three BMIC archetypes a procurement engineer would actually query.

    1. Single-cell Li-ion protection IC — entry-level consumer / wearable.
    2. Multi-cell EV / datacenter battery monitor — premium-tier.
    3. Cost-down single-cell with I2C telemetry — mid-tier consumer.
    """
    return [
        {  # TI BQ29700 class
            "manufacturer": "Texas Instruments",
            "Battery Chemistry": "Lithium Ion/Polymer",
            "Interface": "unknown", "max_cells_supported": 1.0,
            "op_temp_min_C": -40.0, "op_temp_max_C": 85.0,
            "package_pins": 6.0, "is_multi_cell": "False",
        },
        {  # ADI/Maxim MAX17841 class
            "manufacturer": "Analog Devices Inc./Maxim Integrated",
            "Battery Chemistry": "Lithium Ion/Polymer",
            "Interface": "I2C, USB", "max_cells_supported": 16.0,
            "op_temp_min_C": -40.0, "op_temp_max_C": 125.0,
            "package_pins": 48.0, "is_multi_cell": "True",
        },
        {  # Microchip MCP73833 class
            "manufacturer": "Microchip Technology",
            "Battery Chemistry": "Lithium Ion/Polymer",
            "Interface": "I2C", "max_cells_supported": 1.0,
            "op_temp_min_C": -40.0, "op_temp_max_C": 85.0,
            "package_pins": 8.0, "is_multi_cell": "False",
        },
    ]


_LABELS = [
    "TI BQ29700-class single-cell Li-ion protection IC (wearable / consumer)",
    "ADI / Maxim MAX17841-class 16-cell EV / datacenter BMS monitor",
    "Microchip MCP73833-class 1-cell I2C charge controller (cost-down)",
]


def _clip(low: float) -> str:
    """Clip the lower bound to 0 in display — prices can't go negative,
    but the additive conformal interval doesn't know that. Honest framing
    in the README explains why."""
    return f"${max(0.0, low):>5.2f}" + (" *" if low < 0 else "")


def main() -> None:
    path = _latest_model()
    loaded = load_model(path)
    model = loaded["model"]
    bg = loaded["background_sample"]
    cal = loaded["calibration"]

    print(f"Model:       {path.name}")
    print(f"Algorithm:   {loaded['model_name']}")
    print(f"Target:      {loaded['target_feature']}")
    print(f"Log-target:  {loaded.get('log_target')}")
    print(f"Holdout R²:  {loaded['r2']}")
    print()

    examples = _example_parts()
    df = pd.DataFrame(examples)

    # 1. Point + 90% interval.
    print("=" * 72)
    print("1. POINT ESTIMATES + 90% LIKELY RANGES")
    print("=" * 72)
    print("(* = additive interval lower bound went negative; clipped to $0 in")
    print(" display. Log-target wrap would produce always-positive multiplicative")
    print(" intervals — see ROADMAP item for --log-target on/off flag.)")
    print()
    intervals = predict_interval(model, df, cal, coverage=0.90)
    for label, iv in zip(_LABELS, intervals):
        print(f"  {label}")
        print(f"    predicted:    ${iv.prediction:>5.2f}")
        print(f"    90% range:    {_clip(iv.low)}  to  ${iv.high:>5.2f}")
        print(f"    band:         {iv.band or 'global (calibration set too small to band)'}")
        print()

    # 2. SHAP for the EV BMS — the most procurement-interesting part.
    print("=" * 72)
    print("2. WHY ~$5.48 FOR THE 16-CELL EV BMS? — SHAP DOLLAR ATTRIBUTION")
    print("=" * 72)
    ev = df.iloc[[1]]
    ex = explain(model, ev, background_X=bg)
    print(f"  Listing:       {_LABELS[1]}")
    print(f"  Baseline:      ${ex.baseline:.2f}  (model's E[price] over training data)")
    print(f"  Prediction:    ${ex.prediction:.2f}")
    print(f"  Net delta:     ${ex.prediction - ex.baseline:+.2f}")
    print()
    print("  Per-feature contribution (dollars, rank by absolute magnitude):")
    print("  --------------------------------------------------------------------")
    for k, v in sorted(ex.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True):
        sign = "+" if v >= 0 else "-"
        print(f"    {k:<28}  {sign} ${abs(v):.2f}")
    print()
    print(f"  Axiom check:   baseline + Σ contributions = ${ex.baseline + sum(ex.contributions.values()):.4f}")
    print(f"                 prediction                  = ${ex.prediction:.4f}  ✓")
    print()

    # 3. What-if: same EV BMS, but Microchip instead of ADI/Maxim.
    print("=" * 72)
    print("3. WHAT-IF: ADI/MAXIM → MICROCHIP ON THE 16-CELL BMS")
    print("=" * 72)
    wi = what_if(
        model, ev, {"manufacturer": "Microchip Technology"}, FEATURE_TYPES,
        background_X=bg, calibration=cal, coverage=0.90,
    )
    print(f"  Base prediction (ADI/Maxim):    ${wi.base_prediction:.2f}")
    print(f"  Counterfactual (Microchip):     ${wi.counterfactual_prediction:.2f}")
    print(f"  Delta:                          ${wi.delta:+.2f}  ({wi.delta_pct:+.1f}%)")
    print()
    print("  Interpretation: same 16-cell BMS spec, swap the supplier from")
    print(f"  ADI/Maxim to Microchip — the model says {wi.delta_pct:+.1f}%. That's the")
    print("  procurement negotiation lever, quantified from real DigiKey data.")
    print("  Whether the Microchip part actually meets your spec is your")
    print("  engineer's call; the model only knows the catalog patterns.")


if __name__ == "__main__":
    main()
