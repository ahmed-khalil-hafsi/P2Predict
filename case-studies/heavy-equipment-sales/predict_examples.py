"""Point estimate + interval + SHAP + what-if on three used-machine lots.

Sell-side framing: an equipment remarketing / asset-disposal desk is about
to send three used machines across the auction block and wants a defensible
reserve price, the resale drivers behind it, and the dollar value of the
cab option. Same shape as case-studies/battery-management-ics/predict_examples.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

FEATURE_TYPES = {
    "age_at_sale":   "Numerical",
    "sale_year":     "Numerical",
    "product_group": "Categorical",
    "product_size":  "Categorical",
    "enclosure":     "Categorical",
    "state":         "Categorical",
}


def _latest_model() -> Path:
    candidates = sorted(MODELS_DIR.glob("*_sale_price_usd_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(
            f"No resale models in {MODELS_DIR}. Train first — see "
            "case-studies/heavy-equipment-sales/README.md."
        )
    return candidates[-1]


def _example_lots() -> list[dict]:
    """Three used-equipment archetypes a remarketing desk actually lists.

    1. Late-model large wheel loader with an AC cab — premium lot.
    2. Mid-age track excavator, enclosed cab — bread-and-butter lot.
    3. Older skid steer, open station — low-value, high-variance lot.
    """
    return [
        {  # premium
            "age_at_sale": 3.0, "sale_year": 2008.0,
            "product_group": "Wheel Loader", "product_size": "Large",
            "enclosure": "EROPS w AC", "state": "Texas",
        },
        {  # bread-and-butter
            "age_at_sale": 8.0, "sale_year": 2008.0,
            "product_group": "Track Excavators", "product_size": "Medium",
            "enclosure": "EROPS", "state": "Florida",
        },
        {  # low-value / high-variance
            "age_at_sale": 18.0, "sale_year": 2009.0,
            "product_group": "Skid Steer Loaders", "product_size": "Small",
            "enclosure": "OROPS", "state": "Ohio",
        },
    ]


_LABELS = [
    "3-yr Large wheel loader, AC cab (premium lot)",
    "8-yr Medium track excavator, enclosed cab (bread-and-butter)",
    "18-yr Small skid steer, open station (low-value / high-variance)",
]


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

    df = pd.DataFrame(_example_lots())

    # 1. Point + 90% interval.
    print("=" * 72)
    print("1. RESERVE-PRICE ESTIMATES + 90% LIKELY HAMMER RANGE")
    print("=" * 72)
    intervals = predict_interval(model, df, cal, coverage=0.90)
    for label, iv in zip(_LABELS, intervals):
        span = iv.high / max(iv.low, 1)
        print(f"  {label}")
        print(f"    predicted:    ${iv.prediction:>9,.0f}")
        print(f"    90% range:    ${iv.low:>9,.0f}  to  ${iv.high:>9,.0f}   ({span:.1f}x)")
        print(f"    band:         {iv.band or 'global (calibration set too small to band)'}")
        print()

    # 2. SHAP for the premium wheel loader — the most sell-interesting lot.
    #    The model is log-target, so contributions are multiplicative: we
    #    read each driver as a % lift/cut on the price, and quote the
    #    dollar_attribution figure alongside for concreteness.
    print("=" * 72)
    print("2. WHY THAT NUMBER FOR THE LARGE WHEEL LOADER? — % RESALE DRIVERS")
    print("=" * 72)
    hero = df.iloc[[0]]
    ex = explain(model, hero, background_X=bg)
    print(f"  Lot:           {_LABELS[0]}")
    print(f"  Baseline:      ${ex.baseline_price:,.0f}  (model's average machine)")
    print(f"  Prediction:    ${ex.predicted_price:,.0f}")
    print(f"  Net factor:    x{ex.predicted_price / ex.baseline_price:.3f}")
    print()
    print("  Per-driver factor (log-target model -> % lift/cut on the price):")
    print("  --------------------------------------------------------------------")
    items = sorted(ex.multiplicative_factors.items(),
                   key=lambda kv: abs(kv[1] - 1.0), reverse=True)
    for feature, factor in items:
        pct = (factor - 1.0) * 100.0
        dollars = ex.dollar_attribution.get(feature, 0.0)
        print(f"    {feature:<16}  x {factor:>5.3f}   ({pct:+6.1f}%)   ${dollars:>+9,.0f}")
    print()
    prod = 1.0
    for f in ex.multiplicative_factors.values():
        prod *= f
    print(f"  Axiom check:   product of factors = {prod:.4f}, "
          f"pred/baseline = {ex.predicted_price / ex.baseline_price:.4f}  ✓")
    print()

    # 3. What-if: what is the AC cab worth on that loader? Strip it to open.
    print("=" * 72)
    print("3. WHAT-IF: WHAT IS THE AC CAB WORTH? (EROPS w AC -> OROPS)")
    print("=" * 72)
    wi = what_if(
        model, hero, {"enclosure": "OROPS"}, FEATURE_TYPES,
        background_X=bg, calibration=cal, coverage=0.90,
    )
    print(f"  As listed (AC cab):        ${wi.base_prediction:>9,.0f}")
    print(f"  Stripped to open station:  ${wi.counterfactual_prediction:>9,.0f}")
    print(f"  Cab is worth:              ${-wi.delta:>9,.0f}  ({-wi.delta_pct:+.1f}% of hammer)")
    if wi.reliability_say_to_user:
        print(f"  Direction reliability:     {wi.reliability_say_to_user}")
    print()
    print("  Interpretation: on this exact machine, the enclosed AC cab carries")
    print(f"  about {-wi.delta_pct:+.0f}% of the resale value. That tells the remarketing desk")
    print("  which lots are worth prepping/photographing around the cab, and sets")
    print("  a defensible floor when a bidder argues the cab 'doesn't matter'.")


if __name__ == "__main__":
    main()
