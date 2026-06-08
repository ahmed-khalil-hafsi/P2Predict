"""Point estimate + 90% interval + SHAP + what-if on three bolt archetypes.

Same shape as the BMIC case study's predict_examples.py. The headline lever
here is the **material premium**: hold size and style fixed and swap the
material from commodity alloy steel to an aerospace grade (titanium / A286
superalloy), and the model puts a per-each dollar figure on what the grade
costs you.

NOTE: the feature dicts below are placeholders to confirm against the trained
model's actual columns once data/bolts_clean.csv exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from p2predict import explain, load_model, predict_interval, what_if

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

FEATURE_TYPES = {
    "material":              "Categorical",
    "head_style":            "Categorical",
    "thread_diameter_in":    "Numerical",
    "length_in":             "Numerical",
    "thread_class":          "Categorical",
    "thread_series":         "Categorical",
    "finish":                "Categorical",
    "tensile_strength_psi":  "Numerical",
    "threads_per_inch":      "Numerical",
    "width_across_flats_in": "Numerical",
}

_LABELS = [
    "Commodity alloy-steel hex bolt, 1/4-28 x 1.0in (the value anchor)",
    "Aerospace titanium 12-point bolt, 1/4-28 x 1.0in (the premium grade)",
    "CRES hex bolt, 1/4-28 x 1.0in (the mid tier)",
]


def _example_parts() -> list[dict]:
    """Three archetypes that share size and thread so the material grade is the
    only thing moving — the columns and category strings match the cleaned
    bolts_clean.csv exactly (head styles and thread series are kept in PUB LOG's
    own clear-text form)."""
    return [
        {"material": "Alloy Steel", "head_style": "HEXAGON",
         "thread_diameter_in": 0.25, "length_in": 1.0, "thread_class": "3A",
         "thread_series": "UNF", "finish": "Cadmium",
         "tensile_strength_psi": 125000, "threads_per_inch": 28,
         "width_across_flats_in": 0.4375},
        {"material": "Titanium", "head_style": "DOUBLE HEXAGON",
         "thread_diameter_in": 0.25, "length_in": 1.0, "thread_class": "3A",
         "thread_series": "UNIFIED NATIONAL JOINT FINE (UNJF)",
         "finish": "Passivated", "tensile_strength_psi": 160000,
         "threads_per_inch": 28, "width_across_flats_in": 0.4375},
        {"material": "Corrosion Resisting Steel", "head_style": "HEXAGON",
         "thread_diameter_in": 0.25, "length_in": 1.0, "thread_class": "3A",
         "thread_series": "UNF", "finish": "Passivated",
         "tensile_strength_psi": 125000, "threads_per_inch": 28,
         "width_across_flats_in": 0.4375},
    ]


def _latest_model() -> Path:
    cands = sorted(MODELS_DIR.glob("*_unit_price_each_usd_*.model"))
    if not cands:
        sys.exit(f"No fastener price models in {MODELS_DIR}. Train first — "
                 "see case-studies/aerospace-fasteners/README.md.")
    return cands[-1]


def main() -> None:
    path = _latest_model()
    loaded = load_model(path)
    model, bg, cal = loaded["model"], loaded["background_sample"], loaded["calibration"]

    print(f"Model: {path.name}   algo: {loaded['model_name']}   "
          f"log-target: {loaded.get('log_target')}   R2: {loaded['r2']}\n")

    df = pd.DataFrame(_example_parts())

    print("=" * 72)
    print("1. POINT ESTIMATES + 90% LIKELY RANGES")
    print("=" * 72)
    for label, iv in zip(_LABELS, predict_interval(model, df, cal, coverage=0.90)):
        print(f"  {label}\n    predicted: ${iv.prediction:>7.2f}   "
              f"90% range: ${iv.low:>7.2f} to ${iv.high:>7.2f}\n")

    print("=" * 72)
    print("2. WHY THIS PRICE? — SHAP ATTRIBUTION (titanium bolt)")
    print("=" * 72)
    ti = df.iloc[[1]]
    ex = explain(model, ti, background_X=bg)
    print(f"  Baseline:   ${ex.baseline:.2f}")
    print(f"  Prediction: ${ex.prediction:.2f}\n")
    for k, v in sorted(ex.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"    {k:<26} {'+' if v >= 0 else '-'} {abs(v):.3f}")
    print()

    print("=" * 72)
    print("3. WHAT-IF: TITANIUM -> ALLOY STEEL (the material premium, quantified)")
    print("=" * 72)
    wi = what_if(model, ti, {"material": "Alloy Steel"}, FEATURE_TYPES,
                 background_X=bg, calibration=cal, coverage=0.90)
    print(f"  Base (Titanium):        ${wi.base_prediction:.2f}")
    print(f"  Counterfactual (Steel): ${wi.counterfactual_prediction:.2f}")
    print(f"  Delta:                  ${wi.delta:+.2f}  ({wi.delta_pct:+.1f}%)")


if __name__ == "__main__":
    main()
