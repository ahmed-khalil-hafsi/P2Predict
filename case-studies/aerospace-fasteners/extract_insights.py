"""One-feature-at-a-time sweeps behind the README's "So what?" findings.

Holds a baseline commodity bolt fixed and sweeps a single feature to expose
how PUB LOG prices each cost driver:

  * material:          alloy steel -> CRES -> A286 -> titanium  (the premium ladder)
  * thread_diameter_in: the dimensional cost ruler
  * length_in:          the length cost ruler
  * head_style:         hex vs socket vs 12-point

Mirrors the BMIC extract_insights.py. Fill the baseline + sweep values against
the trained model once data/bolts_clean.csv exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from p2predict import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# Baseline commodity bolt — columns + category strings match bolts_clean.csv.
BASELINE = {
    "material": "Alloy Steel", "head_style": "HEXAGON",
    "thread_diameter_in": 0.25, "length_in": 1.0, "thread_class": "3A",
    "thread_series": "UNF", "finish": "Cadmium",
    "tensile_strength_psi": 125000, "threads_per_inch": 28,
    "width_across_flats_in": 0.4375,
}
MATERIAL_LADDER = ["Alloy Steel", "Corrosion Resisting Steel",
                   "A286 Superalloy", "Nickel Alloy", "Titanium"]
DIAMETER_GRID = [0.164, 0.190, 0.250, 0.3125, 0.375, 0.500]
LENGTH_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def _latest_model() -> Path:
    cands = sorted(MODELS_DIR.glob("*_unit_price_each_usd_*.model"))
    if not cands:
        sys.exit("Train a fastener model first — see README.")
    return cands[-1]


def _sweep(model, feature: str, values: list) -> None:
    rows = []
    for v in values:
        part = dict(BASELINE)
        part[feature] = v
        rows.append(part)
    preds = model.predict(pd.DataFrame(rows))
    base = preds[0]
    print(f"\n{feature}:")
    for v, p in zip(values, preds):
        print(f"  {str(v):<26} ${p:>7.2f}   {(p / base - 1) * 100:+5.0f}%")


def main() -> None:
    model = load_model(_latest_model())["model"]
    _sweep(model, "material", MATERIAL_LADDER)
    _sweep(model, "thread_diameter_in", DIAMETER_GRID)
    _sweep(model, "length_in", LENGTH_GRID)


if __name__ == "__main__":
    main()
