"""Probe the trained model for substantive findings about used-car pricing.

This is the source of the README's "So what?" section. It runs
"hold everything else fixed, vary one feature" sweeps against the most
recently trained Ridge model and prints the resulting price tables:

  * Brand premium      — same vehicle as 25 different manufacturers
  * State premium      — same vehicle in each of the 50 US states
  * Body type premium  — sedan vs SUV vs pickup vs …
  * Condition premium  — new vs excellent vs good vs fair vs salvage
  * Drive premium      — fwd vs rwd vs 4wd
  * Fuel premium       — gas vs diesel vs electric vs hybrid
  * Mileage curve      — what every 10k miles is worth
  * Year curve         — what every 4-year jump is worth

Run after a model has been trained::

    python case-studies/used-cars/extract_insights.py

The baseline used for every sweep is intentionally bland (a 2015 Honda
sedan, gas, automatic, FWD, silver, CA, 100k miles, good condition) so
each comparison reads cleanly as "what's the effect of *only* this
feature." Re-run with a different ``BASE_VEHICLE`` to probe other
configurations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p2predict import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def _latest_ridge() -> Path:
    candidates = sorted(MODELS_DIR.glob("ridge_price_*.model"))
    return candidates[-1]


BASE_VEHICLE = {
    "year": 2015, "odometer": 100_000,
    "manufacturer": "honda", "condition": "good",
    "fuel": "gas", "transmission": "automatic",
    "drive": "fwd", "type": "sedan",
    "state": "ca", "paint_color": "silver",
}


def vary(model, col: str, values: list) -> pd.DataFrame:
    rows = []
    for v in values:
        row = dict(BASE_VEHICLE)
        row[col] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    preds = model.predict(df)
    return pd.DataFrame({col: values, "predicted": preds})


def main() -> None:
    m = load_model(_latest_ridge())["model"]

    print("Base vehicle for all sweeps:")
    for k, v in BASE_VEHICLE.items():
        print(f"  {k:<14} {v}")
    base = m.predict(pd.DataFrame([BASE_VEHICLE]))[0]
    print(f"\nBase prediction: ${base:,.0f}\n")

    print("=" * 72)
    print("MANUFACTURER PREMIUM (vs base $%.0f, gas sedan, fwd, 2015, 100k mi, good, CA)" % base)
    print("=" * 72)
    mfrs = ["toyota", "honda", "ford", "chevrolet", "nissan", "jeep", "ram",
            "gmc", "bmw", "mercedes-benz", "audi", "lexus", "subaru", "kia",
            "hyundai", "volkswagen", "mazda", "dodge", "tesla", "porsche",
            "land rover", "cadillac", "lincoln", "buick", "chrysler"]
    r = vary(m, "manufacturer", mfrs).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        print(f"  {row['manufacturer']:<18} ${row['predicted']:>8,.0f}   ×{factor:.2f}  "
              f"({(factor-1)*100:+.0f}%)")

    print()
    print("=" * 72)
    print("STATE PREMIUM (vs base in CA)")
    print("=" * 72)
    states = ["ca", "tx", "fl", "ny", "wa", "or", "co", "mi", "oh", "il",
              "az", "ga", "nc", "pa", "ma", "mn", "wi", "tn", "in", "mo",
              "nv", "ut", "id", "ok", "ks", "ne", "ia", "ar", "la", "ms",
              "al", "ky", "wv", "va", "md", "de", "nj", "ct", "ri", "nh",
              "me", "vt", "mt", "wy", "nd", "sd", "ak", "hi", "nm", "sc"]
    r = vary(m, "state", states).sort_values("predicted", ascending=False)
    print("Top 5 priciest states:")
    for _, row in r.head(5).iterrows():
        factor = row["predicted"] / base
        print(f"  {row['state'].upper():<4} ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+.1f}%)")
    print("\nBottom 5 cheapest states:")
    for _, row in r.tail(5).iterrows():
        factor = row["predicted"] / base
        print(f"  {row['state'].upper():<4} ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+.1f}%)")
    print(f"\nSpread (priciest / cheapest): {r.predicted.max() / r.predicted.min():.2f}x")

    print()
    print("=" * 72)
    print("BODY TYPE PREMIUM")
    print("=" * 72)
    types = ["sedan", "SUV", "pickup", "truck", "coupe", "hatchback",
             "wagon", "van", "convertible", "mini-van", "bus"]
    r = vary(m, "type", types).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        print(f"  {row['type']:<14} ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+.0f}%)")

    print()
    print("=" * 72)
    print("CONDITION PREMIUM")
    print("=" * 72)
    conds = ["new", "like new", "excellent", "good", "fair", "salvage", "unknown"]
    r = vary(m, "condition", conds).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        print(f"  {row['condition']:<14} ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+.0f}%)")

    print()
    print("=" * 72)
    print("DRIVE PREMIUM")
    print("=" * 72)
    drives = ["4wd", "rwd", "fwd", "unknown"]
    r = vary(m, "drive", drives).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        print(f"  {row['drive']:<8} ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+.0f}%)")

    print()
    print("=" * 72)
    print("FUEL PREMIUM")
    print("=" * 72)
    fuels = ["diesel", "electric", "gas", "hybrid", "other"]
    r = vary(m, "fuel", fuels).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        print(f"  {row['fuel']:<10} ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+.0f}%)")

    print()
    print("=" * 72)
    print("MILEAGE DEPRECIATION CURVE")
    print("=" * 72)
    miles = [10_000, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 200_000, 250_000]
    r = vary(m, "odometer", miles)
    prev = None
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        delta_per_10k = "" if prev is None else f"   Δ per 10k mi: ${(row['predicted']-prev)/(row['odometer']-r.iloc[r[r.predicted==prev].index[0],0])*10000:+,.0f}"
        if prev is not None:
            prev_miles = r[r.predicted == prev].iloc[0].odometer
            delta_per_10k = f"   Δ per 10k mi: ${(row['predicted']-prev) / (row['odometer']-prev_miles) * 10000:+,.0f}"
        print(f"  {row['odometer']:>7,} mi  ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+5.1f}%){delta_per_10k}")
        prev = row['predicted']

    print()
    print("=" * 72)
    print("YEAR DEPRECIATION CURVE")
    print("=" * 72)
    years = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
    r = vary(m, "year", years)
    for _, row in r.iterrows():
        factor = row["predicted"] / base
        print(f"  {row['year']}  ${row['predicted']:>8,.0f}   "
              f"({(factor-1)*100:+5.1f}%)")


if __name__ == "__main__":
    main()
