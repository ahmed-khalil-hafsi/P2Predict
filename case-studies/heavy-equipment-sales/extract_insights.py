"""Probe the trained heavy-equipment resale model for substantive findings.

Hold every feature fixed except one; sweep that one across plausible
values; record the price effect. The output of this script is the source
of the README's "So what?" section — written for an equipment
remarketing / fleet asset-disposal desk deciding reserve prices, not a
data scientist.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from p2predict import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def _latest_model() -> Path:
    # Sort by mtime, not filename — the algorithm prefix precedes the
    # timestamp, so an alphabetical sort can rank an older model above a
    # newer one. mtime = the model actually trained last.
    candidates = sorted(MODELS_DIR.glob("*_sale_price_usd_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No resale models in {MODELS_DIR}. Train first — see "
            "case-studies/heavy-equipment-sales/README.md."
        )
    return candidates[-1]


# Baseline: a deliberately bland mid-tier machine. Each sweep below reads
# as "what if we change ONLY this attribute of the machine on the block?".
BASE_MACHINE = {
    "age_at_sale": 8.0,
    "sale_year": 2008.0,
    "product_group": "Wheel Loader",
    "product_size": "Medium",
    "enclosure": "EROPS",
    "state": "Florida",
}


def vary(model, col: str, values: list) -> pd.DataFrame:
    rows = []
    for v in values:
        r = dict(BASE_MACHINE)
        r[col] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    preds = model.predict(df)
    return pd.DataFrame({col: values, "predicted": preds})


def main() -> None:
    m = load_model(_latest_model())["model"]
    base = m.predict(pd.DataFrame([BASE_MACHINE]))[0]
    print("Baseline machine for all sweeps:")
    for k, v in BASE_MACHINE.items():
        print(f"  {k:<16} {v}")
    print(f"\nBase predicted hammer price: ${base:,.0f}\n")

    print("=" * 72)
    print(f"CAB / ENCLOSURE PREMIUM (vs base ${base:,.0f}, enclosed cab)")
    print("=" * 72)
    encl = ["OROPS", "EROPS", "EROPS w AC"]
    r = vary(m, "enclosure", encl).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['enclosure']:<14} ${row['predicted']:>8,.0f}  "
              f"({delta:+,.0f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("PRODUCT SIZE SCALING (Mini -> Large)")
    print("=" * 72)
    sizes = ["Mini", "Compact", "Small", "Medium", "Large / Medium", "Large"]
    r = vary(m, "product_size", sizes)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['product_size']:<16} ${row['predicted']:>8,.0f}  "
              f"({delta:+,.0f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("MACHINE CATEGORY PREMIUM (product group)")
    print("=" * 72)
    groups = ["Motor Graders", "Wheel Loader", "Track Type Tractors",
              "Track Excavators", "Backhoe Loaders", "Skid Steer Loaders"]
    r = vary(m, "product_group", groups).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['product_group']:<22} ${row['predicted']:>8,.0f}  "
              f"({delta:+,.0f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("AGE DEPRECIATION (years old at sale)")
    print("=" * 72)
    ages = [0, 2, 5, 8, 12, 16, 20, 25, 30]
    r = vary(m, "age_at_sale", [float(a) for a in ages])
    prev = None
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        step = "" if prev is None else f"   step {row['predicted']-prev:+,.0f}"
        print(f"  {int(row['age_at_sale']):>2} yr   ${row['predicted']:>8,.0f}  "
              f"({delta:+,.0f}, {pct:+.0f}%){step}")
        prev = row["predicted"]

    print()
    print("=" * 72)
    print("AUCTION-YEAR CYCLE (market softness, holds machine fixed)")
    print("=" * 72)
    years = [2001, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
    r = vary(m, "sale_year", [float(y) for y in years])
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {int(row['sale_year'])}   ${row['predicted']:>8,.0f}  "
              f"({delta:+,.0f}, {pct:+.0f}%)")

    print()
    print("=" * 72)
    print("AUCTION GEOGRAPHY (sample of states)")
    print("=" * 72)
    states = ["Florida", "Texas", "California", "Washington", "Georgia",
              "Ohio", "Maryland", "Nevada", "Colorado", "New York"]
    r = vary(m, "state", states).sort_values("predicted", ascending=False)
    for _, row in r.iterrows():
        delta = row["predicted"] - base
        pct = 100 * delta / base
        print(f"  {row['state']:<14} ${row['predicted']:>8,.0f}  "
              f"({delta:+,.0f}, {pct:+.0f}%)")


if __name__ == "__main__":
    main()
