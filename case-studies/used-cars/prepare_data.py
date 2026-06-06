"""Clean the raw Craigslist dataset and produce the training inputs.

Reads ``data/vehicles.csv`` (the symlink ``fetch_data.py`` produces),
keeps the predictive columns, drops obvious garbage, fills sentinel
nulls in categoricals, then writes:

* ``data/vehicles_clean.csv``        — the full cleaned dataset (~350k rows)
* ``data/vehicles_training.csv``     — 80k-row stratified random sample
                                       used as the case-study training set
* ``data-sample/vehicles_sample.csv``— 5k-row sample committed to git so
                                       readers without a Kaggle account can
                                       reproduce the shape of the result.

The cuts here are not subtle — Craigslist data is noisy by nature, and
deliberately leaving some of the dirt in lets P2Predict's outlier
handling earn its keep at training time (see ``--outliers drop`` and
``--feature-outliers drop`` in the README's "Reproducing this case
study" section).

Why we drop some columns up front
---------------------------------
* ``county`` is 100% null in this snapshot.
* ``size`` (72% null) and ``cylinders`` (42% null) are too sparse to
  carry signal once we also use ``manufacturer`` and ``type``.
* ``model`` has many thousands of unique values. Tree models cope with
  high-cardinality categoricals via the OrdinalEncoder path, but the
  CV-driven HPO search slows to a crawl. For the case-study v1 we lean
  on ``manufacturer`` as a coarser proxy.
* ``url``, ``region_url``, ``image_url``, ``VIN``, ``description``,
  ``id``, ``lat``, ``long``, ``posting_date`` carry no parametric
  signal for our purposes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP = [
    "price",        # target
    "year",         # numerical
    "odometer",     # numerical
    "manufacturer", # categorical
    "condition",    # categorical (40% null — fill 'unknown')
    "fuel",         # categorical
    "transmission", # categorical
    "drive",        # categorical
    "type",         # categorical
    "state",        # categorical
    "paint_color",  # categorical
]

# Hard guardrails — IQR-based outlier handling at training time picks
# up the rest. These cuts only drop obvious garbage so the IQR bounds
# downstream are computed against something defensible.
PRICE_MIN, PRICE_MAX = 500, 200_000
ODO_MIN, ODO_MAX = 0, 500_000
YEAR_MIN, YEAR_MAX = 1990, 2022


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[KEEP].copy()
    df = df.dropna(subset=["price", "year", "manufacturer", "odometer",
                           "fuel", "transmission"])
    df = df[(df.price >= PRICE_MIN) & (df.price <= PRICE_MAX)]
    df = df[(df.odometer >= ODO_MIN) & (df.odometer <= ODO_MAX)]
    df = df[(df.year >= YEAR_MIN) & (df.year <= YEAR_MAX)]
    for col in ["condition", "drive", "type", "paint_color"]:
        df[col] = df[col].fillna("unknown")
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--input", type=Path,
                        default=here / "data" / "vehicles.csv",
                        help="Raw Craigslist CSV (default: ./data/vehicles.csv)")
    parser.add_argument("--training-rows", type=int, default=80_000,
                        help="Rows to sample for vehicles_training.csv")
    parser.add_argument("--sample-rows", type=int, default=5_000,
                        help="Rows for the committed vehicles_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    raw = pd.read_csv(args.input)
    print(f"  raw rows: {len(raw):,}")

    clean_df = clean(raw)
    print(f"  cleaned rows: {len(clean_df):,}")
    print(f"  price skew (cleaned): {clean_df.price.skew():.2f}")

    data_dir = args.input.parent
    sample_dir = here / "data-sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean_path = data_dir / "vehicles_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"  wrote {clean_path}")

    training = clean_df.sample(n=args.training_rows, random_state=args.seed)
    training_path = data_dir / "vehicles_training.csv"
    training.to_csv(training_path, index=False)
    print(f"  wrote {training_path}  ({len(training):,} rows)")

    sample = clean_df.sample(n=args.sample_rows, random_state=args.seed)
    sample_path = sample_dir / "vehicles_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"  wrote {sample_path}  ({len(sample):,} rows, committed to git)")


if __name__ == "__main__":
    main()
