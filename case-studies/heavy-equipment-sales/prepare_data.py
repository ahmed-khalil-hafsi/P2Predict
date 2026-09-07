"""Clean the raw Blue Book for Bulldozers auction file into a sell-side,
training-ready CSV.

Reads ``data/bulldozers_train.csv`` (what ``fetch_data.py`` symlinks) and
writes:

  * ``data/bulldozers_training.csv``   — an 80k-row random sample of the
                                          cleaned data, ready for
                                          ``p2predict-train``. The full
                                          cleaned set is ~360k rows;
                                          XGBoost HPO on all of it is slow
                                          and adds little, so we sample.
  * ``data-sample/bulldozers_sample.csv`` — a 5k-row sample committed to
                                          git for the "no Kaggle account
                                          needed" tutorial path. The Kaggle
                                          re-upload is openly downloadable,
                                          so unlike the DigiKey studies we
                                          can check a real slice in.

The question this dataset answers is a *sales* one: given a used machine's
vintage, category, size, cab type, and where it's being sold, what will it
fetch at auction — and which of those attributes lift or sink the number.

Cleaning steps
--------------
1. Parse ``saledate`` into ``sale_year`` (the auction-market year; the
   2009 downturn is visible in the data and worth keeping as a feature).
2. ``YearMade`` uses 1000 as an "unknown build year" sentinel on ~38k
   rows. A resale model is anchored on machine age, so we drop rows with
   an unknown or implausible (< 1920) build year rather than impute a
   fake age.
3. Derive ``age_at_sale = sale_year - YearMade``. Drop the handful of rows
   with a negative age (sale dated before the build year — data errors).
4. ``ProductGroupDesc`` -> ``product_group`` (the readable 6-way machine
   category: Track Excavators, Backhoe Loaders, ...).
5. ``ProductSize`` -> ``product_size``. 53% is blank; fill the blanks with
   the explicit string "unknown" so the row survives and the blank
   becomes its own honest category.
6. ``Enclosure`` -> ``enclosure``, the operator-station / cab type and the
   headline resale lever: OROPS (open station) vs EROPS (enclosed cab) vs
   EROPS w AC (enclosed cab with air conditioning). We fold the tiny
   "EROPS AC" spelling into "EROPS w AC" and map the ~330 blanks /
   "NO ROPS" / "None or Unspecified" rows to "unknown".
7. ``state`` -> ``state`` (auction geography, 53 values, fully populated).
8. Target: ``SalePrice`` -> ``sale_price_usd``.

Note on machine hours: ``MachineHoursCurrentMeter`` is 0 or blank on ~83%
of rows (the meter reading simply wasn't recorded at auction), so it can't
carry a usage signal without masquerading missingness as "brand new". We
leave it out and let machine age carry the wear story. See the README's
Limitations section.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "data" / "bulldozers_train.csv"
TRAINING_CSV = HERE / "data" / "bulldozers_training.csv"
SAMPLE_DIR = HERE / "data-sample"
SAMPLE_CSV = SAMPLE_DIR / "bulldozers_sample.csv"

FINAL_COLUMNS = [
    "sale_price_usd",
    "age_at_sale",
    "sale_year",
    "product_group",
    "product_size",
    "enclosure",
    "state",
]

_ENCLOSURE_MAP = {
    "EROPS AC": "EROPS w AC",
    "NO ROPS": "unknown",
    "None or Unspecified": "unknown",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Auction-market year.
    df["sale_year"] = pd.to_datetime(df["saledate"], errors="coerce").dt.year

    # 2. Drop unknown / implausible build years (the 1000 sentinel + noise).
    df = df[df["YearMade"] >= 1920]

    # 3. Machine age at time of sale.
    df["age_at_sale"] = df["sale_year"] - df["YearMade"]
    df = df[df["age_at_sale"] >= 0]

    # 4. Readable machine category.
    df["product_group"] = df["ProductGroupDesc"]

    # 5. Size tier, blanks made explicit.
    df["product_size"] = df["ProductSize"].fillna("unknown")

    # 6. Cab type — the resale lever.
    df["enclosure"] = (
        df["Enclosure"].replace(_ENCLOSURE_MAP).fillna("unknown")
    )

    # 7. Auction geography.
    df["state"] = df["state"]

    # 8. Target.
    df["sale_price_usd"] = df["SalePrice"].astype(float)

    out = df[FINAL_COLUMNS].reset_index(drop=True)
    # Drop any residual rows missing a kept field (keeps the trainer from
    # silently dropping them later, and keeps the row counts honest).
    out = out.dropna(subset=FINAL_COLUMNS).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--training-rows", type=int, default=80_000,
                        help="Rows to sample for bulldozers_training.csv")
    parser.add_argument("--sample-rows", type=int, default=5_000,
                        help="Rows for the committed bulldozers_sample.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  rows: {len(df):,}   columns: {df.shape[1]}")

    clean_df = clean(df)
    print(f"\nAfter cleaning: {len(clean_df):,} rows × {clean_df.shape[1]} columns")
    print(f"  columns: {list(clean_df.columns)}")
    price = clean_df["sale_price_usd"]
    print(f"  price stats: median ${price.median():,.0f}, min ${price.min():,.0f}, "
          f"max ${price.max():,.0f}, skew {price.skew():.3f}")
    print(f"  age_at_sale: median {clean_df.age_at_sale.median():.0f} yr, "
          f"max {clean_df.age_at_sale.max():.0f} yr")
    print(f"  enclosure mix: {clean_df.enclosure.value_counts().to_dict()}")

    n_train = min(args.training_rows, len(clean_df))
    training = clean_df.sample(n=n_train, random_state=args.seed).reset_index(drop=True)
    TRAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(TRAINING_CSV, index=False)
    print(f"\n  wrote {TRAINING_CSV}  ({len(training):,} rows, for training)")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    n_sample = min(args.sample_rows, len(clean_df))
    sample = clean_df.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"  wrote {SAMPLE_CSV}  ({len(sample):,} rows, committed to git)")


if __name__ == "__main__":
    main()
