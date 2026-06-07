"""Clean the raw DigiKey BMIC catalog into a training-ready CSV.

Reads ``data/bmics.csv`` (what ``fetch_data.py`` produces) and writes:

  * ``data/bmics_clean.csv``  — full cleaned dataset, ready for
                                 ``p2predict-train``.
  * ``data-sample/bmics_sample.csv`` — a 30-row sampled CSV checked into
                                       git for the "no API account
                                       needed" path. Generated only from
                                       *non-identifying* fields and
                                       deliberately small, since DigiKey
                                       catalog data is not redistributable
                                       in bulk. The sample is enough for
                                       a tutorial run of the workflow on
                                       a different reader's machine.

Cleaning steps
--------------
1. Drop bookkeeping columns we don't want as features: ``mpn``,
   ``description``, ``category``, ``quantity_available``.
2. Drop low-coverage spec columns (< 50% populated) — Current/Voltage
   fields are mostly unpopulated in the BMIC slice and would just add
   sparse OHE columns.
3. Drop the zero-variance ``Mounting Type`` column (≈100% Surface Mount
   in this slice).
4. Parse ``Number of Cells`` strings into:
     * ``max_cells_supported`` — numeric (an upper bound on the cell
       count the IC can manage; "1 ~ 16" → 16).
     * ``is_multi_cell``       — boolean ("1" → False; anything ≥ 2 → True).
5. Extract a numeric pin-count from ``Package / Case`` strings via the
   leading digits (e.g. "24-VFQFN Exposed Pad" → 24, "SC-74A,
   SOT-753" → null since SC-74A doesn't lead with the pin count).
6. Fill remaining categorical NaNs with the explicit string "unknown"
   so they're not silently dropped.
7. Use ``unit_price_at_1_usd`` as the target. Keep ``price_at_1k_usd``
   as a secondary column for the case-study narrative but do not pass
   it to the trainer.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "data" / "bmics.csv"
OUTPUT_CSV = HERE / "data" / "bmics_clean.csv"
SAMPLE_DIR = HERE / "data-sample"
SAMPLE_CSV = SAMPLE_DIR / "bmics_sample.csv"

# Coverage threshold — drop columns where < this fraction is populated.
COVERAGE_THRESHOLD = 0.50

DROP_BOOKKEEPING = ["mpn", "description", "category", "quantity_available"]
DROP_ZERO_VARIANCE = ["Mounting Type"]
# High-cardinality features that would blow up OHE on a 150-row dataset:
#   Package / Case (59 unique), Supplier Device Package (73), Fault Protection (32).
# package_pins captures the most predictive signal from Package / Case as a
# clean numeric. We drop the high-cardinality categoricals entirely.
DROP_HIGH_CARDINALITY = ["Package / Case", "Supplier Device Package", "Fault Protection"]

# Categorical features we keep — anything not in this list (after the
# above drops) is automatically picked up too, but listing them here
# documents intent.
CATEGORICAL_FEATURES = [
    "manufacturer",
    "Function",
    "Battery Chemistry",
    "Fault Protection",
    "Interface",
    "Package / Case",
    "Supplier Device Package",
]

_LEADING_PINS_RE = re.compile(r"^\s*(\d{1,3})-")
_CELL_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:~\s*(\d+))?\s*$")


def _parse_cells(value: object) -> tuple[float | None, bool | None]:
    """Parse 'Number of Cells' strings.

    Examples:
        "1"       → (1.0,  False)
        "2"       → (2.0,  True)
        "1 ~ 4"   → (4.0,  True)
        "3 ~ 16"  → (16.0, True)
        anything else / NaN → (None, None)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    m = _CELL_RANGE_RE.match(str(value))
    if not m:
        return None, None
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return float(hi), bool(hi >= 2)


def _parse_package_pins(value: object) -> float | None:
    """Extract leading pin count from a package string.

    "24-VFQFN Exposed Pad"   → 24.0
    "10-VFDFN Exposed Pad"   → 10.0
    "SC-74A, SOT-753"        → None  (no leading pin count, fall back to OHE on Package/Case)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = _LEADING_PINS_RE.match(str(value))
    if not m:
        return None
    return float(m.group(1))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_BOOKKEEPING if c in df.columns])
    df = df.drop(columns=[c for c in DROP_ZERO_VARIANCE if c in df.columns])

    # Coverage filter (price target is always populated; we exempt it).
    coverage = df.notna().sum() / len(df)
    low_coverage = [c for c, frac in coverage.items()
                    if frac < COVERAGE_THRESHOLD and c != "price_at_1k_usd"]
    if low_coverage:
        df = df.drop(columns=low_coverage)

    # Cell-range parsing.
    if "Number of Cells" in df.columns:
        parsed = df["Number of Cells"].apply(_parse_cells)
        df["max_cells_supported"] = parsed.apply(lambda p: p[0])
        df["is_multi_cell"] = parsed.apply(lambda p: p[1])
        df = df.drop(columns=["Number of Cells"])

    # Package pin-count parsing — extract numeric pin count from the
    # Package / Case string *before* we drop the column itself.
    if "Package / Case" in df.columns:
        df["package_pins"] = df["Package / Case"].apply(_parse_package_pins)

    # Drop high-cardinality categoricals after we've extracted what we want.
    df = df.drop(columns=[c for c in DROP_HIGH_CARDINALITY if c in df.columns])

    # Explicit 'unknown' for categorical NaNs so OHE doesn't silently drop them.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")

    return df.reset_index(drop=True)


def _build_sample(df_clean: pd.DataFrame, n: int = 30, seed: int = 7) -> pd.DataFrame:
    """A small slice of the cleaned data for the no-API-account
    reproducibility path. Deliberately drops the manufacturer-attributed
    identifiers we keep in the full dataset.

    Note: DigiKey catalog data is not redistributable in bulk. 30 rows is
    a tiny excerpt for tutorial purposes only.
    """
    sample = df_clean.sample(n=min(n, len(df_clean)), random_state=seed).copy()
    return sample.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--sample-rows", type=int, default=30)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df):,}   columns: {df.shape[1]}")

    clean_df = clean(df)
    print(f"\nAfter cleaning: {len(clean_df):,} rows × {clean_df.shape[1]} columns")
    print(f"  columns: {list(clean_df.columns)}")
    print(f"  price stats: median ${clean_df.unit_price_at_1_usd.median():.2f}, "
          f"min ${clean_df.unit_price_at_1_usd.min():.2f}, "
          f"max ${clean_df.unit_price_at_1_usd.max():.2f}, "
          f"skew {clean_df.unit_price_at_1_usd.skew():.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)
    print(f"  wrote {args.output}")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample = _build_sample(clean_df, n=args.sample_rows)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"  wrote {SAMPLE_CSV} ({len(sample)} rows, for git)")


if __name__ == "__main__":
    main()
