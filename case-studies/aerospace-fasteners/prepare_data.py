"""Clean the filtered PUB LOG FSC 5306 (Bolts) extract into a training-ready CSV.

Reads the three filtered CSVs produced by ``fetch_data.py``:

  * data/identification_5306.csv  — NIIN, FSC, INC, ITEM_NAME
  * data/characteristics_5306.csv — long (NIIN, MRC, REQUIREMENTS_STATEMENT,
                                    CLEAR_TEXT_REPLY) spec rows
  * data/management_5306.csv      — NIIN, UI, UNIT_PRICE, EFFECTIVE_DATE

and writes:

  * data/bolts_clean.csv          — cleaned, training-ready, target =
                                    unit_price_each_usd
  * data-sample/bolts_sample.csv  — a small committed sample. PUB LOG is U.S.
                                    Government public domain, so unlike the BMIC
                                    sample this is a faithful slice, not a
                                    redacted one.

The shape of the work
---------------------
PUB LOG characteristics arrive in **long** format — one row per
(NIIN, MRC, requirement-name, value), e.g.::

    000011993 | MATT | MATERIAL                | STEEL COMP 4140 OR STEEL COMP E4340 ...
    000011993 | CMLP | THREAD QUANTITY PER INCH| 20
    000011993 | AASA | THREAD LENGTH           | 0.405 INCHES MINIMUM AND 0.452 INCHES MAXIMUM

Step 1 pivots that to one row per NIIN with one column per spec. Then we join
the unit price, normalise it to per-each via the unit-of-issue, parse the
datasheet-style strings into clean numerics/categoricals, and coverage-filter.

Two load-bearing data-hygiene steps (the fastener equivalents of the BMIC
"150 -> 102 drop"):

  1. **Unit-of-issue normalisation.** A PUB LOG unit price is per unit-of-issue.
     "EA" is per-each; "HD" is per-hundred; "TH" is per-thousand. We divide to a
     true per-each price before modelling, or a $2.00/hundred bolt looks 200x
     pricier than a $2.00/each one.

  2. **Coverage filter.** Many FSC 5306 NIINs are missing a price, or carry only
     a few of the spec MRCs. We keep NIINs that have a valid per-each price AND
     the core dimensional specs (diameter, length, material), and drop spec
     columns populated on < COVERAGE_THRESHOLD of the survivors.

The real PUB LOG strings are messy free text, so the parsing does real work:

  * **material** is grouped into a small grade ladder — the headline lever.
    FLIS encodes the A286 iron-base superalloy as "IRON ALLOY 660", titanium as
    "TITANIUM ALLOY ...", CRES as "STEEL CORROSION RESISTING", and everything
    else "STEEL ..." collapses to commodity alloy steel.
  * **dimensional** strings ("0.250 INCHES", "1.000 INCHES NOMINAL",
    "0.367 INCHES MINIMUM AND 0.376 INCHES MAXIMUM") -> the leading float.
  * **finish** ("CADMIUM AND CHROMATE OVERALL") -> the leading treatment word.

The headline procurement lever this dataset is built to surface is the
**material premium** — what an aerospace-grade material (titanium, A286
superalloy, CRES) costs over plain alloy steel, holding size and style fixed —
quantified with ``--whatif "material:Titanium"``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SAMPLE_DIR = HERE / "data-sample"

IDENT_CSV = DATA_DIR / "identification_5306.csv"
CHAR_CSV = DATA_DIR / "characteristics_5306.csv"
MGMT_CSV = DATA_DIR / "management_5306.csv"
OUTPUT_CSV = DATA_DIR / "bolts_clean.csv"
SAMPLE_CSV = SAMPLE_DIR / "bolts_sample.csv"

COVERAGE_THRESHOLD = 0.50

# Domain price sanity bounds (per-each, USD). These are a DATA-QUALITY filter,
# not a statistical outlier policy: a fastener cataloged above ~$2,000 each is
# almost always a mis-cataloged assembly or kit (we found "bolts" listed at
# $55k-$88k each), and a sub-cent price is a normalisation artefact. We keep the
# full natural spread of real bolt prices — cents to a few hundred dollars,
# which is exactly the heavy right skew the log-target is for — and only clip
# the physically-implausible ends. ~0.6% of rows fall above the ceiling.
MIN_PRICE_EACH = 0.01
MAX_PRICE_EACH = 2000.0

# Unit-of-issue -> how many "each" are in one issue unit (the price divisor).
# EA dominates FSC 5306; the rest appear in the long tail. Anything not listed
# is treated as un-normalisable and dropped (we won't guess a price scale).
UOI_TO_EACH = {
    "EA": 1, "HD": 100, "TH": 1000, "PR": 2, "DZ": 12, "GR": 144,
    "PG": 1, "BX": 1, "PZ": 1, "HK": 100, "ST": 1, "SE": 1,
}

# REQUIREMENTS_STATEMENT (decoded spec name) -> our modelling column. These are
# the well-populated, price-relevant specs confirmed against the real extract.
SPEC_COLUMNS = {
    "MATERIAL": "material",
    "NOMINAL THREAD DIAMETER": "thread_diameter_in",
    "FASTENER LENGTH": "length_in",
    "THREAD CLASS": "thread_class",
    "HEAD STYLE": "head_style",
    "THREAD SERIES DESIGNATOR": "thread_series",
    "SURFACE TREATMENT": "finish",
    "MINIMUM TENSILE STRENGTH": "tensile_strength_psi",
    "THREAD QUANTITY PER INCH": "threads_per_inch",
    "WIDTH BETWEEN FLATS": "width_across_flats_in",
}

# Specs every modelled bolt must have — no point pricing a bolt with no size.
CORE_SPECS = ["material", "thread_diameter_in", "length_in"]

# Numeric specs (everything else stays categorical, one-hot at train time).
NUMERIC_SPECS = ["thread_diameter_in", "length_in", "tensile_strength_psi",
                 "threads_per_inch", "width_across_flats_in"]


# --------------------------------------------------------------------------- #
# Parsers — PUB LOG clear-text replies are messy datasheet prose.
# --------------------------------------------------------------------------- #
def _material_grade(value: object) -> str:
    """Collapse free-text material prose into a small price-relevant ladder.

    The order matters: check the aerospace grades before the generic "STEEL"
    fallback, because many exotic entries still contain the word STEEL.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    s = str(value).upper()
    if "TITANIUM" in s:
        return "Titanium"
    if "IRON ALLOY 660" in s or "A286" in s or "A-286" in s:
        return "A286 Superalloy"
    if "NICKEL ALLOY 718" in s or "INCONEL" in s or "NICKEL ALLOY" in s:
        return "Nickel Alloy"
    if "CORROSION RESISTING" in s or "CRES" in s or "STAINLESS" in s:
        return "Corrosion Resisting Steel"
    if "ALUMINUM" in s or "ALUMINIUM" in s:
        return "Aluminum"
    if "STEEL" in s or "IRON" in s:
        return "Alloy Steel"
    return "Other"


def _finish_group(value: object) -> str:
    """Leading surface-treatment word -> a tidy finish category."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    s = str(value).upper()
    for key, label in (
        ("CADMIUM", "Cadmium"), ("PASSIVATE", "Passivated"), ("ZINC", "Zinc"),
        ("ALUMINUM", "Aluminum"), ("PHOSPHATE", "Phosphate"),
        ("NICKEL", "Nickel"), ("CHROMATE", "Chromate"), ("ANODIZE", "Anodised"),
        ("SILVER", "Silver"), ("TIN", "Tin"),
    ):
        if key in s:
            return label
    return "Other"


def _dedup_doubled(value: object) -> object:
    """Collapse PUB LOG's exactly-doubled clear-text replies — 'HEXAGONHEXAGON'
    -> 'HEXAGON', 'UNFUNF' -> 'UNF' — which otherwise fragment a category. Only
    fires when the string is precisely its own first half repeated."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    s = str(value)
    n = len(s)
    if n % 2 == 0 and s[: n // 2] == s[n // 2:]:
        return s[: n // 2]
    return s


def _thread_class(value: object) -> str:
    """'3A3A' / '2A NUT THREAD A' / '3A' -> a clean {1A,2A,3A,...} token."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    s = str(value).strip().upper()
    for cls in ("3A", "2A", "1A", "3B", "2B", "1B"):
        if s.startswith(cls):
            return cls
    return s.split()[0] if s else "unknown"


def _leading_float(value: object) -> float | None:
    """'0.250 INCHES', '1.000 INCHES NOMINAL', '0.367 INCHES MINIMUM AND ...',
    '160000 POUNDS PER SQUARE INCH', '24' -> the leading number."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    tok = str(value).strip().split()
    if not tok:
        return None
    try:
        return float(tok[0])
    except ValueError:
        return None


def _normalise_price(price: object, uoi: object) -> float | None:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    divisor = UOI_TO_EACH.get(str(uoi).strip().upper())
    if not divisor or p <= 0:
        return None
    return p / divisor


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _pivot_characteristics(char: pd.DataFrame) -> pd.DataFrame:
    """Long (NIIN, requirement-name, value) -> wide one-row-per-NIIN spec table."""
    keep = char[char["REQUIREMENTS_STATEMENT"].isin(SPEC_COLUMNS)].copy()
    wide = keep.pivot_table(
        index="NIIN", columns="REQUIREMENTS_STATEMENT",
        values="CLEAR_TEXT_REPLY", aggfunc="first")
    wide = wide.rename(columns=SPEC_COLUMNS).reset_index()
    wide.columns.name = None
    return wide


def _one_price_per_niin(mgmt: pd.DataFrame) -> pd.DataFrame:
    """Collapse the many service-specific management rows to one per-each price
    per NIIN, taking the most recent EFFECTIVE_DATE."""
    mgmt = mgmt.copy()
    mgmt["unit_price_each_usd"] = [
        _normalise_price(p, u)
        for p, u in zip(mgmt["UNIT_PRICE"], mgmt["UI"])]
    mgmt = mgmt.dropna(subset=["unit_price_each_usd"])
    mgmt["_eff"] = pd.to_datetime(mgmt["EFFECTIVE_DATE"], format="%d-%b-%Y",
                                  errors="coerce")
    mgmt = mgmt.sort_values("_eff").drop_duplicates("NIIN", keep="last")
    return mgmt[["NIIN", "unit_price_each_usd"]]


def clean() -> pd.DataFrame:
    char = pd.read_csv(CHAR_CSV, dtype=str, keep_default_na=False)
    mgmt = pd.read_csv(MGMT_CSV, dtype=str, keep_default_na=False)

    specs = _pivot_characteristics(char)
    price = _one_price_per_niin(mgmt)

    df = specs.merge(price, on="NIIN", how="inner")

    # collapse PUB LOG's doubled clear-text on the free-text categoricals
    for col in ("head_style", "thread_series"):
        if col in df.columns:
            df[col] = df[col].apply(_dedup_doubled)

    # parse the messy strings into clean modelling values
    if "material" in df.columns:
        df["material"] = df["material"].apply(_material_grade)
    if "finish" in df.columns:
        df["finish"] = df["finish"].apply(_finish_group)
    if "thread_class" in df.columns:
        df["thread_class"] = df["thread_class"].apply(_thread_class)
    for col in NUMERIC_SPECS:
        if col in df.columns:
            df[col] = df[col].apply(_leading_float)

    # domain price sanity bounds (drop mis-cataloged assemblies + artefacts)
    df = df[df["unit_price_each_usd"].between(MIN_PRICE_EACH, MAX_PRICE_EACH)]

    # require the core specs + a price
    df = df.dropna(subset=[c for c in CORE_SPECS if c in df.columns])
    # a few rows carry nonsense materials after grouping
    df = df[df["material"] != "unknown"]

    # coverage filter on the remaining spec columns
    coverage = df.notna().sum() / len(df)
    low = [c for c, frac in coverage.items()
           if frac < COVERAGE_THRESHOLD
           and c not in ("unit_price_each_usd", "NIIN")]
    if low:
        df = df.drop(columns=low)

    # explicit 'unknown' for any categorical NaNs so OHE keeps the row
    for col in df.columns:
        if df[col].dtype == "object" and col != "NIIN":
            df[col] = df[col].fillna("unknown")

    return df.drop(columns=["NIIN"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-rows", type=int, default=60)
    args = ap.parse_args()

    df = clean()
    print(f"Cleaned: {len(df):,} bolts x {df.shape[1]} columns")
    print(f"  columns: {list(df.columns)}")
    if "material" in df.columns:
        print("\n  material ladder (count):")
        for mat, n in df["material"].value_counts().items():
            print(f"    {mat:<28} {n:>7,}")
    if "unit_price_each_usd" in df.columns and len(df):
        p = df["unit_price_each_usd"]
        print(f"\n  price/each: median ${p.median():.2f}  min ${p.min():.2f}  "
              f"max ${p.max():.2f}  skew {p.skew():.2f}")
        print("  -> skew well above 1.0: train with --log-target on")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  wrote {OUTPUT_CSV}")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    n = min(args.sample_rows, len(df))
    df.sample(n=n, random_state=7).to_csv(SAMPLE_CSV, index=False)
    print(f"  wrote {SAMPLE_CSV} ({n}-row committed sample — PUB LOG is public domain)")


if __name__ == "__main__":
    main()
