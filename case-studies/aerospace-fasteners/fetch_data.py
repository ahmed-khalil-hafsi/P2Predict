"""Acquire the raw fastener data from DLA PUB LOG — disk-safe.

Unlike the Battery Management ICs case study (which pulls from a live API),
the fastener data comes from the U.S. Defense Logistics Agency's **PUB LOG**
public release of the Federal Logistics Information System (FLIS) catalog.
This is the same master catalog NASA, the DoD, and their contractors order
fasteners from, by National Stock Number (NSN). It is **public domain** and
free to redistribute — the one source across our case studies where we can
check a cleaned sample straight into git with zero ToS worry.

Why this script doesn't download for you
----------------------------------------
DLA's site sits behind Akamai bot protection that 403s automated requests,
so the archives have to be downloaded from a real browser. You do that one
step; this script does everything after it.

What to download (from the FLIS Electronic Reading Room)
--------------------------------------------------------
https://www.dla.mil/Information-Operations/FLIS-Data-Electronic-Reading-Room/

The Reading Room offers the catalog split into per-segment zips (much kinder
to disk than the 2 GB monolith). For a specs -> unit-price study of bolts we
need exactly three segments:

  * IDENTIFICATION.zip   -> NSN, FSC, item-name (lets us filter to FSC 5306).
                            Inner file: P_FLIS_NSN.CSV.
  * CHARACTERISTICS.zip  -> DECODED physical/performance features (our specs:
                            thread diameter, length, material, head style,
                            grade, finish). DLA already translated the MRC
                            codes to clear text, so we don't decode them.
                            Inner file: V_CHARACTERISTICS.CSV.
  * MANAGEMENT.zip       -> unit price + unit-of-issue (our target). The
                            unit-of-issue field is load-bearing: a price may
                            be per-each (EA), per-hundred (HD) or per-thousand
                            (TH), and must be normalised to per-each.
                            Inner file: V_FLIS_MANAGEMENT.CSV.

Save the zips anywhere (e.g. ~/Downloads or "~/DLA P2Predict") and point this
script at them with ``--src-dir``.

How the disk-safe filtering works
---------------------------------
The uncompressed members are large (V_CHARACTERISTICS.CSV alone is ~3 GB), so
we never fully unpack them. Python's ``zipfile`` lets us **stream** one member
at a time straight from the compressed archive, decoding line by line. We keep
only FSC 5306 rows, which collapses gigabytes down to a few MB on disk.

Crucially the FSC lives **only** in the identification file (NSN-level); the
characteristics and management files are keyed by the 9-digit NIIN with no FSC
column. So the filter is two-stage:

  1. Stream P_FLIS_NSN.CSV, collect the NIINs whose FSC == 5306 (the bolts).
  2. Stream V_CHARACTERISTICS.CSV and V_FLIS_MANAGEMENT.CSV, keeping only rows
     whose NIIN is in that bolt set.

  python fetch_data.py --src-dir "~/DLA P2Predict"

Output (all gitignored under data/):
  data/identification_5306.csv   — NIIN, FSC, INC, ITEM_NAME for each bolt NSN
  data/characteristics_5306.csv  — long (NIIN, MRC, REQUIREMENTS_STATEMENT,
                                    CLEAR_TEXT_REPLY) spec rows for bolts
  data/management_5306.csv        — NIIN, UI, UNIT_PRICE, EFFECTIVE_DATE per bolt

Then run ``prepare_data.py`` to pivot, join, normalise, and clean these into
``data/bolts_clean.csv``. You can delete the downloaded zips afterwards.
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# Federal Supply Class for Bolts. (5305 = Screws, 5310 = Nuts/Washers.)
FSC = "5306"

# Inner CSV member name within each segment zip (verified against the real
# PUB LOG release).
IDENT_MEMBER = "P_FLIS_NSN.CSV"
CHAR_MEMBER = "V_CHARACTERISTICS.CSV"
MGMT_MEMBER = "V_FLIS_MANAGEMENT.CSV"

# Columns we carry forward from each segment (a small, explicit subset — the
# raw files have many more we don't need).
IDENT_COLS = ["NIIN", "FSC", "INC", "ITEM_NAME"]
CHAR_COLS = ["NIIN", "MRC", "REQUIREMENTS_STATEMENT", "CLEAR_TEXT_REPLY"]
MGMT_COLS = ["NIIN", "UI", "UNIT_PRICE", "EFFECTIVE_DATE"]

# PUB LOG ships latin-1 (a few non-ASCII bytes in clear-text replies).
ENCODING = "latin-1"


def _open_member(zip_path: Path, member: str):
    """Yield decoded text lines from one zip member without unpacking to disk."""
    zf = zipfile.ZipFile(zip_path)
    raw = zf.open(member)
    return zf, raw


def _collect_bolt_niins(zip_path: Path, fsc: str) -> dict[str, list[str]]:
    """Stream the identification file; return {NIIN: [FSC, INC, ITEM_NAME]} for
    every NSN in the target FSC."""
    bolts: dict[str, list[str]] = {}
    zf, raw = _open_member(zip_path, IDENT_MEMBER)
    try:
        reader = csv.reader((ln.decode(ENCODING) for ln in raw))
        header = next(reader)
        idx = {c: header.index(c) for c in IDENT_COLS}
        for row in reader:
            if row[idx["FSC"]] == fsc:
                bolts[row[idx["NIIN"]]] = [
                    row[idx["FSC"]], row[idx["INC"]], row[idx["ITEM_NAME"]]]
    finally:
        raw.close()
        zf.close()
    return bolts


def _stream_filter_by_niin(zip_path: Path, member: str, cols: list[str],
                           bolt_niins: set[str], out_path: Path) -> int:
    """Stream a NIIN-keyed segment, writing only the chosen columns for rows
    whose NIIN is a bolt. Returns rows kept."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    zf, raw = _open_member(zip_path, member)
    try:
        reader = csv.reader((ln.decode(ENCODING) for ln in raw))
        header = next(reader)
        idx = {c: header.index(c) for c in cols}
        niin_i = header.index("NIIN")
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for row in reader:
                if len(row) <= niin_i:
                    continue
                if row[niin_i] in bolt_niins:
                    writer.writerow([row[idx[c]] for c in cols])
                    kept += 1
    finally:
        raw.close()
        zf.close()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="Directory holding the downloaded PUB LOG segment zips.")
    ap.add_argument("--fsc", default=FSC,
                    help="Federal Supply Class to keep (default 5306 Bolts).")
    args = ap.parse_args()

    src = args.src_dir.expanduser()
    ident_zip = src / "IDENTIFICATION.zip"
    char_zip = src / "CHARACTERISTICS.zip"
    mgmt_zip = src / "MANAGEMENT.zip"
    for z in (ident_zip, char_zip, mgmt_zip):
        if not z.exists():
            raise SystemExit(
                f"Missing {z} — download it from the FLIS Electronic Reading Room.")

    print(f"Filtering PUB LOG to FSC {args.fsc} (streaming, disk-safe)...\n")

    # Stage 1: which NIINs are bolts?
    print(f"  [1/3] {IDENT_MEMBER}: collecting FSC {args.fsc} NIINs ...")
    bolts = _collect_bolt_niins(ident_zip, args.fsc)
    bolt_niins = set(bolts)
    print(f"        {len(bolt_niins):,} bolt NIINs")

    ident_out = DATA_DIR / "identification_5306.csv"
    ident_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ident_out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(IDENT_COLS)
        for niin, (fsc, inc, name) in bolts.items():
            writer.writerow([niin, fsc, inc, name])
    print(f"        -> {ident_out.name}")

    # Stage 2: characteristics for those NIINs.
    print(f"  [2/3] {CHAR_MEMBER}: keeping bolt rows (this reads ~3 GB, streamed) ...")
    n_char = _stream_filter_by_niin(
        char_zip, CHAR_MEMBER, CHAR_COLS, bolt_niins,
        DATA_DIR / "characteristics_5306.csv")
    print(f"        {n_char:,} characteristic rows -> characteristics_5306.csv")

    # Stage 3: management (price) for those NIINs.
    print(f"  [3/3] {MGMT_MEMBER}: keeping bolt rows ...")
    n_mgmt = _stream_filter_by_niin(
        mgmt_zip, MGMT_MEMBER, MGMT_COLS, bolt_niins,
        DATA_DIR / "management_5306.csv")
    print(f"        {n_mgmt:,} management rows -> management_5306.csv")

    print("\nNext: python prepare_data.py")
    print("Once this succeeds you can delete the downloaded zips to reclaim space.")


if __name__ == "__main__":
    main()
