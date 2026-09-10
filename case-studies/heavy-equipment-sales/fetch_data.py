"""Fetch the *Blue Book for Bulldozers* used-equipment auction dataset.

This is the sell-side counterpart to the procurement case studies: instead
of "what should I pay for this part", the question is "what will this used
machine sell for at auction, and which specs lift or sink that number".

Data provenance
---------------
The dataset originates from the Kaggle *Blue Book for Bulldozers*
competition (sponsored by Fast Iron / Ritchie Bros.). The competition
archive itself is gated behind rules acceptance on the Kaggle website, so
this script pulls an openly-downloadable re-upload of the same
``Train.csv`` (401,125 auction records, 53 columns, 1989-2011) that does
not require accepting competition rules:

    https://www.kaggle.com/datasets/farhanreynaldo/blue-book-for-bulldozer

The schema is identical to the competition's ``Train.csv``. If you have
accepted the competition rules and prefer the canonical source, swap
``KAGGLE_DATASET`` below for ``kagglehub.competition_download(
"bluebook-for-bulldozers")`` and point at its ``Train/Train.csv``.

Setup
-----
1. Get an API token: https://www.kaggle.com/settings -> "Create New Token"
2. Save it to ``~/.kaggle/api_token`` and ``chmod 600`` it::

       mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
       printf 'KGAT_...' > ~/.kaggle/api_token
       chmod 600 ~/.kaggle/api_token

   (Or just export ``KAGGLE_API_TOKEN`` in your shell; this script
   accepts either path.)
3. ``pip install 'kagglehub>=0.4.1'``

Usage
-----
::

    python fetch_data.py [--out data/]

After the dataset downloads into the kagglehub cache, the script symlinks
``bulldozers_train.csv`` into the project's data directory so the file is
at a stable path for later cleaning. The cache stays intact, so re-running
is free.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KAGGLE_DATASET = "farhanreynaldo/blue-book-for-bulldozer"
TOKEN_FILE = Path.home() / ".kaggle" / "api_token"


def _load_token() -> None:
    """Populate ``KAGGLE_API_TOKEN`` from the token file if it isn't
    already set in the environment. kagglehub picks the env var up
    automatically, so we just make sure it's there before importing.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if not TOKEN_FILE.exists():
        sys.exit(
            f"No Kaggle API token found.\n"
            f"  Expected env var KAGGLE_API_TOKEN or file at {TOKEN_FILE}.\n"
            f"  See the docstring at the top of this file."
        )
    os.environ["KAGGLE_API_TOKEN"] = TOKEN_FILE.read_text().strip()


def fetch(out_dir: Path) -> Path:
    """Download the dataset (cached by kagglehub) and symlink the training
    CSV into ``out_dir``. Returns the path to the symlink.
    """
    _load_token()
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub not installed. Run: pip install 'kagglehub>=0.4.1'")

    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    # The mirror ships Train/Train.csv; other re-uploads flatten it to
    # Train.csv. Accept either.
    candidates = [cache_dir / "Train" / "Train.csv", cache_dir / "Train.csv"]
    source_csv = next((c for c in candidates if c.exists()), None)
    if source_csv is None:
        contents = "\n".join(sorted(str(p.relative_to(cache_dir))
                                    for p in cache_dir.rglob("*") if p.is_file()))
        sys.exit(
            f"Downloaded archive does not contain a Train.csv. Got:\n{contents}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "bulldozers_train.csv"
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(source_csv)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to symlink the CSV into. Default: ./data/",
    )
    args = parser.parse_args()

    csv_path = fetch(args.out)
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"\n  bulldozers_train.csv  ({size_mb:,.0f} MB)\n  -> {csv_path}\n")
    print("Next: clean and sample, then train. See README.md in this directory.")


if __name__ == "__main__":
    main()
