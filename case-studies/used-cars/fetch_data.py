"""Fetch the Craigslist Cars+Trucks dataset from Kaggle.

Uses ``kagglehub`` (>= 0.4.1) with the new-style Kaggle API token. The
older ``{"username": ..., "key": ...}`` credential format is *not*
required — the new token (a single opaque string, prefixed with
``KGAT_``) is enough on its own.

Setup
-----
1. Get an API token: https://www.kaggle.com/settings → "Create New Token"
2. Save it to ``~/.kaggle/api_token`` and ``chmod 600`` it::

       mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
       echo "KGAT_..." > ~/.kaggle/api_token
       chmod 600 ~/.kaggle/api_token

   (Or just export ``KAGGLE_API_TOKEN`` in your shell; this script
   accepts either path.)
3. ``pip install 'kagglehub>=0.4.1'``

Usage
-----
::

    python fetch_data.py [--out data/]

After the dataset downloads into the kagglehub cache, the script
symlinks ``vehicles.csv`` into the project's data directory so the
file is at a stable path for later training commands. The cache stays
intact, so re-running is free.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"
TOKEN_FILE = Path.home() / ".kaggle" / "api_token"


def _load_token() -> None:
    """Populate ``KAGGLE_API_TOKEN`` from the token file if it isn't
    already set in the environment.

    kagglehub picks the env var up automatically, so we only need to
    make sure it's there before importing the library.
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
    """Download the dataset (cached by kagglehub) and symlink the CSV
    into ``out_dir``. Returns the path to the symlink.
    """
    _load_token()
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub not installed. Run: pip install 'kagglehub>=0.4.1'")

    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    source_csv = cache_dir / "vehicles.csv"
    if not source_csv.exists():
        # The Kaggle dataset has shipped under slightly different names
        # historically; surface what we got rather than failing silently.
        contents = "\n".join(sorted(p.name for p in cache_dir.iterdir()))
        sys.exit(
            f"Downloaded archive does not contain vehicles.csv. Got:\n{contents}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "vehicles.csv"
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(source_csv)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to symlink the CSV into. Default: ./data/",
    )
    args = parser.parse_args()

    csv_path = fetch(args.out)
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"\n  vehicles.csv  ({size_mb:,.0f} MB)\n  -> {csv_path}\n")
    print("Next: clean and sample, then train. See README.md in this directory.")


if __name__ == "__main__":
    main()
