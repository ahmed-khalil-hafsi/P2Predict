"""Fetch the Craigslist Cars+Trucks dataset from Kaggle.

Uses the Kaggle CLI for authentication and download. Requires:
  pip install kaggle
  # And Kaggle API credentials at ~/.kaggle/kaggle.json
  # (https://www.kaggle.com/docs/api).

Run once before train.py and predict_examples.py.

Usage:
    python fetch_data.py [--out data/]

Status: TEMPLATE — the kaggle CLI call is sketched but the actual
invocation needs to be wired and tested.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"


def fetch(out_dir: Path) -> Path:
    """Run `kaggle datasets download` and unzip into `out_dir`.

    Returns the path to the downloaded CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(out_dir),
        "--unzip",
    ]
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(
            "kaggle CLI not found. Install with `pip install kaggle` and "
            "place your API token at ~/.kaggle/kaggle.json. "
            "See https://www.kaggle.com/docs/api"
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"kaggle download failed: {exc}")

    # TODO: the Kaggle dataset unzips to a specific filename. Confirm
    # what it actually produces and return the right path.
    csv_path = out_dir / "vehicles.csv"
    if not csv_path.exists():
        sys.exit(
            f"expected {csv_path} after unzip, but it's not there. "
            "Check what the Kaggle archive actually contains."
        )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to download into.",
    )
    args = parser.parse_args()

    csv_path = fetch(args.out)
    print(f"\n→ {csv_path}\n"
          f"\nNow run:\n  p2predict-train -i {csv_path} -t price "
          "--outliers drop --feature-outliers drop --budget thorough\n")


if __name__ == "__main__":
    main()
