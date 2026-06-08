"""Generate the procurement-style model-quality PDF report for the bolts.

Adapted from the BMIC case study's generate_quality_report.py. Calls
P2Predict's built-in ``plotting.plot_results_pdf`` on the trained model and
writes a multi-page PDF to ``assets/``, plus per-page PNG previews via macOS
``sips`` for inline embedding in the README.

Run after a model has been trained::

    python case-studies/aerospace-fasteners/generate_quality_report.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from p2predict import (
    apply_feature_outlier_policy,
    apply_outlier_policy,
    load_model,
    plotting,
)
from p2predict.prepare_data import prepare_data
from p2predict.trained_model_io import load_csv_file
from p2predict.training import extract_feature_importances

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
TRAINING_CSV = HERE / "data" / "bolts_clean.csv"

# Outlier policies the case study trains with — keep in sync with the
# Reproducing section of the README. TODO: confirm once the data lands.
TARGET_OUTLIER_POLICY = "warn"
FEATURE_OUTLIER_POLICY = "warn"


def _latest_model() -> Path:
    cands = sorted(MODELS_DIR.glob("*_unit_price_each_usd_*.model"))
    if not cands:
        sys.exit(f"No *_unit_price_each_usd_*.model in {MODELS_DIR}. Train first.")
    return cands[-1]


def _rebuild_test_set(loaded: dict) -> tuple:
    if not TRAINING_CSV.exists():
        sys.exit(f"Expected {TRAINING_CSV}. Run prepare_data.py first.")
    target = loaded["target_feature"]
    features = list(loaded["features"])

    df = load_csv_file(str(TRAINING_CSV))
    df, _ = apply_outlier_policy(df, target, policy=TARGET_OUTLIER_POLICY)
    numerical = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    df, _ = apply_feature_outlier_policy(df, numerical, policy=FEATURE_OUTLIER_POLICY)

    X_train, X_test, y_train, y_test, _num, _cat = prepare_data(
        df, features, target, test_size=0.2)
    return X_train, X_test, y_train, y_test


def _convert_pdf_to_png(pdf_path: Path) -> list[Path]:
    out: list[Path] = []
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(str(pdf_path))
        for i in range(len(reader.pages)):
            single = pdf_path.with_name(f".tmp_page_{i+1}.pdf")
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            with open(single, "wb") as f:
                writer.write(f)
            png = pdf_path.with_name(pdf_path.stem + f"_page_{i+1}.png")
            subprocess.run(["sips", "-s", "format", "png", str(single),
                            "--out", str(png)], check=True, capture_output=True)
            single.unlink(missing_ok=True)
            out.append(png)
        return out
    except ImportError:
        png = pdf_path.with_name(pdf_path.stem + "_page_1.png")
        subprocess.run(["sips", "-s", "format", "png", str(pdf_path),
                        "--out", str(png)], check=True, capture_output=True)
        return [png]


def main() -> None:
    model_path = _latest_model()
    print(f"Loading {model_path.name} ...")
    loaded = load_model(model_path)
    model = loaded["model"]

    X_train, X_test, y_train, y_test = _rebuild_test_set(loaded)
    print(f"  X_train: {len(X_train):,}   X_test: {len(X_test):,}")
    y_pred = model.predict(X_test)

    try:
        feat_imp = extract_feature_importances(model, X_train)
    except Exception as exc:
        print(f"  feature importances unavailable ({exc!r}) — skipping page 3.")
        feat_imp = None

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = ASSETS_DIR / "model_quality_report.pdf"
    plotting.plot_results_pdf(
        y_test, y_pred, str(pdf_path),
        target_name=loaded["target_feature"],
        model_name=loaded["model_name"],
        n_train=len(X_train),
        training_date=loaded.get("training_date"),
        feature_importances=feat_imp,
    )
    print(f"  wrote {pdf_path}")
    for p in _convert_pdf_to_png(pdf_path):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
