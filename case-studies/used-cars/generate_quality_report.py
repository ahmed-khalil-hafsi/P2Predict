"""Generate the procurement-style model-quality PDF report.

This calls P2Predict's built-in ``plotting.plot_results_pdf`` on the
latest trained price model and writes a multi-page PDF to ``assets/``. It also
shells out to ``sips`` (macOS) to produce PNG previews of each page so
they can be embedded inline in the README.

The PDF report is the same one that ``p2predict-train`` itself produces
in expert + interactive mode. The current CLI gates it behind that
combination only, so auto-mode case-study runs have to call the API
directly — which is what this script does.

Run after a model has been trained::

    python case-studies/used-cars/generate_quality_report.py

Outputs:

  * assets/model_quality_report.pdf
  * assets/model_quality_report_page_1.png
  * assets/model_quality_report_page_2.png
  * assets/model_quality_report_page_3.png   (if feature importance is available)
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
)
from p2predict import plotting
from p2predict.prepare_data import prepare_data
from p2predict.training import extract_feature_importances

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
TRAINING_CSV = HERE / "data" / "vehicles_training.csv"


def _latest_price_model() -> Path:
    candidates = sorted(MODELS_DIR.glob("*_price_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(
            f"No *_price_*.model in {MODELS_DIR}. Train one first — see "
            "the case study README's Reproducing section."
        )
    return candidates[-1]


def _rebuild_test_set(loaded: dict) -> tuple:
    """Re-derive the exact train/test split P2Predict used at training time.

    The saved model carries the calibration residuals but not the test
    set itself. To re-create the test set we:

      1. Re-load the same training CSV the case study used (80k rows).
      2. Apply the same target outlier policy (``warn`` — keep all rows
         but report).
      3. Apply the same feature outlier policy (``drop`` — remove rows
         with year or odometer outside the Tukey fences).
      4. Use ``prepare_data`` with the same ``test_size=0.2`` and
         ``random_state=0`` defaults that the train CLI uses.

    Returns ``(model, X_test, y_test, n_train)``.
    """
    if not TRAINING_CSV.exists():
        sys.exit(
            f"Expected {TRAINING_CSV}. Run prepare_data.py first."
        )
    target = loaded["target_feature"]
    features = list(loaded["features"])

    df = pd.read_csv(TRAINING_CSV)

    # Step 2: target outlier policy (warn — no rows dropped).
    df, _ = apply_outlier_policy(df, target, policy="warn")

    # Step 3: feature outlier policy (drop) on the numeric feature columns.
    numerical = [c for c in features
                 if pd.api.types.is_numeric_dtype(df[c])]
    df, _ = apply_feature_outlier_policy(df, numerical, policy="drop")

    # Step 4: same split as the trainer used.
    X_train, X_test, y_train, y_test, _num, _cat = prepare_data(
        df, features, target, test_size=0.2,
    )
    return X_train, X_test, y_train, y_test


def _convert_pdf_to_png(pdf_path: Path) -> list[Path]:
    """Use macOS's ``sips`` to convert each PDF page to a PNG preview.

    sips converts only the first page of a multi-page PDF directly, so we
    first split the PDF using Quartz (via a tiny ``python``-Quartz one-liner)
    if available, falling back to "first page only" if not.

    Returns the list of generated PNG paths.
    """
    # Quartz/CoreGraphics route via PyObjC — best when available.
    try:
        from Quartz import (CGPDFDocumentCreateWithURL, CGPDFDocumentGetNumberOfPages,
                            CGPDFDocumentGetPage, CGPDFPageGetBoxRect,
                            kCGPDFCropBox)
        from Quartz import CGBitmapContextCreate, CGContextDrawPDFPage
        from Quartz.CoreGraphics import (CGColorSpaceCreateDeviceRGB,
                                          kCGImageAlphaPremultipliedLast,
                                          CGContextSetRGBFillColor,
                                          CGContextFillRect)
        from Quartz.ImageIO import (CGImageDestinationCreateWithURL,
                                     CGImageDestinationAddImage,
                                     CGImageDestinationFinalize)
        from Quartz import CGBitmapContextCreateImage
        from CoreFoundation import CFURLCreateFromFileSystemRepresentation
        # ...Skipping the heavyweight path. Fall back to sips below.
    except ImportError:
        pass

    # Pure-sips fallback: convert the PDF as-is, sips emits a multi-image
    # PNG when given a multi-page PDF. The simplest reliable path is to
    # split the PDF into per-page PDFs first via Python's pypdf if
    # installed; if not, we settle for "page 1 only".
    out_paths: list[Path] = []
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(str(pdf_path))
        for i in range(len(reader.pages)):
            single = pdf_path.with_name(f".tmp_page_{i+1}.pdf")
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            with open(single, "wb") as f:
                writer.write(f)
            png = pdf_path.with_name(
                pdf_path.stem + f"_page_{i+1}.png"
            )
            subprocess.run(
                ["sips", "-s", "format", "png", str(single), "--out", str(png)],
                check=True, capture_output=True,
            )
            single.unlink(missing_ok=True)
            out_paths.append(png)
        return out_paths
    except ImportError:
        pass

    # Last-resort: page 1 only.
    png = pdf_path.with_name(pdf_path.stem + "_page_1.png")
    subprocess.run(
        ["sips", "-s", "format", "png", str(pdf_path), "--out", str(png)],
        check=True, capture_output=True,
    )
    return [png]


def main() -> None:
    model_path = _latest_price_model()
    print(f"Loading {model_path.name} ...")
    loaded = load_model(model_path)
    model = loaded["model"]

    print("Re-deriving the test set with the same split + outlier policies ...")
    X_train, X_test, y_train, y_test = _rebuild_test_set(loaded)
    print(f"  X_train: {len(X_train):,} rows   X_test: {len(X_test):,} rows")

    print("Predicting on holdout ...")
    y_pred = model.predict(X_test)

    print("Extracting feature importances ...")
    try:
        feat_imp = extract_feature_importances(model, X_train)
    except Exception as exc:
        print(f"  feature importances unavailable ({exc!r}) — skipping page 3.")
        feat_imp = None

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = ASSETS_DIR / "model_quality_report.pdf"

    print(f"Writing {pdf_path} ...")
    plotting.plot_results_pdf(
        y_test, y_pred, str(pdf_path),
        target_name=loaded["target_feature"],
        model_name=loaded["model_name"],
        n_train=len(X_train),
        training_date=loaded.get("training_date"),
        feature_importances=feat_imp,
    )
    print(f"  wrote {pdf_path}  ({pdf_path.stat().st_size / 1024:,.0f} KB)")

    print("Converting PDF pages to PNG previews ...")
    pngs = _convert_pdf_to_png(pdf_path)
    for p in pngs:
        print(f"  wrote {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
