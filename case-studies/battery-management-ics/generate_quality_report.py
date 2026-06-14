"""Generate the procurement-style model-quality PDF report for the BMICs.

This calls P2Predict's built-in ``plotting.plot_results_pdf`` on the
trained Ridge model and writes a multi-page PDF to ``assets/``. It also
shells out to ``sips`` (macOS) to produce PNG previews of each page so
they can be embedded inline in the README.

The PDF report is the same one that ``p2predict-train`` itself produces
in expert + interactive mode. The current CLI gates it behind that
combination only, so auto-mode case-study runs have to call the API
directly — which is what this script does.

Run after a model has been trained::

    python case-studies/battery-management-ics/generate_quality_report.py

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
from p2predict.trained_model_io import load_csv_file
from p2predict.training import extract_feature_importances

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
TRAINING_CSV = HERE / "data" / "bmics_clean.csv"


def _latest_ridge() -> Path:
    candidates = sorted(MODELS_DIR.glob("ridge_unit_price_at_1_usd_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(
            f"No ridge_unit_price_at_1_usd_*.model in {MODELS_DIR}. Train one "
            "first — see the case study README's Reproducing section."
        )
    return candidates[-1]


def _rebuild_test_set(loaded: dict) -> tuple:
    """Re-derive the exact train/test split P2Predict used at training time.

    The saved model carries the calibration residuals but not the test
    set itself. To re-create the test set we:

      1. Re-load the same cleaned CSV through ``load_csv_file``, which is
         exactly what the train CLI does. It drops rows with NA in *any*
         column (the BMIC slice has sparse ``package_pins`` /
         ``max_cells_supported`` / ``price_at_1k_usd``), taking 150 rows
         down to the 102 the model actually trained on.
      2. Apply the same target outlier policy (``warn`` — keep all rows,
         just report). This is the CLI default and what the case study used.
      3. Apply the same feature outlier policy (``warn`` — keep all rows).
         The BMIC slice is tiny and mostly -40/85C industrial; ``drop``
         collapsed the temperature columns to a constant, so the case
         study deliberately uses ``warn``.
      4. Use ``prepare_data`` with the same ``test_size=0.2`` default that
         the train CLI uses, so the split is byte-for-byte the same.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    if not TRAINING_CSV.exists():
        sys.exit(
            f"Expected {TRAINING_CSV}. Run prepare_data.py first."
        )
    target = loaded["target_feature"]
    features = list(loaded["features"])

    df = load_csv_file(str(TRAINING_CSV))

    # Step 2: target outlier policy (warn — no rows dropped).
    df, _ = apply_outlier_policy(df, target, policy="warn")

    # Step 3: feature outlier policy (warn — no rows dropped).
    numerical = [c for c in features
                 if pd.api.types.is_numeric_dtype(df[c])]
    df, _ = apply_feature_outlier_policy(df, numerical, policy="warn")

    # Step 4: same split as the trainer used.
    X_train, X_test, y_train, y_test, _num, _cat = prepare_data(
        df, features, target, test_size=0.2,
    )
    return X_train, X_test, y_train, y_test


def _convert_pdf_to_png(pdf_path: Path) -> list[Path]:
    """Use macOS's ``sips`` to convert each PDF page to a PNG preview.

    Splits the PDF into per-page PDFs first via ``pypdf`` if installed; if
    not, settles for "page 1 only". Returns the list of generated PNGs.
    """
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
            png = pdf_path.with_name(pdf_path.stem + f"_page_{i+1}.png")
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
    model_path = _latest_ridge()
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
