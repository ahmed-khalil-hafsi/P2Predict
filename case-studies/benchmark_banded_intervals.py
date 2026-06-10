"""Cross-case-study benchmark: global vs banded (Mondrian) likely-ranges.

Runs the three case studies — battery-management ICs, aerospace fasteners,
used cars — through the same honest protocol and reports what banded
conformal calibration changes about the 90% likely-range:

  1. 60/20/20 split: train / calibration / evaluation. The model is trained
     on the train split, the conformal residuals come from the calibration
     split, and every number reported below is measured on the evaluation
     split — points neither the model nor the calibration ever saw.
  2. Train exactly as the case study does (same target, features, outlier
     policy, log-target setting; XGBoost with tuning).
  3. Compute intervals twice from the same calibration data:
       * GLOBAL — the pre-banding behaviour (calibration dict with the
         stored predictions removed, which is exactly what an old model
         file provides), and
       * BANDED — the new per-band quantiles.
  4. Report, per band of the evaluation set: how wide the quoted range is
     and how often it actually contains the true price (coverage).

Width metric: for log-target models the multiplicative ratio high/low
(a ×9 band means "high is 9x low"); for additive models (BMIC) the dollar
width high − low.

Run:  python case-studies/benchmark_banded_intervals.py [bmic|fasteners|cars]
      (no argument = all three sequentially)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from p2predict.intervals import compute_calibration_residuals, predict_interval
from p2predict.outliers import apply_feature_outlier_policy, apply_outlier_policy
from p2predict.prepare_data import Get_Column_Types
from p2predict.training import resolve_log_target, start_training

DATASETS = {
    "bmic": dict(
        csv=HERE / "battery-management-ics/data/bmics_clean.csv",
        target="unit_price_at_1_usd",
        features=["manufacturer", "Battery Chemistry", "Interface",
                  "max_cells_supported", "op_temp_min_C", "op_temp_max_C",
                  "package_pins", "is_multi_cell"],
        outliers="warn", feature_outliers="warn", log_target="auto",
        budget="thorough",
    ),
    "fasteners": dict(
        csv=HERE / "aerospace-fasteners/data/bolts_clean.csv",
        target="unit_price_each_usd",
        features=["material", "head_style", "thread_diameter_in", "length_in",
                  "thread_class", "thread_series", "finish",
                  "tensile_strength_psi", "threads_per_inch",
                  "width_across_flats_in"],
        outliers="warn", feature_outliers="warn", log_target="on",
        budget="fast",
    ),
    "cars": dict(
        csv=HERE / "used-cars/data/vehicles_training.csv",
        target="price",
        features=["year", "odometer", "manufacturer", "condition", "fuel",
                  "transmission", "drive", "type", "state", "paint_color"],
        outliers="warn", feature_outliers="drop", log_target="auto",
        budget="fast",
    ),
}


def width(iv, log_space: bool) -> float:
    return iv.high / iv.low if log_space else iv.high - iv.low


def run(name: str) -> None:
    cfg = DATASETS[name]
    print("=" * 76)
    print(f"{name.upper()} — {cfg['csv'].name}")
    print("=" * 76)
    t0 = time.time()

    df = pd.read_csv(cfg["csv"])
    df = df[df[cfg["target"]].notna()]
    df, _ = apply_outlier_policy(df, cfg["target"], policy=cfg["outliers"])
    numeric_feats = [c for c in cfg["features"]
                     if pd.api.types.is_numeric_dtype(df[c])]
    df, _ = apply_feature_outlier_policy(
        df, numeric_feats, policy=cfg["feature_outliers"])
    df = df.reset_index(drop=True)

    X, y = df[cfg["features"]], df[cfg["target"]]
    # 60/20/20: the first split mirrors the product's train/test split; the
    # product's test set then becomes calibration, and we carve a final
    # evaluation set out of the training pool? No — evaluation must be
    # untouched: split 80/20 first (train+cal / eval), then 75/25 inside.
    X_pool, X_eval, y_pool, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_pool, y_pool, test_size=0.25, random_state=0)
    num, cat = Get_Column_Types(X_train)

    log_target, decision = resolve_log_target(y_train, mode=cfg["log_target"])
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="xgboost",
        budget=cfg["budget"], tune=True, log_target=log_target,
    )
    print(f"rows: train {len(X_train):,} / calibration {len(X_cal):,} / "
          f"evaluation {len(X_eval):,}   log-target: {log_target} ({decision})"
          f"   trained in {time.time()-t0:.0f}s")

    calibration = compute_calibration_residuals(model, X_cal, y_cal)
    legacy = {k: v for k, v in calibration.items() if k != "predictions"}
    banded = predict_interval(model, X_eval, calibration, coverage=0.90)
    global_ = predict_interval(model, X_eval, legacy, coverage=0.90)
    actual = y_eval.to_numpy()
    in_log = bool(calibration["in_log_space"])
    unit = "x" if in_log else "$"

    rows = []
    bands = sorted({iv.band for iv in banded}, key=lambda b: (b is None, str(b)))
    for band in bands:
        idx = [i for i, iv in enumerate(banded) if iv.band == band]
        cov_b = np.mean([banded[i].low <= actual[i] <= banded[i].high for i in idx])
        cov_g = np.mean([global_[i].low <= actual[i] <= global_[i].high for i in idx])
        w_b = np.median([width(banded[i], in_log) for i in idx])
        w_g = np.median([width(global_[i], in_log) for i in idx])
        rows.append((band or "(global fallback)", len(idx), w_g, cov_g, w_b, cov_b))

    print(f"\n{'band':<34} {'n':>5}  {'global width':>13} {'cov':>6}"
          f"  {'banded width':>13} {'cov':>6}")
    for band, n, w_g, cov_g, w_b, cov_b in rows:
        print(f"{band:<34} {n:>5}  {unit}{w_g:>11,.1f} {cov_g*100:>5.1f}%"
              f"  {unit}{w_b:>11,.1f} {cov_b*100:>5.1f}%")

    cov_all_b = np.mean([iv.low <= a <= iv.high for iv, a in zip(banded, actual)])
    cov_all_g = np.mean([iv.low <= a <= iv.high for iv, a in zip(global_, actual)])
    print(f"\noverall coverage: global {cov_all_g*100:.1f}%  "
          f"banded {cov_all_b*100:.1f}%  (target 90%)")
    print()


def main() -> None:
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        run(name)


if __name__ == "__main__":
    main()
