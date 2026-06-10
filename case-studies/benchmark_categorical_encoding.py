"""Benchmark: ordinal vs target encoding of categoricals for tree models.

Why this exists
---------------
P2Predict used to feed tree models (RandomForest/XGBoost) an ``OrdinalEncoder``,
which assigns each category an *arbitrary alphabetical integer*. A tree can only
make a threshold split on that integer, so it groups alphabetically-adjacent
categories into the same leaf — which destroys the signal for high-cardinality
*nominal* features (supplier / manufacturer / brand: the most common procurement
categorical). The smoking gun lived in the used-cars study: a 2021 like-new
**Tesla** priced at ~$6k (the code next to "toyota") instead of a realistic
~$40–55k.

This script reproduces the comparison across all three case-study datasets,
training the *same* XGBoost twice — once with ordinal codes, once with
``TargetEncoder`` (each category -> its cross-fitted, smoothed mean target, so
the code orders by price) — on an identical train/test split.

Run:  python case-studies/benchmark_categorical_encoding.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

XGB = dict(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=5, n_jobs=-1, random_state=0)

DATASETS = {
    "used-cars": dict(
        csv=HERE / "used-cars/data/vehicles_training.csv", target="price", log=True,
        features=["year", "odometer", "manufacturer", "condition", "fuel",
                  "transmission", "drive", "type", "state", "paint_color"],
        # a premium, sparse category and a commodity one to contrast
        probe={"year": 2021, "odometer": 22000, "manufacturer": "tesla",
               "condition": "like new", "fuel": "electric", "transmission": "other",
               "drive": "rwd", "type": "sedan", "state": "wa", "paint_color": "white"},
        probe_label="2021 like-new Tesla"),
    "fasteners": dict(
        csv=HERE / "aerospace-fasteners/data/bolts_clean.csv",
        target="unit_price_each_usd", log=True,
        features=["material", "head_style", "thread_diameter_in", "length_in",
                  "thread_class", "thread_series", "finish", "tensile_strength_psi",
                  "threads_per_inch", "width_across_flats_in"],
        probe=None, probe_label=None),
    "BMIC": dict(
        csv=HERE / "battery-management-ics/data/bmics_clean.csv",
        target="unit_price_at_1_usd", log=False,
        features=["manufacturer", "Battery Chemistry", "Interface",
                  "max_cells_supported", "op_temp_min_C", "op_temp_max_C",
                  "package_pins", "is_multi_cell"],
        probe=None, probe_label=None),
}


def _score(y_true, y_pred, log):
    if log:
        return r2_score(np.log(np.clip(y_true, 1e-9, None)),
                        np.log(np.clip(y_pred, 1e-9, None)))
    return r2_score(y_true, y_pred)


def run(name, cfg):
    df = pd.read_csv(cfg["csv"]).dropna(subset=[cfg["target"]])
    feats, target, log = cfg["features"], cfg["target"], cfg["log"]
    cats = [c for c in feats if df[c].dtype == object]
    X, y = df[feats].copy(), df[target].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    yt = np.log(ytr) if log else ytr.to_numpy()
    inv = np.exp if log else (lambda v: v)
    probe = pd.DataFrame([cfg["probe"]]) if cfg["probe"] else None

    def fit_eval(encoder_name):
        a, b, p = Xtr.copy(), Xte.copy(), (probe.copy() if probe is not None else None)
        if encoder_name == "ordinal":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            enc = TargetEncoder(target_type="continuous", smooth="auto")
        a[cats] = enc.fit_transform(Xtr[cats], yt)
        b[cats] = enc.transform(Xte[cats])
        m = XGBRegressor(**XGB).fit(a, yt)
        r2 = _score(yte, inv(m.predict(b)), log)
        pred = None
        if p is not None:
            p[cats] = enc.transform(probe[cats])
            pred = inv(m.predict(p))[0]
        return r2, pred

    r2_o, probe_o = fit_eval("ordinal")
    r2_t, probe_t = fit_eval("target")
    metric = "logR2" if log else "R2"
    line = (f"{name:<11} {len(df):>6,} rows, {len(cats)} cat cols   "
            f"{metric}: ordinal {r2_o:.3f} -> target {r2_t:.3f}")
    if probe_o is not None:
        line += (f"   |  {cfg['probe_label']}: "
                 f"ordinal ${probe_o:,.0f} -> target ${probe_t:,.0f}")
    print(line)


def main():
    print("=" * 92)
    print("Categorical encoding for tree models — ordinal codes vs target encoding")
    print("=" * 92)
    for name, cfg in DATASETS.items():
        if not cfg["csv"].exists():
            print(f"{name:<11} (skipped — {cfg['csv'].name} not present)")
            continue
        run(name, cfg)
    print("\nTakeaway: target encoding orders categories by price, so a single tree")
    print("split separates premium from commodity. It recovers the sparse premium")
    print("brand the ordinal code collapses, helps small data (BMIC) via smoothing,")
    print("and keeps a single numeric column per feature (SHAP/intervals unaffected).")


if __name__ == "__main__":
    sys.exit(main())
