"""Probe: does anything in the predict path notice an out-of-domain part?

Reproduction script for `research/out_of_domain_flag.md`. Read-only — it never
writes a model or touches core. Two parts:

  Part A  a small synthetic bracket-pricing model, trained here, good enough
          that the shipped 'trust' verdict is reachable. Self-contained: no
          Kaggle account, no case-study downloads, no files in models/.
  Part B  the real case-study models, if they are present locally.

Run from the repo root:  python research/out_of_domain_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p2predict.intervals import (  # noqa: E402
    compute_calibration_residuals,
    predict_interval,
)
from p2predict.model_utils import extract_feature_info, inner_pipeline  # noqa: E402
from p2predict.quality import interval_reliability, interval_say_to_user  # noqa: E402
from p2predict.training import start_training  # noqa: E402

COVERAGE = 0.90


def verdict_row(model, calibration, features, feat: dict) -> dict:
    x = pd.DataFrame([feat])[features]
    ir = predict_interval(model, x, calibration, coverage=COVERAGE)[0]
    return {
        "prediction": ir.prediction,
        "low": ir.low,
        "high": ir.high,
        "band": ir.band,
        "width_pct_of_pred": (ir.high - ir.low) / ir.prediction * 100,
        "reliability": interval_reliability(ir.low, ir.prediction, ir.high),
        "say_to_user": interval_say_to_user(ir.low, ir.prediction, ir.high),
    }


def print_table(rows: list[dict]) -> None:
    print(f"    {'part':34} {'price':>10} {'likely range':>22} {'width':>7}  verdict")
    for r in rows:
        rng = f"{r['low']:,.2f} - {r['high']:,.2f}"
        print(f"    {r['variant']:34} {r['prediction']:>10,.2f} {rng:>22} "
              f"{r['width_pct_of_pred']:>6.0f}%  {r['reliability']}")


# ---------------------------------------------------------------- Part A
def synthetic_case() -> dict:
    """A well-fit model on parts whose price is a clean function of their specs.

    Deliberately easy: strong signal, 5% noise. That is the point — it is the
    regime where the reliability flag is *supposed* to earn a buyer's trust.
    """
    rng = np.random.default_rng(7)
    n = 1200
    mass = rng.uniform(0.2, 4.0, n)
    holes = rng.integers(2, 13, n)
    tol = rng.choice([0.05, 0.1, 0.25], n)
    material = rng.choice(
        ["Aluminium 6061", "Steel S235", "Stainless 316"], n, p=[0.5, 0.3, 0.2]
    )
    supplier = rng.choice(["Alpha", "Bravo", "Charlie", "Delta"], n)
    mat_f = pd.Series(material).map(
        {"Aluminium 6061": 1.0, "Steel S235": 0.85, "Stainless 316": 1.9}
    ).to_numpy()
    sup_f = pd.Series(supplier).map(
        {"Alpha": 1.0, "Bravo": 1.12, "Charlie": 0.94, "Delta": 1.3}
    ).to_numpy()
    price = (8 + 11 * mass + 1.4 * holes + 3.0 / tol) * mat_f * sup_f * rng.normal(1, 0.05, n)

    df = pd.DataFrame({
        "mass_kg": mass, "machined_holes": holes, "tolerance_mm": tol,
        "material": material, "supplier": supplier, "price_eur": price,
    })
    numeric = ["mass_kg", "machined_holes", "tolerance_mm"]
    categorical = ["material", "supplier"]
    features = numeric + categorical

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["price_eur"], test_size=0.25, random_state=7
    )
    model, _, _ = start_training(
        X_train, y_train, numeric, categorical, "random_forest", log_target=True
    )
    calibration = compute_calibration_residuals(model, X_test, y_test)

    base = {"mass_kg": 1.5, "machined_holes": 6, "tolerance_mm": 0.1,
            "material": "Aluminium 6061", "supplier": "Alpha"}
    variants = {
        "in-domain: 1.5kg alu bracket": base,
        "unseen supplier (new vendor)": base | {"supplier": "Zeta Werke"},
        "unseen material (titanium)": base | {"material": "Titanium Grade 5"},
        "900 kg (225x observed max)": base | {"mass_kg": 900},
        "everything impossible": {
            "mass_kg": 900, "machined_holes": 4000, "tolerance_mm": 0.00001,
            "material": "Unobtainium", "supplier": "Zeta Werke",
        },
    }

    rows = [{"variant": k} | verdict_row(model, calibration, features, v)
            for k, v in variants.items()]

    r2 = float(model.score(X_test, y_test))
    print(f"\n=== Part A: synthetic brackets (random_forest, holdout R2 {r2:.3f}, "
          f"{len(df):,} rows)")
    print(f"    observed price EUR {df['price_eur'].min():.2f}-{df['price_eur'].max():.2f}; "
          f"mass 0.2-4.0 kg, holes 2-12, tolerance 0.05-0.25 mm")
    print_table(rows)
    print(f"\n    the impossible part is described to the agent as:\n"
          f"      \"{rows[-1]['say_to_user']}\"")

    return {"case": "synthetic brackets", "algorithm": "random_forest",
            "holdout_r2": r2, "training_rows": len(df),
            "n_calibration": calibration["n_calibration"], "results": rows}


# ---------------------------------------------------------------- Part B
CASES = [
    ("used cars", "models/xgboost_price_20260610_152137.model",
     "case-studies/used-cars/data/vehicles_training.csv", "manufacturer"),
    ("aerospace fasteners", "models/xgboost_unit_price_each_usd_20260626_132941.model",
     "case-studies/aerospace-fasteners/data/bolts_clean.csv", "material"),
    ("heavy equipment", "models/xgboost_sale_price_usd_20260831_155833.model",
     "case-studies/heavy-equipment-sales/data/bulldozers_training.csv", "product_group"),
]

UNSEEN = "___NOT_IN_ANY_CATALOG___"


def real_case(label: str, model_path: str, csv_path: str, swap_col: str) -> dict:
    bundle = joblib.load(REPO / model_path)
    model, features = bundle["model"], bundle["features"]
    calibration = bundle.get("calibration")
    ftypes, categories = extract_feature_info(inner_pipeline(model))

    df = pd.read_csv(REPO / csv_path, low_memory=False)
    numeric = [f for f in features if ftypes.get(f) == "Numerical"]
    domain = {f: (float(pd.to_numeric(df[f], errors="coerce").min()),
                  float(pd.to_numeric(df[f], errors="coerce").max()))
              for f in numeric}

    base = {f: (float(pd.to_numeric(df[f], errors="coerce").median())
                if ftypes.get(f) == "Numerical" else df[f].mode().iloc[0])
            for f in features}

    variants = {"in-domain baseline": base, f"unseen {swap_col}": base | {swap_col: UNSEEN}}
    if numeric:
        f0 = numeric[0]
        variants[f"{f0} = 2x observed max"] = base | {f0: domain[f0][1] * 2}
        variants[f"{f0} = 50x observed max"] = base | {f0: domain[f0][1] * 50}
    variants["every spec out of domain"] = {
        f: (domain[f][1] * 50 if ftypes.get(f) == "Numerical" else UNSEEN) for f in features
    }

    rows = []
    for name, feat in variants.items():
        rows.append({"variant": name} | verdict_row(model, calibration, features, feat))

    target = bundle["target_feature"]
    r2 = float(bundle["r2"])
    print(f"\n=== Part B: {label} ({bundle.get('model_name')}, holdout R2 {r2:.3f}, "
          f"{len(df):,} rows)")
    print(f"    target {target} observed "
          f"{df[target].min():,.0f}-{df[target].max():,.0f}")
    print_table(rows)

    return {"case": label, "model": Path(model_path).name,
            "algorithm": bundle.get("model_name"), "target": target,
            "log_target": bundle.get("log_target"), "holdout_r2": r2,
            "n_calibration": calibration.get("n_calibration"),
            "training_rows": len(df), "numeric_domain": domain,
            "n_known_categories": {c: len(v) for c, v in categories.items()},
            "results": rows}


def main() -> None:
    out = [synthetic_case()]
    for label, model_path, csv_path, swap_col in CASES:
        if not (REPO / model_path).exists() or not (REPO / csv_path).exists():
            print(f"\n=== Part B: {label} — skipped (model or data not present locally)")
            continue
        out.append(real_case(label, model_path, csv_path, swap_col))

    dest = REPO / "research" / "out_of_domain_probe_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
