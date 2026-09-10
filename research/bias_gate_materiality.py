"""Is the 'unreliable' gate measuring bias, or measuring sample size?

Reproduction script for `research/bias_gate_materiality.md`. Read-only: it
trains throwaway models in memory and never writes a model or touches core.

Three experiments per dataset:
  1. price-space vs log-space bias test, and the verdict each produces
  2. effect size (mean/median residual %) alongside the p-value
  3. the same model judged on holdouts of growing size -- does the verdict
     track the bias, or track n?

Run from the repo root:  python research/bias_gate_materiality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p2predict.quality import assess_model, residual_bias_p  # noqa: E402
from p2predict.training import start_training  # noqa: E402

SEED = 11

# (label, csv (full), csv (committed sample fallback), target, features)
DATASETS = [
    ("used cars",
     "case-studies/used-cars/data/vehicles_training.csv",
     "case-studies/used-cars/data-sample/vehicles_sample.csv",
     "price",
     ["year", "odometer", "manufacturer", "condition", "fuel",
      "transmission", "drive", "type", "state", "paint_color"]),
    ("heavy equipment",
     "case-studies/heavy-equipment-sales/data/bulldozers_training.csv",
     "case-studies/heavy-equipment-sales/data-sample/bulldozers_sample.csv",
     "sale_price_usd",
     ["age_at_sale", "sale_year", "product_group", "product_size",
      "enclosure", "state"]),
    ("battery mgmt ICs",
     "case-studies/battery-management-ics/data/bmics_clean.csv",
     "case-studies/battery-management-ics/data-sample/bmics_sample.csv",
     "unit_price_at_1_usd",
     ["manufacturer", "Battery Chemistry", "Interface", "max_cells_supported",
      "op_temp_min_C", "op_temp_max_C", "package_pins", "is_multi_cell"]),
]


def log_space_bias_p(y_true, y_pred) -> float:
    """The same t-test, run where a log-target model is actually unbiased."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ok = (yt > 0) & (yp > 0)
    resid = np.log(yt[ok]) - np.log(yp[ok])
    return float(ttest_1samp(resid, 0.0).pvalue) if resid.size >= 2 else float("nan")


def fit(df: pd.DataFrame, target: str, features: list[str]):
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    X, y = df[features], df[target]
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(X[f])]
    categorical = [f for f in features if f not in numeric]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model, _, log_target = start_training(
        X_train, y_train, numeric, categorical, "xgboost", log_target=None
    )
    return model, X_test, y_test, log_target


def run(label: str, csv: str, sample: str, target: str, features: list[str]) -> dict:
    path = REPO / csv
    used = csv
    if not path.exists():
        path, used = REPO / sample, sample
    if not path.exists():
        print(f"skip {label}: no data", file=sys.stderr)
        return {}

    df = pd.read_csv(path, low_memory=False)
    model, X_test, y_test, log_target = fit(df, target, features)
    y_pred = np.asarray(model.predict(X_test), float)
    y_true = np.asarray(y_test, float)

    from sklearn.metrics import r2_score
    r2 = float(r2_score(y_true, y_pred))
    n = len(y_true)

    resid_pct = (y_true - y_pred) / y_true * 100
    mean_pct, median_pct = float(np.mean(resid_pct)), float(np.median(resid_pct))
    p_price = residual_bias_p(y_true, y_pred)
    p_log = log_space_bias_p(y_true, y_pred)

    now = assess_model(r2, p_price, n)["verdict"]
    in_log = assess_model(r2, p_log, n)["verdict"]

    print(f"\n=== {label}  ({used.split('/')[-1]}, log_target={log_target}, "
          f"n_holdout={n:,}, R2 {r2:.3f})")
    print(f"    mean residual   {mean_pct:+.1f}%      median residual {median_pct:+.1f}%")
    print(f"    bias p, price space {p_price:.2e}  -> verdict '{now}'")
    print(f"    bias p, log space   {p_log:.2e}  -> verdict '{in_log}'")

    # Experiment 3: hold the model and its bias FIXED, vary only the holdout
    # size. 200 random draws per size, so the rate is not one lucky sample.
    print(f"    same model, same bias -- only n varies (200 draws each)")
    print(f"    {'n holdout':>10} {'median mean-resid':>18} {'median bias p':>14} "
          f"{'% unreliable':>13}")
    ladder = []
    rng = np.random.default_rng(SEED)
    for k in [50, 100, 250, 500, 1000, 2500, 5000, 10000, n]:
        if k > n:
            continue
        draws = 1 if k == n else 200
        ps, ms, unreliable = [], [], 0
        for _ in range(draws):
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            p_k = residual_bias_p(y_true[idx], y_pred[idx])
            ps.append(p_k)
            ms.append(float(np.mean((y_true[idx] - y_pred[idx]) / y_true[idx] * 100)))
            unreliable += assess_model(r2, p_k, k)["verdict"] == "unreliable"
        pct = unreliable / draws * 100
        ladder.append({"n": int(k), "draws": draws,
                       "median_mean_resid_pct": float(np.median(ms)),
                       "median_bias_p": float(np.median(ps)),
                       "pct_flagged_unreliable": pct})
        print(f"    {k:>10,} {np.median(ms):>17.1f}% {np.median(ps):>14.2e} "
              f"{pct:>12.0f}%")

    return {"dataset": label, "data_file": used, "log_target": bool(log_target),
            "n_holdout": n, "r2": r2, "mean_residual_pct": mean_pct,
            "median_residual_pct": median_pct, "bias_p_price_space": p_price,
            "bias_p_log_space": p_log, "verdict_today": now,
            "verdict_log_space_test": in_log, "holdout_ladder": ladder}


def main() -> None:
    out = [r for r in (run(*d) for d in DATASETS) if r]
    dest = REPO / "research" / "bias_gate_materiality_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
