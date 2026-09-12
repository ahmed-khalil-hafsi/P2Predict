"""Findings probe: retransformation bias in P2Predict's log-target models.

Question
--------
P2Predict wraps skewed targets with a log transform and reports dollars by
exponentiating the model's log-scale prediction. That back-transform is a
plain ``exp()`` with no bias correction. This probe asks three things, with
numbers rather than assertions:

  1. Is the resulting dollar prediction systematically biased, and in which
     sense — biased for the *mean* of price, the *median*, or both?
  2. Does the standard fix (Duan's 1983 smearing estimator) remove the
     *mean* bias, and what does it cost elsewhere (R², intervals, SHAP)?
  3. Because the honesty layer's ``residual_bias_p`` test checks the *mean*
     residual, does it end up labelling an otherwise-good, essentially
     *median-unbiased* log model "unreliable"?

Nothing here changes core code. It trains a throwaway model on the
committed 5k heavy-equipment sample so the finding is reproducible with no
Kaggle account, and (if present) additionally reports the on-disk 80k
case-study model as the real-world instance.

Run::

    python research/log_retransformation_bias.py

Writes ``research/log_retransformation_bias_results.json`` and prints a
summary.

Math, briefly
-------------
If the model predicts ``m(x) ≈ E[log Y | x]`` and reports ``exp(m(x))``,
then ``E[Y|x] = exp(m(x)) · E[exp(ε)|x]`` with ``E[exp(ε)] ≥ 1`` (Jensen),
so ``exp(m(x))`` under-estimates the dollar *mean*. But ``exp(E[log Y])`` is
the *geometric mean*, which for a right-skewed price sits at roughly the
*median* — so the same predictor is close to median-unbiased. Duan's
smearing multiplies by ``S = mean(exp(ε_train))`` to target the mean; the
log-normal analogue is ``exp(σ²/2)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from p2predict import (
    apply_feature_outlier_policy,
    apply_outlier_policy,
    auto_train,
    load_model,
    quality,
)
from p2predict.prepare_data import prepare_data
from p2predict.trained_model_io import load_csv_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MODELS_DIR = REPO_ROOT / "models"
SAMPLE_CSV = REPO_ROOT / "case-studies" / "heavy-equipment-sales" / "data-sample" / "bulldozers_sample.csv"
FEATURES = ["age_at_sale", "sale_year", "product_group", "product_size",
            "enclosure", "state"]
TARGET = "sale_price_usd"


def _analyse(name: str, model, X_test: pd.DataFrame, y_test: pd.Series,
             X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Compute the mean/median bias picture and the smearing correction."""
    y_test = np.asarray(y_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    yp = np.asarray(model.predict(X_test), dtype=float)
    yp_tr = np.asarray(model.predict(X_train), dtype=float)

    res = y_test - yp
    mean_res, median_res = float(res.mean()), float(np.median(res))

    # Duan smearing factor from TRAIN residuals in log space (correct
    # methodology: never estimated on the evaluation set).
    log_resid_tr = np.log(y_train) - np.log(yp_tr)
    smear = float(np.mean(np.exp(log_resid_tr)))
    sigma = float(np.std(log_resid_tr, ddof=1))
    lognormal_factor = float(np.exp(sigma ** 2 / 2.0))

    yp_corr = yp * smear
    res_corr = y_test - yp_corr

    def bias_p(pred):
        return float(quality.residual_bias_p(pd.Series(y_test), np.asarray(pred)))

    return {
        "model": name,
        "n_test": int(len(y_test)),
        "mean_actual": float(y_test.mean()),
        "raw": {
            "mean_pred": float(yp.mean()),
            "mean_residual": mean_res,
            "mean_residual_pct": 100 * mean_res / float(y_test.mean()),
            "median_residual": median_res,
            "median_residual_pct": 100 * median_res / float(np.median(y_test)),
            "residual_bias_p": bias_p(yp),
            "r2": float(r2_score(y_test, yp)),
        },
        "smearing": {
            "duan_factor": smear,
            "lognormal_factor": lognormal_factor,
            "log_resid_sigma": sigma,
        },
        "corrected": {
            "mean_pred": float(yp_corr.mean()),
            "mean_residual": float(res_corr.mean()),
            "mean_residual_pct": 100 * float(res_corr.mean()) / float(y_test.mean()),
            "residual_bias_p": bias_p(yp_corr),
            "r2": float(r2_score(y_test, yp_corr)),
        },
    }


def _print(block: dict) -> None:
    r, s, c = block["raw"], block["smearing"], block["corrected"]
    print(f"\n=== {block['model']}  (holdout n={block['n_test']:,}, "
          f"mean actual ${block['mean_actual']:,.0f}) ===")
    print(f"  RAW exp() back-transform:")
    print(f"    mean residual   : ${r['mean_residual']:>8,.0f}  "
          f"({r['mean_residual_pct']:+.1f}%)   <- biased for the MEAN")
    print(f"    median residual : ${r['median_residual']:>8,.0f}  "
          f"({r['median_residual_pct']:+.1f}%)   <- ~unbiased for the MEDIAN")
    print(f"    residual_bias_p : {r['residual_bias_p']:.3g}   "
          f"(honesty-layer mean test; < 0.05 => flagged 'biased')")
    print(f"    R2              : {r['r2']:.3f}")
    print(f"  Duan smearing factor x{s['duan_factor']:.4f}  "
          f"(log-normal analogue x{s['lognormal_factor']:.4f})")
    print(f"  AFTER smearing correction:")
    print(f"    mean residual   : ${c['mean_residual']:>8,.0f}  "
          f"({c['mean_residual_pct']:+.1f}%)")
    print(f"    residual_bias_p : {c['residual_bias_p']:.3g}   "
          f"({'PASS' if c['residual_bias_p'] >= 0.05 else 'still flagged'})")
    print(f"    R2              : {c['r2']:.3f}")


def main() -> None:
    results = {}

    # 1. Reproducible instance: train on the committed 5k sample.
    print("Training a throwaway log-target model on the committed 5k sample ...")
    df = pd.read_csv(SAMPLE_CSV)
    Xtr, Xte, ytr, yte, num, cat = prepare_data(df, FEATURES, TARGET, test_size=0.2)
    model, model_name, _params, log_target = auto_train(Xtr, ytr, num, cat, budget="fast")
    print(f"  won: {model_name}, log_target={log_target}")
    results["sample_5k"] = _analyse(f"5k sample ({model_name}, log_target={log_target})",
                                    model, Xte, yte, Xtr, ytr)
    _print(results["sample_5k"])

    # 2. Real-world instance: the on-disk 80k case-study model, if present.
    disk = sorted(MODELS_DIR.glob(f"*_{TARGET}_*.model"), key=lambda p: p.stat().st_mtime)
    if disk:
        loaded = load_model(disk[-1])
        m = loaded["model"]
        big = load_csv_file(str(REPO_ROOT / "case-studies" / "heavy-equipment-sales"
                                 / "data" / "bulldozers_training.csv")) \
            if (REPO_ROOT / "case-studies" / "heavy-equipment-sales" / "data"
                / "bulldozers_training.csv").exists() else None
        if big is not None:
            big, _ = apply_outlier_policy(big, TARGET, policy="warn")
            numf = [c for c in loaded["features"]
                    if pd.api.types.is_numeric_dtype(big[c])]
            big, _ = apply_feature_outlier_policy(big, numf, policy="warn")
            Xtr2, Xte2, ytr2, yte2, _n, _c = prepare_data(
                big, list(loaded["features"]), TARGET, test_size=0.2)
            results["case_study_80k"] = _analyse(
                f"80k case-study model ({loaded['model_name']})",
                m, Xte2, yte2, Xtr2, ytr2)
            _print(results["case_study_80k"])
        else:
            print("\n(80k training CSV not present — run prepare_data.py to "
                  "include the real-world instance. Skipping.)")
    else:
        print("\n(No on-disk sale_price_usd model — skipping real-world instance.)")

    out = HERE / "log_retransformation_bias_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
