"""Does the likely-range hold up on a 150-part catalog?

Reproduction script for `research/small_n_conformal.md`. Read-only: it
trains throwaway models in memory and never writes a model or touches core.

P2Predict calibrates its likely-range with **split conformal** — fit on 80%
of the rows, take the residuals on the 20% holdout, and read a quantile off
them. On the 16k-400k-row case studies that holdout is thousands of rows and
the quantile is rock solid. On the 50-300-part catalog the project is
actually built for, the holdout is 10-60 rows, and the 90% quantile of 30
residuals is the third-largest one. This script asks what that costs, and
whether the cross-validation conformal methods of Barber, Candes, Ramdas &
Tibshirani (2021, *Ann. Statist.* 49(1)) buy it back.

Three calibrations, graded against a disjoint 3,000-row evaluation sample:

  split     what ships today. Model trained on 0.8n; q-hat from the 0.2n
            holdout residuals.
  cv_cross  K-fold cross-conformal. Out-of-fold residuals over all n rows
            give q-hat; the shipped model is trained on all n. Cheap: only
            the residual list changes, so the persisted calibration dict
            keeps its current shape.
  cv_plus   CV+ proper (BCRT 2021, Sec. 3). Keeps the K fold-models and
            builds the interval from the ensemble. Carries the theoretical
            1-2*alpha worst-case guarantee, at the cost of persisting K
            models instead of one.

Reported per method: empirical coverage, median half-width, and -- the
number this study is really about -- the **spread of the half-width across
resamples**, i.e. how much the range a buyer is quoted depends on which
rows happened to land in the holdout.

Run from the repo root:  .venv/bin/python research/small_n_conformal.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import small_n_harness as H  # noqa: E402

from p2predict.intervals import _conformal_quantile  # noqa: E402
from p2predict.training import start_training  # noqa: E402

warnings.filterwarnings("ignore")

COVERAGE = 0.90
ALPHA = 1.0 - COVERAGE
K_FOLDS = 10
TRIALS = 30
ALGORITHMS = ("ridge", "xgboost")


def _fit(X, y, cat, algorithm):
    """Fit one pipeline with the log-target wrap forced on.

    Forced rather than auto so half-widths are directly comparable across
    trials: in log space the conformal bound is multiplicative, so a
    half-width reads as a percentage of the price regardless of whether the
    trial happened to draw a skewed sample. Every target here is a strictly
    positive price, which is the case the wrap exists for.
    """
    model, _, _ = start_training(
        X, y, cat.numeric, cat.categorical, algorithm, log_target=True
    )
    return model


def _log_resid(model, X, y) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=float)
    ok = (pred > 0) & (y > 0)
    out = np.full(y.shape, np.nan)
    out[ok] = np.abs(np.log(y[ok]) - np.log(pred[ok]))
    return out


def _grade(low, high, pred, y_eval) -> dict:
    """Coverage and half-width of one interval set on the eval sample."""
    ok = np.isfinite(low) & np.isfinite(high) & np.isfinite(pred) & (pred > 0)
    covered = (y_eval[ok] >= low[ok]) & (y_eval[ok] <= high[ok])
    # Half-width as a percentage of the prediction: the "+/- x%" a buyer sees.
    half_pct = (high[ok] - low[ok]) / 2.0 / pred[ok] * 100
    return {
        "coverage": float(np.mean(covered)),
        "half_width_pct": float(np.median(half_pct)),
        "median_ape": H.median_ape(y_eval[ok], pred[ok]),
    }


def _split_conformal(cat, X, y, X_eval, y_eval, algorithm) -> dict:
    """What ships today."""
    X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.2, random_state=0)
    model = _fit(X_tr, y_tr, cat, algorithm)
    resid = _log_resid(model, X_cal, y_cal)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return {}
    q = _conformal_quantile(resid, ALPHA)
    pred = np.asarray(model.predict(X_eval), dtype=float)
    out = _grade(pred * np.exp(-q), pred * np.exp(q), pred, y_eval)
    out["n_calibration"] = int(resid.size)
    return out


def _oof(cat, X, y, algorithm):
    """K-fold out-of-fold residuals, plus the fold models and fold index."""
    kf = KFold(n_splits=min(K_FOLDS, len(y)), shuffle=True, random_state=0)
    resid = np.full(len(y), np.nan)
    fold_of = np.zeros(len(y), dtype=int)
    models = []
    for k, (tr, te) in enumerate(kf.split(X)):
        m = _fit(X.iloc[tr], y[tr], cat, algorithm)
        resid[te] = _log_resid(m, X.iloc[te], y[te])
        fold_of[te] = k
        models.append(m)
    return resid, fold_of, models


def _cv_cross(cat, X, y, X_eval, y_eval, algorithm, oof) -> dict:
    """K-fold cross-conformal: OOF residuals, model trained on all n rows."""
    resid, _, _ = oof
    r = resid[np.isfinite(resid)]
    if r.size == 0:
        return {}
    q = _conformal_quantile(r, ALPHA)
    full = _fit(X, y, cat, algorithm)
    pred = np.asarray(full.predict(X_eval), dtype=float)
    out = _grade(pred * np.exp(-q), pred * np.exp(q), pred, y_eval)
    out["n_calibration"] = int(r.size)
    return out


def _cv_plus(cat, X, y, X_eval, y_eval, algorithm, oof) -> dict:
    """CV+ (BCRT 2021 Sec. 3): interval from the fold-model ensemble.

    For test point x the interval endpoints are order statistics of
    ``mu_{-k(i)}(x) -/+ R_i`` over the n training points, where ``R_i`` is
    point i's out-of-fold residual and ``mu_{-k(i)}`` is the model fitted
    without i's fold. All arithmetic is in log space; the endpoints are
    exponentiated back to price space at the end.
    """
    resid, fold_of, models = oof
    keep = np.isfinite(resid)
    n = int(keep.sum())
    if n == 0:
        return {}

    # (K, n_eval) log-space predictions, one row per fold model.
    fold_pred = np.vstack([
        np.log(np.clip(np.asarray(m.predict(X_eval), dtype=float), 1e-12, None))
        for m in models
    ])
    mu = fold_pred[fold_of[keep]]           # (n, n_eval)
    r = resid[keep][:, None]                # (n, 1)

    lo_sorted = np.sort(mu - r, axis=0)
    hi_sorted = np.sort(mu + r, axis=0)
    k_lo = int(np.floor(ALPHA * (n + 1)))
    k_hi = int(np.ceil((1.0 - ALPHA) * (n + 1)))
    low = lo_sorted[max(k_lo - 1, 0)]
    high = hi_sorted[min(k_hi, n) - 1]

    # The point estimate CV+ pairs with is the fold-model ensemble median.
    pred = np.exp(np.median(fold_pred, axis=0))
    out = _grade(np.exp(low), np.exp(high), pred, y_eval)
    out["n_calibration"] = n
    return out


def run_cell(cat: H.Catalog, n: int, algorithm: str, rng) -> dict:
    """TRIALS resamples of an n-part catalog; aggregate each method."""
    acc: dict[str, list[dict]] = {"split": [], "cv_cross": [], "cv_plus": []}
    for _ in range(TRIALS):
        X, y, X_eval, y_eval = cat.draw(n, rng)
        try:
            oof = _oof(cat, X, y, algorithm)
            results = {
                "split": _split_conformal(cat, X, y, X_eval, y_eval, algorithm),
                "cv_cross": _cv_cross(cat, X, y, X_eval, y_eval, algorithm, oof),
                "cv_plus": _cv_plus(cat, X, y, X_eval, y_eval, algorithm, oof),
            }
        except Exception as exc:  # a degenerate draw shouldn't kill the run
            print(f"    trial failed ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        for name, res in results.items():
            if res:
                acc[name].append(res)

    summary = {}
    for name, rows in acc.items():
        if not rows:
            continue
        cov = np.array([r["coverage"] for r in rows])
        hw = np.array([r["half_width_pct"] for r in rows])
        ape = np.array([r["median_ape"] for r in rows])
        summary[name] = {
            "trials": len(rows),
            "n_calibration": int(np.median([r["n_calibration"] for r in rows])),
            "coverage_mean": float(cov.mean()),
            "coverage_sd": float(cov.std(ddof=1)) if len(cov) > 1 else 0.0,
            "coverage_min": float(cov.min()),
            "half_width_pct_mean": float(hw.mean()),
            # The headline instability number: how much the quoted range
            # moves when only the luck of the split changes.
            "half_width_pct_sd": float(hw.std(ddof=1)) if len(hw) > 1 else 0.0,
            "half_width_pct_iqr": float(np.percentile(hw, 75) - np.percentile(hw, 25)),
            "median_ape_mean": float(ape.mean()),
        }
    return summary


def main() -> None:
    t0 = time.time()
    out = []
    for cat in H.load_all():
        for algorithm in ALGORITHMS:
            for n in H.CATALOG_SIZES:
                rng = np.random.default_rng(H.SEED)
                summary = run_cell(cat, n, algorithm, rng)
                if not summary:
                    continue
                out.append({"dataset": cat.label, "data_file": cat.source,
                            "algorithm": algorithm, "catalog_n": n,
                            "coverage_target": COVERAGE, "k_folds": K_FOLDS,
                            "trials": TRIALS, "methods": summary})
                print(f"\n=== {cat.label} / {algorithm} / n={n} "
                      f"({TRIALS} resamples, target coverage {COVERAGE:.0%})")
                print(f"    {'method':<10} {'cal n':>6} {'coverage':>10} "
                      f"{'cov sd':>8} {'half-width':>12} {'hw sd':>8} {'mAPE':>7}")
                for name in ("split", "cv_cross", "cv_plus"):
                    s = summary.get(name)
                    if not s:
                        continue
                    print(f"    {name:<10} {s['n_calibration']:>6} "
                          f"{s['coverage_mean']:>9.1%} {s['coverage_sd']:>8.1%} "
                          f"{s['half_width_pct_mean']:>11.1f}% "
                          f"{s['half_width_pct_sd']:>7.1f} "
                          f"{s['median_ape_mean']:>6.1f}%")

    dest = H.REPO / "research" / "small_n_conformal_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(H.REPO)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
