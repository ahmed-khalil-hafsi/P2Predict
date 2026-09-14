"""Can P2Predict tell a buyer how many more parts would actually help?

Reproduction script for `research/learning_curve.md`. Read-only: it trains
throwaway models in memory and never writes a model or touches core.

A category manager with 150 parts and a +/-22% likely-range has exactly one
lever P2Predict never mentions: **collect more parts.** Today the tool says
how good the model is; it never says whether that is a ceiling or a
starting point. "Another 100 parts would take you to about +/-15%" is
actionable in a way that "R^2 = 0.71" is not.

But a learning curve is only a feature if it can be *extrapolated*, so this
script runs both halves:

Experiment 1 -- does the curve exist, and is it steep in the 50-300 range?
    Ground truth. Build catalogs at each size, grade on a disjoint 3,000-row
    evaluation sample, and record both the point error (median APE) and the
    conformal half-width, which is the number a buyer actually reads.

Experiment 2 -- can it be predicted from the rows the buyer already has?
    The honest test. Using *only* an n-part catalog, estimate the curve by
    internal subsampled cross-validation at sizes up to n, fit the classic
    power law `err(m) = a * m^-b + c`, and extrapolate to 2n and 3n. Then
    compare against Experiment 1's ground truth at those sizes. What matters
    is not the absolute level but the **predicted gain**: if the tool says
    "doubling gets you 6 points better", is it right?

Calibration here uses K-fold cross-conformal (see `small_n_conformal.md`),
because split conformal's half-width at n=50 is too unstable to read a
curve through.

Run from the repo root:  .venv/bin/python research/learning_curve.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import small_n_harness as H  # noqa: E402

from p2predict.intervals import _conformal_quantile  # noqa: E402
from p2predict.training import build_pipeline  # noqa: E402

warnings.filterwarnings("ignore")

COVERAGE = 0.90
ALPHA = 1.0 - COVERAGE
K_FOLDS = 5
ALGORITHMS = ("ridge", "xgboost")

# Experiment 1: the ground-truth curve.
# The large sizes are exactly 2x and 3x the anchors below, so Experiment 2
# has a ground-truth point to score its extrapolation against.
CURVE_SIZES = (50, 75, 100, 150, 200, 300, 450, 600, 900)
CURVE_TRIALS = 20

# Experiment 2: extrapolate from an n-part catalog to 2n and 3n.
ANCHORS = (150, 300)
EXTRAP_TRIALS = 15
# Fractions of the buyer's own catalog used to trace the internal curve.
INTERNAL_FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
INTERNAL_REPEATS = 3


def _fit(cat, X, y, algorithm):
    p = build_pipeline(algorithm, cat.numeric, cat.categorical, log_target=True)
    p.fit(X, y)
    return p


def _half_width_pct(cat, X, y, algorithm) -> float:
    """Cross-conformal half-width, as a percentage of the prediction."""
    kf = KFold(n_splits=min(K_FOLDS, len(y)), shuffle=True, random_state=0)
    resid = []
    for tr, te in kf.split(X):
        m = _fit(cat, X.iloc[tr], y[tr], algorithm)
        pred = np.asarray(m.predict(X.iloc[te]), dtype=float)
        ok = (pred > 0) & (y[te] > 0)
        resid.append(np.abs(np.log(y[te][ok]) - np.log(pred[ok])))
    r = np.concatenate(resid)
    if r.size == 0:
        return float("nan")
    q = _conformal_quantile(r, ALPHA)
    # Multiplicative bounds: pred*exp(-q) .. pred*exp(q). Half-width relative
    # to the prediction is therefore (exp(q) - exp(-q)) / 2.
    return float((np.exp(q) - np.exp(-q)) / 2.0 * 100)


def _internal_curve(cat, X, y, algorithm) -> tuple[np.ndarray, np.ndarray]:
    """Trace an error-vs-size curve using only the buyer's own rows."""
    n = len(y)
    rng = np.random.default_rng(0)
    sizes, errs = [], []
    for frac in INTERNAL_FRACTIONS:
        m = max(int(round(frac * n)), 3 * K_FOLDS)
        if m > n:
            m = n
        reps = []
        for _ in range(INTERNAL_REPEATS if m < n else 1):
            idx = rng.choice(n, size=m, replace=False)
            Xi, yi = X.iloc[idx], y[idx]
            kf = KFold(n_splits=min(K_FOLDS, m), shuffle=True, random_state=0)
            preds = np.full(m, np.nan)
            for tr, te in kf.split(Xi):
                mod = _fit(cat, Xi.iloc[tr], yi[tr], algorithm)
                preds[te] = np.asarray(mod.predict(Xi.iloc[te]), dtype=float)
            reps.append(H.median_ape(yi, preds))
        sizes.append(m)
        errs.append(float(np.mean(reps)))
    return np.array(sizes, float), np.array(errs, float)


def _power_law(m, a, b, c):
    return a * np.power(m, -b) + c


def _fit_power_law(sizes, errs):
    """Fit err(m) = a*m^-b + c. Returns the callable, or None if it won't fit."""
    ok = np.isfinite(sizes) & np.isfinite(errs)
    sizes, errs = sizes[ok], errs[ok]
    if sizes.size < 4:
        return None
    # c is the irreducible floor: bounded below by 0 and above by the best
    # error actually observed, so the fit cannot claim a floor it never saw.
    p0 = [float(errs[0] - errs[-1] + 1e-6) * sizes[0] ** 0.5, 0.5,
          float(max(errs.min() - 1.0, 0.0))]
    bounds = ([0.0, 0.05, 0.0], [np.inf, 2.0, float(errs.min())+1e-9])
    try:
        popt, _ = curve_fit(_power_law, sizes, errs, p0=p0, bounds=bounds,
                            maxfev=20000)
    except Exception:
        return None
    return lambda m: float(_power_law(float(m), *popt))


def experiment_1(cat, algorithm, rng) -> list[dict]:
    """The ground-truth curve: error and half-width vs catalog size."""
    rows = []
    for n in CURVE_SIZES:
        apes, hws = [], []
        for _ in range(CURVE_TRIALS):
            X, y, X_eval, y_eval = cat.draw(n, rng)
            try:
                model = _fit(cat, X, y, algorithm)
                pred = np.asarray(model.predict(X_eval), dtype=float)
                apes.append(H.median_ape(y_eval, pred))
                hws.append(_half_width_pct(cat, X, y, algorithm))
            except Exception as exc:
                print(f"    trial failed ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
        if not apes:
            continue
        rows.append({
            "n": n,
            "median_ape_mean": float(np.nanmean(apes)),
            "median_ape_sd": float(np.nanstd(apes, ddof=1)),
            "half_width_pct_mean": float(np.nanmean(hws)),
            "half_width_pct_sd": float(np.nanstd(hws, ddof=1)),
        })
    return rows


def experiment_2(cat, algorithm, truth: dict, rng) -> list[dict]:
    """Extrapolate from n to 2n/3n and score the predicted *gain*."""
    rows = []
    for anchor in ANCHORS:
        preds_2n, preds_3n, base = [], [], []
        fit_failures = 0
        for _ in range(EXTRAP_TRIALS):
            X, y, _, _ = cat.draw(anchor, rng)
            try:
                sizes, errs = _internal_curve(cat, X, y, algorithm)
            except Exception as exc:
                print(f"    trial failed ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                continue
            f = _fit_power_law(sizes, errs)
            if f is None:
                # A curve too flat or too noisy to fit is itself a result:
                # the feature must stay silent rather than invent a number.
                fit_failures += 1
                continue
            base.append(f(anchor))
            preds_2n.append(f(2 * anchor))
            preds_3n.append(f(3 * anchor))
        if not preds_2n:
            continue

        row = {"anchor_n": anchor, "trials": len(preds_2n),
               "fit_failures": fit_failures,
               "pct_fit_failed": float(
                   fit_failures / (fit_failures + len(preds_2n)) * 100)}
        for mult, preds in ((2, preds_2n), (3, preds_3n)):
            target_n = mult * anchor
            actual = truth.get(target_n)
            actual_base = truth.get(anchor)
            if actual is None or actual_base is None:
                continue
            pred_gain = np.array(base) - np.array(preds)
            actual_gain = actual_base - actual
            row[f"x{mult}"] = {
                "target_n": target_n,
                "predicted_mape_mean": float(np.mean(preds)),
                "actual_mape": float(actual),
                "level_error_pp": float(np.mean(preds) - actual),
                "predicted_gain_pp": float(np.mean(pred_gain)),
                "actual_gain_pp": float(actual_gain),
                "gain_error_pp": float(np.mean(pred_gain) - actual_gain),
                # Does it at least get the direction and rough size right?
                "pct_gain_within_3pp": float(np.mean(
                    np.abs(pred_gain - actual_gain) <= 3.0) * 100),
                "pct_gain_right_sign": float(np.mean(
                    np.sign(pred_gain) == np.sign(actual_gain)) * 100),
            }
        rows.append(row)
    return rows


def main() -> None:
    t0 = time.time()
    out = []
    for cat in H.load_all():
        for algorithm in ALGORITHMS:
            rng = np.random.default_rng(H.SEED)
            curve = experiment_1(cat, algorithm, rng)
            truth = {r["n"]: r["median_ape_mean"] for r in curve}
            extrap = experiment_2(cat, algorithm, truth, rng)
            out.append({"dataset": cat.label, "data_file": cat.source,
                        "algorithm": algorithm, "coverage_target": COVERAGE,
                        "curve": curve, "extrapolation": extrap})

            print(f"\n=== {cat.label} / {algorithm}")
            print(f"    {'catalog n':>10} {'median APE':>12} {'+/- half-width':>16}")
            for r in curve:
                print(f"    {r['n']:>10} {r['median_ape_mean']:>11.1f}% "
                      f"{r['half_width_pct_mean']:>15.1f}%")
            for r in extrap:
                for key in ("x2", "x3"):
                    e = r.get(key)
                    if not e:
                        continue
                    print(f"    from n={r['anchor_n']}, predicting n="
                          f"{e['target_n']}: gain said "
                          f"{e['predicted_gain_pp']:+.1f}pp, actual "
                          f"{e['actual_gain_pp']:+.1f}pp "
                          f"(error {e['gain_error_pp']:+.1f}pp, "
                          f"{e['pct_gain_within_3pp']:.0f}% of draws within 3pp)")

    dest = H.REPO / "research" / "learning_curve_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(H.REPO)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
