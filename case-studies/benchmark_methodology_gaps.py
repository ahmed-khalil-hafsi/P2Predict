"""Quantify three methodology gaps on the case-study datasets (findings PR).

This script produces the numbers behind ``analysis/methodology_review.md``.
It changes nothing in core — it measures how much each gap actually costs on
the aerospace-fasteners and used-cars case studies, so the follow-up core
fixes can be sized by evidence rather than by principle.

Experiment 1 — retransformation (smearing) bias
    Log-target models back-transform with plain ``exp()``. Because
    ``E[exp(yhat + eps)] != exp(yhat)``, the price-space point prediction is
    systematically low by roughly ``E[exp(eps)]``. Duan's smearing estimator
    corrects it: multiply the naive prediction by ``S = mean(exp(eps_i))``
    over *out-of-sample* residuals (a boosted tree's training residuals are
    near zero, so S must come from data the model never fit — here a
    calibration split; in the product it could come from the holdout).
    Protocol: 80/20 pool/eval, then 75/25 train/cal inside the pool. Train
    exactly as the case study does (XGBoost, tuned). Report eval-set mean
    residual, MAE, R² and the bias t-test before and after smearing.

Experiment 2 — pre-split leakage
    The product computes outlier bounds and RF-importance feature selection
    on the FULL dataset before the train/test split, so the holdout has
    leaked into those decisions — and, for drop policies, the holdout itself
    is cleaned of the extremes production will see. Two mechanisms, measured
    separately with fixed (untuned) XGBoost so the only difference is where
    the decision was fitted:
      * selection — auto-mode top-6 features chosen on pool+eval (leaky,
        mirrors the product) vs pool only (clean); same eval set for both.
      * outliers  — drop-policy IQR bounds fitted on pool+eval (leaky) vs
        pool only (clean). Reported three ways: clean fit on raw eval
        (honest), leaky fit on raw eval (bounds leakage alone), leaky fit on
        bounds-cleaned eval (what the product reports today).
    Run at three catalog sizes (400 / 2,000 / full rows) because leakage is
    a small-data problem — the case studies are large, a typical procurement
    CSV is not.

Experiment 3 — residual-bias test space
    The trustworthy/unreliable verdict hangs on a one-sample t-test of RAW
    PRICE residuals (quality.residual_bias_p) even for log-target models,
    where residuals are heavily skewed. Reuse Experiment 1's model: report
    the p-value in price space (product today), in log space, and a Wilcoxon
    signed-rank test, plus the verdict each one implies.

Run:  python case-studies/benchmark_methodology_gaps.py [fasteners|cars]
      (no argument = both sequentially)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# sklearn 1.9 deprecation chatter from TargetEncoder inside the product
# pipeline; irrelevant to the measurements and it drowns the results.
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from p2predict.feature_selection import get_most_predictable_features
from p2predict.outliers import detect_outliers
from p2predict.prepare_data import Get_Column_Types
from p2predict.quality import assess_model, residual_bias_p
from p2predict.training import resolve_log_target, start_training

AUTO_MODE_MAX_FEATURES = 6  # cli/train.py auto-mode default

DATASETS = {
    "fasteners": dict(
        csv=HERE / "aerospace-fasteners/data/bolts_clean.csv",
        target="unit_price_each_usd",
        features=["material", "head_style", "thread_diameter_in", "length_in",
                  "thread_class", "thread_series", "finish",
                  "tensile_strength_psi", "threads_per_inch",
                  "width_across_flats_in"],
        log_target="on", budget="fast",
    ),
    "cars": dict(
        csv=HERE / "used-cars/data/vehicles_training.csv",
        target="price",
        features=["year", "odometer", "manufacturer", "condition", "fuel",
                  "transmission", "drive", "type", "state", "paint_color"],
        log_target="auto", budget="fast",
    ),
}

# (rows, seeds) — leakage shrinks with n, so measure at catalog sizes too.
LEAKAGE_SCALES = ((400, 10), (2000, 5), (None, 2))


def load(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg["csv"])
    return df[df[cfg["target"]].notna()].reset_index(drop=True)


def fit_xgb(X_train, y_train, cfg, tune):
    num, cat = Get_Column_Types(X_train)
    log_target, _ = resolve_log_target(y_train, mode=cfg["log_target"])
    model, _, log_target = start_training(
        X_train, y_train, num, cat, algorithm="xgboost",
        budget=cfg["budget"], tune=tune, log_target=log_target,
    )
    return model, log_target


# ---------------------------------------------------------------------------
# Experiments 1 & 3 — smearing bias and the bias-test space
# ---------------------------------------------------------------------------

def run_smearing_and_bias_test(name: str, cfg) -> None:
    print("-" * 76)
    print(f"[{name}] EXPERIMENT 1 — retransformation (smearing) bias")
    print("-" * 76)
    t0 = time.time()
    df = load(cfg)
    X, y = df[cfg["features"]], df[cfg["target"]]
    X_pool, X_eval, y_pool, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_pool, y_pool, test_size=0.25, random_state=0)

    model, log_target = fit_xgb(X_train, y_train, cfg, tune=True)
    if not log_target:
        print(f"log-target resolved OFF for {name}; smearing does not apply.")
        return
    print(f"rows: train {len(X_train):,} / cal {len(X_cal):,} / "
          f"eval {len(X_eval):,}   trained in {time.time() - t0:.0f}s")

    # Duan smearing factor from out-of-sample (calibration) log residuals.
    pred_cal = np.asarray(model.predict(X_cal), dtype=float)
    resid_cal = np.log(y_cal.to_numpy()) - np.log(pred_cal)
    smear = float(np.mean(np.exp(resid_cal)))
    lognormal = float(np.exp(np.var(resid_cal) / 2.0))

    y_true = y_eval.to_numpy(dtype=float)
    naive = np.asarray(model.predict(X_eval), dtype=float)
    smeared = naive * smear
    eval_factor = float(np.mean(np.exp(np.log(y_true) - np.log(naive))))

    print(f"smearing factor S — Duan (calibration): {smear:.4f}   "
          f"lognormal exp(sigma^2/2): {lognormal:.4f}   "
          f"same factor measured on eval: {eval_factor:.4f}   "
          f"(1.0 = no retransformation bias)")
    print(f"{'':<28} {'naive exp()':>14} {'x S (smeared)':>14}")
    rows = [
        ("mean residual $ (pred-act)", np.mean(naive - y_true), np.mean(smeared - y_true)),
        ("median residual $", np.median(naive - y_true), np.median(smeared - y_true)),
        ("MAE $", mean_absolute_error(y_true, naive), mean_absolute_error(y_true, smeared)),
        ("median APE %",
         np.median(np.abs(naive - y_true) / y_true) * 100,
         np.median(np.abs(smeared - y_true) / y_true) * 100),
        ("R^2 (price space)", r2_score(y_true, naive), r2_score(y_true, smeared)),
        ("bias t-test p (price)", residual_bias_p(y_true, naive), residual_bias_p(y_true, smeared)),
    ]
    for label, a, b in rows:
        print(f"{label:<28} {a:>14,.4f} {b:>14,.4f}")

    print()
    print("-" * 76)
    print(f"[{name}] EXPERIMENT 3 — residual-bias test space (naive predictions)")
    print("-" * 76)
    resid_price = y_true - naive
    resid_log = np.log(y_true) - np.log(naive)
    p_price_t = float(ttest_1samp(resid_price, 0.0).pvalue)
    p_log_t = float(ttest_1samp(resid_log, 0.0).pvalue)
    p_wilcoxon = float(wilcoxon(resid_price).pvalue)
    r2 = r2_score(y_true, naive)
    n = len(y_true)
    v_price = assess_model(r2, p_price_t, n)["verdict"]
    v_log = assess_model(r2, p_log_t, n)["verdict"]
    v_wilcoxon = assess_model(r2, p_wilcoxon, n)["verdict"]
    print(f"{'test':<38} {'p-value':>10}  verdict (gate at p<=0.05)")
    print(f"{'t-test, price space (product today)':<38} {p_price_t:>10.4f}  {v_price}")
    print(f"{'t-test, log space':<38} {p_log_t:>10.4f}  {v_log}")
    print(f"{'Wilcoxon signed-rank, price space':<38} {p_wilcoxon:>10.4f}  {v_wilcoxon}")
    print()


# ---------------------------------------------------------------------------
# Experiment 2 — pre-split leakage (selection + outlier bounds)
# ---------------------------------------------------------------------------

def iqr_bounds(frame: pd.DataFrame, columns) -> dict:
    """Tukey bounds per column, fitted on ``frame`` (mirrors detect_outliers)."""
    bounds = {}
    for col in columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        _, lower, upper = detect_outliers(frame[col])
        bounds[col] = (lower, upper)
    return bounds


def within_bounds(frame: pd.DataFrame, bounds: dict) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for col, (lower, upper) in bounds.items():
        if np.isnan(lower):
            continue
        mask &= frame[col].between(lower, upper) | frame[col].isna()
    return mask


def select_top_features(frame: pd.DataFrame, target: str) -> list[str]:
    ranked = get_most_predictable_features(frame, target, output_only_headers=True)
    return ranked.head(AUTO_MODE_MAX_FEATURES).tolist()


def eval_r2(model, X_eval, y_eval) -> float:
    return float(r2_score(y_eval.to_numpy(dtype=float),
                          np.asarray(model.predict(X_eval), dtype=float)))


def run_leakage(name: str, cfg) -> None:
    df_all = load(cfg)
    target = cfg["target"]
    features = cfg["features"]

    print("-" * 76)
    print(f"[{name}] EXPERIMENT 2 — pre-split leakage (fixed untuned XGBoost)")
    print("-" * 76)

    for n_rows, n_seeds in LEAKAGE_SCALES:
        n = len(df_all) if n_rows is None else min(n_rows, len(df_all))
        sel_deltas, out_honest, out_leaky_raw, out_leaky_clean = [], [], [], []
        sel_changed = 0
        t0 = time.time()
        for seed in range(n_seeds):
            df = df_all.sample(n=n, random_state=seed).reset_index(drop=True)
            pool_idx, eval_idx = train_test_split(
                df.index, test_size=0.2, random_state=seed)
            pool, eval_ = df.loc[pool_idx], df.loc[eval_idx]

            # -- mechanism A: feature selection fitted pre- vs post-split
            frame_cols = features + [target]
            top_leaky = select_top_features(df[frame_cols], target)
            top_clean = select_top_features(pool[frame_cols], target)
            sel_changed += int(set(top_leaky) != set(top_clean))
            r2_leaky = eval_r2(
                fit_xgb(pool[top_leaky], pool[target], cfg, tune=False)[0],
                eval_[top_leaky], eval_[target])
            r2_clean = eval_r2(
                fit_xgb(pool[top_clean], pool[target], cfg, tune=False)[0],
                eval_[top_clean], eval_[target])
            sel_deltas.append(r2_leaky - r2_clean)

            # -- mechanism B: drop-policy outlier bounds fitted pre- vs post-split
            numeric_feats = [c for c in features
                             if pd.api.types.is_numeric_dtype(df[c])]
            cols = numeric_feats + [target]
            b_leaky = iqr_bounds(df, cols)       # product: bounds see eval rows
            b_clean = iqr_bounds(pool, cols)
            pool_leaky = pool[within_bounds(pool, b_leaky)]
            pool_clean = pool[within_bounds(pool, b_clean)]
            eval_cleaned = eval_[within_bounds(eval_, b_leaky)]

            m_leaky = fit_xgb(pool_leaky[features], pool_leaky[target], cfg, tune=False)[0]
            m_clean = fit_xgb(pool_clean[features], pool_clean[target], cfg, tune=False)[0]
            out_honest.append(eval_r2(m_clean, eval_[features], eval_[target]))
            out_leaky_raw.append(eval_r2(m_leaky, eval_[features], eval_[target]))
            if len(eval_cleaned) >= 2:
                out_leaky_clean.append(
                    eval_r2(m_leaky, eval_cleaned[features], eval_cleaned[target]))

        def ms(vals):
            return f"{np.mean(vals):+.4f} ± {np.std(vals):.4f}"

        print(f"\nn={n:,} rows, {n_seeds} seed(s)  ({time.time() - t0:.0f}s)")
        print(f"  selection: eval R^2 (leaky top-6) − (clean top-6): {ms(sel_deltas)}"
              f"   [top-6 set differed in {sel_changed}/{n_seeds} seeds]")
        print(f"  outliers (drop): eval R^2 — honest (clean bounds, raw eval): "
              f"{np.mean(out_honest):.4f}")
        print(f"                   leaky bounds, raw eval:                     "
              f"{np.mean(out_leaky_raw):.4f}   (bounds leakage alone: "
              f"{np.mean(out_leaky_raw) - np.mean(out_honest):+.4f})")
        if out_leaky_clean:
            print(f"                   leaky bounds, CLEANED eval (product today): "
                  f"{np.mean(out_leaky_clean):.4f}   (reported-vs-honest gap: "
                  f"{np.mean(out_leaky_clean) - np.mean(out_honest):+.4f})")
    print()


def main() -> None:
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        cfg = DATASETS[name]
        print("=" * 76)
        print(f"{name.upper()} — {cfg['csv'].name}")
        print("=" * 76)
        run_smearing_and_bias_test(name, cfg)
        run_leakage(name, cfg)


if __name__ == "__main__":
    main()
