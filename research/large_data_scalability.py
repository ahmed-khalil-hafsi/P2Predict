"""Reproduce the measurements in research/large_data_scalability.md.

Answers: does P2Predict still work when a category has hundreds of thousands
of rows rather than a few hundred parts?

Four probes, each printing what the write-up quotes:

  1. cost      — feature ranking and HPO tuning time at N rows.
  2. artifact  — how big the saved model gets, and how much of that is the
                 stored holdout / calibration lists.
  3. payload   — how big an MCP predict_from_csv response would be per row.
  4. bands     — whether more calibration data would buy tighter intervals if
                 the band count were allowed to grow past the pinned 3.

Run from the repo root::

    python research/large_data_scalability.py                 # 1-3, synthetic
    python research/large_data_scalability.py --rows 200000
    python research/large_data_scalability.py --bands models/<a-large>.model

Probe 4 needs a model trained on a large dataset (>= ~5k calibration points);
the write-up used the shipped heavy-equipment model. Probes 1-3 are synthetic
and self-contained. Timings are single runs on a dev Mac — order-of-magnitude
evidence, not benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """A procurement-shaped catalogue: 4 numeric specs, 3 categoricals, a
    40-supplier premium structure, multiplicative lognormal noise."""
    rng = np.random.default_rng(seed)
    suppliers = [f"SUP{i:03d}" for i in range(40)]
    df = pd.DataFrame({
        "weight_g": rng.lognormal(3, 0.6, n),
        "length_mm": rng.uniform(5, 200, n),
        "qty": rng.integers(1, 5000, n),
        "tolerance_um": rng.uniform(1, 50, n),
        "supplier": rng.choice(suppliers, n),
        "material": rng.choice(["steel", "alu", "brass", "peek", "nylon"], n),
        "finish": rng.choice(["none", "anod", "zinc"], n),
    })
    premium = df.supplier.map({s: 1 + (i % 7) * 0.08 for i, s in enumerate(suppliers)})
    df["price"] = (
        (0.4 + 0.02 * df.weight_g + 0.01 * df.length_mm - 0.00002 * df.qty)
        * premium * rng.lognormal(0, 0.25, n)
    )
    return df


# --------------------------------------------------------------------------
# 1. What does it cost to rank features and tune, at N rows?
# --------------------------------------------------------------------------

def probe_cost(df: pd.DataFrame, target: str = "price") -> None:
    from p2predict.feature_selection import (
        find_leaky_features,
        get_most_predictable_features,
    )
    from p2predict.prepare_data import prepare_data
    from p2predict.training import _tune, build_pipeline, should_log_target

    n = len(df)
    print(f"\n=== 1. cost at {n:,} rows x {df.shape[1] - 1} specs ===")

    t0 = time.perf_counter()
    find_leaky_features(df, target)
    print(f"find_leaky_features                {time.perf_counter() - t0:8.2f}s")

    # Ranking is a SCREENING step. If a subsample ranks the same, the full-data
    # fit is pure cost -- that is the claim the write-up makes.
    for rows in (n, n // 2, n // 4, n // 10):
        if rows < 500:
            continue
        sub = df if rows == n else df.sample(rows, random_state=0)
        t0 = time.perf_counter()
        ranked = get_most_predictable_features(sub, target, output_only_headers=True)
        print(f"get_most_predictable_features @{rows:>8,} {time.perf_counter() - t0:8.2f}s"
              f"  {list(ranked)}")
    print(f"peak RSS after ranking             {_peak_rss_mb():8.0f} MB")

    feats = [c for c in df.columns if c != target]
    X_train, _, y_train, _, num, cat = prepare_data(df, feats, target)
    log_target = should_log_target(y_train)
    total = 0.0
    for algo in ("ridge", "xgboost", "random_forest"):
        t0 = time.perf_counter()
        _tune(build_pipeline(algo, num, cat, log_target=log_target),
              X_train, y_train, algo, "fast", log_target)
        el = time.perf_counter() - t0
        total += el
        print(f"tune {algo:<14}                {el:8.2f}s")
    print(f"auto_train total (3 algorithms)    {total:8.2f}s")


# --------------------------------------------------------------------------
# 2. How much of the saved model is stored holdout / calibration?
# --------------------------------------------------------------------------

def probe_artifact(df: pd.DataFrame, target: str = "price") -> dict:
    import joblib

    from p2predict.domain import numeric_domain_from_frame
    from p2predict.intervals import compute_calibration_residuals
    from p2predict.prepare_data import prepare_data
    from p2predict.trained_model_io import SaveModel, Serialize_Trained_Model
    from p2predict.training import start_training

    print(f"\n=== 2. model artifact at {len(df):,} rows ===")
    feats = [c for c in df.columns if c != target]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(df, feats, target)
    model, _, log_target = start_training(X_train, y_train, num, cat, "xgboost")

    calibration = compute_calibration_residuals(model, X_test, y_test)
    metadata = Serialize_Trained_Model(
        "xgboost", feats, target, model, 0.9, log_target=log_target,
        background_sample=X_train.sample(min(100, len(X_train)), random_state=0),
        calibration=calibration,
        feature_domain=numeric_domain_from_frame(X_train),
    )
    # What the MCP train tool adds on top of the CLI artifact.
    metadata["holdout_y_test"] = y_test.tolist()
    metadata["holdout_y_pred"] = model.predict(X_test).tolist()

    full = os.path.join(SCRATCH, "p2p_scale_full.model")
    lean = os.path.join(SCRATCH, "p2p_scale_lean.model")
    SaveModel(metadata, full)
    stripped = {k: v for k, v in metadata.items()
                if k not in ("calibration", "holdout_y_test", "holdout_y_pred")}
    SaveModel(stripped, lean)

    t0 = time.perf_counter()
    joblib.load(full)
    load_s = time.perf_counter() - t0

    print(f"calibration points stored          {calibration['n_calibration']:>8,}")
    print(f"model file, as shipped             {os.path.getsize(full) / 1e6:8.2f} MB")
    print(f"model file, holdout+calibration stripped "
          f"{os.path.getsize(lean) / 1e6:6.2f} MB")
    print(f"model load                         {load_s:8.2f}s")
    return metadata


# --------------------------------------------------------------------------
# 3. How big would an MCP predict_from_csv response be?
# --------------------------------------------------------------------------

def probe_payload(df: pd.DataFrame, metadata: dict, target: str = "price",
                  sample_rows: int = 2000) -> None:
    """Rebuild predict_from_csv's per-row dict and measure it.

    The MCP layer json.dumps the whole list with no cap, so the per-row cost
    extrapolates linearly to whatever the user's CSV holds.
    """
    from p2predict.domain import check_part, training_domain

    print(f"\n=== 3. predict_from_csv payload ===")
    feats = metadata["features"]
    model = metadata["model"]
    rows_in = df.head(sample_rows)
    domain = training_domain(metadata)

    t0 = time.perf_counter()
    preds = model.predict(rows_in[feats])
    payload = []
    for i in range(len(rows_in)):
        row_input = {f: rows_in[f].iloc[i] for f in feats}
        payload.append({
            "input": row_input,
            "prediction": float(preds[i]),
            "in_domain": check_part(row_input, domain),
        })
    elapsed = time.perf_counter() - t0
    size = len(json.dumps(payload, default=str))

    per_row = size / len(rows_in)
    print(f"{len(rows_in):,} rows                        {size / 1e6:8.2f} MB "
          f"in {elapsed:.2f}s")
    for target_n in (100_000, 400_000):
        print(f"extrapolated to {target_n:>7,} rows    "
              f"{per_row * target_n / 1e6:8.0f} MB, "
              f"{elapsed / len(rows_in) * target_n:6.0f}s to assemble")


# --------------------------------------------------------------------------
# 4. Would more bands buy tighter intervals on a large calibration set?
# --------------------------------------------------------------------------

def probe_bands(model_path: str, coverage: float = 0.90, seed: int = 0) -> None:
    """Split the model's calibration set in half: build bands on one half,
    measure realised coverage and width on the other.

    This is the honest test of "does extra calibration data buy anything under
    a band count that is allowed to grow?" -- the pinned N_BANDS = 3 is the
    baseline row.
    """
    import joblib

    import p2predict.intervals as intervals

    print(f"\n=== 4. interval banding on {model_path} ===")
    metadata = joblib.load(model_path)
    calibration = metadata.get("calibration") or {}
    residuals = np.asarray(calibration.get("residuals", []), dtype=float)
    cal_preds = np.asarray(calibration.get("predictions", []), dtype=float)
    if residuals.size == 0 or cal_preds.size != residuals.size:
        print("model has no banded calibration data (pre-banding artifact) — skipped")
        return
    log_space = bool(calibration.get("in_log_space", False))
    print(f"calibration points {residuals.size:,} | log space {log_space}")

    alpha = 1.0 - coverage
    order = np.random.default_rng(seed).permutation(residuals.size)
    fit, evaluate = order[:residuals.size // 2], order[residuals.size // 2:]

    def width(q):
        # Report what the user sees: a +/- % of the prediction for a
        # multiplicative model, raw target units otherwise.
        return (np.exp(q) - np.exp(-q)) * 100 if log_space else 2 * q

    unit = "%" if log_space else " (target units)"
    q_global = intervals._conformal_quantile(residuals[fit], alpha)
    print(f"global      coverage {float((residuals[evaluate] <= q_global).mean() * 100):5.1f}%"
          f"  median width +/-{width(q_global):7.1f}{unit}")

    original = intervals.N_BANDS
    try:
        for n_bands in (3, 5, 10, 20):
            intervals.N_BANDS = n_bands
            built = intervals._build_bands(cal_preds[fit], residuals[fit], alpha)
            if built is None:
                print(f"{n_bands:2d} bands    not built (too few calibration points)")
                continue
            edges, q_hats, labels = built
            band_of = np.searchsorted(edges, cal_preds[evaluate], side="right")
            q = np.asarray(q_hats, dtype=float)[band_of]
            covered = residuals[evaluate] <= q
            widths = width(q)
            per_band = [
                (labels[k], float(covered[band_of == k].mean() * 100),
                 float(np.median(widths[band_of == k])), int((band_of == k).sum()))
                for k in range(n_bands) if (band_of == k).any()
            ]
            print(f"{n_bands:2d} bands    coverage {float(covered.mean() * 100):5.1f}%"
                  f"  median width +/-{float(np.median(widths)):7.1f}{unit}"
                  f"  | coverage spread {min(p[1] for p in per_band):5.1f}-"
                  f"{max(p[1] for p in per_band):5.1f}%"
                  f"  cheapest band +/-{per_band[0][2]:7.1f}{unit}")
            for label, cov, wide, n in per_band:
                print(f"              {label:34s} n={n:6,} cov {cov:5.1f}%"
                      f"  +/-{wide:7.1f}{unit}")
    finally:
        intervals.N_BANDS = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000,
                        help="synthetic dataset size for probes 1-3")
    parser.add_argument("--bands", metavar="MODEL_PATH", default=None,
                        help="run probe 4 against this saved model")
    parser.add_argument("--skip-cost", action="store_true",
                        help="skip probe 1 (the slow one: HPO tuning)")
    args = parser.parse_args()

    if args.bands:
        probe_bands(args.bands)
        return

    df = synthetic(args.rows)
    if not args.skip_cost:
        probe_cost(df)
    metadata = probe_artifact(df)
    probe_payload(df, metadata)


if __name__ == "__main__":
    main()
