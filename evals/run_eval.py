"""Run P2Predict against a TDC ADMET benchmark under the official protocol.

Protocol (TDC's, not ours -- this is the point):
  * TDC supplies the scaffold-split train_val / test partition. We never
    choose a split, so we can't accidentally choose a favourable one.
  * For each of 5 seeds, TDC hands us a train/valid partition of train_val.
  * We fit on `train` only. `valid` is used for conformal calibration --
    its intended purpose, and it never touches model fitting.
  * We predict `test` and hand the predictions to group.evaluate_many(),
    which computes the leaderboard metric and its across-seed spread.

P2Predict is driven at full capability: budget="thorough" (the wider HPO
search), all descriptors offered as features (the CLI's 6-feature auto cap
is a CLI default, not an engine limit), and log-target left on `auto`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

from featurize import featurize, clean
from p2predict import auto_train
from p2predict.intervals import compute_calibration_residuals, predict_interval
from p2predict.model_evals import evaluate_model
from p2predict.prepare_data import Get_Column_Types
from p2predict.trained_model_io import P2PREDICT_VERSION
from p2predict.training import resolve_log_target

TARGET = "Y"
DROP = ("Y", "Drug_ID")

# Results are filed per P2Predict version so a later release's run sits
# alongside the previous one instead of overwriting it -- the before/after
# delta on a public leaderboard is the whole point of keeping this harness.
VERSION = P2PREDICT_VERSION.lstrip("v")


def run_seed(group, benchmark_name, seed, budget, feature_cap=None):
    train, valid = group.get_train_valid_split(
        benchmark=benchmark_name, split_type="default", seed=seed
    )
    test = group.get(benchmark_name)["test"]

    ftrain, fvalid, ftest = clean([featurize(train), featurize(valid), featurize(test)])
    features = [c for c in ftrain.columns if c not in DROP]

    if feature_cap:
        # P2Predict's own ranker, the same one the CLI uses to pick features.
        from p2predict.feature_selection import get_most_predictable_features

        ranked = get_most_predictable_features(
            ftrain[features + [TARGET]], TARGET, output_only_headers=True
        )
        features = list(ranked.head(feature_cap))

    X_train, y_train = ftrain[features], ftrain[TARGET]
    X_valid, y_valid = fvalid[features], fvalid[TARGET]
    X_test, y_test = ftest[features], ftest[TARGET]

    num, cat = Get_Column_Types(X_train)
    log_target, decision = resolve_log_target(y_train, mode="auto")

    started = time.time()
    model, algorithm, scores, log_target = auto_train(
        X_train, y_train, num, cat, budget=budget, log_target=log_target
    )
    elapsed = time.time() - started

    # Conformal calibration on valid -- held out from fitting.
    calibration = compute_calibration_residuals(model, X_valid, y_valid)
    intervals = predict_interval(model, X_test, calibration, coverage=0.90)
    lows = np.array([i.low for i in intervals])
    highs = np.array([i.high for i in intervals])
    covered = float(((y_test.values >= lows) & (y_test.values <= highs)).mean())

    y_pred = model.predict(X_test)
    mae, r2, _, rmse = evaluate_model(X_test, y_test, model)

    return {
        "seed": seed,
        "algorithm": algorithm,
        "cv_scores": {k: round(float(v), 4) for k, v in scores.items()},
        "log_target": bool(log_target),
        "log_target_decision": decision,
        "n_features": len(features),
        "n_train": int(len(X_train)),
        "test_mae": round(float(mae), 4),
        "test_rmse": round(float(rmse), 4),
        "test_r2": round(float(r2), 4),
        "interval_90_coverage": round(covered, 4),
        "mean_interval_width": round(float((highs - lows).mean()), 4),
        "train_seconds": round(elapsed, 1),
        "_predictions": list(map(float, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="Caco2_Wang")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budget", default="thorough", choices=["fast", "thorough"])
    ap.add_argument("--feature-cap", type=int, default=None,
                    help="Use P2Predict's ranker to keep only the top N descriptors")
    ap.add_argument("--tag", default="all-features")
    args = ap.parse_args()

    group = admet_group(path="data/")
    name = group.get(args.benchmark)["name"]

    results, preds = [], []
    for seed in args.seeds:
        print(f"--- seed {seed} ({args.budget}, cap={args.feature_cap}) ---", flush=True)
        r = run_seed(group, args.benchmark, seed, args.budget, args.feature_cap)
        preds.append({name: r.pop("_predictions")})
        results.append(r)
        print(f"    {r['algorithm']:<14} MAE {r['test_mae']:.4f}  R² {r['test_r2']:.3f}  "
              f"90% coverage {r['interval_90_coverage']:.1%}  ({r['train_seconds']}s)",
              flush=True)

    official = group.evaluate_many(preds) if len(preds) > 1 else group.evaluate(preds[0])

    out = {
        "benchmark": name,
        "p2predict_version": VERSION,
        "run_date": datetime.date.today().isoformat(),
        "budget": args.budget,
        "feature_cap": args.feature_cap,
        "official_tdc_metric": official,
        "per_seed": results,
    }
    out_dir = Path("results") / VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{args.tag}_{args.budget}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n=== OFFICIAL TDC SCORE ===\n{official}")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
