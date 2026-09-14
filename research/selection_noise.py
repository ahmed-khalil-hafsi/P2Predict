"""At 150 parts, is auto-mode picking the best model or the luckiest one?

Reproduction script for `research/selection_noise.md`. Read-only: it trains
throwaway models in memory and never writes a model or touches core.

`auto_train` scores ridge, random forest and XGBoost by cross-validation and
keeps the argmax:

    if score > best_score:        # training.py
        best_score, best_model, best_algorithm = score, model, algorithm

Nothing in that comparison knows how *precise* the scores are. On the
default `fast` budget the CV is **3-fold**, so on a 150-part catalog each
score is an average over three folds of fifty rows. This script measures
what that costs by giving each trial an honest grader: draw an n-part
catalog, let the selection rule choose, then score every candidate on a
disjoint 3,000-row evaluation sample and see whether the rule picked the
one that actually generalises best.

Measured per trial:
  * does the CV winner match the eval winner?
  * **regret** -- how much accuracy the pick gives up against the best
    available family, in median APE percentage points
  * **winner's curse** -- the winner's own CV score re-measured on an
    independent fold shuffle of the *same rows*. Taking a maximum over three
    noisy scores biases the winning score upward; this measures the drop.
  * **selection stability** -- reshuffle the folds, change nothing else, and
    see whether the same family still wins
  * is the winner's margin over the runner-up even bigger than one standard
    error of the fold scores?

Three rules are compared on identical draws:
  argmax     what ships today
  one_se     among families within 1 SE of the best mean CV score, take the
             simplest (ridge < random forest < XGBoost)
  ridge      never choose at all -- the do-nothing baseline

The rule under test selects on R^2, because that is what `auto_train` does.
The *grader* is median APE instead, for a reason the fasteners runs made
unavoidable: price-space R^2 is unbounded below, and a single exploded
prediction (ridge under the log wrap occasionally emits an astronomical
price on a heavy-tailed catalog) drives a family's score to -1e60 and makes
every average meaningless. Median APE is bounded, robust, and is the error a
buyer actually feels. The rate of those explosions is reported separately --
it is a finding in its own right.

Tuning is deliberately off: every family gets its library defaults. That
isolates the *selection* question, and it makes the result a lower bound --
the shipped path also chooses hyperparameters from the same small CV, which
adds noise rather than removing it.

Run from the repo root:  .venv/bin/python research/selection_noise.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import small_n_harness as H  # noqa: E402

from p2predict.training import ALGORITHMS, build_pipeline  # noqa: E402

warnings.filterwarnings("ignore")

TRIALS = 40
# 3 folds is what `budget="fast"` -- the default everywhere -- actually uses.
CV_FOLDS = 3
# Simple to complex. The 1-SE rule breaks ties toward the front of this list.
COMPLEXITY = {"ridge": 0, "random_forest": 1, "xgboost": 2}


def _cv_fold_scores(cat, X, y, algorithm, seed=0) -> np.ndarray:
    pipeline = build_pipeline(algorithm, cat.numeric, cat.categorical,
                              log_target=True)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return cross_val_score(pipeline, X, y, cv=kf, scoring="r2")


def _eval_fit(cat, X, y, X_eval, y_eval, algorithm) -> dict:
    pipeline = build_pipeline(algorithm, cat.numeric, cat.categorical,
                              log_target=True)
    pipeline.fit(X, y)
    pred = np.asarray(pipeline.predict(X_eval), dtype=float)
    return {"r2": float(r2_score(y_eval, pred)),
            "mape": H.median_ape(y_eval, pred)}


def _one_se_pick(means: dict, ses: dict) -> str:
    """Simplest family whose mean CV score is within 1 SE of the best."""
    best = max(means, key=lambda a: means[a])
    threshold = means[best] - ses[best]
    eligible = [a for a in ALGORITHMS if means[a] >= threshold]
    return min(eligible, key=lambda a: COMPLEXITY[a])


def run_cell(cat: H.Catalog, n: int, rng) -> dict:
    rows = []
    for _ in range(TRIALS):
        X, y, X_eval, y_eval = cat.draw(n, rng)
        try:
            folds = {a: _cv_fold_scores(cat, X, y, a, seed=0)
                     for a in ALGORITHMS}
            # Same rows, different fold shuffle. Any disagreement between the
            # two is selection noise and nothing else.
            replica = {a: float(np.mean(_cv_fold_scores(cat, X, y, a, seed=1)))
                       for a in ALGORITHMS}
            truth = {a: _eval_fit(cat, X, y, X_eval, y_eval, a)
                     for a in ALGORITHMS}
        except Exception as exc:
            print(f"    trial failed ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            continue

        means = {a: float(np.mean(folds[a])) for a in ALGORITHMS}
        ses = {a: float(np.std(folds[a], ddof=1) / np.sqrt(CV_FOLDS))
               for a in ALGORITHMS}

        argmax = max(means, key=lambda a: means[a])
        runner_up = max((a for a in ALGORITHMS if a != argmax),
                        key=lambda a: means[a])
        picks = {"argmax": argmax, "one_se": _one_se_pick(means, ses),
                 "ridge": "ridge"}

        true_best = min(ALGORITHMS, key=lambda a: truth[a]["mape"])
        best_mape = truth[true_best]["mape"]

        rows.append({
            "picks": picks,
            "true_best": true_best,
            "margin_over_runner_up": means[argmax] - means[runner_up],
            "se_of_winner": ses[argmax],
            # Is the winner's lead even distinguishable from fold noise?
            "margin_inside_1se": bool(
                means[argmax] - means[runner_up] < ses[argmax]
            ),
            # Winner's curse: the max of three noisy scores does not
            # replicate on an independent shuffle of the same data.
            "winners_curse_r2": means[argmax] - replica[argmax],
            "pick_changed_on_reshuffle": bool(
                argmax != max(replica, key=lambda a: replica[a])
            ),
            "regret_mape": {r: truth[p]["mape"] - best_mape
                            for r, p in picks.items()},
            "eval_mape": {a: truth[a]["mape"] for a in ALGORITHMS},
            # A family whose eval R2 has gone below -10 has emitted at least
            # one absurd price. Selection still "works" (it will never pick
            # that family), but the CV score it reports is meaningless.
            "exploded": {a: bool(truth[a]["r2"] < -10.0) for a in ALGORITHMS},
        })

    if not rows:
        return {}

    def agg(field, rule):
        return np.array([r[field][rule] for r in rows], dtype=float)

    rules = {}
    for rule in ("argmax", "one_se", "ridge"):
        reg = agg("regret_mape", rule)
        rules[rule] = {
            "pct_picked_true_best": float(np.mean(
                [r["picks"][rule] == r["true_best"] for r in rows]) * 100),
            "mean_regret_mape_pp": float(np.mean(reg)),
            "median_regret_mape_pp": float(np.median(reg)),
            "p90_regret_mape_pp": float(np.percentile(reg, 90)),
            "pick_counts": {a: int(sum(r["picks"][rule] == a for r in rows))
                            for a in ALGORITHMS},
        }

    spread = np.array([max(r["eval_mape"].values()) - min(r["eval_mape"].values())
                       for r in rows])
    return {
        "trials": len(rows),
        "rules": rules,
        "pct_margin_inside_1se": float(np.mean(
            [r["margin_inside_1se"] for r in rows]) * 100),
        "median_winners_curse_r2": float(np.median(
            [r["winners_curse_r2"] for r in rows])),
        "pct_pick_changed_on_reshuffle": float(np.mean(
            [r["pick_changed_on_reshuffle"] for r in rows]) * 100),
        "median_eval_mape_spread_pp": float(np.median(spread)),
        "pct_any_family_exploded": float(np.mean(
            [any(r["exploded"].values()) for r in rows]) * 100),
        "true_best_counts": {
            a: int(sum(r["true_best"] == a for r in rows)) for a in ALGORITHMS},
        # Kept so a re-analysis never needs a re-run.
        "trial_detail": rows,
    }


def main() -> None:
    t0 = time.time()
    out = []
    for cat in H.load_all():
        for n in H.CATALOG_SIZES:
            rng = np.random.default_rng(H.SEED)
            s = run_cell(cat, n, rng)
            if not s:
                continue
            out.append({"dataset": cat.label, "data_file": cat.source,
                        "catalog_n": n, "cv_folds": CV_FOLDS,
                        "trials": TRIALS, **s})
            print(f"\n=== {cat.label} / n={n}  ({s['trials']} resamples, "
                  f"{CV_FOLDS}-fold CV as shipped)")
            print(f"    winner's CV margin is inside 1 SE of fold noise in "
                  f"{s['pct_margin_inside_1se']:.0f}% of draws")
            print(f"    the winning CV score drops "
                  f"{s['median_winners_curse_r2']:+.3f} R2 (median) when the "
                  f"folds are reshuffled")
            print(f"    reshuffling the folds alone changes the chosen family "
                  f"in {s['pct_pick_changed_on_reshuffle']:.0f}% of draws")
            print(f"    best-to-worst family gap on the eval sample: "
                  f"{s['median_eval_mape_spread_pp']:.1f}pp of median APE")
            if s["pct_any_family_exploded"]:
                print(f"    at least one family emitted an absurd price in "
                      f"{s['pct_any_family_exploded']:.0f}% of draws")
            print(f"    {'rule':<8} {'picked best':>12} {'mean regret':>12} "
                  f"{'median':>9} {'p90':>8}")
            for rule, r in s["rules"].items():
                print(f"    {rule:<8} {r['pct_picked_true_best']:>11.0f}% "
                      f"{r['mean_regret_mape_pp']:>11.1f}pp "
                      f"{r['median_regret_mape_pp']:>8.1f}pp "
                      f"{r['p90_regret_mape_pp']:>7.1f}pp")

    dest = H.REPO / "research" / "selection_noise_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(H.REPO)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
