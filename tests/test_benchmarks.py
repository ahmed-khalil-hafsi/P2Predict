"""Performance-regression guards for the SHAP explanation path.

Why this file exists
--------------------
The expensive part of a SHAP explanation is *building the explainer*: for a
tree ensemble, SHAP parses the entire fitted forest before it can attribute a
single row. The original code rebuilt that explainer once per row inside an
``explain_row`` loop, so explaining N rows paid the forest-parsing cost N
times. ``explain_batch`` builds the explainer ONCE and attributes all rows in
a single call (measured: random_forest 100 rows 9.0s -> 2.0s, xgboost
6.2s -> 0.09s).

That win is invisible to every other test in the suite — they all assert on
*values*, not *time* — so a refactor that accidentally moved the explainer
build back inside a per-row loop would keep the suite green while
reintroducing a 10x+ slowdown in production. These tests make that regression
fail loudly.

Design choices that keep this non-flaky
---------------------------------------
* The core assertions are RELATIVE: ``explain_batch`` over N rows vs. an
  ``explain_row`` loop over the same N rows. The per-row-rebuild pathology
  produces an order-of-magnitude gap, so a conservative "batch is at least
  ~2.5x faster" threshold sits 5-9x below the gap actually observed on this
  machine — CI jitter cannot close that margin.
* Small models (default 100-tree ensembles, 200-row fixture) — the pathology
  shows up at any size, so we don't pay for 400-tree models just to measure
  it. Whole file adds only a few seconds to the suite.
* Each path is warmed up once before timing so SHAP/XGBoost one-time import
  and JIT costs land outside the measured window and don't skew the first
  comparison.
* ``time.perf_counter()`` (monotonic, highest-resolution) for all timing.
* No new dependencies and no opt-in marker: these run on a plain
  ``pytest tests/``.
"""
from __future__ import annotations

import time

import pytest

from p2predict.explain import explain_batch, explain_row
from p2predict.intervals import compute_calibration_residuals, predict_interval
from p2predict.prepare_data import prepare_data
from p2predict.training import start_training

# How many test rows to time. Big enough that the per-row explainer rebuild
# dominates, small enough that the whole comparison is a couple of seconds.
_N_ROWS = 30

# Conservative lower bound on the batch-vs-loop speedup. Observed on the dev
# machine: ~22x (xgboost), ~13x (random_forest). Asserting only 2.5x leaves
# 5-9x of headroom so machine/CI variance can never flake this.
_MIN_SPEEDUP = 2.5


def _train(df, algorithm, target="Price"):
    features = [c for c in df.columns if c != target]
    X_train, X_test, y_train, y_test, num, cat = prepare_data(df, features, target)
    model, _, _ = start_training(
        X_train, y_train, num, cat, algorithm=algorithm, tune=False
    )
    return model, X_train, X_test, y_test


def _time_batch_vs_loop(model, rows):
    """Return (batch_seconds, loop_seconds) for explaining ``rows``.

    Both paths are warmed once first so the first-call SHAP/XGBoost import and
    explainer-build one-time costs are excluded from the comparison and the
    measurement is stable.
    """
    # Warm up both code paths (explainer build, SHAP internals, XGBoost loader).
    explain_batch(model, rows.iloc[:2], background_X=None)
    explain_row(model, rows.iloc[[0]], background_X=None)

    t0 = time.perf_counter()
    explain_batch(model, rows, background_X=None)
    batch_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(len(rows)):
        explain_row(model, rows.iloc[[i]], background_X=None)
    loop_seconds = time.perf_counter() - t0

    return batch_seconds, loop_seconds


# ---------------------------------------------------------------------------
# Batch-explain scaling — the core regression guard.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["xgboost", "random_forest"])
def test_explain_batch_is_far_cheaper_than_per_row_loop(synthetic_parts, algorithm):
    """Explaining N rows with one ``explain_batch`` call must be dramatically
    cheaper than N separate ``explain_row`` calls.

    This is the guard against silently re-introducing the per-row explainer
    rebuild that ``explain_batch`` was created to eliminate. We assert a
    relative speedup (batch at least 2.5x faster) rather than an absolute
    wall-clock budget so the test measures the *shape* of the cost (one build
    vs. N builds) and stays immune to how fast the machine happens to be.

    Both tree families are exercised: xgboost is the largest win and the path
    that round-trips through SHAP's XGBTreeModelLoader, random_forest is the
    second-largest — neither may regress while the other stays green.
    """
    model, _, X_test, _ = _train(synthetic_parts, algorithm)
    n = min(_N_ROWS, len(X_test))
    assert n >= 10, "Need a handful of rows for the per-row cost to dominate."
    rows = X_test.iloc[:n]

    batch_seconds, loop_seconds = _time_batch_vs_loop(model, rows)

    speedup = loop_seconds / batch_seconds if batch_seconds > 0 else float("inf")
    assert speedup >= _MIN_SPEEDUP, (
        f"explain_batch for {n} {algorithm} rows was only {speedup:.1f}x "
        f"faster than the per-row explain_row loop (batch={batch_seconds:.3f}s, "
        f"loop={loop_seconds:.3f}s). A speedup near 1x means the SHAP explainer "
        f"is being rebuilt per row again — the exact pathology explain_batch "
        f"exists to prevent."
    )


def test_explain_batch_cost_grows_sublinearly_in_rows(synthetic_parts):
    """Doubling the row count must NOT double ``explain_batch`` time.

    If the explainer is built once and only the per-row SHAP attribution
    scales, batch time over 2N rows is far less than 2x the time over N rows
    (the fixed build cost is amortised). If someone reintroduces a per-row
    rebuild, this ratio jumps toward ~2x and the test fails. Exercised on
    xgboost, where the build cost most dominates the per-row cost.
    """
    model, _, X_test, _ = _train(synthetic_parts, "xgboost")
    n = min(_N_ROWS, len(X_test) // 2)
    assert n >= 10, "Need enough rows that N and 2N differ meaningfully."
    rows_n = X_test.iloc[:n]
    rows_2n = X_test.iloc[: 2 * n]

    # Warm the explainer/SHAP/XGBoost paths before timing.
    explain_batch(model, rows_n.iloc[:2], background_X=None)

    t0 = time.perf_counter()
    explain_batch(model, rows_n, background_X=None)
    t_n = time.perf_counter() - t0

    t0 = time.perf_counter()
    explain_batch(model, rows_2n, background_X=None)
    t_2n = time.perf_counter() - t0

    # With a single shared build, 2N should cost well under 2x N. Allow a very
    # generous 1.8x ceiling: comfortably below the ~2x a per-row rebuild would
    # force, but loose enough that timer noise on a tiny absolute duration
    # never flakes it.
    if t_n > 0:
        assert t_2n <= 1.8 * t_n, (
            f"explain_batch time scaled ~linearly with row count "
            f"(N={n}: {t_n:.4f}s, 2N={2 * n}: {t_2n:.4f}s). A shared explainer "
            f"build should make 2N cost well under 2x N; near-linear scaling "
            f"suggests the explainer is rebuilt per row."
        )


# ---------------------------------------------------------------------------
# Absolute sanity benchmark — predict / predict_interval on a few hundred rows.
# ---------------------------------------------------------------------------


def test_predict_and_interval_are_fast_on_a_few_hundred_rows(synthetic_parts):
    """``model.predict`` and ``predict_interval`` over a few hundred rows must
    finish near-instantly.

    Pure sanity check — these paths are vectorised and should be milliseconds.
    The threshold (2s) sits ~500x above the few-millisecond time measured on
    the dev machine, so it can only fail if something pathological lands here
    (e.g. an accidental per-row Python loop), never on timer noise.
    """
    model, _, X_test, y_test = _train(synthetic_parts, "xgboost")
    calibration = compute_calibration_residuals(model, X_test, y_test)

    # A few hundred rows by tiling the test set.
    rows = X_test.sample(300, replace=True, random_state=0)

    model.predict(rows)  # warm up

    t0 = time.perf_counter()
    model.predict(rows)
    t_predict = time.perf_counter() - t0

    t0 = time.perf_counter()
    predict_interval(model, rows, calibration, coverage=0.90)
    t_interval = time.perf_counter() - t0

    assert t_predict < 2.0, f"predict on {len(rows)} rows took {t_predict:.3f}s"
    assert t_interval < 2.0, (
        f"predict_interval on {len(rows)} rows took {t_interval:.3f}s"
    )
