"""Locks in the plotting contract so future refactors can't silently regress
the model-quality PDF (layout/visuals are not asserted — those are hard to
test reliably; what we lock in is the public API surface, page count, metric
math, and the edge cases that bit us during the v0.3 plotting overhaul).
"""
import numpy as np
import pandas as pd
import pytest

# Force a non-interactive backend before modules.plotting imports pyplot.
import matplotlib
matplotlib.use("Agg")

from modules.plotting import (
    _abs_pct_errors,
    _error_by_price_band,
    _summary_metrics,
    plot_histograms,
    plot_results_pdf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def holdout():
    """Synthetic holdout shaped like a real procurement target (log-normal,
    ~8% noise) so band-bucketing and percentile metrics have something to chew on."""
    rng = np.random.default_rng(0)
    y_test = pd.Series(np.exp(rng.normal(4, 0.7, 200)))
    y_pred = y_test * (1 + rng.normal(0, 0.08, 200))
    return y_test, y_pred


@pytest.fixture
def feat_imp():
    return [
        ("weight_kg", 0.42),
        ("material", 0.21),
        ("length_mm", 0.18),
        ("supplier", 0.11),
        ("finish", 0.08),
    ]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_summary_metrics_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = _summary_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert m["mape"] == 0.0
    assert m["median_ape"] == 0.0
    assert m["p90_ape"] == 0.0
    assert m["n_test"] == 4


def test_summary_metrics_known_errors():
    # Two points, each 10% off in absolute terms.
    y_test = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    m = _summary_metrics(y_test, y_pred)
    assert m["mae"] == pytest.approx(15.0)
    assert m["median_ape"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(10.0)


def test_abs_pct_errors_drops_zero_actuals():
    # Zero-actual rows would explode the percentage; helper must mask them.
    y_test = np.array([0.0, 100.0, 0.0, 50.0])
    y_pred = np.array([5.0, 110.0, 1.0, 60.0])
    ape = _abs_pct_errors(y_test, y_pred)
    assert len(ape) == 2
    np.testing.assert_allclose(sorted(ape), [10.0, 20.0])


def test_abs_pct_errors_all_zero_actuals_returns_empty():
    ape = _abs_pct_errors(np.zeros(5), np.ones(5))
    assert len(ape) == 0


def test_summary_metrics_all_zero_actuals_yields_nan_percentages():
    # Without any non-zero actuals, percentage metrics are undefined — they
    # must come back as NaN rather than crash or quietly produce garbage.
    y_test = np.zeros(5)
    y_pred = np.ones(5)
    m = _summary_metrics(y_test, y_pred)
    assert np.isnan(m["mape"])
    assert np.isnan(m["median_ape"])
    assert np.isnan(m["p90_ape"])
    assert m["mae"] == 1.0  # absolute error is still well-defined


def test_error_by_price_band_returns_none_for_tiny_input():
    # Default n_bins=10; fewer rows than bins should bail out cleanly.
    assert _error_by_price_band(np.arange(5.0), np.arange(5.0)) is None


def test_error_by_price_band_basic_shape(holdout):
    y_test, y_pred = holdout
    bands = _error_by_price_band(y_test.values, y_pred.values, n_bins=10)
    assert bands is not None
    labels, medians, counts = bands
    assert len(labels) == len(medians) == len(counts)
    assert 1 <= len(labels) <= 10
    assert sum(counts) <= len(y_test)
    # All medians are non-negative percentages.
    assert all(m >= 0 for m in medians)


# ---------------------------------------------------------------------------
# plot_histograms — covers the latent 1-column-DataFrame crash we fixed.
# ---------------------------------------------------------------------------


def test_plot_histograms_single_column(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)
    df = pd.DataFrame({"weight_kg": np.random.default_rng(0).random(20)})
    plot_histograms(df)  # would have raised before the fix


def test_plot_histograms_multi_column(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"weight_kg": rng.random(20), "length_mm": rng.random(20)})
    plot_histograms(df)


# ---------------------------------------------------------------------------
# plot_results_pdf — public contract.
# ---------------------------------------------------------------------------


def _count_pdf_pages(path):
    """Count /Type /Page objects in the PDF, excluding the /Pages catalog node.

    matplotlib's PdfPages writes uncompressed page objects, so a substring
    scan of the file bytes is reliable for this purpose.
    """
    with open(path, "rb") as f:
        data = f.read()
    needle = b"/Type /Page"
    n, i = 0, 0
    while True:
        i = data.find(needle, i)
        if i == -1:
            return n
        # Exclude "/Type /Pages" (the page-tree root).
        if data[i + len(needle):i + len(needle) + 1] != b"s":
            n += 1
        i += len(needle)


def test_plot_results_pdf_writes_non_empty_file(holdout, tmp_path, feat_imp):
    y_test, y_pred = holdout
    out = tmp_path / "report.pdf"
    plot_results_pdf(
        y_test, y_pred, str(out),
        target_name="Revenue", model_name="xgboost", n_train=800,
        training_date="2026-05-29 14:30", feature_importances=feat_imp,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_results_pdf_three_pages_with_importances(holdout, tmp_path, feat_imp):
    y_test, y_pred = holdout
    out = tmp_path / "report.pdf"
    plot_results_pdf(
        y_test, y_pred, str(out),
        target_name="Revenue", feature_importances=feat_imp,
    )
    assert _count_pdf_pages(str(out)) == 3


def test_plot_results_pdf_two_pages_without_importances(holdout, tmp_path):
    y_test, y_pred = holdout
    out = tmp_path / "report.pdf"
    plot_results_pdf(
        y_test, y_pred, str(out),
        target_name="Revenue", feature_importances=None,
    )
    assert _count_pdf_pages(str(out)) == 2


def test_plot_results_pdf_two_pages_with_empty_importances(holdout, tmp_path):
    y_test, y_pred = holdout
    out = tmp_path / "report.pdf"
    plot_results_pdf(
        y_test, y_pred, str(out),
        target_name="Revenue", feature_importances=[],
    )
    assert _count_pdf_pages(str(out)) == 2


def test_plot_results_pdf_two_pages_when_importances_sum_zero(holdout, tmp_path):
    # Linear models may emit all-zero coefficients on a degenerate fit; the
    # importance page should be skipped rather than render an empty bar chart.
    y_test, y_pred = holdout
    out = tmp_path / "report.pdf"
    plot_results_pdf(
        y_test, y_pred, str(out),
        target_name="Revenue", feature_importances=[("a", 0.0), ("b", 0.0)],
    )
    assert _count_pdf_pages(str(out)) == 2


def test_plot_results_pdf_embeds_metadata(holdout, tmp_path):
    y_test, y_pred = holdout
    out = tmp_path / "report.pdf"
    plot_results_pdf(
        y_test, y_pred, str(out),
        target_name="Revenue", model_name="xgboost",
    )
    with open(out, "rb") as f:
        data = f.read()
    assert b"/Title" in data
    assert b"P2Predict" in data
    # Target name should be reflected in the document title metadata.
    assert b"Revenue" in data


def test_plot_results_pdf_target_name_flows_through(holdout, tmp_path):
    # A user training on `Cost` should not see "Revenue" or "Price" in the PDF.
    y_test, y_pred = holdout
    out = tmp_path / "cost.pdf"
    plot_results_pdf(y_test, y_pred, str(out), target_name="Cost")
    with open(out, "rb") as f:
        data = f.read()
    assert b"Cost" in data


def test_plot_results_pdf_handles_tiny_holdout(tmp_path):
    # Below the band-bucketing threshold; calibration chart must degrade
    # gracefully (no crash, still produces a valid PDF).
    y_test = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0, 51.0])
    out = tmp_path / "tiny.pdf"
    plot_results_pdf(y_test, y_pred, str(out), target_name="Price")
    assert out.exists()
    assert out.stat().st_size > 0
