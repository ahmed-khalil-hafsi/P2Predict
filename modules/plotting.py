# Plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Theme — restrained navy/amber palette inspired by consulting-firm reports.
# ---------------------------------------------------------------------------

THEME = {
    "navy":   "#1F3A5F",   # primary lines, bars, headings
    "navy_2": "#3C5A7E",   # secondary
    "amber":  "#D97706",   # accent for reference lines, highlights
    "ink":    "#1F2937",   # body text
    "muted":  "#6B7280",   # secondary text, axis labels
    "grid":   "#E5E7EB",   # gridlines, table separators
    "panel":  "#F9FAFB",   # subtle panel background
    "rule":   "#9CA3AF",   # header/footer rules
}


def _apply_axes_style(ax, *, grid_axis="both"):
    """Strip chart chrome down to the essentials: no top/right spines, light grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(THEME["grid"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=THEME["muted"], labelsize=9)
    ax.xaxis.label.set_color(THEME["muted"])
    ax.yaxis.label.set_color(THEME["muted"])
    ax.title.set_color(THEME["ink"])
    if grid_axis in ("both", "x"):
        ax.grid(True, axis="x", color=THEME["grid"], linewidth=0.6, alpha=0.8)
    if grid_axis in ("both", "y"):
        ax.grid(True, axis="y", color=THEME["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _draw_insight_strip(fig, lines, *, x=0.05, y_start=0.832, line_step=0.022):
    """Render 1–2 lines of narrative takeaway just below the page subtitle.

    These are the "so what?" lines a CPO reads first; charts justify them.
    """
    for i, line in enumerate(lines):
        if not line:
            continue
        fig.text(
            x, y_start - i * line_step, line,
            fontsize=10.5, color=THEME["navy"], style="italic", weight="normal",
        )


def _draw_definitions_footer(fig, *, y=0.085):
    """One-line glossary so non-technical readers can decode the metric table."""
    fig.text(
        0.05, y,
        "Definitions",
        fontsize=7.5, color=THEME["navy"], weight="bold",
    )
    fig.text(
        0.13, y,
        "Median = typical prediction error    "
        "MAPE = average prediction error    "
        "P90 = 9-in-10 worst-case error    "
        "Holdout = parts the model never trained on",
        fontsize=7.5, color=THEME["muted"],
    )


def _draw_page_chrome(fig, *, page_label, target_name, model_name, training_date):
    """Add a thin top rule with brand/title and a footer strip with provenance.

    Keeps every page looking like part of one report rather than three loose charts.
    """
    # Top rule
    fig.add_artist(plt.Line2D(
        [0.05, 0.95], [0.965, 0.965],
        transform=fig.transFigure, color=THEME["rule"], linewidth=0.6,
    ))
    fig.text(0.05, 0.975, "P 2 P R E D I C T", fontsize=8.5,
             color=THEME["navy"], weight="bold")
    fig.text(0.95, 0.975, "Model Quality Report", fontsize=8.5,
             color=THEME["muted"], ha="right")

    # Footer rule + metadata
    footer_bits = []
    if target_name:
        footer_bits.append(f"Target: {target_name}")
    if model_name:
        footer_bits.append(f"Model: {model_name}")
    if training_date:
        footer_bits.append(training_date)
    footer_left = "  ·  ".join(footer_bits) if footer_bits else ""
    fig.add_artist(plt.Line2D(
        [0.05, 0.95], [0.035, 0.035],
        transform=fig.transFigure, color=THEME["rule"], linewidth=0.6,
    ))
    fig.text(0.05, 0.018, footer_left, fontsize=7.5, color=THEME["muted"])
    fig.text(0.95, 0.018, page_label, fontsize=7.5, color=THEME["muted"], ha="right")


def plot_histograms(df):
    n = len(df.columns)
    fig, axs = plt.subplots(1, n, figsize=(n * 5, 4), constrained_layout=True)

    # When df has a single column, plt.subplots returns a bare Axes, not an array.
    if n == 1:
        axs = [axs]

    sns.set_style("dark")
    for ax, column in zip(axs, df.columns):
        sns.histplot(df[column], color="darkblue", bins=30, ax=ax)
        ax.set_title(f"Distribution of {column}", fontsize=12)
        ax.set_xlabel(column, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)

    plt.show()


# ---------------------------------------------------------------------------
# Metric helpers for the procurement-style report.
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _abs_pct_errors(y_test, y_pred):
    """Return absolute percentage errors (%), dropping rows where y_test == 0."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_test) > _EPS
    if not mask.any():
        return np.array([])
    return np.abs(y_test[mask] - y_pred[mask]) / np.abs(y_test[mask]) * 100.0


def _summary_metrics(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_test - y_pred
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(r2_score(y_test, y_pred))
    ape = _abs_pct_errors(y_test, y_pred)
    if ape.size:
        mape = float(np.mean(ape))
        median_ape = float(np.median(ape))
        p90_ape = float(np.quantile(ape, 0.9))
    else:
        mape = median_ape = p90_ape = float("nan")
    return {
        "n_test": int(len(y_test)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ape": median_ape,
        "p90_ape": p90_ape,
    }


def _error_by_price_band(y_test, y_pred, n_bins=10):
    """Bucket holdout points by actual-price quantile and return median APE per bucket.

    Returns (labels, median_apes, counts) or None if data is too thin to bin.
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_test) < n_bins:
        return None
    edges = np.unique(np.quantile(y_test, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    # np.digitize with the interior edges; clip to valid bin range.
    bin_idx = np.clip(np.digitize(y_test, edges[1:-1]), 0, len(edges) - 2)
    labels, medians, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ape = _abs_pct_errors(y_test[mask], y_pred[mask])
        if ape.size == 0:
            continue
        labels.append(f"{edges[b]:,.0f}–{edges[b + 1]:,.0f}")
        medians.append(float(np.median(ape)))
        counts.append(n)
    if not labels:
        return None
    return labels, medians, counts


# ---------------------------------------------------------------------------
# Page renderers.
# ---------------------------------------------------------------------------


def _draw_metric_table(ax, *, title, rows, value_color=None):
    """Render a flat, borderless metric table with a small-caps section header."""
    ax.axis("off")
    # Spaced uppercase mimics the small-caps eyebrow used in consulting decks.
    ax.text(0.0, 1.0, " ".join(title.upper()), fontsize=9, color=THEME["muted"],
            weight="bold", transform=ax.transAxes, va="top")
    # One thin rule under the section header.
    ax.add_artist(plt.Line2D(
        [0.0, 1.0], [0.965, 0.965],
        transform=ax.transAxes, color=THEME["grid"], linewidth=0.8,
    ))
    n = len(rows)
    # Rows laid out top-down from y=0.92 to y=0.05.
    top = 0.91
    bottom = 0.05
    row_h = (top - bottom) / max(n, 1)
    for i, (label, value) in enumerate(rows):
        y = top - (i + 0.5) * row_h
        ax.text(0.0, y, label, fontsize=10, color=THEME["muted"],
                transform=ax.transAxes, va="center")
        ax.text(1.0, y, value, fontsize=10.5,
                color=value_color or THEME["ink"], weight="bold",
                transform=ax.transAxes, va="center", ha="right")
        # Thin separator between rows (except after the last).
        if i < n - 1:
            sep_y = y - row_h / 2
            ax.add_artist(plt.Line2D(
                [0.0, 1.0], [sep_y, sep_y],
                transform=ax.transAxes, color=THEME["grid"], linewidth=0.4,
            ))


def _render_summary_page(pdf, y_test, y_pred, *, target_name, model_name, n_train,
                        training_date, metrics, total_pages):
    """Page 1: provenance + headline metrics on the left, predicted vs actual on the right."""
    fig = plt.figure(figsize=(14, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.90, "Model Quality Report", fontsize=22, color=THEME["ink"],
             weight="bold")
    fig.text(0.05, 0.865, "Holdout performance summary for parametric estimation",
             fontsize=11, color=THEME["muted"], style="italic")

    # Narrative insight line — the executive takeaway.
    insight_lines = []
    if np.isfinite(metrics["median_ape"]) and np.isfinite(metrics["p90_ape"]):
        insight_lines.append(
            f"The model predicts within {metrics['median_ape']:.1f}% of actuals "
            f"for half the holdout, and within {metrics['p90_ape']:.1f}% for nine in ten."
        )
    _draw_insight_strip(fig, insight_lines)

    # Body grid: two tables stacked on the left, scatter on the right.
    gs = fig.add_gridspec(
        2, 2,
        left=0.05, right=0.95, top=0.77, bottom=0.17,
        width_ratios=[1.0, 1.5], height_ratios=[1.0, 1.25],
        wspace=0.30, hspace=0.18,
    )
    ax_prov = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])

    _draw_metric_table(ax_prov, title="Provenance", rows=[
        ("Model", model_name or "—"),
        ("Target", target_name),
        ("Training rows", f"{n_train:,}" if n_train is not None else "—"),
        ("Holdout rows", f"{metrics['n_test']:,}"),
        ("Report date", training_date or "—"),
    ])
    _draw_metric_table(ax_metrics, title="Performance", rows=[
        ("Median % error", f"{metrics['median_ape']:.1f}%"),
        ("Mean % error (MAPE)", f"{metrics['mape']:.1f}%"),
        ("P90 % error", f"{metrics['p90_ape']:.1f}%"),
        ("MAE", f"{metrics['mae']:,.2f}"),
        ("RMSE", f"{metrics['rmse']:,.2f}"),
        ("R²", f"{metrics['r2']:.3f}"),
    ], value_color=THEME["navy"])

    # --- Predicted vs Actual scatter ---
    ax_scatter.scatter(y_test, y_pred,
                       color=THEME["navy"], alpha=0.6, s=28, linewidths=0)
    lo, hi = float(np.min(y_test)), float(np.max(y_test))
    ax_scatter.plot([lo, hi], [lo, hi],
                    color=THEME["amber"], linewidth=1.2, linestyle="--")
    ax_scatter.set_title(
        f"Predicted vs actual  ·  holdout n = {metrics['n_test']}",
        fontsize=12, loc="left", pad=12, weight="bold",
    )
    # Inline line label — replaces a floating legend with a labeled reference.
    ax_scatter.text(
        0.985, 0.985, "Perfect prediction",
        transform=ax_scatter.transAxes,
        fontsize=8.5, color=THEME["amber"], style="italic",
        ha="right", va="top",
    )
    ax_scatter.set_xlabel(f"Actual {target_name}")
    ax_scatter.set_ylabel(f"Predicted {target_name}")
    _apply_axes_style(ax_scatter)

    _draw_definitions_footer(fig, y=0.085)
    _draw_page_chrome(fig, page_label=f"Page 1 of {total_pages}",
                      target_name=target_name, model_name=model_name,
                      training_date=training_date)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _render_calibration_page(pdf, y_test, y_pred, *, target_name, model_name,
                             training_date, metrics, total_pages):
    """Page 2: distribution of % errors + median % error by price band."""
    fig = plt.figure(figsize=(14, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.90, "Error Distribution & Calibration", fontsize=20,
             color=THEME["ink"], weight="bold")
    fig.text(0.05, 0.865,
             "Where the model lands and which bands of the target it handles best",
             fontsize=11, color=THEME["muted"], style="italic")

    # Compute bands up front so we can mention the worst/best in the insight strip.
    bands = _error_by_price_band(y_test, y_pred)
    insight_lines = []
    if bands is not None:
        labels, medians, counts = bands
        worst_i = int(np.argmax(medians))
        best_i = int(np.argmin(medians))
        insight_lines.append(
            f"Accuracy is strongest in the {labels[best_i]} band ({medians[best_i]:.1f}% median error) "
            f"and weakest in {labels[worst_i]} ({medians[worst_i]:.1f}%)."
        )
    elif np.isfinite(metrics["median_ape"]):
        insight_lines.append(
            f"Holdout was too small to bucket by {target_name.lower()} band; "
            f"overall median error is {metrics['median_ape']:.1f}%."
        )
    _draw_insight_strip(fig, insight_lines)

    gs = fig.add_gridspec(
        1, 2, left=0.06, right=0.96, top=0.77, bottom=0.13, wspace=0.22,
    )
    ax_pct = fig.add_subplot(gs[0, 0])
    ax_band = fig.add_subplot(gs[0, 1])

    # --- % error histogram ---
    ape = _abs_pct_errors(y_test, y_pred)
    if ape.size:
        ax_pct.hist(ape, bins=25, color=THEME["navy"], alpha=0.85,
                    edgecolor="white", linewidth=0.6)
        ax_pct.axvline(metrics["median_ape"], color=THEME["amber"], linestyle="--",
                       linewidth=1.3, label=f"Median: {metrics['median_ape']:.1f}%")
        ax_pct.axvline(metrics["p90_ape"], color=THEME["amber"], linestyle=":",
                       linewidth=1.3, label=f"P90: {metrics['p90_ape']:.1f}%")
        ax_pct.legend(frameon=False, fontsize=9, loc="upper right")
    ax_pct.set_title("Distribution of absolute % errors",
                     fontsize=12, loc="left", pad=10, weight="bold")
    ax_pct.set_xlabel("Absolute percentage error (%)")
    ax_pct.set_ylabel("Holdout parts")
    _apply_axes_style(ax_pct, grid_axis="y")

    # --- Error by price band (already computed above for the insight strip) ---
    if bands is not None:
        labels, medians, counts = bands
        positions = np.arange(len(labels))
        overall = metrics["median_ape"]
        # Above-average bands get the accent colour; the rest stay navy.
        colors = [THEME["amber"] if m > overall else THEME["navy"] for m in medians]
        ax_band.bar(positions, medians, color=colors, edgecolor="white", linewidth=0.6)
        ax_band.set_xticks(positions)
        ax_band.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        if len(set(counts)) > 1:
            for x, y, n in zip(positions, medians, counts):
                ax_band.text(x, y, f"n={n}", ha="center", va="bottom",
                             fontsize=7, color=THEME["muted"])
        ax_band.axhline(
            overall, color=THEME["muted"], linestyle="--", linewidth=1,
            label=f"Overall median: {overall:.1f}%",
        )
        ax_band.legend(loc="upper left", frameon=False, fontsize=9)
        ax_band.set_title(
            f"Median % error by {target_name.lower()} band",
            fontsize=12, loc="left", pad=10, weight="bold",
        )
        # X-axis label dropped — the chart title already names the dimension,
        # and the band tick labels are wide enough to overlap a redundant label.
        ax_band.set_xlabel("")
        ax_band.set_ylabel("Median absolute % error")
        _apply_axes_style(ax_band, grid_axis="y")
    else:
        ax_band.axis("off")
        ax_band.text(0.5, 0.5,
                     f"Not enough holdout rows to bucket by {target_name.lower()} band.",
                     ha="center", va="center", fontsize=11, color=THEME["muted"])

    _draw_page_chrome(fig, page_label=f"Page 2 of {total_pages}",
                      target_name=target_name, model_name=model_name,
                      training_date=training_date)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _render_feature_importance_page(pdf, feature_importances, *, target_name,
                                    model_name, training_date, top_features,
                                    total_pages):
    """Page 3 (optional): horizontal bar chart of top-N source feature importances."""
    if not feature_importances:
        return
    items = list(feature_importances)[:top_features]
    names = [n for n, _ in items]
    weights = [w for _, w in items]
    total = sum(weights)
    if total <= 0:
        return
    # Normalise so the x-axis reads as a share — easier to talk about than raw weights.
    shares = [w / total for w in weights]

    fig = plt.figure(figsize=(14, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.90, "Feature Importance", fontsize=20,
             color=THEME["ink"], weight="bold")
    fig.text(0.05, 0.865,
             f"Top {len(items)} attributes the model relies on when estimating {target_name.lower()}",
             fontsize=11, color=THEME["muted"], style="italic")

    # Cumulative-share insight — the headline a CPO will quote back.
    k = min(3, len(items))
    top_k_share = sum(shares[:k])
    top_k_names = ", ".join(names[:k])
    insight_lines = [
        f"The top {k} features ({top_k_names}) explain "
        f"{top_k_share * 100:.0f}% of the model's decisions."
    ]
    _draw_insight_strip(fig, insight_lines)

    gs = fig.add_gridspec(1, 1, left=0.18, right=0.92, top=0.77, bottom=0.11)
    ax = fig.add_subplot(gs[0, 0])

    positions = np.arange(len(items))
    # Plot smallest at the top of the bar list so the largest sits at the top of the chart.
    bar_colors = [THEME["amber"]] + [THEME["navy"]] * (len(items) - 1)
    ax.barh(positions, shares[::-1], color=bar_colors[::-1],
            edgecolor="white", linewidth=0.6)
    ax.set_yticks(positions)
    ax.set_yticklabels(names[::-1], fontsize=10, color=THEME["ink"])
    ax.set_xlabel("Share of total importance")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    # Inline value labels at the end of each bar.
    for pos, share in zip(positions, shares[::-1]):
        ax.text(share, pos, f"  {share * 100:.1f}%",
                va="center", ha="left", fontsize=9, color=THEME["muted"])
    _apply_axes_style(ax, grid_axis="x")
    # Give the labels room.
    ax.set_xlim(0, max(shares) * 1.15)

    _draw_page_chrome(fig, page_label=f"Page {total_pages} of {total_pages}",
                      target_name=target_name, model_name=model_name,
                      training_date=training_date)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def plot_results_pdf(
    y_test,
    y_pred,
    filename,
    *,
    target_name="Price",
    model_name=None,
    n_train=None,
    training_date=None,
    feature_importances=None,
    top_features=15,
):
    """Write a procurement-friendly model-quality PDF.

    Layout:
      Page 1 — Summary metrics table + Predicted vs Actual scatter.
      Page 2 — Absolute % error distribution + median % error by price band.
      Page 3 — (optional) Top-N feature importance, when ``feature_importances`` is provided.

    Parameters
    ----------
    y_test, y_pred : array-like
        Holdout actuals and aligned predictions. Caller is responsible for
        passing the true holdout — metrics are otherwise misleadingly optimistic.
    filename : str
        Output PDF path.
    target_name : str
        Name of the target column (e.g. "Price", "Revenue", "Cost"). Drives all
        titles and axis labels.
    model_name : str, optional
        Algorithm or pipeline label shown in the summary table (e.g. "xgboost").
    n_train : int, optional
        Number of training rows; shown for provenance.
    training_date : str, optional
        Date/time string shown in the summary table. Caller picks the format.
    feature_importances : list[tuple[str, float]], optional
        Sorted (desc) list of ``(feature_name, importance)`` pairs, as produced by
        ``modules.training.extract_feature_importances``. When omitted or empty,
        the feature-importance page is skipped.
    top_features : int
        Max number of features to render on the importance page.
    """
    metrics = _summary_metrics(y_test, y_pred)
    # Page 3 only renders when we have importances; reflect that in the footer counter.
    has_importance_page = bool(feature_importances) and sum(
        w for _, w in list(feature_importances)[:top_features]
    ) > 0
    total_pages = 3 if has_importance_page else 2

    # Keep PDF /Info metadata strictly ASCII. matplotlib 3.7 switches the
    # whole Info dict to UTF-16BE hex encoding the moment any field contains
    # non-Latin-1 (e.g. an em-dash), which buries fields like the target name
    # in hex and confuses any consumer that scans the file for text. The
    # visible em-dashes in page titles/subtitles are rendered by matplotlib
    # directly, so this only affects file-properties metadata.
    pdf_metadata = {
        "Title": f"P2Predict - Model Quality Report ({target_name})",
        "Author": "P2Predict",
        "Subject": (
            f"Parametric estimation holdout performance for {target_name}"
            + (f" - model: {model_name}" if model_name else "")
        ),
        "Keywords": "P2Predict, parametric estimation, model quality, procurement",
    }

    with PdfPages(filename, metadata=pdf_metadata) as pdf:
        _render_summary_page(
            pdf, y_test, y_pred,
            target_name=target_name,
            model_name=model_name,
            n_train=n_train,
            training_date=training_date,
            metrics=metrics,
            total_pages=total_pages,
        )
        _render_calibration_page(
            pdf, y_test, y_pred,
            target_name=target_name,
            model_name=model_name,
            training_date=training_date,
            metrics=metrics,
            total_pages=total_pages,
        )
        _render_feature_importance_page(
            pdf, feature_importances,
            target_name=target_name,
            model_name=model_name,
            training_date=training_date,
            top_features=top_features,
            total_pages=total_pages,
        )
