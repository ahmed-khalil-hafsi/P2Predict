"""Generate the three case-study marketing charts.

Three figures carry the case study on their own for a non-developer
reader:

  1. ``civic_attribution.png`` — horizontal bar chart of the SHAP
     multiplicative factors for the 2019 Honda Civic, sorted by magnitude.
     The "why is this price what it is" picture.

  2. ``intervals_comparison.png`` — point estimate + 90% likely range for
     the three example vehicles. Visually carries the "honest uncertainty"
     story: the Tesla's interval is wide because the model has thin
     training data there.

  3. ``mileage_curve.png`` — predicted price as a function of odometer
     for the Civic, holding every other feature fixed, with the 90%
     interval as a shaded band. Reads as the depreciation curve the
     model learned, not a rule of thumb.

Run after a model has been trained::

    python case-studies/used-cars/generate_charts.py

Outputs go to ``case-studies/used-cars/assets/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stop matplotlib from interpreting ``$...$`` pairs as math mode — we have
# dollar signs all over the labels and don't need TeX-style rendering.
plt.rcParams["text.parse_math"] = False

from p2predict import explain, load_model, predict_interval


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Pleasant, readable palette. We avoid red/green for accessibility — using
# a deep blue for "increases price" and a warm orange for "decreases price".
COLOR_UP = "#1f77b4"      # tab:blue
COLOR_DOWN = "#ff7f0e"    # tab:orange
COLOR_NEUTRAL = "#7f7f7f" # tab:gray
COLOR_INTERVAL = "#1f77b4"


def _latest_ridge_model() -> Path:
    candidates = sorted(MODELS_DIR.glob("ridge_price_*.model"))
    if not candidates:
        candidates = sorted(MODELS_DIR.glob("*_price_*.model"))
    if not candidates:
        raise SystemExit(
            f"No price models in {MODELS_DIR}. Train first — see README."
        )
    return candidates[-1]


def _civic_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "year": 2019, "odometer": 45_000,
        "manufacturer": "honda", "condition": "excellent",
        "fuel": "gas", "transmission": "automatic",
        "drive": "fwd", "type": "sedan",
        "state": "ca", "paint_color": "silver",
    }])


def chart_civic_attribution(model_data: dict, out: Path) -> None:
    """Horizontal bar chart of SHAP multiplicative factors for the Civic."""
    model = model_data["model"]
    bg = model_data["background_sample"]
    civic = _civic_row()
    ex = explain(model, civic, background_X=bg)

    items = sorted(
        ex.multiplicative_factors.items(),
        key=lambda kv: abs(1 - kv[1]),
        reverse=True,
    )
    features = [k for k, _ in items]
    pct = [(v - 1.0) * 100.0 for _, v in items]
    colors = [COLOR_UP if p >= 0 else COLOR_DOWN for p in pct]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    y = np.arange(len(features))
    ax.barh(y, pct, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Effect on price vs. the model's average vehicle (%)",
                  fontsize=11)
    ax.set_title(
        f"Why ${ex.predicted_price:,.0f}? — per-feature attribution\n"
        f"for the 2019 Honda Civic (45,000 mi, excellent, CA)",
        fontsize=12, pad=14, loc="left",
    )

    # Label each bar with its percentage.
    for i, p in enumerate(pct):
        offset = 1.5 if p >= 0 else -1.5
        ha = "left" if p >= 0 else "right"
        ax.text(p + offset, i, f"{p:+.1f}%",
                va="center", ha=ha, fontsize=10, color="black")

    # Padding so the labels don't clip.
    span = max(abs(min(pct)), abs(max(pct)))
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Footer with the axiom check — this is the line that separates SHAP
    # from any other importance heuristic. Putting it on the chart turns
    # the chart into something a procurement reviewer can defend.
    product = float(np.prod(list(ex.multiplicative_factors.values())))
    ratio = ex.predicted_price / ex.baseline_price
    fig.text(0.01, 0.01,
             f"Axiom check: product of factors = {product:.3f} = pred/baseline "
             f"({ratio:.3f})  ✓     "
             f"Baseline price ${ex.baseline_price:,.0f} → predicted "
             f"${ex.predicted_price:,.0f}",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_intervals_comparison(model_data: dict, out: Path) -> None:
    """Point + 90% interval for the three example vehicles."""
    model = model_data["model"]
    cal = model_data["calibration"]

    vehicles = [
        ("2019 Honda Civic\n45,000 mi · excellent · CA", {
            "year": 2019, "odometer": 45_000,
            "manufacturer": "honda", "condition": "excellent",
            "fuel": "gas", "transmission": "automatic",
            "drive": "fwd", "type": "sedan",
            "state": "ca", "paint_color": "silver"}),
        ("2008 Ford F-150\n180,000 mi · good · 4wd · TX", {
            "year": 2008, "odometer": 180_000,
            "manufacturer": "ford", "condition": "good",
            "fuel": "gas", "transmission": "automatic",
            "drive": "4wd", "type": "pickup",
            "state": "tx", "paint_color": "white"}),
        ("2021 Tesla Model 3\n22,000 mi · like new · WA", {
            "year": 2021, "odometer": 22_000,
            "manufacturer": "tesla", "condition": "like new",
            "fuel": "electric", "transmission": "other",
            "drive": "rwd", "type": "sedan",
            "state": "wa", "paint_color": "white"}),
    ]
    df = pd.DataFrame([v for _, v in vehicles])
    intervals = predict_interval(model, df, cal, coverage=0.90)
    labels = [lbl for lbl, _ in vehicles]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    y = np.arange(len(vehicles))[::-1]

    for i, iv in enumerate(intervals):
        width = iv.high - iv.low
        ax.barh(y[i], width, left=iv.low,
                color=COLOR_INTERVAL, alpha=0.20, edgecolor="none",
                height=0.55)
        # Marker for the point estimate.
        ax.plot(iv.prediction, y[i], "o", color=COLOR_INTERVAL,
                markersize=10, zorder=3)
        # Labels.
        ax.text(iv.prediction, y[i] + 0.3,
                f"${iv.prediction:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(iv.low, y[i] - 0.35,
                f"${iv.low:,.0f}", ha="center", va="top",
                fontsize=8, color="#444444")
        ax.text(iv.high, y[i] - 0.35,
                f"${iv.high:,.0f}", ha="center", va="top",
                fontsize=8, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted price ($)", fontsize=11)
    ax.set_title(
        "Honest uncertainty: wider range where the model has less data\n"
        "(90% likely range from split-conformal calibration)",
        fontsize=12, pad=14, loc="left",
    )
    ax.set_xlim(0, max(iv.high for iv in intervals) * 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_mileage_curve(model_data: dict, out: Path) -> None:
    """Predicted price vs odometer for the Civic, with 90% interval band."""
    model = model_data["model"]
    cal = model_data["calibration"]

    miles_grid = np.linspace(0, 250_000, 60)
    rows = []
    civic = _civic_row().iloc[0].to_dict()
    for m in miles_grid:
        row = dict(civic)
        row["odometer"] = float(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    intervals = predict_interval(model, df, cal, coverage=0.90)

    preds = np.array([iv.prediction for iv in intervals])
    lows = np.array([iv.low for iv in intervals])
    highs = np.array([iv.high for iv in intervals])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.fill_between(miles_grid, lows, highs,
                    color=COLOR_INTERVAL, alpha=0.18,
                    label="90% likely range")
    ax.plot(miles_grid, preds, color=COLOR_INTERVAL,
            linewidth=2.2, label="Predicted price")

    # Annotate two reference points: 45k mi (the listing's actual miles)
    # and 90k mi (the what-if counterfactual from predict_examples.py).
    for m_ref, label, dy in [(45_000, "Listing\n(45k mi)", 0.04),
                              (90_000, "What-if\n(90k mi)", -0.10)]:
        idx = int(np.argmin(np.abs(miles_grid - m_ref)))
        ax.plot(m_ref, preds[idx], "o", color="black", markersize=7, zorder=4)
        ax.annotate(
            f"{label}\n${preds[idx]:,.0f}",
            xy=(m_ref, preds[idx]),
            xytext=(m_ref + 18_000, preds[idx] * (1 + dy)),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="black", linewidth=0.7),
        )

    ax.set_xlabel("Odometer (miles)", fontsize=11)
    ax.set_ylabel("Predicted price ($)", fontsize=11)
    ax.set_title(
        "Mileage depreciation, learned — not assumed\n"
        "2019 Honda Civic, excellent, CA — sliding odometer with every other spec fixed",
        fontsize=12, pad=14, loc="left",
    )
    ax.set_xlim(0, 250_000)
    ax.set_ylim(0, max(highs) * 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y/1000:.0f}k"))
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    model_path = _latest_ridge_model()
    print(f"Loading {model_path.name} ...")
    model_data = load_model(model_path)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    chart_civic_attribution(model_data, ASSETS_DIR / "civic_attribution.png")
    chart_intervals_comparison(model_data, ASSETS_DIR / "intervals_comparison.png")
    chart_mileage_curve(model_data, ASSETS_DIR / "mileage_curve.png")
    print(f"\nDone. {len(list(ASSETS_DIR.glob('*.png')))} charts in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
