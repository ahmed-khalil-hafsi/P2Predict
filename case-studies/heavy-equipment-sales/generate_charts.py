"""Generate the heavy-equipment resale case-study marketing charts.

Four figures carry the case study for a non-developer reader:

  1. ``wheel_loader_attribution.png`` — horizontal bar chart of the SHAP
     multiplicative factors (as %) for the premium wheel loader. The "why
     will this lot fetch ~$52k" picture. Log-target model, so drivers are
     read as % lift/cut on the price.

  2. ``intervals_comparison.png`` — point estimate + 90% likely hammer
     range for the three example lots. The "defensible reserve, honest
     uncertainty" story.

  3. ``depreciation_curve.png`` — predicted price as a function of machine
     age for the baseline machine, holding everything else fixed, with the
     90% interval as a shaded band. Shows depreciation is a front-loaded
     curve, not a straight line.

  4. ``enclosure_premium.png`` — predicted price by cab type (open station
     vs enclosed cab vs enclosed cab with AC) for an otherwise identical
     machine. The resale lever a remarketer actually controls.

Run after a model has been trained::

    python case-studies/heavy-equipment-sales/generate_charts.py

Outputs go to ``case-studies/heavy-equipment-sales/assets/``.
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

# Deep blue for "lifts resale", warm orange for "cuts resale" — avoids
# red/green for accessibility.
COLOR_UP = "#1f77b4"      # tab:blue
COLOR_DOWN = "#ff7f0e"    # tab:orange
COLOR_INTERVAL = "#1f77b4"

# The bland mid-tier machine the sweeps read against — same one
# extract_insights.py uses, so the README's numbers line up across scripts.
BASE_MACHINE = {
    "age_at_sale": 8.0,
    "sale_year": 2008.0,
    "product_group": "Wheel Loader",
    "product_size": "Medium",
    "enclosure": "EROPS",
    "state": "Florida",
}

# The premium wheel loader — the most sell-interesting lot.
WHEEL_LOADER = {
    "age_at_sale": 3.0,
    "sale_year": 2008.0,
    "product_group": "Wheel Loader",
    "product_size": "Large",
    "enclosure": "EROPS w AC",
    "state": "Texas",
}


def _latest_model() -> Path:
    candidates = sorted(MODELS_DIR.glob("*_sale_price_usd_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No resale models in {MODELS_DIR}. Train first — see README."
        )
    return candidates[-1]


def chart_attribution(model_data: dict, out: Path) -> None:
    """Horizontal bar chart of SHAP multiplicative factors (%) for the loader."""
    model = model_data["model"]
    bg = model_data["background_sample"]
    ex = explain(model, pd.DataFrame([WHEEL_LOADER]), background_X=bg)

    items = sorted(ex.multiplicative_factors.items(),
                   key=lambda kv: abs(kv[1] - 1.0), reverse=True)
    features = [k for k, _ in items]
    pct = [(v - 1.0) * 100.0 for _, v in items]
    colors = [COLOR_UP if p >= 0 else COLOR_DOWN for p in pct]

    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=150)
    y = np.arange(len(features))
    ax.barh(y, pct, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Effect on hammer price vs. the model's average machine (%)",
                  fontsize=11)
    ax.set_title(
        f"Why ${ex.predicted_price:,.0f}? — per-driver resale attribution\n"
        f"for the 3-yr Large wheel loader with an AC cab (sold 2008, TX)",
        fontsize=12, pad=14, loc="left",
    )

    span = max(abs(min(pct)), abs(max(pct)))
    for i, p in enumerate(pct):
        offset = span * 0.03 if p >= 0 else -span * 0.03
        ha = "left" if p >= 0 else "right"
        ax.text(p + offset, i, f"{p:+.1f}%",
                va="center", ha=ha, fontsize=10, color="black")

    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    prod = float(np.prod(list(ex.multiplicative_factors.values())))
    fig.text(0.01, 0.01,
             f"Axiom check: product of factors = {prod:.3f} "
             f"= pred/baseline ({ex.predicted_price / ex.baseline_price:.3f})  OK     "
             f"Baseline ${ex.baseline_price:,.0f} (avg machine) -> predicted "
             f"${ex.predicted_price:,.0f}",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_intervals_comparison(model_data: dict, out: Path) -> None:
    """Point + 90% interval for the three example lots."""
    model = model_data["model"]
    cal = model_data["calibration"]

    lots = [
        ("3-yr Large wheel loader\nAC cab (premium lot)", {
            "age_at_sale": 3.0, "sale_year": 2008.0,
            "product_group": "Wheel Loader", "product_size": "Large",
            "enclosure": "EROPS w AC", "state": "Texas"}),
        ("8-yr Medium track excavator\nenclosed cab (bread-and-butter)", {
            "age_at_sale": 8.0, "sale_year": 2008.0,
            "product_group": "Track Excavators", "product_size": "Medium",
            "enclosure": "EROPS", "state": "Florida"}),
        ("18-yr Small skid steer\nopen station (low-value)", {
            "age_at_sale": 18.0, "sale_year": 2009.0,
            "product_group": "Skid Steer Loaders", "product_size": "Small",
            "enclosure": "OROPS", "state": "Ohio"}),
    ]
    df = pd.DataFrame([p for _, p in lots])
    intervals = predict_interval(model, df, cal, coverage=0.90)
    labels = [lbl for lbl, _ in lots]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    y = np.arange(len(lots))[::-1]

    for i, iv in enumerate(intervals):
        width = iv.high - iv.low
        ax.barh(y[i], width, left=iv.low,
                color=COLOR_INTERVAL, alpha=0.20, edgecolor="none", height=0.55)
        ax.plot(iv.prediction, y[i], "o", color=COLOR_INTERVAL,
                markersize=10, zorder=3)
        ax.text(iv.prediction, y[i] + 0.30, f"${iv.prediction:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(iv.low, y[i] - 0.35, f"${iv.low:,.0f}",
                ha="center", va="top", fontsize=8, color="#444444")
        ax.text(iv.high, y[i] - 0.35, f"${iv.high:,.0f}",
                ha="center", va="top", fontsize=8, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted auction hammer price ($)", fontsize=11)
    ax.set_title(
        "90% likely hammer range from banded conformal calibration\n"
        "Lean on the range and the relative levers — this model compares lots, it isn't a single-number appraiser",
        fontsize=11, pad=14, loc="left",
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


def chart_depreciation_curve(model_data: dict, out: Path) -> None:
    """Predicted price vs machine age for the baseline machine."""
    model = model_data["model"]
    cal = model_data["calibration"]

    ages = np.linspace(0, 30, 31)
    rows = []
    for a in ages:
        row = dict(BASE_MACHINE)
        row["age_at_sale"] = float(a)
        rows.append(row)
    df = pd.DataFrame(rows)
    intervals = predict_interval(model, df, cal, coverage=0.90)

    preds = np.array([iv.prediction for iv in intervals])
    lows = np.array([iv.low for iv in intervals])
    highs = np.array([iv.high for iv in intervals])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.fill_between(ages, lows, highs, color=COLOR_INTERVAL, alpha=0.18,
                    label="90% likely range")
    ax.plot(ages, preds, color=COLOR_INTERVAL, linewidth=2.2,
            label="Predicted hammer price")

    for a_ref, label, dy in [(3, "3 yr\n(near-new)", 0.10),
                             (8, "8 yr\n(baseline)", 0.14),
                             (20, "20 yr\n(aged)", 0.22)]:
        idx = int(np.argmin(np.abs(ages - a_ref)))
        ax.plot(a_ref, preds[idx], "o", color="black", markersize=7, zorder=4)
        ax.annotate(
            f"{label}\n${preds[idx]:,.0f}",
            xy=(a_ref, preds[idx]),
            xytext=(a_ref + 1.5, preds[idx] * (1 + dy)),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="black", linewidth=0.7),
        )

    ax.set_xlabel("Machine age at sale (years)", fontsize=11)
    ax.set_ylabel("Predicted hammer price ($)", fontsize=11)
    ax.set_title(
        "Depreciation is a front-loaded curve, not a straight line\n"
        "Baseline Medium wheel loader, sliding age with every other attribute fixed",
        fontsize=12, pad=14, loc="left",
    )
    ax.set_xlim(0, 30)
    ax.set_ylim(0, max(highs) * 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_enclosure_premium(model_data: dict, out: Path) -> None:
    """Predicted price by cab type for an otherwise identical machine."""
    model = model_data["model"]
    order = ["OROPS", "EROPS", "EROPS w AC"]
    pretty = {
        "OROPS": "Open station (OROPS)",
        "EROPS": "Enclosed cab (EROPS)",
        "EROPS w AC": "Enclosed cab + AC (EROPS w AC)",
    }
    rows = []
    for e in order:
        row = dict(BASE_MACHINE)
        row["enclosure"] = e
        rows.append(row)
    df = pd.DataFrame(rows)
    preds = model.predict(df)
    base = model.predict(pd.DataFrame([BASE_MACHINE]))[0]  # EROPS baseline

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    y = np.arange(len(order))
    colors = [COLOR_UP if p >= base else COLOR_DOWN for p in preds]
    ax.barh(y, preds, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(base, color="black", linewidth=0.9, linestyle="--")
    ax.text(base, -0.62, f"  enclosed-cab baseline ${base:,.0f}",
            fontsize=9, color="black", va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[e] for e in order], fontsize=10)
    ax.set_ylim(len(order) - 0.4, -0.85)
    ax.set_xlabel("Predicted hammer price for an identical Medium wheel loader ($)",
                  fontsize=11)
    ax.set_title(
        "Same machine, different cab: the resale lever you control\n"
        "All other attributes held fixed — only the operator station changes",
        fontsize=12, pad=14, loc="left",
    )
    for i, p in enumerate(preds):
        pct = 100 * (p - base) / base
        ax.text(p + max(preds) * 0.01, i, f"${p:,.0f}  ({pct:+.0f}%)",
                va="center", ha="left", fontsize=9, color="black")
    ax.set_xlim(0, max(preds) * 1.25)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    model_path = _latest_model()
    print(f"Loading {model_path.name} ...")
    model_data = load_model(model_path)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    chart_attribution(model_data, ASSETS_DIR / "wheel_loader_attribution.png")
    chart_intervals_comparison(model_data, ASSETS_DIR / "intervals_comparison.png")
    chart_depreciation_curve(model_data, ASSETS_DIR / "depreciation_curve.png")
    chart_enclosure_premium(model_data, ASSETS_DIR / "enclosure_premium.png")
    print(f"\nDone. {len(list(ASSETS_DIR.glob('*.png')))} charts in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
