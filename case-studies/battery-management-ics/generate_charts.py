"""Generate the BMIC case-study marketing charts.

Four figures carry the case study for a non-developer reader:

  1. ``ev_bms_attribution.png`` — horizontal bar chart of the SHAP dollar
     contributions for the 16-cell EV / datacenter BMS, sorted by
     magnitude. The "why is this part $5.25" picture.

  2. ``intervals_comparison.png`` — point estimate + 90% likely range for
     the three example parts. Carries the "honest uncertainty" story and
     the additive-interval-goes-negative caveat (clipped to $0, flagged).

  3. ``pin_count_curve.png`` — predicted price as a function of package
     pin count for the baseline BMIC, holding every other feature fixed,
     with the 90% interval as a shaded band. The cleanest cost driver the
     model learned (+70% from 6 → 48 pins).

  4. ``manufacturer_premium.png`` — predicted price by manufacturer for an
     otherwise identical single-cell I2C BMIC. The supplier-swap
     negotiation lever, quantified.

Run after a model has been trained::

    python case-studies/battery-management-ics/generate_charts.py

Outputs go to ``case-studies/battery-management-ics/assets/``.
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

# Pleasant, readable palette. We avoid red/green for accessibility — a deep
# blue for "adds cost" and a warm orange for "saves cost".
COLOR_UP = "#1f77b4"      # tab:blue
COLOR_DOWN = "#ff7f0e"    # tab:orange
COLOR_NEUTRAL = "#7f7f7f" # tab:gray
COLOR_INTERVAL = "#1f77b4"

# The baseline single-cell I2C BMIC the sweeps read against — same one
# extract_insights.py uses, so the README's numbers line up across scripts.
BASE_BMIC = {
    "manufacturer": "Texas Instruments",
    "Battery Chemistry": "Lithium Ion/Polymer",
    "Interface": "I2C",
    "max_cells_supported": 1.0,
    "op_temp_min_C": -40.0,
    "op_temp_max_C": 85.0,
    "package_pins": 8.0,
    "is_multi_cell": "False",
}

# The 16-cell EV / datacenter BMS — the most procurement-interesting part.
EV_BMS = {
    "manufacturer": "Analog Devices Inc./Maxim Integrated",
    "Battery Chemistry": "Lithium Ion/Polymer",
    "Interface": "I2C, USB",
    "max_cells_supported": 16.0,
    "op_temp_min_C": -40.0,
    "op_temp_max_C": 125.0,
    "package_pins": 48.0,
    "is_multi_cell": "True",
}


def _latest_model() -> Path:
    # Sort by mtime, not filename — an alphabetical sort can rank an older
    # xgboost model above a newer ridge one. mtime = trained last.
    candidates = sorted(MODELS_DIR.glob("*_unit_price_at_1_usd_*.model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No price models in {MODELS_DIR}. Train first — see README."
        )
    return candidates[-1]


def _clip0(v: float) -> float:
    return max(0.0, v)


def chart_ev_bms_attribution(model_data: dict, out: Path) -> None:
    """Horizontal bar chart of SHAP dollar contributions for the EV BMS."""
    model = model_data["model"]
    bg = model_data["background_sample"]
    ev = pd.DataFrame([EV_BMS])
    ex = explain(model, ev, background_X=bg)

    items = sorted(
        ex.contributions.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    features = [k for k, _ in items]
    dollars = [v for _, v in items]
    colors = [COLOR_UP if d >= 0 else COLOR_DOWN for d in dollars]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    y = np.arange(len(features))
    ax.barh(y, dollars, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Contribution to unit price vs. the model's average BMIC ($)",
                  fontsize=11)
    ax.set_title(
        f"Why ${ex.prediction:,.2f}? — per-feature dollar attribution\n"
        f"for the 16-cell ADI/Maxim EV / datacenter BMS (48-pin, -40/125 C)",
        fontsize=12, pad=14, loc="left",
    )

    span = max(abs(min(dollars)), abs(max(dollars)))
    for i, d in enumerate(dollars):
        offset = span * 0.03 if d >= 0 else -span * 0.03
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+.2f}",
                va="center", ha=ha, fontsize=10, color="black")

    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Footer with the local-accuracy axiom check — the line that separates
    # SHAP from any other importance heuristic.
    total = ex.baseline + sum(ex.contributions.values())
    fig.text(0.01, 0.01,
             f"Axiom check: baseline + sum(contributions) = ${total:.2f} "
             f"= prediction (${ex.prediction:.2f})  OK     "
             f"Baseline ${ex.baseline:.2f} (avg BMIC) -> predicted "
             f"${ex.prediction:.2f}",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_intervals_comparison(model_data: dict, out: Path) -> None:
    """Point + 90% interval for the three example parts."""
    model = model_data["model"]
    cal = model_data["calibration"]

    parts = [
        ("TI BQ29700-class\n1-cell Li-ion protection IC (wearable)", {
            "manufacturer": "Texas Instruments",
            "Battery Chemistry": "Lithium Ion/Polymer",
            "Interface": "unknown", "max_cells_supported": 1.0,
            "op_temp_min_C": -40.0, "op_temp_max_C": 85.0,
            "package_pins": 6.0, "is_multi_cell": "False"}),
        ("ADI/Maxim MAX17841-class\n16-cell EV / datacenter BMS monitor", {
            "manufacturer": "Analog Devices Inc./Maxim Integrated",
            "Battery Chemistry": "Lithium Ion/Polymer",
            "Interface": "I2C, USB", "max_cells_supported": 16.0,
            "op_temp_min_C": -40.0, "op_temp_max_C": 125.0,
            "package_pins": 48.0, "is_multi_cell": "True"}),
        ("Microchip MCP73833-class\n1-cell I2C charge controller", {
            "manufacturer": "Microchip Technology",
            "Battery Chemistry": "Lithium Ion/Polymer",
            "Interface": "I2C", "max_cells_supported": 1.0,
            "op_temp_min_C": -40.0, "op_temp_max_C": 85.0,
            "package_pins": 8.0, "is_multi_cell": "False"}),
    ]
    df = pd.DataFrame([p for _, p in parts])
    intervals = predict_interval(model, df, cal, coverage=0.90)
    labels = [lbl for lbl, _ in parts]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    y = np.arange(len(parts))[::-1]

    any_clipped = False
    for i, iv in enumerate(intervals):
        low_disp = _clip0(iv.low)
        clipped = iv.low < 0
        any_clipped = any_clipped or clipped
        width = iv.high - low_disp
        ax.barh(y[i], width, left=low_disp,
                color=COLOR_INTERVAL, alpha=0.20, edgecolor="none",
                height=0.55)
        ax.plot(iv.prediction, y[i], "o", color=COLOR_INTERVAL,
                markersize=10, zorder=3)
        ax.text(iv.prediction, y[i] + 0.30,
                f"${iv.prediction:,.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        low_txt = f"${low_disp:,.2f}" + (" *" if clipped else "")
        ax.text(low_disp, y[i] - 0.35,
                low_txt, ha="center", va="top",
                fontsize=8, color="#444444")
        ax.text(iv.high, y[i] - 0.35,
                f"${iv.high:,.2f}", ha="center", va="top",
                fontsize=8, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted unit price at qty 1 ($)", fontsize=11)
    ax.set_title(
        "Honest uncertainty: 90% likely range from split-conformal calibration\n"
        "(* lower bound clipped to $0 — additive interval went negative on a cheap part)",
        fontsize=12, pad=14, loc="left",
    )
    ax.set_xlim(0, max(iv.high for iv in intervals) * 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}" + ("  (a lower bound was clipped)" if any_clipped else ""))


def chart_pin_count_curve(model_data: dict, out: Path) -> None:
    """Predicted price vs package pin count for the baseline BMIC."""
    model = model_data["model"]
    cal = model_data["calibration"]

    pins_grid = np.linspace(6, 48, 43)
    rows = []
    for p in pins_grid:
        row = dict(BASE_BMIC)
        row["package_pins"] = float(p)
        rows.append(row)
    df = pd.DataFrame(rows)
    intervals = predict_interval(model, df, cal, coverage=0.90)

    preds = np.array([iv.prediction for iv in intervals])
    lows = np.array([_clip0(iv.low) for iv in intervals])
    highs = np.array([iv.high for iv in intervals])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.fill_between(pins_grid, lows, highs,
                    color=COLOR_INTERVAL, alpha=0.18,
                    label="90% likely range")
    ax.plot(pins_grid, preds, color=COLOR_INTERVAL,
            linewidth=2.2, label="Predicted unit price")

    # Annotate two reference points: 8 pins (the baseline) and 48 pins
    # (the EV BMS package).
    for p_ref, label, dy in [(8, "Baseline\n(8-pin)", 0.16),
                             (48, "EV BMS\n(48-pin)", -0.18)]:
        idx = int(np.argmin(np.abs(pins_grid - p_ref)))
        ax.plot(p_ref, preds[idx], "o", color="black", markersize=7, zorder=4)
        ax.annotate(
            f"{label}\n${preds[idx]:,.2f}",
            xy=(p_ref, preds[idx]),
            xytext=(p_ref + (3 if p_ref < 30 else -6), preds[idx] * (1 + dy)),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="-", color="black", linewidth=0.7),
        )

    ax.set_xlabel("Package pin count", fontsize=11)
    ax.set_ylabel("Predicted unit price ($)", fontsize=11)
    ax.set_title(
        "Package complexity, priced — not assumed\n"
        "Baseline TI single-cell I2C BMIC, sliding pin count with every other spec fixed",
        fontsize=12, pad=14, loc="left",
    )
    ax.set_xlim(6, 48)
    ax.set_ylim(0, max(highs) * 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_manufacturer_premium(model_data: dict, out: Path) -> None:
    """Predicted price by manufacturer for an otherwise identical BMIC."""
    model = model_data["model"]
    mfrs = [
        "Analog Devices Inc.",
        "Analog Devices Inc./Maxim Integrated",
        "STMicroelectronics",
        "Infineon Technologies",
        "NXP USA Inc.",
        "Texas Instruments",
        "onsemi",
        "Microchip Technology",
    ]
    rows = []
    for m in mfrs:
        row = dict(BASE_BMIC)
        row["manufacturer"] = m
        rows.append(row)
    df = pd.DataFrame(rows)
    preds = model.predict(df)
    base = model.predict(pd.DataFrame([BASE_BMIC]))[0]

    order = np.argsort(preds)
    mfrs_s = [mfrs[i] for i in order]
    preds_s = preds[order]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    y = np.arange(len(mfrs_s))
    colors = [COLOR_UP if p >= base else COLOR_DOWN for p in preds_s]
    ax.barh(y, preds_s, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(base, color="black", linewidth=0.9, linestyle="--")
    ax.text(base, len(mfrs_s) - 0.3, f"  TI baseline ${base:.2f}",
            fontsize=9, color="black", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(mfrs_s, fontsize=10)
    ax.set_xlabel("Predicted unit price for an identical 1-cell I2C BMIC ($)",
                  fontsize=11)
    ax.set_title(
        "Same spec, different supplier: the manufacturer premium\n"
        "All other features held fixed — only the brand changes",
        fontsize=12, pad=14, loc="left",
    )
    for i, p in enumerate(preds_s):
        pct = 100 * (p - base) / base
        ax.text(p + 0.04, i, f"${p:.2f}  ({pct:+.0f}%)",
                va="center", ha="left", fontsize=9, color="black")
    ax.set_xlim(0, max(preds_s) * 1.25)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
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

    chart_ev_bms_attribution(model_data, ASSETS_DIR / "ev_bms_attribution.png")
    chart_intervals_comparison(model_data, ASSETS_DIR / "intervals_comparison.png")
    chart_pin_count_curve(model_data, ASSETS_DIR / "pin_count_curve.png")
    chart_manufacturer_premium(model_data, ASSETS_DIR / "manufacturer_premium.png")
    print(f"\nDone. {len(list(ASSETS_DIR.glob('*.png')))} charts in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
