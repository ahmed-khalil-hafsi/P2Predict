"""Generate the case-study charts into assets/.

Three figures that carry the honest-study narrative:

  * material_premium.png  — the headline lever: per-each price up the material
                            grade ladder (alloy steel -> CRES -> A286 -> Ti ->
                            Nickel), holding size and style fixed. A what-if
                            sweep on the trained model.
  * dimension_curve.png   — price vs bolt length, the cleanest cost ruler.
  * noise_floor.png       — the methodology centerpiece: among bolts with
                            IDENTICAL specs, how wide is the cataloged price
                            band? Visualises why R² is capped (see
                            diagnose_noise.py).

Run after training::

    python case-studies/aerospace-fasteners/generate_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from p2predict import load_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
TRAINING_CSV = HERE / "data" / "bolts_clean.csv"

INK = "#1a2b4a"
ACCENT = "#c8783c"

# Baseline commodity bolt (matches extract_insights.py / bolts_clean.csv).
BASELINE = {
    "material": "Alloy Steel", "head_style": "HEXAGON",
    "thread_diameter_in": 0.25, "length_in": 1.0, "thread_class": "3A",
    "thread_series": "UNF", "finish": "Cadmium",
    "tensile_strength_psi": 125000, "threads_per_inch": 28,
    "width_across_flats_in": 0.4375,
}
MATERIAL_LADDER = ["Alloy Steel", "Corrosion Resisting Steel",
                   "A286 Superalloy", "Titanium", "Nickel Alloy"]
LENGTH_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def _latest_model() -> Path:
    cands = sorted(MODELS_DIR.glob("*_unit_price_each_usd_*.model"))
    if not cands:
        sys.exit("Train a fastener model first — see README.")
    return cands[-1]


def _sweep(model, feature: str, values: list) -> list[float]:
    rows = []
    for v in values:
        part = dict(BASELINE)
        part[feature] = v
        rows.append(part)
    return list(model.predict(pd.DataFrame(rows)))


def chart_material_premium(model) -> None:
    preds = _sweep(model, "material", MATERIAL_LADDER)
    base = preds[0]
    labels = ["Alloy Steel\n(commodity)", "CRES", "A286\nsuperalloy",
              "Titanium", "Nickel\nalloy"]
    colors = [INK] + [ACCENT] * (len(preds) - 1)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, preds, color=colors)
    for b, p in zip(bars, preds):
        prem = (p / base - 1) * 100
        tag = "baseline" if abs(prem) < 1 else f"+{prem:.0f}%"
        ax.text(b.get_x() + b.get_width() / 2, p, f"${p:.2f}\n{tag}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("Predicted per-each price (USD)")
    ax.set_title("Material grade premium — same 1/4-28 x 1.0\" bolt, "
                 "size & style held fixed", fontsize=11, color=INK)
    ax.set_ylim(0, max(preds) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "material_premium.png", dpi=130)
    plt.close(fig)


def chart_dimension_curve(model) -> None:
    preds = _sweep(model, "length_in", LENGTH_GRID)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(LENGTH_GRID, preds, "-o", color=ACCENT, lw=2, markersize=7)
    for x, p in zip(LENGTH_GRID, preds):
        ax.text(x, p, f"  ${p:.2f}", va="bottom", ha="left", fontsize=9, color=INK)
    ax.set_xlabel("Bolt length (inches)")
    ax.set_ylabel("Predicted per-each price (USD)")
    ax.set_title("Length is a clean cost ruler — commodity alloy-steel hex, "
                 "1/4-28", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "dimension_curve.png", dpi=130)
    plt.close(fig)


def chart_noise_floor(df: pd.DataFrame) -> None:
    """For bolts whose full spec signature repeats, plot the spread of cataloged
    prices within each signature — the irreducible noise the model can't beat."""
    sig = ["material", "head_style", "thread_class", "thread_series", "finish",
           "thread_diameter_in", "length_in", "tensile_strength_psi",
           "threads_per_inch", "width_across_flats_in"]
    sig = [c for c in sig if c in df.columns]
    g = df.groupby(sig, dropna=False)["unit_price_each_usd"]
    sizes = g.transform("size")
    dup = df[sizes >= 2].copy()
    dup["_band"] = g.transform(lambda s: s.max() / s.min())[sizes >= 2]
    ratios = dup.groupby(sig, dropna=False)["unit_price_each_usd"].apply(
        lambda s: s.max() / s.min())

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(np.clip(ratios, 1, 20), bins=40, color=INK)
    med = ratios.median()
    ax.axvline(med, color=ACCENT, lw=2,
               label=f"median {med:.1f}x price band")
    ax.set_xlabel("max ÷ min cataloged price among IDENTICAL-spec bolts")
    ax.set_ylabel("Spec signatures")
    ax.set_title("Why the ceiling is ~0.60: identical specs, very different "
                 "prices", fontsize=11, color=INK)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "noise_floor.png", dpi=130)
    plt.close(fig)


def main() -> None:
    model = load_model(_latest_model())["model"]
    df = pd.read_csv(TRAINING_CSV)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    chart_material_premium(model)
    chart_dimension_curve(model)
    chart_noise_floor(df)
    print(f"Wrote material_premium.png, dimension_curve.png, noise_floor.png "
          f"to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
