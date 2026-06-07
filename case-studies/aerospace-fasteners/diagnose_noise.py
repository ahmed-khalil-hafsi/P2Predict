"""Is the data noisy, or is the model just bad? Two heuristics that answer it.

This is the methodology centerpiece of the fastener case study. When a model
caps out at a modest R² it is tempting to keep tuning — but sometimes the data
itself sets a hard ceiling and no amount of tuning gets past it. Before blaming
the model, measure the ceiling. You do that by looking at rows that share
**identical feature values** and asking how much the target still moves.

Heuristic 1 — signature uniqueness
----------------------------------
Group every row by its full feature signature (here: material, head style,
thread spec, dimensions, finish ...). What fraction of rows are *one-offs* — the
only bolt with that exact signature? If most rows are one-offs the model can
never *interpolate* within a known spec; it must *extrapolate* across specs, and
you cannot even measure the noise floor from data alone. High uniqueness is a
yellow flag on its own.

Heuristic 2 — the duplicate-signature noise floor
-------------------------------------------------
For the rows that DO share a signature, split the target's variance into:
  * *between-signature* variance — differences the features CAN explain, and
  * *within-signature* variance — differences between bolts that are, as far as
    the features are concerned, identical. This part is **irreducible**: no
    model can predict it, because the inputs are the same.

The best R² any model can reach is therefore::

    ceiling = 1 - within_variance / total_variance      (on the duplicate subset)

THE GOTCHA: compute this on the *duplicate subset only*. If you divide the
within-variance by the variance over ALL rows, the one-off signatures (which add
zero within-variance by construction) mechanically drag the ceiling toward 1.0 —
a falsely reassuring number. On this dataset that trap turns a fake "0.93
ceiling" into the honest ~0.60.

Run it::

    python case-studies/aerospace-fasteners/diagnose_noise.py

and read the ceiling next to your model's actual R² (see the README Results).
If the model is already near the ceiling, stop tuning — go find better features
or accept that the target is intrinsically noisy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAINING_CSV = HERE / "data" / "bolts_clean.csv"
TARGET = "unit_price_each_usd"

# The feature signature: two bolts with the same values here are "identical"
# as far as the model can see.
SIGNATURE = [
    "material", "head_style", "thread_class", "thread_series", "finish",
    "thread_diameter_in", "length_in", "tensile_strength_psi",
    "threads_per_inch", "width_across_flats_in",
]


def noise_report(df: pd.DataFrame, signature: list[str], target: str) -> dict:
    """Return the two heuristics as plain numbers (also usable from tests)."""
    sig_cols = [c for c in signature if c in df.columns]
    df = df.copy()
    df["_logt"] = np.log(df[target])

    sizes = df.groupby(sig_cols, dropna=False).size()
    n_rows = len(df)
    n_sigs = len(sizes)
    singleton_rows = int(sizes[sizes == 1].sum())

    # noise floor — measured ONLY on rows whose signature repeats
    dup_mask = df.groupby(sig_cols, dropna=False)[target].transform("size") >= 2
    d = df[dup_mask]
    grp_mean = d.groupby(sig_cols, dropna=False)["_logt"].transform("mean")
    ss_within = float(((d["_logt"] - grp_mean) ** 2).sum())
    ss_total = float(((d["_logt"] - d["_logt"].mean()) ** 2).sum())
    irreducible = ss_within / ss_total if ss_total else float("nan")
    ceiling = 1 - irreducible

    # how wide is the price band on identical specs?
    ratios = d.groupby(sig_cols, dropna=False)[target].apply(lambda g: g.max() / g.min())

    return {
        "rows": n_rows,
        "signatures": n_sigs,
        "singleton_rows": singleton_rows,
        "singleton_frac": singleton_rows / n_rows,
        "dup_rows": int(len(d)),
        "irreducible_frac": irreducible,
        "r2_ceiling": ceiling,
        "median_price_band": float(ratios.median()),
    }


def main() -> None:
    if not TRAINING_CSV.exists():
        raise SystemExit(f"Expected {TRAINING_CSV}. Run prepare_data.py first.")
    df = pd.read_csv(TRAINING_CSV)
    r = noise_report(df, SIGNATURE, TARGET)

    print("=" * 68)
    print("NOISY-DATA DIAGNOSIS — fastener catalog prices")
    print("=" * 68)
    print(f"rows = {r['rows']:,}   distinct spec signatures = {r['signatures']:,}")

    print("\nHeuristic 1 — signature uniqueness")
    print(f"  {r['singleton_frac']*100:.0f}% of bolts are one-offs "
          f"(the only bolt with that exact spec).")
    print(f"  -> the model must generalise across specs, not look up a known one.")

    print(f"\nHeuristic 2 — noise floor (on the {r['dup_rows']:,} duplicate-signature rows)")
    print(f"  irreducible within-signature variance = {r['irreducible_frac']*100:.0f}% of total")
    print(f"  => best achievable R² (log price) ceiling ≈ {r['r2_ceiling']:.2f}")
    print(f"  identical specs are cataloged across a {r['median_price_band']:.1f}x "
          f"price band (median).")

    print("\nRead this next to the model's actual R² (README Results). A model")
    print("already near the ceiling isn't under-tuned — the target is noisy.")


if __name__ == "__main__":
    main()
