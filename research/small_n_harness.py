"""Shared dataset harness for the small-n findings.

The three small-n studies (`small_n_conformal`, `selection_noise`,
`learning_curve`) all ask the same shape of question: **what happens at the
dataset size P2Predict actually sees?** The case-study CSVs are 16k-400k
rows, which is not that size. This module turns each of them into a
simulator for a realistic procurement catalog:

    draw `n` rows as "the buyer's dataset", and keep a large disjoint
    sample as ground truth to grade against.

That disjoint eval set is the point. A 150-part catalog cannot both train a
model and honestly measure it, which is precisely the problem under study —
so the measurement has to come from outside the simulated catalog.

Read-only: trains throwaway models in memory, never writes a model or
touches core.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SEED = 11

# Sizes a procurement catalog actually comes in. 150 is the anchor: the
# battery-management-ICs case study is a real 150-part catalog, and the
# project's working assumption is 50-300 parts.
CATALOG_SIZES = (75, 150, 300)

# Rows held back to grade against. Large enough that the grade itself is
# not the noisy thing in the experiment.
EVAL_N = 3000

# (label, csv (full), csv (committed sample fallback), target, features)
DATASETS = [
    ("used cars",
     "case-studies/used-cars/data/vehicles_training.csv",
     "case-studies/used-cars/data-sample/vehicles_sample.csv",
     "price",
     ["year", "odometer", "manufacturer", "condition", "fuel",
      "transmission", "drive", "type", "state", "paint_color"]),
    ("heavy equipment",
     "case-studies/heavy-equipment-sales/data/bulldozers_training.csv",
     "case-studies/heavy-equipment-sales/data-sample/bulldozers_sample.csv",
     "sale_price_usd",
     ["age_at_sale", "sale_year", "product_group", "product_size",
      "enclosure", "state"]),
    ("aerospace fasteners",
     "case-studies/aerospace-fasteners/data/bolts_clean.csv",
     "case-studies/aerospace-fasteners/data-sample/bolts_sample.csv",
     "unit_price_each_usd",
     ["length_in", "head_style", "material", "tensile_strength_psi",
      "thread_diameter_in", "finish", "thread_class", "threads_per_inch",
      "thread_series", "width_across_flats_in"]),
]


class Catalog:
    """One case-study CSV, ready to be sampled into synthetic catalogs."""

    def __init__(self, label: str, source: str, df: pd.DataFrame,
                 target: str, features: list[str]):
        self.label = label
        self.source = source
        self.target = target
        self.features = features
        self.df = df
        self.numeric = [f for f in features
                        if pd.api.types.is_numeric_dtype(df[f])]
        self.categorical = [f for f in features if f not in self.numeric]

    def draw(self, n: int, rng: np.random.Generator):
        """Return ``(X_cat, y_cat, X_eval, y_eval)``.

        ``_cat`` is the simulated n-part catalog the buyer would hand
        P2Predict; ``_eval`` is a disjoint sample used only to grade.
        """
        idx = rng.permutation(len(self.df))
        cat_idx = idx[:n]
        eval_idx = idx[n:n + EVAL_N]
        cat = self.df.iloc[cat_idx]
        ev = self.df.iloc[eval_idx]
        return (cat[self.features], cat[self.target].to_numpy(float),
                ev[self.features], ev[self.target].to_numpy(float))


def load(label: str, csv: str, sample: str, target: str,
         features: list[str]) -> Catalog | None:
    path, used = REPO / csv, csv
    if not path.exists():
        path, used = REPO / sample, sample
    if not path.exists():
        print(f"skip {label}: no data at {csv} or {sample}", file=sys.stderr)
        return None

    df = pd.read_csv(path, low_memory=False)
    keep = [c for c in features if c in df.columns]
    if len(keep) != len(features):
        missing = set(features) - set(keep)
        print(f"skip {label}: missing columns {sorted(missing)}", file=sys.stderr)
        return None

    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    # Every study here forces the log-target wrap (see the study docs), so
    # the target must be strictly positive and finite.
    df = df[np.isfinite(df[target])]
    return Catalog(label, used, df.reset_index(drop=True), target, features)


def load_all() -> list[Catalog]:
    return [c for c in (load(*d) for d in DATASETS) if c is not None]


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed relative error in log space — the scale prices live on."""
    ok = (y_true > 0) & (y_pred > 0) & np.isfinite(y_pred)
    out = np.full(y_true.shape, np.nan)
    out[ok] = np.log(y_pred[ok]) - np.log(y_true[ok])
    return out


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error — the error a buyer feels."""
    ok = (y_true > 0) & np.isfinite(y_pred)
    if not ok.any():
        return float("nan")
    return float(np.median(np.abs(y_pred[ok] - y_true[ok]) / y_true[ok]) * 100)
