"""Fetch a TDC ADMET benchmark and turn SMILES into a flat descriptor table.

P2Predict eats tabular spec sheets, not molecular graphs. This is the only
domain-specific step in the whole eval: RDKit turns each molecule into ~210
numeric 2D descriptors (molecular weight, logP, ring counts, topological
indices...), which is exactly the shape of a P2Predict parts CSV -- one row
per entity, one column per spec.

Descriptors are computed per-molecule and independently, so nothing leaks
across the train/test boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")


def descriptors_for(smiles: str) -> dict | None:
    """All RDKit 2D descriptors for one SMILES string, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.CalcMolDescriptors(mol)


def featurize(df: pd.DataFrame, smiles_col: str = "Drug") -> pd.DataFrame:
    """Expand a TDC split frame into Drug_ID | <descriptors...> | Y."""
    rows, keep = [], []
    for idx, smiles in zip(df.index, df[smiles_col]):
        desc = descriptors_for(smiles)
        if desc is not None:
            rows.append(desc)
            keep.append(idx)

    feats = pd.DataFrame(rows, index=keep)
    out = df.loc[keep].drop(columns=[smiles_col]).join(feats)
    dropped = len(df) - len(out)
    if dropped:
        print(f"    (dropped {dropped} unparseable SMILES)")
    return out


def clean(frames: list[pd.DataFrame], target: str = "Y") -> list[pd.DataFrame]:
    """Drop descriptor columns that are unusable, deciding purely on TRAIN.

    frames[0] is treated as the training frame; the column decision is made
    there and applied to the rest, so the test set never influences which
    features the model gets to see.
    """
    train = frames[0]
    feature_cols = [c for c in train.columns if c not in (target, "Drug_ID")]

    numeric = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    # Unusable = mostly missing, or constant (no signal, and it breaks scaling).
    mostly_missing = numeric.columns[numeric.isna().mean() > 0.10]
    constant = numeric.columns[numeric.nunique(dropna=True) <= 1]
    drop = sorted(set(mostly_missing) | set(constant))
    if drop:
        print(f"    (dropped {len(drop)} unusable descriptor columns)")

    cleaned = []
    for frame in frames:
        out = frame.drop(columns=drop)
        cols = [c for c in out.columns if c not in (target, "Drug_ID")]
        out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
        # Median-impute the few remaining gaps using TRAIN medians only.
        out[cols] = out[cols].fillna(train[cols].median(numeric_only=True))
        cleaned.append(out)
    return cleaned
