import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the p2predict package importable when running tests without first
# doing `pip install -e .`. In CI we install the package properly; this
# fallback keeps local "git clone + pytest" working too.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for path in (SRC, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def _synthetic_parts(n=200, seed=0, skewed=False, with_date=False):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(1, 50, n)
    region = rng.choice(["EU", "CN", "SG", "US"], n)
    supplier = rng.choice(["A", "B", "C"], n)
    size = rng.choice(["Small", "Standard", "Large"], n)

    base = 0.08 * weight + np.where(region == "EU", 0.5, 0.0) + np.where(size == "Large", 0.7, 0.0)
    noise = rng.normal(0, 0.1, n)
    price = np.clip(base + noise, 0.05, None)
    if skewed:
        # Exponentiate to guarantee a strongly right-skewed (log-normal) target.
        price = np.exp(price)

    df = pd.DataFrame({
        "Weight": weight,
        "Region": region,
        "Supplier": supplier,
        "Size": size,
        "Price": price,
    })
    if with_date:
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        # Mild upward drift over time so chronological vs random splits actually differ.
        df["Date"] = dates
        drift = np.linspace(0, 0.5, n)
        df["Price"] = df["Price"] + drift
    return df


@pytest.fixture
def synthetic_parts():
    return _synthetic_parts(n=200)


@pytest.fixture
def synthetic_parts_skewed():
    return _synthetic_parts(n=200, skewed=True)


@pytest.fixture
def synthetic_parts_with_date():
    return _synthetic_parts(n=200, with_date=True)


@pytest.fixture
def tiny_parts():
    """Small dataset for quick fits where HPO budgets aren't worth the cost."""
    return _synthetic_parts(n=60, seed=1)


@pytest.fixture
def csv_path_with_nas(tmp_path):
    df = pd.DataFrame({
        "Weight": [1.0, 2.0, None, 4.0],
        "Region": ["EU", "CN", "SG", None],
        "Price": [1.0, 2.0, 3.0, 4.0],
    })
    p = tmp_path / "with_nas.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_path_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    return str(p)


@pytest.fixture
def csv_path_clean(tmp_path, synthetic_parts):
    p = tmp_path / "clean.csv"
    synthetic_parts.to_csv(p, index=False)
    return str(p)
