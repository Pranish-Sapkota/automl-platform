"""Shared pytest configuration and fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def iris_df() -> pd.DataFrame:
    """Iris dataset as a DataFrame."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df


@pytest.fixture(scope="session")
def titanic_like_df() -> pd.DataFrame:
    """Synthetic Titanic-like dataset."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "pclass": rng.choice([1, 2, 3], n),
        "age": rng.normal(35, 15, n).clip(1, 80),
        "fare": rng.exponential(30, n),
        "sex": rng.choice(["male", "female"], n),
        "embarked": rng.choice(["S", "C", "Q", None], n),
        "sibsp": rng.integers(0, 5, n),
        "parch": rng.integers(0, 3, n),
        "survived": rng.choice([0, 1], n),
    })


@pytest.fixture(scope="session")
def housing_like_df() -> pd.DataFrame:
    """Synthetic housing regression dataset."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=400, n_features=10, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["price"] = y
    return df


@pytest.fixture
def small_binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, 50),
        "x2": rng.uniform(0, 5, 50),
        "cat": rng.choice(["a", "b"], 50),
        "label": rng.choice([0, 1], 50),
    })
