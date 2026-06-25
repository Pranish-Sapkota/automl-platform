"""Shared utility helpers."""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def human_readable_size(num_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def df_memory_usage(df: pd.DataFrame) -> str:
    """Return human-readable memory usage for a DataFrame."""
    mem = df.memory_usage(deep=True).sum()
    return human_readable_size(mem)


def hash_dataframe(df: pd.DataFrame) -> str:
    """Generate a stable hash for a DataFrame."""
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return hashlib.md5(buf.getvalue()).hexdigest()
    except Exception:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.md5(csv_bytes).hexdigest()


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns default instead of ZeroDivisionError."""
    return a / b if b != 0 else default


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Heuristically detect ID-like columns to exclude from modelling."""
    candidates: list[str] = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            candidates.append(col)
        elif col.lower() in {"id", "index", "uuid", "key", "row_id", "record_id"}:
            candidates.append(col)
    return list(set(candidates))


def infer_problem_type(series: pd.Series) -> str:
    """Infer ML problem type from target variable."""
    n_unique = series.nunique()
    dtype = series.dtype

    if pd.api.types.is_float_dtype(dtype):
        return "regression"
    if pd.api.types.is_integer_dtype(dtype):
        if n_unique <= 20:
            return "binary_classification" if n_unique == 2 else "multiclass_classification"
        return "regression"
    if pd.api.types.is_bool_dtype(dtype):
        return "binary_classification"
    # Categorical / object
    if n_unique == 2:
        return "binary_classification"
    if n_unique <= 20:
        return "multiclass_classification"
    return "regression"


def timer(func):  # type: ignore[return]
    """Simple timing decorator."""
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__qualname__} took {elapsed:.3f}s")
        return result

    return wrapper
