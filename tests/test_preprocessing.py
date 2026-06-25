"""Tests for AutoPreprocessor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import AutoPreprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "num_a": rng.normal(0, 1, 100),
        "num_b": rng.uniform(0, 10, 100),
        "cat_x": rng.choice(["apple", "banana", "cherry"], 100),
        "cat_y": rng.choice(["red", "blue"], 100),
        "target": rng.choice([0, 1], 100),
    })


@pytest.fixture
def missing_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 80).tolist() + [None] * 20,
        "b": rng.uniform(0, 5, 100),
        "cat": rng.choice(["x", "y", None], 100),
        "target": rng.choice([0, 1], 100),
    })
    return df


class TestAutoPreprocessorFitTransform:
    def test_basic_fit_transform(self, sample_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
        )
        X, y = pp.fit_transform(sample_df)
        assert X.shape[0] == sample_df.shape[0]
        assert "target" not in X.columns
        assert y.shape[0] == sample_df.shape[0]

    def test_no_missing_after_imputation(self, missing_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
        )
        X, y = pp.fit_transform(missing_df)
        assert X.isnull().sum().sum() == 0

    def test_scaling_applied(self, sample_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
            scaling="standard",
        )
        X, _ = pp.fit_transform(sample_df)
        numeric = X.select_dtypes(include=[np.number])
        # Standard scaling should produce roughly mean=0
        assert abs(numeric.mean().mean()) < 1.0

    def test_label_encoding(self, sample_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
            cat_encoding="label",
        )
        X, _ = pp.fit_transform(sample_df)
        # All values should be numeric
        assert X.select_dtypes(include=[object]).shape[1] == 0

    def test_onehot_encoding(self, sample_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
            cat_encoding="onehot",
        )
        X, _ = pp.fit_transform(sample_df)
        # Should have more columns than original (OHE expands cats)
        assert X.shape[1] > sample_df.shape[1] - 1  # -1 for target

    def test_id_column_dropped(self, sample_df):
        sample_df["id"] = range(len(sample_df))
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
            id_columns=["id"],
        )
        X, _ = pp.fit_transform(sample_df)
        assert "id" not in X.columns

    def test_report_populated(self, sample_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
        )
        pp.fit_transform(sample_df)
        assert len(pp.report.steps_applied) > 0

    def test_transform_after_fit(self, sample_df):
        pp = AutoPreprocessor(
            target_column="target",
            problem_type="binary_classification",
        )
        pp.fit_transform(sample_df)
        new_data = sample_df.copy()
        X_new, _ = pp.transform(new_data)
        assert X_new.shape[1] == pp.report.feature_names.__len__()

    def test_regression_target_not_encoded(self, sample_df):
        sample_df["reg_target"] = np.random.rand(len(sample_df))
        pp = AutoPreprocessor(
            target_column="reg_target",
            problem_type="regression",
        )
        _, y = pp.fit_transform(sample_df)
        assert pp.target_classes is None
        assert y.dtype in [np.float64, np.float32]
