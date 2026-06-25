"""Tests for DataService."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.services.data_service import DataService
from src.utils.helpers import infer_problem_type, detect_id_columns


@pytest.fixture
def svc() -> DataService:
    return DataService()


@pytest.fixture
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, 200),
        "feature_b": rng.uniform(0, 10, 200),
        "category": rng.choice(["cat", "dog", "bird"], 200),
        "target": rng.choice([0, 1], 200),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, 300),
        "x2": rng.uniform(0, 5, 300),
        "target": rng.normal(10, 2, 300),
    })


class TestInferProblemType:
    def test_binary(self):
        s = pd.Series([0, 1, 0, 1, 1, 0])
        assert infer_problem_type(s) == "binary_classification"

    def test_multiclass(self):
        s = pd.Series([0, 1, 2, 3, 0, 1])
        assert infer_problem_type(s) == "multiclass_classification"

    def test_regression_float(self):
        s = pd.Series([1.0, 2.5, 3.7, 4.2])
        assert infer_problem_type(s) == "regression"

    def test_regression_large_int(self):
        s = pd.Series(list(range(100)))
        assert infer_problem_type(s) == "regression"

    def test_bool(self):
        s = pd.Series([True, False, True])
        assert infer_problem_type(s) == "binary_classification"


class TestDetectIdColumns:
    def test_all_unique(self):
        df = pd.DataFrame({"id": range(100), "val": range(100)})
        result = detect_id_columns(df)
        assert "id" in result

    def test_named_id(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": [4, 5, 6]})
        result = detect_id_columns(df)
        assert "id" in result

    def test_no_id(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = detect_id_columns(df)
        assert result == []


class TestDataServiceProfile:
    def test_profile_shape(self, svc, binary_df):
        profile = svc.profile(binary_df, "target")
        assert profile.shape == binary_df.shape

    def test_profile_problem_type(self, svc, binary_df):
        profile = svc.profile(binary_df, "target")
        assert profile.problem_type == "binary_classification"

    def test_profile_regression(self, svc, regression_df):
        profile = svc.profile(regression_df, "target")
        assert profile.problem_type == "regression"

    def test_missing_report_no_missing(self, svc, binary_df):
        profile = svc.profile(binary_df)
        assert profile.missing_report["has_missing"] is False

    def test_missing_report_with_missing(self, svc):
        df = pd.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        })
        profile = svc.profile(df, "target")
        assert profile.missing_report["has_missing"] is True
        assert "a" in profile.missing_report["by_column"]

    def test_duplicate_report(self, svc):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        profile = svc.profile(df)
        assert profile.duplicate_report["duplicate_rows"] == 1

    def test_column_type_detection(self, svc, binary_df):
        profile = svc.profile(binary_df)
        assert "feature_a" in profile.numeric_columns
        assert "feature_b" in profile.numeric_columns
        assert "category" in profile.categorical_columns

    def test_recommendations_generated(self, svc, binary_df):
        profile = svc.profile(binary_df, "target")
        assert isinstance(profile.recommendations, list)
        assert len(profile.recommendations) > 0
