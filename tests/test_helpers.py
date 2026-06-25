"""Tests for utility helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.helpers import (
    human_readable_size,
    df_memory_usage,
    hash_dataframe,
    safe_divide,
    detect_id_columns,
    infer_problem_type,
)


class TestHumanReadableSize:
    def test_bytes(self):
        assert human_readable_size(512) == "512.0 B"

    def test_kilobytes(self):
        assert human_readable_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert human_readable_size(1_048_576) == "1.0 MB"

    def test_gigabytes(self):
        assert human_readable_size(1_073_741_824) == "1.0 GB"


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_zero_denominator(self):
        assert safe_divide(10, 0) == 0.0

    def test_custom_default(self):
        assert safe_divide(5, 0, default=-1.0) == -1.0

    def test_float_result(self):
        assert abs(safe_divide(1, 3) - 0.3333) < 0.001


class TestDfMemoryUsage:
    def test_returns_string(self):
        df = pd.DataFrame({"a": range(100)})
        result = df_memory_usage(df)
        assert isinstance(result, str)
        assert any(unit in result for unit in ["B", "KB", "MB"])


class TestHashDataframe:
    def test_consistent_hash(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h1 = hash_dataframe(df)
        h2 = hash_dataframe(df)
        assert h1 == h2

    def test_different_data_different_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_hash_is_string(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert isinstance(hash_dataframe(df), str)


class TestDetectIdColumns:
    def test_all_unique_int(self):
        df = pd.DataFrame({"row_id": range(50), "val": range(50)})
        ids = detect_id_columns(df)
        assert "row_id" in ids

    def test_named_id(self):
        df = pd.DataFrame({"id": [1, 2, 3], "x": [4, 5, 6]})
        ids = detect_id_columns(df)
        assert "id" in ids

    def test_uuid_name(self):
        df = pd.DataFrame({"uuid": ["a", "b", "c"], "v": [1, 2, 3]})
        ids = detect_id_columns(df)
        assert "uuid" in ids

    def test_no_id_column(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 3, 4, 4]})
        ids = detect_id_columns(df)
        assert ids == []


class TestInferProblemType:
    @pytest.mark.parametrize("vals,expected", [
        ([0, 1, 0, 1], "binary_classification"),
        ([0, 1, 2, 0, 1, 2], "multiclass_classification"),
        ([1.0, 2.5, 3.7], "regression"),
        (list(range(50)), "regression"),
        ([True, False, True], "binary_classification"),
    ])
    def test_infer(self, vals, expected):
        assert infer_problem_type(pd.Series(vals)) == expected

    def test_string_binary(self):
        s = pd.Series(["yes", "no", "yes", "no"])
        assert infer_problem_type(s) == "binary_classification"

    def test_string_multiclass(self):
        s = pd.Series(["cat", "dog", "bird", "cat", "dog"])
        assert infer_problem_type(s) == "multiclass_classification"
