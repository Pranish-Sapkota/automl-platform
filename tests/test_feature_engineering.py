"""Tests for AutoFeatureEngineer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import AutoFeatureEngineer


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.uniform(1, 10, 100),
        "c": rng.normal(5, 2, 100),
        "target": rng.choice([0, 1], 100),
    })


@pytest.fixture
def date_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "event_date": dates,
        "value": rng.normal(0, 1, 100),
        "target": rng.choice([0, 1], 100),
    })


class TestAutoFeatureEngineer:
    def test_fit_transform_returns_dataframe(self, numeric_df):
        fe = AutoFeatureEngineer(target_column="target")
        result = fe.fit_transform(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_interaction_features_created(self, numeric_df):
        fe = AutoFeatureEngineer(
            create_interactions=True,
            create_date_features=False,
            create_aggregations=False,
            target_column="target",
        )
        result = fe.fit_transform(numeric_df)
        interaction_cols = [c for c in result.columns if "_x_" in c or "_div_" in c]
        assert len(interaction_cols) > 0

    def test_aggregation_features_created(self, numeric_df):
        fe = AutoFeatureEngineer(
            create_interactions=False,
            create_date_features=False,
            create_aggregations=True,
            target_column="target",
        )
        result = fe.fit_transform(numeric_df)
        agg_cols = [c for c in result.columns if c.startswith("_agg_")]
        assert len(agg_cols) > 0

    def test_date_features_created(self, date_df):
        fe = AutoFeatureEngineer(
            create_interactions=False,
            create_date_features=True,
            create_aggregations=False,
            target_column="target",
        )
        result = fe.fit_transform(date_df)
        date_cols = [c for c in result.columns if "year" in c or "month" in c or "day" in c]
        assert len(date_cols) > 0
        # Original date column should be dropped
        assert "event_date" not in result.columns

    def test_date_features_weekend(self, date_df):
        fe = AutoFeatureEngineer(create_date_features=True, target_column="target")
        result = fe.fit_transform(date_df)
        assert "event_date_is_weekend" in result.columns

    def test_more_features_than_original(self, numeric_df):
        fe = AutoFeatureEngineer(target_column="target")
        result = fe.fit_transform(numeric_df)
        assert result.shape[1] > numeric_df.shape[1]

    def test_transform_consistency(self, numeric_df):
        fe = AutoFeatureEngineer(
            create_interactions=True,
            create_aggregations=True,
            create_date_features=False,
            target_column="target",
        )
        result_fit = fe.fit_transform(numeric_df)
        result_transform = fe.transform(numeric_df.copy())
        assert result_fit.shape[1] == result_transform.shape[1]

    def test_report_populated(self, numeric_df):
        fe = AutoFeatureEngineer(target_column="target")
        fe.fit_transform(numeric_df)
        assert fe.report.final_feature_count >= fe.report.original_feature_count

    def test_no_features_if_all_disabled(self, numeric_df):
        fe = AutoFeatureEngineer(
            create_interactions=False,
            create_date_features=False,
            create_aggregations=False,
            target_column="target",
        )
        result = fe.fit_transform(numeric_df)
        assert result.shape[1] == numeric_df.shape[1]

    def test_target_not_used_in_interactions(self, numeric_df):
        fe = AutoFeatureEngineer(
            create_interactions=True,
            create_aggregations=False,
            create_date_features=False,
            target_column="target",
        )
        result = fe.fit_transform(numeric_df)
        # No interaction column should reference 'target'
        bad = [c for c in result.columns if "target" in c and ("_x_" in c or "_div_" in c)]
        assert len(bad) == 0
