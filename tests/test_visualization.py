"""Tests for visualization chart builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.charts import (
    missing_values_heatmap,
    correlation_heatmap,
    distribution_plot,
    target_distribution,
    outlier_boxplots,
    leaderboard_chart,
    shap_importance_bar,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.uniform(0, 10, 100),
        "c": rng.choice(["x", "y", "z"], 100),
        "target": rng.choice([0, 1], 100),
    })


class TestChartBuilders:
    def test_missing_heatmap_no_missing(self, sample_df):
        fig = missing_values_heatmap(sample_df)
        assert isinstance(fig, go.Figure)

    def test_missing_heatmap_with_missing(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, None]})
        fig = missing_values_heatmap(df)
        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap(self, sample_df):
        fig = correlation_heatmap(sample_df)
        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap_single_column(self):
        df = pd.DataFrame({"only": [1, 2, 3]})
        fig = correlation_heatmap(df)
        assert isinstance(fig, go.Figure)

    def test_distribution_numeric(self, sample_df):
        fig = distribution_plot(sample_df["a"], "a")
        assert isinstance(fig, go.Figure)

    def test_distribution_categorical(self, sample_df):
        fig = distribution_plot(sample_df["c"], "c")
        assert isinstance(fig, go.Figure)

    def test_target_distribution_classification(self, sample_df):
        fig = target_distribution(sample_df["target"], "binary_classification")
        assert isinstance(fig, go.Figure)

    def test_target_distribution_regression(self):
        s = pd.Series(np.random.rand(100))
        fig = target_distribution(s, "regression")
        assert isinstance(fig, go.Figure)

    def test_outlier_boxplots(self, sample_df):
        fig = outlier_boxplots(sample_df, ["a", "b"])
        assert isinstance(fig, go.Figure)

    def test_leaderboard_chart(self):
        df = pd.DataFrame({
            "Model": ["RF", "XGB", "LGBM"],
            "F1": [0.92, 0.95, 0.91],
        })
        fig = leaderboard_chart(df, "F1")
        assert isinstance(fig, go.Figure)

    def test_shap_importance_bar(self):
        imp_df = pd.DataFrame({
            "Feature": ["feat_a", "feat_b", "feat_c"],
            "Importance": [0.5, 0.3, 0.2],
        })
        fig = shap_importance_bar(imp_df)
        assert isinstance(fig, go.Figure)
