"""Tests for AutoMLEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.automl import AutoMLEngine, AutoMLResult


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    series_y = pd.Series(y, name="target")
    return train_test_split(df_X, series_y, test_size=0.2, random_state=42)


@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=300, n_features=8, n_classes=3,
        n_informative=6, random_state=42
    )
    df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    series_y = pd.Series(y, name="target")
    return train_test_split(df_X, series_y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    series_y = pd.Series(y, name="target")
    return train_test_split(df_X, series_y, test_size=0.2, random_state=42)


class TestAutoMLEngineBinary:
    def test_returns_automl_result(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        engine = AutoMLEngine("binary_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        assert isinstance(result, AutoMLResult)

    def test_leaderboard_not_empty(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        engine = AutoMLEngine("binary_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        assert len(result.leaderboard) > 0

    def test_best_model_set(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        engine = AutoMLEngine("binary_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        assert result.best_model is not None
        assert result.best_model_name != ""

    def test_metrics_present(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        engine = AutoMLEngine("binary_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        best = result.leaderboard[0]
        assert "accuracy" in best.metrics
        assert "f1" in best.metrics
        assert 0.0 <= best.metrics["accuracy"] <= 1.0

    def test_leaderboard_sorted_by_f1(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        engine = AutoMLEngine("binary_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        f1_scores = [r.metrics.get("f1", 0) for r in result.leaderboard]
        assert f1_scores == sorted(f1_scores, reverse=True)

    def test_model_can_predict(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        engine = AutoMLEngine("binary_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        preds = result.best_model.predict(X_test)
        assert len(preds) == len(y_test)


class TestAutoMLEngineRegression:
    def test_regression_metrics(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        engine = AutoMLEngine("regression", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        best = result.leaderboard[0]
        assert "rmse" in best.metrics
        assert "mae" in best.metrics
        assert "r2" in best.metrics

    def test_regression_sorted_by_r2(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        engine = AutoMLEngine("regression", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        r2_scores = [r.metrics.get("r2", 0) for r in result.leaderboard]
        assert r2_scores == sorted(r2_scores, reverse=True)

    def test_regression_training_time_recorded(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        engine = AutoMLEngine("regression", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        for r in result.leaderboard:
            assert r.train_time > 0


class TestAutoMLEngineMulticlass:
    def test_multiclass_returns_result(self, multiclass_data):
        X_train, X_test, y_train, y_test = multiclass_data
        engine = AutoMLEngine("multiclass_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        assert isinstance(result, AutoMLResult)
        assert len(result.leaderboard) > 0

    def test_multiclass_accuracy_in_bounds(self, multiclass_data):
        X_train, X_test, y_train, y_test = multiclass_data
        engine = AutoMLEngine("multiclass_classification", time_budget=10)
        result = engine.train(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
        )
        best = result.leaderboard[0]
        assert 0.0 <= best.metrics.get("accuracy", 0) <= 1.0
