"""AutoML engine: FLAML primary + individual model training + leaderboard."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    name: str
    algorithm: str
    model: Any
    metrics: dict[str, float]
    train_time: float
    params: dict[str, Any]
    is_flaml: bool = False


@dataclass
class AutoMLResult:
    problem_type: str
    best_model_name: str
    best_model: Any
    leaderboard: list[ModelResult] = field(default_factory=list)
    flaml_model: Any = None
    feature_names: list[str] = field(default_factory=list)
    target_classes: list | None = None
    training_time: float = 0.0


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
) -> dict[str, float]:
    avg = "binary" if n_classes == 2 else "weighted"
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


class AutoMLEngine:
    """Trains multiple models, runs FLAML, and builds a leaderboard."""

    def __init__(
        self,
        problem_type: str,
        time_budget: int = 60,
        random_state: int = 42,
        cv_folds: int = 5,
    ) -> None:
        self.problem_type = problem_type
        self.time_budget = time_budget
        self.random_state = random_state
        self.cv_folds = cv_folds
        self._is_classification = problem_type in (
            "binary_classification", "multiclass_classification"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        target_classes: list | None = None,
        progress_callback: Any = None,
    ) -> AutoMLResult:
        t0 = time.perf_counter()
        results: list[ModelResult] = []
        n_classes = y_train.nunique()

        models = self._get_model_zoo()
        total = len(models) + 1  # +1 for FLAML

        for i, (name, model) in enumerate(models.items()):
            if progress_callback:
                progress_callback(i / total, f"Training {name}…")
            try:
                result = self._train_single(
                    name, model, X_train, y_train, X_test, y_test, n_classes
                )
                results.append(result)
                logger.info("✓ %s — metrics: %s", name, result.metrics)
            except Exception as exc:
                logger.warning("✗ %s failed: %s", name, exc)

        # FLAML
        if progress_callback:
            progress_callback((len(models)) / total, "Running FLAML AutoML…")
        flaml_result = self._run_flaml(X_train, y_train, X_test, y_test, n_classes)
        if flaml_result:
            results.append(flaml_result)

        if progress_callback:
            progress_callback(1.0, "Finalising leaderboard…")

        # Sort leaderboard
        sort_key = "f1" if self._is_classification else "r2"
        results.sort(key=lambda r: r.metrics.get(sort_key, 0), reverse=True)

        best = results[0]
        total_time = time.perf_counter() - t0

        return AutoMLResult(
            problem_type=self.problem_type,
            best_model_name=best.name,
            best_model=best.model,
            leaderboard=results,
            flaml_model=flaml_result.model if flaml_result else None,
            feature_names=feature_names,
            target_classes=target_classes,
            training_time=total_time,
        )

    def save_model(self, model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(path))
        logger.info("Model saved to %s", path)

    def load_model(self, path: Path) -> Any:
        return joblib.load(str(path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model_zoo(self) -> dict[str, Any]:
        rs = self.random_state
        if self._is_classification:
            return {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1),
                "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=rs, n_jobs=-1),
                "Decision Tree": DecisionTreeClassifier(random_state=rs),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
            }
        else:
            return {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=-1),
                "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=rs, n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(random_state=rs),
                "Ridge Regression": Ridge(),
            }

    def _train_single(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_classes: int,
    ) -> ModelResult:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        if self._is_classification:
            metrics = _classification_metrics(
                y_test.values, y_pred, y_proba, n_classes
            )
        else:
            metrics = _regression_metrics(y_test.values, y_pred)

        return ModelResult(
            name=name,
            algorithm=type(model).__name__,
            model=model,
            metrics=metrics,
            train_time=train_time,
            params=model.get_params(),
        )

    def _run_flaml(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_classes: int,
    ) -> ModelResult | None:
        try:
            from flaml import AutoML  # type: ignore

            task = (
                "classification" if self._is_classification else "regression"
            )
            automl = AutoML()
            t0 = time.perf_counter()
            automl.fit(
                X_train, y_train,
                task=task,
                time_budget=self.time_budget,
                estimator_list=["xgboost", "lgbm", "catboost", "rf", "extra_tree"],
                verbose=0,
                seed=self.random_state,
            )
            train_time = time.perf_counter() - t0

            y_pred = automl.predict(X_test)
            y_proba = automl.predict_proba(X_test) if self._is_classification else None

            if self._is_classification:
                metrics = _classification_metrics(y_test.values, y_pred, y_proba, n_classes)
            else:
                metrics = _regression_metrics(y_test.values, y_pred)

            return ModelResult(
                name="FLAML AutoML",
                algorithm=str(automl.best_estimator),
                model=automl,
                metrics=metrics,
                train_time=train_time,
                params=automl.best_config,
                is_flaml=True,
            )
        except Exception as exc:
            logger.error("FLAML failed: %s", exc)
            return None

    def _try_boosting(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_classes: int,
    ) -> ModelResult | None:
        """Try XGBoost / LightGBM / CatBoost individually."""
        rs = self.random_state
        try:
            if name == "XGBoost":
                import xgboost as xgb  # type: ignore
                objective = "binary:logistic" if n_classes == 2 else (
                    "multi:softprob" if self._is_classification else "reg:squarederror"
                )
                params = {"n_estimators": 200, "random_state": rs, "n_jobs": -1,
                          "objective": objective, "verbosity": 0}
                if n_classes > 2 and self._is_classification:
                    params["num_class"] = n_classes
                model = xgb.XGBClassifier(**params) if self._is_classification else xgb.XGBRegressor(**params)
            elif name == "LightGBM":
                import lightgbm as lgb  # type: ignore
                model = (lgb.LGBMClassifier(random_state=rs, verbosity=-1, n_jobs=-1)
                         if self._is_classification
                         else lgb.LGBMRegressor(random_state=rs, verbosity=-1, n_jobs=-1))
            elif name == "CatBoost":
                import catboost as cb  # type: ignore
                model = (cb.CatBoostClassifier(random_state=rs, verbose=0, iterations=200)
                         if self._is_classification
                         else cb.CatBoostRegressor(random_state=rs, verbose=0, iterations=200))
            else:
                return None

            return self._train_single(name, model, X_train, y_train, X_test, y_test, n_classes)
        except Exception as exc:
            logger.warning("Boosting model %s failed: %s", name, exc)
            return None

    def train_with_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        target_classes: list | None = None,
        progress_callback: Any = None,
    ) -> AutoMLResult:
        """Full training including XGB, LGB, CatBoost."""
        t0 = time.perf_counter()
        results: list[ModelResult] = []
        n_classes = y_train.nunique()

        base_models = self._get_model_zoo()
        boost_models = ["XGBoost", "LightGBM", "CatBoost"]
        all_names = list(base_models.keys()) + boost_models + ["FLAML AutoML"]
        total = len(all_names)

        for i, (name, model) in enumerate(base_models.items()):
            if progress_callback:
                progress_callback(i / total, f"Training {name}…")
            try:
                r = self._train_single(name, model, X_train, y_train, X_test, y_test, n_classes)
                results.append(r)
            except Exception as exc:
                logger.warning("✗ %s: %s", name, exc)

        for j, bname in enumerate(boost_models):
            if progress_callback:
                progress_callback((len(base_models) + j) / total, f"Training {bname}…")
            r = self._try_boosting(bname, X_train, y_train, X_test, y_test, n_classes)
            if r:
                results.append(r)

        if progress_callback:
            progress_callback((total - 1) / total, "Running FLAML AutoML…")
        flaml_result = self._run_flaml(X_train, y_train, X_test, y_test, n_classes)
        if flaml_result:
            results.append(flaml_result)

        if progress_callback:
            progress_callback(1.0, "Done!")

        sort_key = "f1" if self._is_classification else "r2"
        results.sort(key=lambda r: r.metrics.get(sort_key, 0), reverse=True)
        best = results[0]

        return AutoMLResult(
            problem_type=self.problem_type,
            best_model_name=best.name,
            best_model=best.model,
            leaderboard=results,
            flaml_model=flaml_result.model if flaml_result else None,
            feature_names=feature_names,
            target_classes=target_classes,
            training_time=time.perf_counter() - t0,
        )
