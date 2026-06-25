"""SHAP-based model explainability: global importance, local explanations, plots."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExplainabilityResult:
    feature_names: list[str] = field(default_factory=list)
    global_importance: dict[str, float] = field(default_factory=dict)
    shap_values: Any = None
    expected_value: Any = None
    sample_shap_values: Any = None
    sample_index: int = 0


class SHAPExplainer:
    """Unified SHAP explainer that handles Tree, Linear, and Kernel explainers."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self._explainer: Any = None

    def build_explainer(self, X_background: pd.DataFrame) -> None:
        """Create the appropriate SHAP explainer for the model."""
        import shap

        model = self.model
        # Unwrap FLAML AutoML if needed
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "estimator"):
            model = model.estimator

        try:
            self._explainer = shap.TreeExplainer(model)
            logger.info("Using SHAP TreeExplainer")
            return
        except Exception:
            pass

        try:
            self._explainer = shap.LinearExplainer(model, X_background)
            logger.info("Using SHAP LinearExplainer")
            return
        except Exception:
            pass

        # Fallback: KernelExplainer on a small sample
        bg = shap.sample(X_background, min(100, len(X_background)))
        self._explainer = shap.KernelExplainer(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            bg,
        )
        logger.info("Using SHAP KernelExplainer (slow fallback)")

    def explain(
        self, X: pd.DataFrame, max_samples: int = 200
    ) -> ExplainabilityResult:
        """Compute SHAP values for a dataset."""
        import shap

        if self._explainer is None:
            self.build_explainer(X)

        sample = X.iloc[:max_samples] if len(X) > max_samples else X
        result = ExplainabilityResult(feature_names=self.feature_names)

        try:
            shap_vals = self._explainer.shap_values(sample)

            # For multi-output: average across classes
            if isinstance(shap_vals, list):
                arr = np.array(shap_vals)
                # shape: (n_classes, n_samples, n_features)
                mean_abs = np.abs(arr).mean(axis=(0, 1))
                shap_for_plot = arr[1] if arr.shape[0] == 2 else arr.mean(axis=0)
            else:
                mean_abs = np.abs(shap_vals).mean(axis=0)
                shap_for_plot = shap_vals

            result.shap_values = shap_for_plot
            result.expected_value = (
                self._explainer.expected_value[1]
                if isinstance(self._explainer.expected_value, (list, np.ndarray))
                else self._explainer.expected_value
            )
            result.global_importance = {
                name: float(val)
                for name, val in zip(self.feature_names, mean_abs)
            }
            result.sample_shap_values = shap_for_plot
        except Exception as exc:
            logger.error("SHAP explain failed: %s", exc)

        return result

    def get_sorted_importance(
        self, result: ExplainabilityResult, top_n: int = 20
    ) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances."""
        if not result.global_importance:
            return pd.DataFrame()
        df = pd.DataFrame(
            list(result.global_importance.items()),
            columns=["Feature", "Importance"],
        )
        df = df.sort_values("Importance", ascending=False).head(top_n)
        df["Importance"] = df["Importance"].round(4)
        return df.reset_index(drop=True)

    def explain_single(
        self, X: pd.DataFrame, index: int = 0
    ) -> dict[str, float]:
        """Explain a single prediction."""
        import shap

        if self._explainer is None:
            self.build_explainer(X)

        sample = X.iloc[[index]]
        try:
            shap_vals = self._explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0] if len(shap_vals) == 2 else shap_vals[0][0]
            else:
                vals = shap_vals[0]
            return {name: float(v) for name, v in zip(self.feature_names, vals)}
        except Exception as exc:
            logger.error("Single SHAP explanation failed: %s", exc)
            return {}
