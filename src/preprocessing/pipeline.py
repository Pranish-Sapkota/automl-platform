"""Auto-preprocessing pipeline: imputation, encoding, scaling, outlier treatment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingReport:
    steps_applied: list[str] = field(default_factory=list)
    original_shape: tuple[int, int] = (0, 0)
    final_shape: tuple[int, int] = (0, 0)
    dropped_columns: list[str] = field(default_factory=list)
    encoded_columns: list[str] = field(default_factory=list)
    imputed_columns: list[str] = field(default_factory=list)
    scaled_columns: list[str] = field(default_factory=list)
    outlier_treated_columns: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


class AutoPreprocessor:
    """Stateful auto-preprocessing pipeline with fit/transform semantics."""

    def __init__(
        self,
        target_column: str,
        problem_type: str,
        scaling: str = "robust",          # "standard" | "minmax" | "robust" | "none"
        cat_encoding: str = "auto",       # "label" | "onehot" | "auto"
        outlier_treatment: str = "clip",  # "clip" | "drop" | "none"
        missing_strategy_num: str = "median",
        missing_strategy_cat: str = "most_frequent",
        drop_high_cardinality: int = 50,
        id_columns: list[str] | None = None,
    ) -> None:
        self.target_column = target_column
        self.problem_type = problem_type
        self.scaling = scaling
        self.cat_encoding = cat_encoding
        self.outlier_treatment = outlier_treatment
        self.missing_strategy_num = missing_strategy_num
        self.missing_strategy_cat = missing_strategy_cat
        self.drop_high_cardinality = drop_high_cardinality
        self.id_columns: list[str] = id_columns or []

        self._num_imputer: SimpleImputer | None = None
        self._cat_imputer: SimpleImputer | None = None
        self._scaler: Any = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._onehot_encoder: OneHotEncoder | None = None
        self._onehot_cols: list[str] = []
        self._label_cols: list[str] = []
        self._target_encoder: LabelEncoder | None = None
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._clip_bounds: dict[str, tuple[float, float]] = {}
        self.report = PreprocessingReport()
        self._fitted = False

    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit the pipeline and transform in one pass."""
        self.report.original_shape = df.shape
        df = df.copy()

        # Separate target
        y = df[self.target_column].copy()
        X = df.drop(columns=[self.target_column])

        # Drop ID / irrelevant columns
        drop_cols = [c for c in self.id_columns if c in X.columns]
        X = X.drop(columns=drop_cols)
        self.report.dropped_columns.extend(drop_cols)
        if drop_cols:
            self.report.steps_applied.append(f"Dropped ID columns: {drop_cols}")

        # Identify column types
        self._num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        raw_cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Drop high-cardinality categoricals
        hi_card = [c for c in raw_cat_cols if X[c].nunique() > self.drop_high_cardinality]
        if hi_card:
            X = X.drop(columns=hi_card)
            self.report.dropped_columns.extend(hi_card)
            self.report.steps_applied.append(f"Dropped high-cardinality cols: {hi_card}")
        self._cat_cols = [c for c in raw_cat_cols if c not in hi_card]

        # Numeric imputation
        if self._num_cols:
            self._num_imputer = SimpleImputer(strategy=self.missing_strategy_num)
            X[self._num_cols] = self._num_imputer.fit_transform(X[self._num_cols])
            imputed = [c for c in self._num_cols if df[c].isnull().any()]
            self.report.imputed_columns.extend(imputed)
            if imputed:
                self.report.steps_applied.append(f"Numeric imputation ({self.missing_strategy_num})")

        # Categorical imputation
        if self._cat_cols:
            self._cat_imputer = SimpleImputer(strategy=self.missing_strategy_cat)
            X[self._cat_cols] = self._cat_imputer.fit_transform(X[self._cat_cols])
            self.report.steps_applied.append(f"Categorical imputation ({self.missing_strategy_cat})")

        # Outlier treatment
        if self.outlier_treatment == "clip" and self._num_cols:
            for col in self._num_cols:
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                self._clip_bounds[col] = (lo, hi)
                X[col] = X[col].clip(lo, hi)
            self.report.outlier_treated_columns = self._num_cols[:]
            self.report.steps_applied.append("Outlier clipping (3×IQR)")

        # Categorical encoding
        X = self._fit_encode_categoricals(X)

        # Scaling
        X = self._fit_scale(X)

        # Target encoding for classification
        y = self._fit_encode_target(y)

        self.report.final_shape = X.shape
        self.report.feature_names = X.columns.tolist()
        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform new data using fitted pipeline."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        df = df.copy()
        has_target = self.target_column in df.columns
        y = df[self.target_column].copy() if has_target else None
        X = df.drop(columns=[self.target_column], errors="ignore")

        # Drop same columns as training
        drop_cols = [c for c in self.report.dropped_columns if c in X.columns]
        X = X.drop(columns=drop_cols, errors="ignore")

        if self._num_cols:
            present_num = [c for c in self._num_cols if c in X.columns]
            X[present_num] = self._num_imputer.transform(X[present_num])  # type: ignore

        if self._cat_cols:
            present_cat = [c for c in self._cat_cols if c in X.columns]
            X[present_cat] = self._cat_imputer.transform(X[present_cat])  # type: ignore

        if self.outlier_treatment == "clip":
            for col, (lo, hi) in self._clip_bounds.items():
                if col in X.columns:
                    X[col] = X[col].clip(lo, hi)

        # Encode categoricals
        X = self._transform_encode_categoricals(X)

        # Scale
        X = self._transform_scale(X)

        if y is not None and self._target_encoder is not None:
            y = pd.Series(
                self._target_encoder.transform(y.astype(str)),
                index=y.index,
                name=self.target_column,
            )
        return X, y

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._cat_cols:
            return X

        n_unique_max = max((X[c].nunique() for c in self._cat_cols), default=0)

        if self.cat_encoding == "auto":
            encoding = "onehot" if n_unique_max <= 10 else "label"
        else:
            encoding = self.cat_encoding

        if encoding == "label":
            self._label_cols = self._cat_cols[:]
            for col in self._label_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._label_encoders[col] = le
            self.report.encoded_columns.extend(self._label_cols)
            self.report.steps_applied.append("Label encoding")
        else:
            self._onehot_cols = self._cat_cols[:]
            self._onehot_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", drop="first"
            )
            ohe_arr = self._onehot_encoder.fit_transform(X[self._onehot_cols])
            ohe_cols = self._onehot_encoder.get_feature_names_out(self._onehot_cols).tolist()
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)
            X = X.drop(columns=self._onehot_cols).join(ohe_df)
            self.report.encoded_columns.extend(self._onehot_cols)
            self.report.steps_applied.append("One-Hot encoding")
        return X

    def _transform_encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._label_cols:
            for col in self._label_cols:
                if col in X.columns:
                    le = self._label_encoders[col]
                    X[col] = X[col].astype(str).map(
                        lambda v, _le=le: (
                            _le.transform([v])[0]
                            if v in _le.classes_
                            else -1
                        )
                    )
        if self._onehot_cols and self._onehot_encoder:
            present = [c for c in self._onehot_cols if c in X.columns]
            if present:
                ohe_arr = self._onehot_encoder.transform(X[present])
                ohe_cols = self._onehot_encoder.get_feature_names_out(present).tolist()
                ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)
                X = X.drop(columns=present).join(ohe_df)
        return X

    def _fit_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols or self.scaling == "none":
            return X
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        self._scaler = scaler_map.get(self.scaling, RobustScaler())
        X[num_cols] = self._scaler.fit_transform(X[num_cols])
        self.report.scaled_columns = num_cols
        self.report.steps_applied.append(f"Feature scaling ({self.scaling})")
        return X

    def _transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            return X
        num_cols = [c for c in self.report.scaled_columns if c in X.columns]
        if num_cols:
            X[num_cols] = self._scaler.transform(X[num_cols])
        return X

    def _fit_encode_target(self, y: pd.Series) -> pd.Series:
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if not pd.api.types.is_numeric_dtype(y):
                self._target_encoder = LabelEncoder()
                y = pd.Series(
                    self._target_encoder.fit_transform(y.astype(str)),
                    index=y.index,
                    name=self.target_column,
                )
                self.report.steps_applied.append("Target label encoding")
        return y

    @property
    def target_classes(self) -> list | None:
        if self._target_encoder:
            return self._target_encoder.classes_.tolist()
        return None
