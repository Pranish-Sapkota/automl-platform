"""Automated feature engineering: date features, interactions, aggregations."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringReport:
    new_features: list[str] = field(default_factory=list)
    date_features: list[str] = field(default_factory=list)
    interaction_features: list[str] = field(default_factory=list)
    aggregation_features: list[str] = field(default_factory=list)
    original_feature_count: int = 0
    final_feature_count: int = 0


class AutoFeatureEngineer:
    """Generate new features automatically from existing columns."""

    def __init__(
        self,
        max_interaction_features: int = 20,
        create_date_features: bool = True,
        create_interactions: bool = True,
        create_aggregations: bool = True,
        target_column: str | None = None,
    ) -> None:
        self.max_interaction_features = max_interaction_features
        self.create_date_features = create_date_features
        self.create_interactions = create_interactions
        self.create_aggregations = create_aggregations
        self.target_column = target_column
        self.report = FeatureEngineeringReport()
        self._date_cols: list[str] = []
        self._interaction_pairs: list[tuple[str, str]] = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.report.original_feature_count = df.shape[1]
        self._original_cols = set(df.columns.tolist())

        # Date / datetime features
        if self.create_date_features:
            df = self._extract_date_features(df)

        # Numeric interactions
        if self.create_interactions:
            df = self._create_interaction_features(df)

        # Statistical aggregations per row
        if self.create_aggregations:
            df = self._create_aggregation_features(df)

        new = [c for c in df.columns if c not in self._original_cols]
        self.report.new_features = new
        self.report.final_feature_count = df.shape[1]
        self._fitted = True
        logger.info("Feature engineering: %d → %d features", self.report.original_feature_count, df.shape[1])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.create_date_features:
            df = self._extract_date_features(df, fit=False)
        if self.create_interactions:
            df = self._apply_interaction_features(df)
        if self.create_aggregations:
            df = self._create_aggregation_features(df)
        return df

    # ------------------------------------------------------------------

    def _extract_date_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit:
            self._date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            # Try to parse object columns as dates
            for col in df.select_dtypes(include=["object"]).columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                        self._date_cols.append(col)
                    except Exception:
                        pass
            self._original_cols = set(df.columns.tolist())

        for col in self._date_cols:
            if col not in df.columns:
                continue
            prefix = col
            df[f"{prefix}_year"] = df[col].dt.year
            df[f"{prefix}_month"] = df[col].dt.month
            df[f"{prefix}_day"] = df[col].dt.day
            df[f"{prefix}_dayofweek"] = df[col].dt.dayofweek
            df[f"{prefix}_quarter"] = df[col].dt.quarter
            df[f"{prefix}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
            new = [
                f"{prefix}_year", f"{prefix}_month", f"{prefix}_day",
                f"{prefix}_dayofweek", f"{prefix}_quarter", f"{prefix}_is_weekend"
            ]
            self.report.date_features.extend(new)
            df = df.drop(columns=[col])
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_original_cols"):
            self._original_cols = set(df.columns.tolist())
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column and self.target_column in num_cols:
            num_cols.remove(self.target_column)

        pairs = list(combinations(num_cols[:10], 2))  # cap at 10 cols
        self._interaction_pairs = pairs[: self.max_interaction_features // 2]

        for col_a, col_b in self._interaction_pairs:
            feat_name = f"{col_a}_x_{col_b}"
            df[feat_name] = df[col_a] * df[col_b]
            self.report.interaction_features.append(feat_name)

            ratio_name = f"{col_a}_div_{col_b}"
            df[ratio_name] = df[col_a] / (df[col_b].replace(0, np.nan))
            df[ratio_name] = df[ratio_name].fillna(0)
            self.report.interaction_features.append(ratio_name)

        return df

    def _apply_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col_a, col_b in self._interaction_pairs:
            if col_a in df.columns and col_b in df.columns:
                df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
                df[f"{col_a}_div_{col_b}"] = df[col_a] / (df[col_b].replace(0, np.nan))
                df[f"{col_a}_div_{col_b}"] = df[f"{col_a}_div_{col_b}"].fillna(0)
        return df

    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column and self.target_column in num_cols:
            num_cols.remove(self.target_column)
        if len(num_cols) < 2:
            return df

        df["_agg_row_mean"] = df[num_cols].mean(axis=1)
        df["_agg_row_std"] = df[num_cols].std(axis=1).fillna(0)
        df["_agg_row_min"] = df[num_cols].min(axis=1)
        df["_agg_row_max"] = df[num_cols].max(axis=1)
        df["_agg_row_range"] = df["_agg_row_max"] - df["_agg_row_min"]
        df["_agg_row_sum"] = df[num_cols].sum(axis=1)

        new_agg = ["_agg_row_mean", "_agg_row_std", "_agg_row_min",
                   "_agg_row_max", "_agg_row_range", "_agg_row_sum"]
        self.report.aggregation_features.extend(new_agg)
        return df
