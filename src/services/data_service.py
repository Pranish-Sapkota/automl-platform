"""Dataset ingestion, validation, and profiling service."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import CONFIG, get_logger, infer_problem_type, detect_id_columns

logger = get_logger(__name__)


@dataclass
class DataProfile:
    """Complete data profile report."""
    shape: tuple[int, int] = (0, 0)
    memory_usage_mb: float = 0.0
    dtypes: dict[str, str] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    missing_report: dict[str, Any] = field(default_factory=dict)
    duplicate_report: dict[str, Any] = field(default_factory=dict)
    outlier_report: dict[str, Any] = field(default_factory=dict)
    correlation_matrix: dict[str, Any] = field(default_factory=dict)
    distribution_stats: dict[str, Any] = field(default_factory=dict)
    target_analysis: dict[str, Any] = field(default_factory=dict)
    id_columns: list[str] = field(default_factory=list)
    problem_type: str = ""
    recommendations: list[str] = field(default_factory=list)


class DataService:
    """Handles dataset loading, validation, and full profiling."""

    MAX_ROWS = CONFIG.max_rows
    MAX_COLS = CONFIG.max_columns

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dataframe(self, uploaded_file: Any) -> pd.DataFrame:
        """Load a DataFrame from a Streamlit uploaded file object."""
        fname = uploaded_file.name.lower()
        raw = uploaded_file.read()
        buf = io.BytesIO(raw)

        if fname.endswith(".csv"):
            df = pd.read_csv(buf)
        elif fname.endswith((".xls", ".xlsx")):
            df = pd.read_excel(buf)
        elif fname.endswith(".parquet"):
            df = pd.read_parquet(buf)
        elif fname.endswith(".json"):
            df = pd.read_json(buf)
        else:
            raise ValueError(f"Unsupported file format: {fname}")

        if df.shape[0] > self.MAX_ROWS:
            logger.warning("Dataset truncated to %d rows", self.MAX_ROWS)
            df = df.iloc[: self.MAX_ROWS]
        if df.shape[1] > self.MAX_COLS:
            df = df.iloc[:, : self.MAX_COLS]

        logger.info("Loaded dataset: %s rows × %s cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame, target_column: str | None = None) -> DataProfile:
        """Generate a complete DataProfile for the given DataFrame."""
        p = DataProfile()
        p.shape = df.shape
        p.memory_usage_mb = df.memory_usage(deep=True).sum() / 1e6
        p.dtypes = {col: str(df[col].dtype) for col in df.columns}

        p.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        p.categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        p.datetime_columns = df.select_dtypes(include=["datetime64"]).columns.tolist()
        p.id_columns = detect_id_columns(df)

        p.missing_report = self._missing_report(df)
        p.duplicate_report = self._duplicate_report(df)
        p.outlier_report = self._outlier_report(df)
        p.correlation_matrix = self._correlation(df)
        p.distribution_stats = self._distribution_stats(df)

        if target_column and target_column in df.columns:
            p.target_analysis = self._target_analysis(df[target_column])
            p.problem_type = infer_problem_type(df[target_column])

        p.recommendations = self._generate_recommendations(p, df)
        return p

    # ------------------------------------------------------------------
    # Sub-analyses
    # ------------------------------------------------------------------

    def _missing_report(self, df: pd.DataFrame) -> dict:
        total = len(df)
        missing = df.isnull().sum()
        pct = (missing / total * 100).round(2)
        cols_with_missing = missing[missing > 0].index.tolist()
        return {
            "total_missing": int(missing.sum()),
            "pct_missing": float((missing.sum() / (total * df.shape[1]) * 100).round(2)),
            "by_column": {
                col: {"count": int(missing[col]), "pct": float(pct[col])}
                for col in cols_with_missing
            },
            "has_missing": len(cols_with_missing) > 0,
            "cols_with_missing": cols_with_missing,
        }

    def _duplicate_report(self, df: pd.DataFrame) -> dict:
        n_dup = int(df.duplicated().sum())
        return {
            "duplicate_rows": n_dup,
            "pct_duplicates": round(n_dup / len(df) * 100, 2),
            "has_duplicates": n_dup > 0,
        }

    def _outlier_report(self, df: pd.DataFrame) -> dict:
        report: dict[str, dict] = {}
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            s = numeric[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers_iqr = int(((s < lower) | (s > upper)).sum())

            _, p_value = stats.shapiro(s.sample(min(len(s), 500), random_state=42))
            z_scores = np.abs(stats.zscore(s))
            outliers_z = int((z_scores > 3).sum())

            report[col] = {
                "iqr_outliers": outliers_iqr,
                "iqr_pct": round(outliers_iqr / len(s) * 100, 2),
                "z_outliers": outliers_z,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "shapiro_p": float(p_value),
                "is_normal": p_value > 0.05,
            }
        return report

    def _correlation(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return {}
        corr = numeric.corr()
        return corr.to_dict()

    def _distribution_stats(self, df: pd.DataFrame) -> dict:
        stats_dict: dict[str, dict] = {}
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            s = numeric[col].dropna()
            stats_dict[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
                "skewness": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
            }
        for col in df.select_dtypes(include=["object", "category"]).columns:
            vc = df[col].value_counts()
            stats_dict[col] = {
                "unique": int(df[col].nunique()),
                "top": str(vc.index[0]) if len(vc) > 0 else "",
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_pct": round(vc.iloc[0] / len(df) * 100, 2) if len(vc) > 0 else 0.0,
            }
        return stats_dict

    def _target_analysis(self, target: pd.Series) -> dict:
        problem = infer_problem_type(target)
        result: dict[str, Any] = {"problem_type": problem, "n_unique": int(target.nunique())}
        if problem in ("binary_classification", "multiclass_classification"):
            vc = target.value_counts()
            result["class_distribution"] = vc.to_dict()
            result["class_balance_ratio"] = float(vc.min() / vc.max())
            result["is_imbalanced"] = result["class_balance_ratio"] < 0.2
        else:
            result["mean"] = float(target.mean())
            result["std"] = float(target.std())
            result["min"] = float(target.min())
            result["max"] = float(target.max())
        return result

    def _generate_recommendations(self, p: DataProfile, df: pd.DataFrame) -> list[str]:
        recs: list[str] = []
        if p.missing_report.get("pct_missing", 0) > 30:
            recs.append("⚠️ High missing rate (>30%). Consider removing low-quality columns or using advanced imputation.")
        elif p.missing_report.get("has_missing"):
            recs.append("📋 Missing values detected. Automatic imputation will be applied during preprocessing.")
        if p.duplicate_report.get("has_duplicates"):
            recs.append(f"🔁 {p.duplicate_report['duplicate_rows']} duplicate rows detected. Consider deduplication.")
        outlier_cols = [c for c, v in p.outlier_report.items() if v.get("iqr_pct", 0) > 5]
        if outlier_cols:
            recs.append(f"📊 Significant outliers in: {', '.join(outlier_cols[:5])}. Capping or removal recommended.")
        ta = p.target_analysis
        if ta.get("is_imbalanced"):
            recs.append("⚖️ Target class imbalance detected. SMOTE or class weighting will be applied.")
        if len(p.id_columns) > 0:
            recs.append(f"🔑 Potential ID columns detected: {', '.join(p.id_columns)}. These will be excluded from training.")
        high_cardinality = [
            c for c in p.categorical_columns
            if df[c].nunique() > 50
        ]
        if high_cardinality:
            recs.append(f"🏷️ High-cardinality categoricals: {', '.join(high_cardinality[:3])}. Target encoding recommended.")
        if not recs:
            recs.append("✅ Dataset looks clean and ready for AutoML training.")
        return recs
