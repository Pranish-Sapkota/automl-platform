"""Data Profiling / EDA page."""
from __future__ import annotations

import streamlit as st

from src.components import page_header, section_header, info_box, metric_row, styled_dataframe
from src.services.data_service import DataService
from src.visualization import (
    missing_values_heatmap,
    correlation_heatmap,
    distribution_plot,
    target_distribution,
    outlier_boxplots,
)
from src.utils import get_logger

logger = get_logger(__name__)
_svc = DataService()


def render() -> None:
    page_header("🔍 Data Profiling", "Automated EDA, missing value analysis, and statistical summaries.")

    df = st.session_state.get("df")
    if df is None:
        info_box("Please upload a dataset first.", "warning")
        return

    target = st.session_state.get("target_column")

    # Auto-profile or use cached
    if "profile" not in st.session_state:
        with st.spinner("Profiling dataset…"):
            profile = _svc.profile(df, target)
            st.session_state["profile"] = profile
    else:
        profile = st.session_state["profile"]

    # ── Overview metrics ──
    metric_row([
        {"label": "Rows", "value": f"{profile.shape[0]:,}", "icon": "📋"},
        {"label": "Columns", "value": f"{profile.shape[1]:,}", "icon": "🏷️"},
        {"label": "Memory", "value": f"{profile.memory_usage_mb:.1f} MB", "icon": "💾"},
        {"label": "Missing %", "value": f"{profile.missing_report.get('pct_missing', 0):.1f}%", "icon": "❓"},
        {"label": "Duplicates", "value": f"{profile.duplicate_report.get('duplicate_rows', 0):,}", "icon": "🔁"},
        {"label": "Problem Type", "value": profile.problem_type.replace("_", " ").title() if profile.problem_type else "Unknown", "icon": "🎯"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations ──
    section_header("AI Recommendations", "", "💡")
    for rec in profile.recommendations:
        st.markdown(
            f'<div style="background:#1A1D2E;border-left:3px solid #6366F1;'
            f'border-radius:6px;padding:0.6rem 1rem;margin:0.3rem 0;'
            f'color:#E2E8F0;font-size:0.88rem">{rec}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs([
        "📋 Schema", "❓ Missing Values", "🔁 Duplicates",
        "📊 Distributions", "🔗 Correlations", "📦 Outliers", "🎯 Target Analysis"
    ])

    with tabs[0]:
        _schema_tab(df, profile)

    with tabs[1]:
        _missing_tab(df, profile)

    with tabs[2]:
        _duplicate_tab(profile)

    with tabs[3]:
        _distribution_tab(df, profile)

    with tabs[4]:
        _correlation_tab(df)

    with tabs[5]:
        _outlier_tab(df, profile)

    with tabs[6]:
        _target_tab(df, profile, target)


def _schema_tab(df, profile) -> None:
    import pandas as pd

    section_header("Dataset Schema", "Column types and basic info", "📋")

    schema_data = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = missing / len(df) * 100
        schema_data.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-Null": len(df) - missing,
            "Missing": missing,
            "Missing %": f"{pct:.1f}%",
            "Unique": df[col].nunique(),
            "Sample": str(df[col].dropna().iloc[0]) if df[col].dropna().any() else "—",
        })

    schema_df = pd.DataFrame(schema_data)
    styled_dataframe(schema_df, height=450, key="schema_table")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numeric Columns", len(profile.numeric_columns))
    with col2:
        st.metric("Categorical Columns", len(profile.categorical_columns))
    with col3:
        st.metric("Datetime Columns", len(profile.datetime_columns))


def _missing_tab(df, profile) -> None:
    section_header("Missing Value Analysis", "", "❓")
    mr = profile.missing_report

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total Missing Cells", f"{mr.get('total_missing', 0):,}")
        st.metric("Overall Missing %", f"{mr.get('pct_missing', 0):.2f}%")
        st.metric("Columns With Missing", len(mr.get("cols_with_missing", [])))

        if mr.get("cols_with_missing"):
            import pandas as pd
            mc_df = pd.DataFrame([
                {"Column": c, "Missing": v["count"], "Missing %": v["pct"]}
                for c, v in mr["by_column"].items()
            ]).sort_values("Missing", ascending=False)
            st.dataframe(mc_df, use_container_width=True)

    with col2:
        fig = missing_values_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)


def _duplicate_tab(profile) -> None:
    section_header("Duplicate Row Analysis", "", "🔁")
    dr = profile.duplicate_report

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duplicate Rows", dr.get("duplicate_rows", 0))
    with col2:
        st.metric("Duplicate %", f"{dr.get('pct_duplicates', 0):.2f}%")
    with col3:
        status = "🔴 Found" if dr.get("has_duplicates") else "✅ None"
        st.metric("Status", status)

    if dr.get("has_duplicates"):
        info_box("Duplicate rows will be removed during preprocessing.", "warning")
    else:
        info_box("No duplicate rows detected.", "success")


def _distribution_tab(df, profile) -> None:
    section_header("Feature Distributions", "Select a column to explore its distribution", "📊")

    all_cols = df.columns.tolist()
    col = st.selectbox("Select Column", all_cols, key="dist_col")

    if col:
        fig = distribution_plot(df[col], col)
        st.plotly_chart(fig, use_container_width=True)

        if col in profile.distribution_stats:
            stats = profile.distribution_stats[col]
            if "mean" in stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{stats['mean']:.4f}")
                c2.metric("Std Dev", f"{stats['std']:.4f}")
                c3.metric("Skewness", f"{stats['skewness']:.4f}")
                c4.metric("Kurtosis", f"{stats['kurtosis']:.4f}")


def _correlation_tab(df) -> None:
    section_header("Correlation Analysis", "Pearson correlation between numeric features", "🔗")

    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        info_box("Need at least 2 numeric columns for correlation analysis.", "warning")
        return

    fig = correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)

    # Strongest correlations
    import numpy as np, pandas as pd
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "Feature A": corr.columns[i],
                "Feature B": corr.columns[j],
                "Correlation": round(corr.iloc[i, j], 4),
                "Abs Correlation": round(abs(corr.iloc[i, j]), 4),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Abs Correlation", ascending=False).head(15)
    st.markdown("**Top Correlated Pairs**")
    st.dataframe(pairs_df.drop(columns=["Abs Correlation"]), use_container_width=True)


def _outlier_tab(df, profile) -> None:
    section_header("Outlier Analysis", "IQR and Z-score based detection", "📦")

    or_ = profile.outlier_report
    if not or_:
        info_box("No numeric columns found for outlier analysis.", "warning")
        return

    import pandas as pd
    outlier_df = pd.DataFrame([
        {
            "Column": col,
            "IQR Outliers": v["iqr_outliers"],
            "IQR %": f"{v['iqr_pct']:.1f}%",
            "Z-Score Outliers": v["z_outliers"],
            "Is Normal": "✅" if v["is_normal"] else "❌",
        }
        for col, v in or_.items()
    ]).sort_values("IQR Outliers", ascending=False)

    st.dataframe(outlier_df, use_container_width=True)

    num_cols = list(or_.keys())[:8]
    if num_cols:
        fig = outlier_boxplots(df, num_cols)
        st.plotly_chart(fig, use_container_width=True)


def _target_tab(df, profile, target) -> None:
    section_header("Target Variable Analysis", "", "🎯")

    if not target or target not in df.columns:
        info_box("No target column selected. Please set one on the Upload page.", "warning")
        return

    ta = profile.target_analysis
    pt = ta.get("problem_type", "unknown")

    st.info(f"**Problem Type:** {pt.replace('_', ' ').title()}")

    fig = target_distribution(df[target], pt)
    st.plotly_chart(fig, use_container_width=True)

    if "class_distribution" in ta:
        import pandas as pd
        cd = ta["class_distribution"]
        cd_df = pd.DataFrame(
            [{"Class": str(k), "Count": v, "Percentage": f"{v / sum(cd.values()) * 100:.1f}%"}
             for k, v in cd.items()]
        )
        st.dataframe(cd_df, use_container_width=True)

        if ta.get("is_imbalanced"):
            info_box(
                f"⚖️ Class imbalance detected (ratio: {ta['class_balance_ratio']:.2f}). "
                "Consider using class weights or SMOTE during training.",
                "warning",
            )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{ta.get('mean', 0):.4f}")
        c2.metric("Std Dev", f"{ta.get('std', 0):.4f}")
        c3.metric("Min", f"{ta.get('min', 0):.4f}")
        c4.metric("Max", f"{ta.get('max', 0):.4f}")
