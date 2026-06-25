"""Data Cleaning & Preprocessing page."""
from __future__ import annotations

import streamlit as st

from src.components import page_header, section_header, info_box, metric_row
from src.preprocessing import AutoPreprocessor
from src.feature_engineering import AutoFeatureEngineer
from src.utils import get_logger

logger = get_logger(__name__)


def render() -> None:
    page_header("⚙️ Data Cleaning & Preprocessing", "Configure and apply the preprocessing pipeline.")

    df = st.session_state.get("df")
    if df is None:
        info_box("Please upload a dataset first.", "warning")
        return

    target = st.session_state.get("target_column")
    if not target:
        info_box("Please select a target column on the Upload page.", "warning")
        return

    profile = st.session_state.get("profile")

    # ── Configuration ──
    section_header("Preprocessing Configuration", "", "⚙️")

    col1, col2, col3 = st.columns(3)

    with col1:
        scaling = st.selectbox(
            "Feature Scaling",
            ["robust", "standard", "minmax", "none"],
            help="RobustScaler handles outliers well",
        )
        missing_num = st.selectbox(
            "Numeric Imputation",
            ["median", "mean", "most_frequent"],
        )

    with col2:
        encoding = st.selectbox(
            "Categorical Encoding",
            ["auto", "onehot", "label"],
            help="Auto chooses based on cardinality",
        )
        missing_cat = st.selectbox(
            "Categorical Imputation",
            ["most_frequent", "constant"],
        )

    with col3:
        outlier_treatment = st.selectbox(
            "Outlier Treatment",
            ["clip", "none"],
            help="Clip caps outliers at 3×IQR bounds",
        )
        drop_cardinality = st.slider(
            "Max Cardinality (OHE)",
            min_value=10, max_value=200, value=50,
        )

    # ── Feature Engineering Config ──
    section_header("Feature Engineering", "", "🧬")

    fe_col1, fe_col2, fe_col3 = st.columns(3)
    with fe_col1:
        do_date_features = st.checkbox("Extract Date Features", value=True)
    with fe_col2:
        do_interactions = st.checkbox("Create Interaction Features", value=True)
    with fe_col3:
        do_aggregations = st.checkbox("Add Aggregation Features", value=True)

    # ── ID columns ──
    id_cols_default = profile.id_columns if profile else []
    all_feature_cols = [c for c in df.columns if c != target]
    id_columns = st.multiselect(
        "ID / Irrelevant Columns to Drop",
        options=all_feature_cols,
        default=[c for c in id_cols_default if c in all_feature_cols],
        help="These columns will be excluded from model training",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Apply Pipeline ──
    if st.button("▶️ Apply Preprocessing Pipeline", type="primary"):
        _apply_pipeline(
            df=df,
            target=target,
            scaling=scaling,
            encoding=encoding,
            outlier_treatment=outlier_treatment,
            missing_num=missing_num,
            missing_cat=missing_cat,
            drop_cardinality=drop_cardinality,
            id_columns=id_columns,
            do_date_features=do_date_features,
            do_interactions=do_interactions,
            do_aggregations=do_aggregations,
            problem_type=st.session_state.get("problem_type", "binary_classification"),
        )

    # ── Show results ──
    preproc_result = st.session_state.get("preproc_result")
    if preproc_result:
        _show_results(preproc_result)


def _apply_pipeline(df, target, scaling, encoding, outlier_treatment,
                    missing_num, missing_cat, drop_cardinality, id_columns,
                    do_date_features, do_interactions, do_aggregations, problem_type) -> None:
    with st.spinner("Applying preprocessing pipeline…"):
        try:
            # Feature engineering first (on raw df, excluding target)
            fe = AutoFeatureEngineer(
                create_date_features=do_date_features,
                create_interactions=do_interactions,
                create_aggregations=do_aggregations,
                target_column=target,
            )
            df_fe = fe.fit_transform(df)

            # Now preprocess
            preprocessor = AutoPreprocessor(
                target_column=target,
                problem_type=problem_type,
                scaling=scaling,
                cat_encoding=encoding,
                outlier_treatment=outlier_treatment,
                missing_strategy_num=missing_num,
                missing_strategy_cat=missing_cat,
                drop_high_cardinality=drop_cardinality,
                id_columns=id_columns,
            )
            X, y = preprocessor.fit_transform(df_fe)

            # Store in session
            st.session_state["preprocessor"] = preprocessor
            st.session_state["feature_engineer"] = fe
            st.session_state["X_processed"] = X
            st.session_state["y_processed"] = y
            st.session_state["preproc_result"] = {
                "prep_report": preprocessor.report,
                "fe_report": fe.report,
                "X_shape": X.shape,
                "y_shape": y.shape,
                "target_classes": preprocessor.target_classes,
            }

            # Reset downstream
            for key in ["automl_result", "shap_result"]:
                st.session_state.pop(key, None)

            st.success("✅ Preprocessing complete!")
            st.rerun()

        except Exception as exc:
            info_box(f"Preprocessing failed: {exc}", "error")
            logger.error("Preprocessing error: %s", exc, exc_info=True)


def _show_results(result: dict) -> None:
    import pandas as pd

    prep_report = result["prep_report"]
    fe_report = result["fe_report"]

    section_header("Pipeline Results", "", "📊")

    metric_row([
        {"label": "Original Features", "value": fe_report.original_feature_count, "icon": "🔢"},
        {"label": "After FE", "value": fe_report.final_feature_count, "icon": "🧬"},
        {"label": "Final Features", "value": result["X_shape"][1], "icon": "✅"},
        {"label": "Training Samples", "value": f"{result['X_shape'][0]:,}", "icon": "📋"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("Preprocessing Steps", "", "⚙️")
        for step in prep_report.steps_applied:
            st.markdown(
                f'<div style="background:#1A1D2E;border-left:3px solid #10B981;'
                f'border-radius:6px;padding:0.4rem 0.8rem;margin:0.25rem 0;'
                f'color:#E2E8F0;font-size:0.85rem">✓ {step}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        section_header("Feature Engineering", "", "🧬")
        if fe_report.date_features:
            st.markdown(f"**Date Features ({len(fe_report.date_features)}):** `{', '.join(fe_report.date_features[:5])}`")
        if fe_report.interaction_features:
            st.markdown(f"**Interaction Features ({len(fe_report.interaction_features)}):** `{', '.join(fe_report.interaction_features[:5])}`")
        if fe_report.aggregation_features:
            st.markdown(f"**Aggregation Features ({len(fe_report.aggregation_features)}):** `{', '.join(fe_report.aggregation_features)}`")
        if not (fe_report.date_features or fe_report.interaction_features or fe_report.aggregation_features):
            st.markdown("No new features generated.")

    # Dropped columns
    if prep_report.dropped_columns:
        st.markdown("<br>", unsafe_allow_html=True)
        info_box(f"Dropped columns: {', '.join(prep_report.dropped_columns)}", "info")

    if result.get("target_classes"):
        st.info(f"**Target Classes:** {result['target_classes']}")
