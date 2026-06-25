"""Model Explainability page (SHAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.components import page_header, section_header, info_box, metric_row
from src.explainability import SHAPExplainer
from src.visualization import (
    shap_importance_bar,
    shap_waterfall,
    feature_importance_scatter,
)
from src.utils import get_logger

logger = get_logger(__name__)


def render() -> None:
    page_header("🔮 Model Explainability", "SHAP-based global and local feature importance.")

    result = st.session_state.get("automl_result")
    if result is None:
        info_box("Please train models first.", "warning")
        return

    X_test = st.session_state.get("X_test")
    if X_test is None:
        info_box("Test dataset not found in session.", "warning")
        return

    # ── Model selection ──
    model_names = [r.name for r in result.leaderboard]
    sel_model = st.selectbox("Select Model to Explain", model_names)
    selected = next(r for r in result.leaderboard if r.name == sel_model)

    # ── Run SHAP (with caching per model) ──
    cache_key = f"shap_{sel_model}"
    if cache_key not in st.session_state:
        with st.spinner(f"Computing SHAP values for {sel_model}… (this may take a minute)"):
            try:
                explainer = SHAPExplainer(
                    model=selected.model,
                    feature_names=result.feature_names,
                )
                explainer.build_explainer(X_test)
                shap_result = explainer.explain(X_test, max_samples=200)
                imp_df = explainer.get_sorted_importance(shap_result, top_n=25)
                st.session_state[cache_key] = {
                    "explainer": explainer,
                    "shap_result": shap_result,
                    "imp_df": imp_df,
                }
            except Exception as exc:
                info_box(f"SHAP computation failed: {exc}", "error")
                logger.error("SHAP error: %s", exc, exc_info=True)
                return
    else:
        info_box("Using cached SHAP values.", "info")

    cached = st.session_state.get(cache_key)
    if not cached:
        return

    explainer: SHAPExplainer = cached["explainer"]
    shap_result = cached["shap_result"]
    imp_df: pd.DataFrame = cached["imp_df"]

    # ── Top features ──
    if not imp_df.empty:
        top5 = imp_df.head(5)["Feature"].tolist()
        metric_row([
            {"label": "Total Features", "value": len(result.feature_names), "icon": "🏷️"},
            {"label": "Top Feature", "value": imp_df.iloc[0]["Feature"] if not imp_df.empty else "—", "icon": "⭐"},
            {"label": "Top Importance", "value": f"{imp_df.iloc[0]['Importance']:.4f}" if not imp_df.empty else "—", "icon": "📊"},
            {"label": "Samples Explained", "value": min(200, len(X_test)), "icon": "🔢"},
        ])

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "🌍 Global Importance", "💧 Waterfall Plot", "🫧 Bubble Chart",
        "🔍 Local Explanation", "📋 Raw Values"
    ])

    with tabs[0]:
        _global_tab(imp_df)

    with tabs[1]:
        _waterfall_tab(shap_result, result.feature_names)

    with tabs[2]:
        _bubble_tab(imp_df)

    with tabs[3]:
        _local_tab(explainer, X_test, result.feature_names)

    with tabs[4]:
        _raw_tab(shap_result, result.feature_names)


def _global_tab(imp_df: pd.DataFrame) -> None:
    section_header("Global Feature Importance", "Average |SHAP value| across all predictions", "🌍")

    if imp_df.empty:
        info_box("No importance data available.", "warning")
        return

    fig = shap_importance_bar(imp_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Feature Importance Table**")
    imp_df_display = imp_df.copy()
    imp_df_display.index = range(1, len(imp_df_display) + 1)
    st.dataframe(imp_df_display, use_container_width=True)


def _waterfall_tab(shap_result, feature_names: list[str]) -> None:
    section_header("SHAP Waterfall Plot", "How each feature pushes the prediction from the base value", "💧")

    if shap_result.shap_values is None:
        info_box("Waterfall data not available.", "warning")
        return

    X_test = st.session_state.get("X_test")
    max_idx = len(X_test) - 1 if X_test is not None else 10
    sample_idx = st.slider("Sample Index", 0, min(max_idx, 199), 0)

    try:
        vals = shap_result.shap_values
        base = shap_result.expected_value or 0.0

        fig = shap_waterfall(vals, feature_names, float(base), sample_idx)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        info_box(f"Waterfall plot failed: {exc}", "error")


def _bubble_tab(imp_df: pd.DataFrame) -> None:
    section_header("Feature Importance Bubbles", "Size and color represent importance magnitude", "🫧")

    if imp_df.empty:
        info_box("No data available.", "warning")
        return

    fig = feature_importance_scatter(imp_df)
    st.plotly_chart(fig, use_container_width=True)


def _local_tab(explainer: SHAPExplainer, X_test: pd.DataFrame, feature_names: list[str]) -> None:
    section_header("Local Explanation", "Explain a single prediction", "🔍")

    idx = st.number_input("Sample Index", min_value=0, max_value=len(X_test) - 1, value=0)
    if st.button("Explain This Prediction", type="primary"):
        with st.spinner("Computing local SHAP values…"):
            contrib = explainer.explain_single(X_test, int(idx))

        if contrib:
            contrib_df = pd.DataFrame(
                sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True),
                columns=["Feature", "SHAP Value"],
            ).head(15)
            contrib_df["Direction"] = contrib_df["SHAP Value"].apply(
                lambda v: "🔴 Increases prediction" if v > 0 else "🔵 Decreases prediction"
            )

            st.dataframe(contrib_df, use_container_width=True)

            # Mini bar chart
            import plotly.graph_objects as go

            colors = [
                "#EF4444" if v > 0 else "#3B82F6"
                for v in contrib_df["SHAP Value"]
            ]
            fig = go.Figure(go.Bar(
                x=contrib_df["SHAP Value"],
                y=contrib_df["Feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                paper_bgcolor="#0F1117", plot_bgcolor="#1A1D2E",
                font=dict(color="#E2E8F0"),
                title="Local Feature Contributions",
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            info_box("Could not compute local explanation.", "error")


def _raw_tab(shap_result, feature_names: list[str]) -> None:
    section_header("Raw SHAP Values", "First 20 samples × top features", "📋")

    if shap_result.shap_values is None:
        info_box("No SHAP values available.", "warning")
        return

    vals = shap_result.shap_values
    if hasattr(vals, "shape") and len(vals.shape) == 2:
        n_show = min(20, vals.shape[0])
        raw_df = pd.DataFrame(
            vals[:n_show],
            columns=feature_names[: vals.shape[1]],
        )
        # Show only top-10 columns by mean abs SHAP
        top_cols = (
            raw_df.abs().mean().sort_values(ascending=False).head(10).index.tolist()
        )
        st.dataframe(raw_df[top_cols].round(4), use_container_width=True)
    else:
        info_box("SHAP values format not supported for display.", "info")
