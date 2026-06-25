"""Home / Dashboard page."""
from __future__ import annotations

import streamlit as st

from src.components import page_header, metric_row, section_header, info_box


def render() -> None:
    page_header(
        "🤖 AutoML Platform",
        "Upload a dataset → Get trained models, insights, and AI explanations in minutes.",
    )

    # ── Quick-start pipeline graphic ──
    st.markdown(
        """
        <div style="display:flex;gap:0.5rem;align-items:center;justify-content:center;
                    flex-wrap:wrap;margin-bottom:2rem">
        """ + "".join([
            f"""<div style="display:flex;align-items:center;gap:0.5rem">
                <div style="background:#1A1D2E;border:1px solid #2D3748;border-radius:10px;
                            padding:0.6rem 1rem;text-align:center;min-width:100px">
                    <div style="font-size:1.4rem">{icon}</div>
                    <div style="color:#94A3B8;font-size:0.72rem;margin-top:0.2rem">{label}</div>
                </div>
                {"<div style='color:#6366F1;font-size:1.2rem'>→</div>" if i < 5 else ""}
            </div>"""
            for i, (icon, label) in enumerate([
                ("📂", "Upload"), ("🔍", "Profile"), ("⚙️", "Preprocess"),
                ("🧪", "Train"), ("📊", "Explain"), ("💬", "Chat"),
            ])
        ]) + "</div>",
        unsafe_allow_html=True,
    )

    # ── Session metrics ──
    df = st.session_state.get("df")
    automl_result = st.session_state.get("automl_result")
    profile = st.session_state.get("profile")

    metrics = [
        {
            "label": "Dataset Rows",
            "value": f"{df.shape[0]:,}" if df is not None else "—",
            "icon": "📋",
        },
        {
            "label": "Features",
            "value": f"{df.shape[1]:,}" if df is not None else "—",
            "icon": "🏷️",
        },
        {
            "label": "Models Trained",
            "value": len(automl_result.leaderboard) if automl_result else "—",
            "icon": "🧠",
        },
        {
            "label": "Problem Type",
            "value": profile.problem_type.replace("_", " ").title() if profile else "—",
            "icon": "🎯",
        },
    ]
    metric_row(metrics)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature highlights ──
    section_header("Platform Features", "What this AutoML platform can do for you", "⚡")

    col1, col2, col3 = st.columns(3)
    features = [
        (col1, [
            ("📊", "Smart Data Profiling", "Automatic missing value, outlier, and correlation analysis"),
            ("🔧", "Auto Preprocessing", "Imputation, encoding, scaling, and outlier treatment"),
            ("🧬", "Feature Engineering", "Date features, interactions, and aggregations"),
        ]),
        (col2, [
            ("🤖", "AutoML Engine", "FLAML + XGBoost + LightGBM + CatBoost + more"),
            ("🏆", "Model Leaderboard", "Compare all models with full metrics"),
            ("📈", "Model Explainability", "SHAP global & local feature importance"),
        ]),
        (col3, [
            ("💬", "AI Chat Assistant", "Ask questions about your data in plain English"),
            ("💡", "Business Insights", "AI-powered recommendations from Mistral"),
            ("📦", "Model Export", "Download trained models as .pkl files"),
        ]),
    ]

    for col, feat_list in features:
        with col:
            for icon, title, desc in feat_list:
                st.markdown(
                    f"""
                    <div style="background:#1A1D2E;border:1px solid #2D3748;border-radius:10px;
                                padding:1rem;margin-bottom:0.7rem">
                        <div style="font-size:1.2rem">{icon}</div>
                        <div style="color:#E2E8F0;font-weight:600;font-size:0.9rem;
                                    margin-top:0.2rem">{title}</div>
                        <div style="color:#64748B;font-size:0.78rem;margin-top:0.2rem">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Getting started ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Getting Started", "Follow these steps to train your first model", "🚀")

    if df is None:
        info_box(
            "👉 Start by uploading a dataset using the <b>Upload Dataset</b> page in the sidebar.",
            "info",
        )
    elif automl_result is None:
        info_box(
            "✅ Dataset loaded! Head to <b>AutoML Training</b> to train your models.",
            "success",
        )
    else:
        info_box(
            f"🏆 Training complete! Best model: <b>{automl_result.best_model_name}</b>. "
            "Explore the Leaderboard and Explainability pages.",
            "success",
        )

    # ── Supported algorithms ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Supported Algorithms", "", "🧠")

    alg_cols = st.columns(7)
    algorithms = [
        ("🌲", "Random Forest"),
        ("🌳", "Extra Trees"),
        ("⚡", "XGBoost"),
        ("💡", "LightGBM"),
        ("🐱", "CatBoost"),
        ("📐", "Logistic Reg."),
        ("🌿", "Decision Tree"),
    ]
    for col, (icon, name) in zip(alg_cols, algorithms):
        with col:
            st.markdown(
                f"""<div style="text-align:center;background:#1A1D2E;border:1px solid #2D3748;
                               border-radius:8px;padding:0.6rem 0.3rem">
                    <div style="font-size:1.2rem">{icon}</div>
                    <div style="color:#94A3B8;font-size:0.65rem;margin-top:0.2rem">{name}</div>
                </div>""",
                unsafe_allow_html=True,
            )
