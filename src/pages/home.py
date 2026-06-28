"""Home / Dashboard page — theme-aware, mobile-first."""
from __future__ import annotations

import streamlit as st

from src.components.ui import page_header, section_header, metric_row, info_box, feature_card, _t


def render() -> None:
    T = _t()

    page_header(
        "🤖 AutoML Platform",
        "Upload a dataset → get trained models, SHAP insights & AI explanations in minutes.",
    )

    # ── Pipeline flow graphic ──────────────────────────────────────────
    steps = [
        ("📂", "Upload"),
        ("🔍", "Profile"),
        ("⚙️", "Clean"),
        ("🧪", "Train"),
        ("📊", "Explain"),
        ("💬", "Chat"),
    ]
    step_html = ""
    for i, (icon, label) in enumerate(steps):
        arrow = (
            f'<div style="color:{T["primary"]};font-size:1rem;'
            f'padding:0 0.3rem;align-self:center">›</div>'
            if i < len(steps) - 1 else ""
        )
        step_html += f"""
        <div style="display:flex;align-items:center">
            <div style="background:{T["surface"]};border:1px solid {T["border"]};
                        border-radius:10px;padding:0.55rem 0.9rem;
                        text-align:center;min-width:72px">
                <div style="font-size:1.3rem">{icon}</div>
                <div style="color:{T["muted"]};font-size:0.67rem;
                            margin-top:0.15rem;font-weight:600">{label}</div>
            </div>
            {arrow}
        </div>"""

    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:0.3rem;align-items:center;
                    justify-content:center;margin-bottom:1.6rem;
                    padding:1rem;background:{T["surface2"]};
                    border:1px solid {T["border"]};border-radius:14px">
            {step_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session metrics ────────────────────────────────────────────────
    df      = st.session_state.get("df")
    result  = st.session_state.get("automl_result")
    profile = st.session_state.get("profile")

    metric_row([
        {"label": "Dataset Rows",   "value": f"{df.shape[0]:,}" if df is not None else "—", "icon": "📋"},
        {"label": "Features",       "value": f"{df.shape[1]:,}" if df is not None else "—", "icon": "🏷️"},
        {"label": "Models Trained", "value": len(result.leaderboard) if result else "—",     "icon": "🧠"},
        {"label": "Problem Type",   "value": (
            profile.problem_type.replace("_", " ").title()
            if profile and profile.problem_type else "—"
        ), "icon": "🎯"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Features grid ──────────────────────────────────────────────────
    section_header("Platform Features", "Everything you need in one place", "⚡")

    col1, col2, col3 = st.columns(3)

    with col1:
        feature_card("📊", "Smart Data Profiling",   "Missing values, outliers, correlations & class imbalance")
        feature_card("🔧", "Auto Preprocessing",     "Imputation, encoding, scaling & outlier treatment")
        feature_card("🧬", "Feature Engineering",    "Date features, interactions & row aggregations")

    with col2:
        feature_card("🤖", "AutoML Engine",          "FLAML + XGBoost + LightGBM + CatBoost + more")
        feature_card("🏆", "Model Leaderboard",      "Compare all models with full metric breakdowns")
        feature_card("📈", "SHAP Explainability",    "Global & local feature importance with waterfall plots")

    with col3:
        feature_card("💬", "AI Chat Assistant",      "Ask questions about your data in plain English")
        feature_card("💡", "Business Insights",      "AI-powered recommendations powered by Mistral")
        feature_card("📦", "Model Export",           "Download .pkl models + batch predictions as CSV")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status CTA ────────────────────────────────────────────────────
    section_header("Getting Started", "", "🚀")

    if df is None:
        info_box(
            "👉 Start by uploading a dataset on the <b>Upload Dataset</b> page.",
            "info",
        )
    elif result is None:
        info_box(
            "✅ Dataset loaded! Go to <b>Data Cleaning</b> then <b>AutoML Training</b>.",
            "success",
        )
    else:
        info_box(
            f"🏆 Training complete — best model: <b>{result.best_model_name}</b>. "
            "Explore the <b>Leaderboard</b> and <b>Explainability</b> pages.",
            "success",
        )

    # ── Algorithm badges ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Supported Algorithms", "", "🧠")

    algos = [
        ("🌲", "Random Forest"),
        ("🌳", "Extra Trees"),
        ("⚡", "XGBoost"),
        ("💡", "LightGBM"),
        ("🐱", "CatBoost"),
        ("📐", "Logistic Reg"),
        ("🌿", "Decision Tree"),
        ("🔬", "FLAML Auto"),
    ]
    cols = st.columns(len(algos))
    for col, (icon, name) in zip(cols, algos):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;background:{T["surface"]};
                            border:1px solid {T["border"]};border-radius:10px;
                            padding:0.65rem 0.3rem;cursor:default;
                            transition:border-color 0.18s ease"
                     onmouseover="this.style.borderColor='{T["primary"]}'"
                     onmouseout="this.style.borderColor='{T["border"]}'">
                    <div style="font-size:1.2rem">{icon}</div>
                    <div style="color:{T["muted"]};font-size:0.62rem;
                                margin-top:0.2rem;font-weight:600">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
