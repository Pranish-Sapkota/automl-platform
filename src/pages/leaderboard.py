"""Model Leaderboard page."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.components import page_header, section_header, info_box, metric_row
from src.visualization import (
    leaderboard_chart,
    metrics_radar,
    roc_curve_chart,
    confusion_matrix_chart,
    training_time_chart,
)
from src.utils import get_logger

logger = get_logger(__name__)


def render() -> None:
    page_header("🏆 Model Leaderboard", "Compare all trained models and explore their metrics.")

    result = st.session_state.get("automl_result")
    if result is None:
        info_box("Please run AutoML Training first.", "warning")
        return

    is_clf = "classification" in result.problem_type

    # ── Build leaderboard DataFrame ──
    records = []
    for r in result.leaderboard:
        row: dict = {"Model": r.name, "Algorithm": r.algorithm, "Train Time (s)": round(r.train_time, 2)}
        row.update({k.upper(): round(v, 4) for k, v in r.metrics.items()})
        row["🏅"] = "🥇" if r.name == result.best_model_name else ""
        records.append(row)

    lb_df = pd.DataFrame(records)

    # ── Top metrics ──
    best = result.leaderboard[0]
    if is_clf:
        metric_row([
            {"label": "Best Model", "value": best.name, "icon": "🏆"},
            {"label": "Accuracy", "value": f"{best.metrics.get('accuracy', 0):.4f}", "icon": "🎯"},
            {"label": "F1 Score", "value": f"{best.metrics.get('f1', 0):.4f}", "icon": "📊"},
            {"label": "ROC-AUC", "value": f"{best.metrics.get('roc_auc', 0):.4f}", "icon": "📈"},
            {"label": "Models Trained", "value": len(result.leaderboard), "icon": "🧠"},
        ])
    else:
        metric_row([
            {"label": "Best Model", "value": best.name, "icon": "🏆"},
            {"label": "R²", "value": f"{best.metrics.get('r2', 0):.4f}", "icon": "📈"},
            {"label": "RMSE", "value": f"{best.metrics.get('rmse', 0):.4f}", "icon": "📉"},
            {"label": "MAE", "value": f"{best.metrics.get('mae', 0):.4f}", "icon": "📊"},
            {"label": "Models Trained", "value": len(result.leaderboard), "icon": "🧠"},
        ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full table ──
    section_header("Full Leaderboard", "Sorted by primary metric", "📋")

    def _highlight_best(row):
        if row.get("🏅") == "🥇":
            return ["background-color: rgba(99,102,241,0.15)"] * len(row)
        return [""] * len(row)

    st.dataframe(lb_df.style.apply(_highlight_best, axis=1), use_container_width=True, height=350)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart tabs ──
    tabs = st.tabs(["📊 Metric Comparison", "⏱️ Training Time", "🕸️ Radar Chart",
                    "📈 ROC Curve", "🔷 Confusion Matrix", "⚙️ Model Details"])

    with tabs[0]:
        primary_metric = "F1" if is_clf else "R2"
        available = [c for c in lb_df.columns if c not in ["Model", "Algorithm", "Train Time (s)", "🏅"]]
        sel_metric = st.selectbox("Select Metric", available,
                                   index=available.index(primary_metric) if primary_metric in available else 0)
        plot_df = lb_df[["Model", sel_metric]].rename(columns={sel_metric: sel_metric})
        plot_df.columns = ["Model", sel_metric]
        fig = leaderboard_chart(plot_df, sel_metric)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig = training_time_chart(lb_df)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        model_names = lb_df["Model"].tolist()
        sel_model = st.selectbox("Model for Radar", model_names, key="radar_model")
        selected = next(r for r in result.leaderboard if r.name == sel_model)
        # Normalize metrics to 0-1 range for radar
        metrics_for_radar = {
            k: min(max(float(v), 0.0), 1.0)
            for k, v in selected.metrics.items()
            if k not in ("rmse", "mae")  # exclude unbounded metrics
        }
        if metrics_for_radar:
            fig = metrics_radar(metrics_for_radar, sel_model)
            st.plotly_chart(fig, use_container_width=True)
        else:
            info_box("Radar chart not available for regression metrics.", "info")

    with tabs[3]:
        if is_clf:
            _roc_tab(result)
        else:
            info_box("ROC Curve is only available for classification tasks.", "info")

    with tabs[4]:
        if is_clf:
            _cm_tab(result)
        else:
            info_box("Confusion Matrix is only available for classification tasks.", "info")

    with tabs[5]:
        _details_tab(result)


def _roc_tab(result) -> None:
    y_test = st.session_state.get("y_test")
    if y_test is None:
        info_box("Test data not found in session.", "warning")
        return

    model_names = [r.name for r in result.leaderboard]
    sel = st.selectbox("Model", model_names, key="roc_model")
    selected = next(r for r in result.leaderboard if r.name == sel)

    X_test = st.session_state.get("X_test")
    if X_test is None:
        return

    try:
        model = selected.model
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                fig = roc_curve_chart(fpr, tpr, roc_auc)
                st.plotly_chart(fig, use_container_width=True)
            else:
                info_box("ROC curve shown for binary classification only.", "info")
        else:
            info_box("This model does not support probability predictions.", "info")
    except Exception as exc:
        info_box(f"ROC curve generation failed: {exc}", "error")


def _cm_tab(result) -> None:
    y_test = st.session_state.get("y_test")
    X_test = st.session_state.get("X_test")
    if y_test is None or X_test is None:
        info_box("Test data not found.", "warning")
        return

    model_names = [r.name for r in result.leaderboard]
    sel = st.selectbox("Model", model_names, key="cm_model")
    selected = next(r for r in result.leaderboard if r.name == sel)

    try:
        y_pred = selected.model.predict(X_test)
        labels = sorted(y_test.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = confusion_matrix_chart(cm, labels)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        info_box(f"Confusion matrix failed: {exc}", "error")


def _details_tab(result) -> None:
    model_names = [r.name for r in result.leaderboard]
    sel = st.selectbox("Select Model", model_names, key="detail_model")
    selected = next(r for r in result.leaderboard if r.name == sel)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Metrics**")
        metrics_df = pd.DataFrame(
            [{"Metric": k.upper(), "Value": round(v, 6)} for k, v in selected.metrics.items()]
        )
        st.dataframe(metrics_df, use_container_width=True)

    with col2:
        st.markdown("**Hyperparameters**")
        if isinstance(selected.params, dict) and selected.params:
            params_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in list(selected.params.items())[:20]]
            )
            st.dataframe(params_df, use_container_width=True)
        else:
            st.info("No hyperparameters available.")
