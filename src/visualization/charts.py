"""Plotly chart builders for EDA, model metrics, and explainability."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "bg": "#0F1117",
    "surface": "#1A1D2E",
    "text": "#E2E8F0",
    "muted": "#64748B",
}

PALETTE = [
    "#6366F1", "#8B5CF6", "#10B981", "#F59E0B",
    "#EF4444", "#3B82F6", "#EC4899", "#14B8A6",
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["surface"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=COLORS["text"])), **LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="#2D3748", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="#2D3748", showgrid=True, zeroline=False)
    return fig


# ─────────────────────────── EDA Charts ──────────────────────────────

def missing_values_heatmap(df: pd.DataFrame) -> go.Figure:
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values!", showarrow=False,
                           font=dict(size=20, color=COLORS["success"]))
        return _apply_defaults(fig, "Missing Values")

    fig = go.Figure(go.Bar(
        x=missing_pct.index.tolist(),
        y=missing_pct.values.tolist(),
        marker_color=[
            COLORS["danger"] if v > 30 else COLORS["warning"] if v > 10 else COLORS["info"]
            for v in missing_pct.values
        ],
        text=[f"{v:.1f}%" for v in missing_pct.values],
        textposition="outside",
    ))
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="30% threshold")
    return _apply_defaults(fig, "Missing Values by Column (%)")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns", showarrow=False)
        return _apply_defaults(fig, "Correlation Matrix")

    corr = numeric.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="r", tickfont=dict(color=COLORS["text"])),
    ))
    return _apply_defaults(fig, "Correlation Matrix")


def distribution_plot(series: pd.Series, col_name: str) -> go.Figure:
    if pd.api.types.is_numeric_dtype(series):
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Distribution", "Box Plot"])
        fig.add_trace(
            go.Histogram(x=series.dropna(), name=col_name,
                         marker_color=COLORS["primary"], nbinsx=40, opacity=0.8),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=series.dropna(), name=col_name,
                   marker_color=COLORS["secondary"], boxmean=True),
            row=1, col=2,
        )
    else:
        vc = series.value_counts().head(20)
        fig = go.Figure(go.Bar(
            x=vc.index.tolist(), y=vc.values.tolist(),
            marker_color=COLORS["primary"], opacity=0.85,
        ))
    return _apply_defaults(fig, f"Distribution: {col_name}")


def target_distribution(series: pd.Series, problem_type: str) -> go.Figure:
    if "classification" in problem_type:
        vc = series.value_counts()
        fig = go.Figure(go.Bar(
            x=[str(v) for v in vc.index],
            y=vc.values.tolist(),
            marker_color=PALETTE[:len(vc)],
            text=vc.values.tolist(),
            textposition="outside",
        ))
        return _apply_defaults(fig, "Target Class Distribution")
    else:
        fig = go.Figure(go.Histogram(
            x=series.dropna(), nbinsx=50,
            marker_color=COLORS["primary"], opacity=0.85,
        ))
        return _apply_defaults(fig, "Target Variable Distribution")


def outlier_boxplots(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    cols = columns[:8]
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean=True,
        ))
    return _apply_defaults(fig, "Outlier Analysis (Box Plots)")


def scatter_matrix(df: pd.DataFrame, target: str | None = None) -> go.Figure:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    if target and target not in numeric and target in df.columns:
        color_col = None
        dimensions = numeric
    else:
        color_col = target if target in numeric else None
        dimensions = [c for c in numeric if c != target][:5]

    fig = px.scatter_matrix(
        df[dimensions + ([target] if target and target in df.columns else [])],
        dimensions=dimensions,
        color=target if target in df.columns else None,
        color_continuous_scale="Viridis",
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
    return _apply_defaults(fig, "Scatter Matrix")


# ─────────────────────────── Model Charts ──────────────────────────────

def leaderboard_chart(leaderboard_df: pd.DataFrame, metric: str) -> go.Figure:
    df = leaderboard_df.sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(
        x=df[metric],
        y=df["Model"],
        orientation="h",
        marker=dict(
            color=df[metric],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=metric, tickfont=dict(color=COLORS["text"])),
        ),
        text=[f"{v:.4f}" for v in df[metric]],
        textposition="outside",
    ))
    return _apply_defaults(fig, f"Model Leaderboard — {metric.upper()}")


def metrics_radar(metrics: dict[str, float], model_name: str) -> go.Figure:
    categories = list(metrics.keys())
    values = list(metrics.values())
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba(99,102,241,0.3)",
        line=dict(color=COLORS["primary"], width=2),
        name=model_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2D3748"),
            angularaxis=dict(gridcolor="#2D3748"),
            bgcolor=COLORS["surface"],
        ),
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    fig.update_layout(title=dict(text=f"Metrics Radar — {model_name}",
                                 font=dict(size=16, color=COLORS["text"])))
    return fig


def roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=COLORS["primary"], width=2),
        name=f"ROC (AUC = {auc:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=COLORS["muted"], dash="dash"),
        name="Random",
    ))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _apply_defaults(fig, "ROC Curve")


def confusion_matrix_chart(cm: np.ndarray, labels: list) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorbar=dict(tickfont=dict(color=COLORS["text"])),
    ))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return _apply_defaults(fig, "Confusion Matrix")


def training_time_chart(leaderboard_df: pd.DataFrame) -> go.Figure:
    df = leaderboard_df.sort_values("Train Time (s)", ascending=False)
    fig = go.Figure(go.Bar(
        x=df["Train Time (s)"],
        y=df["Model"],
        orientation="h",
        marker_color=COLORS["secondary"],
        text=[f"{v:.2f}s" for v in df["Train Time (s)"]],
        textposition="outside",
    ))
    return _apply_defaults(fig, "Training Time Comparison")


# ─────────────────────────── SHAP Charts ──────────────────────────────

def shap_importance_bar(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.sort_values("Importance", ascending=True).tail(20)
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(
            color=df["Importance"],
            colorscale="Purples",
            showscale=True,
            colorbar=dict(title="Mean |SHAP|", tickfont=dict(color=COLORS["text"])),
        ),
    ))
    return _apply_defaults(fig, "Global Feature Importance (SHAP)")


def shap_waterfall(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_value: float,
    sample_idx: int = 0,
) -> go.Figure:
    vals = shap_values[sample_idx] if len(shap_values.shape) > 1 else shap_values
    sorted_idx = np.argsort(np.abs(vals))[::-1][:15]
    top_vals = vals[sorted_idx]
    top_names = [feature_names[i] for i in sorted_idx]

    cumulative = base_value
    x_positions: list[float] = []
    measures: list[str] = []
    texts: list[str] = []

    x_positions.append(base_value)
    measures.append("absolute")
    texts.append(f"Base: {base_value:.3f}")

    for val, name in zip(top_vals, top_names):
        x_positions.append(val)
        measures.append("relative")
        texts.append(f"{val:+.3f}")
        cumulative += val

    x_positions.append(cumulative)
    measures.append("total")
    texts.append(f"Pred: {cumulative:.3f}")

    labels = ["Base Value"] + top_names + ["Prediction"]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measures,
        x=labels,
        y=x_positions,
        text=texts,
        textposition="outside",
        connector={"line": {"color": COLORS["muted"]}},
        increasing={"marker": {"color": COLORS["danger"]}},
        decreasing={"marker": {"color": COLORS["info"]}},
        totals={"marker": {"color": COLORS["success"]}},
    ))
    fig.update_xaxes(tickangle=45)
    return _apply_defaults(fig, f"SHAP Waterfall — Sample {sample_idx}")


def feature_importance_scatter(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.head(20)
    fig = go.Figure(go.Scatter(
        x=df["Importance"],
        y=df["Feature"],
        mode="markers",
        marker=dict(
            size=df["Importance"] / df["Importance"].max() * 30 + 8,
            color=df["Importance"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance", tickfont=dict(color=COLORS["text"])),
        ),
        text=[f"{v:.4f}" for v in df["Importance"]],
    ))
    return _apply_defaults(fig, "Feature Importance Bubble Chart")
