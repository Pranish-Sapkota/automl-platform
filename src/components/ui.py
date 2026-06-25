"""Reusable Streamlit UI components."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# ─────────────────────────── Metric Cards ──────────────────────────────

def metric_card(label: str, value: Any, delta: str | None = None, icon: str = "📊") -> None:
    """Render a styled metric card."""
    delta_html = f'<span style="color:#10B981;font-size:0.75rem">{delta}</span>' if delta else ""
    st.markdown(
        f"""
        <div style="
            background: #1A1D2E;
            border: 1px solid #2D3748;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            text-align: center;
        ">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="color:#64748B;font-size:0.78rem;text-transform:uppercase;
                        letter-spacing:0.05em;margin-top:0.3rem">{label}</div>
            <div style="color:#E2E8F0;font-size:1.6rem;font-weight:700;
                        margin-top:0.2rem">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[dict[str, Any]]) -> None:
    """Render a row of metric cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            metric_card(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                delta=m.get("delta"),
                icon=m.get("icon", "📊"),
            )


# ─────────────────────────── Status Badges ──────────────────────────────

def status_badge(text: str, status: str = "info") -> str:
    colors = {
        "success": ("#10B981", "#064E3B"),
        "warning": ("#F59E0B", "#451A03"),
        "danger": ("#EF4444", "#450A0A"),
        "info": ("#3B82F6", "#172554"),
        "purple": ("#8B5CF6", "#2E1065"),
    }
    fg, bg = colors.get(status, colors["info"])
    return (
        f'<span style="background:{bg};color:{fg};padding:0.2rem 0.6rem;'
        f'border-radius:9999px;font-size:0.72rem;font-weight:600;'
        f'border:1px solid {fg}">{text}</span>'
    )


def render_badge(text: str, status: str = "info") -> None:
    st.markdown(status_badge(text, status), unsafe_allow_html=True)


# ─────────────────────────── Section Headers ──────────────────────────────

def section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    icon_html = f'<span style="font-size:1.5rem;margin-right:0.5rem">{icon}</span>' if icon else ""
    st.markdown(
        f"""
        <div style="margin: 1rem 0 0.5rem 0;">
            <div style="display:flex;align-items:center">
                {icon_html}
                <h2 style="color:#E2E8F0;margin:0;font-size:1.4rem;font-weight:700">{title}</h2>
            </div>
            {f'<p style="color:#64748B;margin:0.2rem 0 0 0;font-size:0.85rem">{subtitle}</p>' if subtitle else ""}
        </div>
        <hr style="border:none;border-top:1px solid #2D3748;margin:0.5rem 0 1rem 0">
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1A1D2E 0%, #0F1117 100%);
            border: 1px solid #2D3748;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        ">
            <h1 style="color:#E2E8F0;margin:0;font-size:2rem;font-weight:800">{title}</h1>
            {f'<p style="color:#64748B;margin:0.5rem 0 0 0;font-size:1rem">{subtitle}</p>' if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Info/Warning Boxes ──────────────────────────────

def info_box(message: str, type_: str = "info") -> None:
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    colors = {
        "info": ("#3B82F6", "#172554"),
        "success": ("#10B981", "#064E3B"),
        "warning": ("#F59E0B", "#451A03"),
        "error": ("#EF4444", "#450A0A"),
    }
    fg, bg = colors.get(type_, colors["info"])
    icon = icons.get(type_, "ℹ️")
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:4px solid {fg};
            border-radius:8px;
            padding:0.8rem 1rem;
            margin:0.5rem 0;
            color:{fg};
            font-size:0.9rem;
        ">{icon} {message}</div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Progress Steps ──────────────────────────────

def pipeline_steps(steps: list[dict[str, Any]]) -> None:
    """Render horizontal pipeline step indicators."""
    n = len(steps)
    cols = st.columns(n)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            done = step.get("done", False)
            active = step.get("active", False)
            color = "#10B981" if done else "#6366F1" if active else "#2D3748"
            text_color = "#E2E8F0" if (done or active) else "#64748B"
            st.markdown(
                f"""
                <div style="text-align:center;padding:0.5rem">
                    <div style="
                        width:32px;height:32px;border-radius:50%;
                        background:{color};color:white;
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto;font-weight:bold;font-size:0.85rem
                    ">{'✓' if done else str(i + 1)}</div>
                    <div style="color:{text_color};font-size:0.72rem;margin-top:0.3rem">
                        {step['label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────── DataFrame Display ──────────────────────────────

def styled_dataframe(
    df: pd.DataFrame,
    height: int = 400,
    key: str | None = None,
) -> None:
    """Display a styled DataFrame with AgGrid if available, else st.dataframe."""
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode  # type: ignore

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(
            resizable=True, sortable=True, filter=True, wrapHeaderText=True
        )
        grid_opts = gb.build()
        AgGrid(
            df,
            gridOptions=grid_opts,
            height=height,
            update_mode=GridUpdateMode.NO_UPDATE,
            theme="balham-dark",
            key=key,
        )
    except Exception:
        st.dataframe(df, use_container_width=True, height=height)


# ─────────────────────────── Sidebar Navigation ──────────────────────────────

def render_sidebar_nav() -> None:
    """Render styled sidebar navigation."""
    st.sidebar.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.5rem 0">
            <div style="font-size:2rem">🤖</div>
            <div style="color:#6366F1;font-size:1.2rem;font-weight:800;
                        letter-spacing:0.05em">AutoML Platform</div>
            <div style="color:#64748B;font-size:0.7rem">Powered by FLAML + Mistral AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def session_state_summary() -> None:
    """Show mini dataset / model status in sidebar."""
    st.sidebar.markdown("---")
    has_data = st.session_state.get("df") is not None
    has_model = st.session_state.get("automl_result") is not None

    st.sidebar.markdown(
        f"""
        <div style="padding:0.5rem 0">
            <div style="font-size:0.75rem;color:#64748B;text-transform:uppercase;
                        letter-spacing:0.05em;margin-bottom:0.4rem">Session Status</div>
            <div style="display:flex;gap:0.4rem;flex-wrap:wrap">
                {'<span style="background:#064E3B;color:#10B981;padding:0.15rem 0.5rem;border-radius:9999px;font-size:0.7rem">✓ Dataset</span>' if has_data else '<span style="background:#1F2937;color:#64748B;padding:0.15rem 0.5rem;border-radius:9999px;font-size:0.7rem">○ No Dataset</span>'}
                {'<span style="background:#2E1065;color:#8B5CF6;padding:0.15rem 0.5rem;border-radius:9999px;font-size:0.7rem">✓ Model</span>' if has_model else '<span style="background:#1F2937;color:#64748B;padding:0.15rem 0.5rem;border-radius:9999px;font-size:0.7rem">○ No Model</span>'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
