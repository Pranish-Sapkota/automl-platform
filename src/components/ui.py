"""
Reusable Streamlit UI components — fully theme-aware (dark + light).
All colours are read from st.session_state["theme"] at render time.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# ── Theme helpers ─────────────────────────────────────────────────────

def _t() -> dict:
    """Return the current theme token dict."""
    theme = st.session_state.get("theme", "dark")
    return {
        "dark": {
            "bg":         "#0F1117",
            "surface":    "#1A1D2E",
            "surface2":   "#141624",
            "border":     "#2D3748",
            "primary":    "#6366F1",
            "pg":         "rgba(99,102,241,0.18)",
            "secondary":  "#8B5CF6",
            "success":    "#10B981",
            "warning":    "#F59E0B",
            "danger":     "#EF4444",
            "info":       "#3B82F6",
            "text":       "#E2E8F0",
            "muted":      "#94A3B8",
            "faint":      "#475569",
            "shadow":     "rgba(0,0,0,0.35)",
        },
        "light": {
            "bg":         "#F8FAFC",
            "surface":    "#FFFFFF",
            "surface2":   "#F1F5F9",
            "border":     "#E2E8F0",
            "primary":    "#4F46E5",
            "pg":         "rgba(79,70,229,0.12)",
            "secondary":  "#7C3AED",
            "success":    "#059669",
            "warning":    "#D97706",
            "danger":     "#DC2626",
            "info":       "#2563EB",
            "text":       "#0F172A",
            "muted":      "#475569",
            "faint":      "#94A3B8",
            "shadow":     "rgba(0,0,0,0.08)",
        },
    }[theme]


# ─────────────────────────── Page Header ─────────────────────────────

def page_header(title: str, subtitle: str = "") -> None:
    T = _t()
    sub_html = (
        f'<p style="color:{T["muted"]};margin:0.4rem 0 0 0;'
        f'font-size:0.95rem;font-weight:400">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {T["surface"]} 0%, {T["surface2"]} 100%);
            border: 1px solid {T["border"]};
            border-radius: 16px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px {T["shadow"]};
        ">
            <h1 style="color:{T["text"]};margin:0;font-size:1.75rem;
                       font-weight:800;letter-spacing:-0.02em">{title}</h1>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Section Header ──────────────────────────

def section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    T = _t()
    icon_html = (
        f'<span style="font-size:1.3rem;margin-right:0.45rem">{icon}</span>'
        if icon else ""
    )
    sub_html = (
        f'<p style="color:{T["muted"]};margin:0.15rem 0 0 0;font-size:0.82rem">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="margin:1.2rem 0 0.5rem 0">
            <div style="display:flex;align-items:center">
                {icon_html}
                <h2 style="color:{T["text"]};margin:0;font-size:1.15rem;
                           font-weight:700;letter-spacing:-0.01em">{title}</h2>
            </div>
            {sub_html}
        </div>
        <div style="height:2px;background:linear-gradient(90deg,{T["primary"]},transparent);
                    border-radius:9999px;margin:0.4rem 0 1rem 0"></div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Metric Card ─────────────────────────────

def metric_card(
    label: str,
    value: Any,
    delta: str | None = None,
    icon: str = "📊",
    color: str | None = None,
) -> None:
    T = _t()
    accent = color or T["primary"]
    delta_html = (
        f'<div style="color:{T["success"]};font-size:0.72rem;margin-top:0.15rem">{delta}</div>'
        if delta else ""
    )
    st.markdown(
        f"""
        <div style="
            background:{T["surface"]};
            border:1px solid {T["border"]};
            border-top:3px solid {accent};
            border-radius:12px;
            padding:1.1rem 1.2rem;
            text-align:center;
            transition:transform 0.18s ease,box-shadow 0.18s ease;
            box-shadow:0 2px 8px {T["shadow"]};
        " onmouseover="this.style.transform='translateY(-3px)';
                       this.style.boxShadow='0 8px 24px {T["shadow"]}'"
          onmouseout="this.style.transform='';this.style.boxShadow='0 2px 8px {T["shadow"]}'">
            <div style="font-size:1.5rem;line-height:1">{icon}</div>
            <div style="color:{T["muted"]};font-size:0.7rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin-top:0.35rem;font-weight:600">
                {label}
            </div>
            <div style="color:{T["text"]};font-size:1.4rem;font-weight:800;
                        margin-top:0.2rem;letter-spacing:-0.01em">
                {value}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[dict[str, Any]]) -> None:
    cols = st.columns(len(metrics))
    colors = [None, "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#3B82F6"]
    for i, (col, m) in enumerate(zip(cols, metrics)):
        with col:
            metric_card(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                delta=m.get("delta"),
                icon=m.get("icon", "📊"),
                color=m.get("color", colors[i % len(colors)]),
            )


# ─────────────────────────── Info Box ────────────────────────────────

def info_box(message: str, type_: str = "info") -> None:
    T = _t()
    cfg = {
        "info":    (T["info"],    "rgba(59,130,246,0.08)",  "ℹ️"),
        "success": (T["success"], "rgba(16,185,129,0.08)",  "✅"),
        "warning": (T["warning"], "rgba(245,158,11,0.08)",  "⚠️"),
        "error":   (T["danger"],  "rgba(239,68,68,0.08)",   "❌"),
    }
    fg, bg, icon = cfg.get(type_, cfg["info"])
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:4px solid {fg};
            border-radius:0 10px 10px 0;
            padding:0.75rem 1rem;
            margin:0.5rem 0;
            color:{fg};
            font-size:0.875rem;
            line-height:1.5;
        ">{icon} {message}</div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Status Badge ────────────────────────────

def status_badge(text: str, status: str = "info") -> str:
    T = _t()
    cfg = {
        "success": (T["success"], "rgba(16,185,129,0.15)"),
        "warning": (T["warning"], "rgba(245,158,11,0.15)"),
        "danger":  (T["danger"],  "rgba(239,68,68,0.15)"),
        "info":    (T["info"],    "rgba(59,130,246,0.15)"),
        "purple":  (T["secondary"], "rgba(139,92,246,0.15)"),
    }
    fg, bg = cfg.get(status, cfg["info"])
    return (
        f'<span style="background:{bg};color:{fg};padding:0.2rem 0.65rem;'
        f'border-radius:9999px;font-size:0.72rem;font-weight:700;'
        f'border:1px solid {fg}44">{text}</span>'
    )


def render_badge(text: str, status: str = "info") -> None:
    st.markdown(status_badge(text, status), unsafe_allow_html=True)


# ─────────────────────────── Pipeline Steps ──────────────────────────

def pipeline_steps(steps: list[dict[str, Any]]) -> None:
    T = _t()
    n = len(steps)
    cols = st.columns(n)
    for i, (col, step) in enumerate(zip(cols, steps)):
        done   = step.get("done", False)
        active = step.get("active", False)
        if done:
            ring_color = T["success"]
            num_or_check = "✓"
            label_color = T["success"]
            bg_color = "rgba(16,185,129,0.12)"
        elif active:
            ring_color = T["primary"]
            num_or_check = str(i + 1)
            label_color = T["primary"]
            bg_color = T["pg"]
        else:
            ring_color = T["border"]
            num_or_check = str(i + 1)
            label_color = T["muted"]
            bg_color = T["surface2"]

        with col:
            st.markdown(
                f"""
                <div style="text-align:center;padding:0.4rem 0.2rem">
                    <div style="
                        width:34px;height:34px;border-radius:50%;
                        background:{bg_color};
                        border:2px solid {ring_color};
                        color:{ring_color};
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto;font-weight:700;font-size:0.82rem;
                        transition:all 0.2s ease;
                    ">{num_or_check}</div>
                    <div style="color:{label_color};font-size:0.68rem;
                                margin-top:0.3rem;font-weight:600;
                                letter-spacing:0.02em">
                        {step["label"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────── Sidebar Nav ─────────────────────────────

def render_sidebar_nav() -> None:
    """Called from app.py — intentionally minimal; app.py renders its own header."""
    pass


def session_state_summary() -> None:
    """No-op — handled inline in app.py with full theme awareness."""
    pass


# ─────────────────────────── DataFrame ───────────────────────────────

def styled_dataframe(
    df: pd.DataFrame,
    height: int = 400,
    key: str | None = None,
) -> None:
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode  # type: ignore

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(
            resizable=True, sortable=True, filter=True, wrapHeaderText=True,
        )
        grid_opts = gb.build()
        AgGrid(
            df,
            gridOptions=grid_opts,
            height=height,
            update_mode=GridUpdateMode.NO_UPDATE,
            theme="balham-dark" if st.session_state.get("theme", "dark") == "dark" else "balham",
            key=key,
        )
    except Exception:
        st.dataframe(df, use_container_width=True, height=height)


# ─────────────────────────── Feature Card ────────────────────────────

def feature_card(icon: str, title: str, desc: str) -> None:
    T = _t()
    st.markdown(
        f"""
        <div style="
            background:{T["surface"]};
            border:1px solid {T["border"]};
            border-radius:12px;
            padding:1rem 1.1rem;
            margin-bottom:0.65rem;
            transition:transform 0.18s ease,box-shadow 0.18s ease;
        " onmouseover="this.style.transform='translateY(-2px)';
                       this.style.boxShadow='0 6px 20px {T["shadow"]}';
                       this.style.borderColor='{T["primary"]}'"
          onmouseout="this.style.transform='';
                      this.style.boxShadow='';
                      this.style.borderColor='{T["border"]}'">
            <span style="font-size:1.2rem">{icon}</span>
            <div style="color:{T["text"]};font-weight:600;font-size:0.88rem;
                        margin-top:0.2rem">{title}</div>
            <div style="color:{T["muted"]};font-size:0.76rem;
                        margin-top:0.15rem;line-height:1.45">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
