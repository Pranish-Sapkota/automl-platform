"""
AutoML Platform — Enhanced UI
Dark/Light mode · Responsive · Sidebar hide/show · Mobile-first
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Pranish-Sapkota/automl-platform",
        "Report a bug": "https://github.com/Pranish-Sapkota/automl-platform/issues",
        "About": "AutoML Platform — FLAML · SHAP · Mistral AI",
    },
)

from src.storage import init_db
from src.components import render_sidebar_nav, session_state_summary
from src.pages import (
    home, upload, profiling, cleaning,
    training, leaderboard, explainability, chat, export,
)
from src.utils import get_logger

logger = get_logger("app")


@st.cache_resource
def _init_storage() -> None:
    init_db()


_init_storage()

# ── Initialise theme in session state ────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"   # default: dark

THEME = st.session_state["theme"]
IS_DARK = THEME == "dark"

# ── Theme token maps ──────────────────────────────────────────────────
T = {
    "dark": {
        "bg":           "#0F1117",
        "surface":      "#1A1D2E",
        "surface2":     "#141624",
        "border":       "#2D3748",
        "primary":      "#6366F1",
        "primary_glow": "rgba(99,102,241,0.25)",
        "secondary":    "#8B5CF6",
        "success":      "#10B981",
        "warning":      "#F59E0B",
        "danger":       "#EF4444",
        "info":         "#3B82F6",
        "text":         "#E2E8F0",
        "text_muted":   "#94A3B8",
        "text_faint":   "#475569",
        "sidebar_bg":   "#0D0F1A",
        "input_bg":     "#1E2130",
        "shadow":       "rgba(0,0,0,0.4)",
        "toggle_bg":    "#1A1D2E",
        "toggle_icon":  "☀️",
        "toggle_label": "Light mode",
        "code_bg":      "#0D1117",
    },
    "light": {
        "bg":           "#F8FAFC",
        "surface":      "#FFFFFF",
        "surface2":     "#F1F5F9",
        "border":       "#E2E8F0",
        "primary":      "#4F46E5",
        "primary_glow": "rgba(79,70,229,0.15)",
        "secondary":    "#7C3AED",
        "success":      "#059669",
        "warning":      "#D97706",
        "danger":       "#DC2626",
        "info":         "#2563EB",
        "text":         "#0F172A",
        "text_muted":   "#475569",
        "text_faint":   "#94A3B8",
        "sidebar_bg":   "#F1F5F9",
        "input_bg":     "#FFFFFF",
        "shadow":       "rgba(0,0,0,0.08)",
        "toggle_bg":    "#FFFFFF",
        "toggle_icon":  "🌙",
        "toggle_label": "Dark mode",
        "code_bg":      "#F8FAFC",
    },
}[THEME]


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ═══════════════════════════════════════════════════════════
           ROOT & RESET
        ═══════════════════════════════════════════════════════════ */
        :root {{
            --bg:           {T["bg"]};
            --surface:      {T["surface"]};
            --surface2:     {T["surface2"]};
            --border:       {T["border"]};
            --primary:      {T["primary"]};
            --primary-glow: {T["primary_glow"]};
            --secondary:    {T["secondary"]};
            --success:      {T["success"]};
            --warning:      {T["warning"]};
            --danger:       {T["danger"]};
            --info:         {T["info"]};
            --text:         {T["text"]};
            --text-muted:   {T["text_muted"]};
            --text-faint:   {T["text_faint"]};
            --shadow:       {T["shadow"]};
        }}

        /* ── Force background & text everywhere ── */
        html, body, [data-testid="stApp"] {{
            background-color: {T["bg"]} !important;
            color: {T["text"]} !important;
        }}

        /* ── Block container ── */
        .block-container {{
            padding: 1.2rem 1.5rem 2rem 1.5rem !important;
            max-width: 1400px !important;
            background-color: {T["bg"]} !important;
        }}

        /* ── All text elements ── */
        p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {{
            color: {T["text"]} !important;
        }}

        /* ── Streamlit markdown ── */
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span {{
            color: {T["text"]} !important;
        }}

        /* ── Code blocks ── */
        code, pre, .stCode {{
            background-color: {T["code_bg"]} !important;
            color: {T["text"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 6px !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           HIDE STREAMLIT CHROME
        ═══════════════════════════════════════════════════════════ */
        #MainMenu, footer, header {{ visibility: hidden !important; }}

        /* ═══════════════════════════════════════════════════════════
           SCROLLBAR
        ═══════════════════════════════════════════════════════════ */
        ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
        ::-webkit-scrollbar-track {{ background: {T["bg"]}; }}
        ::-webkit-scrollbar-thumb {{
            background: {T["primary"]};
            border-radius: 4px;
        }}

        /* ═══════════════════════════════════════════════════════════
           SIDEBAR
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {{
            background: {T["sidebar_bg"]} !important;
            border-right: 1px solid {T["border"]} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {T["text"]} !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            color: {T["text_muted"]} !important;
            font-size: 0.87rem !important;
            padding: 0.35rem 0.6rem !important;
            border-radius: 7px !important;
            transition: all 0.18s ease !important;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background: {T["primary_glow"]} !important;
            color: {T["primary"]} !important;
        }}
        /* Active nav item */
        [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
        [data-testid="stSidebar"] .stRadio input:checked + div label {{
            background: {T["primary_glow"]} !important;
            color: {T["primary"]} !important;
            font-weight: 600 !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           BUTTONS
        ═══════════════════════════════════════════════════════════ */
        .stButton > button {{
            border-radius: 9px !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            padding: 0.45rem 1.1rem !important;
            transition: all 0.2s ease !important;
            border: 1px solid {T["border"]} !important;
            background: {T["surface"]} !important;
            color: {T["text"]} !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px {T["shadow"]} !important;
            border-color: {T["primary"]} !important;
            color: {T["primary"]} !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {T["primary"]}, {T["secondary"]}) !important;
            border: none !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px {T["primary_glow"]} !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            opacity: 0.92 !important;
            box-shadow: 0 8px 24px {T["primary_glow"]} !important;
            color: #FFFFFF !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           TABS
        ═══════════════════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {{
            background: {T["surface2"]} !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 3px !important;
            border: 1px solid {T["border"]} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 9px !important;
            color: {T["text_muted"]} !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            padding: 0.4rem 0.9rem !important;
            transition: all 0.18s ease !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: {T["primary_glow"]} !important;
            color: {T["primary"]} !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {T["primary"]} !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           INPUTS — selectbox, text, number, slider
        ═══════════════════════════════════════════════════════════ */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea textarea {{
            background: {T["input_bg"]} !important;
            color: {T["text"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div:focus-within {{
            border-color: {T["primary"]} !important;
            box-shadow: 0 0 0 3px {T["primary_glow"]} !important;
        }}

        /* Slider track & thumb */
        [data-testid="stSlider"] > div > div > div > div {{
            background: {T["primary"]} !important;
        }}
        [data-testid="stSlider"] [role="slider"] {{
            background: {T["primary"]} !important;
            box-shadow: 0 0 0 4px {T["primary_glow"]} !important;
        }}
        [data-testid="stSlider"] p {{
            color: {T["text"]} !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           METRICS
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stMetric"] {{
            background: {T["surface"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 12px !important;
            padding: 1rem 1.2rem !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px {T["shadow"]} !important;
        }}
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] span {{
            color: {T["text_muted"]} !important;
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {T["text"]} !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           DATAFRAME / TABLE
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] * {{
            background-color: {T["surface"]} !important;
            color: {T["text"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 10px !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           FILE UPLOADER
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stFileUploadDropzone"] {{
            background: {T["surface"]} !important;
            border: 2px dashed {T["primary"]} !important;
            border-radius: 14px !important;
            transition: all 0.2s ease !important;
        }}
        [data-testid="stFileUploadDropzone"]:hover {{
            background: {T["primary_glow"]} !important;
        }}
        [data-testid="stFileUploadDropzone"] * {{
            color: {T["text"]} !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           EXPANDER
        ═══════════════════════════════════════════════════════════ */
        .streamlit-expanderHeader {{
            background: {T["surface"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 10px !important;
            color: {T["text"]} !important;
        }}
        .streamlit-expanderContent {{
            background: {T["surface"]} !important;
            border: 1px solid {T["border"]} !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           ALERTS / INFO BOXES
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stInfo"],
        [data-testid="stSuccess"],
        [data-testid="stWarning"],
        [data-testid="stError"] {{
            border-radius: 10px !important;
            border-left-width: 4px !important;
        }}
        [data-testid="stInfo"] * {{ color: {T["info"]} !important; }}
        [data-testid="stSuccess"] * {{ color: {T["success"]} !important; }}
        [data-testid="stWarning"] * {{ color: {T["warning"]} !important; }}
        [data-testid="stError"] * {{ color: {T["danger"]} !important; }}

        /* ═══════════════════════════════════════════════════════════
           PROGRESS BAR
        ═══════════════════════════════════════════════════════════ */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {T["primary"]}, {T["secondary"]}) !important;
            border-radius: 9999px !important;
        }}
        .stProgress > div > div {{
            background: {T["surface2"]} !important;
            border-radius: 9999px !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           CHAT MESSAGES
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stChatMessage"] {{
            background: {T["surface"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 14px !important;
        }}
        [data-testid="stChatMessage"] * {{
            color: {T["text"]} !important;
        }}
        [data-testid="stChatInput"] textarea {{
            background: {T["surface"]} !important;
            color: {T["text"]} !important;
            border: 1px solid {T["border"]} !important;
            border-radius: 12px !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           CHECKBOX & RADIO
        ═══════════════════════════════════════════════════════════ */
        .stCheckbox label, .stRadio label {{
            color: {T["text"]} !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           PLOTLY CHARTS — force bg to match theme
        ═══════════════════════════════════════════════════════════ */
        .js-plotly-plot .plotly {{
            background: {T["surface"]} !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           MOBILE RESPONSIVE  (≤ 768px)
        ═══════════════════════════════════════════════════════════ */
        @media (max-width: 768px) {{
            .block-container {{
                padding: 0.8rem 0.8rem 1.5rem 0.8rem !important;
            }}
            /* Stack columns vertically on mobile */
            [data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
            }}
            [data-testid="stHorizontalBlock"] > div {{
                width: 100% !important;
                min-width: 100% !important;
            }}
            /* Bigger tap targets */
            .stButton > button {{
                padding: 0.6rem 1rem !important;
                font-size: 0.95rem !important;
                width: 100% !important;
            }}
            /* Full-width tabs */
            .stTabs [data-baseweb="tab"] {{
                font-size: 0.78rem !important;
                padding: 0.35rem 0.55rem !important;
            }}
            /* Readable metric cards */
            [data-testid="stMetricValue"] {{
                font-size: 1.2rem !important;
            }}
            /* Sidebar overlay on mobile */
            [data-testid="stSidebar"] {{
                min-width: 260px !important;
            }}
        }}

        @media (max-width: 480px) {{
            .block-container {{
                padding: 0.6rem 0.5rem 1.2rem 0.5rem !important;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 1rem !important;
            }}
        }}

        /* ═══════════════════════════════════════════════════════════
           CUSTOM CARD CLASS (used in page_header etc.)
        ═══════════════════════════════════════════════════════════ */
        .automl-card {{
            background: {T["surface"]};
            border: 1px solid {T["border"]};
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }}
        .automl-card:hover {{
            box-shadow: 0 8px 28px {T["shadow"]};
            transform: translateY(-2px);
        }}

        /* ═══════════════════════════════════════════════════════════
           DIVIDER
        ═══════════════════════════════════════════════════════════ */
        hr {{
            border-color: {T["border"]} !important;
            margin: 0.8rem 0 !important;
        }}

        /* ═══════════════════════════════════════════════════════════
           SPINNER
        ═══════════════════════════════════════════════════════════ */
        [data-testid="stSpinner"] * {{
            color: {T["primary"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()


# ── Theme toggle (top-right floating button) ─────────────────────────
def _theme_toggle() -> None:
    """Render the dark/light mode toggle in the sidebar header."""
    col1, col2 = st.sidebar.columns([3, 1])
    with col2:
        if st.button(
            T["toggle_icon"],
            key="theme_toggle_btn",
            help=T["toggle_label"],
        ):
            st.session_state["theme"] = "light" if IS_DARK else "dark"
            st.rerun()


# ── Sidebar logo & toggle ─────────────────────────────────────────────
def _render_sidebar_header() -> None:
    _theme_toggle()
    st.sidebar.markdown(
        f"""
        <div style="text-align:center;padding:0.6rem 0 1.2rem 0">
            <div style="font-size:2.2rem;line-height:1">🤖</div>
            <div style="color:{T["primary"]};font-size:1.15rem;font-weight:800;
                        letter-spacing:0.04em;margin-top:0.3rem">AutoML Platform</div>
            <div style="color:{T["text_faint"]};font-size:0.68rem;margin-top:0.15rem">
                FLAML · SHAP · Mistral AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Session status pill ───────────────────────────────────────────────
def _session_status() -> None:
    has_data  = st.session_state.get("df") is not None
    has_model = st.session_state.get("automl_result") is not None

    d_col = T["success"] if has_data  else T["text_faint"]
    m_col = T["secondary"] if has_model else T["text_faint"]
    d_bg  = "rgba(16,185,129,0.12)" if has_data  else T["surface2"]
    m_bg  = "rgba(139,92,246,0.12)" if has_model else T["surface2"]

    st.sidebar.markdown(
        f"""
        <div style="padding:0.5rem 0.8rem;margin:0.4rem 0 0.8rem 0;
                    border:1px solid {T["border"]};border-radius:10px;
                    background:{T["surface"]}">
            <div style="font-size:0.68rem;color:{T["text_faint"]};
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem">
                Session
            </div>
            <div style="display:flex;gap:0.4rem;flex-wrap:wrap">
                <span style="background:{d_bg};color:{d_col};
                             padding:0.15rem 0.55rem;border-radius:9999px;
                             font-size:0.7rem;font-weight:600;
                             border:1px solid {d_col}33">
                    {"✓ Dataset" if has_data else "○ No Data"}
                </span>
                <span style="background:{m_bg};color:{m_col};
                             padding:0.15rem 0.55rem;border-radius:9999px;
                             font-size:0.7rem;font-weight:600;
                             border:1px solid {m_col}33">
                    {"✓ Model" if has_model else "○ No Model"}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Navigation pages ──────────────────────────────────────────────────
PAGES = {
    "🏠  Home":               home,
    "📂  Upload Dataset":     upload,
    "🔍  Data Profiling":     profiling,
    "⚙️  Data Cleaning":      cleaning,
    "🧪  AutoML Training":    training,
    "🏆  Model Leaderboard":  leaderboard,
    "🔮  Explainability":     explainability,
    "💬  AI Chat Assistant":  chat,
    "📦  Model Export":       export,
}

# ── Build sidebar ─────────────────────────────────────────────────────
_render_sidebar_header()

with st.sidebar:
    st.markdown(
        f'<div style="font-size:0.72rem;color:{T["text_faint"]};'
        f'text-transform:uppercase;letter-spacing:0.07em;'
        f'margin-bottom:0.3rem;padding-left:0.2rem">Navigation</div>',
        unsafe_allow_html=True,
    )
    selected_page = st.radio(
        "nav",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav_page",
    )

    _session_status()

    st.sidebar.markdown(
        f"""
        <div style="padding:0.6rem 0.8rem;margin-top:0.5rem;
                    border:1px solid {T["border"]};border-radius:10px;
                    background:{T["surface"]};font-size:0.68rem;color:{T["text_faint"]}">
            <div style="color:{T["primary"]};font-weight:700;font-size:0.75rem">
                🤖 AutoML Platform
            </div>
            <div style="margin-top:0.25rem">v1.0.0 · MIT License</div>
            <div style="margin-top:0.1rem">
                <a href="https://github.com/Pranish-Sapkota/automl-platform"
                   style="color:{T["primary"]};text-decoration:none">
                   GitHub ↗
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render page ───────────────────────────────────────────────────────
try:
    PAGES[selected_page].render()
except Exception as exc:
    st.error(f"❌ Page error: {exc}")
    logger.error("Page error on '%s': %s", selected_page, exc, exc_info=True)
    with st.expander("🔧 Error Details"):
        import traceback
        st.code(traceback.format_exc())
