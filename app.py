"""
AutoML Platform — Main Application
Sidebar always shows collapse/expand button • No theme toggle • Professional dark UI
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


# ─────────────────────────── CSS ──────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ══════════════════════════════════════════════════
           GLOBAL BASE
        ══════════════════════════════════════════════════ */
        html, body, [data-testid="stApp"] {
            background-color: #0F1117 !important;
            color: #E2E8F0 !important;
        }
        .block-container {
            padding: 1.4rem 1.8rem 2rem 1.8rem !important;
            max-width: 1400px !important;
        }

        /* ══════════════════════════════════════════════════
           HIDE DEFAULT CHROME
        ══════════════════════════════════════════════════ */
        #MainMenu { visibility: hidden !important; }
        footer    { visibility: hidden !important; }
        header    { visibility: hidden !important; }

        /* ══════════════════════════════════════════════════
           SIDEBAR — body
        ══════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: #0D0F1A !important;
            border-right: 1px solid #1E2030 !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }

        /* ══════════════════════════════════════════════════
           SIDEBAR COLLAPSE BUTTON — always visible & styled
          (Streamlit renders this as button[data-testid="collapsedControl"]
           and button[data-testid="baseButton-headerNoPadding"])
        ══════════════════════════════════════════════════ */

        /* The ‹ / › collapse button */
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 9999 !important;
            background: #6366F1 !important;
            border: 2px solid #4F46E5 !important;
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.45) !important;
            cursor: pointer !important;
            transition: background 0.2s, box-shadow 0.2s !important;
        }
        [data-testid="collapsedControl"]:hover {
            background: #4F46E5 !important;
            box-shadow: 0 6px 20px rgba(99,102,241,0.6) !important;
        }
        [data-testid="collapsedControl"] svg {
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
            width: 14px !important;
            height: 14px !important;
        }

        /* Sidebar open: button sits at right edge of sidebar */
        [data-testid="stSidebar"][aria-expanded="true"]
            ~ * [data-testid="collapsedControl"],
        section[data-testid="stSidebar"][aria-expanded="true"]
            + div [data-testid="collapsedControl"] {
            left: calc(var(--sidebar-width, 240px) - 16px) !important;
        }

        /* Sidebar closed: button sits at left edge of viewport */
        [data-testid="stSidebar"][aria-expanded="false"]
            ~ * [data-testid="collapsedControl"] {
            left: 8px !important;
        }

        /* Streamlit also uses this button class for the toggle */
        button[kind="headerNoPadding"],
        [data-testid="baseButton-headerNoPadding"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 9999 !important;
            background: #6366F1 !important;
            border: 2px solid #4F46E5 !important;
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.45) !important;
            cursor: pointer !important;
            padding: 0 !important;
            min-height: unset !important;
        }
        button[kind="headerNoPadding"]:hover,
        [data-testid="baseButton-headerNoPadding"]:hover {
            background: #4F46E5 !important;
            box-shadow: 0 6px 20px rgba(99,102,241,0.6) !important;
        }
        button[kind="headerNoPadding"] svg,
        [data-testid="baseButton-headerNoPadding"] svg {
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        /* ══════════════════════════════════════════════════
           NAV RADIO ITEMS
        ══════════════════════════════════════════════════ */
        [data-testid="stSidebar"] .stRadio label {
            color: #94A3B8 !important;
            font-size: 0.875rem !important;
            padding: 0.4rem 0.7rem !important;
            border-radius: 8px !important;
            transition: all 0.18s ease !important;
            display: block !important;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(99,102,241,0.12) !important;
            color: #6366F1 !important;
        }
        [data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) label {
            background: rgba(99,102,241,0.15) !important;
            color: #6366F1 !important;
            font-weight: 600 !important;
        }

        /* ══════════════════════════════════════════════════
           SCROLLBAR
        ══════════════════════════════════════════════════ */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #0F1117; }
        ::-webkit-scrollbar-thumb {
            background: #6366F1;
            border-radius: 4px;
        }

        /* ══════════════════════════════════════════════════
           BUTTONS
        ══════════════════════════════════════════════════ */
        .stButton > button {
            border-radius: 9px !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            transition: all 0.2s ease !important;
            border: 1px solid #2D3748 !important;
            background: #1A1D2E !important;
            color: #E2E8F0 !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
            border-color: #6366F1 !important;
            color: #6366F1 !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            border: none !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.92 !important;
            box-shadow: 0 8px 24px rgba(99,102,241,0.5) !important;
            color: #FFFFFF !important;
        }

        /* ══════════════════════════════════════════════════
           TABS
        ══════════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            background: #141624 !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 3px !important;
            border: 1px solid #2D3748 !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px !important;
            color: #64748B !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            transition: all 0.18s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(99,102,241,0.1) !important;
            color: #6366F1 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #6366F1 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        /* ══════════════════════════════════════════════════
           INPUTS
        ══════════════════════════════════════════════════ */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea textarea {
            background: #1E2130 !important;
            color: #E2E8F0 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
        }
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div:focus-within {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
        }
        /* Dropdown options */
        [data-baseweb="select"] [role="listbox"],
        [data-baseweb="popover"] ul {
            background: #1A1D2E !important;
            border: 1px solid #2D3748 !important;
        }
        [data-baseweb="select"] [role="option"],
        [data-baseweb="popover"] li {
            color: #E2E8F0 !important;
        }
        [data-baseweb="select"] [role="option"]:hover {
            background: rgba(99,102,241,0.15) !important;
        }

        /* ══════════════════════════════════════════════════
           SLIDER — track, thumb, labels
        ══════════════════════════════════════════════════ */
        [data-testid="stSlider"] > div > div > div > div {
            background: #6366F1 !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #6366F1 !important;
            border: 3px solid #E2E8F0 !important;
            box-shadow: 0 0 0 4px rgba(99,102,241,0.3) !important;
            width: 18px !important;
            height: 18px !important;
        }
        [data-testid="stSlider"] p,
        [data-testid="stSlider"] span,
        [data-testid="stSlider"] div {
            color: #E2E8F0 !important;
        }
        /* Slider min/max value labels */
        [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
            color: #94A3B8 !important;
            font-size: 0.78rem !important;
        }

        /* ══════════════════════════════════════════════════
           METRICS
        ══════════════════════════════════════════════════ */
        [data-testid="stMetric"] {
            background: #1A1D2E !important;
            border: 1px solid #2D3748 !important;
            border-radius: 12px !important;
            padding: 1rem 1.2rem !important;
            transition: transform 0.18s, box-shadow 0.18s !important;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
        }
        [data-testid="stMetricLabel"] p { color: #94A3B8 !important; }
        [data-testid="stMetricValue"]   { color: #E2E8F0 !important; font-weight: 700 !important; }

        /* ══════════════════════════════════════════════════
           DATAFRAME
        ══════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
        }
        [data-testid="stDataFrame"] * {
            background-color: #1A1D2E !important;
            color: #E2E8F0 !important;
        }

        /* ══════════════════════════════════════════════════
           FILE UPLOADER
        ══════════════════════════════════════════════════ */
        [data-testid="stFileUploadDropzone"] {
            background: #1A1D2E !important;
            border: 2px dashed #6366F1 !important;
            border-radius: 14px !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            background: rgba(99,102,241,0.08) !important;
        }
        [data-testid="stFileUploadDropzone"] * { color: #E2E8F0 !important; }

        /* ══════════════════════════════════════════════════
           EXPANDER
        ══════════════════════════════════════════════════ */
        .streamlit-expanderHeader {
            background: #1A1D2E !important;
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
            color: #E2E8F0 !important;
        }
        .streamlit-expanderContent {
            background: #1A1D2E !important;
            border: 1px solid #2D3748 !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
        }

        /* ══════════════════════════════════════════════════
           PROGRESS BAR
        ══════════════════════════════════════════════════ */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
            border-radius: 9999px !important;
        }
        .stProgress > div > div {
            background: #2D3748 !important;
            border-radius: 9999px !important;
        }

        /* ══════════════════════════════════════════════════
           CHAT
        ══════════════════════════════════════════════════ */
        [data-testid="stChatMessage"] {
            background: #1A1D2E !important;
            border: 1px solid #2D3748 !important;
            border-radius: 14px !important;
        }
        [data-testid="stChatMessage"] * { color: #E2E8F0 !important; }
        [data-testid="stChatInput"] textarea {
            background: #1A1D2E !important;
            color: #E2E8F0 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 12px !important;
        }

        /* ══════════════════════════════════════════════════
           CHECKBOXES & RADIO
        ══════════════════════════════════════════════════ */
        .stCheckbox label, .stRadio label { color: #E2E8F0 !important; }

        /* ══════════════════════════════════════════════════
           ALERTS
        ══════════════════════════════════════════════════ */
        [data-testid="stInfo"]    * { color: #3B82F6 !important; }
        [data-testid="stSuccess"] * { color: #10B981 !important; }
        [data-testid="stWarning"] * { color: #F59E0B !important; }
        [data-testid="stError"]   * { color: #EF4444 !important; }

        /* ══════════════════════════════════════════════════
           MOBILE  (≤ 768px)
        ══════════════════════════════════════════════════ */
        @media (max-width: 768px) {
            .block-container {
                padding: 0.8rem 0.8rem 1.5rem 0.8rem !important;
            }
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            [data-testid="stHorizontalBlock"] > div {
                width: 100% !important;
                min-width: 100% !important;
            }
            .stButton > button {
                width: 100% !important;
                padding: 0.6rem 1rem !important;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 0.75rem !important;
                padding: 0.3rem 0.5rem !important;
            }
            /* Keep the sidebar toggle accessible on mobile */
            [data-testid="collapsedControl"],
            [data-testid="baseButton-headerNoPadding"] {
                width: 36px !important;
                height: 36px !important;
                top: 12px !important;
                transform: none !important;
            }
        }

        @media (max-width: 480px) {
            .block-container { padding: 0.5rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()


# ─────────────────────────── Sidebar ─────────────────────────────────
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

with st.sidebar:
    # ── Logo ──
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.4rem 0">
            <div style="font-size:2.2rem;line-height:1">🤖</div>
            <div style="color:#6366F1;font-size:1.15rem;font-weight:800;
                        letter-spacing:0.04em;margin-top:0.35rem">
                AutoML Platform
            </div>
            <div style="color:#475569;font-size:0.68rem;margin-top:0.15rem">
                FLAML · SHAP · Mistral AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation ──
    st.markdown(
        """
        <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:0.3rem;padding-left:0.2rem">
            Navigation
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_page = st.radio(
        "nav",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav_page",
    )

    # ── Session status ──
    has_data  = st.session_state.get("df") is not None
    has_model = st.session_state.get("automl_result") is not None

    st.markdown(
        f"""
        <div style="margin-top:1rem;padding:0.65rem 0.8rem;
                    background:#141624;border:1px solid #1E2030;
                    border-radius:10px">
            <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;
                        letter-spacing:0.07em;margin-bottom:0.4rem">Session</div>
            <div style="display:flex;gap:0.35rem;flex-wrap:wrap">
                <span style="
                    background:{'rgba(16,185,129,0.12)' if has_data else '#1A1D2E'};
                    color:{'#10B981' if has_data else '#475569'};
                    border:1px solid {'#10B98144' if has_data else '#2D3748'};
                    padding:0.15rem 0.55rem;border-radius:9999px;
                    font-size:0.68rem;font-weight:600">
                    {'✓ Dataset' if has_data else '○ No Data'}
                </span>
                <span style="
                    background:{'rgba(139,92,246,0.12)' if has_model else '#1A1D2E'};
                    color:{'#8B5CF6' if has_model else '#475569'};
                    border:1px solid {'#8B5CF644' if has_model else '#2D3748'};
                    padding:0.15rem 0.55rem;border-radius:9999px;
                    font-size:0.68rem;font-weight:600">
                    {'✓ Model' if has_model else '○ No Model'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Footer ──
    st.markdown(
        """
        <div style="margin-top:1.2rem;padding:0.6rem 0.8rem;
                    background:#141624;border:1px solid #1E2030;
                    border-radius:10px;font-size:0.67rem;color:#475569">
            <div style="color:#6366F1;font-weight:700;font-size:0.74rem">
                🤖 AutoML Platform v1.0.0
            </div>
            <div style="margin-top:0.2rem">MIT License</div>
            <a href="https://github.com/Pranish-Sapkota/automl-platform"
               style="color:#6366F1;text-decoration:none;font-size:0.7rem">
                GitHub ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────── Render page ─────────────────────────────
try:
    PAGES[selected_page].render()
except Exception as exc:
    st.error(f"❌ Page error: {exc}")
    logger.error("Page error '%s': %s", selected_page, exc, exc_info=True)
    with st.expander("🔧 Error Details"):
        import traceback
        st.code(traceback.format_exc())
