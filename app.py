"""
AutoML Platform — Main Application Entry Point
==============================================
Production-grade AutoML platform deployable on Streamlit Cloud.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure src/ is importable ──────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Pranish-Sapkota/automl-platform",
        "Report a bug": "https://github.com/Pranish-Sapkota/automl-platform/issues",
        "About": "AutoML Platform — Powered by FLAML, SHAP & Mistral AI",
    },
)

# ── Imports after page config ─────────────────────────────────────────
from src.storage import init_db
from src.components import render_sidebar_nav, session_state_summary
from src.pages import home, upload, profiling, cleaning, training, leaderboard, explainability, chat, export, settings
from src.utils import get_logger

logger = get_logger("app")

# ── Initialise persistent storage ────────────────────────────────────
@st.cache_resource
def _init_storage() -> None:
    init_db()
    logger.info("Storage initialised")

_init_storage()

# ── Global CSS ────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Theme-Aware Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--background-color); }
        ::-webkit-scrollbar-thumb { background: var(--primary-color); border-radius: 3px; }

        /* ── Mobile-Safe Header Adjustment ── */
        /* Keeps the hamburger menu button functional on mobile while removing the default app chrome */
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* ── Responsive Container Padding ── */
        .block-container { 
            padding: 1rem 0.75rem 2rem 0.75rem; 
            max-width: 1400px; 
        }
        @media (min-width: 768px) {
            .block-container {
                padding: 1.5rem 2rem 2rem 2rem;
            }
        }

        /* ── Sidebar Custom Styling ── */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-background-color) !important;
            border-right: 1px solid rgba(128, 128, 128, 0.1);
        }

        /* ── Premium UI Buttons ── */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            border: none !important;
            color: white !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
        }

        /* ── Modern Segmented Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--secondary-background-color) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
            border: 1px solid rgba(128, 128, 128, 0.15);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            color: var(--text-color) !important;
            opacity: 0.7;
            font-weight: 500 !important;
            padding: 6px 14px !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary-color) !important;
            color: white !important;
            opacity: 1 !important;
        }

        /* ── Metric Cards ── */
        [data-testid="stMetric"] {
            background: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }

        /* ── Input Element Consistency ── */
        .stSelectbox > div, .stTextInput > div, .stSlider > div {
            border-radius: 8px !important;
        }

        /* ── File Uploader Dropzone ── */
        [data-testid="stFileUploadDropzone"] {
            background: var(--secondary-background-color) !important;
            border: 2px dashed var(--primary-color) !important;
            border-radius: 12px !important;
        }

        /* ── Expanders ── */
        .streamlit-expanderHeader {
            background: var(--secondary-background-color) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
        }

        /* ── DataFrames ── */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-radius: 8px !important;
        }

        /* ── Chat Interface Bubbles ── */
        [data-testid="stChatMessage"] {
            background: var(--secondary-background-color) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            padding: 0.75rem !important;
        }

        /* ── Accent-Aligned Progress Bar ── */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
        }

        /* ── Sidebar Built-in Navigation Items ── */
        [data-testid="stSidebarNav"] a {
            color: var(--text-color) !important;
            opacity: 0.8;
            border-radius: 8px !important;
            padding: 0.5rem 0.75rem !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(99, 102, 241, 0.12) !important;
            color: var(--primary-color) !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_inject_css()

# ── Navigation ────────────────────────────────────────────────────────
PAGES = {
    "🏠 Home": home,
    "📂 Upload Dataset": upload,
    "🔍 Data Profiling": profiling,
    "⚙️ Data Cleaning": cleaning,
    "🧪 AutoML Training": training,
    "🏆 Model Leaderboard": leaderboard,
    "🔮 Explainability": explainability,
    "💬 AI Chat Assistant": chat,
    "📦 Model Export": export,
    "⚙️ Settings": settings,
}

# ── Sidebar ───────────────────────────────────────────────────────────
render_sidebar_nav()

with st.sidebar:
    st.markdown("### Navigation")
    selected_page = st.radio(
        "Navigate to",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav_page",
    )
    session_state_summary()

    st.markdown("---")
    st.markdown(
        """
        <div style="padding:0.5rem 0; opacity: 0.7; font-size:0.75rem">
            <div>🤖 <b>AutoML Platform</b> v1.0.0</div>
            <div style="margin-top:0.2rem">FLAML · SHAP · Mistral AI</div>
            <div style="margin-top:0.2rem">SQLite · Streamlit Cloud</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render selected page ──────────────────────────────────────────────
page_module = PAGES[selected_page]
try:
    page_module.render()
except Exception as exc:
    st.error(f"❌ Page error: {exc}")
    logger.error("Page render error on '%s': %s", selected_page, exc, exc_info=True)
    with st.expander("🔧 Error Details"):
        import traceback
        st.code(traceback.format_exc())
