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
        "Get Help": "https://github.com/your-username/automl-platform",
        "Report a bug": "https://github.com/your-username/automl-platform/issues",
        "About": "AutoML Platform — Powered by FLAML, SHAP & Mistral AI",
    },
)

# ── Imports after page config ─────────────────────────────────────────
from src.storage import init_db
from src.components import render_sidebar_nav, session_state_summary
from src.pages import home, upload, profiling, cleaning, training, leaderboard, explainability, chat, export
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
        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0F1117; }
        ::-webkit-scrollbar-thumb { background: #6366F1; border-radius: 3px; }

        /* ── Hide Streamlit chrome ── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        /* ── Main container padding ── */
        .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #0D0F1A !important;
            border-right: 1px solid #1E2030;
        }

        /* ── Buttons ── */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            border: none !important;
            color: white !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: #1A1D2E !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            color: #64748B !important;
            font-weight: 500 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #6366F1 !important;
            color: white !important;
        }

        /* ── Cards ── */
        [data-testid="stMetric"] {
            background: #1A1D2E !important;
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }

        /* ── Selectbox, text input ── */
        .stSelectbox > div, .stTextInput > div {
            border-radius: 8px !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploadDropzone"] {
            background: #1A1D2E !important;
            border: 2px dashed #6366F1 !important;
            border-radius: 12px !important;
        }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: #1A1D2E !important;
            border-radius: 8px !important;
        }

        /* ── DataFrame ── */
        [data-testid="stDataFrame"] {
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
        }

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {
            background: #1A1D2E !important;
            border-radius: 12px !important;
            border: 1px solid #2D3748 !important;
            padding: 0.5rem !important;
        }

        /* ── Progress bar ── */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
        }

        /* ── Sidebar nav items ── */
        [data-testid="stSidebarNav"] a {
            color: #94A3B8 !important;
            border-radius: 8px !important;
            padding: 0.5rem 0.75rem !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(99,102,241,0.15) !important;
            color: #6366F1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_inject_css()

# ── Navigation (Settings removed) ────────────────────────────────────
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
        <div style="padding:0.5rem 0;color:#475569;font-size:0.7rem">
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
