"""Upload Dataset page."""
from __future__ import annotations

import streamlit as st

from src.components import page_header, section_header, info_box, metric_row, styled_dataframe
from src.services.data_service import DataService
from src.utils import get_logger

logger = get_logger(__name__)
_svc = DataService()

SAMPLE_DATASETS = {
    "Iris (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic (Classification)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Wine Quality (Regression)": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
}


def render() -> None:
    page_header("📂 Upload Dataset", "Upload a CSV, Excel, Parquet, or JSON file to get started.")

    tab1, tab2 = st.tabs(["📤 Upload File", "🌐 Load Sample Dataset"])

    with tab1:
        _upload_tab()

    with tab2:
        _sample_tab()

    # Show loaded dataset preview
    df = st.session_state.get("df")
    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Dataset Preview", f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns", "👁️")

        metric_row([
            {"label": "Rows", "value": f"{df.shape[0]:,}", "icon": "📋"},
            {"label": "Columns", "value": f"{df.shape[1]:,}", "icon": "🏷️"},
            {"label": "Missing %", "value": f"{df.isnull().mean().mean() * 100:.1f}%", "icon": "❓"},
            {"label": "Duplicates", "value": f"{df.duplicated().sum():,}", "icon": "🔁"},
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        styled_dataframe(df.head(100), height=350, key="upload_preview")

        st.markdown("<br>", unsafe_allow_html=True)
        _target_selector(df)


def _upload_tab() -> None:
    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "xlsx", "xls", "parquet", "json"],
        help="Maximum file size: 200 MB",
    )

    if uploaded is not None:
        with st.spinner("Loading dataset…"):
            try:
                df = _svc.load_dataframe(uploaded)
                st.session_state["df"] = df
                st.session_state["dataset_name"] = uploaded.name
                # Reset downstream state
                for key in ["profile", "preprocessor", "automl_result", "shap_result"]:
                    st.session_state.pop(key, None)
                info_box(f"✅ Loaded <b>{uploaded.name}</b> — {df.shape[0]:,} rows × {df.shape[1]:,} columns", "success")
            except Exception as exc:
                info_box(f"Failed to load file: {exc}", "error")
                logger.error("Upload error: %s", exc)


def _sample_tab() -> None:
    import pandas as pd

    st.markdown("Load a built-in sample dataset to explore the platform quickly.")
    choice = st.selectbox("Select a sample dataset", list(SAMPLE_DATASETS.keys()))

    if st.button("Load Sample", type="primary"):
        url = SAMPLE_DATASETS[choice]
        with st.spinner(f"Downloading {choice}…"):
            try:
                sep = ";" if "wine" in url else ","
                df = pd.read_csv(url, sep=sep)
                st.session_state["df"] = df
                st.session_state["dataset_name"] = choice
                for key in ["profile", "preprocessor", "automl_result", "shap_result"]:
                    st.session_state.pop(key, None)
                info_box(f"✅ Loaded <b>{choice}</b> — {df.shape[0]:,} rows × {df.shape[1]:,} columns", "success")
                st.rerun()
            except Exception as exc:
                info_box(f"Failed to load sample: {exc}", "error")


def _target_selector(df) -> None:
    section_header("Target Column Selection", "Choose the variable you want to predict", "🎯")

    col1, col2 = st.columns([2, 1])
    with col1:
        current_target = st.session_state.get("target_column", df.columns[-1])
        if current_target not in df.columns:
            current_target = df.columns[-1]

        target = st.selectbox(
            "Target Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(current_target),
            help="The column you want the model to predict",
        )
        st.session_state["target_column"] = target

    with col2:
        from src.utils import infer_problem_type
        if target:
            pt = infer_problem_type(df[target])
            icons = {
                "binary_classification": "🔵 Binary Classification",
                "multiclass_classification": "🟣 Multi-Class Classification",
                "regression": "📈 Regression",
            }
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Detected:** {icons.get(pt, pt)}")
            st.session_state["problem_type"] = pt

    if st.button("✅ Confirm Dataset & Target", type="primary"):
        info_box(
            f"Target set to <b>{target}</b>. You can now proceed to <b>Data Profiling</b>.",
            "success",
        )
