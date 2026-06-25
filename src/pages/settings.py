"""Settings page."""
from __future__ import annotations

import streamlit as st

from src.components import page_header, section_header, info_box, metric_row
from src.storage import get_experiments
from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


def render() -> None:
    page_header("⚙️ Settings", "Configure API keys, preferences, and view session info.")

    tabs = st.tabs(["🔑 API Keys", "🧠 ML Config", "📊 Session Info", "🗃️ Experiment History"])

    with tabs[0]:
        _api_keys_tab()

    with tabs[1]:
        _ml_config_tab()

    with tabs[2]:
        _session_tab()

    with tabs[3]:
        _history_tab()


def _api_keys_tab() -> None:
    section_header("Mistral AI API Key", "", "🔑")

    import os
    existing = st.session_state.get("mistral_api_key") or os.environ.get("MISTRAL_API_KEY", "")

    st.markdown(
        "Get your free API key at [console.mistral.ai](https://console.mistral.ai/). "
        "The key is stored **only in your browser session** — it is never saved to disk."
    )

    api_key = st.text_input(
        "Mistral API Key",
        value=existing,
        type="password",
        placeholder="Enter your Mistral API key…",
    )

    model_choice = st.selectbox(
        "Mistral Model",
        ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "open-mistral-7b"],
        help="mistral-large-latest gives best results; open-mistral-7b is free",
    )

    if st.button("💾 Save API Key", type="primary"):
        st.session_state["mistral_api_key"] = api_key
        CONFIG.mistral.api_key = api_key
        CONFIG.mistral.model = model_choice
        if api_key:
            info_box("✅ API key saved for this session.", "success")
        else:
            info_box("API key cleared.", "warning")

    # Test connection
    if existing and st.button("🧪 Test Connection"):
        from src.ai import MistralClient
        client = MistralClient(api_key=existing)
        with st.spinner("Testing Mistral connection…"):
            resp = client.chat([{"role": "user", "content": "Reply with: OK"}])
        if "OK" in resp or len(resp) > 0:
            info_box("✅ Connection successful!", "success")
        else:
            info_box(f"Connection test returned: {resp}", "warning")


def _ml_config_tab() -> None:
    section_header("Machine Learning Configuration", "", "🧠")

    col1, col2 = st.columns(2)

    with col1:
        time_budget = st.slider(
            "Default FLAML Time Budget (s)",
            30, 600,
            st.session_state.get("default_time_budget", 60),
        )
        test_size = st.slider(
            "Default Test Split %",
            10, 40,
            st.session_state.get("default_test_size", 20),
        )

    with col2:
        cv_folds = st.selectbox(
            "Default CV Folds",
            [3, 5, 10],
            index=1,
        )
        random_state = st.number_input(
            "Global Random Seed",
            value=st.session_state.get("random_state", 42),
        )

    if st.button("💾 Save ML Config", type="primary"):
        st.session_state["default_time_budget"] = time_budget
        st.session_state["default_test_size"] = test_size
        st.session_state["default_cv_folds"] = cv_folds
        st.session_state["random_state"] = int(random_state)
        info_box("✅ ML configuration saved.", "success")


def _session_tab() -> None:
    section_header("Current Session", "", "📊")

    df = st.session_state.get("df")
    result = st.session_state.get("automl_result")
    profile = st.session_state.get("profile")

    metrics = [
        {"label": "Dataset", "value": "✅ Loaded" if df is not None else "❌ None", "icon": "📋"},
        {"label": "Rows", "value": f"{df.shape[0]:,}" if df is not None else "—", "icon": "🔢"},
        {"label": "Preprocessed", "value": "✅" if st.session_state.get("X_processed") is not None else "❌", "icon": "⚙️"},
        {"label": "Models Trained", "value": len(result.leaderboard) if result else "0", "icon": "🧠"},
    ]
    metric_row(metrics)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Session (Reset All)", type="secondary"):
        keys_to_clear = [
            "df", "target_column", "problem_type", "profile",
            "preprocessor", "feature_engineer", "X_processed", "y_processed",
            "automl_result", "X_train", "X_test", "y_train", "y_test",
            "experiment_id", "preproc_result", "chat_messages",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        # Clear SHAP caches
        shap_keys = [k for k in st.session_state if k.startswith("shap_")]
        for k in shap_keys:
            st.session_state.pop(k, None)
        st.success("Session cleared!")
        st.rerun()

    # Raw session state (debug)
    with st.expander("🔧 Raw Session State (Debug)"):
        safe_state = {
            k: (str(v)[:100] if not isinstance(v, (int, float, str, bool, list, dict)) else v)
            for k, v in st.session_state.items()
            if not k.startswith("shap_")
        }
        st.json(safe_state)


def _history_tab() -> None:
    section_header("Experiment History", "All past training runs (SQLite)", "🗃️")

    import pandas as pd

    try:
        experiments = get_experiments()
        if not experiments:
            info_box("No experiments found.", "info")
            return

        records = []
        for exp in experiments:
            import json as _json
            metrics = _json.loads(exp.get("metrics") or "{}")
            records.append({
                "Name": exp["name"],
                "Dataset": exp.get("dataset_name", "—"),
                "Problem": exp.get("problem_type", "—"),
                "Target": exp.get("target_column", "—"),
                "Status": exp.get("status", "—"),
                "Created": exp["created_at"][:19],
                "Best Metric": str(list(metrics.values())[0])[:8] if metrics else "—",
            })

        exp_df = pd.DataFrame(records)
        st.dataframe(exp_df, use_container_width=True, height=400)

    except Exception as exc:
        info_box(f"Could not load history: {exc}", "error")
