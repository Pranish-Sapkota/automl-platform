"""AI Chat Assistant page (Mistral AI)."""
from __future__ import annotations

import json

import streamlit as st

from src.ai import MistralClient
from src.components import page_header, section_header, info_box
from src.storage import save_chat_message, get_chat_history
from src.utils import get_logger

logger = get_logger(__name__)

SUGGESTED_PROMPTS = [
    "Summarize this dataset for me.",
    "Why might this model be performing poorly?",
    "What are the most important features and why?",
    "How can I improve model accuracy?",
    "Are there any data quality issues I should address?",
    "What business insights can you draw from this data?",
    "Explain what the ROC-AUC score means for my model.",
    "Should I be concerned about class imbalance?",
]


def render() -> None:
    page_header(
        "💬 AI Chat Assistant",
        "Ask questions about your dataset and model using Mistral AI.",
    )

    # API key is loaded automatically from Streamlit secrets or environment variable.
    # No manual entry needed — it is set securely in Streamlit Cloud secrets.
    client = MistralClient()

    if not client.is_configured():
        info_box(
            "Mistral API key is not configured. Please add <code>MISTRAL_API_KEY</code> "
            "to your Streamlit Cloud secrets.",
            "warning",
        )

    # ── Context panel ──
    with st.expander("📋 Current Session Context", expanded=False):
        _show_context()

    # ── Quick prompts ──
    section_header("Suggested Questions", "", "💡")
    cols = st.columns(4)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 4]:
            if st.button(prompt[:40] + ("…" if len(prompt) > 40 else ""), key=f"qp_{i}"):
                st.session_state["pending_prompt"] = prompt

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat interface ──
    section_header("Chat", "", "💬")

    # Initialise chat history in session
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Display history
    _render_chat_history()

    # Input
    pending = st.session_state.pop("pending_prompt", None)
    user_input = st.chat_input("Ask anything about your dataset or model…")

    message_to_send = pending or user_input
    if message_to_send:
        _handle_message(message_to_send, client)

    # ── Actions ──
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

    with col2:
        if st.button("📊 Generate EDA Summary"):
            _generate_eda_summary(client)

    with col3:
        if st.button("💡 Generate Business Insights"):
            _generate_business_insights(client)


def _render_chat_history() -> None:
    messages = st.session_state.get("chat_messages", [])
    for msg in messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])


def _handle_message(message: str, client: MistralClient) -> None:
    # Add user message
    st.session_state["chat_messages"].append({"role": "user", "content": message})

    # Build context
    system_prompt = client.build_system_prompt(
        df_info=_get_df_context(),
        model_info=_get_model_context(),
    )

    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            if client.is_configured():
                for chunk in client.stream_chat(
                    messages=st.session_state["chat_messages"],
                    system_prompt=system_prompt,
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            else:
                full_response = (
                    "⚠️ **Mistral API key not set.** "
                    "Please add `MISTRAL_API_KEY` to Streamlit Cloud secrets."
                )
                response_placeholder.markdown(full_response)
        except Exception as exc:
            full_response = f"❌ Error: {str(exc)}"
            response_placeholder.markdown(full_response)
            logger.error("Chat error: %s", exc)

    st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})

    # Persist to DB
    exp_id = st.session_state.get("experiment_id")
    save_chat_message(exp_id, "user", message)
    save_chat_message(exp_id, "assistant", full_response)

    st.rerun()


def _generate_eda_summary(client: MistralClient) -> None:
    profile = st.session_state.get("profile")
    if not profile:
        info_box("Run Data Profiling first to generate an EDA summary.", "warning")
        return

    profile_dict = {
        "shape": profile.shape,
        "problem_type": profile.problem_type,
        "missing_report": profile.missing_report,
        "duplicate_report": profile.duplicate_report,
        "outlier_cols": list(profile.outlier_report.keys())[:10],
        "numeric_columns": profile.numeric_columns[:15],
        "categorical_columns": profile.categorical_columns[:10],
        "recommendations": profile.recommendations,
    }

    with st.spinner("Generating EDA summary…"):
        summary = client.generate_eda_summary(profile_dict)

    msg = f"📊 **EDA Summary:**\n\n{summary}"
    st.session_state["chat_messages"].append({"role": "assistant", "content": msg})
    st.rerun()


def _generate_business_insights(client: MistralClient) -> None:
    result = st.session_state.get("automl_result")
    if not result:
        info_box("Train models first to generate business insights.", "warning")
        return

    best = result.leaderboard[0]
    top_features = list(
        sorted(best.metrics.items(), key=lambda x: abs(x[1]), reverse=True)
    )[:5]

    df_info = _get_df_context()
    with st.spinner("Generating business insights…"):
        insights = client.generate_business_insights(
            df_info=df_info,
            top_features=[str(f) for f in top_features],
        )

    msg = f"💡 **Business Insights:**\n\n{insights}"
    st.session_state["chat_messages"].append({"role": "assistant", "content": msg})
    st.rerun()


def _get_df_context() -> dict:
    profile = st.session_state.get("profile")
    df = st.session_state.get("df")
    if profile:
        return {
            "shape": profile.shape,
            "problem_type": profile.problem_type,
            "target_column": st.session_state.get("target_column"),
            "numeric_columns": profile.numeric_columns[:10],
            "categorical_columns": profile.categorical_columns[:10],
            "missing_pct": profile.missing_report.get("pct_missing", 0),
            "duplicate_rows": profile.duplicate_report.get("duplicate_rows", 0),
            "recommendations": profile.recommendations,
        }
    elif df is not None:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist()[:20],
            "dtypes": {col: str(df[col].dtype) for col in df.columns[:10]},
        }
    return {}


def _get_model_context() -> dict:
    result = st.session_state.get("automl_result")
    if not result:
        return {}
    best = result.leaderboard[0]
    return {
        "problem_type": result.problem_type,
        "best_model": best.name,
        "best_metrics": best.metrics,
        "models_trained": len(result.leaderboard),
        "training_time_s": result.training_time,
        "all_models": [r.name for r in result.leaderboard],
        "feature_count": len(result.feature_names),
        "top_features": result.feature_names[:10],
    }


def _show_context() -> None:
    df_ctx = _get_df_context()
    model_ctx = _get_model_context()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Context**")
        if df_ctx:
            st.json(df_ctx)
        else:
            st.info("No dataset loaded.")

    with col2:
        st.markdown("**Model Context**")
        if model_ctx:
            st.json(model_ctx)
        else:
            st.info("No model trained.")
