"""Model Export page."""
from __future__ import annotations

import io
import json
import pickle
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from src.components import page_header, section_header, info_box, metric_row
from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


def render() -> None:
    page_header("📦 Model Export", "Download trained models and generate prediction reports.")

    result = st.session_state.get("automl_result")
    if result is None:
        info_box("Please train models first.", "warning")
        return

    section_header("Select Model to Export", "", "🎯")

    model_names = [r.name for r in result.leaderboard]
    sel_model = st.selectbox("Model", model_names)
    selected = next(r for r in result.leaderboard if r.name == sel_model)

    metric_row([
        {"label": "Model", "value": selected.name, "icon": "🧠"},
        {"label": "Algorithm", "value": selected.algorithm, "icon": "⚙️"},
        {"label": "Train Time", "value": f"{selected.train_time:.2f}s", "icon": "⏱️"},
    ] + [
        {"label": k.upper(), "value": f"{v:.4f}", "icon": "📊"}
        for k, v in list(selected.metrics.items())[:3]
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download options ──
    tabs = st.tabs(["📁 Model File", "📊 Metrics Report", "🔮 Batch Predictions", "📋 Experiment Summary"])

    with tabs[0]:
        _model_download_tab(selected)

    with tabs[1]:
        _metrics_report_tab(result, selected)

    with tabs[2]:
        _batch_prediction_tab(result, selected)

    with tabs[3]:
        _experiment_summary_tab(result)


def _model_download_tab(selected) -> None:
    section_header("Download Trained Model", "Export as .pkl file for use in production", "📁")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Details**")
        st.json({
            "name": selected.name,
            "algorithm": selected.algorithm,
            "train_time_seconds": round(selected.train_time, 3),
            "metrics": {k: round(v, 4) for k, v in selected.metrics.items()},
            "is_flaml": selected.is_flaml,
        })

    with col2:
        st.markdown("**Export Options**")

        # Pickle export
        model_bytes = io.BytesIO()
        joblib.dump(selected.model, model_bytes)
        model_bytes.seek(0)

        fname = f"{selected.name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=model_bytes.getvalue(),
            file_name=fname,
            mime="application/octet-stream",
            type="primary",
        )

        # Also export with preprocessor bundled
        preprocessor = st.session_state.get("preprocessor")
        fe = st.session_state.get("feature_engineer")
        if preprocessor:
            bundle = {
                "model": selected.model,
                "preprocessor": preprocessor,
                "feature_engineer": fe,
                "model_name": selected.name,
                "metrics": selected.metrics,
                "feature_names": st.session_state.get("automl_result").feature_names,
            }
            bundle_bytes = io.BytesIO()
            joblib.dump(bundle, bundle_bytes)
            bundle_bytes.seek(0)

            bundle_fname = f"pipeline_{fname}"
            st.download_button(
                label="⬇️ Download Full Pipeline (.pkl)",
                data=bundle_bytes.getvalue(),
                file_name=bundle_fname,
                mime="application/octet-stream",
            )
            st.caption("Includes preprocessor + feature engineering for end-to-end inference")

    # Usage code
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Usage in Python**")
    st.code(f"""
import joblib
import pandas as pd

# Load model
model = joblib.load("{fname}")

# Load full pipeline (preprocessor + model)
pipeline = joblib.load("pipeline_{fname}")
preprocessor = pipeline["preprocessor"]
feature_engineer = pipeline["feature_engineer"]
model = pipeline["model"]

# Predict on new data
new_data = pd.read_csv("new_data.csv")

# Option A: Using full pipeline
df_fe = feature_engineer.transform(new_data)
X, _ = preprocessor.transform(df_fe)
predictions = model.predict(X)

# Option B: Using model directly (if already preprocessed)
predictions = model.predict(X_preprocessed)
print(predictions)
""", language="python")


def _metrics_report_tab(result, selected) -> None:
    section_header("Metrics Report", "Full leaderboard and best model metrics as CSV", "📊")

    records = []
    for r in result.leaderboard:
        row = {"Model": r.name, "Algorithm": r.algorithm, "Train_Time_s": round(r.train_time, 3)}
        row.update({k: round(v, 6) for k, v in r.metrics.items()})
        records.append(row)

    lb_df = pd.DataFrame(records)

    st.dataframe(lb_df, use_container_width=True)

    csv = lb_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Leaderboard CSV",
        data=csv,
        file_name=f"leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary",
    )

    # JSON report
    report = {
        "experiment": {
            "problem_type": result.problem_type,
            "training_time_s": result.training_time,
            "models_trained": len(result.leaderboard),
        },
        "best_model": {
            "name": result.best_model_name,
            "algorithm": selected.algorithm,
            "metrics": {k: round(v, 6) for k, v in selected.metrics.items()},
        },
        "leaderboard": records,
    }
    json_str = json.dumps(report, indent=2)
    st.download_button(
        label="⬇️ Download Report JSON",
        data=json_str,
        file_name=f"automl_report_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
    )


def _batch_prediction_tab(result, selected) -> None:
    section_header("Batch Predictions", "Upload new data to generate predictions", "🔮")

    uploaded = st.file_uploader("Upload prediction data (CSV)", type=["csv"])

    if uploaded is not None:
        try:
            import pandas as pd
            new_df = pd.read_csv(uploaded)
            st.dataframe(new_df.head(), use_container_width=True)

            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating predictions…"):
                    preprocessor = st.session_state.get("preprocessor")
                    fe = st.session_state.get("feature_engineer")

                    df_to_predict = new_df.copy()

                    if fe:
                        df_to_predict = fe.transform(df_to_predict)

                    if preprocessor:
                        X_new, _ = preprocessor.transform(df_to_predict)
                    else:
                        X_new = df_to_predict

                    preds = selected.model.predict(X_new)
                    result_df = new_df.copy()
                    result_df["prediction"] = preds

                    if hasattr(selected.model, "predict_proba"):
                        try:
                            probas = selected.model.predict_proba(X_new)
                            for i, col in enumerate(probas.T):
                                result_df[f"prob_class_{i}"] = col.round(4)
                        except Exception:
                            pass

                    st.success(f"✅ Generated {len(preds):,} predictions!")
                    st.dataframe(result_df.head(50), use_container_width=True)

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary",
                    )
        except Exception as exc:
            info_box(f"Prediction failed: {exc}", "error")


def _experiment_summary_tab(result) -> None:
    section_header("Experiment Summary", "", "📋")

    preprocessor = st.session_state.get("preprocessor")
    profile = st.session_state.get("profile")

    summary = {
        "experiment": {
            "dataset": st.session_state.get("dataset_name", "unknown"),
            "target": st.session_state.get("target_column", "unknown"),
            "problem_type": result.problem_type,
            "training_time_s": round(result.training_time, 2),
        },
        "dataset": {
            "shape": list(profile.shape) if profile else "unknown",
            "features": len(result.feature_names),
            "problem_type": profile.problem_type if profile else "unknown",
        } if profile else {},
        "preprocessing": {
            "steps": preprocessor.report.steps_applied if preprocessor else [],
            "dropped_columns": preprocessor.report.dropped_columns if preprocessor else [],
            "encoded_columns": preprocessor.report.encoded_columns if preprocessor else [],
        } if preprocessor else {},
        "models": [
            {
                "name": r.name,
                "algorithm": r.algorithm,
                "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
            }
            for r in result.leaderboard
        ],
        "best_model": result.best_model_name,
    }

    st.json(summary)

    json_str = json.dumps(summary, indent=2)
    st.download_button(
        "⬇️ Download Experiment Summary",
        data=json_str,
        file_name=f"experiment_summary_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        type="primary",
    )
