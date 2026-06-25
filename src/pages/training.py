"""AutoML Training page."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.automl import AutoMLEngine
from src.components import page_header, section_header, info_box, metric_row
from src.storage import create_experiment, update_experiment_status, save_model_record
from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


def render() -> None:
    page_header("🧪 AutoML Training", "Configure and run the full AutoML pipeline.")

    # Check prerequisites
    if st.session_state.get("df") is None:
        info_box("Please upload a dataset first.", "warning")
        return

    X = st.session_state.get("X_processed")
    y = st.session_state.get("y_processed")

    if X is None or y is None:
        info_box("Please run Data Preprocessing first.", "warning")
        return

    target = st.session_state.get("target_column", "target")
    problem_type = st.session_state.get("problem_type", "binary_classification")
    preprocessor = st.session_state.get("preprocessor")
    target_classes = preprocessor.target_classes if preprocessor else None

    # ── Configuration ──
    section_header("Training Configuration", "", "⚙️")

    col1, col2, col3 = st.columns(3)
    with col1:
        time_budget = st.slider(
            "FLAML Time Budget (seconds)",
            min_value=30, max_value=600, value=60, step=30,
            help="How long FLAML searches for the best hyperparameters",
        )
        test_size = st.slider("Test Split %", min_value=10, max_value=40, value=20) / 100

    with col2:
        cv_folds = st.selectbox("CV Folds", [3, 5, 10], index=1)
        random_state = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

    with col3:
        use_boosting = st.checkbox("Include XGBoost / LightGBM / CatBoost", value=True,
                                   help="Adds ~60s but often improves results")
        exp_name = st.text_input(
            "Experiment Name",
            value=f"exp_{int(time.time())}",
            help="Name for this training run",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset summary ──
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Features", X.shape[1])
    col_b.metric("Samples", f"{X.shape[0]:,}")
    col_c.metric("Problem Type", problem_type.replace("_", " ").title())
    col_d.metric("Target", target)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Train button ──
    if st.button("🚀 Start AutoML Training", type="primary"):
        _run_training(
            X=X, y=y,
            problem_type=problem_type,
            time_budget=time_budget,
            test_size=test_size,
            cv_folds=cv_folds,
            random_state=int(random_state),
            use_boosting=use_boosting,
            exp_name=exp_name,
            target=target,
            target_classes=target_classes,
        )

    # ── Show results summary if available ──
    result = st.session_state.get("automl_result")
    if result:
        _show_training_summary(result)


def _run_training(
    X, y, problem_type, time_budget, test_size, cv_folds,
    random_state, use_boosting, exp_name, target, target_classes
) -> None:
    progress_bar = st.progress(0, text="Initialising…")
    status_text = st.empty()

    def progress_cb(pct: float, msg: str) -> None:
        progress_bar.progress(min(pct, 1.0), text=msg)
        status_text.text(f"⏳ {msg}")

    try:
        # Create experiment in DB
        dataset_name = st.session_state.get("dataset_name", "unknown")
        exp_id = create_experiment(
            name=exp_name,
            dataset_name=dataset_name,
            problem_type=problem_type,
            target_column=target,
            config={"time_budget": time_budget, "test_size": test_size},
        )
        st.session_state["experiment_id"] = exp_id

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if "classification" in problem_type else None,
        )

        # Train
        engine = AutoMLEngine(
            problem_type=problem_type,
            time_budget=time_budget,
            random_state=random_state,
            cv_folds=cv_folds,
        )

        if use_boosting:
            result = engine.train_with_boosting(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                feature_names=X.columns.tolist(),
                target_classes=target_classes,
                progress_callback=progress_cb,
            )
        else:
            result = engine.train(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                feature_names=X.columns.tolist(),
                target_classes=target_classes,
                progress_callback=progress_cb,
            )

        # Store in session
        st.session_state["automl_result"] = result
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["engine"] = engine

        # Persist best model + save records
        model_dir = CONFIG.models_dir / exp_id
        best_metrics = result.leaderboard[0].metrics
        update_experiment_status(exp_id, "completed", best_metrics)

        for mr in result.leaderboard:
            artifact_path = model_dir / f"{mr.name.replace(' ', '_')}.pkl"
            engine.save_model(mr.model, artifact_path)
            save_model_record(
                experiment_id=exp_id,
                name=mr.name,
                algorithm=mr.algorithm,
                metrics=mr.metrics,
                params=mr.params if isinstance(mr.params, dict) else {},
                artifact_path=str(artifact_path),
            )

        progress_bar.progress(1.0, text="✅ Training complete!")
        st.success(
            f"🏆 Training complete! Best model: **{result.best_model_name}** "
            f"in {result.training_time:.1f}s"
        )
        st.rerun()

    except Exception as exc:
        progress_bar.empty()
        info_box(f"Training failed: {exc}", "error")
        logger.error("Training error: %s", exc, exc_info=True)


def _show_training_summary(result) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Training Summary", "", "📊")

    best = result.leaderboard[0]
    is_clf = "classification" in result.problem_type

    top_metrics = []
    if is_clf:
        top_metrics = [
            {"label": "Best Model", "value": best.name, "icon": "🏆"},
            {"label": "Accuracy", "value": f"{best.metrics.get('accuracy', 0):.4f}", "icon": "🎯"},
            {"label": "F1 Score", "value": f"{best.metrics.get('f1', 0):.4f}", "icon": "📊"},
            {"label": "ROC-AUC", "value": f"{best.metrics.get('roc_auc', 0):.4f}", "icon": "📈"},
        ]
    else:
        top_metrics = [
            {"label": "Best Model", "value": best.name, "icon": "🏆"},
            {"label": "RMSE", "value": f"{best.metrics.get('rmse', 0):.4f}", "icon": "📉"},
            {"label": "MAE", "value": f"{best.metrics.get('mae', 0):.4f}", "icon": "📊"},
            {"label": "R²", "value": f"{best.metrics.get('r2', 0):.4f}", "icon": "📈"},
        ]

    metric_row(top_metrics)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"✅ {len(result.leaderboard)} models trained in {result.training_time:.1f}s. "
            "View the full leaderboard on the **Model Leaderboard** page.")
