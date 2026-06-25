"""Tests for SQLite storage layer."""
from __future__ import annotations

import pytest

from src.storage.database import (
    init_db,
    create_experiment,
    update_experiment_status,
    save_model_record,
    get_experiments,
    get_models_for_experiment,
    save_chat_message,
    get_chat_history,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Redirect DB to a temp path for each test."""
    import src.storage.database as db_module
    db_module.DB_PATH = tmp_path / "test_automl.db"
    init_db()
    yield


class TestExperiments:
    def test_create_and_retrieve(self):
        exp_id = create_experiment(
            name="test_exp",
            dataset_name="titanic.csv",
            problem_type="binary_classification",
            target_column="survived",
        )
        assert exp_id is not None
        experiments = get_experiments()
        assert len(experiments) == 1
        assert experiments[0]["name"] == "test_exp"

    def test_update_status(self):
        exp_id = create_experiment("exp2", "data.csv", "regression", "price")
        update_experiment_status(exp_id, "completed", {"r2": 0.92})
        exps = get_experiments()
        assert exps[0]["status"] == "completed"

    def test_multiple_experiments(self):
        for i in range(5):
            create_experiment(f"exp_{i}", "data.csv", "regression", "y")
        exps = get_experiments()
        assert len(exps) == 5

    def test_experiment_ordered_by_created_at_desc(self):
        create_experiment("first", "a.csv", "regression", "y")
        create_experiment("second", "b.csv", "regression", "y")
        exps = get_experiments()
        assert exps[0]["name"] == "second"


class TestModels:
    def test_save_and_retrieve_model(self):
        exp_id = create_experiment("exp", "d.csv", "binary_classification", "target")
        model_id = save_model_record(
            experiment_id=exp_id,
            name="Random Forest",
            algorithm="RandomForestClassifier",
            metrics={"accuracy": 0.95, "f1": 0.94},
            params={"n_estimators": 100},
            artifact_path="/tmp/model.pkl",
        )
        assert model_id is not None
        models = get_models_for_experiment(exp_id)
        assert len(models) == 1
        assert models[0]["name"] == "Random Forest"

    def test_multiple_models_per_experiment(self):
        exp_id = create_experiment("exp", "d.csv", "binary_classification", "target")
        for algo in ["RF", "XGB", "LGBM"]:
            save_model_record(exp_id, algo, algo, {"f1": 0.9}, {}, "/tmp/m.pkl")
        models = get_models_for_experiment(exp_id)
        assert len(models) == 3


class TestChatMessages:
    def test_save_and_retrieve(self):
        exp_id = create_experiment("chat_exp", "data.csv", "regression", "y")
        save_chat_message(exp_id, "user", "Hello!")
        save_chat_message(exp_id, "assistant", "Hi there!")
        msgs = get_chat_history(exp_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_no_experiment_chat(self):
        save_chat_message(None, "user", "test message")
        msgs = get_chat_history(None)
        assert len(msgs) >= 1
