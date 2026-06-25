"""SQLite-based session and experiment storage."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)

DB_PATH = CONFIG.data_dir / "automl.db"


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        dataset_name TEXT,
        problem_type TEXT,
        target_column TEXT,
        status TEXT DEFAULT 'created',
        config JSON,
        metrics JSON,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES experiments(id),
        name TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        created_at TEXT NOT NULL,
        metrics JSON,
        params JSON,
        artifact_path TEXT
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        experiment_id TEXT REFERENCES experiments(id),
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_models_exp ON models(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_chat_exp ON chat_messages(experiment_id);
    """
    with get_connection() as conn:
        conn.executescript(ddl)
    logger.info("Database initialised at %s", DB_PATH)


def create_experiment(
    name: str,
    dataset_name: str,
    problem_type: str,
    target_column: str,
    config: dict[str, Any] | None = None,
) -> str:
    """Insert a new experiment record and return its ID."""
    exp_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO experiments (id, name, created_at, updated_at,
               dataset_name, problem_type, target_column, config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, name, now, now, dataset_name, problem_type, target_column,
             json.dumps(config or {})),
        )
    return exp_id


def update_experiment_status(exp_id: str, status: str, metrics: dict | None = None) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE experiments SET status=?, updated_at=?, metrics=? WHERE id=?",
            (status, now, json.dumps(metrics or {}), exp_id),
        )


def save_model_record(
    experiment_id: str,
    name: str,
    algorithm: str,
    metrics: dict[str, Any],
    params: dict[str, Any],
    artifact_path: str,
) -> str:
    model_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO models (id, experiment_id, name, algorithm, created_at,
               metrics, params, artifact_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, experiment_id, name, algorithm, now,
             json.dumps(metrics), json.dumps(params), artifact_path),
        )
    return model_id


def get_experiments() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_models_for_experiment(exp_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM models WHERE experiment_id=? ORDER BY created_at DESC",
            (exp_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_chat_message(experiment_id: Optional[str], role: str, content: str) -> None:
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, experiment_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (msg_id, experiment_id, role, content, now),
        )


def get_chat_history(experiment_id: Optional[str] = None) -> list[dict]:
    with get_connection() as conn:
        if experiment_id:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE experiment_id=? ORDER BY created_at",
                (experiment_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_messages ORDER BY created_at"
            ).fetchall()
    return [dict(r) for r in rows]
