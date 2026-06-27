"""Central configuration management using Pydantic v2."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class MLConfig(BaseModel):
    time_budget: int = Field(default=120)
    max_iter: int = Field(default=50)
    n_jobs: int = Field(default=-1)
    cv_folds: int = Field(default=5)
    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    estimator_list: list[str] = Field(
        default=["rf", "extra_tree", "xgboost", "lgbm", "catboost", "lrl1", "dt"]
    )


class MistralConfig(BaseModel):
    model: str = Field(default="mistral-small-2506")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.3)
    api_key: Optional[str] = Field(default=None)


class AppConfig(BaseModel):
    app_name: str = "AutoML Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    max_rows: int = 500_000
    max_columns: int = 500
    base_dir: Path = Field(default=BASE_DIR)
    data_dir: Path = Field(default=DATA_DIR)
    models_dir: Path = Field(default=MODELS_DIR)
    reports_dir: Path = Field(default=REPORTS_DIR)
    ml: MLConfig = Field(default_factory=MLConfig)
    mistral: MistralConfig = Field(default_factory=MistralConfig)

    model_config = {"arbitrary_types_allowed": True}


def _get_mistral_api_key() -> Optional[str]:
    """
    Load API key at CALL TIME (not import time) so st.secrets is always ready.
    Priority: Streamlit secrets → environment variable.
    """
    # 1. Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        key = st.secrets.get("MISTRAL_API_KEY", None)
        if key:
            return str(key)
    except Exception:
        pass

    # 2. Environment variable (local dev)
    return os.environ.get("MISTRAL_API_KEY") or None


def get_config() -> AppConfig:
    """Build config — always resolves secrets at runtime, not import time."""
    mistral_cfg = MistralConfig(
        api_key=_get_mistral_api_key(),
        model=os.environ.get("MISTRAL_MODEL", "mistral-small-2506"),
    )
    return AppConfig(mistral=mistral_cfg)


# Module-level CONFIG is used for non-secret settings only.
# MistralClient always calls get_config() fresh to pick up st.secrets.
CONFIG: AppConfig = get_config()
