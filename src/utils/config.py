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
    """ML training configuration."""
    time_budget: int = Field(default=120, description="FLAML time budget in seconds")
    max_iter: int = Field(default=50, description="Max FLAML iterations")
    n_jobs: int = Field(default=-1, description="Parallel jobs")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    test_size: float = Field(default=0.2, description="Test split ratio")
    random_state: int = Field(default=42, description="Random seed")
    estimator_list: list[str] = Field(
        default=["rf", "extra_tree", "xgboost", "lgbm", "catboost", "lrl1", "dt"],
        description="Models to train"
    )


class MistralConfig(BaseModel):
    """Mistral AI configuration."""
    model: str = Field(default="mistral-large-latest", description="Mistral model")
    max_tokens: int = Field(default=2048, description="Max response tokens")
    temperature: float = Field(default=0.3, description="Generation temperature")
    api_key: Optional[str] = Field(default=None, description="API key from env")


class AppConfig(BaseModel):
    """Application configuration."""
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


def get_config() -> AppConfig:
    """Get application configuration with environment overrides."""
    mistral_cfg = MistralConfig(
        api_key=os.environ.get("MISTRAL_API_KEY"),
        model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
    )
    return AppConfig(mistral=mistral_cfg)


CONFIG: AppConfig = get_config()
