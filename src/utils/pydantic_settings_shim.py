"""
Lightweight BaseSettings shim so config.py works without pydantic-settings installed.
Not needed when pydantic-settings is in requirements.txt, but kept as a fallback.
"""
from pydantic import BaseModel


class BaseSettings(BaseModel):
    """Minimal BaseSettings replacement using Pydantic BaseModel."""

    model_config = {"extra": "ignore", "arbitrary_types_allowed": True}
