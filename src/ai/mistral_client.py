"""Mistral AI client for dataset chat assistant."""
from __future__ import annotations

import json
from typing import Any, Generator

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


class MistralClient:
    """Thin wrapper around the Mistral AI API for streaming chat."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or CONFIG.mistral.api_key
        self.model = CONFIG.mistral.model
        self.max_tokens = CONFIG.mistral.max_tokens
        self.temperature = CONFIG.mistral.temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from mistralai import Mistral  # type: ignore
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_system_prompt(
        self,
        df_info: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> str:
        """Build a rich system prompt with dataset and model context."""
        parts = [
            "You are an expert data scientist and ML engineer assistant embedded in an AutoML platform.",
            "Your role is to help users understand their datasets, model results, and feature importance.",
            "Be concise, accurate, and actionable. Use bullet points and markdown when helpful.",
            "Never hallucinate — only reference what you know from the provided context.",
        ]

        if df_info:
            parts.append(f"\n## Dataset Context\n```json\n{json.dumps(df_info, indent=2, default=str)}\n```")

        if model_info:
            parts.append(f"\n## Model Context\n```json\n{json.dumps(model_info, indent=2, default=str)}\n```")

        return "\n".join(parts)

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """Send a chat request and return the full response."""
        if not self.is_configured():
            return (
                "⚠️ **Mistral API key not configured.**\n\n"
                "Please add your `MISTRAL_API_KEY` in the Settings page or as an environment variable."
            )

        try:
            client = self._get_client()
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)

            response = client.chat.complete(
                model=self.model,
                messages=all_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Mistral API error: %s", exc)
            return f"❌ Mistral API error: {str(exc)}"

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream chat tokens as they arrive."""
        if not self.is_configured():
            yield (
                "⚠️ **Mistral API key not configured.**\n\n"
                "Please add your `MISTRAL_API_KEY` in the Settings page."
            )
            return

        try:
            client = self._get_client()
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)

            with client.chat.stream(
                model=self.model,
                messages=all_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ) as stream:
                for chunk in stream:
                    delta = chunk.data.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as exc:
            logger.error("Mistral streaming error: %s", exc)
            yield f"❌ Streaming error: {str(exc)}"

    def generate_eda_summary(self, profile_dict: dict) -> str:
        """Generate a natural-language EDA summary."""
        prompt = f"""
Analyze this dataset profile and provide:
1. A 3-sentence executive summary
2. Top 3 data quality concerns
3. Top 3 ML readiness recommendations

Dataset Profile:
{json.dumps(profile_dict, indent=2, default=str)}
"""
        return self.chat([{"role": "user", "content": prompt}])

    def explain_model_performance(
        self, metrics: dict, problem_type: str, feature_importance: dict
    ) -> str:
        """Explain why a model performed well or poorly."""
        prompt = f"""
Explain this ML model's performance in plain English:

Problem Type: {problem_type}
Metrics: {json.dumps(metrics, indent=2)}
Top Features: {json.dumps(dict(list(feature_importance.items())[:10]), indent=2)}

Provide:
1. Is this good/average/poor performance? Why?
2. What do the metrics tell us?
3. Which features are most influential and why?
4. 3 specific improvement recommendations
"""
        return self.chat([{"role": "user", "content": prompt}])

    def generate_business_insights(self, df_info: dict, top_features: list[str]) -> str:
        """Generate business insights from model features."""
        prompt = f"""
Given this dataset and the most important predictive features, generate actionable business insights:

Dataset: {json.dumps(df_info, indent=2, default=str)}
Most Important Features: {top_features}

Provide:
1. What business question does this model answer?
2. Key drivers of the outcome (in business terms)
3. 3 actionable recommendations for the business
4. Potential risks or biases to watch for
"""
        return self.chat([{"role": "user", "content": prompt}])
