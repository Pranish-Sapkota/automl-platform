"""Mistral AI client — compatible with mistralai v0.x, v1.x, and direct HTTP."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Generator

from src.utils import get_logger
# NOTE: we import get_config (not CONFIG) so the API key is resolved
# at MistralClient() instantiation time, after st.secrets is loaded.
from src.utils.config import get_config

logger = get_logger(__name__)


def _detect_sdk_version() -> str:
    """Return 'v1', 'v0', or 'none' based on installed mistralai SDK."""
    try:
        from mistralai import Mistral  # noqa: F401
        return "v1"
    except ImportError:
        pass
    try:
        from mistralai.client import MistralClient  # noqa: F401
        return "v0"
    except ImportError:
        pass
    return "none"


SDK_VERSION = _detect_sdk_version()
logger.info("Detected mistralai SDK version: %s", SDK_VERSION)


class MistralClient:
    """
    Thin wrapper around the Mistral AI API.
    Supports mistralai v1.x, v0.x, and a pure-HTTP fallback (no SDK needed).

    API key is resolved fresh on every __init__ call via get_config() so that
    Streamlit secrets are always available by the time the client is created.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # If an explicit key is passed, use it.
        # Otherwise call get_config() fresh — this reads st.secrets at runtime.
        if api_key:
            self.api_key = api_key
            cfg = get_config()
            self.model = cfg.mistral.model
            self.max_tokens = cfg.mistral.max_tokens
            self.temperature = cfg.mistral.temperature
        else:
            cfg = get_config()           # ← fresh call: st.secrets is ready here
            self.api_key = cfg.mistral.api_key
            self.model = cfg.mistral.model
            self.max_tokens = cfg.mistral.max_tokens
            self.temperature = cfg.mistral.temperature

        self._client: Any = None

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if SDK_VERSION == "v1":
            from mistralai import Mistral  # type: ignore
            self._client = Mistral(api_key=self.api_key)
        elif SDK_VERSION == "v0":
            from mistralai.client import MistralClient as _Old  # type: ignore
            self._client = _Old(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Core chat (non-streaming)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        if not self.is_configured():
            return (
                "⚠️ **Mistral API key not configured.**\n\n"
                "Please add `MISTRAL_API_KEY` to your Streamlit Cloud secrets."
            )

        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        try:
            if SDK_VERSION == "v1":
                return self._chat_v1(all_messages)
            elif SDK_VERSION == "v0":
                return self._chat_v0(all_messages)
            else:
                return self._chat_http(all_messages)
        except Exception as exc:
            logger.error("Mistral API error: %s", exc)
            return f"❌ Mistral API error: {str(exc)}"

    def _chat_v1(self, messages: list[dict]) -> str:
        client = self._get_client()
        response = client.chat.complete(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _chat_v0(self, messages: list[dict]) -> str:
        from mistralai.models.chat_completion import ChatMessage  # type: ignore
        client = self._get_client()
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"]) for m in messages
        ]
        response = client.chat(
            model=self.model,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _chat_http(self, messages: list[dict]) -> str:
        """Pure urllib fallback — zero SDK dependency."""
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        if not self.is_configured():
            yield (
                "⚠️ **Mistral API key not configured.**\n\n"
                "Please add `MISTRAL_API_KEY` to your Streamlit Cloud secrets."
            )
            return

        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        try:
            if SDK_VERSION == "v1":
                yield from self._stream_v1(all_messages)
            elif SDK_VERSION == "v0":
                yield from self._stream_v0(all_messages)
            else:
                yield from self._stream_http(all_messages)
        except Exception as exc:
            logger.error("Mistral streaming error: %s", exc)
            try:
                yield self._chat_http(all_messages)
            except Exception as exc2:
                yield f"❌ Error: {str(exc2)}"

    def _stream_v1(self, messages: list[dict]) -> Generator[str, None, None]:
        client = self._get_client()
        with client.chat.stream(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as stream:
            for chunk in stream:
                try:
                    delta = chunk.data.choices[0].delta.content
                    if delta:
                        yield delta
                except (AttributeError, IndexError):
                    continue

    def _stream_v0(self, messages: list[dict]) -> Generator[str, None, None]:
        from mistralai.models.chat_completion import ChatMessage  # type: ignore
        client = self._get_client()
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"]) for m in messages
        ]
        for chunk in client.chat_stream(
            model=self.model,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ):
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            except (AttributeError, IndexError):
                continue

    def _stream_http(self, messages: list[dict]) -> Generator[str, None, None]:
        """SSE streaming via urllib — no SDK needed."""
        import ssl
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception:
            yield self._chat_http(messages)

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        df_info: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> str:
        parts = [
            "You are an expert data scientist and ML engineer assistant embedded in an AutoML platform.",
            "Your role is to help users understand their datasets, model results, and feature importance.",
            "Be concise, accurate, and actionable. Use bullet points and markdown when helpful.",
            "Never hallucinate — only reference what you know from the provided context.",
        ]
        if df_info:
            parts.append(
                f"\n## Dataset Context\n```json\n{json.dumps(df_info, indent=2, default=str)}\n```"
            )
        if model_info:
            parts.append(
                f"\n## Model Context\n```json\n{json.dumps(model_info, indent=2, default=str)}\n```"
            )
        return "\n".join(parts)

    def generate_eda_summary(self, profile_dict: dict) -> str:
        prompt = (
            "Analyze this dataset profile and provide:\n"
            "1. A 3-sentence executive summary\n"
            "2. Top 3 data quality concerns\n"
            "3. Top 3 ML readiness recommendations\n\n"
            f"Dataset Profile:\n{json.dumps(profile_dict, indent=2, default=str)}"
        )
        return self.chat([{"role": "user", "content": prompt}])

    def explain_model_performance(
        self, metrics: dict, problem_type: str, feature_importance: dict
    ) -> str:
        prompt = (
            f"Explain this ML model's performance in plain English:\n\n"
            f"Problem Type: {problem_type}\n"
            f"Metrics: {json.dumps(metrics, indent=2)}\n"
            f"Top Features: {json.dumps(dict(list(feature_importance.items())[:10]), indent=2)}\n\n"
            "Provide:\n"
            "1. Is this good/average/poor performance? Why?\n"
            "2. What do the metrics tell us?\n"
            "3. Which features are most influential and why?\n"
            "4. 3 specific improvement recommendations"
        )
        return self.chat([{"role": "user", "content": prompt}])

    def generate_business_insights(self, df_info: dict, top_features: list[str]) -> str:
        prompt = (
            "Given this dataset and the most important predictive features, "
            "generate actionable business insights:\n\n"
            f"Dataset: {json.dumps(df_info, indent=2, default=str)}\n"
            f"Most Important Features: {top_features}\n\n"
            "Provide:\n"
            "1. What business question does this model answer?\n"
            "2. Key drivers of the outcome (in business terms)\n"
            "3. 3 actionable recommendations for the business\n"
            "4. Potential risks or biases to watch for"
        )
        return self.chat([{"role": "user", "content": prompt}])
