"""Provider-agnostic chat client over any OpenAI-compatible endpoint.

Works with Groq (default), OpenAI, or a local vLLM/Ollama-compatible server by
changing ``base_url`` and ``model``. The underlying ``openai`` SDK already
handles connection pooling and retries with exponential backoff; we configure
those and wrap provider errors in a single :class:`LLMError` so callers don't
depend on the SDK's exception types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..logging_config import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM request ultimately fails (after retries)."""


@runtime_checkable
class ChatClient(Protocol):
    """Minimal chat interface the agents depend on (easily faked in tests)."""

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant message content for a single-turn prompt."""
        ...


class OpenAICompatibleClient:
    """Concrete :class:`ChatClient` backed by the ``openai`` SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise LLMError(
                "No LLM API key configured. Set GROQ_API_KEY (or SCRAG_LLM_API_KEY)."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            logger.error("LLM request failed for model %s: %s", model, exc)
            raise LLMError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()
