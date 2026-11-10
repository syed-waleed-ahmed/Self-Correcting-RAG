"""Provider-agnostic chat client over any OpenAI-compatible endpoint.

Works with Groq (default), OpenAI, or a local vLLM/Ollama-compatible server by
changing ``base_url`` and ``model``. The underlying ``openai`` SDK already
handles connection pooling and retries with exponential backoff (including on
HTTP 429); we configure those and wrap provider errors in :class:`LLMError` (or
:class:`LLMRateLimitError` for quota/rate limits) so callers don't depend on the
SDK's exception types.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol, runtime_checkable

from ..errors import LLMError, LLMRateLimitError
from ..logging_config import get_logger
from ..observability import metrics, set_attributes, span
from ..usage import current_ledger

__all__ = ["ChatClient", "LLMError", "LLMRateLimitError", "OpenAICompatibleClient"]

logger = get_logger(__name__)

# Some reasoning models (e.g. Qwen3 in "raw" mode) inline their reasoning as
# <think>...</think> before the answer; it must never reach parsers or users.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(content: str) -> str:
    """Remove inline ``<think>`` blocks (and an unterminated trailing one)."""
    content = _THINK_BLOCK_RE.sub("", content)
    open_tag = content.lower().find("<think>")
    if open_tag != -1:
        content = content[:open_tag]
    return content.strip()


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
            raise LLMError("No LLM API key configured. Set GROQ_API_KEY (or SCRAG_LLM_API_KEY).")
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
        kwargs: dict[str, Any] = {
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

        start = time.perf_counter()
        with span(
            "llm.chat",
            gen_ai__system="openai_compatible",
            gen_ai__operation__name="chat",
            gen_ai__request__model=model,
        ) as current:
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - normalize all provider errors
                rate_limited = getattr(exc, "status_code", None) == 429
                metrics.LLM_CALLS.labels(model, "rate_limited" if rate_limited else "error").inc()
                metrics.LLM_DURATION.labels(model).observe(time.perf_counter() - start)
                # Full provider detail goes to server logs only; the API layer
                # returns a generic, client-safe message.
                logger.error("LLM request failed for model %s: %s", model, exc)
                if rate_limited:
                    raise LLMRateLimitError(f"LLM rate limited: {exc}") from exc
                raise LLMError(f"LLM request failed: {exc}") from exc

            metrics.LLM_CALLS.labels(model, "ok").inc()
            metrics.LLM_DURATION.labels(model).observe(time.perf_counter() - start)
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion = int(getattr(usage, "completion_tokens", 0) or 0)
                metrics.LLM_TOKENS.labels(model, "prompt").inc(prompt)
                metrics.LLM_TOKENS.labels(model, "completion").inc(completion)
                set_attributes(
                    current,
                    gen_ai__usage__input_tokens=prompt,
                    gen_ai__usage__output_tokens=completion,
                )
                ledger = current_ledger()
                if ledger is not None:
                    ledger.record(model, prompt, completion)
        content = response.choices[0].message.content
        return strip_reasoning(content or "")
