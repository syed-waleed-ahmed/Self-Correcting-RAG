"""Provider-agnostic chat client over any OpenAI-compatible endpoint.

Works with Groq (default), OpenAI, or a local vLLM/Ollama-compatible server by
changing ``base_url`` and ``model``. The underlying ``openai`` SDK already
handles connection pooling and retries with exponential backoff (including on
HTTP 429); we configure those and wrap provider errors in :class:`LLMError` (or
:class:`LLMRateLimitError` for quota/rate limits) so callers don't depend on the
SDK's exception types.

Optional ``fallback_models``: when a model is rate limited or no longer served,
the same call is retried once on each fallback in order. Free-tier quotas are
per model, so a fallback also adds headroom; providers retire free models with
little notice (Groq did twice in 2026).
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..errors import LLMError, LLMModelUnavailableError, LLMRateLimitError
from ..logging_config import get_logger
from ..observability import metrics, set_attributes, span
from ..usage import current_ledger

__all__ = [
    "ChatClient",
    "LLMError",
    "LLMModelUnavailableError",
    "LLMRateLimitError",
    "OpenAICompatibleClient",
]

# Provider error codes that mean "this model can't be used", not "this request is bad".
_MODEL_GONE_CODES = {"model_not_found", "model_decommissioned", "model_not_active"}


def _model_unavailable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    return status == 404 or (status == 400 and code in _MODEL_GONE_CODES)


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
        fallback_models: Sequence[str] = (),
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
        self.fallback_models = [m for m in fallback_models if m]

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
        """Return the assistant message, falling back to other models if this one is
        rate limited or no longer served (other errors are raised at once)."""
        candidates = [model, *(m for m in self.fallback_models if m != model)]
        for i, candidate in enumerate(candidates):
            try:
                return self._complete_once(
                    candidate, system, user, temperature, max_tokens, json_mode
                )
            except (LLMRateLimitError, LLMModelUnavailableError) as exc:
                if i + 1 == len(candidates):
                    raise
                reason = "rate limited" if isinstance(exc, LLMRateLimitError) else "unavailable"
                logger.warning(
                    "LLM model %s %s; falling back to %s", candidate, reason, candidates[i + 1]
                )
        raise AssertionError("unreachable: the last candidate either returns or raises")

    def _complete_once(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
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
                if _model_unavailable(exc):
                    raise LLMModelUnavailableError(f"LLM model unavailable: {exc}") from exc
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
