"""Per-run LLM usage accounting.

A :class:`UsageLedger` is bound to the current execution context with
:func:`track_usage`. :class:`~scrag.llm.client.OpenAICompatibleClient` records
the provider-reported token counts of every call made inside that context, so a
workflow can report its real cost without threading a ledger through every
agent. Calls made outside a tracked context are not recorded.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_current: ContextVar[UsageLedger | None] = ContextVar("scrag_usage_ledger", default=None)


@dataclass
class UsageLedger:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            entry = self.by_model.setdefault(
                model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
            }


def current_ledger() -> UsageLedger | None:
    return _current.get()


@contextmanager
def track_usage(ledger: UsageLedger) -> Iterator[UsageLedger]:
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)
