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
    # Set by track_usage when nested: usage is also forwarded to the outer ledger
    # (e.g. an evaluation run that measures several pipeline runs).
    parent: UsageLedger | None = field(default=None, repr=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        if self.parent is not None:
            self.parent.record(model, prompt_tokens, completion_tokens)
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
    outer = _current.get()
    if outer is not None and outer is not ledger:
        ledger.parent = outer
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)


def estimate_cost(
    by_model: dict[str, dict[str, int]], prices: dict[str, dict[str, float]]
) -> float | None:
    """Estimated USD cost at configured list prices (per million tokens).

    ``prices`` maps a model id to ``{"input": usd, "output": usd}``. Returns None
    when any model used has no configured price: SCRAG never guesses prices.
    The demo runs on a free tier, so its real cost is 0; this answers "what
    would this run cost on a paid plan at these prices?".
    """
    if not by_model:
        return 0.0
    total = 0.0
    for model, usage in by_model.items():
        price = prices.get(model)
        if price is None or "input" not in price or "output" not in price:
            return None
        total += usage.get("prompt_tokens", 0) * price["input"] / 1_000_000
        total += usage.get("completion_tokens", 0) * price["output"] / 1_000_000
    return round(total, 6)
