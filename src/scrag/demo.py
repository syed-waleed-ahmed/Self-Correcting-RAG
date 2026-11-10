"""Public-demo protections and answer caching.

* :class:`RateLimiter` - per-client token bucket (in-process: correct for the
  single-instance free tier; a shared store is needed once there are several
  instances).
* :class:`DailyBudget` - global cap on live LLM pipeline runs per UTC day, so the
  provider's free quota can't be drained. It resets on restart (it is a quota
  guard, not billing); with a database configured,
  :class:`scrag.persistence.PersistentDailyBudget` keeps the count across restarts.
* :class:`ClientQuota` - each client's share of that daily budget, so one
  visitor can't spend all of it.
* :class:`AnswerCache` - recorded sample answers plus a small TTL cache of recent
  live answers. Cached responses are always labelled as cached with their
  timestamp; they are real past executions, never synthesized.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from .errors import ClientQuotaExceededError, DemoBudgetExhaustedError, RateLimitedError
from .logging_config import get_logger
from .models import Chunk

logger = get_logger(__name__)


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split()).rstrip(" ?!.")


def corpus_fingerprint(chunks: Iterable[Chunk]) -> str:
    """Identify an index's corpus by its (source, content hash) pairs."""
    pairs = sorted({(c.source, c.metadata.get("doc_hash", "")) for c in chunks})
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


class RateLimiter:
    """Token bucket per key: ``burst`` requests at once, refilled at ``per_minute``."""

    def __init__(self, per_minute: float, burst: int, *, clock: Any = time.monotonic) -> None:
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Consume one token for ``key`` or raise :class:`RateLimitedError`."""
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                retry_after = math.ceil((1.0 - tokens) / self.rate) if self.rate else 60
                self._buckets[key] = (tokens, now)
                raise RateLimitedError(retry_after=retry_after)
            self._buckets[key] = (tokens - 1.0, now)
            if len(self._buckets) > 10_000:  # bound memory: drop the stalest keys
                for stale in sorted(self._buckets, key=lambda k: self._buckets[k][1])[:5_000]:
                    del self._buckets[stale]


class Budget(Protocol):
    """A daily quota of live LLM runs (in-process, or persisted in Postgres)."""

    limit: int

    def consume(self, units: int = 1) -> None:
        """Take ``units`` from today's quota; raises DemoBudgetExhaustedError when spent."""

    def refund(self, units: int = 1) -> None:
        """Give units back (the run failed before producing an answer)."""

    def status(self) -> dict[str, Any]:
        """``limit``, ``remaining`` and ``resets_at`` (ISO 8601, UTC) for today."""


class DailyBudget:
    """At most ``limit`` live pipeline runs per UTC day, across all clients."""

    def __init__(self, limit: int, *, now: Any = None) -> None:
        self.limit = limit
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._day = self._now().date()
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = self._now().date()
        if today != self._day:
            self._day, self._used = today, 0

    def consume(self, units: int = 1) -> None:
        with self._lock:
            self._roll()
            if self._used + units > self.limit:
                raise DemoBudgetExhaustedError(retry_after=self._seconds_until_reset())
            self._used += units

    def refund(self, units: int = 1) -> None:
        """Give units back (the run failed before producing an answer)."""
        with self._lock:
            self._used = max(0, self._used - units)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._roll()
            reset = datetime.combine(self._day + timedelta(days=1), datetime.min.time())
            return {
                "limit": self.limit,
                "remaining": max(0, self.limit - self._used),
                "resets_at": reset.replace(tzinfo=timezone.utc).isoformat(),
            }

    def _seconds_until_reset(self) -> int:
        tomorrow = datetime.combine(self._day + timedelta(days=1), datetime.min.time())
        delta = tomorrow.replace(tzinfo=timezone.utc) - self._now()
        return max(1, int(delta.total_seconds()))


class ClientQuota:
    """At most ``limit`` live runs per client per UTC day.

    The global :class:`DailyBudget` protects the provider quota; this protects
    the other visitors, so one client (IP) can't spend the whole day's budget.
    In-process, like :class:`RateLimiter`: correct for the single-instance demo.
    Memory is bounded: past ``max_clients`` keys, the lightest users are
    forgotten (which only ever gives them a fresh share, never blocks anyone).
    """

    def __init__(self, limit: int, *, now: Any = None, max_clients: int = 10_000) -> None:
        self.limit = limit
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._day = self._now().date()
        self._used: dict[str, int] = {}
        self._max_clients = max_clients
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = self._now().date()
        if today != self._day:
            self._day = today
            self._used.clear()

    def consume(self, key: str, units: int = 1) -> None:
        """Charge ``units`` to ``key`` or raise :class:`ClientQuotaExceededError`."""
        with self._lock:
            self._roll()
            used = self._used.get(key, 0)
            if used + units > self.limit:
                tomorrow = datetime.combine(self._day + timedelta(days=1), datetime.min.time())
                wait = (tomorrow.replace(tzinfo=timezone.utc) - self._now()).total_seconds()
                raise ClientQuotaExceededError(
                    f"client used {used} of {self.limit} live runs today",
                    retry_after=max(1, int(wait)),
                )
            if key not in self._used and len(self._used) >= self._max_clients:
                lightest = sorted(self._used, key=self._used.__getitem__)
                for stale in lightest[: self._max_clients // 2]:
                    del self._used[stale]
            self._used[key] = used + units

    def refund(self, key: str, units: int = 1) -> None:
        """Give units back (the run produced no answer)."""
        with self._lock:
            if key in self._used:
                self._used[key] = max(0, self._used[key] - units)

    def remaining(self, key: str) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used.get(key, 0))


@dataclass(frozen=True)
class CachedAnswer:
    response: dict[str, Any]
    cached_at: str
    source: Literal["sample", "recent"]  # recorded run shipped with the demo, or recent


class AnswerCache:
    """Recorded sample answers + a TTL/LRU cache of recent live answers."""

    def __init__(self, *, ttl_seconds: int, max_entries: int, clock: Any = time.time) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._recent: OrderedDict[tuple[str, int, str], tuple[float, CachedAnswer]] = OrderedDict()
        self._samples: dict[str, CachedAnswer] = {}
        self._samples_fingerprint: str | None = None
        self.sample_questions: list[str] = []
        self._lock = threading.Lock()

    def load_samples(self, path: Path) -> int:
        """Load recorded sample answers (see ``scripts/record_samples.py``)."""
        if not path.is_file():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        self._samples_fingerprint = data.get("corpus_fingerprint")
        self.sample_questions = [item["question"] for item in data.get("items", [])]
        self._samples = {
            normalize_query(item["question"]): CachedAnswer(
                response=item["response"], cached_at=data["recorded_at"], source="sample"
            )
            for item in data.get("items", [])
        }
        return len(self._samples)

    def get(self, query: str, top_k: int, fingerprint: str) -> CachedAnswer | None:
        key = normalize_query(query)
        # Recorded samples are only valid for the corpus they were recorded on.
        sample = self._samples.get(key)
        if sample is not None and fingerprint == self._samples_fingerprint:
            return sample
        with self._lock:
            entry = self._recent.get((key, top_k, fingerprint))
            if entry is None:
                return None
            stored_at, answer = entry
            if self._clock() - stored_at > self.ttl:
                del self._recent[(key, top_k, fingerprint)]
                return None
            self._recent.move_to_end((key, top_k, fingerprint))
            return answer

    def put(self, query: str, top_k: int, fingerprint: str, response: dict[str, Any]) -> None:
        if self.max_entries <= 0 or self.ttl <= 0:
            return
        now = self._clock()
        stamp = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        with self._lock:
            key = (normalize_query(query), top_k, fingerprint)
            self._recent[key] = (
                now,
                CachedAnswer(response=response, cached_at=stamp, source="recent"),
            )
            self._recent.move_to_end(key)
            while len(self._recent) > self.max_entries:
                self._recent.popitem(last=False)
