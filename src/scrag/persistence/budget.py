"""The demo's daily live-run budget, stored in Postgres so it survives restarts.

The in-process :class:`~scrag.demo.DailyBudget` resets whenever the free-tier
instance restarts, and it restarts after every 15-minute sleep, so "30 per day"
really meant "30 per wake-up". This version keeps the counter in
``scrag.usage_counters``: one row per subject and UTC day, updated atomically
with a conditional upsert.

If the database is unavailable, it falls back to the in-process budget, so the
core demo keeps working (a quota guard, not billing: counts may be slightly
off across a database outage).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..demo import ClientQuota, DailyBudget
from ..errors import (
    ClientQuotaExceededError,
    DatabaseUnavailableError,
    DemoBudgetExhaustedError,
)
from ..logging_config import get_logger
from .database import Database

logger = get_logger(__name__)


class PersistentDailyBudget:
    def __init__(
        self,
        db: Database,
        limit: int,
        *,
        subject: str = "demo:live_runs",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.db = db
        self.limit = limit
        self.subject = subject
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.fallback = DailyBudget(limit, now=self._now)

    def _today(self) -> date:
        return self._now().date()

    def _reset_at(self, day: date) -> datetime:
        return datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

    def _retry_after(self, day: date) -> int:
        return max(1, int((self._reset_at(day) - self._now()).total_seconds()))

    def consume(self, units: int = 1) -> None:
        day = self._today()
        if units > self.limit:
            raise DemoBudgetExhaustedError(retry_after=self._retry_after(day))
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "INSERT INTO scrag.usage_counters AS u (subject, window_start, used)"
                    " VALUES (%(s)s, %(d)s, %(n)s)"
                    " ON CONFLICT (subject, window_start) DO UPDATE SET used = u.used + %(n)s"
                    " WHERE u.used + %(n)s <= %(limit)s"
                    " RETURNING used",
                    {"s": self.subject, "d": day, "n": units, "limit": self.limit},
                ).fetchone()
        except DatabaseUnavailableError:
            logger.warning("Budget database unavailable; using the in-process budget")
            self.fallback.consume(units)
            return
        if row is None:
            raise DemoBudgetExhaustedError(retry_after=self._retry_after(day))

    def refund(self, units: int = 1) -> None:
        try:
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE scrag.usage_counters SET used = GREATEST(0, used - %s)"
                    " WHERE subject = %s AND window_start = %s",
                    (units, self.subject, self._today()),
                )
        except DatabaseUnavailableError:
            self.fallback.refund(units)

    def status(self) -> dict[str, Any]:
        day = self._today()
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "SELECT used FROM scrag.usage_counters"
                    " WHERE subject = %s AND window_start = %s",
                    (self.subject, day),
                ).fetchone()
        except DatabaseUnavailableError:
            return self.fallback.status()
        used = int(row["used"]) if row else 0
        return {
            "limit": self.limit,
            "remaining": max(0, self.limit - used),
            "resets_at": self._reset_at(day).isoformat(),
        }


class PersistentClientQuota:
    """Each client's daily share of live runs, stored in Postgres.

    The in-process :class:`~scrag.demo.ClientQuota` resets whenever the free
    instance restarts, and it restarts after every 15-minute sleep, so a client
    could get a fresh share by waiting. Counters live in ``scrag.usage_counters``
    like the global budget, under ``client:`` plus a keyed hash (HMAC-SHA256
    with a server secret) of the client key: the address itself is never
    stored, and :meth:`purge_old` deletes the rows after a day. Without the
    database it falls back to the in-process quota.
    """

    PREFIX = "client:"

    def __init__(
        self,
        db: Database,
        limit: int,
        *,
        secret: bytes,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.db = db
        self.limit = limit
        self._secret = secret
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.fallback = ClientQuota(limit, now=self._now)

    def subject(self, key: str) -> str:
        digest = hmac.new(self._secret, key.encode("utf-8"), hashlib.sha256).hexdigest()
        return self.PREFIX + digest[:32]

    def _retry_after(self, day: date) -> int:
        reset = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        return max(1, int((reset - self._now()).total_seconds()))

    def consume(self, key: str, units: int = 1) -> None:
        day = self._now().date()
        if units > self.limit:
            raise ClientQuotaExceededError(retry_after=self._retry_after(day))
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "INSERT INTO scrag.usage_counters AS u (subject, window_start, used)"
                    " VALUES (%(s)s, %(d)s, %(n)s)"
                    " ON CONFLICT (subject, window_start) DO UPDATE SET used = u.used + %(n)s"
                    " WHERE u.used + %(n)s <= %(limit)s"
                    " RETURNING used",
                    {"s": self.subject(key), "d": day, "n": units, "limit": self.limit},
                ).fetchone()
        except DatabaseUnavailableError:
            logger.warning("Client quota database unavailable; using the in-process quota")
            self.fallback.consume(key, units)
            return
        if row is None:
            raise ClientQuotaExceededError(retry_after=self._retry_after(day))

    def refund(self, key: str, units: int = 1) -> None:
        try:
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE scrag.usage_counters SET used = GREATEST(0, used - %s)"
                    " WHERE subject = %s AND window_start = %s",
                    (units, self.subject(key), self._now().date()),
                )
        except DatabaseUnavailableError:
            self.fallback.refund(key, units)

    def remaining(self, key: str) -> int:
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "SELECT used FROM scrag.usage_counters"
                    " WHERE subject = %s AND window_start = %s",
                    (self.subject(key), self._now().date()),
                ).fetchone()
        except DatabaseUnavailableError:
            return self.fallback.remaining(key)
        return max(0, self.limit - (int(row["used"]) if row else 0))

    def purge_old(self) -> int:
        """Delete client counters from before yesterday. Returns how many were removed."""
        return purge_client_counters(self.db, today=self._now().date())


def purge_client_counters(db: Database, *, today: date | None = None) -> int:
    """Delete per-client counters from before yesterday (they only matter for today)."""
    today = today or datetime.now(timezone.utc).date()
    with db.connection() as conn:
        cur = conn.execute(
            "DELETE FROM scrag.usage_counters WHERE subject LIKE %s AND window_start < %s",
            (PersistentClientQuota.PREFIX + "%", today - timedelta(days=1)),
        )
        return int(cur.rowcount or 0)
