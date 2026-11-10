"""Postgres connection management.

Design constraints (free tier, see TARGET_ARCHITECTURE D2/D3):

* **The API must start without the database.** A paused free-tier project or a
  network blip must never break sample-corpus Q&A, so nothing here connects at
  construction time and every failure becomes :class:`DatabaseUnavailableError`
  (HTTP 503) on the endpoints that need persistence.
* **Fail fast while the database is down.** After a connection failure, calls
  fail immediately for ``retry_interval_s`` instead of each waiting for a
  connect timeout (a simple circuit breaker).
* **No idle traffic.** The pool keeps no minimum connections and closes idle
  ones, so the server never pings the database on its own. That would be
  keep-alive traffic, which the free-tier rules forbid.
* **Poolers.** Prepared statements are disabled, so the URL can point at a
  transaction-mode pooler (Supabase Supavisor on port 6543, PgBouncer).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from ..errors import DatabaseUnavailableError
from ..logging_config import get_logger

logger = get_logger(__name__)

DatabaseStatus = Literal["unknown", "ok", "unavailable"]


def _driver() -> tuple[Any, Any, Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool, PoolTimeout
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'Persistence needs the postgres extra: pip install "self-correcting-rag[postgres]"'
        ) from exc
    return psycopg, dict_row, ConnectionPool, PoolTimeout


class Database:
    """A lazily connecting, migration-aware Postgres connection pool."""

    def __init__(
        self,
        url: str,
        *,
        pool_max: int = 4,
        connect_timeout_s: float = 5.0,
        retry_interval_s: float = 30.0,
        migrate: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        psycopg, dict_row, connection_pool, pool_timeout = _driver()
        self._psycopg = psycopg
        self._unavailable_errors: tuple[type[BaseException], ...] = (
            psycopg.OperationalError,
            pool_timeout,
        )
        self._pool = connection_pool(
            url,
            min_size=0,  # connect on demand only: no idle connections, no pings
            max_size=pool_max,
            max_idle=120,
            timeout=connect_timeout_s + 2,
            reconnect_timeout=60,
            open=False,
            check=connection_pool.check_connection,
            name="scrag",
            kwargs={
                "prepare_threshold": None,  # transaction poolers can't keep them
                "connect_timeout": max(1, round(connect_timeout_s)),
                "row_factory": dict_row,
                "application_name": "scrag",
            },
        )
        self._migrate = migrate
        self._retry_interval = retry_interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._opened = False
        self._ready = False
        self._unavailable_until = float("-inf")
        self.status: DatabaseStatus = "unknown"

    # --- lifecycle -----------------------------------------------------------------
    def open(self) -> None:
        with self._lock:
            if not self._opened:
                self._pool.open(wait=False)  # min_size=0: this opens no connection
                self._opened = True

    def close(self) -> None:
        with self._lock:
            if self._opened:
                self._pool.close()
                self._opened = False

    @property
    def ready(self) -> bool:
        return self._ready

    def ensure_ready(self) -> None:
        """Connect (and apply migrations) once; raise ``DatabaseUnavailableError`` if down."""
        if self._ready:
            return
        self.open()
        with self._lock:
            if self._ready:
                return
            self._check_breaker()
            try:
                with self._pool.connection() as conn:
                    if self._migrate:
                        from .migrations import apply_migrations

                        applied = apply_migrations(conn)
                        if applied:
                            logger.info("Applied database migrations: %s", applied)
                    else:
                        conn.execute("SELECT 1")
            except self._psycopg.Error as exc:  # includes PoolTimeout and failed migrations
                self._trip(exc)
                raise DatabaseUnavailableError(f"database not ready: {exc}") from exc
            self._ready = True
            self.status = "ok"

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """A pooled connection inside one transaction (commit on success, else rollback)."""
        self.ensure_ready()
        self._check_breaker()
        try:
            with self._pool.connection() as conn:
                yield conn
        except self._unavailable_errors as exc:
            self._trip(exc)
            raise DatabaseUnavailableError(f"database error: {exc}") from exc
        self.status = "ok"

    # --- circuit breaker -----------------------------------------------------------
    def _check_breaker(self) -> None:
        if self._clock() < self._unavailable_until:
            raise DatabaseUnavailableError("database marked unavailable; retrying shortly")

    def _trip(self, exc: BaseException) -> None:
        self.status = "unavailable"
        self._unavailable_until = self._clock() + self._retry_interval
        logger.warning("Database unavailable (%s): %s", type(exc).__name__, exc)
