"""A durable Postgres job queue and a single background worker.

Semantics:

* **At-least-once.** A worker claims a job with ``FOR UPDATE SKIP LOCKED`` and
  takes a lease (``locked_until``). If the process dies, the lease expires and
  any worker claims the job again, so work interrupted by a restart resumes.
  Handlers must therefore be idempotent (document ingestion replaces a
  document's chunks rather than appending).
* **Fenced completion.** Completing or failing a job checks ``locked_by``, so a
  worker whose lease expired can't overwrite the result of the worker that
  took over.
* **Retries.** A failure requeues the job with exponential backoff until
  ``max_attempts``. :class:`PermanentJobError` fails it at once. A handler's
  ``on_failure`` hook runs when a job finally fails.
* **No idle polling.** When the queue is empty the worker sleeps until an
  enqueue in this process wakes it, so it never pings the database by itself.
  It polls on an interval only while jobs are pending (retries scheduled for
  later, or jobs whose leases may expire).
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import DatabaseUnavailableError
from ..logging_config import get_logger
from .database import Database
from .records import JobRecord

logger = get_logger(__name__)

_JOB_COLUMNS = (
    "id, kind, payload, session_id, status, attempts, max_attempts, run_after,"
    " locked_by, last_error, result, created_at, updated_at"
)


class PermanentJobError(Exception):
    """A failure that retrying can't fix (e.g. the input is invalid)."""

    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class JobHandler:
    run: Callable[[JobRecord], dict[str, Any] | None]
    on_failure: Callable[[JobRecord, BaseException], None] | None = None


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


class JobQueue:
    def __init__(self, db: Database, *, max_attempts: int = 3, backoff_base_s: float = 5.0) -> None:
        self.db = db
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self._listeners: list[Callable[[], None]] = []

    def on_enqueue(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def notify(self) -> None:
        for listener in self._listeners:
            listener()

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        session_id: uuid.UUID | None = None,
        conn: Any = None,
    ) -> uuid.UUID:
        """Add a job. Pass ``conn`` to enqueue inside the caller's transaction
        (the job then exists if and only if the caller's writes commit); the
        caller must call :meth:`notify` after committing."""
        from psycopg.types.json import Jsonb

        job_id = uuid.uuid4()
        params = (job_id, kind, Jsonb(payload), session_id, self.max_attempts)
        sql = (
            "INSERT INTO scrag.jobs (id, kind, payload, session_id, max_attempts)"
            " VALUES (%s, %s, %s, %s, %s)"
        )
        if conn is not None:
            conn.execute(sql, params)
            return job_id
        with self.db.connection() as own:
            own.execute(sql, params)
        self.notify()
        return job_id

    def claim(self, worker_id: str, lease_s: float) -> JobRecord | None:
        """Take the next runnable job: queued and due, or running with an expired lease."""
        with self.db.connection() as conn:
            row = conn.execute(
                "UPDATE scrag.jobs SET status = 'running', attempts = attempts + 1,"
                " locked_by = %(worker)s, locked_until = now() + make_interval(secs => %(lease)s),"
                " updated_at = now()"
                " WHERE id = ("
                "   SELECT id FROM scrag.jobs"
                "   WHERE (status = 'queued' AND run_after <= now())"
                "      OR (status = 'running' AND locked_until < now())"
                "   ORDER BY run_after, created_at"
                "   FOR UPDATE SKIP LOCKED LIMIT 1)"
                f" RETURNING {_JOB_COLUMNS}",
                {"worker": worker_id, "lease": float(lease_s)},
            ).fetchone()
        return JobRecord(**row) if row else None

    def complete(self, job: JobRecord, worker_id: str, result: dict[str, Any] | None) -> bool:
        from psycopg.types.json import Jsonb

        with self.db.connection() as conn:
            cur = conn.execute(
                "UPDATE scrag.jobs SET status = 'succeeded', result = %s, last_error = NULL,"
                " locked_by = NULL, locked_until = NULL, updated_at = now()"
                " WHERE id = %s AND locked_by = %s AND status = 'running'",
                (Jsonb(result or {}), job.id, worker_id),
            )
            return bool(cur.rowcount)

    def fail(
        self, job: JobRecord, worker_id: str, error: str, *, permanent: bool = False
    ) -> str | None:
        """Requeue with backoff, or mark failed. Returns the new status (None if fenced out)."""
        delay = min(300.0, self.backoff_base_s * (2 ** max(0, job.attempts - 1)))
        with self.db.connection() as conn:
            row = conn.execute(
                "UPDATE scrag.jobs SET"
                " status = CASE WHEN %(permanent)s OR attempts >= max_attempts"
                "   THEN 'failed' ELSE 'queued' END,"
                " run_after = now() + make_interval(secs => %(delay)s),"
                " last_error = %(error)s, locked_by = NULL, locked_until = NULL,"
                " updated_at = now()"
                " WHERE id = %(id)s AND locked_by = %(worker)s AND status = 'running'"
                " RETURNING status",
                {
                    "permanent": permanent,
                    "delay": delay,
                    "error": error[:500],
                    "id": job.id,
                    "worker": worker_id,
                },
            ).fetchone()
        return row["status"] if row else None

    def get(self, job_id: uuid.UUID, *, session_id: uuid.UUID | None = None) -> JobRecord | None:
        """A job by id; with ``session_id``, only if it belongs to that session."""
        sql = f"SELECT {_JOB_COLUMNS} FROM scrag.jobs WHERE id = %s"
        params: tuple[Any, ...] = (job_id,)
        if session_id is not None:
            sql += " AND session_id = %s"
            params = (job_id, session_id)
        with self.db.connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return JobRecord(**row) if row else None

    def has_pending(self) -> bool:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM scrag.jobs WHERE status IN ('queued', 'running'))"
                " AS pending"
            ).fetchone()
        return bool(row["pending"])


class JobWorker:
    """Runs jobs one at a time on a daemon thread (the free tier has 0.1 CPU)."""

    def __init__(
        self,
        queue: JobQueue,
        handlers: dict[str, JobHandler],
        *,
        lease_s: float = 120.0,
        poll_interval_s: float = 30.0,
        maintenance: Callable[[], None] | None = None,
        maintenance_interval_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self.lease_s = lease_s
        self.poll_interval_s = poll_interval_s
        self.maintenance = maintenance
        self.maintenance_interval_s = maintenance_interval_s
        self.worker_id = _worker_id()
        self._clock = clock
        self._last_maintenance: float | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        queue.on_enqueue(self.wake)

    # --- thread control ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scrag-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def wake(self) -> None:
        self._wake.set()

    # --- work ----------------------------------------------------------------------
    def run_once(self) -> bool:
        """Claim and run one job. Returns False when nothing was runnable."""
        job = self.queue.claim(self.worker_id, self.lease_s)
        if job is None:
            return False
        handler = self.handlers.get(job.kind)
        if handler is None:
            self._finish_failed(job, PermanentJobError("unknown_job_kind", job.kind), None)
            return True
        if job.attempts > job.max_attempts:
            # Its last attempt died mid-run (lease expired): don't try again.
            self._finish_failed(job, PermanentJobError("lease_expired"), handler)
            return True
        started = time.perf_counter()
        try:
            result = handler.run(job)
        except PermanentJobError as exc:
            self._finish_failed(job, exc, handler)
        except DatabaseUnavailableError:
            raise  # the lease expires and the job is retried once the database is back
        except Exception as exc:  # noqa: BLE001 - any handler failure is retried
            logger.warning("Job %s (%s) attempt %d failed: %r", job.id, job.kind, job.attempts, exc)
            status = self.queue.fail(job, self.worker_id, f"{type(exc).__name__}: {exc}")
            if status == "failed" and handler.on_failure is not None:
                handler.on_failure(job, exc)
        else:
            if self.queue.complete(job, self.worker_id, result):
                logger.info(
                    "Job %s (%s) succeeded in %.0f ms",
                    job.id,
                    job.kind,
                    (time.perf_counter() - started) * 1000,
                )
        return True

    def run_until_idle(self, max_jobs: int = 100) -> int:
        """Run jobs until none is runnable (used by tests and the CLI)."""
        count = 0
        while count < max_jobs and self.run_once():
            count += 1
        return count

    def run_maintenance(self, *, force: bool = False) -> None:
        if self.maintenance is None:
            return
        now = self._clock()
        due = self._last_maintenance is None or (
            now - self._last_maintenance >= self.maintenance_interval_s
        )
        if force or due:
            self._last_maintenance = now
            self.maintenance()

    def _finish_failed(
        self, job: JobRecord, exc: PermanentJobError, handler: JobHandler | None
    ) -> None:
        logger.warning("Job %s (%s) failed permanently: %s", job.id, job.kind, exc.code)
        status = self.queue.fail(job, self.worker_id, exc.code, permanent=True)
        if status == "failed" and handler is not None and handler.on_failure is not None:
            handler.on_failure(job, exc)

    def _loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            timeout: float | None
            try:
                self.queue.db.ensure_ready()
                self.run_maintenance()
                while not self._stop.is_set() and self.run_once():
                    pass
                # Sleep until an enqueue wakes us; poll on a timer only while jobs
                # are pending (retries due later, or leases that may expire).
                timeout = self.poll_interval_s if self.queue.has_pending() else None
                backoff = 1.0
            except DatabaseUnavailableError:
                backoff = min(300.0, backoff * 2)
                timeout = backoff
            except Exception:  # noqa: BLE001 - keep the worker alive
                logger.exception("Job worker loop error")
                timeout = self.poll_interval_s
            if self._stop.is_set():
                break
            self._wake.wait(timeout)
            self._wake.clear()
