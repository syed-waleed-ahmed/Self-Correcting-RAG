"""Anonymous demo sessions.

A session is an opaque bearer token (``scs_`` + 256 random bits). Only its
SHA-256 hash is stored, so a database leak does not leak usable tokens. Sessions
expire after a fixed TTL from creation (no sliding renewal): everything a
visitor uploaded or asked is deleted with the session, at the latest one TTL
after they started.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import timedelta
from typing import Any

from ..errors import SessionCapacityError
from .database import Database
from .records import SessionRecord

TOKEN_PREFIX = "scs_"
_MAX_TOKEN_LENGTH = 128


def hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class SessionStore:
    def __init__(self, db: Database, *, ttl_hours: float, max_live: int | None = None) -> None:
        self.db = db
        self.ttl = timedelta(hours=ttl_hours)
        self.max_live = max_live

    def create(self) -> tuple[SessionRecord, str]:
        """Create an anonymous session. Returns the record and the token (shown once).

        Refused with :class:`SessionCapacityError` while ``max_live`` sessions are
        live: sessions are cheap rows, but an unbounded number of them (from many
        addresses, each within its rate limit) would still fill a free database.
        The check and the insert are one statement; under heavy concurrency the
        cap can be overshot by a few rows, never by an unbounded number.
        """
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        with self.db.connection() as conn:
            row = conn.execute(
                "INSERT INTO scrag.sessions (id, token_hash, expires_at)"
                " SELECT %s, %s, now() + %s"
                " WHERE %s::int IS NULL"
                " OR (SELECT count(*) FROM scrag.sessions WHERE expires_at > now()) < %s::int"
                " RETURNING id, kind, created_at, expires_at",
                (uuid.uuid4(), hash_token(token), self.ttl, self.max_live, self.max_live),
            ).fetchone()
        if row is None:
            raise SessionCapacityError(f"{self.max_live} live sessions")
        return SessionRecord(**row), token

    def authenticate(self, token: str | None) -> SessionRecord | None:
        """The live session for ``token``, or None (unknown, malformed or expired)."""
        if not token or not token.startswith(TOKEN_PREFIX) or len(token) > _MAX_TOKEN_LENGTH:
            return None  # don't spend a database round trip on garbage
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT id, kind, created_at, expires_at FROM scrag.sessions"
                " WHERE token_hash = %s AND expires_at > now()",
                (hash_token(token),),
            ).fetchone()
        return SessionRecord(**row) if row else None

    def usage(self, session_id: uuid.UUID) -> dict[str, Any]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT"
                " (SELECT count(*) FROM scrag.documents WHERE session_id = %(s)s) AS documents,"
                " (SELECT count(*) FROM scrag.conversations WHERE session_id = %(s)s)"
                "   AS conversations,"
                " (SELECT count(*) FROM scrag.messages WHERE session_id = %(s)s) AS messages",
                {"s": session_id},
            ).fetchone()
        return dict(row)

    def delete(self, session_id: uuid.UUID) -> bool:
        """Delete a session and, by cascade, all of its documents, chunks, history and jobs."""
        with self.db.connection() as conn:
            cur = conn.execute("DELETE FROM scrag.sessions WHERE id = %s", (session_id,))
            return bool(cur.rowcount)

    def purge_expired(self) -> int:
        """Delete expired sessions (and their data). Returns how many were removed."""
        with self.db.connection() as conn:
            cur = conn.execute("DELETE FROM scrag.sessions WHERE expires_at <= now()")
            return int(cur.rowcount or 0)

    def stats(self) -> dict[str, int]:
        """Operator view: row counts across all sessions (no contents)."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT"
                " (SELECT count(*) FROM scrag.sessions WHERE expires_at > now()) AS sessions,"
                " (SELECT count(*) FROM scrag.documents) AS documents,"
                " (SELECT count(*) FROM scrag.chunks) AS chunks,"
                " (SELECT count(*) FROM scrag.messages) AS messages,"
                " (SELECT count(*) FROM scrag.jobs WHERE status IN ('queued', 'running'))"
                "   AS pending_jobs,"
                " (SELECT count(*) FROM scrag.jobs WHERE status = 'failed') AS failed_jobs",
            ).fetchone()
        return {key: int(value) for key, value in row.items()}
