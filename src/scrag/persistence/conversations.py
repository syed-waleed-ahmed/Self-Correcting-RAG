"""Conversation history, scoped to a session.

Every query filters on ``session_id``; a conversation from another session is
indistinguishable from one that does not exist (404, never 403).
"""

from __future__ import annotations

import uuid
from typing import Any

from ..errors import NotFoundError
from .database import Database
from .records import ConversationDetail, ConversationRecord, MessageRecord

_TITLE_CHARS = 80


def make_title(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= _TITLE_CHARS else title[: _TITLE_CHARS - 1].rstrip() + "…"


class ConversationStore:
    def __init__(self, db: Database, *, max_messages_per_session: int = 400) -> None:
        self.db = db
        self.max_messages = max_messages_per_session

    def ensure(self, session_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        """Raise :class:`NotFoundError` unless the conversation belongs to the session."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM scrag.conversations WHERE id = %s AND session_id = %s",
                (conversation_id, session_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("conversation not found in this session")

    def for_session(self, session_id: uuid.UUID, *, limit: int = 50) -> list[ConversationRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at,"
                " (SELECT count(*) FROM scrag.messages m WHERE m.conversation_id = c.id)"
                "   AS message_count"
                " FROM scrag.conversations c WHERE c.session_id = %s"
                " ORDER BY c.updated_at DESC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        return [ConversationRecord(**row) for row in rows]

    def get(
        self, session_id: uuid.UUID, conversation_id: uuid.UUID, *, message_limit: int = 200
    ) -> ConversationDetail:
        with self.db.connection() as conn:
            head = conn.execute(
                "SELECT id, title, created_at, updated_at FROM scrag.conversations"
                " WHERE id = %s AND session_id = %s",
                (conversation_id, session_id),
            ).fetchone()
            if head is None:
                raise NotFoundError("conversation not found in this session")
            rows = conn.execute(
                "SELECT id, role, content, metadata, created_at FROM scrag.messages"
                " WHERE conversation_id = %s AND session_id = %s ORDER BY id LIMIT %s",
                (conversation_id, session_id, message_limit),
            ).fetchall()
        messages = [MessageRecord(**row) for row in rows]
        return ConversationDetail(**head, message_count=len(messages), messages=messages)

    def delete(self, session_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        with self.db.connection() as conn:
            cur = conn.execute(
                "DELETE FROM scrag.conversations WHERE id = %s AND session_id = %s",
                (conversation_id, session_id),
            )
            if not cur.rowcount:
                raise NotFoundError("conversation not found in this session")

    def append_turn(
        self,
        session_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
        question: str,
        answer: str,
        metadata: dict[str, Any],
    ) -> uuid.UUID | None:
        """Record a question and its answer. Creates the conversation when
        ``conversation_id`` is None. Returns the conversation id, or None when the
        session's message quota is used up (nothing is recorded then)."""
        from psycopg.types.json import Jsonb

        with self.db.connection() as conn:
            # Serialize turns per session so the quota check can't be raced.
            conn.execute("SELECT 1 FROM scrag.sessions WHERE id = %s FOR UPDATE", (session_id,))
            used = conn.execute(
                "SELECT count(*) AS n FROM scrag.messages WHERE session_id = %s", (session_id,)
            ).fetchone()["n"]
            if used + 2 > self.max_messages:
                return None
            if conversation_id is None:
                conversation_id = uuid.uuid4()
                conn.execute(
                    "INSERT INTO scrag.conversations (id, session_id, title) VALUES (%s, %s, %s)",
                    (conversation_id, session_id, make_title(question)),
                )
            else:
                cur = conn.execute(
                    "UPDATE scrag.conversations SET updated_at = now()"
                    " WHERE id = %s AND session_id = %s",
                    (conversation_id, session_id),
                )
                if not cur.rowcount:
                    raise NotFoundError("conversation not found in this session")
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO scrag.messages"
                    " (conversation_id, session_id, role, content, metadata)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    [
                        (conversation_id, session_id, "user", question, Jsonb({})),
                        (conversation_id, session_id, "assistant", answer, Jsonb(metadata)),
                    ],
                )
        return conversation_id
