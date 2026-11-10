"""Uploaded documents, their chunks, and session-scoped search over them.

Isolation: every read and write filters on ``session_id`` (chunks carry it
denormalized, so a search can never join into another session's rows), and a
document from another session is reported as not found.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import NotFoundError, SessionQuotaExceededError, StorageFullError
from ..sparse import tokenize
from .database import Database
from .jobs import JobQueue
from .records import DocumentRecord

INGEST_JOB = "ingest_document"
_DOC_COLUMNS = (
    "id, filename, content_hash, size_bytes, status, error, chunk_count, created_at, updated_at"
)
_MAX_QUERY_TERMS = 32


def vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    """pgvector's text form, e.g. ``[0.1,0.2]`` (avoids a client-side type adapter)."""
    return "[" + ",".join(format(float(x), ".8g") for x in vector) + "]"


def keyword_query(text: str) -> str | None:
    """An OR-of-terms ``to_tsquery`` string, or None if the text has no usable terms.

    Terms come from the same tokenizer as the in-process BM25 index, reduced to
    ``[a-z0-9]`` so no tsquery operator can be injected. OR semantics (not
    ``websearch_to_tsquery``'s AND) because questions rarely contain every term
    of the passage that answers them.
    """
    terms: list[str] = []
    for token in tokenize(text):
        for part in re.findall(r"[a-z0-9]+", token):
            if part not in terms:
                terms.append(part)
    return " | ".join(terms[:_MAX_QUERY_TERMS]) or None


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    position: int
    text: str
    score: float


class DocumentStore:
    def __init__(
        self,
        db: Database,
        jobs: JobQueue,
        *,
        max_per_session: int = 5,
        max_total: int = 300,
    ) -> None:
        self.db = db
        self.jobs = jobs
        self.max_per_session = max_per_session
        self.max_total = max_total

    # --- visitor-facing -------------------------------------------------------------
    def create(
        self,
        session_id: uuid.UUID,
        *,
        filename: str,
        content: str,
        content_hash: str,
        size_bytes: int,
    ) -> tuple[DocumentRecord, uuid.UUID | None]:
        """Store an upload and enqueue its ingestion in one transaction.

        Returns ``(document, job_id)``. Uploading the same content twice in a
        session returns the existing document and ``job_id=None``.
        """
        with self.db.connection() as conn:
            # Serialize uploads per session so the quota check can't be raced.
            conn.execute("SELECT 1 FROM scrag.sessions WHERE id = %s FOR UPDATE", (session_id,))
            existing = conn.execute(
                f"SELECT {_DOC_COLUMNS} FROM scrag.documents"
                " WHERE session_id = %s AND content_hash = %s",
                (session_id, content_hash),
            ).fetchone()
            if existing is not None:
                return DocumentRecord(**existing), None
            counts = conn.execute(
                "SELECT (SELECT count(*) FROM scrag.documents WHERE session_id = %s) AS mine,"
                " (SELECT count(*) FROM scrag.documents) AS total",
                (session_id,),
            ).fetchone()
            if counts["mine"] >= self.max_per_session:
                raise SessionQuotaExceededError(f"session has {counts['mine']} documents")
            if counts["total"] >= self.max_total:
                raise StorageFullError(f"{counts['total']} documents stored in total")
            row = conn.execute(
                "INSERT INTO scrag.documents"
                " (id, session_id, filename, content_hash, size_bytes, content)"
                " VALUES (%s, %s, %s, %s, %s, %s)"
                f" RETURNING {_DOC_COLUMNS}",
                (uuid.uuid4(), session_id, filename, content_hash, size_bytes, content),
            ).fetchone()
            document = DocumentRecord(**row)
            job_id = self.jobs.enqueue(
                INGEST_JOB,
                {"document_id": str(document.id)},
                session_id=session_id,
                conn=conn,
            )
        self.jobs.notify()  # after commit: the worker can now see the job
        return document, job_id

    def for_session(self, session_id: uuid.UUID) -> list[DocumentRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT {_DOC_COLUMNS} FROM scrag.documents WHERE session_id = %s"
                " ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [DocumentRecord(**row) for row in rows]

    def get(self, session_id: uuid.UUID, document_id: uuid.UUID) -> DocumentRecord:
        with self.db.connection() as conn:
            row = conn.execute(
                f"SELECT {_DOC_COLUMNS} FROM scrag.documents WHERE id = %s AND session_id = %s",
                (document_id, session_id),
            ).fetchone()
        if row is None:
            raise NotFoundError("document not found in this session")
        return DocumentRecord(**row)

    def delete(self, session_id: uuid.UUID, document_id: uuid.UUID) -> None:
        with self.db.connection() as conn:
            cur = conn.execute(
                "DELETE FROM scrag.documents WHERE id = %s AND session_id = %s",
                (document_id, session_id),
            )
            if not cur.rowcount:
                raise NotFoundError("document not found in this session")

    def ready_count(self, session_id: uuid.UUID) -> int:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM scrag.documents"
                " WHERE session_id = %s AND status = 'ready'",
                (session_id,),
            ).fetchone()
        return int(row["n"])

    # --- worker-facing -----------------------------------------------------------------
    def start_processing(self, document_id: uuid.UUID) -> tuple[uuid.UUID, str] | None:
        """Mark a document as processing; returns ``(session_id, text)`` or None if deleted."""
        with self.db.connection() as conn:
            row = conn.execute(
                "UPDATE scrag.documents SET status = 'processing', error = NULL,"
                " updated_at = now() WHERE id = %s RETURNING session_id, content",
                (document_id,),
            ).fetchone()
        return (row["session_id"], row["content"]) if row else None

    def store_chunks(
        self,
        document_id: uuid.UUID,
        texts: list[str],
        vectors: np.ndarray,
        *,
        embedding_model: str,
    ) -> int | None:
        """Replace a document's chunks and mark it ready (idempotent: safe to retry).

        Returns the chunk count, or None if the document was deleted meanwhile.
        """
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")
        with self.db.connection() as conn:
            doc = conn.execute(
                "SELECT session_id FROM scrag.documents WHERE id = %s FOR UPDATE", (document_id,)
            ).fetchone()
            if doc is None:
                return None
            conn.execute("DELETE FROM scrag.chunks WHERE document_id = %s", (document_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO scrag.chunks"
                    " (document_id, session_id, position, text, embedding, embedding_model)"
                    " VALUES (%s, %s, %s, %s, %s::vector, %s)",
                    [
                        (
                            document_id,
                            doc["session_id"],
                            position,
                            text,
                            vector_literal(vectors[position]),
                            embedding_model,
                        )
                        for position, text in enumerate(texts)
                    ],
                )
            conn.execute(
                "UPDATE scrag.documents SET status = 'ready', error = NULL, chunk_count = %s,"
                " updated_at = now() WHERE id = %s",
                (len(texts), document_id),
            )
        return len(texts)

    def mark_failed(self, document_id: uuid.UUID, code: str) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE scrag.documents SET status = 'failed', error = %s, updated_at = now()"
                " WHERE id = %s",
                (code[:64], document_id),
            )

    # --- search ------------------------------------------------------------------------
    def search_dense(
        self,
        session_id: uuid.UUID,
        query_vector: Sequence[float] | np.ndarray,
        k: int,
        *,
        embedding_model: str,
    ) -> list[ChunkHit]:
        """Cosine-nearest chunks among the session's ready documents (exact scan)."""
        literal = vector_literal(query_vector)
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT c.id AS chunk_id, c.document_id, d.filename, c.position, c.text,"
                " 1 - (c.embedding <=> %(q)s::vector) AS score"
                " FROM scrag.chunks c JOIN scrag.documents d ON d.id = c.document_id"
                " WHERE c.session_id = %(s)s AND d.session_id = %(s)s"
                "   AND c.embedding_model = %(m)s AND d.status = 'ready'"
                " ORDER BY c.embedding <=> %(q)s::vector, c.id LIMIT %(k)s",
                {"q": literal, "s": session_id, "m": embedding_model, "k": k},
            ).fetchall()
        # Cosine distance involving a zero vector is NaN in pgvector; never let
        # a NaN reach ranking or a JSON response.
        return [
            ChunkHit(**{**row, "score": row["score"] if math.isfinite(row["score"]) else 0.0})
            for row in rows
        ]

    def search_keyword(self, session_id: uuid.UUID, query: str, k: int) -> list[ChunkHit]:
        """Full-text matches (OR of query terms) ranked by ``ts_rank_cd``."""
        terms = keyword_query(query)
        if terms is None:
            return []
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT c.id AS chunk_id, c.document_id, d.filename, c.position, c.text,"
                " ts_rank_cd(c.tsv, q.query) AS score"
                " FROM scrag.chunks c JOIN scrag.documents d ON d.id = c.document_id,"
                " to_tsquery('english', %(t)s) AS q(query)"
                " WHERE c.session_id = %(s)s AND d.session_id = %(s)s"
                "   AND d.status = 'ready' AND c.tsv @@ q.query"
                " ORDER BY score DESC, c.id LIMIT %(k)s",
                {"t": terms, "s": session_id, "k": k},
            ).fetchall()
        return [ChunkHit(**row) for row in rows]
