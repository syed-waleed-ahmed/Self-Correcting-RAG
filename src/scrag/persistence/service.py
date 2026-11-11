"""Wiring for the persistence layer: stores, the job worker and the ingest handler."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from ..chunking import Chunker
from ..config import Settings
from ..embeddings import Embedder, embed_checked
from ..logging_config import get_logger
from ..retriever import SupportsRetrieve
from .budget import purge_client_counters
from .conversations import ConversationStore
from .database import Database, DatabaseStatus
from .documents import INGEST_JOB, DocumentStore
from .jobs import JobHandler, JobQueue, JobWorker, PermanentJobError
from .records import JobRecord
from .retrieval import SessionRetriever
from .sessions import SessionStore

logger = get_logger(__name__)


@dataclass
class Persistence:
    settings: Settings
    db: Database
    sessions: SessionStore
    documents: DocumentStore
    conversations: ConversationStore
    jobs: JobQueue
    worker: JobWorker
    embedder: Embedder

    def start(self) -> None:
        """Open the pool and start the worker. Never blocks on, or fails for, the database."""
        self.db.open()
        if self.settings.job_worker_enabled:
            self.worker.start()  # its first step connects and migrates, off the startup path

    def stop(self) -> None:
        self.worker.stop()
        self.db.close()

    @property
    def status(self) -> DatabaseStatus:
        return self.db.status

    def retriever(
        self, session_id: uuid.UUID, *, samples: SupportsRetrieve | None = None
    ) -> SessionRetriever:
        s = self.settings
        return SessionRetriever(
            self.embedder,
            self.documents,
            session_id,
            embedding_model=s.embedding_model,
            embedding_dim=s.embedding_dim,
            default_top_k=s.top_k,
            candidate_pool=s.hybrid_candidate_pool,
            rrf_k=s.rrf_k,
            samples=samples,
        )

    def maintenance(self) -> None:
        removed = self.sessions.purge_expired()
        if removed:
            logger.info("Deleted %d expired sessions and their data", removed)
        counters = purge_client_counters(self.db)
        if counters:
            logger.info("Deleted %d per-client quota counters from earlier days", counters)


def ingest_document_handler(
    documents: DocumentStore, embedder: Embedder, chunker: Chunker, settings: Settings
) -> JobHandler:
    """Chunk and embed one uploaded document, then publish its chunks (idempotent)."""

    def run(job: JobRecord) -> dict[str, object]:
        document_id = uuid.UUID(str(job.payload["document_id"]))
        started = documents.start_processing(document_id)
        if started is None:
            return {"skipped": "document deleted"}
        _, text = started
        pieces = chunker.split(text)
        if not pieces:
            raise PermanentJobError("empty_document")
        if len(pieces) > settings.document_max_chunks:
            raise PermanentJobError("document_too_long")
        batch = settings.embedding_batch_size
        vectors = np.vstack(
            [
                embed_checked(embedder, pieces[i : i + batch], settings.embedding_dim)
                for i in range(0, len(pieces), batch)
            ]
        )
        stored = documents.store_chunks(
            document_id, pieces, vectors, embedding_model=settings.embedding_model
        )
        if stored is None:
            return {"skipped": "document deleted"}
        return {"chunks": stored}

    def on_failure(job: JobRecord, exc: BaseException) -> None:
        code = exc.code if isinstance(exc, PermanentJobError) else "processing_failed"
        documents.mark_failed(uuid.UUID(str(job.payload["document_id"])), code)

    return JobHandler(run=run, on_failure=on_failure)


def build_persistence(
    settings: Settings,
    *,
    embedder: Embedder,
    chunker: Chunker,
    database: Database | None = None,
) -> Persistence:
    db = database or Database(
        settings.database_url or "",
        pool_max=settings.db_pool_max,
        connect_timeout_s=settings.db_connect_timeout_s,
        retry_interval_s=settings.db_retry_interval_s,
        migrate=settings.db_migrate_on_start,
    )
    jobs = JobQueue(db, max_attempts=settings.job_max_attempts)
    documents = DocumentStore(
        db,
        jobs,
        max_per_session=settings.session_max_documents,
        max_total=settings.persistence_max_documents,
    )
    sessions = SessionStore(
        db, ttl_hours=settings.session_ttl_hours, max_live=settings.persistence_max_sessions
    )
    worker = JobWorker(
        jobs,
        {INGEST_JOB: ingest_document_handler(documents, embedder, chunker, settings)},
        lease_s=settings.job_lease_seconds,
        poll_interval_s=settings.job_poll_interval_s,
        maintenance_interval_s=settings.session_cleanup_interval_s,
    )
    persistence = Persistence(
        settings=settings,
        db=db,
        sessions=sessions,
        documents=documents,
        conversations=ConversationStore(db),
        jobs=jobs,
        worker=worker,
        embedder=embedder,
    )
    worker.maintenance = persistence.maintenance
    return persistence
