"""Postgres-backed persistence: sessions, uploads, history, jobs and quotas (Phase 4).

Optional: the API runs without it (``SCRAG_DATABASE_URL`` unset), and only the
session, upload and history endpoints need it. Requires the ``postgres`` extra.
"""

from .budget import PersistentDailyBudget
from .conversations import ConversationStore
from .database import Database
from .documents import DocumentStore
from .jobs import JobHandler, JobQueue, JobWorker, PermanentJobError
from .migrations import apply_migrations, current_version, rollback_migrations
from .records import (
    ConversationDetail,
    ConversationRecord,
    DocumentRecord,
    JobRecord,
    JobView,
    MessageRecord,
    SessionRecord,
)
from .retrieval import SessionRetriever
from .service import Persistence, build_persistence
from .sessions import SessionStore

__all__ = [
    "ConversationDetail",
    "ConversationRecord",
    "ConversationStore",
    "Database",
    "DocumentRecord",
    "DocumentStore",
    "JobHandler",
    "JobQueue",
    "JobRecord",
    "JobView",
    "JobWorker",
    "MessageRecord",
    "PermanentJobError",
    "Persistence",
    "PersistentDailyBudget",
    "SessionRecord",
    "SessionRetriever",
    "SessionStore",
    "apply_migrations",
    "current_version",
    "build_persistence",
    "rollback_migrations",
]
