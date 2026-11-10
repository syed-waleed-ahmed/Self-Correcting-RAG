"""Records returned by the persistence layer (also used as API response bodies).

None of these carry secrets or internal error text: session token hashes, raw
document content and job error messages stay in the database.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

DocumentStatus = Literal["queued", "processing", "ready", "failed"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class SessionRecord(BaseModel):
    id: UUID
    kind: Literal["anonymous", "user"] = "anonymous"
    created_at: datetime
    expires_at: datetime


class DocumentRecord(BaseModel):
    id: UUID
    filename: str
    content_hash: str
    size_bytes: int
    status: DocumentStatus
    error: str | None = Field(default=None, description="Stable error code when status=failed.")
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class MessageRecord(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ConversationRecord(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationRecord):
    messages: list[MessageRecord] = Field(default_factory=list)


class JobRecord(BaseModel):
    id: UUID
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    session_id: UUID | None = None
    status: JobStatus
    attempts: int
    max_attempts: int
    run_after: datetime
    locked_by: str | None = None
    last_error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class SessionCreated(BaseModel):
    session_id: UUID
    token: str = Field(description="Send as X-Session-Token. Shown once; store it client-side.")
    expires_at: datetime
    limits: dict[str, int]


class SessionInfo(BaseModel):
    session_id: UUID
    kind: str
    created_at: datetime
    expires_at: datetime
    usage: dict[str, int]
    limits: dict[str, int]


class DocumentUploadResponse(BaseModel):
    document: DocumentRecord
    job_id: UUID | None = Field(description="Ingestion job, or null if this was a duplicate.")
    duplicate: bool = False


class JobView(BaseModel):
    """What a client may see about a job (no payload internals or error text)."""

    id: UUID
    kind: str
    status: JobStatus
    attempts: int
    max_attempts: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, job: JobRecord) -> JobView:
        return cls(**job.model_dump(include=set(cls.model_fields)))
