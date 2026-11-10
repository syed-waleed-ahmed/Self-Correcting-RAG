"""Session-scoped routes (Phase 4): sessions, uploads, jobs, conversation history.

All of them need a configured database (else 503 ``persistence_disabled``) and,
except ``POST /sessions``, an ``X-Session-Token``. Resources of other sessions
answer 404, exactly like resources that don't exist.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile

from ..errors import (
    DocumentTooLongError,
    NotFoundError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    UploadTooLargeError,
)
from ..ingest import content_hash
from ..loaders import SUPPORTED_SUFFIXES, LoaderError, extract_upload, safe_filename
from ..models import ErrorResponse
from ..persistence.records import (
    ConversationDetail,
    ConversationRecord,
    DocumentRecord,
    DocumentUploadResponse,
    JobView,
    SessionCreated,
    SessionInfo,
)
from .deps import get_container, get_persistence, rate_limit, require_operator, require_session

router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse} for status in (401, 404, 409, 413, 415, 422, 429, 503)
}


def _limits(request: Request) -> dict[str, int]:
    s = get_container(request).settings
    return {
        "max_documents": s.session_max_documents,
        "upload_max_bytes": s.upload_max_bytes,
        "document_max_chunks": s.document_max_chunks,
        "ttl_hours": int(s.session_ttl_hours),
    }


# --- sessions -------------------------------------------------------------------------
@router.post(
    "/sessions",
    response_model=SessionCreated,
    status_code=201,
    tags=["sessions"],
    responses=_ERRORS,
)
def create_session(request: Request) -> SessionCreated:
    """Start an anonymous session. Everything in it is deleted when it expires."""
    persistence = get_persistence(request)
    rate_limit(request)
    record, token = persistence.sessions.create()
    persistence.worker.wake()  # a good moment to purge expired sessions, if due
    return SessionCreated(
        session_id=record.id, token=token, expires_at=record.expires_at, limits=_limits(request)
    )


@router.get("/sessions/me", response_model=SessionInfo, tags=["sessions"], responses=_ERRORS)
def get_session(request: Request) -> SessionInfo:
    session = require_session(request)
    usage = get_persistence(request).sessions.usage(session.id)
    return SessionInfo(
        session_id=session.id,
        kind=session.kind,
        created_at=session.created_at,
        expires_at=session.expires_at,
        usage={key: int(value) for key, value in usage.items()},
        limits=_limits(request),
    )


@router.delete("/sessions/me", status_code=204, tags=["sessions"], responses=_ERRORS)
def delete_session(request: Request) -> Response:
    """Delete the session and all of its documents, history and jobs, immediately."""
    session = require_session(request)
    get_persistence(request).sessions.delete(session.id)
    return Response(status_code=204)


# --- documents ------------------------------------------------------------------------
@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=202,
    tags=["documents"],
    responses=_ERRORS,
)
def upload_document(
    request: Request, response: Response, file: UploadFile = File(...)
) -> DocumentUploadResponse:
    """Upload a document to this session. Text is extracted now; chunking and
    embedding run as a background job (poll ``GET /jobs/{job_id}``)."""
    session = require_session(request)
    persistence = get_persistence(request)
    rate_limit(request)
    settings = get_container(request).settings

    filename = safe_filename(file.filename)
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(f"suffix {suffix!r}")
    data = file.file.read(settings.upload_max_bytes + 1)
    if len(data) > settings.upload_max_bytes:
        raise UploadTooLargeError(f"upload exceeds {settings.upload_max_bytes} bytes")
    try:
        text = extract_upload(
            data,
            filename,
            max_bytes=settings.upload_max_bytes,
            max_pdf_pages=min(settings.max_pdf_pages, 50),
        )
    except LoaderError as exc:
        raise UnreadableDocumentError(str(exc)) from exc
    if not text:
        raise UnreadableDocumentError("no text extracted")
    # Reject oversized documents now instead of failing the job later.
    if len(get_container(request).chunker.split(text)) > settings.document_max_chunks:
        raise DocumentTooLongError(f"more than {settings.document_max_chunks} chunks")

    document, job_id = persistence.documents.create(
        session.id,
        filename=filename,
        content=text,
        content_hash=content_hash(text),
        size_bytes=len(data),
    )
    if job_id is None:
        response.status_code = 200  # duplicate: nothing new to process
    return DocumentUploadResponse(document=document, job_id=job_id, duplicate=job_id is None)


@router.get(
    "/documents", response_model=list[DocumentRecord], tags=["documents"], responses=_ERRORS
)
def list_documents(request: Request) -> list[DocumentRecord]:
    session = require_session(request)
    return get_persistence(request).documents.for_session(session.id)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentRecord,
    tags=["documents"],
    responses=_ERRORS,
)
def get_document(request: Request, document_id: UUID) -> DocumentRecord:
    session = require_session(request)
    return get_persistence(request).documents.get(session.id, document_id)


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"], responses=_ERRORS)
def delete_document(request: Request, document_id: UUID) -> Response:
    session = require_session(request)
    get_persistence(request).documents.delete(session.id, document_id)
    return Response(status_code=204)


@router.get("/jobs/{job_id}", response_model=JobView, tags=["documents"], responses=_ERRORS)
def get_job(request: Request, job_id: UUID) -> JobView:
    session = require_session(request)
    job = get_persistence(request).jobs.get(job_id, session_id=session.id)
    if job is None:
        raise NotFoundError("job not found in this session")
    return JobView.of(job)


# --- conversations --------------------------------------------------------------------
@router.get(
    "/conversations",
    response_model=list[ConversationRecord],
    tags=["conversations"],
    responses=_ERRORS,
)
def list_conversations(request: Request) -> list[ConversationRecord]:
    session = require_session(request)
    return get_persistence(request).conversations.for_session(session.id)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    tags=["conversations"],
    responses=_ERRORS,
)
def get_conversation(request: Request, conversation_id: UUID) -> ConversationDetail:
    session = require_session(request)
    return get_persistence(request).conversations.get(session.id, conversation_id)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    tags=["conversations"],
    responses=_ERRORS,
)
def delete_conversation(request: Request, conversation_id: UUID) -> Response:
    session = require_session(request)
    get_persistence(request).conversations.delete(session.id, conversation_id)
    return Response(status_code=204)


# --- operator -------------------------------------------------------------------------
@router.get(
    "/admin/stats",
    tags=["ops"],
    dependencies=[Depends(require_operator)],
    responses=_ERRORS,
)
def admin_stats(request: Request) -> dict[str, Any]:
    """Row counts and database status (operator only; never any contents)."""
    persistence = get_persistence(request)
    return {"database": persistence.status, **persistence.sessions.stats()}
