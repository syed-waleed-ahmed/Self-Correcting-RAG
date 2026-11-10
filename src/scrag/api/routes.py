"""HTTP routes: /health, /query, /ingest."""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import __version__
from ..container import Container
from ..errors import IndexEmptyError, LLMNotConfiguredError
from ..models import (
    ChunkView,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PipelineResult,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse} for status in (401, 409, 500, 502, 503)
}


def _container(request: Request) -> Container:
    return request.app.state.container


def require_auth(request: Request) -> None:
    """Enforce the optional shared-secret token when one is configured.

    Accepts ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``.
    Uses a constant-time comparison to avoid timing side channels.
    """
    expected = request.app.state.settings.api_auth_token
    if not expected:
        return
    provided = request.headers.get("x-api-key")
    if provided is None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    # Liveness only: deliberately no index reload or other I/O here, so a slow
    # reload can never fail a container health check.
    c = _container(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_configured=c.pipeline is not None,
        embedding_model=c.settings.embedding_model,
        vector_backend=c.settings.vector_backend,
        index_size=len(c.store),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth)],
    responses=_ERRORS,
)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    c = _container(request)
    if c.pipeline is None:
        raise LLMNotConfiguredError()
    c.refresh_index()  # pick up an index another worker/process just ingested
    if len(c.store) == 0:
        raise IndexEmptyError()
    result: PipelineResult = c.pipeline.run(body.query, top_k=body.top_k)
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        evaluation_score=result.evaluation_score,
        evaluation_explanation=result.evaluation_explanation,
        evaluation_status=result.evaluation_status,
        attempts=result.attempts,
        best_attempt=result.best_attempt,
        attempt_history=result.attempt_history,
        used_chunks=[
            ChunkView(
                source=rc.chunk.source,
                text=rc.chunk.text,
                score=rc.score,
                guardrail_score=rc.guardrail_score,
            )
            for rc in result.used_chunks
        ],
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth)],
    responses=_ERRORS,
)
def ingest_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    stats = _container(request).run_ingest(reset=body.reset)
    return IngestResponse(
        documents=stats.documents,
        chunks=stats.chunks,
        index_size=stats.index_size,
        added_documents=stats.added_documents,
        updated_documents=stats.updated_documents,
        skipped_documents=stats.skipped_documents,
    )
