"""HTTP routes: /health, /query, /ingest."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import __version__
from ..container import Container
from ..ingest import ingest
from ..models import (
    ChunkView,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PipelineResult,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()


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
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
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
)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    c = _container(request)
    if c.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Set GROQ_API_KEY to enable /query.",
        )
    if len(c.store) == 0:
        raise HTTPException(
            status_code=409,
            detail="Index is empty. Ingest documents first (POST /ingest).",
        )
    result: PipelineResult = c.pipeline.run(body.query, top_k=body.top_k)
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        evaluation_score=result.evaluation_score,
        evaluation_explanation=result.evaluation_explanation,
        attempts=result.attempts,
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
)
def ingest_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    c = _container(request)
    stats = ingest(
        embedder=c.embedder,
        store=c.store,
        chunker=c.chunker,
        docs_dir=c.settings.docs_dir,
        batch_size=c.settings.embedding_batch_size,
        reset=body.reset,
    )
    return IngestResponse(
        documents=stats.documents,
        chunks=stats.chunks,
        index_size=stats.index_size,
    )
