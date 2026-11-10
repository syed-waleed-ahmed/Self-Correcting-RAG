"""HTTP routes: /health, /demo, /search, /query, /ingest.

Mounted twice by the app factory: under ``/api/v1`` and at the original
unversioned paths.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import __version__
from ..container import Container
from ..errors import (
    ForbiddenError,
    IndexEmptyError,
    LLMNotConfiguredError,
    QueryTooLongError,
    ScragError,
)
from ..models import (
    ChunkView,
    DemoInfo,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PipelineResult,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse} for status in (400, 401, 403, 409, 429, 500, 502, 503)
}


def _container(request: Request) -> Container:
    return request.app.state.container


def _token_ok(request: Request) -> bool | None:
    """True/False if a token is configured and presented correctly; None if no token is set."""
    expected = request.app.state.settings.api_auth_token
    if not expected:
        return None
    provided = request.headers.get("x-api-key")
    if provided is None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided:
        return False
    return secrets.compare_digest(provided.encode(), expected.encode())


def require_auth(request: Request) -> None:
    """Enforce the optional shared-secret token when one is configured.

    Accepts ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``.
    Uses a constant-time comparison to avoid timing side channels.
    """
    if _token_ok(request) is False:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def client_key(request: Request) -> str:
    """Rate-limit key: the client IP, read from a trusted proxy header if configured."""
    settings = request.app.state.settings
    header = settings.forwarded_ip_header
    if header:
        value = request.headers.get(header, "")
        hops = [part.strip() for part in value.split(",") if part.strip()]
        if len(hops) >= settings.forwarded_ip_hops:
            return hops[-settings.forwarded_ip_hops]
    return request.client.host if request.client else "unknown"


def _demo_limits(request: Request, query: str) -> None:
    c = _container(request)
    if not c.settings.demo_mode:
        return
    if len(query) > c.settings.demo_max_query_chars:
        raise QueryTooLongError(
            f"query of {len(query)} chars exceeds {c.settings.demo_max_query_chars}"
        )
    if c.rate_limiter is not None:
        c.rate_limiter.check(client_key(request))


def _top_k(c: Container, requested: int | None) -> int:
    k = requested or c.settings.top_k
    return min(k, c.settings.demo_top_k) if c.settings.demo_mode else k


def _chunk_view(rc: RetrievedChunk) -> ChunkView:
    return ChunkView(
        source=rc.chunk.source,
        text=rc.chunk.text,
        score=rc.score,
        fusion_score=rc.fusion_score,
        keyword_score=rc.keyword_score,
        guardrail_score=rc.guardrail_score,
    )


def _to_response(result: PipelineResult) -> QueryResponse:
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        evaluation_score=result.evaluation_score,
        evaluation_explanation=result.evaluation_explanation,
        evaluation_status=result.evaluation_status,
        attempts=result.attempts,
        best_attempt=result.best_attempt,
        attempt_history=result.attempt_history,
        used_chunks=[_chunk_view(rc) for rc in result.used_chunks],
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
        demo_mode=c.settings.demo_mode,
    )


@router.get("/demo", response_model=DemoInfo, tags=["ops"])
def demo_info(request: Request) -> DemoInfo:
    """What a demo client needs: sample questions, limits and today's budget."""
    c = _container(request)
    s = c.settings
    return DemoInfo(
        demo_mode=s.demo_mode,
        llm_configured=c.pipeline is not None,
        model=s.generator_model,
        retrieval_mode=s.retrieval_mode,
        sample_questions=c.answers.sample_questions if c.answers else [],
        limits={
            "max_query_chars": s.demo_max_query_chars,
            "top_k": s.demo_top_k,
            "rate_limit_per_minute": s.demo_rate_limit_per_minute,
            "rate_limit_burst": s.demo_rate_limit_burst,
            "max_self_correct_steps": s.max_self_correct_steps,
        }
        if s.demo_mode
        else {},
        budget=c.budget.status() if c.budget else None,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth)],
    responses=_ERRORS,
)
def search(request: Request, body: SearchRequest) -> SearchResponse:
    """Retrieval only: ranked chunks for the query, without calling an LLM."""
    c = _container(request)
    query = body.query.strip()
    _demo_limits(request, query)
    c.refresh_index()
    if len(c.store) == 0:
        raise IndexEmptyError()
    results = c.retriever.retrieve(query, _top_k(c, body.top_k))
    return SearchResponse(
        query=query,
        retrieval_mode=getattr(c.retriever, "mode", "dense"),
        chunks=[_chunk_view(rc) for rc in results],
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
    _demo_limits(request, body.query.strip())
    c.refresh_index()  # pick up an index another worker/process just ingested
    if len(c.store) == 0:
        raise IndexEmptyError()
    top_k = _top_k(c, body.top_k)
    fingerprint = c.corpus_fingerprint()

    if body.use_cache and c.answers is not None:
        hit = c.answers.get(body.query, top_k, fingerprint)
        if hit is not None:
            return QueryResponse(
                **{**hit.response, "query": body.query},
                cached=True,
                cached_at=hit.cached_at,
                cache_source=hit.source,
            )

    if c.budget is not None:
        c.budget.consume()
    try:
        result = c.pipeline.run(body.query, top_k=top_k)
    except ScragError:
        # The visitor got no answer (e.g. the provider was down or rate limited
        # us), so the run doesn't count against the daily budget.
        if c.budget is not None:
            c.budget.refund()
        raise
    response = _to_response(result)
    if c.answers is not None:
        c.answers.put(
            body.query,
            top_k,
            fingerprint,
            response.model_dump(exclude={"cached", "cached_at", "cache_source"}),
        )
    return response


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth)],
    responses=_ERRORS,
)
def ingest_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    # On the public demo, ingestion needs the operator token: a configured
    # token that was presented (require_auth already rejected a wrong one).
    if _container(request).settings.demo_mode and _token_ok(request) is not True:
        raise ForbiddenError("ingest attempted without operator token in demo mode")
    stats = _container(request).run_ingest(reset=body.reset)
    return IngestResponse(
        documents=stats.documents,
        chunks=stats.chunks,
        index_size=stats.index_size,
        added_documents=stats.added_documents,
        updated_documents=stats.updated_documents,
        skipped_documents=stats.skipped_documents,
    )
