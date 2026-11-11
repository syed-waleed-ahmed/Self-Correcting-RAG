"""HTTP routes: /health, /demo, /search, /query[/stream], /agent/*, /ingest.

Mounted twice by the app factory: under ``/api/v1`` and at the original
unversioned paths. Session, upload and history routes live in
:mod:`scrag.api.workspace` (``/api/v1`` only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from .. import __version__
from ..container import Container
from ..errors import (
    DatabaseUnavailableError,
    ForbiddenError,
    IndexEmptyError,
    LLMNotConfiguredError,
    PersistenceDisabledError,
    QueryTooLongError,
    ScragError,
    SessionRequiredError,
)
from ..logging_config import get_logger
from ..models import (
    AgentRunRequest,
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
    Scope,
    SearchRequest,
    SearchResponse,
    UsageSummary,
)
from ..observability import metrics
from ..orchestration import AgentRunResult, ToolRegistry, calculator_tool, document_search_tool
from ..retriever import SupportsRetrieve
from ..usage import estimate_cost
from .deps import (
    client_key,
    get_container,
    operator_status,
    optional_session,
    rate_limit,
    require_auth,
    require_auth_unless_demo,
    require_operator,
)
from .streaming import Emit, sse_response

if TYPE_CHECKING:
    from ..persistence import SessionRecord

logger = get_logger(__name__)
router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse} for status in (400, 401, 403, 404, 409, 429, 500, 502, 503)
}

# Fields of a QueryResponse that describe one delivery, not the answer itself.
_DELIVERY_FIELDS = {"cached", "cached_at", "cache_source", "conversation_id", "history_saved"}


def _demo_limits(request: Request, query: str) -> None:
    c = get_container(request)
    if not c.settings.demo_mode:
        return
    if len(query) > c.settings.demo_max_query_chars:
        raise QueryTooLongError(
            f"query of {len(query)} chars exceeds {c.settings.demo_max_query_chars}"
        )
    rate_limit(request)


def _quota_client(request: Request) -> str | None:
    """The key a live run is charged to (None when no per-client quota applies)."""
    return client_key(request) if get_container(request).client_quota is not None else None


def _charge(c: Container, client: str | None, units: int) -> Callable[[], None]:
    """Charge a live run to the client's share and the global budget.

    Returns the matching refund, for runs that end up producing no answer.
    """
    quota = c.client_quota if client is not None else None
    if quota is not None and client is not None:
        quota.consume(client, units)
    if c.budget is not None:
        try:
            c.budget.consume(units)
        except ScragError:
            if quota is not None and client is not None:
                quota.refund(client, units)
            raise

    def refund() -> None:
        if quota is not None and client is not None:
            quota.refund(client, units)
        if c.budget is not None:
            c.budget.refund(units)

    return refund


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


def _usage(raw: dict[str, Any] | None, prices: dict[str, dict[str, float]]) -> UsageSummary | None:
    if raw is None:
        return None
    return UsageSummary(
        **{k: v for k, v in raw.items() if k in UsageSummary.model_fields},
        estimated_cost_usd=estimate_cost(raw.get("by_model", {}), prices),
    )


def _to_response(result: PipelineResult, prices: dict[str, dict[str, float]]) -> QueryResponse:
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
        usage=_usage(result.usage, prices),
        timings_ms=result.timings_ms or None,
    )


def _session_for(request: Request, scope: Scope) -> tuple[SessionRecord | None, bool]:
    """The caller's session (if a token was sent) and whether history is degraded.

    Sample-corpus questions must keep working while the database is down (D2),
    so for ``scope="samples"`` an unavailable database only means the turn
    can't be saved; the other scopes need the database and fail with 503.
    """
    if get_container(request).persistence is None:
        if scope != "samples":
            raise PersistenceDisabledError()
        return None, False
    try:
        return optional_session(request), False
    except DatabaseUnavailableError:
        if scope != "samples":
            raise
        logger.warning("Database unavailable: answering without saving history")
        return None, True


def _resolve_retriever(
    request: Request, scope: Scope, session: SessionRecord | None
) -> SupportsRetrieve | None:
    """The retriever for ``scope`` (None: the default sample-corpus retriever).
    Raises before any LLM work is spent."""
    c = get_container(request)
    if scope == "samples":
        c.refresh_index()  # pick up an index another worker/process just ingested
        if len(c.store) == 0:
            raise IndexEmptyError()
        return None
    if session is None or c.persistence is None:
        raise SessionRequiredError(f"scope={scope!r} needs a session")
    samples: SupportsRetrieve | None = None
    if scope == "all":
        c.refresh_index()
        samples = c.retriever
    return c.persistence.retriever(session.id, samples=samples)


def _record_turn(
    c: Container, session: SessionRecord, body: QueryRequest, response: QueryResponse
) -> QueryResponse:
    """Save the question and answer to the session's history (best effort)."""
    assert c.persistence is not None
    metadata = {
        "scope": response.scope,
        "evaluation_status": response.evaluation_status,
        "evaluation_score": response.evaluation_score,
        "attempts": response.attempts,
        "cached": response.cached,
        "sources": sorted({chunk.source for chunk in response.used_chunks}),
    }
    try:
        conversation_id = c.persistence.conversations.append_turn(
            session.id, body.conversation_id, body.query, response.answer, metadata
        )
    except ScragError as exc:  # the answer is already paid for: still return it
        logger.warning("Could not save conversation turn: %s", exc)
        return response.model_copy(
            update={"conversation_id": body.conversation_id, "history_saved": False}
        )
    return response.model_copy(
        update={"conversation_id": conversation_id, "history_saved": conversation_id is not None}
    )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    # Liveness only: deliberately no index reload, database query or other I/O
    # here, so a slow dependency can never fail a container health check.
    c = get_container(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_configured=c.pipeline is not None,
        embedding_model=c.settings.embedding_model,
        vector_backend=c.settings.vector_backend,
        index_size=len(c.store),
        demo_mode=c.settings.demo_mode,
        persistence="disabled" if c.persistence is None else c.persistence.status,
    )


@router.get(
    "/metrics",
    tags=["ops"],
    dependencies=[Depends(require_operator)],
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}}},
)
def metrics_endpoint() -> Response:
    """Prometheus metrics. Operator-only on the public demo; open otherwise."""
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@router.get("/demo", response_model=DemoInfo, tags=["ops"])
def demo_info(request: Request) -> DemoInfo:
    """What a demo client needs: sample questions, limits and today's budget."""
    c = get_container(request)
    s = c.settings
    budget = c.budget.status() if c.budget else None
    if budget is not None and c.client_quota is not None:
        budget["client_remaining"] = c.client_quota.remaining(client_key(request))
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
            "client_daily_runs": s.demo_client_daily_runs,
            "agent_budget_cost": s.agent_budget_cost,
        }
        if s.demo_mode
        else {},
        budget=budget,
        features={
            "sessions": c.persistence is not None,
            "agent": c.agent is not None,
            "streaming": True,
        },
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth_unless_demo)],
    responses=_ERRORS,
)
def search(request: Request, body: SearchRequest) -> SearchResponse:
    """Retrieval only: ranked chunks for the query, without calling an LLM."""
    c = get_container(request)
    query = body.query.strip()
    _demo_limits(request, query)
    session, _ = _session_for(request, body.scope)
    active = _resolve_retriever(request, body.scope, session) or c.retriever
    results = active.retrieve(query, _top_k(c, body.top_k))
    return SearchResponse(
        query=query,
        retrieval_mode=getattr(active, "mode", "dense"),
        chunks=[_chunk_view(rc) for rc in results],
    )


@dataclass
class _QueryPlan:
    """Everything /query decided before spending any LLM work."""

    body: QueryRequest
    session: SessionRecord | None
    history_degraded: bool
    retriever: SupportsRetrieve | None
    top_k: int
    fingerprint: str | None
    client: str | None = None  # whose daily share a live run is charged to


def _prepare_query(request: Request, body: QueryRequest) -> _QueryPlan:
    """Validation, limits and session checks: these fail as ordinary HTTP errors."""
    c = get_container(request)
    if c.pipeline is None:
        raise LLMNotConfiguredError()
    _demo_limits(request, body.query.strip())
    session, history_degraded = _session_for(request, body.scope)
    retriever = _resolve_retriever(request, body.scope, session)
    if body.conversation_id is not None and not history_degraded:
        if session is None or c.persistence is None:
            raise SessionRequiredError("conversation_id needs a session")
        c.persistence.conversations.ensure(session.id, body.conversation_id)
    return _QueryPlan(
        body=body,
        session=session,
        history_degraded=history_degraded,
        retriever=retriever,
        top_k=_top_k(c, body.top_k),
        # Cached answers exist only for the shared sample corpus.
        fingerprint=c.corpus_fingerprint() if body.scope == "samples" else None,
        client=_quota_client(request),
    )


def _answer(c: Container, plan: _QueryPlan, emit: Emit | None = None) -> QueryResponse:
    """Serve a cached answer or run the pipeline, then record the turn."""
    assert c.pipeline is not None
    body = plan.body
    response: QueryResponse | None = None
    if plan.fingerprint is not None and body.use_cache and c.answers is not None:
        hit = c.answers.get(body.query, plan.top_k, plan.fingerprint)
        metrics.CACHE_LOOKUPS.labels("miss" if hit is None else f"hit_{hit.source}").inc()
        if hit is not None:
            response = QueryResponse(
                **{**hit.response, "query": body.query},
                cached=True,
                cached_at=hit.cached_at,
                cache_source=hit.source,
            )
            if emit is not None:
                emit({"stage": "cache", "status": "hit", "source": hit.source})

    if response is None:
        refund = _charge(c, plan.client, 1)
        try:
            result = c.pipeline.run(
                body.query, top_k=plan.top_k, retriever=plan.retriever, on_event=emit
            )
        except ScragError:
            # The visitor got no answer (e.g. the provider was down or rate limited
            # us), so the run doesn't count against the daily budget.
            refund()
            raise
        if result.attempts == 0:
            refund()  # nothing was retrieved, so no LLM call was made
        response = _to_response(result, c.settings.llm_prices).model_copy(
            update={"scope": body.scope}
        )
        if plan.fingerprint is not None and c.answers is not None:
            c.answers.put(
                body.query,
                plan.top_k,
                plan.fingerprint,
                response.model_dump(exclude=_DELIVERY_FIELDS),
            )

    if plan.session is not None:
        response = _record_turn(c, plan.session, body, response)
    elif plan.history_degraded:
        response = response.model_copy(
            update={"conversation_id": body.conversation_id, "history_saved": False}
        )
    return response


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth_unless_demo)],
    responses=_ERRORS,
)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    return _answer(get_container(request), _prepare_query(request, body))


@router.post(
    "/query/stream",
    tags=["rag"],
    dependencies=[Depends(require_auth_unless_demo)],
    responses={**_ERRORS, 200: {"content": {"text/event-stream": {}}}},
)
def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Like /query, streamed as Server-Sent Events: `progress` per pipeline stage,
    then `result` (the /query response body) or `error`."""
    c = get_container(request)
    plan = _prepare_query(request, body)
    return sse_response(lambda emit: _answer(c, plan, emit).model_dump(mode="json"))


@router.get("/agent/tools", tags=["agent"])
def agent_tools(request: Request) -> list[dict[str, Any]]:
    """The tool catalog: names, descriptions, permission levels and input schemas."""
    tools = get_container(request).tools
    return tools.specs() if tools else []


@dataclass
class _AgentPlan:
    task: str
    approved: frozenset[str]
    registry: ToolRegistry | None
    client: str | None = None


def _prepare_agent(request: Request, body: AgentRunRequest) -> _AgentPlan:
    c = get_container(request)
    if c.agent is None:
        raise LLMNotConfiguredError()
    task = body.task.strip()
    _demo_limits(request, task)
    session, _ = _session_for(request, body.scope)
    retriever = _resolve_retriever(request, body.scope, session)  # refreshes the index
    # On the public demo only the operator may approve sensitive tools.
    approved = frozenset(body.approved_tools)
    if c.settings.demo_mode and operator_status(request) is not True:
        approved = frozenset()
    registry: ToolRegistry | None = None
    if retriever is not None:
        # The visitor's own documents: search them, never anything state-changing.
        registry = ToolRegistry()
        registry.register(document_search_tool(lambda q, k: retriever.retrieve(q, k)))
        registry.register(calculator_tool())
    return _AgentPlan(
        task=task, approved=approved, registry=registry, client=_quota_client(request)
    )


def _run_agent(c: Container, plan: _AgentPlan, emit: Emit | None = None) -> AgentRunResult:
    assert c.agent is not None
    refund = _charge(c, plan.client, c.settings.agent_budget_cost)
    try:
        result = c.agent.run(
            plan.task, approved=plan.approved, registry=plan.registry, on_event=emit
        )
    except ScragError:
        refund()
        raise
    if result.status != "completed":
        refund()  # nothing was answered (approval pending or failure)
    result.usage["estimated_cost_usd"] = estimate_cost(
        result.usage.get("by_model", {}), c.settings.llm_prices
    )
    return result


@router.post(
    "/agent/run",
    response_model=AgentRunResult,
    tags=["agent"],
    dependencies=[Depends(require_auth_unless_demo)],
    responses=_ERRORS,
)
def agent_run(request: Request, body: AgentRunRequest) -> AgentRunResult:
    """Plan -> run tools -> write -> verify, within hard step/LLM/time limits."""
    return _run_agent(get_container(request), _prepare_agent(request, body))


@router.post(
    "/agent/run/stream",
    tags=["agent"],
    dependencies=[Depends(require_auth_unless_demo)],
    responses={**_ERRORS, 200: {"content": {"text/event-stream": {}}}},
)
def agent_run_stream(request: Request, body: AgentRunRequest) -> StreamingResponse:
    """Like /agent/run, streamed as Server-Sent Events: `progress` per trace
    event (plan, tool, synthesize, verify), then `result` or `error`."""
    c = get_container(request)
    plan = _prepare_agent(request, body)
    return sse_response(lambda emit: _run_agent(c, plan, emit).model_dump(mode="json"))


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["rag"],
    dependencies=[Depends(require_auth)],
    responses=_ERRORS,
)
def ingest_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    # On the public demo, ingestion needs proven operator credentials (a wrong
    # credential was already rejected with 401 by require_auth).
    c = get_container(request)
    if c.settings.demo_mode and operator_status(request) is not True:
        raise ForbiddenError("ingest attempted without operator token in demo mode")
    stats = c.run_ingest(reset=body.reset)
    return IngestResponse(
        documents=stats.documents,
        chunks=stats.chunks,
        index_size=stats.index_size,
        added_documents=stats.added_documents,
        updated_documents=stats.updated_documents,
        skipped_documents=stats.skipped_documents,
    )
