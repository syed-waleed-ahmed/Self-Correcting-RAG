"""Composition root: build and wire all components from Settings.

Centralizing construction here keeps the API and CLI thin and makes the whole
graph trivially swappable in tests (build a container with fakes and inject it).
The embedder is constructed eagerly but loads its model lazily, so building a
container never imports torch. The pipeline is only built when an LLM key is
configured; without one, retrieval/ingestion still work and ``/query`` reports
a clear 503.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .agents import EvaluatorAgent, GeneratorAgent, GuardrailAgent
from .auth import JWTVerifier, build_verifier
from .chunking import Chunker
from .config import Settings
from .demo import AnswerCache, Budget, DailyBudget, RateLimiter, corpus_fingerprint
from .embeddings import Embedder, create_embedder
from .errors import IngestInProgressError
from .ingest import IngestStats, ingest
from .llm.client import ChatClient, OpenAICompatibleClient
from .logging_config import get_logger
from .orchestration import (
    AgentWorkflow,
    ToolRegistry,
    WorkflowLimits,
    calculator_tool,
    document_search_tool,
    reindex_tool,
)
from .pipeline import SelfCorrectingRAGPipeline
from .retriever import HybridRetriever, Retriever
from .vectorstore import create_vector_store
from .vectorstore.base import VectorStore

if TYPE_CHECKING:
    from .persistence import Database, Persistence

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    chunker: Chunker
    retriever: Retriever
    pipeline: SelfCorrectingRAGPipeline | None  # None when no LLM key configured
    agent: AgentWorkflow | None = None  # None when no LLM key configured
    tools: ToolRegistry | None = None
    answers: AnswerCache | None = None
    rate_limiter: RateLimiter | None = None  # demo mode only
    budget: Budget | None = None  # demo mode only (persisted when a database is configured)
    persistence: Persistence | None = None  # None when no database is configured
    jwt_verifier: JWTVerifier | None = None  # None unless JWT admin auth is configured
    # One ingestion at a time per process; concurrent requests get a 409.
    ingest_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _fingerprint: tuple[int, str] | None = field(default=None, repr=False)

    def corpus_fingerprint(self) -> str:
        """Fingerprint of the indexed corpus, cached per store revision."""
        revision, _, chunks = self.store.versioned_snapshot()
        cached = self._fingerprint
        if cached is not None and revision != -1 and cached[0] == revision:
            return cached[1]
        value = corpus_fingerprint(chunks)
        self._fingerprint = (revision, value)
        return value

    def load_index(self) -> bool:
        """Load a persisted index into the store if one exists."""
        try:
            return self.store.load()
        except Exception as exc:  # noqa: BLE001 - surface but don't crash startup
            logger.warning("Could not load persisted index: %s", exc)
            return False

    def refresh_index(self) -> bool:
        """Pick up an index persisted by another process (e.g. another API worker).

        On failure the currently loaded index keeps serving.
        """
        try:
            return self.store.refresh_if_stale()
        except Exception as exc:  # noqa: BLE001 - degrade to the loaded index
            logger.warning("Could not refresh index; serving the loaded one: %s", exc)
            return False

    def run_ingest(self, *, reset: bool = True) -> IngestStats:
        """Ingest the configured docs directory, rejecting concurrent runs."""
        if not self.ingest_lock.acquire(blocking=False):
            raise IngestInProgressError()
        try:
            if not reset:
                # Upsert against the newest persisted index, not a stale copy
                # (another worker may have ingested since this one loaded).
                self.refresh_index()
            return ingest(
                embedder=self.embedder,
                store=self.store,
                chunker=self.chunker,
                docs_dir=self.settings.docs_dir,
                batch_size=self.settings.embedding_batch_size,
                reset=reset,
                max_document_bytes=self.settings.max_document_bytes,
                max_pdf_pages=self.settings.max_pdf_pages,
            )
        finally:
            self.ingest_lock.release()


def build_retriever(
    settings: Settings, embedder: Embedder, store: VectorStore, mode: str | None = None
) -> Retriever:
    """Build the retriever for ``mode`` (defaults to ``settings.retrieval_mode``)."""
    if (mode or settings.retrieval_mode) == "hybrid":
        return HybridRetriever(
            embedder,
            store,
            default_top_k=settings.top_k,
            candidate_pool=settings.hybrid_candidate_pool,
            rrf_k=settings.rrf_k,
        )
    return Retriever(embedder, store, default_top_k=settings.top_k)


def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    chat_client: ChatClient | None = None,
    database: Database | None = None,
) -> Container:
    """Construct a fully wired :class:`Container`.

    The optional overrides exist so tests (and alternative deployments) can inject
    fakes without monkeypatching. Building never connects to the database.
    """
    embedder = embedder or create_embedder(settings)
    store = store or create_vector_store(settings)
    chunker = Chunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    retriever = build_retriever(settings, embedder, store)

    pipeline: SelfCorrectingRAGPipeline | None = None
    client = chat_client
    if client is None and settings.has_llm:
        client = OpenAICompatibleClient(
            api_key=settings.llm_api_key or "",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    if client is not None:
        pipeline = SelfCorrectingRAGPipeline(
            retriever=retriever,
            guardrail=GuardrailAgent(
                client,
                settings.guardrail_model,
                threshold=settings.guardrail_threshold,
                max_workers=settings.guardrail_max_workers,
                max_tokens=settings.guardrail_max_tokens,
                mode=settings.guardrail_mode,
            ),
            generator=GeneratorAgent(client, settings.generator_model),
            evaluator=EvaluatorAgent(client, settings.evaluator_model),
            top_k=settings.top_k,
            eval_threshold=settings.eval_threshold,
            max_self_correct_steps=settings.max_self_correct_steps,
        )
    else:
        logger.warning("No LLM key configured — /query will be unavailable.")

    tools = ToolRegistry()
    tools.register(document_search_tool(lambda q, k: retriever.retrieve(q, k)))
    tools.register(calculator_tool())

    answers = AnswerCache(
        ttl_seconds=settings.answer_cache_ttl_seconds, max_entries=settings.answer_cache_size
    )
    try:
        loaded = answers.load_samples(settings.samples_path)
        if loaded:
            logger.info("Loaded %d recorded sample answers", loaded)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not load sample answers from %s: %s", settings.samples_path, exc)

    persistence: Persistence | None = None
    budget: Budget | None = None
    if database is not None or settings.has_persistence:
        from .persistence import PersistentDailyBudget, build_persistence

        persistence = build_persistence(
            settings, embedder=embedder, chunker=chunker, database=database
        )
        if settings.demo_mode:  # shared by every instance through Postgres
            budget = PersistentDailyBudget(persistence.db, settings.demo_daily_query_budget)
    elif settings.demo_mode:
        budget = DailyBudget(settings.demo_daily_query_budget)

    container = Container(
        settings=settings,
        embedder=embedder,
        store=store,
        chunker=chunker,
        retriever=retriever,
        pipeline=pipeline,
        tools=tools,
        answers=answers,
        rate_limiter=(
            RateLimiter(settings.demo_rate_limit_per_minute, settings.demo_rate_limit_burst)
            if settings.demo_mode
            else None
        ),
        budget=budget,
        persistence=persistence,
        jwt_verifier=build_verifier(settings),
    )
    # Sensitive, state-changing tool: registered everywhere (so the approval
    # flow is visible) but disabled on the public demo.
    tools.register(
        reindex_tool(
            lambda reset: container.run_ingest(reset=reset), enabled=not settings.demo_mode
        )
    )
    if client is not None and pipeline is not None:
        container.agent = AgentWorkflow(
            client,
            settings.generator_model,
            tools,
            pipeline.evaluator,
            limits=WorkflowLimits(
                max_steps=settings.agent_max_steps,
                max_llm_calls=settings.agent_max_llm_calls,
                timeout_s=settings.agent_timeout_s,
            ),
            eval_threshold=settings.eval_threshold,
        )
    return container
