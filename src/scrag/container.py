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

from .agents import EvaluatorAgent, GeneratorAgent, GuardrailAgent
from .chunking import Chunker
from .config import Settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .errors import IngestInProgressError
from .ingest import IngestStats, ingest
from .llm.client import ChatClient, OpenAICompatibleClient
from .logging_config import get_logger
from .pipeline import SelfCorrectingRAGPipeline
from .retriever import Retriever
from .vectorstore import create_vector_store
from .vectorstore.base import VectorStore

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    chunker: Chunker
    retriever: Retriever
    pipeline: SelfCorrectingRAGPipeline | None  # None when no LLM key configured
    # One ingestion at a time per process; concurrent requests get a 409.
    ingest_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
            )
        finally:
            self.ingest_lock.release()


def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    chat_client: ChatClient | None = None,
) -> Container:
    """Construct a fully wired :class:`Container`.

    The optional overrides exist so tests (and alternative deployments) can inject
    fakes without monkeypatching.
    """
    embedder = embedder or SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )
    store = store or create_vector_store(settings)
    chunker = Chunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    retriever = Retriever(embedder, store, default_top_k=settings.top_k)

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
            ),
            generator=GeneratorAgent(client, settings.generator_model),
            evaluator=EvaluatorAgent(client, settings.evaluator_model),
            top_k=settings.top_k,
            eval_threshold=settings.eval_threshold,
            max_self_correct_steps=settings.max_self_correct_steps,
        )
    else:
        logger.warning("No LLM key configured — /query will be unavailable.")

    return Container(
        settings=settings,
        embedder=embedder,
        store=store,
        chunker=chunker,
        retriever=retriever,
        pipeline=pipeline,
    )
