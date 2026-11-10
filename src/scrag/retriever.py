"""Query-time retrieval: dense (vector) and hybrid (vector + BM25, fused with RRF)."""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from .embeddings import Embedder, embed_checked
from .models import Chunk, RetrievedChunk
from .sparse import BM25Index, reciprocal_rank_fusion
from .vectorstore.base import VectorStore


class SupportsRetrieve(Protocol):
    """Anything that returns ranked chunks for a query."""

    mode: str

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks for ``query``, best first."""


class Retriever:
    """Dense retrieval: embed the query and return the most similar chunks."""

    mode = "dense"

    def __init__(self, embedder: Embedder, store: VectorStore, default_top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        # An empty store can't match anything: skip embedding (and model loading).
        if not query.strip() or len(self.store) == 0:
            return []
        k = top_k or self.default_top_k
        query_vector = embed_checked(self.embedder, [query], self.store.dim)
        return self.store.search(query_vector, k)


class HybridRetriever(Retriever):
    """Dense + BM25 candidates fused with Reciprocal Rank Fusion.

    Each retriever contributes ``candidate_pool`` candidates; RRF (constant
    ``rrf_k``, 60 by default as in Cormack et al.) merges them by rank only, so
    the incompatible score scales never have to be calibrated. Every result
    keeps its true cosine similarity in ``score``; the fused value is in
    ``fusion_score`` and the BM25 value (if matched) in ``keyword_score``.

    Both rankings are computed over one ``versioned_snapshot`` of the store, and
    the BM25 index is cached per store revision.
    """

    mode = "hybrid"

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        default_top_k: int = 5,
        *,
        candidate_pool: int = 20,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(embedder, store, default_top_k)
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        self._cache_lock = threading.Lock()
        self._cache: tuple[int, np.ndarray, list[Chunk], BM25Index] | None = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip() or len(self.store) == 0:
            return []
        k = top_k or self.default_top_k
        vectors, chunks, bm25 = self._snapshot()
        if not chunks:
            return []
        pool = max(k, self.candidate_pool)

        query_vector = embed_checked(self.embedder, [query], self.store.dim)
        sims = vectors @ VectorStore._normalize(query_vector)[0]
        n_dense = min(pool, sims.shape[0])
        dense_top = np.argpartition(-sims, n_dense - 1)[:n_dense]
        dense_ranking = [int(i) for i in dense_top[np.argsort(-sims[dense_top])]]

        keyword_hits = bm25.top(query, pool)
        keyword_scores = dict(keyword_hits)
        fused = reciprocal_rank_fusion(
            [dense_ranking, [idx for idx, _ in keyword_hits]], k=self.rrf_k
        )
        best = sorted(fused, key=lambda i: (-fused[i], -float(sims[i])))[:k]
        return [
            RetrievedChunk(
                chunk=chunks[i],
                score=float(sims[i]),
                fusion_score=fused[i],
                keyword_score=keyword_scores.get(i),
            )
            for i in best
        ]

    def _snapshot(self) -> tuple[np.ndarray, list[Chunk], BM25Index]:
        revision, vectors, chunks = self.store.versioned_snapshot()
        with self._cache_lock:
            cached = self._cache
            if cached is not None and revision != -1 and cached[0] == revision:
                return cached[1], cached[2], cached[3]
            bm25 = BM25Index([c.text for c in chunks])
            if revision != -1:
                self._cache = (revision, vectors, chunks, bm25)
        return vectors, chunks, bm25
