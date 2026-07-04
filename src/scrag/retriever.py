"""Query-time retrieval: embed the query and search the vector store."""

from __future__ import annotations

from .embeddings import Embedder
from .models import RetrievedChunk
from .vectorstore.base import VectorStore


class Retriever:
    """Embeds a query and returns the most similar chunks from the store."""

    def __init__(self, embedder: Embedder, store: VectorStore, default_top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        k = top_k or self.default_top_k
        query_vector = self.embedder.embed([query])
        return self.store.search(query_vector, k)
