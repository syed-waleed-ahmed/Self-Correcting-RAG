"""Pluggable vector store backends."""

from __future__ import annotations

from ..config import Settings
from .base import VectorStore
from .numpy_store import NumpyVectorStore

__all__ = ["VectorStore", "NumpyVectorStore", "create_vector_store"]


def create_vector_store(settings: Settings) -> VectorStore:
    """Instantiate the vector store selected by configuration.

    ``numpy`` is the dependency-free default (exact brute-force cosine search,
    fine into the ~10^5–10^6 chunk range). ``faiss`` is the optional backend for
    larger corpora and is only imported when requested.
    """
    backend = settings.vector_backend
    if backend == "numpy":
        return NumpyVectorStore(dim=settings.embedding_dim, path=settings.index_dir)
    if backend == "faiss":
        from .faiss_store import FaissVectorStore

        return FaissVectorStore(dim=settings.embedding_dim, path=settings.index_dir)
    raise ValueError(f"Unknown vector_backend: {backend!r}")
