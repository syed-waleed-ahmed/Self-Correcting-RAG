"""Vector store interface shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Abstract nearest-neighbour store over chunk embeddings.

    Implementations own persistence and search. Vectors are assumed to be
    L2-normalized on insertion so that inner product equals cosine similarity.
    """

    dim: int

    @abstractmethod
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add ``vectors`` (shape ``(n, dim)``) with their ``chunks``."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks to ``query_vector``."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors and chunks."""

    @abstractmethod
    def save(self) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self) -> bool:
        """Load a persisted index if present. Returns True if anything loaded."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored chunks."""

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize row vectors, guarding against zero norms."""
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms
