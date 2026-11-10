"""Vector store interface shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Abstract nearest-neighbour store over chunk embeddings.

    Implementations own persistence and search. Vectors are assumed to be
    L2-normalized on insertion so that inner product equals cosine similarity.

    Concurrency contract: ``search`` may run concurrently with any write
    (``add``/``replace``/``clear``/``load``) and must observe either the state
    before or after that write, never a mix of the two.
    """

    dim: int

    @abstractmethod
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Append ``vectors`` (shape ``(n, dim)``) with their ``chunks``."""

    @abstractmethod
    def replace(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Atomically swap the whole contents for ``vectors``/``chunks``."""

    @abstractmethod
    def snapshot(self) -> tuple[np.ndarray, list[Chunk]]:
        """Return a consistent copy of the (normalized) vectors and their chunks."""

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

    def refresh_if_stale(self) -> bool:
        """Reload if another process persisted a newer index. Returns True if reloaded.

        The default is a no-op for stores with no shared persistence.
        """
        return False

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize row vectors, guarding against zero norms."""
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms
