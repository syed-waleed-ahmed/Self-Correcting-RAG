"""In-memory NumPy vector store with on-disk persistence.

Exact brute-force cosine search. Uses ``argpartition`` for top-k selection so a
single query stays ``O(n)`` rather than ``O(n log n)``, which keeps it usable
well into the hundreds-of-thousands-of-chunks range before FAISS is warranted.

The live state is a single immutable ``(vectors, chunks)`` tuple. Writers build a
new tuple and swap it in under a lock; ``search`` reads the tuple once, so it
always sees a consistent index even while an ingestion is replacing it.

Payload files in each generation directory (see ``_persistence``):
    vectors.npy    float32 matrix, shape (n, dim), L2-normalized
    chunks.jsonl   one JSON-encoded Chunk per line, row-aligned with vectors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..errors import IndexIncompatibleError
from ..models import Chunk, RetrievedChunk
from ._persistence import PersistentVectorStore, check_count, read_chunks, write_chunks

# (vectors, chunks, revision): replaced as a whole, never mutated.
_State = tuple[np.ndarray, tuple[Chunk, ...], int]


class NumpyVectorStore(PersistentVectorStore):
    backend = "numpy"
    legacy_files = ("vectors.npy", "chunks.jsonl")

    def __init__(self, dim: int, path: Path | str) -> None:
        super().__init__(dim, path)
        self._state: _State = (self._empty(), (), 0)

    def __len__(self) -> int:
        return len(self._state[1])

    # --- writes ----------------------------------------------------------------
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(chunks) == 0:
            return
        normalized = self._validated(vectors, chunks)
        with self._lock:
            current_vectors, current_chunks, _ = self._state
            merged = (
                normalized
                if current_vectors.shape[0] == 0
                else np.vstack([current_vectors, normalized])
            )
            self._publish(merged, current_chunks + tuple(chunks))

    def replace(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        normalized = self._validated(vectors, chunks) if chunks else self._empty()
        with self._lock:
            self._publish(normalized, tuple(chunks))

    def clear(self) -> None:
        with self._lock:
            self._publish(self._empty(), ())

    # --- reads -----------------------------------------------------------------
    def snapshot(self) -> tuple[np.ndarray, list[Chunk]]:
        vectors, chunks, _ = self._state
        return vectors, list(chunks)

    def versioned_snapshot(self) -> tuple[int, np.ndarray, list[Chunk]]:
        vectors, chunks, revision = self._state
        return revision, vectors, list(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        vectors, chunks, _ = self._state  # one read: a consistent snapshot
        if len(chunks) == 0 or top_k <= 0:
            return []
        query = self._normalize(query_vector)[0]
        sims = vectors @ query  # cosine, both sides normalized
        k = min(top_k, sims.shape[0])
        # Partial selection of the k best, then sort just those k.
        top_unsorted = np.argpartition(-sims, k - 1)[:k]
        top_sorted = top_unsorted[np.argsort(-sims[top_unsorted])]
        return [RetrievedChunk(chunk=chunks[i], score=float(sims[i])) for i in top_sorted]

    # --- persistence hooks -------------------------------------------------------
    def _write_payload(self, directory: Path) -> int:
        vectors, chunks, _ = self._state
        np.save(directory / "vectors.npy", vectors)
        write_chunks(directory / "chunks.jsonl", chunks)
        return len(chunks)

    def _read_payload(self, directory: Path, meta: dict[str, Any] | None) -> None:
        vectors = np.load(directory / "vectors.npy").astype("float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise IndexIncompatibleError(
                f"Persisted index dim {vectors.shape} != configured dim {self.dim}. "
                "Re-ingest after changing the embedding model."
            )
        chunks = read_chunks(directory / "chunks.jsonl")
        check_count(directory, meta, vectors.shape[0], len(chunks))
        with self._lock:
            self._publish(vectors, tuple(chunks))

    # --- helpers -----------------------------------------------------------------
    def _empty(self) -> np.ndarray:
        return np.zeros((0, self.dim), dtype="float32")

    def _validated(self, vectors: np.ndarray, chunks: list[Chunk]) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors/chunks length mismatch: {vectors.shape} for {len(chunks)} chunks"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        return self._normalize(vectors)

    def _publish(self, vectors: np.ndarray, chunks: tuple[Chunk, ...]) -> None:
        vectors.setflags(write=False)  # shared with concurrent readers: never mutate
        self._state = (vectors, chunks, self._state[2] + 1)
