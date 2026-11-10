"""Optional FAISS-backed vector store for large corpora.

Uses a flat inner-product index over L2-normalized vectors (exact cosine). Swap
``IndexFlatIP`` for an IVF/HNSW index if you need approximate search at very
large scale; the persistence contract stays the same.

Like the NumPy store, the live state is an immutable ``(index, chunks)`` tuple:
writes build a new FAISS index (copy-on-write) and swap it in, so concurrent
searches never touch an index that is being modified.

Requires the optional dependency::

    pip install "self-correcting-rag[faiss]"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..errors import IndexIncompatibleError
from ..models import Chunk, RetrievedChunk
from ._persistence import PersistentVectorStore, check_count, read_chunks, write_chunks


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The faiss backend requires faiss-cpu. Install it with:\n"
            '    pip install "self-correcting-rag[faiss]"'
        ) from exc
    return faiss


class FaissVectorStore(PersistentVectorStore):
    backend = "faiss"
    legacy_files = ("index.faiss", "chunks.jsonl")

    def __init__(self, dim: int, path: Path | str) -> None:
        super().__init__(dim, path)
        self._faiss = _import_faiss()
        self._state: tuple[Any, tuple[Chunk, ...]] = (self._faiss.IndexFlatIP(dim), ())

    def __len__(self) -> int:
        return len(self._state[1])

    # --- writes ----------------------------------------------------------------
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(chunks) == 0:
            return
        normalized = self._validated(vectors, chunks)
        with self._lock:
            current_index, current_chunks = self._state
            index = self._faiss.clone_index(current_index)
            index.add(normalized)
            self._state = (index, current_chunks + tuple(chunks))

    def replace(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        index = self._faiss.IndexFlatIP(self.dim)
        if chunks:
            index.add(self._validated(vectors, chunks))
        with self._lock:
            self._state = (index, tuple(chunks))

    def clear(self) -> None:
        with self._lock:
            self._state = (self._faiss.IndexFlatIP(self.dim), ())

    # --- reads -----------------------------------------------------------------
    def snapshot(self) -> tuple[np.ndarray, list[Chunk]]:
        index, chunks = self._state
        if index.ntotal == 0:
            return np.zeros((0, self.dim), dtype="float32"), list(chunks)
        return np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32"), list(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        index, chunks = self._state  # one read: a consistent snapshot
        if len(chunks) == 0 or top_k <= 0:
            return []
        query = self._normalize(query_vector)
        k = min(top_k, len(chunks))
        scores, indices = index.search(query, k)
        return [
            RetrievedChunk(chunk=chunks[int(idx)], score=float(score))
            for score, idx in zip(scores[0], indices[0], strict=False)
            if idx >= 0
        ]

    # --- persistence hooks -------------------------------------------------------
    def _write_payload(self, directory: Path) -> int:
        index, chunks = self._state
        self._faiss.write_index(index, str(directory / "index.faiss"))
        write_chunks(directory / "chunks.jsonl", chunks)
        return len(chunks)

    def _read_payload(self, directory: Path, meta: dict[str, Any] | None) -> None:
        index = self._faiss.read_index(str(directory / "index.faiss"))
        if index.d != self.dim:
            raise IndexIncompatibleError(
                f"Persisted FAISS dim {index.d} != configured dim {self.dim}."
            )
        chunks = read_chunks(directory / "chunks.jsonl")
        check_count(directory, meta, int(index.ntotal), len(chunks))
        with self._lock:
            self._state = (index, tuple(chunks))

    # --- helpers -----------------------------------------------------------------
    def _validated(self, vectors: np.ndarray, chunks: list[Chunk]) -> np.ndarray:
        normalized = self._normalize(vectors)
        if normalized.shape[0] != len(chunks):
            raise ValueError(
                f"vectors/chunks length mismatch: {normalized.shape[0]} != {len(chunks)}"
            )
        if normalized.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {normalized.shape[1]}")
        return normalized
