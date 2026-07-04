"""In-memory NumPy vector store with on-disk persistence.

Exact brute-force cosine search. Uses ``argpartition`` for top-k selection so a
single query stays ``O(n)`` rather than ``O(n log n)``, which keeps it usable
well into the hundreds-of-thousands-of-chunks range before FAISS is warranted.

Persistence layout (under ``path``):
    vectors.npy    float32 matrix, shape (n, dim), L2-normalized
    chunks.jsonl   one JSON-encoded Chunk per line, row-aligned with vectors
    meta.json      {"dim": int, "count": int}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..logging_config import get_logger
from ..models import Chunk, RetrievedChunk
from .base import VectorStore

logger = get_logger(__name__)


class NumpyVectorStore(VectorStore):
    def __init__(self, dim: int, path: Path | str) -> None:
        self.dim = dim
        self.path = Path(path)
        self._vectors: np.ndarray = np.zeros((0, dim), dtype="float32")
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(chunks) == 0:
            return
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors/chunks length mismatch: {vectors.shape[0]} != {len(chunks)}"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )
        normalized = self._normalize(vectors)
        self._vectors = (
            normalized
            if self._vectors.shape[0] == 0
            else np.vstack([self._vectors, normalized])
        )
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if len(self._chunks) == 0 or top_k <= 0:
            return []
        query = self._normalize(query_vector)[0]
        sims = self._vectors @ query  # cosine, both sides normalized
        k = min(top_k, sims.shape[0])
        # Partial selection of the k best, then sort just those k.
        top_unsorted = np.argpartition(-sims, k - 1)[:k]
        top_sorted = top_unsorted[np.argsort(-sims[top_unsorted])]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(sims[i]))
            for i in top_sorted
        ]

    def clear(self) -> None:
        self._vectors = np.zeros((0, self.dim), dtype="float32")
        self._chunks = []

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self._vectors)
        with (self.path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(chunk.model_dump_json() + "\n")
        with (self.path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "count": len(self._chunks)}, f)
        logger.info("Saved index (%d chunks) to %s", len(self._chunks), self.path)

    def load(self) -> bool:
        vectors_path = self.path / "vectors.npy"
        chunks_path = self.path / "chunks.jsonl"
        if not (vectors_path.exists() and chunks_path.exists()):
            return False
        vectors = np.load(vectors_path).astype("float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Persisted index dim {vectors.shape} != configured dim {self.dim}. "
                "Re-ingest after changing the embedding model."
            )
        chunks = [
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if vectors.shape[0] != len(chunks):
            raise ValueError("Corrupt index: vector/chunk count mismatch.")
        self._vectors = vectors
        self._chunks = chunks
        logger.info("Loaded index (%d chunks) from %s", len(chunks), self.path)
        return True
