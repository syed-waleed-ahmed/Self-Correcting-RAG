"""Optional FAISS-backed vector store for large corpora.

Uses a flat inner-product index over L2-normalized vectors (exact cosine). Swap
``IndexFlatIP`` for an IVF/HNSW index if you need approximate search at very
large scale; the persistence contract stays the same.

Requires the optional dependency::

    pip install "self-correcting-rag[faiss]"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..logging_config import get_logger
from ..models import Chunk, RetrievedChunk
from .base import VectorStore

logger = get_logger(__name__)


def _import_faiss():  # type: ignore[no-untyped-def]
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The faiss backend requires faiss-cpu. Install it with:\n"
            '    pip install "self-correcting-rag[faiss]"'
        ) from exc
    return faiss


class FaissVectorStore(VectorStore):
    def __init__(self, dim: int, path: Path | str) -> None:
        self.dim = dim
        self.path = Path(path)
        self._faiss = _import_faiss()
        self._index = self._faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(chunks) == 0:
            return
        normalized = self._normalize(vectors)
        if normalized.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {normalized.shape[1]}")
        self._index.add(normalized)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if len(self._chunks) == 0 or top_k <= 0:
            return []
        query = self._normalize(query_vector)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(chunk=self._chunks[int(idx)], score=float(score))
            )
        return results

    def clear(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._chunks = []

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.path / "index.faiss"))
        with (self.path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(chunk.model_dump_json() + "\n")
        with (self.path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "count": len(self._chunks)}, f)
        logger.info("Saved FAISS index (%d chunks) to %s", len(self._chunks), self.path)

    def load(self) -> bool:
        index_path = self.path / "index.faiss"
        chunks_path = self.path / "chunks.jsonl"
        if not (index_path.exists() and chunks_path.exists()):
            return False
        self._index = self._faiss.read_index(str(index_path))
        if self._index.d != self.dim:
            raise ValueError(
                f"Persisted FAISS dim {self._index.d} != configured dim {self.dim}."
            )
        self._chunks = [
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info("Loaded FAISS index (%d chunks) from %s", len(self._chunks), self.path)
        return True
