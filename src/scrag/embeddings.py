"""Text embedding abstraction.

The rest of the system depends only on the :class:`Embedder` protocol, so the
backend can be swapped (or faked in tests) without touching retrieval logic. The
default implementation uses a local ``sentence-transformers`` model; importing
torch is deferred until the first embedding call so the package stays importable
without the heavy stack installed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into fixed-size vectors."""

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array."""
        ...


class SentenceTransformerEmbedder:
    """Local embedder backed by ``sentence-transformers`` (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self._dim = dim
        self.batch_size = batch_size
        self._model = None  # loaded on first use

    @property
    def dim(self) -> int:
        return self._dim

    def _get_model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional stack
                raise RuntimeError(
                    "The local embedder requires sentence-transformers. Install it with:\n"
                    '    pip install "self-correcting-rag[local]"'
                ) from exc
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            # sentence-transformers >=5 renamed this method; support both.
            dim_getter = getattr(
                self._model,
                "get_embedding_dimension",
                getattr(self._model, "get_sentence_embedding_dimension", None),
            )
            loaded_dim = dim_getter() if callable(dim_getter) else None
            if loaded_dim and loaded_dim != self._dim:
                logger.warning(
                    "Configured embedding_dim=%s but model reports %s; using %s",
                    self._dim,
                    loaded_dim,
                    loaded_dim,
                )
                self._dim = loaded_dim
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
