"""Text embedding abstraction.

The rest of the system depends only on the :class:`Embedder` protocol, so the
backend can be swapped (or faked in tests) without touching retrieval logic. The
default implementation uses a local ``sentence-transformers`` model; importing
torch is deferred until the first embedding call so the package stays importable
without the heavy stack installed.

Every embedding that reaches the vector store passes through
:func:`embed_checked`, which verifies the real output shape against the
dimension the index was built with.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .errors import EmbeddingDimensionError, EmbeddingError
from .logging_config import get_logger

if TYPE_CHECKING:
    from .config import Settings

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into fixed-size vectors."""

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array."""


def embed_checked(embedder: Embedder, texts: list[str], expected_dim: int) -> np.ndarray:
    """Embed ``texts`` and verify the result is a finite ``(len(texts), expected_dim)`` array."""
    vectors = np.asarray(embedder.embed(texts), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise EmbeddingDimensionError(
            f"Embedder returned shape {vectors.shape} for {len(texts)} texts; "
            f"expected ({len(texts)}, {expected_dim})."
        )
    if vectors.shape[1] != expected_dim:
        raise EmbeddingDimensionError(
            f"Embedder produces {vectors.shape[1]}-dim vectors but the index expects "
            f"{expected_dim}. Set SCRAG_EMBEDDING_DIM={vectors.shape[1]} and re-ingest "
            "(POST /ingest with reset=true)."
        )
    if vectors.size and not np.isfinite(vectors).all():
        raise EmbeddingError("Embedder returned non-finite values (NaN or inf).")
    return vectors


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
        self._model: Any = None  # loaded on first use
        self._load_lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:  # one load, even when warm-up and a request race
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional stack
            raise RuntimeError(
                "The local embedder requires sentence-transformers. Install it with:\n"
                '    pip install "self-correcting-rag[local]"'
            ) from exc
        logger.info("Loading embedding model %s", self.model_name)
        model = SentenceTransformer(self.model_name)
        # sentence-transformers >=5 renamed this method; support both.
        dim_getter = getattr(
            model,
            "get_embedding_dimension",
            getattr(model, "get_sentence_embedding_dimension", None),
        )
        loaded_dim = dim_getter() if callable(dim_getter) else None
        if loaded_dim and loaded_dim != self._dim:
            # Fail fast: silently adopting the model's dimension would only
            # move the failure to the vector store, with a worse message.
            raise EmbeddingDimensionError(
                f"Embedding model {self.model_name!r} produces {loaded_dim}-dim vectors "
                f"but SCRAG_EMBEDDING_DIM={self._dim}. Set SCRAG_EMBEDDING_DIM={loaded_dim} "
                "and re-ingest."
            )
        return model

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


class FastEmbedEmbedder:
    """Local embedder backed by ``fastembed`` (ONNX Runtime, no PyTorch).

    Runs the same sentence-transformers models with roughly half the memory of
    the torch backend, which is what makes a 512 MB hosting instance viable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        batch_size: int = 32,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        # fastembed names sentence-transformers models with their HF org prefix.
        self.model_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        self._dim = dim
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.threads = threads  # ONNX Runtime threads; None = one per core
        self._model: Any = None
        self._load_lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:  # one load, even when warm-up and a request race
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> Any:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional stack
            raise RuntimeError(
                "The fastembed backend requires fastembed. Install it with:\n"
                '    pip install "self-correcting-rag[onnx]"'
            ) from exc
        logger.info(
            "Loading embedding model %s (fastembed, threads=%s)",
            self.model_name,
            self.threads or "auto",
        )
        model = TextEmbedding(self.model_name, cache_dir=self.cache_dir, threads=self.threads)
        probe = np.asarray(next(iter(model.embed(["dimension probe"]))), dtype="float32")
        if probe.shape[-1] != self._dim:
            raise EmbeddingDimensionError(
                f"Embedding model {self.model_name!r} produces {probe.shape[-1]}-dim "
                f"vectors but SCRAG_EMBEDDING_DIM={self._dim}. Set "
                f"SCRAG_EMBEDDING_DIM={probe.shape[-1]} and re-ingest."
            )
        return model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        model = self._get_model()
        return np.asarray(list(model.embed(texts, batch_size=self.batch_size)), dtype="float32")


def _fastembed_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("fastembed") is not None


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``settings.embedding_backend``."""
    backend = settings.embedding_backend
    if backend == "auto":
        backend = "fastembed" if _fastembed_available() else "sentence-transformers"
    cache_dir = str(settings.embedding_cache_dir) if settings.embedding_cache_dir else None
    if backend == "fastembed":
        return FastEmbedEmbedder(
            model_name=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            cache_dir=cache_dir,
            threads=settings.embedding_threads,
        )
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )
