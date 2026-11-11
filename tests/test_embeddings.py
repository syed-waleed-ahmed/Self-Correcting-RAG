"""Embedder dimension validation, without loading torch."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scrag.embeddings import SentenceTransformerEmbedder, embed_checked
from scrag.errors import EmbeddingDimensionError

from .conftest import FakeEmbedder


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a stand-in ``sentence_transformers`` module reporting 768 dims."""

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_embedding_dimension(self) -> int:
            return 768

        def encode(self, texts, **_):
            return np.ones((len(texts), 768), dtype="float32")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_model_dimension_mismatch_fails_fast(fake_sentence_transformers):
    embedder = SentenceTransformerEmbedder(model_name="big-model", dim=384)
    with pytest.raises(EmbeddingDimensionError, match="SCRAG_EMBEDDING_DIM=768"):
        embedder.embed(["hello"])
    # The mismatched model must not be cached as if it were valid.
    with pytest.raises(EmbeddingDimensionError):
        embedder.embed(["hello again"])


def test_matching_model_dimension_works(fake_sentence_transformers):
    embedder = SentenceTransformerEmbedder(model_name="big-model", dim=768)
    assert embedder.embed(["a", "b"]).shape == (2, 768)


def test_embed_checked_rejects_wrong_row_count():
    class ShortEmbedder(FakeEmbedder):
        def embed(self, texts):
            return super().embed(texts[:-1])

    with pytest.raises(EmbeddingDimensionError):
        embed_checked(ShortEmbedder(), ["a", "b"], expected_dim=ShortEmbedder().dim)


def test_fastembed_threads_reach_onnx_runtime(monkeypatch):
    """SCRAG_EMBEDDING_THREADS is passed to fastembed (the image sets 1, see config)."""
    from scrag.config import Settings
    from scrag.embeddings import FastEmbedEmbedder, create_embedder

    seen: dict[str, object] = {}

    class FakeTextEmbedding:
        def __init__(self, name, cache_dir=None, threads=None):
            seen.update(name=name, threads=threads)

        def embed(self, texts, batch_size=None):
            seen["batch_size"] = batch_size
            return iter(np.ones((len(texts), 384), dtype="float32"))

    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)

    settings = Settings(_env_file=None, embedding_backend="fastembed", embedding_threads=1)
    embedder = create_embedder(settings)
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.embed(["a query"]).shape == (1, 384)
    assert seen == {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "threads": 1,
        "batch_size": 8,  # the default batch size (see config: memory)
    }
    default = create_embedder(Settings(_env_file=None, embedding_backend="fastembed"))
    assert isinstance(default, FastEmbedEmbedder) and default.threads is None


def test_concurrent_first_use_loads_the_model_once(monkeypatch):
    """Warm-up and a visitor's request can race for the first load: one load wins."""
    import threading
    import time

    from scrag.embeddings import FastEmbedEmbedder

    loads: list[int] = []

    class SlowTextEmbedding:
        def __init__(self, name, cache_dir=None, threads=None):
            loads.append(1)
            time.sleep(0.2)  # long enough for the other thread to arrive meanwhile

        def embed(self, texts, batch_size=None):
            return iter(np.ones((len(texts), 384), dtype="float32"))

    module = types.ModuleType("fastembed")
    module.TextEmbedding = SlowTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)

    embedder = FastEmbedEmbedder(dim=384)
    workers = [threading.Thread(target=embedder.embed, args=(["q"],)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert len(loads) == 1
