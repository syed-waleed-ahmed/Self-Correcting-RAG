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
