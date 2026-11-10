"""Ingestion tests: incremental upsert, dedup, and failure handling (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from scrag.chunking import Chunker
from scrag.errors import DocumentsNotFoundError, EmbeddingDimensionError, NoDocumentsError
from scrag.ingest import DOC_HASH_KEY, ingest
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeEmbedder


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("rag combines retrieval and generation.", encoding="utf-8")
    (d / "b.md").write_text("a guardrail agent filters context.", encoding="utf-8")
    return d


@pytest.fixture
def store(tmp_path):
    return NumpyVectorStore(dim=FAKE_DIM, path=tmp_path / "index")


def _run(store, docs, reset, embedder=None):
    return ingest(
        embedder=embedder or FakeEmbedder(),
        store=store,
        chunker=Chunker(chunk_size=200, overlap=20),
        docs_dir=docs,
        reset=reset,
    )


def test_reset_rebuilds_index(store, docs):
    first = _run(store, docs, reset=True)
    second = _run(store, docs, reset=True)
    assert first.index_size == second.index_size == 2
    assert second.added_documents == 2


def test_reingest_without_reset_does_not_duplicate(store, docs):
    """Regression: reset=false used to append every chunk again."""
    _run(store, docs, reset=True)
    generation = store.generation

    stats = _run(store, docs, reset=False)

    assert stats.index_size == 2
    assert stats.skipped_documents == 2
    assert stats.chunks == 0
    assert store.generation == generation  # nothing changed, nothing rewritten


def test_changed_document_replaces_its_old_chunks(store, docs):
    _run(store, docs, reset=True)
    (docs / "a.txt").write_text("an embedding maps text to a vector.", encoding="utf-8")

    stats = _run(store, docs, reset=False)

    assert (stats.updated_documents, stats.skipped_documents, stats.added_documents) == (1, 1, 0)
    texts = [c.text for c in store.snapshot()[1]]
    assert "an embedding maps text to a vector." in texts
    assert "rag combines retrieval and generation." not in texts
    assert len(store) == 2


def test_new_document_is_added_without_touching_others(store, docs):
    _run(store, docs, reset=True)
    (docs / "c.txt").write_text("vector retrieval context.", encoding="utf-8")

    stats = _run(store, docs, reset=False)

    assert (stats.added_documents, stats.skipped_documents) == (1, 2)
    assert len(store) == 3


def test_chunks_without_hash_are_refreshed(store, docs):
    """Indexes built before doc hashes existed self-heal on the next upsert."""
    _run(store, docs, reset=True)
    vectors, chunks = store.snapshot()
    for c in chunks:
        c.metadata.pop(DOC_HASH_KEY)
    store.replace(vectors, chunks)

    stats = _run(store, docs, reset=False)

    assert stats.updated_documents == 2
    assert all(DOC_HASH_KEY in c.metadata for c in store.snapshot()[1])
    assert len(store) == 2


def test_missing_docs_dir_raises_domain_error(store, tmp_path):
    with pytest.raises(DocumentsNotFoundError) as info:
        _run(store, tmp_path / "nope", reset=True)
    assert isinstance(info.value, FileNotFoundError)  # backward compatible


def test_empty_docs_dir_raises_domain_error(store, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(NoDocumentsError) as info:
        _run(store, tmp_path / "empty", reset=True)
    assert isinstance(info.value, ValueError)  # backward compatible


def test_non_utf8_file_is_skipped(store, docs):
    (docs / "binary.txt").write_bytes(b"\xff\xfe\x00bad")
    stats = _run(store, docs, reset=True)
    assert stats.documents == 2


def test_embedding_dimension_mismatch_fails_before_writing(store, docs):
    class WrongDimEmbedder(FakeEmbedder):
        def embed(self, texts):
            return np.ones((len(texts), FAKE_DIM + 1), dtype="float32")

    with pytest.raises(EmbeddingDimensionError):
        _run(store, docs, reset=True, embedder=WrongDimEmbedder())
    assert len(store) == 0
    assert store.generation is None  # nothing persisted


def test_non_finite_embeddings_rejected(store, docs):
    class NaNEmbedder(FakeEmbedder):
        def embed(self, texts):
            return np.full((len(texts), FAKE_DIM), np.nan, dtype="float32")

    from scrag.errors import EmbeddingError

    with pytest.raises(EmbeddingError):
        _run(store, docs, reset=True, embedder=NaNEmbedder())
