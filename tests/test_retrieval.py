"""BM25, Reciprocal Rank Fusion and the hybrid retriever (Phase 2)."""

from __future__ import annotations

import pytest

from scrag.models import Chunk
from scrag.retriever import HybridRetriever, Retriever
from scrag.sparse import BM25Index, reciprocal_rank_fusion, tokenize
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeEmbedder


def test_tokenize_keeps_compounds_and_their_parts():
    tokens = tokenize("The X-Request-ID header and demo_budget_exhausted code")
    assert "x-request-id" in tokens
    assert {"request", "id", "demo_budget_exhausted", "budget"} <= set(tokens)
    assert "the" not in tokens and "and" not in tokens  # stopwords dropped


def test_bm25_prefers_rare_exact_terms():
    index = BM25Index(
        [
            "the evaluator scores answers",
            "error code demo_budget_exhausted means the budget is spent",
            "answers and scores and answers",
        ]
    )
    best, _ = index.top("which code is demo_budget_exhausted", 1)[0]
    assert best == 1


def test_bm25_ignores_unknown_terms_and_empty_index():
    assert BM25Index(["alpha beta"]).top("gamma", 5) == []
    assert BM25Index([]).top("anything", 5) == []


def test_bm25_length_normalization_with_b_zero_ignores_length():
    short, long = "rare", "rare " + "filler " * 50
    with_norm = BM25Index([short, long], b=0.75).scores("rare")
    without = BM25Index([short, long], b=0.0).scores("rare")
    assert with_norm[0] > with_norm[1]
    assert without[0] == pytest.approx(without[1])


def test_reciprocal_rank_fusion_math():
    fused = reciprocal_rank_fusion([[7, 8], [8, 9]], k=60)
    assert fused[8] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[7] == pytest.approx(1 / 61)
    assert fused[9] == pytest.approx(1 / 62)
    assert max(fused, key=lambda i: fused[i]) == 8  # in both lists: wins


def _store(tmp_path, texts):
    embedder = FakeEmbedder()
    chunks = [Chunk(id=f"c{i}", source=f"d{i}.txt", text=t) for i, t in enumerate(texts)]
    store = NumpyVectorStore(dim=FAKE_DIM, path=tmp_path)
    store.replace(embedder.embed(texts), chunks)
    return embedder, store


TEXTS = [
    "rag retrieval generation context answer",
    "guardrail agent filter context",
    "embedding vector retrieval",
    # No fake-vocabulary words: invisible to the fake dense embedder.
    "configure SCRAG_RRF_K to tune fusion smoothing",
]


def test_hybrid_finds_identifier_that_dense_misses(tmp_path):
    embedder, store = _store(tmp_path, TEXTS)
    query = "what does SCRAG_RRF_K tune"
    dense_ids = [r.chunk.id for r in Retriever(embedder, store).retrieve(query, 2)]
    hybrid = HybridRetriever(embedder, store).retrieve(query, 2)
    assert "c3" not in dense_ids
    assert hybrid[0].chunk.id == "c3"
    assert hybrid[0].keyword_score is not None
    assert hybrid[0].fusion_score is not None


def test_hybrid_reports_true_cosine_scores(tmp_path):
    embedder, store = _store(tmp_path, TEXTS)
    query = "guardrail agent filter"
    dense = {r.chunk.id: r.score for r in Retriever(embedder, store).retrieve(query, 4)}
    for rc in HybridRetriever(embedder, store).retrieve(query, 4):
        assert rc.score == pytest.approx(dense[rc.chunk.id])


def test_hybrid_keyword_index_follows_store_changes(tmp_path):
    embedder, store = _store(tmp_path, TEXTS)
    retriever = HybridRetriever(embedder, store)
    assert retriever.retrieve("zebra", 1)[0].keyword_score is None  # nothing matches
    revision = store.versioned_snapshot()[0]

    store.add(embedder.embed(["zebra crossing"]), [Chunk(id="z", source="z.txt", text="zebra")])

    assert store.versioned_snapshot()[0] > revision
    top = retriever.retrieve("zebra", 1)[0]
    assert top.chunk.id == "z"
    assert top.keyword_score is not None


def test_hybrid_on_empty_store_returns_nothing(tmp_path):
    store = NumpyVectorStore(dim=FAKE_DIM, path=tmp_path)
    assert HybridRetriever(FakeEmbedder(), store).retrieve("anything") == []
