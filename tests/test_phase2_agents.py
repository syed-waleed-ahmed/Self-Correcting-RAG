"""Batched guardrail and the fastembed embedder (Phase 2)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scrag import embeddings
from scrag.agents.guardrail import GuardrailAgent
from scrag.config import Settings
from scrag.embeddings import FastEmbedEmbedder, SentenceTransformerEmbedder, create_embedder
from scrag.errors import EmbeddingDimensionError, LLMError
from scrag.models import Chunk, RetrievedChunk

from .conftest import FakeChatClient


def _rcs(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=Chunk(id=str(i), source="s", text=f"chunk {i}"), score=0.5)
        for i in range(n)
    ]


def _agent(client, mode="batch"):
    return GuardrailAgent(client, model="m", threshold=0.6, max_workers=1, mode=mode)


def test_batch_guardrail_uses_one_call(tmp_path):
    client = FakeChatClient(guardrail='{"scores": [0.9, 0.1, "75%"]}')
    kept = _agent(client).filter("q", _rcs(3))
    assert client.calls.count("guardrail") == 1
    assert [rc.chunk.id for rc in kept] == ["0", "2"]
    assert kept[1].guardrail_score == pytest.approx(0.75)


@pytest.mark.parametrize("reply", ["not json", '{"scores": [0.9]}', '{"other": 1}'])
def test_batch_guardrail_falls_back_to_per_chunk(reply):
    client = FakeChatClient(guardrail=[reply, "0.9", "0.2"])
    kept = _agent(client).filter("q", _rcs(2))
    assert client.calls.count("guardrail") == 3  # 1 batch + 2 per-chunk
    assert [rc.guardrail_score for rc in kept] == [0.9]


def test_batch_guardrail_provider_failure_keeps_all_without_more_calls():
    client = FakeChatClient(guardrail=LLMError("down"))
    kept = _agent(client).filter("q", _rcs(4))
    assert client.calls.count("guardrail") == 1
    assert len(kept) == 4
    assert all(rc.guardrail_score is None for rc in kept)


def test_per_chunk_mode_is_still_the_class_default():
    assert GuardrailAgent(FakeChatClient(), model="m").mode == "per_chunk"


# --- fastembed -----------------------------------------------------------------------
@pytest.fixture
def fake_fastembed(monkeypatch):
    calls: dict[str, object] = {}

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            calls["model_name"] = model_name
            calls["cache_dir"] = cache_dir

        def embed(self, texts, batch_size=256):
            return (np.full(384, 0.5, dtype="float32") for _ in texts)

    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return calls


def test_fastembed_prefixes_short_model_names(fake_fastembed):
    embedder = FastEmbedEmbedder(model_name="all-MiniLM-L6-v2", cache_dir="/models")
    assert embedder.embed(["a", "b"]).shape == (2, 384)
    assert fake_fastembed["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert fake_fastembed["cache_dir"] == "/models"


def test_fastembed_dimension_mismatch_fails_fast(fake_fastembed):
    with pytest.raises(EmbeddingDimensionError, match="SCRAG_EMBEDDING_DIM=384"):
        FastEmbedEmbedder(dim=768).embed(["a"])


@pytest.mark.parametrize(
    ("backend", "available", "expected"),
    [
        ("auto", True, FastEmbedEmbedder),
        ("auto", False, SentenceTransformerEmbedder),
        ("fastembed", False, FastEmbedEmbedder),
        ("sentence-transformers", True, SentenceTransformerEmbedder),
    ],
)
def test_create_embedder_selects_backend(monkeypatch, backend, available, expected):
    monkeypatch.setattr(embeddings, "_fastembed_available", lambda: available)
    settings = Settings(_env_file=None, embedding_backend=backend)
    assert isinstance(create_embedder(settings), expected)
