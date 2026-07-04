from __future__ import annotations

import pytest

from scrag.container import build_container
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder


def _pipeline(settings, chat_client, chunks):
    """Build a real pipeline wired to fakes, with `chunks` indexed."""
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    if chunks:
        store.add(embedder.embed([c.text for c in chunks]), chunks)
    container = build_container(
        settings, embedder=embedder, store=store, chat_client=chat_client
    )
    assert container.pipeline is not None
    return container.pipeline


def test_stops_early_when_evaluation_passes(settings, sample_chunks):
    client = FakeChatClient(
        guardrail="0.9",
        generator=["first answer", "second answer"],
        evaluator='{"score": 0.95, "explanation": "great"}',
    )
    pipeline = _pipeline(settings, client, sample_chunks)

    result = pipeline.run("what is a guardrail agent")

    assert result.answer == "first answer"
    assert result.attempts == 1
    assert result.evaluation_score == 0.95
    assert client.calls.count("generator") == 1  # no needless second generation


def test_keeps_best_attempt_not_last(settings, sample_chunks):
    # Attempt 1 (0.5) is better than the "corrected" attempt 2 (0.3): the pipeline
    # must return attempt 1, never regressing to a worse answer.
    client = FakeChatClient(
        guardrail="0.9",
        generator=["good first answer", "worse second answer"],
        evaluator=[
            '{"score": 0.5, "explanation": "partial"}',
            '{"score": 0.3, "explanation": "worse"}',
        ],
    )
    pipeline = _pipeline(settings, client, sample_chunks)

    result = pipeline.run("what is a guardrail agent")

    assert result.answer == "good first answer"
    assert result.evaluation_score == 0.5
    assert result.attempts == 1
    assert client.calls.count("generator") == 2  # it did try to self-correct


def test_guardrail_dropping_all_falls_back_to_raw_retrieval(settings, sample_chunks):
    client = FakeChatClient(
        guardrail="0.0",  # drop everything
        generator="answer from fallback context",
        evaluator='{"score": 0.9, "explanation": "ok"}',
    )
    pipeline = _pipeline(settings, client, sample_chunks)

    result = pipeline.run("what is a guardrail agent")

    assert result.used_chunks  # fallback kept the retrieved chunks
    assert result.answer == "answer from fallback context"


def test_empty_index_returns_graceful_answer(settings):
    client = FakeChatClient()
    pipeline = _pipeline(settings, client, chunks=[])

    result = pipeline.run("anything")

    assert result.attempts == 0
    assert result.used_chunks == []
    assert "could not find" in result.answer.lower()


def test_empty_query_rejected(settings, sample_chunks):
    pipeline = _pipeline(settings, FakeChatClient(), sample_chunks)
    with pytest.raises(ValueError):
        pipeline.run("   ")
