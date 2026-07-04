from __future__ import annotations

from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder


def _client(settings, chat_client=None, chunks=None) -> TestClient:
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    if chunks:
        store.add(embedder.embed([c.text for c in chunks]), chunks)
    container = build_container(
        settings,
        embedder=embedder,
        store=store,
        chat_client=chat_client or FakeChatClient(),
    )
    return TestClient(create_app(settings, container=container))


def test_health(settings):
    with _client(settings) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["vector_backend"] == "numpy"


def test_query_returns_answer(settings, sample_chunks):
    chat = FakeChatClient(
        guardrail="0.9",
        generator="a grounded answer about guardrails",
        evaluator='{"score": 0.9, "explanation": "ok"}',
    )
    with _client(settings, chat, sample_chunks) as client:
        resp = client.post("/query", json={"query": "what is a guardrail agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "a grounded answer about guardrails"
    assert body["attempts"] == 1
    assert len(body["used_chunks"]) > 0


def test_query_on_empty_index_returns_409(settings):
    with _client(settings, chunks=None) as client:
        resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 409


def test_query_validation_error(settings, sample_chunks):
    with _client(settings, chunks=sample_chunks) as client:
        resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_ingest_endpoint(settings):
    docs_dir = settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "a.txt").write_text(
        "rag combines retrieval and generation.", encoding="utf-8"
    )
    (docs_dir / "b.txt").write_text(
        "a guardrail agent filters context.", encoding="utf-8"
    )
    with _client(settings) as client:
        resp = client.post("/ingest", json={"reset": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["documents"] == 2
        assert body["chunks"] >= 2
        assert body["index_size"] == body["chunks"]

        health = client.get("/health").json()
        assert health["index_size"] == body["index_size"]


def test_query_without_llm_returns_503(settings, sample_chunks):
    # Build a container with no chat client and no key → pipeline is None.
    settings.llm_api_key = None
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    container = build_container(settings, embedder=embedder, store=store)
    assert container.pipeline is None
    with TestClient(create_app(settings, container=container)) as client:
        resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 503


def test_api_auth_enforced_when_token_configured(settings, sample_chunks):
    settings.api_auth_token = "s3cret"
    chat = FakeChatClient(
        guardrail="0.9",
        generator="ok",
        evaluator='{"score": 0.9, "explanation": "x"}',
    )
    with _client(settings, chat, sample_chunks) as client:
        assert client.post("/query", json={"query": "hi"}).status_code == 401
        assert (
            client.post(
                "/query", json={"query": "hi"}, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        ok = client.post(
            "/query", json={"query": "hi"}, headers={"X-API-Key": "s3cret"}
        )
        assert ok.status_code == 200
        bearer = client.post(
            "/query",
            json={"query": "hi"},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert bearer.status_code == 200
        # Health probe stays open regardless of auth.
        assert client.get("/health").status_code == 200
