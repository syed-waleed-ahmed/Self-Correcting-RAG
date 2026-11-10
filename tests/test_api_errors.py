"""API error contract, request IDs and multi-worker consistency (Phase 1)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.errors import LLMError
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder


def _app(settings, chat_client=None, chunks=None, store=None):
    embedder = FakeEmbedder()
    store = store or NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    if chunks:
        store.add(embedder.embed([c.text for c in chunks]), chunks)
    container = build_container(
        settings, embedder=embedder, store=store, chat_client=chat_client or FakeChatClient()
    )
    return create_app(settings, container=container), container


def _assert_error(resp, status, code):
    assert resp.status_code == status
    body = resp.json()
    assert body["error"] == code
    assert body["request_id"] == resp.headers["X-Request-ID"]
    return body


def test_provider_error_is_sanitized(settings, sample_chunks):
    chat = FakeChatClient(generator=LLMError("upstream said: key sk-secret-123 invalid"))
    app, _ = _app(settings, chat, sample_chunks)
    with TestClient(app) as client:
        resp = client.post("/query", json={"query": "what is a guardrail agent"})
    body = _assert_error(resp, 502, "llm_error")
    assert "sk-secret" not in resp.text
    assert "upstream" not in body["detail"]


def test_unexpected_exception_returns_generic_500(settings, sample_chunks):
    app, container = _app(settings, chunks=sample_chunks)

    def explode(*_args, **_kwargs):
        raise RuntimeError("internal path C:/secret/config.py line 42")

    container.pipeline.run = explode  # type: ignore[method-assign]
    with TestClient(app) as client:
        resp = client.post("/query", json={"query": "hi"})
    body = _assert_error(resp, 500, "internal_error")
    assert "secret" not in resp.text
    assert body["detail"] == "An internal error occurred."


def test_whitespace_query_is_a_400_not_a_500(settings, sample_chunks):
    app, _ = _app(settings, chunks=sample_chunks)
    with TestClient(app) as client:
        resp = client.post("/query", json={"query": "   "})
    _assert_error(resp, 400, "invalid_query")


def test_missing_docs_dir_is_a_clean_409_without_paths(settings):
    app, _ = _app(settings)  # settings.docs_dir was never created
    with TestClient(app) as client:
        resp = client.post("/ingest", json={"reset": True})
    _assert_error(resp, 409, "documents_not_found")
    assert str(settings.docs_dir.name) not in resp.json()["detail"]


def test_concurrent_ingest_is_rejected(settings):
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "a.txt").write_text("rag retrieval", encoding="utf-8")
    app, container = _app(settings)
    with TestClient(app) as client:
        assert container.ingest_lock.acquire(blocking=False)  # an ingest is "running"
        try:
            _assert_error(client.post("/ingest", json={}), 409, "ingest_in_progress")
        finally:
            container.ingest_lock.release()
        assert client.post("/ingest", json={}).status_code == 200


def test_reingest_without_reset_reports_skipped(settings):
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "a.txt").write_text("rag retrieval", encoding="utf-8")
    app, _ = _app(settings)
    with TestClient(app) as client:
        first = client.post("/ingest", json={"reset": True}).json()
        second = client.post("/ingest", json={"reset": False}).json()
    assert second["index_size"] == first["index_size"]
    assert second["skipped_documents"] == 1
    assert second["chunks"] == 0


def test_query_response_exposes_attempt_history(settings, sample_chunks):
    chat = FakeChatClient(
        guardrail="0.9",
        generator=["draft", "final"],
        evaluator=[
            '{"score": 0.3, "explanation": "missing details"}',
            '{"score": 0.8, "explanation": "good"}',
        ],
    )
    app, _ = _app(settings, chat, sample_chunks)
    with TestClient(app) as client:
        body = client.post("/query", json={"query": "what is a guardrail agent"}).json()
    assert body["answer"] == "final"
    assert (body["attempts"], body["best_attempt"]) == (2, 2)
    assert body["evaluation_status"] == "passed"
    assert [a["score"] for a in body["attempt_history"]] == [0.3, 0.8]


def test_evaluator_outage_is_reported_as_unavailable(settings, sample_chunks):
    app, _ = _app(settings, FakeChatClient(evaluator=LLMError("down")), sample_chunks)
    with TestClient(app) as client:
        body = client.post("/query", json={"query": "what is a guardrail agent"}).json()
    assert body["evaluation_score"] is None
    assert body["evaluation_status"] == "unavailable"


def test_request_id_is_generated_echoed_and_sanitized(settings):
    app, _ = _app(settings)
    with TestClient(app) as client:
        generated = client.get("/health").headers["X-Request-ID"]
        echoed = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})
        bad_chars = client.get("/health", headers={"X-Request-ID": "id with spaces; evil=1"})
        too_long = client.get("/health", headers={"X-Request-ID": "a" * 200})
    assert len(generated) == 32
    assert echoed.headers["X-Request-ID"] == "trace-abc.123"
    assert bad_chars.headers["X-Request-ID"] != "id with spaces; evil=1"
    assert len(too_long.headers["X-Request-ID"]) == 32  # replaced, not echoed


def test_unauthorized_uses_error_contract(settings, sample_chunks):
    settings.api_auth_token = "s3cret"
    app, _ = _app(settings, chunks=sample_chunks)
    with TestClient(app) as client:
        resp = client.post("/query", json={"query": "hi"}, headers={"X-API-Key": "wrong"})
        non_ascii = client.post(
            "/query", json={"query": "hi"}, headers={"X-API-Key": "pässwörd".encode("latin-1")}
        )
    _assert_error(resp, 401, "unauthorized")
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    assert non_ascii.status_code == 401  # used to raise TypeError -> 500


def test_worker_sees_index_ingested_by_another_worker(settings):
    """Two app instances sharing an index directory, like two uvicorn workers."""
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "a.txt").write_text("a guardrail agent filters context", encoding="utf-8")
    worker_a, _ = _app(settings)
    worker_b, _ = _app(settings)

    with TestClient(worker_a) as a, TestClient(worker_b) as b:
        assert b.post("/query", json={"query": "guardrail"}).status_code == 409  # empty
        assert a.post("/ingest", json={"reset": True}).status_code == 200
        resp = b.post("/query", json={"query": "what does a guardrail agent do"})
    assert resp.status_code == 200
    assert resp.json()["used_chunks"][0]["source"] == "a.txt"
