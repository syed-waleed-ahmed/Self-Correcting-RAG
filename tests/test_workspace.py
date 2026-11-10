"""Phase 4 behaviour that needs no running database: disabled persistence, a
database that is down, upload guards and input sanitization."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.errors import DatabaseUnavailableError
from scrag.loaders import LoaderError, extract_upload, safe_filename
from scrag.persistence import Database
from scrag.persistence.documents import keyword_query, vector_literal
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder

# Nothing listens on port 1: connections are refused immediately.
DOWN_URL = "postgresql://scrag@127.0.0.1:1/scrag"


def _client(settings, sample_chunks, *, database=None, **overrides):
    for key, value in {"job_worker_enabled": False, **overrides}.items():
        setattr(settings, key, value)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    container = build_container(
        settings, embedder=embedder, store=store, chat_client=FakeChatClient(), database=database
    )
    return TestClient(create_app(settings, container=container))


# --- persistence disabled (no database configured) ------------------------------------
def test_workspace_endpoints_report_persistence_disabled(settings, sample_chunks):
    with _client(settings, sample_chunks) as client:
        assert client.get("/api/v1/health").json()["persistence"] == "disabled"
        for method, path in [
            ("post", "/api/v1/sessions"),
            ("get", "/api/v1/documents"),
            ("get", "/api/v1/conversations"),
        ]:
            response = getattr(client, method)(path)
            assert response.status_code == 503
            assert response.json()["error"] == "persistence_disabled"
        scoped = client.post("/api/v1/query", json={"query": "rag", "scope": "session"})
        assert scoped.json()["error"] == "persistence_disabled"
        # A session header is ignored for sample questions when there's no database.
        ok = client.post(
            "/api/v1/query", json={"query": "what is rag"}, headers={"X-Session-Token": "scs_x"}
        )
        assert ok.status_code == 200 and ok.json()["conversation_id"] is None


# --- database down ----------------------------------------------------------------------
def test_database_outage_fails_fast_after_the_first_attempt():
    db = Database(DOWN_URL, connect_timeout_s=1, retry_interval_s=60)
    try:
        with pytest.raises(DatabaseUnavailableError):
            db.ensure_ready()
        assert db.status == "unavailable"
        started = time.monotonic()
        with pytest.raises(DatabaseUnavailableError), db.connection():
            pass
        assert time.monotonic() - started < 0.5  # circuit breaker: no new connect attempt
    finally:
        db.close()


def test_sample_questions_still_work_while_the_database_is_down(settings, sample_chunks):
    db = Database(DOWN_URL, connect_timeout_s=1, retry_interval_s=60)
    with _client(
        settings, sample_chunks, database=db, demo_mode=True, demo_rate_limit_burst=10
    ) as client:
        token = {"X-Session-Token": "scs_" + "a" * 43}
        answer = client.post("/api/v1/query", json={"query": "what is rag"}, headers=token)
        assert answer.status_code == 200
        assert answer.json()["history_saved"] is False  # answered, but not saved
        assert client.get("/api/v1/demo").json()["budget"]["remaining"] >= 0  # fallback
        session = client.post("/api/v1/sessions")
        assert session.status_code == 503 and session.json()["error"] == "database_unavailable"
        assert session.headers["Retry-After"] == "30"
        mine = client.post(
            "/api/v1/query", json={"query": "rag", "scope": "session"}, headers=token
        )
        assert mine.status_code == 503
        assert client.get("/api/v1/health").json()["persistence"] == "unavailable"


# --- upload guards ------------------------------------------------------------------------
def test_upload_size_is_enforced_before_the_body_is_parsed(settings, sample_chunks):
    with _client(settings, sample_chunks, upload_max_bytes=10) as client:
        huge = client.post(
            "/api/v1/documents",
            content=b"x" * 100_000,
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
        assert huge.status_code == 413 and huge.json()["error"] == "upload_too_large"
        assert huge.headers.get("x-request-id")

        def chunks():
            yield b"--b\r\n"

        unsized = client.post(
            "/api/v1/documents",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
        assert unsized.status_code == 411


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("notes.md", "notes.md"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("evil\x00name\n.txt", "evil_name_.txt"),
        ("", "upload.txt"),
        (None, "upload.txt"),
        ("..", "upload.txt"),
        ("a" * 300 + ".md", "a" * 117 + ".md"),
    ],
)
def test_safe_filename(raw, safe):
    assert safe_filename(raw) == safe


def test_extract_upload_reuses_the_loaders():
    assert extract_upload(b"<p>Hello <b>RAG</b></p>", "x.html", max_bytes=1000) == "Hello RAG"
    assert extract_upload(b"a,b\n1,2\n", "x.csv", max_bytes=1000) == "a: 1; b: 2"
    with pytest.raises(LoaderError):
        extract_upload(b"MZ", "x.exe", max_bytes=1000)
    with pytest.raises(LoaderError):
        extract_upload(b"\xff\xfe\x00", "x.txt", max_bytes=1000)  # not UTF-8
    with pytest.raises(LoaderError):
        extract_upload(b"x" * 11, "x.txt", max_bytes=10)


# --- SQL input builders --------------------------------------------------------------------
def test_keyword_query_cannot_inject_tsquery_operators():
    # Same tokenizer as the in-process BM25 index; Postgres' english config
    # drops leftovers such as "s" as stopwords.
    assert keyword_query("What's the RRF k-constant?") == "s | rrf | k | constant"
    assert keyword_query("x-request-id & !(drop) | 'quote':*") == (
        "x | request | id | drop | quote"
    )
    assert keyword_query("the and of") is None  # only stopwords


def test_vector_literal():
    assert vector_literal([0.5, -1.0, 1e-9]) == "[0.5,-1,1e-09]"
