"""Phase 4 persistence against a real Postgres + pgvector (needs TEST_DATABASE_URL).

Acceptance criteria covered here:
* data survives a backend restart;
* tenant isolation: session A can't read session B's documents, chunks,
  conversations or jobs;
* a job interrupted by a restart (expired lease) resumes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.chunking import Chunker
from scrag.container import build_container
from scrag.errors import (
    DemoBudgetExhaustedError,
    NotFoundError,
    SessionQuotaExceededError,
    StorageFullError,
)
from scrag.ingest import content_hash
from scrag.persistence import (
    Database,
    JobHandler,
    JobQueue,
    JobWorker,
    PermanentJobError,
    PersistentDailyBudget,
    apply_migrations,
    build_persistence,
    current_version,
    rollback_migrations,
)
from scrag.persistence.sessions import hash_token
from scrag.pipeline import NO_CONTEXT_ANSWER
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, TEST_DATABASE_URL, FakeChatClient, FakeEmbedder

pytestmark = pytest.mark.postgres

GUARDRAIL_DOC = "A guardrail agent can filter off-topic context before generation."
PASS_EVAL = '{"score": 0.9, "explanation": "supported"}'
EMBEDDING_DOC = "An embedding maps text to a vector; vector retrieval finds similar text."


def _persistence(settings, database, **overrides):
    for key, value in {"job_worker_enabled": False, **overrides}.items():
        setattr(settings, key, value)
    chunker = Chunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    return build_persistence(settings, embedder=FakeEmbedder(), chunker=chunker, database=database)


def _upload(p, session_id, text, filename="notes.md"):
    return p.documents.create(
        session_id,
        filename=filename,
        content=text,
        content_hash=content_hash(text),
        size_bytes=len(text),
    )


def _count(database, table):
    with database.connection() as conn:
        return conn.execute(f"SELECT count(*) AS n FROM scrag.{table}").fetchone()["n"]


# --- migrations ----------------------------------------------------------------------
def test_migrations_are_idempotent_and_reversible(database):
    with database.connection() as conn:
        assert current_version(conn) == 1
        assert apply_migrations(conn) == []  # already applied
        assert rollback_migrations(conn, target=0) == [1]
        gone = conn.execute("SELECT to_regclass('scrag.sessions') AS t").fetchone()["t"]
        assert gone is None
        assert apply_migrations(conn) == [1]
        assert current_version(conn) == 1


# --- sessions --------------------------------------------------------------------------
def test_session_token_is_hashed_and_authenticates(settings, database):
    p = _persistence(settings, database)
    record, token = p.sessions.create()
    assert token.startswith("scs_") and len(token) > 40
    assert p.sessions.authenticate(token).id == record.id
    for bad in (None, "", "garbage", token + "x", "scs_" + "a" * 200):
        assert p.sessions.authenticate(bad) is None
    with database.connection() as conn:
        stored = conn.execute("SELECT token_hash FROM scrag.sessions").fetchone()["token_hash"]
    assert bytes(stored) == hash_token(token) and token.encode() not in bytes(stored)


def test_expired_sessions_are_rejected_then_purged(settings, database):
    p = _persistence(settings, database)
    _, token = p.sessions.create()
    with database.connection() as conn:
        conn.execute("UPDATE scrag.sessions SET expires_at = now() - interval '1 second'")
    assert p.sessions.authenticate(token) is None
    assert p.sessions.purge_expired() == 1
    assert _count(database, "sessions") == 0


def test_deleting_a_session_deletes_all_of_its_data(settings, database):
    p = _persistence(settings, database)
    session, _ = p.sessions.create()
    _upload(p, session.id, GUARDRAIL_DOC)
    p.worker.run_until_idle()
    p.conversations.append_turn(session.id, None, "q?", "a.", {})
    assert _count(database, "chunks") == 1 and _count(database, "messages") == 2

    deleted = p.sessions.delete(session.id)
    assert deleted is True
    for table in ("documents", "chunks", "conversations", "messages", "jobs"):
        assert _count(database, table) == 0, table


# --- documents and ingestion -------------------------------------------------------------
def test_upload_is_ingested_by_a_job_and_searchable(settings, database):
    p = _persistence(settings, database)
    session, _ = p.sessions.create()
    document, job_id = _upload(p, session.id, GUARDRAIL_DOC)
    assert document.status == "queued" and job_id is not None

    assert p.worker.run_until_idle() == 1
    ready = p.documents.get(session.id, document.id)
    assert ready.status == "ready" and ready.chunk_count == 1
    assert p.jobs.get(job_id).status == "succeeded"

    query = FakeEmbedder().embed(["guardrail filter"])[0]
    dense = p.documents.search_dense(session.id, query, 5, embedding_model=settings.embedding_model)
    assert [hit.filename for hit in dense] == ["notes.md"] and 0 < dense[0].score <= 1
    keyword = p.documents.search_keyword(session.id, "What does the guardrail do?", 5)
    assert [hit.document_id for hit in keyword] == [document.id]
    # Another embedding model's vectors are never compared with this one's.
    assert p.documents.search_dense(session.id, query, 5, embedding_model="other") == []


def test_duplicate_upload_returns_the_existing_document(settings, database):
    p = _persistence(settings, database)
    session, _ = p.sessions.create()
    first, job = _upload(p, session.id, GUARDRAIL_DOC)
    again, no_job = _upload(p, session.id, GUARDRAIL_DOC, filename="copy.md")
    assert job is not None and no_job is None and again.id == first.id


def test_per_session_and_global_document_quotas(settings, database):
    p = _persistence(settings, database, session_max_documents=1, persistence_max_documents=2)
    a, _ = p.sessions.create()
    b, _ = p.sessions.create()
    c, _ = p.sessions.create()
    _upload(p, a.id, GUARDRAIL_DOC)
    with pytest.raises(SessionQuotaExceededError):
        _upload(p, a.id, EMBEDDING_DOC)
    _upload(p, b.id, EMBEDDING_DOC)
    with pytest.raises(StorageFullError):
        _upload(p, c.id, GUARDRAIL_DOC)


def test_too_long_document_fails_its_job_permanently(settings, database):
    p = _persistence(settings, database, document_max_chunks=1)
    session, _ = p.sessions.create()
    document, job_id = _upload(p, session.id, "rag retrieval " * 400)  # several chunks
    p.worker.run_until_idle()
    failed = p.documents.get(session.id, document.id)
    assert failed.status == "failed" and failed.error == "document_too_long"
    job = p.jobs.get(job_id)
    assert job.status == "failed" and job.attempts == 1  # permanent: no retries


def test_document_deleted_before_processing_is_skipped(settings, database):
    p = _persistence(settings, database)
    session, _ = p.sessions.create()
    document, job_id = _upload(p, session.id, GUARDRAIL_DOC)
    p.documents.delete(session.id, document.id)
    assert p.worker.run_until_idle() == 1
    job = p.jobs.get(job_id)
    assert job.status == "succeeded" and job.result == {"skipped": "document deleted"}
    assert _count(database, "chunks") == 0


# --- job queue semantics ---------------------------------------------------------------
def _queue_with(database, handler, *, max_attempts=3):
    queue = JobQueue(database, max_attempts=max_attempts, backoff_base_s=0)
    return queue, JobWorker(queue, {"test": handler}, lease_s=30)


def test_failed_job_is_retried_then_succeeds(database):
    calls = []

    def flaky(job):
        calls.append(job.attempts)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    queue, worker = _queue_with(database, JobHandler(run=flaky))
    job_id = queue.enqueue("test", {})
    assert worker.run_once() is True
    retrying = queue.get(job_id)
    assert retrying.status == "queued" and retrying.last_error == "RuntimeError: transient"
    assert worker.run_once() is True
    done = queue.get(job_id)
    assert done.status == "succeeded" and done.attempts == 2 and done.result == {"ok": True}
    assert calls == [1, 2]


def test_job_fails_after_max_attempts_and_runs_its_failure_hook(database):
    failures = []
    handler = JobHandler(
        run=lambda job: 1 / 0, on_failure=lambda job, exc: failures.append(type(exc))
    )
    queue, worker = _queue_with(database, handler, max_attempts=2)
    job_id = queue.enqueue("test", {})
    worker.run_until_idle()
    job = queue.get(job_id)
    assert job.status == "failed" and job.attempts == 2
    assert failures == [ZeroDivisionError]


def test_permanent_error_skips_retries(database):
    def invalid(job):
        raise PermanentJobError("bad_input")

    queue, worker = _queue_with(database, JobHandler(run=invalid))
    job_id = queue.enqueue("test", {})
    worker.run_until_idle()
    job = queue.get(job_id)
    assert job.status == "failed" and job.attempts == 1 and job.last_error == "bad_input"


def test_job_interrupted_by_a_restart_resumes_after_its_lease_expires(database):
    runs = []
    queue, worker = _queue_with(database, JobHandler(run=lambda job: runs.append(job) or {}))
    job_id = queue.enqueue("test", {})

    # A previous process claimed the job and died mid-run.
    dead = queue.claim("dead-worker", lease_s=60)
    assert dead is not None and dead.id == job_id
    assert worker.run_once() is False  # the lease is still held

    with database.connection() as conn:  # time passes: the lease expires
        conn.execute("UPDATE scrag.jobs SET locked_until = now() - interval '1 second'")
    assert worker.run_once() is True
    job = queue.get(job_id)
    assert job.status == "succeeded" and job.attempts == 2 and len(runs) == 1
    # The dead worker can't overwrite the result if it comes back.
    assert queue.complete(dead, "dead-worker", {"late": True}) is False


def test_worker_thread_wakes_on_enqueue_and_processes(database):
    done = []
    queue = JobQueue(database, backoff_base_s=0)
    worker = JobWorker(queue, {"test": JobHandler(run=lambda job: done.append(job.id) or {})})
    worker.start()
    try:
        job_id = queue.enqueue("test", {})
        deadline = time.monotonic() + 10
        while not done and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        worker.stop()
    assert done == [job_id]


# --- budget ----------------------------------------------------------------------------------
def test_daily_budget_survives_restarts_and_resets_daily(database):
    now = [datetime(2026, 11, 10, 12, tzinfo=timezone.utc)]
    first = PersistentDailyBudget(database, 3, now=lambda: now[0])
    first.consume(2)
    second = PersistentDailyBudget(database, 3, now=lambda: now[0])  # "after a restart"
    assert second.status()["remaining"] == 1
    second.consume()
    with pytest.raises(DemoBudgetExhaustedError):
        second.consume()
    second.refund()
    assert first.status()["remaining"] == 1
    now[0] += timedelta(days=1)
    assert first.status()["remaining"] == 3


# --- conversations -----------------------------------------------------------------------------
def test_conversation_turns_and_message_quota(settings, database):
    p = _persistence(settings, database)
    p.conversations.max_messages = 4
    session, _ = p.sessions.create()
    conv = p.conversations.append_turn(session.id, None, "What is RAG?", "An approach.", {"x": 1})
    assert p.conversations.append_turn(session.id, conv, "And BM25?", "Keywords.", {}) == conv
    assert p.conversations.append_turn(session.id, conv, "More?", "No.", {}) is None  # quota
    detail = p.conversations.get(session.id, conv)
    assert detail.title == "What is RAG?"
    assert [m.role for m in detail.messages] == ["user", "assistant", "user", "assistant"]
    assert detail.messages[1].metadata == {"x": 1}
    other, _ = p.sessions.create()
    with pytest.raises(NotFoundError):
        p.conversations.get(other.id, conv)


# --- API ----------------------------------------------------------------------------------------
def _app(settings, chat=None, samples=None, **overrides):
    """An app on its own Database instance (as a separate server process would be)."""
    for key, value in {"job_worker_enabled": False, **overrides}.items():
        setattr(settings, key, value)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    if samples:
        store.add(embedder.embed([c.text for c in samples]), samples)
    container = build_container(
        settings,
        embedder=embedder,
        store=store,
        chat_client=chat or FakeChatClient(),
        database=Database(TEST_DATABASE_URL, retry_interval_s=0),
    )
    return TestClient(create_app(settings, container=container)), container


def _session(client):
    response = client.post("/api/v1/sessions")
    assert response.status_code == 201, response.text
    return {"X-Session-Token": response.json()["token"]}


def _upload_via_api(client, container, headers, text=GUARDRAIL_DOC, name="notes.md"):
    response = client.post(
        "/api/v1/documents", headers=headers, files={"file": (name, text.encode(), "text/plain")}
    )
    assert response.status_code == 202, response.text
    container.persistence.worker.run_until_idle()
    return response.json()


def test_api_session_upload_query_and_history(settings, database):
    client, container = _app(settings)
    with client:
        me = _session(client)
        upload = _upload_via_api(client, container, me)
        job = client.get(f"/api/v1/jobs/{upload['job_id']}", headers=me).json()
        assert job["status"] == "succeeded" and "payload" not in job
        docs = client.get("/api/v1/documents", headers=me).json()
        assert [(d["filename"], d["status"]) for d in docs] == [("notes.md", "ready")]

        ask = {"query": "What does the guardrail filter?", "scope": "session"}
        first = client.post("/api/v1/query", headers=me, json=ask).json()
        assert first["scope"] == "session" and first["history_saved"] is True
        assert {c["source"] for c in first["used_chunks"]} == {"notes.md"}
        conv = first["conversation_id"]
        second = client.post(
            "/api/v1/query", headers=me, json={**ask, "conversation_id": conv}
        ).json()
        assert second["conversation_id"] == conv

        listed = client.get("/api/v1/conversations", headers=me).json()
        assert [(c["id"], c["message_count"]) for c in listed] == [(conv, 4)]
        detail = client.get(f"/api/v1/conversations/{conv}", headers=me).json()
        assert detail["messages"][1]["metadata"]["sources"] == ["notes.md"]
        info = client.get("/api/v1/sessions/me", headers=me).json()
        assert info["usage"] == {"documents": 1, "conversations": 1, "messages": 4}

        deleted = client.delete("/api/v1/sessions/me", headers=me)
        assert deleted.status_code == 204
        gone = client.get("/api/v1/sessions/me", headers=me)
        assert gone.status_code == 401 and gone.json()["error"] == "session_required"


def test_api_sessions_are_isolated_from_each_other(settings, database):
    client, container = _app(settings)
    with client:
        alice, bob = _session(client), _session(client)
        upload = _upload_via_api(client, container, alice)
        doc_id, job_id = upload["document"]["id"], upload["job_id"]
        conv = client.post(
            "/api/v1/query", headers=alice, json={"query": "guardrail?", "scope": "session"}
        ).json()["conversation_id"]

        assert client.get("/api/v1/documents", headers=bob).json() == []
        for method, path in [
            ("get", f"/api/v1/documents/{doc_id}"),
            ("delete", f"/api/v1/documents/{doc_id}"),
            ("get", f"/api/v1/jobs/{job_id}"),
            ("get", f"/api/v1/conversations/{conv}"),
            ("delete", f"/api/v1/conversations/{conv}"),
        ]:
            response = getattr(client, method)(path, headers=bob)
            assert response.status_code == 404, (method, path)
            assert response.json()["error"] == "not_found"
        # Bob can't append to Alice's conversation either.
        hijack = client.post(
            "/api/v1/query",
            headers=bob,
            json={"query": "guardrail?", "scope": "session", "conversation_id": conv},
        )
        assert hijack.status_code == 404
        # Bob's searches never see Alice's chunks.
        found = client.post(
            "/api/v1/search", headers=bob, json={"query": "guardrail filter", "scope": "session"}
        ).json()
        assert found["chunks"] == []
        answer = client.post(
            "/api/v1/query", headers=bob, json={"query": "guardrail filter", "scope": "session"}
        ).json()
        assert answer["answer"] == NO_CONTEXT_ANSWER and answer["used_chunks"] == []
        # Alice's data is untouched.
        assert client.get(f"/api/v1/documents/{doc_id}", headers=alice).status_code == 200


def test_api_data_survives_a_backend_restart(settings, database):
    first, container = _app(settings)
    with first:
        me = _session(first)
        _upload_via_api(first, container, me)
        conv = first.post(
            "/api/v1/query", headers=me, json={"query": "guardrail?", "scope": "session"}
        ).json()["conversation_id"]
    # A new process: new container, new connection pool, same database.
    second, _ = _app(settings)
    with second:
        assert second.get("/api/v1/sessions/me", headers=me).status_code == 200
        docs = second.get("/api/v1/documents", headers=me).json()
        assert [d["status"] for d in docs] == ["ready"]
        assert [c["id"] for c in second.get("/api/v1/conversations", headers=me).json()] == [conv]
        again = second.post(
            "/api/v1/query", headers=me, json={"query": "guardrail filter", "scope": "session"}
        ).json()
        assert {c["source"] for c in again["used_chunks"]} == {"notes.md"}


def test_api_scope_all_fuses_uploads_with_the_sample_corpus(settings, database, sample_chunks):
    client, container = _app(settings, samples=sample_chunks)
    with client:
        me = _session(client)
        _upload_via_api(client, container, me, text=GUARDRAIL_DOC, name="mine.md")
        body = client.post(
            "/api/v1/search",
            headers=me,
            json={"query": "guardrail filter context", "scope": "all", "top_k": 4},
        ).json()
    assert body["retrieval_mode"] == "session+samples"
    sources = [c["source"] for c in body["chunks"]]
    assert "mine.md" in sources and any(s.startswith("doc") for s in sources)


def test_api_upload_validation(settings, database):
    client, container = _app(settings, upload_max_bytes=200, session_max_documents=1)
    with client:
        me = _session(client)

        def post(name, data):
            return client.post("/api/v1/documents", headers=me, files={"file": (name, data)})

        assert post("tool.exe", b"MZ").json()["error"] == "unsupported_file_type"
        assert post("big.txt", b"rag " * 100).status_code == 413
        assert post("empty.txt", b"   ").json()["error"] == "unreadable_document"
        assert post("bad.json", b"{not json").json()["error"] == "unreadable_document"
        assert post("../../etc/ok.txt", b"rag retrieval").status_code == 202
        listed = client.get("/api/v1/documents", headers=me).json()
        assert [d["filename"] for d in listed] == ["ok.txt"]  # path stripped
        quota = post("second.txt", b"vector embedding")
        assert quota.status_code == 409 and quota.json()["error"] == "session_quota_exceeded"
        dup = post("same.txt", b"rag retrieval")
        assert dup.status_code == 200 and dup.json()["duplicate"] is True


def test_api_worker_thread_ingests_uploads(settings, database):
    client, _ = _app(settings, job_worker_enabled=True)
    with client:
        me = _session(client)
        response = client.post(
            "/api/v1/documents", headers=me, files={"file": ("n.md", GUARDRAIL_DOC.encode())}
        )
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 15
        status = None
        while time.monotonic() < deadline:
            status = client.get(f"/api/v1/jobs/{job_id}", headers=me).json()["status"]
            if status == "succeeded":
                break
            time.sleep(0.1)
    assert status == "succeeded"


def test_api_demo_budget_is_persisted(settings, database, sample_chunks):
    client, container = _app(
        settings, samples=sample_chunks, demo_mode=True, demo_rate_limit_burst=20
    )
    with client:
        answered = client.post("/api/v1/query", json={"query": "what is rag", "use_cache": False})
        assert answered.status_code == 200, answered.text
        remaining = client.get("/api/v1/demo").json()["budget"]["remaining"]
    assert isinstance(container.budget, PersistentDailyBudget)
    with database.connection() as conn:
        used = conn.execute("SELECT sum(used) AS n FROM scrag.usage_counters").fetchone()["n"]
    assert used == 1 and remaining == settings.demo_daily_query_budget - 1


def test_api_admin_stats_needs_the_operator(settings, database):
    client, _ = _app(settings, demo_mode=True, api_auth_token="op-secret")
    with client:
        _session(client)
        assert client.get("/api/v1/admin/stats").status_code == 401
        stats = client.get("/api/v1/admin/stats", headers={"X-API-Key": "op-secret"}).json()
    assert stats["sessions"] == 1 and stats["database"] == "ok"


def test_api_agent_can_search_only_the_sessions_uploads(settings, database):
    plan = json.dumps({"steps": [{"tool": "document_search", "input": {"query": "guardrail"}}]})
    chat = FakeChatClient(generator=[plan, "It filters context [S1]."], evaluator=PASS_EVAL)
    client, container = _app(settings, chat=chat)
    with client:
        me = _session(client)
        _upload_via_api(client, container, me, text=GUARDRAIL_DOC, name="mine.md")
        body = client.post(
            "/api/v1/agent/run",
            headers=me,
            json={"task": "What is a guardrail?", "scope": "session"},
        ).json()
        anonymous = client.post(
            "/api/v1/agent/run", json={"task": "What is a guardrail?", "scope": "session"}
        )
    assert body["status"] == "completed"
    hits = body["steps"][0]["call"]["output"]
    assert [h["source"] for h in hits] == ["mine.md"]
    # The session's tools are read-only: no reindex tool was offered to the planner.
    assert "reindex_documents" not in chat.prompts_for("generator")[0]
    assert anonymous.status_code == 401
