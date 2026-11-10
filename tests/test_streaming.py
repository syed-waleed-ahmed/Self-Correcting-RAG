"""Phase 5: progress events and the Server-Sent Events endpoints."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.errors import LLMError
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder

PASS = '{"score": 0.9, "explanation": "supported"}'


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs, skipping comments."""
    events = []
    for block in text.split("\n\n"):
        kind, data = "message", []
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            events.append((kind, json.loads("\n".join(data))))
    return events


def _app(settings, sample_chunks, chat, **overrides):
    for key, value in overrides.items():
        setattr(settings, key, value)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    container = build_container(settings, embedder=embedder, store=store, chat_client=chat)
    return TestClient(create_app(settings, container=container)), container


# --- pipeline events ----------------------------------------------------------------
def test_pipeline_reports_each_stage(settings, sample_chunks):
    chat = FakeChatClient(generator=["first try", "better"], evaluator=['{"score": 0.2}', PASS])
    _, container = _app(settings, sample_chunks, chat)
    events: list[dict] = []
    result = container.pipeline.run("what does a guardrail filter", on_event=events.append)

    stages = [(e["stage"], e["status"], e.get("attempt")) for e in events]
    assert stages == [
        ("retrieve", "done", None),
        ("guardrail", "done", None),
        ("generate", "started", 1),
        ("generate", "done", 1),
        ("evaluate", "ok", 1),
        ("generate", "started", 2),
        ("generate", "done", 2),
        ("evaluate", "ok", 2),
    ]
    assert events[0]["count"] == 3 and events[1]["of"] == 3
    assert [e["passed"] for e in events if e["stage"] == "evaluate"] == [False, True]
    assert result.answer == "better" and result.attempts == 2


def test_a_failing_listener_never_breaks_the_pipeline(settings, sample_chunks):
    _, container = _app(settings, sample_chunks, FakeChatClient())

    def broken(event):
        raise RuntimeError("UI went away")

    result = container.pipeline.run("what is rag", on_event=broken)
    assert result.evaluation_status == "passed"


# --- /query/stream --------------------------------------------------------------------
def test_query_stream_sends_progress_then_the_query_result(settings, sample_chunks):
    client, _ = _app(settings, sample_chunks, FakeChatClient(generator="Streamed answer."))
    with client:
        response = client.post(
            "/api/v1/query/stream", json={"query": "what is rag", "use_cache": False}
        )
        plain = client.post("/api/v1/query", json={"query": "what is rag", "use_cache": False})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result" and set(kinds[:-1]) == {"progress"}
    assert [e["stage"] for k, e in events if k == "progress"][:2] == ["retrieve", "guardrail"]
    result = events[-1][1]
    assert result["answer"] == "Streamed answer."
    assert set(result) == set(plain.json())  # the same body as /query


def test_query_stream_reports_a_cache_hit(settings, sample_chunks):
    client, _ = _app(settings, sample_chunks, FakeChatClient(generator="Fresh."))
    with client:
        client.post("/api/v1/query", json={"query": "what is rag"})  # fills the cache
        events = parse_sse(client.post("/api/v1/query/stream", json={"query": "what is rag"}).text)
    assert events[0] == ("progress", {"stage": "cache", "status": "hit", "source": "recent"})
    assert events[-1][0] == "result" and events[-1][1]["cached"] is True


def test_query_stream_failure_is_an_error_event_and_refunds_the_budget(settings, sample_chunks):
    chat = FakeChatClient(generator=LLMError("provider down"))
    client, container = _app(settings, sample_chunks, chat, demo_mode=True)
    with client:
        response = client.post(
            "/api/v1/query/stream", json={"query": "what is rag", "use_cache": False}
        )
    kind, error = parse_sse(response.text)[-1]
    assert kind == "error"
    assert error["error"] == "llm_error" and error["status"] == 502
    assert "provider down" not in json.dumps(error)  # sanitized
    assert error["request_id"] == response.headers["x-request-id"]
    assert container.budget.status()["remaining"] == settings.demo_daily_query_budget


def test_query_stream_checks_fail_as_plain_http_errors(settings, sample_chunks):
    client, _ = _app(settings, sample_chunks, FakeChatClient(), demo_mode=True)
    with client:
        too_long = client.post("/api/v1/query/stream", json={"query": "x" * 400})
        session_scope = client.post(
            "/api/v1/query/stream", json={"query": "rag", "scope": "session"}
        )
    assert too_long.status_code == 400 and too_long.json()["error"] == "query_too_long"
    assert session_scope.status_code == 503  # no database configured
    assert session_scope.json()["error"] == "persistence_disabled"


# --- /agent/run/stream ----------------------------------------------------------------
def test_agent_stream_relays_the_trace_live(settings, sample_chunks):
    plan = json.dumps({"steps": [{"tool": "document_search", "input": {"query": "guardrail"}}]})
    chat = FakeChatClient(generator=[plan, "A guardrail filters context [S1]."], evaluator=PASS)
    client, _ = _app(settings, sample_chunks, chat)
    with client:
        response = client.post("/api/v1/agent/run/stream", json={"task": "What is a guardrail?"})
    events = parse_sse(response.text)
    progress = [e for k, e in events if k == "progress"]
    assert [e["stage"] for e in progress] == ["plan", "tool", "synthesize", "verify"]
    kind, result = events[-1]
    assert kind == "result" and result["status"] == "completed"
    assert [t["seq"] for t in result["trace"]] == [e["seq"] for e in progress]


def test_demo_info_advertises_features(settings, sample_chunks):
    client, _ = _app(settings, sample_chunks, FakeChatClient())
    with client:
        features = client.get("/api/v1/demo").json()["features"]
    assert features == {"sessions": False, "agent": True, "streaming": True}
