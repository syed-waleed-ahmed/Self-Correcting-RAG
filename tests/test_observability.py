"""Phase 6: Prometheus metrics, usage/timing reporting and OpenTelemetry spans."""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.observability import configure_tracing, metrics, shutdown_tracing
from scrag.orchestration import ToolRegistry, calculator_tool
from scrag.usage import UsageLedger, current_ledger, estimate_cost, track_usage
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder

MODEL = "qwen/qwen3.8-27b"  # the configured default for every agent


class MeteredChat(FakeChatClient):
    """A fake client that reports usage like the real one (10 in / 5 out per call)."""

    def complete(self, *, model, **kwargs):  # noqa: ANN001, ANN003
        ledger = current_ledger()
        if ledger is not None:
            ledger.record(model, 10, 5)
        return super().complete(model=model, **kwargs)


def _app(settings, sample_chunks, chat=None, **overrides):
    for key, value in overrides.items():
        setattr(settings, key, value)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    container = build_container(
        settings, embedder=embedder, store=store, chat_client=chat or FakeChatClient()
    )
    return TestClient(create_app(settings, container=container))


# --- /metrics ----------------------------------------------------------------------------
def test_requests_and_pipeline_stages_are_measured(settings, sample_chunks):
    labels = {"method": "POST", "route": "/api/v1/query", "status": "200"}
    before = {
        "requests": metrics.sample("scrag_http_requests_total", labels),
        "retrieve": metrics.sample(
            "scrag_pipeline_stage_duration_seconds_count", {"stage": "retrieve"}
        ),
        "passed": metrics.sample("scrag_pipeline_runs_total", {"status": "passed"}),
    }
    with _app(settings, sample_chunks) as client:
        assert client.post("/api/v1/query", json={"query": "what is rag"}).status_code == 200
        body = client.get("/api/v1/metrics")
    assert body.status_code == 200 and body.headers["content-type"].startswith("text/plain")
    assert metrics.sample("scrag_http_requests_total", labels) == before["requests"] + 1
    assert (
        metrics.sample("scrag_pipeline_stage_duration_seconds_count", {"stage": "retrieve"})
        == before["retrieve"] + 1
    )
    assert metrics.sample("scrag_pipeline_runs_total", {"status": "passed"}) == before["passed"] + 1
    assert 'scrag_build_info{version="' in body.text


def test_route_labels_are_templates_never_raw_paths(settings, sample_chunks):
    with _app(settings, sample_chunks) as client:
        client.get("/api/v1/documents/1b4e28ba-2fa1-11d2-883f-0016d3cca427")
        client.get("/wp-admin/../../etc/passwd-probe-12345")
        text = client.get("/api/v1/metrics").text
    assert 'route="/api/v1/documents/{document_id}"' in text
    assert "1b4e28ba" not in text and "passwd-probe" not in text
    assert 'route="unmatched"' in text


def test_metrics_need_the_operator_on_the_demo(settings, sample_chunks):
    with _app(settings, sample_chunks, demo_mode=True, api_auth_token="op") as client:
        assert client.get("/api/v1/metrics").status_code == 401
        assert client.get("/api/v1/metrics", headers={"X-API-Key": "op"}).status_code == 200
    with _app(settings, sample_chunks, demo_mode=True, api_auth_token=None) as client:
        assert client.get("/api/v1/metrics").status_code == 403  # no operator configured


# --- usage, cost and timings ---------------------------------------------------------------
def test_answers_report_real_usage_timings_and_an_optional_cost(settings, sample_chunks):
    chat = MeteredChat()
    with _app(settings, sample_chunks, chat=chat) as client:
        plain = client.post("/api/v1/query", json={"query": "what is rag", "use_cache": False})
    usage = plain.json()["usage"]
    calls = len(chat.calls)
    assert usage["llm_calls"] == calls and usage["total_tokens"] == calls * 15
    assert usage["by_model"][MODEL]["calls"] == calls
    assert usage["estimated_cost_usd"] is None  # no prices configured: never guessed
    assert set(plain.json()["timings_ms"]) == {
        "retrieve",
        "guardrail",
        "generate",
        "evaluate",
        "total",
    }

    priced = MeteredChat()
    prices = {MODEL: {"input": 1.0, "output": 2.0}}
    with _app(settings, sample_chunks, chat=priced, llm_prices=prices) as client:
        body = client.post("/api/v1/query", json={"query": "what is rag", "use_cache": False})
    expected = len(priced.calls) * (10 * 1.0 + 5 * 2.0) / 1_000_000
    assert body.json()["usage"]["estimated_cost_usd"] == pytest.approx(expected)


def test_nested_ledgers_forward_usage_and_cost_needs_every_price():
    outer, inner = UsageLedger(), UsageLedger()
    with track_usage(outer), track_usage(inner):
        current_ledger().record("a", 100, 10)
    assert inner.total_tokens == 110 and outer.total_tokens == 110
    assert estimate_cost({}, {}) == 0.0
    assert estimate_cost(outer.by_model, {}) is None
    assert estimate_cost(outer.by_model, {"a": {"input": 2.0, "output": 4.0}}) == pytest.approx(
        (100 * 2 + 10 * 4) / 1_000_000
    )


# --- LLM and tool metrics --------------------------------------------------------------------
def test_llm_calls_are_counted_by_outcome(monkeypatch):
    from scrag.errors import LLMRateLimitError
    from scrag.llm.client import OpenAICompatibleClient

    client = OpenAICompatibleClient(api_key="k")
    ok = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))],
        usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )
    before_ok = metrics.sample("scrag_llm_calls_total", {"model": "m9", "outcome": "ok"})
    before_tokens = metrics.sample("scrag_llm_tokens_total", {"model": "m9", "kind": "prompt"})
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: ok)
    client.complete(model="m9", system="s", user="u")

    class Limited(Exception):
        status_code = 429

    def limited(**kw):
        raise Limited("slow down")

    monkeypatch.setattr(client._client.chat.completions, "create", limited)
    with pytest.raises(LLMRateLimitError):
        client.complete(model="m9", system="s", user="u")
    assert (
        metrics.sample("scrag_llm_calls_total", {"model": "m9", "outcome": "ok"}) == before_ok + 1
    )
    assert metrics.sample("scrag_llm_calls_total", {"model": "m9", "outcome": "rate_limited"}) >= 1
    assert (
        metrics.sample("scrag_llm_tokens_total", {"model": "m9", "kind": "prompt"})
        == before_tokens + 7
    )


def test_unknown_tool_names_never_become_metric_labels():
    registry = ToolRegistry()
    registry.register(calculator_tool())
    before = metrics.sample("scrag_tool_calls_total", {"tool": "unknown", "status": "denied"})
    registry.invoke("drop table users; --", {})
    registry.invoke("calculator", {"expression": "2+2"})
    assert (
        metrics.sample("scrag_tool_calls_total", {"tool": "unknown", "status": "denied"})
        == before + 1
    )
    assert metrics.sample("scrag_tool_calls_total", {"tool": "calculator", "status": "ok"}) >= 1


# --- tracing -----------------------------------------------------------------------------------
def test_spans_nest_from_the_http_request_down_to_each_stage(settings, sample_chunks):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    assert configure_tracing("none", span_exporter=exporter, version="test")
    try:
        with _app(settings, sample_chunks, chat=MeteredChat()) as client:
            response = client.post("/api/v1/query", json={"query": "what is rag"})
    finally:
        shutdown_tracing()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    http = spans["HTTP POST /api/v1/query"]
    run = spans["pipeline.run"]
    assert run.parent.span_id == http.context.span_id
    for stage in ("pipeline.retrieve", "pipeline.guardrail", "pipeline.generate"):
        assert spans[stage].parent.span_id == run.context.span_id, stage
    assert http.attributes["http.response.status_code"] == 200
    assert http.attributes["scrag.request_id"] == response.headers["x-request-id"]
    assert run.attributes["scrag.status"] == "passed"


def test_tracing_is_off_by_default_and_costs_nothing():
    from scrag.observability import span, tracing_enabled

    shutdown_tracing()
    assert not tracing_enabled()
    with span("anything", key="value") as current:
        assert current is None


def test_llm_falls_back_when_a_model_is_rate_limited_or_retired(monkeypatch):
    from scrag.errors import LLMError, LLMModelUnavailableError
    from scrag.llm.client import OpenAICompatibleClient

    class Limited(Exception):
        status_code = 429

    class Retired(Exception):
        status_code = 400
        code = "model_decommissioned"

    class Broken(Exception):
        status_code = 500

    ok = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="from fallback"))],
        usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    tried: list[str] = []
    failures = {"primary": Limited("slow down"), "retired": Retired("gone")}

    def create(**kwargs):
        tried.append(kwargs["model"])
        if kwargs["model"] in failures:
            raise failures[kwargs["model"]]
        if kwargs["model"] == "broken":
            raise Broken("server error")
        return ok

    client = OpenAICompatibleClient(api_key="k", fallback_models=["retired", "spare"])
    monkeypatch.setattr(client._client.chat.completions, "create", create)
    assert client.complete(model="primary", system="s", user="u") == "from fallback"
    assert tried == ["primary", "retired", "spare"]

    # Other errors aren't retried on another model: they mean a bad request or outage.
    tried.clear()
    with pytest.raises(LLMError):
        client.complete(model="broken", system="s", user="u")
    assert tried == ["broken"]

    # Without fallbacks, a retired model surfaces as its own error.
    bare = OpenAICompatibleClient(api_key="k")
    monkeypatch.setattr(bare._client.chat.completions, "create", create)
    with pytest.raises(LLMModelUnavailableError):
        bare.complete(model="retired", system="s", user="u")
