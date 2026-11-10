"""Demo-mode protections, answer caching and the v1 API (Phase 2b)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.demo import AnswerCache, DailyBudget, RateLimiter, normalize_query
from scrag.errors import DemoBudgetExhaustedError, LLMError, RateLimitedError
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder

QUESTION = "what is a guardrail agent"


# --- unit: rate limiter, budget, cache ----------------------------------------------
def test_rate_limiter_allows_burst_then_refills():
    now = [0.0]
    limiter = RateLimiter(per_minute=6, burst=2, clock=lambda: now[0])
    limiter.check("ip")
    limiter.check("ip")
    with pytest.raises(RateLimitedError) as info:
        limiter.check("ip")
    assert info.value.headers["Retry-After"] == "10"  # 6/min = one token per 10 s
    limiter.check("other-ip")  # buckets are per key
    now[0] = 10.0
    limiter.check("ip")


def test_daily_budget_exhausts_and_resets_next_utc_day():
    day = [datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)]
    budget = DailyBudget(2, now=lambda: day[0])
    budget.consume()
    budget.consume()
    with pytest.raises(DemoBudgetExhaustedError) as info:
        budget.consume()
    assert int(info.value.headers["Retry-After"]) == 3600
    budget.refund()
    assert budget.status()["remaining"] == 1
    day[0] += timedelta(hours=2)
    assert budget.status()["remaining"] == 2


def test_answer_cache_ttl_lru_and_fingerprint():
    now = [1000.0]
    cache = AnswerCache(ttl_seconds=60, max_entries=2, clock=lambda: now[0])
    cache.put("Q one?", 3, "fp", {"answer": "1"})
    cache.put("q two", 3, "fp", {"answer": "2"})
    assert cache.get("  q ONE ", 3, "fp").response["answer"] == "1"  # normalized key
    assert cache.get("q one", 3, "other-corpus") is None
    assert cache.get("q one", 5, "fp") is None  # top_k is part of the key
    cache.put("q three", 3, "fp", {"answer": "3"})  # evicts LRU ("q two")
    assert cache.get("q two", 3, "fp") is None
    now[0] += 61
    assert cache.get("q one", 3, "fp") is None  # expired


def test_normalize_query():
    assert normalize_query("  What IS  RAG?? ") == "what is rag"


# --- API ---------------------------------------------------------------------------------
def _client(settings, sample_chunks, chat=None, **overrides):
    for key, value in overrides.items():
        setattr(settings, key, value)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    chat = chat or FakeChatClient(guardrail='{"scores": [0.9, 0.9, 0.9]}')
    container = build_container(settings, embedder=embedder, store=store, chat_client=chat)
    return TestClient(create_app(settings, container=container)), container, chat


def test_v1_and_legacy_paths_both_work(settings, sample_chunks):
    client, _, _ = _client(settings, sample_chunks)
    with client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/health").status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/query" in paths
    assert "/query" not in paths  # legacy routes hidden from the schema


def test_search_is_retrieval_only(settings, sample_chunks):
    client, _, chat = _client(settings, sample_chunks)
    with client:
        body = client.post("/api/v1/search", json={"query": "guardrail agent filter"}).json()
    assert body["retrieval_mode"] == "hybrid"
    assert body["chunks"][0]["source"] == "doc1.txt"
    assert body["chunks"][0]["fusion_score"] is not None
    assert chat.calls == []  # no LLM calls at all


def test_repeated_query_is_served_from_cache(settings, sample_chunks):
    client, _, chat = _client(settings, sample_chunks)
    with client:
        first = client.post("/api/v1/query", json={"query": QUESTION}).json()
        calls_after_first = len(chat.calls)
        second = client.post("/api/v1/query", json={"query": QUESTION + "?"}).json()
        assert len(chat.calls) == calls_after_first  # no new LLM calls
        live = client.post("/api/v1/query", json={"query": QUESTION, "use_cache": False}).json()
    assert first["cached"] is False
    assert second["cached"] is True and second["cache_source"] == "recent"
    assert second["cached_at"]
    assert second["answer"] == first["answer"]
    assert live["cached"] is False
    assert len(chat.calls) > calls_after_first


def test_recorded_samples_served_only_for_matching_corpus(settings, sample_chunks, tmp_path):
    from scrag.demo import corpus_fingerprint

    samples = tmp_path / "samples.json"

    def write(fingerprint):
        response = {
            "query": "Sample Q?",
            "answer": "recorded answer",
            "evaluation_score": 0.9,
            "evaluation_explanation": "ok",
            "evaluation_status": "passed",
            "attempts": 1,
            "best_attempt": 1,
            "attempt_history": [],
            "used_chunks": [],
        }
        samples.write_text(
            json.dumps(
                {
                    "recorded_at": "2026-11-10T00:00:00+00:00",
                    "corpus_fingerprint": fingerprint,
                    "items": [{"question": "Sample Q?", "response": response}],
                }
            ),
            encoding="utf-8",
        )

    write(corpus_fingerprint(sample_chunks))
    client, _, chat = _client(settings, sample_chunks, samples_path=samples)
    with client:
        info = client.get("/api/v1/demo").json()
        hit = client.post("/api/v1/query", json={"query": "sample q"}).json()
    assert info["sample_questions"] == ["Sample Q?"]
    assert hit["answer"] == "recorded answer"
    assert hit["cache_source"] == "sample"
    assert hit["cached_at"] == "2026-11-10T00:00:00+00:00"
    assert chat.calls == []

    write("stale-corpus")
    client, _, chat = _client(settings, sample_chunks, samples_path=samples)
    with client:
        miss = client.post("/api/v1/query", json={"query": "sample q"}).json()
    assert miss["cached"] is False  # recorded on a different corpus: run live
    assert chat.calls


def test_demo_rate_limit_returns_429_with_retry_after(settings, sample_chunks):
    client, _, _ = _client(
        settings,
        sample_chunks,
        demo_mode=True,
        demo_rate_limit_burst=2,
        demo_rate_limit_per_minute=1,
    )
    with client:
        assert client.post("/api/v1/search", json={"query": "rag"}).status_code == 200
        assert client.post("/api/v1/search", json={"query": "rag"}).status_code == 200
        resp = client.post("/api/v1/search", json={"query": "rag"})
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited"
    assert resp.headers["Retry-After"] == "60"


def test_demo_budget_exhaustion_and_refund(settings, sample_chunks):
    chat = FakeChatClient(guardrail='{"scores": [0.9, 0.9, 0.9]}', generator=[LLMError("down")])
    client, container, _ = _client(
        settings,
        sample_chunks,
        chat=chat,
        demo_mode=True,
        demo_daily_query_budget=1,
        demo_rate_limit_burst=10,
    )
    with client:
        failed = client.post("/api/v1/query", json={"query": "first question"})
        assert failed.status_code == 502
        assert container.budget.status()["remaining"] == 1  # refunded: no answer given
        assert client.post("/api/v1/query", json={"query": "second question"}).status_code == 200
        exhausted = client.post("/api/v1/query", json={"query": "third question"})
        cached = client.post("/api/v1/query", json={"query": "second question"})
    assert exhausted.status_code == 429
    assert exhausted.json()["error"] == "demo_budget_exhausted"
    assert cached.status_code == 200 and cached.json()["cached"] is True  # still served


def test_demo_query_length_and_top_k_caps(settings, sample_chunks):
    client, _, _ = _client(
        settings,
        sample_chunks,
        demo_mode=True,
        demo_max_query_chars=10,
        demo_top_k=1,
        demo_rate_limit_burst=10,
    )
    with client:
        too_long = client.post("/api/v1/search", json={"query": "x" * 11})
        capped = client.post("/api/v1/search", json={"query": "rag", "top_k": 50}).json()
    assert too_long.status_code == 400
    assert too_long.json()["error"] == "query_too_long"
    assert len(capped["chunks"]) == 1


def test_demo_ingest_requires_operator_token(settings, sample_chunks):
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "a.txt").write_text("rag retrieval", encoding="utf-8")
    client, _, _ = _client(settings, sample_chunks, demo_mode=True)
    with client:
        assert client.post("/api/v1/ingest", json={}).json()["error"] == "forbidden"
    client, _, _ = _client(settings, sample_chunks, demo_mode=True, api_auth_token="op-token")
    with client:
        assert client.post("/api/v1/ingest", json={}).status_code == 401
        ok = client.post("/api/v1/ingest", json={}, headers={"X-API-Key": "op-token"})
    assert ok.status_code == 200


def test_forwarded_ip_uses_trusted_hop(settings, sample_chunks):
    client, _, _ = _client(
        settings,
        sample_chunks,
        demo_mode=True,
        demo_rate_limit_burst=1,
        demo_rate_limit_per_minute=1,
        forwarded_ip_header="x-forwarded-for",
        forwarded_ip_hops=1,
    )
    with client:
        spoof_a = {"X-Forwarded-For": "1.1.1.1, 203.0.113.7"}
        spoof_b = {"X-Forwarded-For": "2.2.2.2, 203.0.113.7"}
        assert (
            client.post("/api/v1/search", json={"query": "rag"}, headers=spoof_a).status_code == 200
        )
        # Same real client (right-most hop), different spoofed left-most value: still limited.
        resp = client.post("/api/v1/search", json={"query": "rag"}, headers=spoof_b)
        other = {"X-Forwarded-For": "198.51.100.9"}
        assert (
            client.post("/api/v1/search", json={"query": "rag"}, headers=other).status_code == 200
        )
    assert resp.status_code == 429


def test_demo_read_endpoints_are_public_even_with_operator_token(settings, sample_chunks):
    """Regression (found on the first live deploy): the operator token must not
    lock anonymous visitors out of /search and /query in demo mode."""
    client, _, _ = _client(
        settings,
        sample_chunks,
        demo_mode=True,
        api_auth_token="op-token",
        demo_rate_limit_burst=10,
    )
    with client:
        assert client.post("/api/v1/search", json={"query": "rag"}).status_code == 200
        assert client.post("/api/v1/query", json={"query": QUESTION}).status_code == 200
        assert client.post("/api/v1/ingest", json={}).status_code == 401  # still protected


def test_token_still_protects_everything_outside_demo_mode(settings, sample_chunks):
    client, _, _ = _client(settings, sample_chunks, api_auth_token="op-token")
    with client:
        assert client.post("/api/v1/search", json={"query": "rag"}).status_code == 401
        assert client.post("/api/v1/query", json={"query": QUESTION}).status_code == 401
