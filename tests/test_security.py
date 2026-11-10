"""Phase 7 hardening: upload sniffing, request limits, headers, secrets, quotas."""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from scrag import loaders
from scrag.api import create_app
from scrag.api.deps import client_address
from scrag.config import Settings
from scrag.container import build_container
from scrag.demo import ClientQuota
from scrag.errors import ClientQuotaExceededError, LLMError
from scrag.loaders import LoaderError, check_docx_container, extract_upload, sniff_upload
from scrag.observability import metrics
from scrag.orchestration.tools import safe_eval
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder
from .test_loaders import _minimal_pdf


def _client(settings, sample_chunks, chat=None, **overrides):
    for key, value in overrides.items():
        setattr(settings, key, value)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    chat = chat or FakeChatClient(guardrail='{"scores": [0.9, 0.9, 0.9]}')
    container = build_container(settings, embedder=embedder, store=store, chat_client=chat)
    return TestClient(create_app(settings, container=container)), container


def _docx(parts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return buffer.getvalue()


_DOCX_PARTS = {"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document/>"}


# --- uploads: content sniffing and parser robustness ---------------------------------------
def test_upload_bytes_must_match_their_extension():
    with pytest.raises(LoaderError, match="not a PDF"):
        sniff_upload(b"just some text", ".pdf")
    with pytest.raises(LoaderError, match="not a ZIP"):
        sniff_upload(b"just some text", ".docx")
    with pytest.raises(LoaderError, match="PDF or ZIP file named .txt"):
        sniff_upload(b"%PDF-1.7 ...", ".txt")
    sniff_upload(_minimal_pdf("hello"), ".pdf")  # a real PDF passes
    sniff_upload(b"junk before the header %PDF-1.4", ".pdf")  # allowed in the first 1024 bytes
    sniff_upload(_docx(_DOCX_PARTS), ".docx")


def test_binary_content_in_a_text_upload_is_refused():
    with pytest.raises(LoaderError, match="binary content"):
        extract_upload(b"valid utf-8 \x00 with a NUL", "notes.txt", max_bytes=1_000)


def test_deeply_nested_json_and_huge_csv_fields_are_loader_errors():
    deep = b"[" * 100_000 + b"]" * 100_000
    with pytest.raises(LoaderError, match="nested too deeply"):
        extract_upload(deep, "deep.json", max_bytes=1_000_000)
    big_field = b"a,b\n" + b"x" * 200_000 + b",1\n"
    with pytest.raises(LoaderError, match="invalid CSV"):
        extract_upload(big_field, "big.csv", max_bytes=1_000_000)


def test_docx_zip_bombs_are_refused_before_inflating(monkeypatch):
    with pytest.raises(LoaderError, match="no word/document.xml"):
        check_docx_container(_docx({"[Content_Types].xml": b"<Types/>"}))
    with pytest.raises(LoaderError, match="corrupt ZIP"):
        check_docx_container(b"PK\x03\x04 truncated")
    # 2 MB of zeros deflates about 1000:1: a bomb's ratio, refused from the directory.
    bomb = _docx({**_DOCX_PARTS, "word/media/blob.bin": b"\x00" * 2_000_000})
    with pytest.raises(LoaderError, match="compression ratio"):
        check_docx_container(bomb)
    monkeypatch.setattr(loaders, "DOCX_MAX_UNCOMPRESSED_BYTES", 1_000)
    with pytest.raises(LoaderError, match="inflates to"):
        check_docx_container(_docx({**_DOCX_PARTS, "word/styles.xml": b"<s/>" * 500}))
    monkeypatch.setattr(loaders, "DOCX_MAX_ENTRIES", 3)
    many = {**_DOCX_PARTS, **{f"word/p{i}.xml": b"<p/>" for i in range(5)}}
    with pytest.raises(LoaderError, match="parts"):
        check_docx_container(_docx(many))


def test_real_docx_still_loads():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Hybrid retrieval fuses BM25 and dense scores.")
    buffer = io.BytesIO()
    document.save(buffer)
    text = extract_upload(buffer.getvalue(), "notes.docx", max_bytes=1_000_000)
    assert "BM25" in text


# --- calculator ---------------------------------------------------------------------------
def test_calculator_refuses_round_with_huge_negative_digits():
    start = time.perf_counter()
    with pytest.raises(ValueError, match="round"):
        safe_eval("round(5, -99999999999)")  # would compute 10**99999999999
    assert time.perf_counter() - start < 0.5
    assert safe_eval("round(3.14159, 2)") == 3.14
    assert safe_eval("round(1234, -2)") == 1200


# --- request limits and headers -----------------------------------------------------------
def test_oversized_json_bodies_are_refused_before_parsing(settings, sample_chunks):
    client, _ = _client(settings, sample_chunks, max_request_bytes=2_048)
    with client:
        ok = client.post("/api/v1/search", json={"query": "rag"})
        declared = client.post("/api/v1/search", json={"query": "x" * 5_000})

        def chunks():  # no Content-Length: counted while it streams
            for _ in range(10):
                yield b" " * 1_000

        streamed = client.post(
            "/api/v1/search", content=chunks(), headers={"content-type": "application/json"}
        )
    assert ok.status_code == 200
    for response in (declared, streamed):
        assert response.status_code == 413, response.text
        assert response.json()["error"] == "payload_too_large"
        assert response.json()["request_id"] == response.headers["x-request-id"]


def test_security_headers_on_api_responses(settings, sample_chunks):
    client, _ = _client(settings, sample_chunks)
    with client:
        health = client.get("/api/v1/health")
        missing = client.get("/api/v1/no-such-route")
        docs = client.get("/docs")
        stream = client.post("/api/v1/query/stream", json={"query": "what is rag"})
    for response in (health, missing):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-security-policy"].startswith("default-src 'none'")
    # The HTML docs load Swagger UI from a CDN, so they don't get the deny-all CSP.
    assert docs.status_code == 200 and "content-security-policy" not in docs.headers
    # A route's own Cache-Control wins (SSE must not be transformed by proxies).
    assert stream.headers["cache-control"] == "no-cache, no-transform"


def test_secrets_never_appear_in_the_settings_repr():
    secrets = ["sk-live-111", "operator-222", "jwt-secret-333", "pg-password-444"]
    settings = Settings(
        _env_file=None,
        llm_api_key=secrets[0],
        api_auth_token=secrets[1],
        auth_jwt_secret=secrets[2],
        database_url=f"postgresql://scrag:{secrets[3]}@db.example/scrag",
    )
    for text in (repr(settings), str(settings)):
        assert not any(secret in text for secret in secrets)


def test_flood_guard_refuses_a_flooding_client_before_routing(settings, sample_chunks):
    client, _ = _client(
        settings,
        sample_chunks,
        demo_mode=True,
        demo_flood_burst=5,
        demo_flood_limit_per_minute=1,
        forwarded_ip_header="x-forwarded-for",
    )
    flood = {"x-forwarded-for": "203.0.113.66"}
    labels = {"method": "GET", "route": "flood_guard", "status": "429"}
    before = metrics.sample("scrag_http_requests_total", labels)
    with client:
        statuses = [client.get("/api/v1/demo", headers=flood).status_code for _ in range(8)]
        refused = client.get("/api/v1/demo", headers=flood)
        health = client.get("/api/v1/health", headers=flood)
        neighbour = client.get("/api/v1/demo", headers={"x-forwarded-for": "198.51.100.1"})
    assert statuses == [200] * 5 + [429] * 3
    body = refused.json()
    assert body["error"] == "rate_limited" and body["request_id"] == refused.headers["x-request-id"]
    assert int(refused.headers["retry-after"]) > 0
    assert refused.headers["x-content-type-options"] == "nosniff"
    assert health.status_code == 200  # liveness probes are never throttled
    assert neighbour.status_code == 200  # other clients are unaffected
    assert metrics.sample("scrag_http_requests_total", labels) - before == 4


def test_client_address_ignores_spoofed_forwarded_hops():
    # The client wrote "6.6.6.6"; Render's proxy appended the real address.
    scope = {"headers": [(b"x-forwarded-for", b"6.6.6.6, 203.0.113.9")], "client": ("10.0.0.1", 1)}
    assert client_address(scope, "x-forwarded-for", 1) == "203.0.113.9"
    assert client_address(scope, "X-Forwarded-For", 2) == "6.6.6.6"  # two trusted proxies
    assert client_address(scope, None, 1) == "10.0.0.1"  # no proxy configured: the peer
    assert client_address({"headers": [], "client": ("10.0.0.1", 1)}, "x-forwarded-for", 1) == (
        "10.0.0.1"
    )


def test_flood_guard_is_off_outside_demo_mode(settings, sample_chunks):
    client, _ = _client(settings, sample_chunks, demo_flood_burst=1, demo_flood_limit_per_minute=1)
    with client:
        assert {client.get("/api/v1/demo").status_code for _ in range(5)} == {200}


# --- per-client quota ---------------------------------------------------------------------
def test_client_quota_is_per_client_refundable_and_daily():
    day = [datetime(2026, 11, 10, 23, 30, tzinfo=timezone.utc)]
    quota = ClientQuota(2, now=lambda: day[0])
    quota.consume("a")
    quota.consume("a")
    with pytest.raises(ClientQuotaExceededError) as info:
        quota.consume("a")
    assert int(info.value.headers["Retry-After"]) == 1800  # until midnight UTC
    quota.consume("b", 2)  # another client has its own share
    quota.refund("a")
    assert quota.remaining("a") == 1
    day[0] += timedelta(hours=1)
    assert quota.remaining("a") == 2 and quota.remaining("b") == 2


def test_client_quota_memory_is_bounded():
    quota = ClientQuota(5, max_clients=4)
    for client in "abcd":
        quota.consume(client, 1 if client != "d" else 3)
    quota.consume("e")  # over the bound: the lightest half is forgotten
    assert len(quota._used) <= 4
    assert quota.remaining("d") == 2  # the heaviest user is still tracked


def test_demo_charges_live_runs_to_each_clients_share(settings, sample_chunks):
    chat = FakeChatClient(
        guardrail='{"scores": [0.9, 0.9, 0.9]}',
        generator=[LLMError("down"), "An answer.", "Another answer."],
    )
    client, container = _client(
        settings,
        sample_chunks,
        chat=chat,
        demo_mode=True,
        demo_client_daily_runs=1,
        demo_daily_query_budget=10,
        demo_rate_limit_burst=20,
        forwarded_ip_header="x-forwarded-for",
    )
    other = {"x-forwarded-for": "203.0.113.7"}
    with client:
        failed = client.post("/api/v1/query", json={"query": "first question"})
        answered = client.post("/api/v1/query", json={"query": "second question"})
        over = client.post("/api/v1/query", json={"query": "third question"})
        agent = client.post("/api/v1/agent/run", json={"task": "a task"}, headers=other)
        elsewhere = client.post("/api/v1/query", json={"query": "third question"}, headers=other)
        demo = client.get("/api/v1/demo").json()
    assert failed.status_code == 502  # no answer: refunded to both quotas
    assert answered.status_code == 200
    assert over.status_code == 429 and over.json()["error"] == "client_quota_exceeded"
    assert int(over.headers["Retry-After"]) > 0
    # An agent run costs agent_budget_cost (2) units: more than one client's share.
    assert agent.status_code == 429 and agent.json()["error"] == "client_quota_exceeded"
    assert elsewhere.status_code == 200  # another visitor still has theirs
    # Only the two answered runs were charged to the global budget.
    assert container.budget.status()["remaining"] == 8
    assert demo["budget"]["client_remaining"] == 0
    assert demo["limits"]["client_daily_runs"] == 1
