"""Phase 7 agent tools: read-only SQL (sql_query) and SSRF-safe fetching (web_fetch)."""

from __future__ import annotations

import json
import socket
import ssl

import pytest
from fastapi.testclient import TestClient

from scrag.api import create_app
from scrag.container import build_container
from scrag.models import Chunk
from scrag.orchestration import CorpusCatalog, SqlRejected, WebFetcher, web_fetch_tool
from scrag.orchestration.web_fetch import (
    FetchBlocked,
    RawResponse,
    _PinnedHTTPSConnection,
    host_allowed,
    is_public_address,
)
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder

PASS = '{"score": 0.9, "explanation": "supported"}'


def _chunks(n_docs: int = 3, per_doc: int = 2) -> list[Chunk]:
    return [
        Chunk(
            id=f"d{d}-{p}",
            source=f"doc{d}.md",
            text=f"document {d} part {p} " + "bm25 " * (d + 1),
            position=p,
        )
        for d in range(n_docs)
        for p in range(per_doc)
    ]


def _catalog(chunks: list[Chunk] | None = None, **kwargs) -> CorpusCatalog:
    data = chunks if chunks is not None else _chunks()
    return CorpusCatalog(lambda: (1, data), **kwargs)


# --- sql_query ----------------------------------------------------------------------------
def test_sql_answers_questions_about_the_corpus():
    catalog = _catalog()
    docs = catalog.query("SELECT source, chunks FROM documents ORDER BY characters DESC")
    assert docs["columns"] == ["source", "chunks"]
    assert docs["rows"][0] == ["doc2.md", 2]  # the most 'bm25' text
    hits = catalog.query("select count(*) as n from chunks where lower(text) like '%bm25%';")
    assert hits["rows"] == [[6]] and hits["truncated"] is False
    cte = catalog.query(
        "WITH big AS (SELECT source FROM documents WHERE characters > 50) SELECT * FROM big"
    )
    assert cte["row_count"] >= 1


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM chunks",
        "WITH x AS (SELECT 1) DELETE FROM chunks",  # passes the prefix check: authorizer refuses
        "WITH x AS (SELECT 1) INSERT INTO documents VALUES ('x', 1, 1)",
        "SELECT * FROM sqlite_master",
        "SELECT 1; DROP TABLE chunks",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA table_info(chunks)",
        "SELECT load_extension('evil')",
        "SELECT randomblob(100000000)",
        "SELECT zeroblob(100000000)",
        "SELECT printf('%.*c', 100000000, 'x')",
        "SELECT group_concat(text) FROM chunks",
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r",
        "SELECT * FROM nonexistent",
    ],
)
def test_sql_refuses_anything_but_reading_the_catalog(sql):
    catalog = _catalog()
    with pytest.raises(SqlRejected):
        catalog.query(sql)
    # ...and the catalog is untouched afterwards.
    assert catalog.query("SELECT count(*) FROM chunks")["rows"] == [[6]]


def test_sql_time_budget_stops_runaway_joins():
    catalog = _catalog(_chunks(n_docs=30, per_doc=10), time_budget_s=0.05)
    with pytest.raises(SqlRejected, match="exceeded"):
        catalog.query("SELECT count(*) FROM chunks a, chunks b, chunks c, chunks d")


def test_sql_caps_rows_and_values():
    long_text = [Chunk(id="long", source="long.md", text="y" * 5_000, position=0)]
    catalog = _catalog(_chunks() + long_text, max_rows=2, max_value_chars=10)
    rows = catalog.query("SELECT id FROM chunks")
    assert rows["row_count"] == 2 and rows["truncated"] is True
    value = catalog.query("SELECT text FROM chunks WHERE id = 'long'")["rows"][0][0]
    assert value == "y" * 10 + "..."


def test_sql_catalog_follows_the_index_revision():
    state = {"revision": 1, "chunks": _chunks(n_docs=1)}
    catalog = CorpusCatalog(lambda: (state["revision"], state["chunks"]))
    assert catalog.query("SELECT count(*) FROM documents")["rows"] == [[1]]
    state["chunks"] = _chunks(n_docs=4)
    assert catalog.query("SELECT count(*) FROM documents")["rows"] == [[1]]  # same revision
    state["revision"] = 2
    assert catalog.query("SELECT count(*) FROM documents")["rows"] == [[4]]


def test_agent_uses_sql_over_the_sample_catalog(settings, sample_chunks):
    plan = json.dumps(
        {
            "steps": [
                {
                    "tool": "sql_query",
                    "input": {"sql": "SELECT source, chunks FROM documents ORDER BY source"},
                }
            ]
        }
    )
    chat = FakeChatClient(generator=[plan, "Three documents [S1]."], evaluator=PASS)
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    container = build_container(settings, embedder=embedder, store=store, chat_client=chat)
    with TestClient(create_app(settings, container=container)) as client:
        body = client.post("/api/v1/agent/run", json={"task": "How many documents?"}).json()
    assert body["status"] == "completed"
    output = body["steps"][0]["call"]["output"]
    assert output["rows"] == [["doc0.txt", 1], ["doc1.txt", 1], ["doc2.txt", 1]]
    writer_prompt = chat.prompts_for("generator")[-1]
    assert "tool:sql_query" in writer_prompt  # the rows reached the writer as evidence


# --- web_fetch: address and URL checks ----------------------------------------------------
@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.215.14", True),
        ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.64.0.1", False),  # CGNAT
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("::1", False),
        ("fd00::1", False),
        ("fe80::1%eth0", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped
        ("2002:7f00:1::1", False),  # 6to4 wrapping 127.0.0.1
        ("64:ff9b::7f00:1", False),  # NAT64 wrapping 127.0.0.1 (is_global says True)
        ("2001:0:4136:e378:8000:63bf:3fff:fdd2", False),  # Teredo
    ],
)
def test_only_public_unicast_addresses_are_allowed(address, public):
    assert is_public_address(address) is public


def test_host_allow_list_matching():
    allowed = ["docs.example.com", "*.wiki.example.org"]
    assert host_allowed("docs.example.com", allowed)
    assert host_allowed("en.wiki.example.org", allowed)
    assert not host_allowed("wiki.example.org", allowed)  # wildcard is subdomains only
    assert not host_allowed("evil-docs.example.com", allowed)
    assert not host_allowed("docs.example.com.evil.net", allowed)
    assert not host_allowed("evilwiki.example.org", allowed)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://docs.example.com/", "only https"),
        ("file:///etc/passwd", "only https"),
        ("https://user:pw@docs.example.com/", "credentials"),
        ("https://127.0.0.1/", "IP-address"),
        ("https://[::1]/", "IP-address"),
        ("https://docs.example.com:8443/", "port"),
        ("https://other.example.com/", "allow-list"),
        ("https://docs.example.com@evil.net/", "credentials"),
        ("https://docs.example.com/a b", "whitespace"),
        ("https://docs.example.com/\r\nHost: evil", "whitespace"),
        ("https://docs.example.com/" + "a" * 2100, "too long"),
    ],
)
def test_urls_are_checked_before_any_lookup(url, reason):
    def no_lookup(host, port):  # pragma: no cover - must not be reached
        raise AssertionError("resolved a refused URL")

    fetcher = WebFetcher(["docs.example.com"], resolver=no_lookup)
    with pytest.raises(FetchBlocked, match=reason):
        fetcher.fetch(url)


# --- web_fetch: fetching with a fake network ----------------------------------------------
class _Net:
    """Fake DNS + transport. ``pages`` maps (host, target) to a RawResponse."""

    def __init__(self, dns: dict[str, list[str]], pages: dict[tuple[str, str], RawResponse]):
        self.dns = dns
        self.pages = pages
        self.connections: list[tuple[str, str, str]] = []

    def resolve(self, host: str, port: int) -> list[str]:
        if host not in self.dns:
            raise socket.gaierror("unknown host")
        return self.dns[host]

    def transport(self, address, host, port, target, timeout, max_bytes) -> RawResponse:
        self.connections.append((address, host, target))
        return self.pages[(host, target)]


def _html(body: str, **headers: str) -> RawResponse:
    return RawResponse(200, {"content-type": "text/html; charset=utf-8", **headers}, body.encode())


def _fetcher(net: _Net, hosts=("docs.example.com", "cdn.example.com"), **kwargs) -> WebFetcher:
    return WebFetcher(list(hosts), resolver=net.resolve, transport=net.transport, **kwargs)


def test_fetch_extracts_text_and_connects_to_the_checked_address():
    page = "<html><script>steal()</script><h1>Hybrid search</h1><p>BM25 + vectors</p></html>"
    net = _Net(
        {"docs.example.com": ["93.184.215.14"]}, {("docs.example.com", "/a?b=1"): _html(page)}
    )
    result = _fetcher(net).fetch("https://docs.example.com/a?b=1#frag")
    assert result["text"] == "Hybrid search\nBM25 + vectors"
    assert result["content_type"] == "text/html" and result["truncated"] is False
    assert net.connections == [("93.184.215.14", "docs.example.com", "/a?b=1")]


@pytest.mark.parametrize(
    "addresses",
    [["10.0.0.5"], ["93.184.215.14", "127.0.0.1"], ["169.254.169.254"], ["::ffff:10.0.0.1"]],
)
def test_fetch_refuses_hosts_that_resolve_internally(addresses):
    net = _Net({"docs.example.com": addresses}, {})
    with pytest.raises(FetchBlocked, match="non-public"):
        _fetcher(net).fetch("https://docs.example.com/")
    assert net.connections == []  # never connected


def test_redirects_are_rechecked_at_every_hop():
    hop = RawResponse(302, {"location": "https://cdn.example.com/page"}, b"")
    net = _Net(
        {"docs.example.com": ["93.184.215.14"], "cdn.example.com": ["93.184.215.15"]},
        {
            ("docs.example.com", "/start"): hop,
            ("cdn.example.com", "/page"): _html("<p>moved here</p>"),
            ("docs.example.com", "/to-internal"): RawResponse(
                301, {"location": "https://intranet.corp/"}, b""
            ),
            ("docs.example.com", "/to-http"): RawResponse(
                307, {"location": "http://docs.example.com/"}, b""
            ),
            ("docs.example.com", "/loop"): RawResponse(302, {"location": "/loop"}, b""),
        },
    )
    fetcher = _fetcher(net, max_redirects=3)
    assert fetcher.fetch("https://docs.example.com/start")["text"] == "moved here"
    with pytest.raises(FetchBlocked, match="allow-list"):
        fetcher.fetch("https://docs.example.com/to-internal")
    with pytest.raises(FetchBlocked, match="only https"):
        fetcher.fetch("https://docs.example.com/to-http")
    with pytest.raises(FetchBlocked, match="redirects"):
        fetcher.fetch("https://docs.example.com/loop")


def test_only_uncompressed_text_responses_are_read():
    def page(status=200, **headers):
        return RawResponse(status, {"content-type": "text/plain", **headers}, "café".encode())

    net = _Net(
        {"docs.example.com": ["93.184.215.14"]},
        {
            ("docs.example.com", "/png"): RawResponse(
                200, {"content-type": "image/png"}, b"\x89PNG"
            ),
            ("docs.example.com", "/gz"): page(**{"content-encoding": "gzip"}),
            ("docs.example.com", "/404"): page(404),
            ("docs.example.com", "/latin1"): RawResponse(
                200, {"content-type": "text/plain; charset=iso-8859-1"}, "café".encode("latin-1")
            ),
            ("docs.example.com", "/big"): RawResponse(
                200, {"content-type": "text/plain"}, b"x" * 50, truncated=True
            ),
        },
    )
    fetcher = _fetcher(net, max_chars=20)
    with pytest.raises(FetchBlocked, match="content type"):
        fetcher.fetch("https://docs.example.com/png")
    with pytest.raises(FetchBlocked, match="compressed"):
        fetcher.fetch("https://docs.example.com/gz")
    with pytest.raises(FetchBlocked, match="HTTP 404"):
        fetcher.fetch("https://docs.example.com/404")
    assert fetcher.fetch("https://docs.example.com/latin1")["text"] == "café"
    big = fetcher.fetch("https://docs.example.com/big")
    assert big["truncated"] is True and len(big["text"]) == 20


def test_unresolvable_hosts_and_transport_errors_are_reported():
    net = _Net({}, {})
    with pytest.raises(FetchBlocked, match="could not resolve"):
        _fetcher(net).fetch("https://docs.example.com/")

    def failing(*args):
        raise ConnectionResetError("reset")

    fetcher = WebFetcher(
        ["docs.example.com"], resolver=lambda h, p: ["93.184.215.14"], transport=failing
    )
    with pytest.raises(FetchBlocked, match="ConnectionResetError"):
        fetcher.fetch("https://docs.example.com/")


def test_pinned_connection_uses_the_checked_address_and_verifies_the_hostname(monkeypatch):
    seen = {}

    class _Socket:
        pass

    class _Context(ssl.SSLContext):  # real verification settings, fake handshake
        def wrap_socket(self, sock, server_hostname=None, **kwargs):
            seen["sni"] = server_hostname
            return sock

    def create_connection(address, timeout):
        seen["address"] = address
        return _Socket()

    monkeypatch.setattr(socket, "create_connection", create_connection)
    conn = _PinnedHTTPSConnection(
        "docs.example.com",
        443,
        address="93.184.215.14",
        timeout=1.0,
        context=_Context(ssl.PROTOCOL_TLS_CLIENT),
    )
    conn.connect()
    assert seen == {"address": ("93.184.215.14", 443), "sni": "docs.example.com"}


def test_web_fetch_tool_is_disabled_without_an_allow_list(settings, sample_chunks):
    assert web_fetch_tool(None).enabled is False
    blocked = web_fetch_tool(WebFetcher(["docs.example.com"], resolver=lambda h, p: ["10.0.0.1"]))
    output = blocked.run(blocked.input_model(url="https://docs.example.com/"))
    assert "non-public" in output["error"]

    settings.http_tool_allowed_hosts = ["docs.example.com"]
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    container = build_container(
        settings, embedder=embedder, store=store, chat_client=FakeChatClient()
    )
    with TestClient(create_app(settings, container=container)) as client:
        tools = {t["name"]: t for t in client.get("/api/v1/agent/tools").json()}
    assert tools["web_fetch"]["enabled"] is True
    assert tools["web_fetch"]["description"] == (
        "Fetch the text of one web page (read-only GET) from an allow-listed host: "
        "docs.example.com."
    )
