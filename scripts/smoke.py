"""Smoke-test a running SCRAG API (local container or the live deployment).

Usage:

    python scripts/smoke.py --base-url http://localhost:8000
    python scripts/smoke.py --base-url https://scrag-api.onrender.com --live-query

Waits for a sleeping free-tier instance to wake up (up to --wake-timeout seconds),
then checks health, the demo info, retrieval, validation, the error contract and
the Phase 7 hardening (security headers, the body cap, each visitor's reported
share of the budget, the read-only SQL tool).
``--live-query`` additionally runs one sample question through /query (a
recorded sample answer is fine; it must not be an error). ``--flood-check``
(demo mode) sends a burst of up to 150 requests and checks that the flood guard
refuses it while health stays open; it runs last, because it leaves this client
rate limited for about half a minute. Exit code 0 = all passed.
Stdlib only, so it runs anywhere without installing the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Per request. A cold 0.1-CPU instance loads its model on the first search.
OPTIONS = {"timeout_s": 90.0}


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, Any, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"content-type": "application/json", "user-agent": "scrag-smoke/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=OPTIONS["timeout_s"]) as resp:
            return resp.status, json.loads(resp.read() or b"null"), dict(resp.headers)
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw.decode(errors="replace")
        return err.code, payload, dict(err.headers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--wake-timeout", type=float, default=180.0)
    parser.add_argument("--live-query", action="store_true")
    parser.add_argument("--flood-check", action="store_true", help="demo mode only; runs last")
    parser.add_argument(
        "--timeout", type=float, default=OPTIONS["timeout_s"], help="seconds per request"
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    OPTIONS["timeout_s"] = args.timeout
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if not ok else ""))
        if not ok:
            failures.append(name)

    deadline = time.monotonic() + args.wake_timeout
    status: int = 0
    health: Any = None
    headers: dict[str, str] = {}
    last_error: object = None
    while time.monotonic() < deadline:
        try:
            status, health, headers = call(base, "GET", "/api/v1/health")
            if status == 200:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc  # still waking up; retry until the deadline
        time.sleep(5)
    check("health is 200 (after wake-up)", status == 200, last_error or status)
    if status != 200:
        return 1
    check("bundled index is loaded", health.get("index_size", 0) > 0, health)
    check("X-Request-ID header present", any(k.lower() == "x-request-id" for k in headers))
    lower = {k.lower(): v for k, v in headers.items()}
    check(
        "security headers on API responses",
        lower.get("x-content-type-options") == "nosniff"
        and lower.get("cache-control") == "no-store"
        and lower.get("content-security-policy", "").startswith("default-src 'none'"),
        lower,
    )

    status, demo, _ = call(base, "GET", "/api/v1/demo")
    check("demo info is 200", status == 200, status)
    samples = (demo or {}).get("sample_questions", [])
    if health.get("demo_mode"):
        budget = (demo or {}).get("budget") or {}
        limits = (demo or {}).get("limits") or {}
        check(
            "each visitor's daily share is reported",
            "client_remaining" in budget and "client_daily_runs" in limits,
            {"budget": budget, "limits": limits},
        )

    status, tools, _ = call(base, "GET", "/api/v1/agent/tools")
    names = {t.get("name"): t for t in tools} if isinstance(tools, list) else {}
    check(
        "agent tools include read-only SQL",
        status == 200 and names.get("sql_query", {}).get("enabled") is True,
        sorted(names),
    )

    status, body, _ = call(base, "POST", "/api/v1/search", {"query": "x" * 70_000})
    check(
        "oversized request body is 413",
        status == 413 and isinstance(body, dict) and body.get("error") == "payload_too_large",
        status,
    )

    status, found, _ = call(base, "POST", "/api/v1/search", {"query": "hybrid search RRF"})
    check("search is 200", status == 200, found)
    check("search returns chunks", bool((found or {}).get("chunks")), found)

    status, body, _ = call(base, "POST", "/api/v1/query", {"query": ""})
    check("empty query is 422", status == 422, status)
    status, body, _ = call(base, "POST", "/api/v1/query", {"query": "   "})
    if health.get("llm_configured"):
        check(
            "blank query is 400 invalid_query",
            status == 400 and body.get("error") == "invalid_query",
            body,
        )
    else:
        check("query without LLM is 503", status == 503, body)

    # Only on the demo, where it must be refused: elsewhere this request would
    # really re-ingest the corpus.
    if health.get("demo_mode"):
        status, body, _ = call(base, "POST", "/api/v1/ingest", {"reset": True})
        check("anonymous ingest is refused on the demo", status in (401, 403), status)

    if args.live_query:
        question = samples[0] if samples else "What triggers self-correction?"
        status, body, _ = call(base, "POST", "/api/v1/query", {"query": question})
        check("sample query is 200", status == 200, body)
        if status == 200:
            check("answer is non-empty", bool(body.get("answer")), body)
            check("answer cites chunks", bool(body.get("used_chunks")), body)
            print(
                f"      status={body.get('evaluation_status')} cached={body.get('cached')} "
                f"attempts={body.get('attempts')}"
            )

    if args.flood_check and health.get("demo_mode"):
        refused: tuple[int, Any] | None = None
        for i in range(150):
            status, body, _ = call(base, "GET", "/api/v1/demo")
            if status == 429:
                refused = (i + 1, body)
                break
        check(
            "flood guard refuses a burst",
            refused is not None
            and isinstance(refused[1], dict)
            and refused[1].get("error") == "rate_limited",
            refused,
        )
        if refused is not None:
            print(f"      refused after {refused[0]} rapid requests")
        status, _, _ = call(base, "GET", "/api/v1/health")
        check("health stays open during a flood", status == 200, status)

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
