"""Smoke-test a running SCRAG API (local container or the live deployment).

Usage:

    python scripts/smoke.py --base-url http://localhost:8000
    python scripts/smoke.py --base-url https://scrag-api.onrender.com --live-query

Waits for a sleeping free-tier instance to wake up (up to --wake-timeout seconds),
then checks health, the demo info, retrieval, validation and the error contract.
``--live-query`` additionally runs one sample question through /query (a
recorded sample answer is fine; it must not be an error). Exit code 0 = all passed.
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


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, Any, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"content-type": "application/json", "user-agent": "scrag-smoke/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
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
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if not ok else ""))
        if not ok:
            failures.append(name)

    deadline = time.monotonic() + args.wake_timeout
    status: int = 0
    health: Any = None
    headers: dict[str, str] = {}
    while time.monotonic() < deadline:
        try:
            status, health, headers = call(base, "GET", "/api/v1/health")
            if status == 200:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(5)
    check("health is 200 (after wake-up)", status == 200, status)
    if status != 200:
        return 1
    check("bundled index is loaded", health.get("index_size", 0) > 0, health)
    check("X-Request-ID header present", any(k.lower() == "x-request-id" for k in headers))

    status, demo, _ = call(base, "GET", "/api/v1/demo")
    check("demo info is 200", status == 200, status)
    samples = (demo or {}).get("sample_questions", [])

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

    status, body, _ = call(base, "POST", "/api/v1/ingest", {"reset": True})
    if health.get("demo_mode"):
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

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
