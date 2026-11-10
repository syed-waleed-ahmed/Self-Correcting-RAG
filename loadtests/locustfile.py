"""Load profiles for the SCRAG API (method and results: docs/load-testing.md).

Run against a local server or container whose LLM points at
``loadtests/mock_llm.py`` (never a real provider). User classes:

* ``Visitor``: a busy visitor (1-3 s between actions): health and demo info,
  retrieval-only search, recorded sample answers, and occasional live
  questions that run the full pipeline. The capacity profile, against a server
  with demo mode off (no rate limits).
* ``DemoVisitor``: the same mix at a reading pace (10-20 s), each user from its
  own client address (``X-Forwarded-For``), against a demo-mode server.
* ``Flooder``: one abusive client (a single address) sending requests as fast
  as it can over ``FLOOD_CONNECTIONS`` concurrent connections (default 1, and
  always exactly that many users, whatever ``-u`` says), to ``/search`` (an
  endpoint with its own rate limit) or, with ``FLOOD_TARGET=demo``, to
  ``GET /demo`` (one without). With ``DemoVisitor``, it shows whether that
  client is contained and how much everyone else slows down.

Examples::

    locust -f loadtests/locustfile.py --headless -u 20 -r 5 -t 2m \\
        --host http://127.0.0.1:8000 --csv out/capacity Visitor
    FLOOD_CONNECTIONS=8 locust -f loadtests/locustfile.py --headless -u 38 -r 38 -t 3m \\
        --host http://127.0.0.1:8000 --csv out/abuse DemoVisitor Flooder
"""

from __future__ import annotations

import itertools
import os
import random

from locust import HttpUser, between, constant, task

SEARCHES = [
    "hybrid search reciprocal rank fusion",
    "what does the guardrail agent check",
    "BM25 parameter k1",
    "prompt injection mitigation",
    "sanitized error response request id",
    "embedding model memory on small instances",
    "self-correction evaluator threshold",
]
_live = itertools.count(1)
_addresses = itertools.count(1)


class Visitor(HttpUser):
    wait_time = between(1, 3)
    headers: dict[str, str] = {}

    def on_start(self) -> None:
        demo = self.client.get("/api/v1/demo", name="/demo", headers=self.headers)
        questions = demo.json().get("sample_questions") if demo.ok else None
        self.samples = questions or ["What triggers self-correction?"]

    @task(1)
    def health(self) -> None:
        self.client.get("/api/v1/health", name="/health", headers=self.headers)

    @task(2)
    def demo_info(self) -> None:
        self.client.get("/api/v1/demo", name="/demo", headers=self.headers)

    @task(4)
    def search(self) -> None:
        self.client.post(
            "/api/v1/search",
            json={"query": random.choice(SEARCHES)},
            name="/search",
            headers=self.headers,
        )

    @task(4)
    def sample_answer(self) -> None:
        self.client.post(
            "/api/v1/query",
            json={"query": random.choice(self.samples)},
            name="/query (recorded sample)",
            headers=self.headers,
        )

    @task(1)
    def live_answer(self) -> None:
        # A unique question: misses every cache, so the full pipeline runs.
        question = f"{random.choice(SEARCHES)}? (load test {next(_live)})"
        self.client.post(
            "/api/v1/query", json={"query": question}, name="/query (live)", headers=self.headers
        )


class DemoVisitor(Visitor):
    wait_time = between(10, 20)

    def on_start(self) -> None:
        n = next(_addresses)
        self.headers = {"x-forwarded-for": f"198.51.100.{n % 250 + 1}"}  # TEST-NET-2
        super().on_start()


class Flooder(HttpUser):
    fixed_count = int(os.environ.get("FLOOD_CONNECTIONS", "1"))
    # "search": an endpoint with its own rate limit; "demo": one without (GET /demo).
    target = os.environ.get("FLOOD_TARGET", "search")
    wait_time = constant(0)
    headers = {"x-forwarded-for": "203.0.113.66"}  # TEST-NET-3: one abusive client

    @task
    def flood(self) -> None:
        if self.target == "demo":
            self.client.get("/api/v1/demo", name="/demo (flood)", headers=self.headers)
        else:
            self.client.post(
                "/api/v1/search",
                json={"query": "flood"},
                name="/search (flood)",
                headers=self.headers,
            )
