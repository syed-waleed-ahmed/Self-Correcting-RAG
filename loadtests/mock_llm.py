"""A stand-in for an OpenAI-compatible LLM provider, for load tests.

Load tests must not spend a real provider's quota (or measure its rate limits
instead of this server), so they point ``SCRAG_LLM_BASE_URL`` here. Every chat
completion waits ``--latency-ms`` (a typical hosted-model response time) and
answers in the shape each agent expects:

* batched guardrail: ``{"scores": [...]}`` with one score per numbered chunk;
* per-chunk guardrail: a bare number;
* evaluator: ``{"score": ..., "explanation": ...}``;
* planner: a one-step ``document_search`` plan;
* generator and writer: a short answer citing ``[S1]``.

Usage::

    python loadtests/mock_llm.py --port 8090 --latency-ms 300

Stdlib only. Not a security boundary: bind it to localhost.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_CHUNK_MARKER = re.compile(r"^\[(\d+)\]$", re.MULTILINE)


def reply_for(system: str, user: str) -> str:
    """The completion an agent with this system prompt can parse."""
    low = system.lower()
    if '{"scores"' in system:
        count = len(_CHUNK_MARKER.findall(user)) or 1
        return json.dumps({"scores": [0.9] * count})
    if "relevance guardrail" in low:
        return "0.9"
    if "evaluator" in low or "'score'" in system:
        return json.dumps({"score": 0.9, "explanation": "Supported by the context (mock)."})
    if "planning agent" in low:
        task = user.split("\n", 1)[-1][:200]
        return json.dumps(
            {"steps": [{"tool": "document_search", "input": {"query": task}, "purpose": "mock"}]}
        )
    return "A mock answer based on the retrieved context [S1]."


class _Handler(BaseHTTPRequestHandler):
    latency_s = 0.3
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("content-length") or 0)
        request: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        messages = request.get("messages") or [{}, {}]
        system = str(messages[0].get("content", ""))
        user = str(messages[-1].get("content", ""))
        time.sleep(self.latency_s)
        content = reply_for(system, user)
        prompt_tokens = (len(system) + len(user)) // 4
        completion_tokens = max(1, len(content) // 4)
        self._send(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "mock"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--latency-ms", type=float, default=300.0)
    args = parser.parse_args()
    _Handler.latency_s = args.latency_ms / 1000
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True
    print(f"mock LLM on http://{args.host}:{args.port}/v1 ({args.latency_ms:g} ms per call)")
    server.serve_forever()


if __name__ == "__main__":
    main()
