"""Server-Sent Events for long-running requests (D8).

The pipeline and the agent are synchronous and take seconds, so a streaming
endpoint runs the work on a thread and relays its progress through a queue:

* ``event: progress``: one per stage (pipeline) or trace event (agent);
* ``event: result``: the final response body, exactly as the non-streaming
  endpoint would return it;
* ``event: error``: ``{"error", "detail", "request_id", "status"}``, the same
  sanitized contract as an HTTP error response (plus the status it would have).

Checks that can fail fast (validation, rate limit, session) run before the
stream starts, so they still come back as ordinary HTTP errors. A comment line
is sent every 15 s while the work is quiet, so proxies keep long runs open.
That is traffic inside an active request, not keep-alive.
"""

from __future__ import annotations

import contextvars
import json
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from fastapi.responses import StreamingResponse

from ..errors import ScragError
from ..logging_config import get_logger, request_id_var

logger = get_logger(__name__)

Emit = Callable[[dict[str, Any]], None]
_HEARTBEAT_S = 15.0
_DONE = object()


def format_event(kind: str, data: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


def sse_response(work: Callable[[Emit], Any]) -> StreamingResponse:
    """Run ``work(emit)`` on a thread and stream its progress, then its result."""
    events: queue.Queue[Any] = queue.Queue()

    def emit(event: dict[str, Any]) -> None:
        events.put(("progress", event))

    def target() -> None:
        try:
            events.put(("result", work(emit)))
        except ScragError as exc:
            if exc.http_status >= 500:
                logger.error("%s in stream: %s", exc.code, exc)
            else:
                logger.info("%s in stream: %s", exc.code, exc)
            body: dict[str, Any] = {
                "error": exc.code,
                "detail": exc.public_message,
                "request_id": request_id_var.get(),
                "status": exc.http_status,
            }
            if "Retry-After" in exc.headers:
                body["retry_after"] = int(exc.headers["Retry-After"])
            events.put(("error", body))
        except Exception:
            logger.exception("Unhandled error in stream")
            events.put(
                (
                    "error",
                    {
                        "error": "internal_error",
                        "detail": "An internal error occurred.",
                        "request_id": request_id_var.get(),
                        "status": 500,
                    },
                )
            )
        finally:
            events.put(_DONE)

    # Copy the request context so logs from the work carry the request ID.
    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(target,), name="scrag-sse", daemon=True).start()

    def stream() -> Iterator[str]:
        while True:
            try:
                item = events.get(timeout=_HEARTBEAT_S)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is _DONE:
                return
            kind, data = item
            yield format_event(kind, data)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
