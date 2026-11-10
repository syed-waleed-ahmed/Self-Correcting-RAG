"""Structured logging setup.

Provides a plain human-readable formatter and an optional single-line JSON
formatter suitable for log aggregation in production. Call :func:`configure_logging`
once at process start (the CLI and API app factory both do this).

Every record carries a ``request_id`` (``"-"`` outside a request) taken from
:data:`request_id_var`, which the API middleware sets per request, so a
client-visible request ID can be matched to server-side log lines.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

_CONFIGURED = False

request_id_var: ContextVar[str] = ContextVar("scrag_request_id", default="-")

# Attributes present on every LogRecord; anything else is treated as structured
# "extra" context and included in the JSON output.
_RESERVED = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime", "taskName"}


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure the root logger. Idempotent within a process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(RequestIdFilter())
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s]: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
