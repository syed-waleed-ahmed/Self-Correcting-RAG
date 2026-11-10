"""Optional OpenTelemetry tracing.

Off by default (``SCRAG_OTEL_EXPORTER=none``): :func:`span` is then a no-op and
the OpenTelemetry SDK is never imported. With ``console`` spans are printed to
stdout; with ``otlp`` they are sent to any OTLP/HTTP collector configured by
the standard ``OTEL_EXPORTER_OTLP_*`` environment variables. Needs the
``otel`` extra.

Spans: one per HTTP request (named by route template), the pipeline run and
each stage, every LLM call (with GenAI semantic-convention attributes), agent
runs and tool calls. Context propagates through ``contextvars``, so spans
started on worker threads (guardrail fan-out, streaming) nest correctly.

The tracer comes from this module's own provider rather than the process-global
one, so configuring tracing twice (tests, several apps) never conflicts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


class _State:
    """The active tracer and its provider (both None while tracing is off)."""

    tracer: Any = None
    provider: Any = None


_state = _State()


def configure_tracing(
    exporter: str,
    *,
    service_name: str = "scrag-api",
    version: str = "",
    span_exporter: Any = None,
) -> bool:
    """Enable tracing. ``span_exporter`` (e.g. an in-memory exporter) overrides
    ``exporter`` and exports synchronously. Returns whether tracing is on."""
    shutdown_tracing()
    if exporter == "none" and span_exporter is None:
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        logger.warning(
            'Tracing requested but OpenTelemetry is not installed: pip install "%s[otel]"',
            "self-correcting-rag",
        )
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": version})
    )
    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    elif exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        logger.warning("Unknown SCRAG_OTEL_EXPORTER %r; tracing stays off", exporter)
        return False
    _state.provider = provider
    _state.tracer = provider.get_tracer("scrag", version or None)
    logger.info("Tracing enabled (exporter=%s)", "custom" if span_exporter else exporter)
    return True


def shutdown_tracing() -> None:
    """Flush and stop the current provider, if any (tracing is then off)."""
    if _state.provider is not None:
        try:
            _state.provider.shutdown()
        except Exception:  # noqa: BLE001 - never fail shutdown over telemetry
            logger.warning("Tracing shutdown failed", exc_info=True)
    _state.tracer = None
    _state.provider = None


def tracing_enabled() -> bool:
    return _state.tracer is not None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A span named ``name`` (no-op when tracing is off). Exceptions are recorded."""
    tracer = _state.tracer
    if tracer is None:
        yield None
        return
    clean = {k.replace("__", "."): v for k, v in attributes.items() if v is not None}
    with tracer.start_as_current_span(name, attributes=clean) as current:
        yield current


def set_attributes(current: Any, **attributes: Any) -> None:
    """Set attributes on a span from :func:`span` (ignored when tracing is off)."""
    if current is None:
        return
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key.replace("__", "."), value)
