"""Observability: Prometheus metrics and optional OpenTelemetry tracing (Phase 6)."""

from . import metrics
from .tracing import configure_tracing, set_attributes, shutdown_tracing, span, tracing_enabled

__all__ = [
    "configure_tracing",
    "metrics",
    "set_attributes",
    "shutdown_tracing",
    "span",
    "tracing_enabled",
]
