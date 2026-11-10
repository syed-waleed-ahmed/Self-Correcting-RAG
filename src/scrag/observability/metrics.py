"""Prometheus metrics (exposed at ``GET /metrics``).

One process-wide registry; every label is low-cardinality by construction:
routes are FastAPI path templates (``/api/v1/documents/{document_id}``, never a
raw path), models come from configuration, statuses from fixed enums. Nothing
user-supplied (queries, filenames, session ids) is ever a label.

On the free tier there is a single instance, so the default in-process registry
is correct; several worker processes would need prometheus_client's
multiprocess mode.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry(auto_describe=True)

# Seconds: from fast in-process steps to slow LLM calls and cold starts.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

BUILD_INFO = Gauge(
    "scrag_build_info", "Build information (always 1).", ["version"], registry=REGISTRY
)
HTTP_REQUESTS = Counter(
    "scrag_http_requests_total",
    "HTTP requests by route template and status code.",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_DURATION = Histogram(
    "scrag_http_request_duration_seconds",
    "HTTP request duration (streaming responses: until the stream ends).",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
STAGE_DURATION = Histogram(
    "scrag_pipeline_stage_duration_seconds",
    "Duration of each pipeline stage (retrieve, guardrail, generate, evaluate).",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
PIPELINE_RUNS = Counter(
    "scrag_pipeline_runs_total",
    "Completed pipeline runs by result status.",
    ["status"],
    registry=REGISTRY,
)
PIPELINE_ATTEMPTS = Histogram(
    "scrag_pipeline_attempts",
    "Generate/evaluate attempts per pipeline run (above 1 means self-correction ran).",
    buckets=(0, 1, 2, 3, 4, 5, 10),
    registry=REGISTRY,
)
LLM_CALLS = Counter(
    "scrag_llm_calls_total",
    "LLM calls by model and outcome (ok, error, rate_limited).",
    ["model", "outcome"],
    registry=REGISTRY,
)
LLM_DURATION = Histogram(
    "scrag_llm_call_duration_seconds",
    "LLM call latency, including the SDK's own retries.",
    ["model"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "scrag_llm_tokens_total",
    "Provider-reported tokens by model and kind (prompt, completion).",
    ["model", "kind"],
    registry=REGISTRY,
)
AGENT_RUNS = Counter(
    "scrag_agent_runs_total",
    "Agent workflow runs by final status.",
    ["status"],
    registry=REGISTRY,
)
TOOL_CALLS = Counter(
    "scrag_tool_calls_total",
    "Agent tool invocations by tool and status.",
    ["tool", "status"],
    registry=REGISTRY,
)
CACHE_LOOKUPS = Counter(
    "scrag_answer_cache_lookups_total",
    "Answer cache lookups: hit_sample, hit_recent or miss.",
    ["result"],
    registry=REGISTRY,
)
JOBS = Counter(
    "scrag_jobs_total",
    "Background job outcomes (succeeded, retried, failed).",
    ["kind", "outcome"],
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """The current metrics in the Prometheus text format, with its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Current value of one sample (0 if absent). For tests and diagnostics."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0
