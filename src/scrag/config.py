"""Typed, environment-driven application configuration.

Settings are read (in priority order) from constructor kwargs, environment
variables, then a local ``.env`` file. Most settings use the ``SCRAG_`` prefix
(e.g. ``SCRAG_TOP_K``). The LLM API key is special-cased to also accept the
conventional ``GROQ_API_KEY`` / ``OPENAI_API_KEY`` names so existing setups keep
working without changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the whole application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SCRAG_",
        case_sensitive=False,
        extra="ignore",
        # Let code pass `llm_api_key=...` as well as the env-style aliases.
        populate_by_name=True,
    )

    # --- LLM (provider-agnostic, OpenAI-compatible) -----------------------
    llm_api_key: str | None = Field(
        default=None,
        # Accept SCRAG_LLM_API_KEY, GROQ_API_KEY or OPENAI_API_KEY.
        validation_alias=AliasChoices("SCRAG_LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"),
        description="API key for the OpenAI-compatible LLM provider.",
        repr=False,  # secrets never appear in repr(settings) or logs of it
    )
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL of the OpenAI-compatible endpoint (Groq by default).",
    )
    # Groq retired llama-3.1-8b-instant / llama-3.3-70b-versatile for free and
    # developer tiers on 2026-08-16. Provider model lists change often: check
    # https://console.groq.com/docs/models when a model ID starts failing.
    generator_model: str = "qwen/qwen3.8-27b"
    evaluator_model: str = "qwen/qwen3.8-27b"
    guardrail_model: str = "qwen/qwen3.8-27b"
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    # --- Embeddings -------------------------------------------------------
    # "auto" uses fastembed (ONNX, ~half the memory of torch) when installed,
    # else sentence-transformers. Both produce the same vectors for the same
    # model, so switching backends does not invalidate an index.
    embedding_backend: Literal["auto", "fastembed", "sentence-transformers"] = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: Path | None = Field(
        default=None, description="Where model files are cached (bake into images)."
    )
    embedding_dim: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0, description="Chunk size in characters.")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap in characters.")

    # --- Ingestion limits -------------------------------------------------
    max_document_bytes: int = Field(default=10_000_000, gt=0, description="Skip larger files.")
    max_pdf_pages: int = Field(default=300, gt=0)

    # --- Vector store -----------------------------------------------------
    vector_backend: Literal["numpy", "faiss"] = "numpy"
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")

    # --- Retrieval / pipeline --------------------------------------------
    top_k: int = Field(default=5, gt=0, le=100)
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    hybrid_candidate_pool: int = Field(default=20, gt=0, le=500)
    rrf_k: int = Field(default=60, gt=0)
    # Small instruct models score relevance conservatively, so a moderate
    # threshold keeps genuinely relevant chunks. The pipeline also falls back to
    # raw retrieval if the guardrail drops everything.
    guardrail_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    guardrail_max_workers: int = Field(
        default=4, gt=0, le=32, description="Concurrent guardrail scoring workers."
    )
    # The guardrail only needs a number back. Reasoning models that spend
    # completion tokens on hidden reasoning (e.g. openai/gpt-oss-*) need ~256.
    guardrail_max_tokens: int = Field(default=8, gt=0, le=4096)
    # "batch" scores all chunks in one LLM call (fewer requests against provider
    # rate limits), falling back to one call per chunk if the reply can't be parsed.
    guardrail_mode: Literal["batch", "per_chunk"] = "batch"
    eval_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Bounded on purpose: each step costs two LLM calls.
    max_self_correct_steps: int = Field(default=2, ge=1, le=10)

    # --- Agent workflow (bounded) ------------------------------------------
    agent_max_steps: int = Field(default=4, ge=1, le=8)
    agent_max_llm_calls: int = Field(default=6, ge=3, le=12)
    agent_timeout_s: float = Field(default=90.0, gt=0, le=600)
    # Live demo budget units charged per agent run (it makes more LLM calls
    # than a single /query).
    agent_budget_cost: int = Field(default=2, ge=1, le=10)
    # Read-only SQL over an in-memory catalog of the sample corpus (sql_query).
    sql_tool_enabled: bool = True
    # web_fetch: HTTPS GETs to these hosts only ("docs.example.com", or
    # "*.example.com" for its subdomains). Empty (default) disables the tool.
    http_tool_allowed_hosts: list[str] = Field(default_factory=list)
    http_tool_max_bytes: int = Field(default=200_000, ge=1_000, le=5_000_000)
    http_tool_timeout_s: float = Field(default=8.0, gt=0, le=60)

    # --- API / server -----------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0, lt=65536)
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Optional shared-secret auth for the write/query endpoints. When set,
    # /query and /ingest require it via `Authorization: Bearer <token>` or the
    # `X-API-Key` header. Unset (default) leaves the API open.
    api_auth_token: str | None = Field(default=None, repr=False)
    # Largest accepted request body outside uploads (JSON requests are tiny; this
    # stops a client from making the server buffer megabytes before validation).
    max_request_bytes: int = Field(default=65_536, ge=1_024, le=10_000_000)

    # Client IP for rate limiting. Behind a reverse proxy, name the header it
    # sets (e.g. "x-forwarded-for") and how many trusted proxy hops append to it;
    # the address `hops` from the right is used, which clients cannot spoof.
    forwarded_ip_header: str | None = None
    forwarded_ip_hops: int = Field(default=1, ge=1, le=10)

    # --- Answer cache ----------------------------------------------------------
    answer_cache_ttl_seconds: int = Field(default=3600, ge=0)
    answer_cache_size: int = Field(default=256, ge=0)
    samples_path: Path = Path("data/demo/samples.json")

    # --- Public demo mode --------------------------------------------------------
    # Enables per-client rate limits, a global daily budget of live LLM runs,
    # query length limits, a top_k cap, and locks /ingest unless an
    # api_auth_token is configured and presented.
    demo_mode: bool = False
    demo_rate_limit_per_minute: float = Field(default=6.0, gt=0)
    demo_rate_limit_burst: int = Field(default=3, ge=1)
    # A coarse per-client cap on *all* requests (health checks exempt), refused
    # before routing. The limit above covers only the expensive endpoints; this
    # one also caps the rest, including database-backed reads, against floods
    # (measured in docs/load-testing.md). Generous for any human visitor.
    demo_flood_limit_per_minute: float = Field(default=120.0, gt=0)
    demo_flood_burst: int = Field(default=60, ge=1)
    demo_daily_query_budget: int = Field(default=30, ge=0)
    # Each client (IP) may spend at most this many of those live runs per UTC day,
    # so one visitor can't use up the whole demo's budget (an agent run costs
    # agent_budget_cost). In-process, like the rate limiter.
    demo_client_daily_runs: int = Field(default=10, ge=1)
    demo_max_query_chars: int = Field(default=300, ge=1)
    # 5, not 3: on the 60-question golden set, hybrid hit@3 misses 2 questions
    # that hit@5 retrieves, and a live comparison showed the pipeline then answers
    # them correctly (docs/evaluation.md), for about 40% more tokens per answer.
    demo_top_k: int = Field(default=5, ge=1, le=20)

    # --- Persistence: sessions, uploads, history, jobs (Phase 4) --------------------
    # Unset (default) keeps the API stateless: sample-corpus Q&A works without a
    # database, and the session/upload/history endpoints answer 503
    # `persistence_disabled`. Any Postgres with pgvector works (Supabase: use the
    # transaction pooler on port 6543; prepared statements are disabled for it).
    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SCRAG_DATABASE_URL", "DATABASE_URL"),
        repr=False,  # contains a password
    )
    db_pool_max: int = Field(default=4, ge=1, le=20)
    db_connect_timeout_s: float = Field(default=5.0, gt=0, le=60)
    # Retry a failed database connection at most this often (requests in between
    # fail fast with 503 `database_unavailable` instead of piling up).
    db_retry_interval_s: float = Field(default=30.0, ge=0, le=3600)
    db_migrate_on_start: bool = True
    # Anonymous sessions and everything in them are deleted after this long.
    session_ttl_hours: float = Field(default=24.0, gt=0, le=24 * 30)
    session_max_documents: int = Field(default=5, ge=1, le=1000)
    upload_max_bytes: int = Field(default=1_000_000, gt=0, le=50_000_000)
    document_max_chunks: int = Field(default=200, ge=1, le=10_000)
    # Global caps across all sessions, to protect a free database's storage quota.
    persistence_max_documents: int = Field(default=300, ge=1)
    persistence_max_sessions: int = Field(default=1_000, ge=1)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_lease_seconds: float = Field(default=120.0, gt=0, le=3600)
    job_poll_interval_s: float = Field(default=30.0, gt=0, le=3600)
    job_worker_enabled: bool = True
    session_cleanup_interval_s: float = Field(default=3600.0, gt=0)

    # --- Admin identity (JWT, e.g. Supabase Auth) ------------------------------------
    # A bearer JWT whose `sub` is listed in `admin_subjects` gets operator rights,
    # like the static api_auth_token. Configure a JWKS URL (asymmetric keys, e.g.
    # https://<project>.supabase.co/auth/v1/.well-known/jwks.json) or a shared
    # HS256 secret (legacy Supabase projects).
    auth_jwks_url: str | None = None
    auth_jwt_secret: str | None = Field(default=None, repr=False)
    auth_issuer: str | None = None
    auth_audience: str = "authenticated"
    admin_subjects: list[str] = Field(default_factory=list)

    # --- Observability (Phase 6) -----------------------------------------------------
    # OpenTelemetry tracing: "none" (default), "console" or "otlp" (configure the
    # collector with the standard OTEL_EXPORTER_OTLP_* variables). Needs [otel].
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_service_name: str = "scrag-api"
    # Optional list prices, USD per million tokens, for cost *estimates*:
    # {"model-id": {"input": 0.2, "output": 0.6}}. Unpriced models report null.
    llm_prices: dict[str, dict[str, float]] = Field(default_factory=dict)

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info: ValidationInfo) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @property
    def has_llm(self) -> bool:
        """Whether an LLM key is configured (agents require this)."""
        return bool(self.llm_api_key)

    @property
    def has_persistence(self) -> bool:
        """Whether a database is configured (sessions, uploads, history, jobs)."""
        return bool(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
