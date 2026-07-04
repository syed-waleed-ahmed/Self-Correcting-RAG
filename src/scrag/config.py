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

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the whole application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SCRAG_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (provider-agnostic, OpenAI-compatible) -----------------------
    llm_api_key: str | None = Field(
        default=None,
        # Accept SCRAG_LLM_API_KEY, GROQ_API_KEY or OPENAI_API_KEY.
        validation_alias=AliasChoices(
            "SCRAG_LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"
        ),
        description="API key for the OpenAI-compatible LLM provider.",
    )
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL of the OpenAI-compatible endpoint (Groq by default).",
    )
    generator_model: str = "llama-3.1-8b-instant"
    evaluator_model: str = "llama-3.1-8b-instant"
    guardrail_model: str = "llama-3.1-8b-instant"
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0, description="Chunk size in characters.")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap in characters.")

    # --- Vector store -----------------------------------------------------
    vector_backend: Literal["numpy", "faiss"] = "numpy"
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")

    # --- Retrieval / pipeline --------------------------------------------
    top_k: int = Field(default=5, gt=0)
    # Small instruct models score relevance conservatively, so a moderate
    # threshold keeps genuinely relevant chunks. The pipeline also falls back to
    # raw retrieval if the guardrail drops everything.
    guardrail_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    guardrail_max_workers: int = Field(
        default=4, gt=0, description="Concurrent guardrail scoring workers."
    )
    eval_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_self_correct_steps: int = Field(default=2, ge=1)

    # --- API / server -----------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0, lt=65536)
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Optional shared-secret auth for the write/query endpoints. When set,
    # /query and /ingest require it via `Authorization: Bearer <token>` or the
    # `X-API-Key` header. Unset (default) leaves the API open.
    api_auth_token: str | None = None

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @property
    def has_llm(self) -> bool:
        """Whether an LLM key is configured (agents require this)."""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
