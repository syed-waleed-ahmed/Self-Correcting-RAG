"""Domain error hierarchy.

Core modules raise these; the API layer maps each one to an HTTP status and a
stable, client-safe message. ``str(exc)`` may carry operator-facing detail
(paths, provider responses) and is only ever logged or shown in the CLI;
``public_message`` is what HTTP clients see and never includes internals.

Several errors also subclass the builtin they replaced (``ValueError``,
``FileNotFoundError``, ``RuntimeError``) so existing ``except`` clauses keep
working.
"""

from __future__ import annotations


class ScragError(Exception):
    """Base class for all expected, classifiable SCRAG failures."""

    code: str = "internal_error"
    http_status: int = 500
    public_message: str = "An internal error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)


# --- Request / state errors (4xx) ---------------------------------------------
class InvalidQueryError(ScragError, ValueError):
    code = "invalid_query"
    http_status = 400
    public_message = "The query must not be empty."


class IndexEmptyError(ScragError):
    code = "index_empty"
    http_status = 409
    public_message = "The index is empty. Ingest documents first (POST /ingest)."


class IngestInProgressError(ScragError):
    code = "ingest_in_progress"
    http_status = 409
    public_message = "An ingestion is already running. Retry when it has finished."


class DocumentsNotFoundError(ScragError, FileNotFoundError):
    code = "documents_not_found"
    http_status = 409
    public_message = "The configured documents directory does not exist."


class NoDocumentsError(ScragError, ValueError):
    code = "no_documents"
    http_status = 409
    public_message = "No ingestable documents (.txt, .md) were found."


# --- Server-side configuration / data errors (5xx) ---------------------------
class ConfigurationError(ScragError):
    code = "configuration_error"
    http_status = 500
    public_message = "The server is misconfigured. The operator has been notified in the logs."


class EmbeddingDimensionError(ConfigurationError, ValueError):
    """Embedding vectors do not have the dimensionality the index expects."""

    code = "embedding_dimension_mismatch"


class EmbeddingError(ScragError):
    """The embedder produced unusable output (e.g. NaN values)."""

    code = "embedding_error"


class IndexIncompatibleError(ConfigurationError, ValueError):
    """A persisted index cannot be loaded with the current configuration."""

    code = "index_incompatible"


# --- LLM provider errors ------------------------------------------------------
class LLMNotConfiguredError(ScragError):
    code = "llm_not_configured"
    http_status = 503
    public_message = "The language model is not configured on this server."


class LLMError(ScragError, RuntimeError):
    """An LLM request ultimately failed (after the SDK's own retries)."""

    code = "llm_error"
    http_status = 502
    public_message = "The language model provider request failed. Please retry."


class LLMRateLimitError(LLMError):
    """The provider rejected the request for rate-limit or quota reasons."""

    code = "llm_rate_limited"
    http_status = 503
    public_message = "The language model provider is rate limiting requests. Please retry later."
