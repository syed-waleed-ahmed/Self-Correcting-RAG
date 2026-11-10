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
        self.headers: dict[str, str] = {}


# --- Demo / quota errors (429, 403) ------------------------------------------------
class RateLimitedError(ScragError):
    code = "rate_limited"
    http_status = 429
    public_message = "Too many requests. Please wait a moment and try again."

    def __init__(self, message: str | None = None, *, retry_after: int = 60) -> None:
        super().__init__(message)
        self.headers = {"Retry-After": str(retry_after)}


class DemoBudgetExhaustedError(ScragError):
    code = "demo_budget_exhausted"
    http_status = 429
    public_message = (
        "The demo's daily budget of live AI queries is used up. Sample questions "
        "still work from recorded results; live queries resume after the daily reset."
    )

    def __init__(self, message: str | None = None, *, retry_after: int = 3600) -> None:
        super().__init__(message)
        self.headers = {"Retry-After": str(retry_after)}


class ClientQuotaExceededError(ScragError):
    code = "client_quota_exceeded"
    http_status = 429
    public_message = (
        "You've used your share of today's live AI queries on the demo. Sample questions "
        "still work from recorded results; your share resets at midnight UTC."
    )

    def __init__(self, message: str | None = None, *, retry_after: int = 3600) -> None:
        super().__init__(message)
        self.headers = {"Retry-After": str(retry_after)}


class ForbiddenError(ScragError):
    code = "forbidden"
    http_status = 403
    public_message = "This operation is not available on the public demo."


class QueryTooLongError(ScragError, ValueError):
    code = "query_too_long"
    http_status = 400
    public_message = "The query is too long for the demo."


# --- Sessions, uploads and persistence (Phase 4) ------------------------------------
class SessionRequiredError(ScragError):
    code = "session_required"
    http_status = 401
    public_message = (
        "A valid session token is required in the X-Session-Token header. "
        "Create one with POST /api/v1/sessions."
    )


class NotFoundError(ScragError, LookupError):
    code = "not_found"
    http_status = 404
    public_message = "The requested resource does not exist."


class UploadTooLargeError(ScragError, ValueError):
    code = "upload_too_large"
    http_status = 413
    public_message = "The uploaded file is too large."


class UnsupportedFileTypeError(ScragError, ValueError):
    code = "unsupported_file_type"
    http_status = 415
    public_message = (
        "Unsupported file type. Upload .txt, .md, .html, .csv, .json, .pdf or .docx files."
    )


class UnreadableDocumentError(ScragError, ValueError):
    code = "unreadable_document"
    http_status = 422
    public_message = "The file could not be read as text (empty, encrypted or malformed)."


class DocumentTooLongError(ScragError, ValueError):
    code = "document_too_long"
    http_status = 413
    public_message = "The document has more text than an upload may contain."


class SessionQuotaExceededError(ScragError):
    code = "session_quota_exceeded"
    http_status = 409
    public_message = (
        "This session has reached its document limit. Delete a document or start a new session."
    )


class StorageFullError(ScragError):
    code = "storage_full"
    http_status = 503
    public_message = "The demo's document storage is full. Please try again later."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message)
        self.headers = {"Retry-After": "3600"}


class SessionCapacityError(ScragError):
    code = "session_capacity"
    http_status = 503
    public_message = "The demo has reached its limit of active sessions. Please try again later."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message)
        self.headers = {"Retry-After": "3600"}


class PersistenceDisabledError(ScragError):
    code = "persistence_disabled"
    http_status = 503
    public_message = "Sessions, uploads and history are not enabled on this server."


class DatabaseUnavailableError(ScragError):
    code = "database_unavailable"
    http_status = 503
    public_message = (
        "The database is unavailable right now. Questions about the sample documents "
        "still work; please retry the rest later."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message)
        self.headers = {"Retry-After": "30"}


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
    public_message = "No ingestable documents were found in the documents directory."


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
