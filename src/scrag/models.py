"""Domain and API data models (pydantic).

These are the serializable contracts shared across the retrieval, agent,
pipeline and API layers. Keeping them in one place makes the data flow explicit
and gives us free validation + JSON schema for the REST API.
"""

from __future__ import annotations

import hashlib
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

# Outcome of a single evaluator call. "error" means the evaluator could not be
# reached; "invalid" means it answered but no score could be parsed. Neither is
# a judgement about answer quality.
EvaluationStatus = Literal["ok", "error", "invalid"]

# Overall verdict for the answer a pipeline run returns. "passed" only means the
# LLM judge's score cleared the configured threshold: a heuristic signal, not
# verification.
ResultStatus = Literal["passed", "below_threshold", "unavailable", "skipped"]


class Chunk(BaseModel):
    """A unit of retrievable text derived from a source document."""

    id: str
    source: str = Field(description="Origin of the chunk, e.g. a filename.")
    text: str
    position: int = Field(default=0, description="Index of the chunk within its source.")
    metadata: dict[str, str] = Field(default_factory=dict)

    @staticmethod
    def make_id(source: str, position: int, text: str) -> str:
        """Deterministic, collision-resistant id for a chunk."""
        digest = hashlib.sha1(f"{source}:{position}:{text}".encode()).hexdigest()
        return digest[:16]


class RetrievedChunk(BaseModel):
    """A chunk plus the scores assigned to it during retrieval/guardrailing."""

    chunk: Chunk
    score: float = Field(description="Cosine similarity between query and chunk.")
    fusion_score: float | None = Field(
        default=None, description="Reciprocal Rank Fusion score (hybrid retrieval only)."
    )
    keyword_score: float | None = Field(
        default=None, description="BM25 score, if the chunk matched any query keyword."
    )
    guardrail_score: float | None = Field(
        default=None, description="Relevance score from the guardrail agent, if run."
    )

    @property
    def rank_key(self) -> float:
        """The value retrieval ranked by: fused score in hybrid mode, else cosine."""
        return self.fusion_score if self.fusion_score is not None else self.score


class Attempt(BaseModel):
    """A single generate→evaluate attempt within the self-correction loop."""

    number: int
    answer: str
    score: float | None = Field(
        description="Evaluator score in [0, 1], or null when the evaluation failed."
    )
    explanation: str
    evaluation_status: EvaluationStatus = "ok"


class PipelineResult(BaseModel):
    """The final result returned by the pipeline for a query."""

    query: str
    answer: str
    evaluation_score: float | None = Field(
        description="Score of the selected attempt, or null if it could not be evaluated."
    )
    evaluation_explanation: str
    evaluation_status: ResultStatus  # required: never default to "passed"
    attempts: int = Field(description="Number of generate/evaluate attempts executed.")
    best_attempt: int = Field(
        default=0, description="Number of the attempt whose answer was returned (0 if none)."
    )
    attempt_history: list[Attempt] = Field(default_factory=list)
    used_chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------
# API request/response schemas
# --------------------------------------------------------------------------
# What a query searches: the bundled sample corpus, the caller's session uploads,
# or both (fused). "session" and "all" need an X-Session-Token.
Scope = Literal["samples", "session", "all"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="The user's question.")
    top_k: int | None = Field(default=None, ge=1, le=100)
    use_cache: bool = Field(
        default=True,
        description="Allow a cached answer (labelled as such). False forces a live run.",
    )
    scope: Scope = Field(default="samples", description="Which documents to search.")
    conversation_id: UUID | None = Field(
        default=None,
        description="Append to this conversation (session only). Omit to start a new one.",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    scope: Scope = "samples"


class ChunkView(BaseModel):
    """Trimmed chunk representation for API responses."""

    source: str
    text: str
    score: float
    fusion_score: float | None = None
    keyword_score: float | None = None
    guardrail_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    evaluation_score: float | None
    evaluation_explanation: str
    evaluation_status: ResultStatus
    attempts: int = Field(description="Number of generate/evaluate attempts executed.")
    best_attempt: int = Field(description="Number of the attempt whose answer was returned.")
    attempt_history: list[Attempt]
    used_chunks: list[ChunkView]
    cached: bool = Field(
        default=False, description="True if this is a stored result of an earlier real run."
    )
    cached_at: str | None = Field(default=None, description="When the cached run happened.")
    cache_source: Literal["sample", "recent"] | None = None
    scope: Scope = "samples"
    conversation_id: UUID | None = Field(
        default=None, description="Conversation this turn was saved to (session requests only)."
    )
    history_saved: bool | None = Field(
        default=None,
        description="Session requests: whether the turn was saved (false: quota or outage).",
    )


class AgentRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=1000, description="What the agent should do.")
    approved_tools: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Sensitive tools a human approved for this run (operator only on the demo).",
    )


class SearchResponse(BaseModel):
    query: str
    retrieval_mode: str
    chunks: list[ChunkView]


class DemoInfo(BaseModel):
    demo_mode: bool
    llm_configured: bool
    model: str
    retrieval_mode: str
    sample_questions: list[str]
    limits: dict[str, float | int]
    budget: dict[str, int | str] | None


class IngestRequest(BaseModel):
    reset: bool = Field(
        default=True, description="Rebuild the index from scratch rather than appending."
    )


class IngestResponse(BaseModel):
    documents: int = Field(description="Documents found in the docs directory.")
    chunks: int = Field(description="Chunks embedded and written during this run.")
    index_size: int = Field(description="Total chunks in the index after this run.")
    added_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = Field(default=0, description="Unchanged documents not re-embedded.")


class ErrorResponse(BaseModel):
    """Body of every non-validation error response."""

    error: str = Field(description="Stable, machine-readable error code.")
    detail: str = Field(description="Human-readable, client-safe message.")
    request_id: str | None = Field(default=None, description="Correlates with server logs.")


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool
    embedding_model: str
    vector_backend: str
    index_size: int
    demo_mode: bool = False
    # "disabled" (no database configured), else the last observed state. Health
    # never queries the database itself (it must stay fast and I/O-free).
    persistence: Literal["disabled", "unknown", "ok", "unavailable"] = "disabled"
