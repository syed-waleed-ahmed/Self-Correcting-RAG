"""Domain and API data models (pydantic).

These are the serializable contracts shared across the retrieval, agent,
pipeline and API layers. Keeping them in one place makes the data flow explicit
and gives us free validation + JSON schema for the REST API.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


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
    score: float = Field(description="Retrieval similarity score (cosine).")
    guardrail_score: float | None = Field(
        default=None, description="Relevance score from the guardrail agent, if run."
    )


class Attempt(BaseModel):
    """A single generate→evaluate attempt within the self-correction loop."""

    number: int
    answer: str
    score: float
    explanation: str


class PipelineResult(BaseModel):
    """The final result returned by the pipeline for a query."""

    query: str
    answer: str
    evaluation_score: float
    evaluation_explanation: str
    attempts: int
    used_chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------
# API request/response schemas
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The user's question.")
    top_k: int | None = Field(default=None, ge=1, le=100)


class ChunkView(BaseModel):
    """Trimmed chunk representation for API responses."""

    source: str
    text: str
    score: float
    guardrail_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    evaluation_score: float
    evaluation_explanation: str
    attempts: int
    used_chunks: list[ChunkView]


class IngestRequest(BaseModel):
    reset: bool = Field(
        default=True, description="Rebuild the index from scratch rather than appending."
    )


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    index_size: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool
    embedding_model: str
    vector_backend: str
    index_size: int
