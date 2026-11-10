"""Shared test fixtures and fakes.

The fakes let the entire pipeline (retrieval → guardrail → generate → evaluate)
and the API run deterministically without network access or the torch/embedding
stack. ``FakeEmbedder`` is a tiny bag-of-words embedder over a fixed vocabulary
so cosine similarity is meaningful; ``FakeChatClient`` is a scriptable
:class:`scrag.llm.client.ChatClient`.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from scrag.config import Settings
from scrag.models import Chunk

VOCAB = [
    "rag",
    "retrieval",
    "generation",
    "guardrail",
    "agent",
    "filter",
    "context",
    "answer",
    "embedding",
    "vector",
]
FAKE_DIM = len(VOCAB)


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary."""

    def __init__(self, dim: int = FAKE_DIM) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        rows = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            rows.append([float(words.count(term)) for term in VOCAB])
        return np.asarray(rows, dtype="float32")


Script = str | Exception | list[str | Exception]


class FakeChatClient:
    """Scriptable ChatClient. Routes each call by role via the system prompt.

    Each of ``guardrail``/``generator``/``evaluator`` may be a fixed value or a
    list consumed one item per call (useful for multi-attempt scenarios). A
    value that is an ``Exception`` instance is raised instead of returned, to
    simulate provider failures. ``calls`` records the role of every call and
    ``prompts`` the ``(role, user_prompt)`` pairs.
    """

    def __init__(
        self,
        guardrail: Script = "0.9",
        generator: Script = "A grounded answer.",
        evaluator: Script = '{"score": 0.9, "explanation": "well grounded"}',
    ) -> None:
        self.guardrail = guardrail
        self.generator = generator
        self.evaluator = evaluator
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    @staticmethod
    def _next(value: Script, fallback: str) -> str:
        item = (value.pop(0) if value else fallback) if isinstance(value, list) else value
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, *, model, system, user, temperature=0.2, max_tokens=None, json_mode=False):  # noqa: ANN001,E501
        role = "generator"
        low = system.lower()
        if "guardrail" in low:
            role = "guardrail"
        elif "evaluator" in low:
            role = "evaluator"
        self.calls.append(role)
        self.prompts.append((role, user))
        if role == "guardrail":
            return self._next(self.guardrail, "0.0")
        if role == "evaluator":
            return self._next(self.evaluator, '{"score": 0.0, "explanation": ""}')
        return self._next(self.generator, "")

    def prompts_for(self, role: str) -> list[str]:
        return [user for r, user in self.prompts if r == role]


# Postgres tests run only when this points at a disposable Postgres with pgvector
# (CI starts one; locally e.g. `docker run -e POSTGRES_PASSWORD=pg -p 5432:5432
# pgvector/pgvector:pg17`). Read once at import, before _hermetic_env runs.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
_SCRAG_TABLES = (
    "scrag.usage_counters, scrag.jobs, scrag.messages, scrag.conversations,"
    " scrag.chunks, scrag.documents, scrag.sessions"
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Keep a developer's real keys and SCRAG_* overrides out of every test."""
    for key in list(os.environ):
        if key.startswith("SCRAG_") or key in {
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            "DATABASE_URL",
        }:
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def database():
    """A migrated, emptied test database (skips without TEST_DATABASE_URL)."""
    if not TEST_DATABASE_URL:
        pytest.skip("set TEST_DATABASE_URL to run the Postgres tests")
    from scrag.persistence import Database

    db = Database(TEST_DATABASE_URL, retry_interval_s=0)
    db.ensure_ready()
    with db.connection() as conn:
        conn.execute(f"TRUNCATE {_SCRAG_TABLES} CASCADE")
    yield db
    db.close()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    texts = [
        "rag combines retrieval and generation to answer questions",
        "a guardrail agent can filter off-topic context before generation",
        "an embedding maps text to a vector for retrieval",
    ]
    return [
        Chunk(id=Chunk.make_id("doc", i, t), source=f"doc{i}.txt", text=t, position=i)
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Hermetic settings: ignore any real .env, use temp dirs and fake dim."""
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        embedding_dim=FAKE_DIM,
        docs_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        top_k=3,
        guardrail_threshold=0.6,
        eval_threshold=0.7,
        max_self_correct_steps=2,
    )
