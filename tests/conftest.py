"""Shared test fixtures and fakes.

The fakes let the entire pipeline (retrieval → guardrail → generate → evaluate)
and the API run deterministically without network access or the torch/embedding
stack. ``FakeEmbedder`` is a tiny bag-of-words embedder over a fixed vocabulary
so cosine similarity is meaningful; ``FakeChatClient`` is a scriptable
:class:`scrag.llm.client.ChatClient`.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from scrag.config import Settings
from scrag.models import Chunk

VOCAB = [
    "rag", "retrieval", "generation", "guardrail", "agent",
    "filter", "context", "answer", "embedding", "vector",
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


class FakeChatClient:
    """Scriptable ChatClient. Routes each call by role via the system prompt.

    Each of ``guardrail``/``generator``/``evaluator`` may be a fixed string or a
    list consumed one item per call (useful for multi-attempt scenarios).
    """

    def __init__(
        self,
        guardrail: str | list[str] = "0.9",
        generator: str | list[str] = "A grounded answer.",
        evaluator: str | list[str] = '{"score": 0.9, "explanation": "well grounded"}',
    ) -> None:
        self.guardrail = guardrail
        self.generator = generator
        self.evaluator = evaluator
        self.calls: list[str] = []

    @staticmethod
    def _next(value: str | list[str], fallback: str) -> str:
        if isinstance(value, list):
            return value.pop(0) if value else fallback
        return value

    def complete(self, *, model, system, user, temperature=0.2, max_tokens=None, json_mode=False):  # noqa: ANN001,E501
        role = "generator"
        low = system.lower()
        if "guardrail" in low:
            role = "guardrail"
        elif "evaluator" in low:
            role = "evaluator"
        self.calls.append(role)
        if role == "guardrail":
            return self._next(self.guardrail, "0.0")
        if role == "evaluator":
            return self._next(self.evaluator, '{"score": 0.0, "explanation": ""}')
        return self._next(self.generator, "")


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
