"""Guardrail agent: score chunk relevance and filter out off-topic context.

Two modes:
    * ``per_chunk`` - one LLM call per chunk, fanned out across a small thread
      pool to keep latency low on larger ``top_k`` values.
    * ``batch`` - all chunks scored in one JSON call, which spends one request
      instead of ``top_k`` against provider rate limits. If the reply can't be
      parsed, it falls back to ``per_chunk``; if the provider is down, every
      chunk is left unscored without firing more doomed calls.

The guardrail fails open: if a chunk cannot be scored (provider error or
unparseable output) it is kept with ``guardrail_score=None``, so a provider
hiccup can't starve the generator of context and no fake score is reported.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from ..errors import LLMError
from ..llm.client import ChatClient
from ..logging_config import get_logger
from ..models import RetrievedChunk
from ._parsing import normalize_score, parse_json_object

logger = get_logger(__name__)

_SYSTEM = (
    "You are a relevance guardrail. Score how much a document chunk helps answer "
    "the user's query, on this scale:\n"
    "  1.0  = directly answers the query\n"
    "  0.7-0.9 = clearly relevant / on-topic\n"
    "  0.4-0.6 = partially or tangentially related\n"
    "  0.1-0.3 = weakly related\n"
    "  0.0  = unrelated\n"
    "Reply with ONLY the number (e.g. 0.8)."
)

_BATCH_SYSTEM = (
    "You are a relevance guardrail. For each numbered document chunk, score how "
    "much it helps answer the user's query: 1.0 = directly answers it, 0.7-0.9 = "
    "clearly relevant, 0.4-0.6 = partially related, 0.1-0.3 = weakly related, "
    '0.0 = unrelated. Return JSON only, in the form {"scores": [0.8, 0.1, ...]} '
    "with exactly one score per chunk, in chunk order."
)


class GuardrailAgent:
    def __init__(
        self,
        client: ChatClient,
        model: str,
        threshold: float = 0.6,
        max_workers: int = 4,
        max_tokens: int = 8,
        mode: Literal["batch", "per_chunk"] = "per_chunk",
    ) -> None:
        self.client = client
        self.model = model
        self.threshold = threshold
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.mode = mode

    def score(self, query: str, chunk_text: str) -> float | None:
        """Relevance in ``[0, 1]``, or ``None`` if the chunk could not be scored."""
        user = f"Query:\n{query}\n\nDocument chunk:\n{chunk_text}\n\nRelevance score (0.0-1.0):"
        try:
            content = self.client.complete(
                model=self.model,
                system=_SYSTEM,
                user=user,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except LLMError as exc:
            logger.warning("Guardrail scoring failed (%s); keeping chunk", type(exc).__name__)
            return None
        score = normalize_score(content)
        if score is None:
            logger.warning("Guardrail output had no parseable score; keeping chunk")
        return score

    def score_batch(self, query: str, texts: list[str]) -> list[float | None] | None:
        """Score all ``texts`` in one call.

        Returns one score (or ``None``) per text, ``[None] * n`` if the provider
        failed, or ``None`` if the reply was unusable (caller falls back).
        """
        listing = "\n\n".join(f"[{i + 1}]\n{text}" for i, text in enumerate(texts))
        user = f"Query:\n{query}\n\nDocument chunks:\n{listing}\n\nReturn the JSON object only."
        # Room for the JSON plus, for reasoning models, the configured extra budget.
        budget = 32 + 12 * len(texts) + (self.max_tokens if self.max_tokens > 16 else 0)
        try:
            content = self.client.complete(
                model=self.model,
                system=_BATCH_SYSTEM,
                user=user,
                temperature=0.0,
                max_tokens=budget,
                json_mode=True,
            )
        except LLMError as exc:
            logger.warning("Batched guardrail failed (%s); keeping all chunks", type(exc).__name__)
            return [None] * len(texts)
        obj = parse_json_object(content)
        raw = obj.get("scores") if obj is not None else None
        if not isinstance(raw, list) or len(raw) != len(texts):
            logger.warning("Batched guardrail reply unusable; scoring chunks one by one")
            return None
        return [normalize_score(str(value)) for value in raw]

    def filter(self, query: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Attach guardrail scores and keep chunks at/above the threshold (or unscored)."""
        if not retrieved:
            return []

        scores: list[float | None] | None = None
        if self.mode == "batch":
            scores = self.score_batch(query, [rc.chunk.text for rc in retrieved])
        if scores is None:
            workers = max(1, min(self.max_workers, len(retrieved)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scores = list(pool.map(lambda rc: self.score(query, rc.chunk.text), retrieved))

        kept: list[RetrievedChunk] = []
        for rc, score in zip(retrieved, scores, strict=True):
            rc.guardrail_score = score
            if score is None or score >= self.threshold:
                kept.append(rc)
        unscored = sum(score is None for score in scores)
        logger.info(
            "Guardrail kept %d/%d chunks (%d unscored)", len(kept), len(retrieved), unscored
        )
        return kept
