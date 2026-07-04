"""Guardrail agent: score chunk relevance and filter out off-topic context.

Scoring each retrieved chunk is embarrassingly parallel, so we fan the calls out
across a small thread pool to keep latency low on larger ``top_k`` values.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..llm.client import ChatClient, LLMError
from ..logging_config import get_logger
from ..models import RetrievedChunk
from ._parsing import parse_score

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


class GuardrailAgent:
    def __init__(
        self,
        client: ChatClient,
        model: str,
        threshold: float = 0.6,
        max_workers: int = 4,
    ) -> None:
        self.client = client
        self.model = model
        self.threshold = threshold
        self.max_workers = max_workers

    def score(self, query: str, chunk_text: str) -> float:
        user = (
            f"Query:\n{query}\n\nDocument chunk:\n{chunk_text}\n\n"
            "Relevance score (0.0-1.0):"
        )
        try:
            content = self.client.complete(
                model=self.model,
                system=_SYSTEM,
                user=user,
                temperature=0.0,
                max_tokens=8,
            )
        except LLMError:
            # On failure, don't silently drop the chunk — keep it (score 1.0) so a
            # provider hiccup can't starve the generator of context.
            logger.warning("Guardrail scoring failed; keeping chunk by default")
            return 1.0
        return parse_score(content)

    def filter(
        self, query: str, retrieved: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Attach guardrail scores and keep chunks at/above the threshold."""
        if not retrieved:
            return []

        workers = max(1, min(self.max_workers, len(retrieved)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(
                pool.map(lambda rc: self.score(query, rc.chunk.text), retrieved)
            )

        kept: list[RetrievedChunk] = []
        for rc, score in zip(retrieved, scores, strict=True):
            rc.guardrail_score = score
            if score >= self.threshold:
                kept.append(rc)
        logger.info("Guardrail kept %d/%d chunks", len(kept), len(retrieved))
        return kept
