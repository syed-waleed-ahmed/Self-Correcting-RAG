"""The self-correcting RAG orchestrator.

Stages:
    1. Retrieval          — embed the query, fetch top-k chunks.
    2. Guardrail filter    — score and drop off-topic chunks (with a fallback so
                             an over-eager guardrail can't leave the generator
                             with nothing).
    3. Generate + evaluate — loop up to ``max_self_correct_steps`` times, feeding
                             the previous answer back for correction, and stop
                             early once the evaluation clears the threshold.

The loop keeps the *best-scoring* attempt, so self-correction can never return a
worse answer than one it already produced.
"""

from __future__ import annotations

from .agents.evaluator import EvaluatorAgent
from .agents.generator import GeneratorAgent
from .agents.guardrail import GuardrailAgent
from .logging_config import get_logger
from .models import Attempt, PipelineResult, RetrievedChunk
from .retriever import Retriever

logger = get_logger(__name__)


class SelfCorrectingRAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        guardrail: GuardrailAgent,
        generator: GeneratorAgent,
        evaluator: EvaluatorAgent,
        *,
        top_k: int = 5,
        eval_threshold: float = 0.7,
        max_self_correct_steps: int = 2,
    ) -> None:
        self.retriever = retriever
        self.guardrail = guardrail
        self.generator = generator
        self.evaluator = evaluator
        self.top_k = top_k
        self.eval_threshold = eval_threshold
        self.max_self_correct_steps = max_self_correct_steps

    def run(self, query: str, top_k: int | None = None) -> PipelineResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # 1. Retrieval
        retrieved = self.retriever.retrieve(query, top_k or self.top_k)

        # 2. Guardrail filtering (with fallback to raw retrieval)
        filtered = self.guardrail.filter(query, retrieved)
        if not filtered:
            logger.info("Guardrail dropped everything; falling back to raw retrieval")
            filtered = retrieved

        if not filtered:
            return PipelineResult(
                query=query,
                answer="I could not find any relevant context to answer this question.",
                evaluation_score=0.0,
                evaluation_explanation="No context was retrieved.",
                attempts=0,
                used_chunks=[],
            )

        # 3 + 4. Generate → evaluate → self-correct, keeping the best attempt.
        best: Attempt | None = None
        previous_answer: str | None = None
        for step in range(1, self.max_self_correct_steps + 1):
            answer = self.generator.generate(query, filtered, previous_answer)
            evaluation = self.evaluator.evaluate(query, answer, filtered)
            logger.info("Attempt %d scored %.2f", step, evaluation.score)

            if best is None or evaluation.score > best.score:
                best = Attempt(
                    number=step,
                    answer=answer,
                    score=evaluation.score,
                    explanation=evaluation.explanation,
                )
            if evaluation.score >= self.eval_threshold:
                break
            previous_answer = answer

        assert best is not None  # loop runs at least once
        return PipelineResult(
            query=query,
            answer=best.answer,
            evaluation_score=best.score,
            evaluation_explanation=best.explanation,
            attempts=best.number,
            used_chunks=self._sort_by_relevance(filtered),
        )

    @staticmethod
    def _sort_by_relevance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return sorted(chunks, key=lambda rc: rc.score, reverse=True)
