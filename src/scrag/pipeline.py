"""The self-correcting RAG orchestrator.

Stages:
    1. Retrieval          — embed the query, fetch top-k chunks.
    2. Guardrail filter    — score and drop off-topic chunks (with a fallback so
                             an over-eager guardrail can't leave the generator
                             with nothing).
    3. Generate + evaluate — loop up to ``max_self_correct_steps`` times. Each
                             retry gets the previous answer *and* the evaluator's
                             explanation of what was wrong with it. The loop stops
                             early once an evaluation clears the threshold.

Selection rule: among attempts that were actually scored, return the
highest-scoring one (earliest wins ties), so self-correction never returns a
worse answer than one it already produced. An unscored attempt is returned only
if no attempt could be scored.

Failure rules:
    * Evaluator failure (``status != "ok"``) is not a low grade. The loop stops,
      because regenerating without a quality signal would be blind.
    * Generator failure on the first attempt propagates as ``LLMError``. On a
      retry it stops the loop and the best earlier attempt is returned.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .agents.evaluator import EvaluatorAgent
from .agents.generator import GeneratorAgent
from .agents.guardrail import GuardrailAgent
from .errors import InvalidQueryError, LLMError
from .logging_config import get_logger
from .models import Attempt, PipelineResult, ResultStatus, RetrievedChunk
from .observability import metrics, set_attributes, span
from .retriever import Retriever, SupportsRetrieve
from .usage import UsageLedger, track_usage

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = "I could not find any relevant context to answer this question."

# Receives one JSON-serializable progress event per pipeline stage.
ProgressListener = Callable[[dict[str, Any]], None]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _safe(listener: ProgressListener | None) -> ProgressListener:
    """Wrap a listener so it can never break the pipeline."""
    if listener is None:
        return lambda event: None

    def emit(event: dict[str, Any]) -> None:
        try:
            listener(event)
        except Exception:  # noqa: BLE001 - progress reporting is best effort
            logger.warning("Progress listener failed", exc_info=True)

    return emit


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
        if max_self_correct_steps < 1:
            raise ValueError("max_self_correct_steps must be at least 1")
        self.retriever = retriever
        self.guardrail = guardrail
        self.generator = generator
        self.evaluator = evaluator
        self.top_k = top_k
        self.eval_threshold = eval_threshold
        self.max_self_correct_steps = max_self_correct_steps

    def run(
        self,
        query: str,
        top_k: int | None = None,
        *,
        retriever: SupportsRetrieve | None = None,
        on_event: ProgressListener | None = None,
    ) -> PipelineResult:
        """Answer ``query``.

        ``retriever`` overrides the default one for this run only (e.g. a
        session-scoped retriever over a visitor's uploads). ``on_event``
        receives one progress event per stage (used for streaming responses);
        a failing listener never breaks the run.
        """
        if not query or not query.strip():
            raise InvalidQueryError("query must not be empty")
        notify = _safe(on_event)
        timings: dict[str, float] = {}

        def emit(event: dict[str, Any]) -> None:
            # Stage timings feed the response, the metrics and the listener.
            ms = event.get("ms")
            if ms is not None and event.get("status") != "started":
                stage = str(event["stage"])
                timings[stage] = round(timings.get(stage, 0.0) + ms, 1)
                metrics.STAGE_DURATION.labels(stage).observe(ms / 1000)
            notify(event)

        active = retriever or self.retriever
        ledger = UsageLedger()
        start = time.perf_counter()
        with (
            span("pipeline.run", scrag__retrieval_mode=getattr(active, "mode", None)) as current,
            track_usage(ledger),
        ):
            try:
                result = self._run(query, top_k or self.top_k, active, emit)
            except Exception:
                metrics.PIPELINE_RUNS.labels("error").inc()
                raise
            set_attributes(
                current,
                scrag__status=result.evaluation_status,
                scrag__attempts=result.attempts,
                scrag__best_attempt=result.best_attempt,
            )
        timings["total"] = _ms(start)
        metrics.PIPELINE_RUNS.labels(result.evaluation_status).inc()
        metrics.PIPELINE_ATTEMPTS.observe(result.attempts)
        return result.model_copy(update={"usage": ledger.summary(), "timings_ms": timings})

    def _run(
        self, query: str, top_k: int, retriever: SupportsRetrieve, emit: ProgressListener
    ) -> PipelineResult:
        # 1. Retrieval
        start = time.perf_counter()
        with span("pipeline.retrieve", scrag__top_k=top_k) as current:
            retrieved = retriever.retrieve(query, top_k)
            set_attributes(current, scrag__chunks=len(retrieved))
        emit({"stage": "retrieve", "status": "done", "count": len(retrieved), "ms": _ms(start)})

        # 2. Guardrail filtering (with fallback to raw retrieval)
        start = time.perf_counter()
        with span("pipeline.guardrail", scrag__candidates=len(retrieved)) as current:
            filtered = self.guardrail.filter(query, retrieved) if retrieved else []
            set_attributes(current, scrag__kept=len(filtered))
        emit(
            {
                "stage": "guardrail",
                "status": "done",
                "kept": len(filtered),
                "of": len(retrieved),
                "fallback": bool(retrieved) and not filtered,
                "ms": _ms(start),
            }
        )
        if not filtered:
            logger.info("Guardrail dropped everything; falling back to raw retrieval")
            filtered = retrieved

        if not filtered:
            return PipelineResult(
                query=query,
                answer=NO_CONTEXT_ANSWER,
                evaluation_score=None,
                evaluation_explanation="No context was retrieved, so no answer was evaluated.",
                evaluation_status="skipped",
                attempts=0,
                best_attempt=0,
                used_chunks=[],
            )

        # 3. Generate → evaluate → self-correct.
        history = self._self_correct(query, filtered, emit)
        best = self._select_best(history)
        return PipelineResult(
            query=query,
            answer=best.answer,
            evaluation_score=best.score,
            evaluation_explanation=best.explanation,
            evaluation_status=self._status(best),
            attempts=len(history),
            best_attempt=best.number,
            attempt_history=history,
            used_chunks=self._sort_by_relevance(filtered),
        )

    def _self_correct(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        emit: ProgressListener = lambda event: None,
    ) -> list[Attempt]:
        history: list[Attempt] = []
        previous_answer: str | None = None
        feedback: str | None = None
        for step in range(1, self.max_self_correct_steps + 1):
            emit({"stage": "generate", "status": "started", "attempt": step})
            start = time.perf_counter()
            try:
                with span("pipeline.generate", scrag__attempt=step):
                    answer = self.generator.generate(query, chunks, previous_answer, feedback)
            except LLMError:
                emit({"stage": "generate", "status": "error", "attempt": step, "ms": _ms(start)})
                if not history:
                    raise
                logger.warning("Generation failed on attempt %d; keeping earlier attempts", step)
                break
            emit({"stage": "generate", "status": "done", "attempt": step, "ms": _ms(start)})

            start = time.perf_counter()
            with span("pipeline.evaluate", scrag__attempt=step) as current:
                evaluation = self.evaluator.evaluate(query, answer, chunks)
                set_attributes(current, scrag__score=evaluation.score)
            emit(
                {
                    "stage": "evaluate",
                    "status": evaluation.status,
                    "attempt": step,
                    "score": evaluation.score,
                    "passed": evaluation.ok
                    and evaluation.score is not None
                    and evaluation.score >= self.eval_threshold,
                    "ms": _ms(start),
                }
            )
            history.append(
                Attempt(
                    number=step,
                    answer=answer,
                    score=evaluation.score,
                    explanation=evaluation.explanation,
                    evaluation_status=evaluation.status,
                )
            )
            if not evaluation.ok or evaluation.score is None:
                logger.info("Attempt %d could not be evaluated (%s)", step, evaluation.status)
                break
            logger.info("Attempt %d scored %.2f", step, evaluation.score)
            if evaluation.score >= self.eval_threshold:
                break
            previous_answer, feedback = answer, evaluation.explanation
        return history

    @staticmethod
    def _select_best(history: list[Attempt]) -> Attempt:
        scored = [a for a in history if a.score is not None]
        if scored:
            return max(scored, key=lambda a: a.score or 0.0)  # max() keeps the earliest tie
        return history[-1]

    def _status(self, attempt: Attempt) -> ResultStatus:
        if attempt.score is None:
            return "unavailable"
        return "passed" if attempt.score >= self.eval_threshold else "below_threshold"

    @staticmethod
    def _sort_by_relevance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return sorted(chunks, key=lambda rc: rc.rank_key, reverse=True)
