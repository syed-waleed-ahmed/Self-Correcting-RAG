"""Evaluator agent: judge whether an answer is grounded in the context.

An evaluation has a ``status`` as well as a score. ``"error"`` (the evaluator
could not be reached) and ``"invalid"`` (no score could be parsed) carry
``score=None``: they say nothing about answer quality and must not be treated as
a low grade.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import LLMError
from ..llm.client import ChatClient
from ..logging_config import get_logger
from ..models import EvaluationStatus, RetrievedChunk
from ._parsing import normalize_score, parse_json_object

logger = get_logger(__name__)

# Explanations are fed back to the generator on retries, so keep them bounded.
MAX_EXPLANATION_CHARS = 1000

_SYSTEM = (
    "You are an evaluator agent. Check whether a candidate answer is factually "
    "supported by the given context. Return JSON with two keys: 'score' (float "
    "between 0.0 and 1.0, higher means better grounded) and 'explanation' (a short "
    "string naming any unsupported or missing points). If the answer contradicts "
    "the context, give a low score (<0.4)."
)


@dataclass(frozen=True)
class Evaluation:
    score: float | None
    explanation: str
    status: EvaluationStatus = "ok"

    @property
    def ok(self) -> bool:
        """True when this evaluation produced a usable quality score."""
        return self.status == "ok" and self.score is not None


class EvaluatorAgent:
    def __init__(self, client: ChatClient, model: str) -> None:
        self.client = client
        self.model = model

    def evaluate(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> Evaluation:
        context = "\n\n".join(rc.chunk.text for rc in chunks)
        user = (
            f"User question:\n{query}\n\nContext:\n{context}\n\n"
            f"Candidate answer:\n{answer}\n\nRespond with the JSON object only."
        )
        try:
            content = self.client.complete(
                model=self.model,
                system=_SYSTEM,
                user=user,
                temperature=0.0,
                json_mode=True,
            )
        except LLMError as exc:
            logger.warning(
                "Evaluator call failed (%s); answer left unevaluated", type(exc).__name__
            )
            return Evaluation(
                score=None,
                explanation="The evaluator was unavailable, so the answer was not evaluated.",
                status="error",
            )

        obj = parse_json_object(content)
        score = normalize_score(str(obj.get("score", ""))) if obj is not None else None
        if obj is None or score is None:
            logger.warning("Evaluator returned output without a parseable score")
            return Evaluation(
                score=None,
                explanation="The evaluator's output could not be parsed into a score.",
                status="invalid",
            )
        explanation = str(obj.get("explanation", "")).strip()[:MAX_EXPLANATION_CHARS]
        return Evaluation(score=score, explanation=explanation)
