"""Evaluator agent: judge whether an answer is grounded in the context."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.client import ChatClient, LLMError
from ..logging_config import get_logger
from ..models import RetrievedChunk
from ._parsing import parse_json_object, parse_score

logger = get_logger(__name__)

_SYSTEM = (
    "You are an evaluator agent. Check whether a candidate answer is factually "
    "supported by the given context. Return JSON with two keys: 'score' (float "
    "between 0.0 and 1.0, higher means better grounded) and 'explanation' (a short "
    "string). If the answer contradicts the context, give a low score (<0.4)."
)


@dataclass
class Evaluation:
    score: float
    explanation: str


class EvaluatorAgent:
    def __init__(self, client: ChatClient, model: str) -> None:
        self.client = client
        self.model = model

    def evaluate(
        self, query: str, answer: str, chunks: list[RetrievedChunk]
    ) -> Evaluation:
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
        except LLMError:
            logger.warning("Evaluator call failed; returning neutral score")
            return Evaluation(score=0.0, explanation="Evaluator unavailable.")

        obj = parse_json_object(content)
        if obj is None:
            return Evaluation(score=0.0, explanation="Failed to parse evaluator output.")
        score = parse_score(str(obj.get("score", 0.0)))
        explanation = str(obj.get("explanation", "")).strip()
        return Evaluation(score=score, explanation=explanation)
