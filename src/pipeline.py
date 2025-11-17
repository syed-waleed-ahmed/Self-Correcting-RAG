from dataclasses import dataclass
from typing import List, Dict, Any

from .retriever import retrieve
from .guardrail_agent import filter_chunks
from .generator_agent import generate_answer
from .evaluator_agent import evaluate_answer
from .config import EVAL_THRESHOLD, MAX_SELF_CORRECT_STEPS


@dataclass
class PipelineResult:
    query: str
    answer: str
    evaluation_score: float
    evaluation_explanation: str
    attempts: int
    used_docs: List[Dict[str, Any]]


class SelfCorrectingRAGPipeline:
    """
    4-stage RAG:
      1. Retrieval
      2. Guardrail relevance filtering
      3. Answer generation
      4. Evaluation + self-correction loop
    """

    def run(self, query: str) -> PipelineResult:
        # 1. Retrieval
        retrieved = retrieve(query)

        # 2. Guardrail filtering
        filtered = filter_chunks(query, retrieved)
        if not filtered:
            # fallback: if guardrail drops everything, use raw retrieved
            filtered = retrieved

        attempts = 0
        answer = None
        score = 0.0
        explanation = ""

        # 3 + 4. Generate → evaluate → self-correct if needed
        while attempts < MAX_SELF_CORRECT_STEPS:
            attempts += 1
            answer = generate_answer(query, filtered, previous_answer=answer)
            score, explanation = evaluate_answer(query, answer, filtered)

            if score >= EVAL_THRESHOLD:
                break  # good enough

        return PipelineResult(
            query=query,
            answer=answer,
            evaluation_score=score,
            evaluation_explanation=explanation,
            attempts=attempts,
            used_docs=filtered,
        )
