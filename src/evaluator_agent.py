from typing import List, Dict, Tuple
import json

from .llm_client import client
from .config import EVALUATOR_MODEL


def evaluate_answer(
    query: str,
    answer: str,
    context_chunks: List[Dict],
) -> Tuple[float, str]:
    """
    Ask the evaluator LLM to judge factual consistency of the answer with the context.
    Returns (score_between_0_and_1, explanation_string).
    """
    context_text = "\n\n".join([c["text"] for c in context_chunks])

    system_prompt = (
        "You are an evaluator agent. Your job is to check whether a candidate answer "
        "is factually supported by the given context.\n\n"
        "Return JSON with two keys: 'score' (float between 0.0 and 1.0) and "
        "'explanation' (short string). Higher score means better factual grounding.\n"
        "If the answer contradicts the context, give a low score (<0.4)."
    )

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Now respond with the JSON object only."
    )

    resp = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content

    try:
        obj = json.loads(content)
        score = float(obj.get("score", 0.0))
        explanation = str(obj.get("explanation", ""))
    except Exception:
        score = 0.0
        explanation = "Failed to parse evaluator output."

    score = max(0.0, min(1.0, score))
    return score, explanation
