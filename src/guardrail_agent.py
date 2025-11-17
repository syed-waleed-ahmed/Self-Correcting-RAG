from typing import List, Dict

from .llm_client import client
from .config import GENERATOR_MODEL, GUARDRAIL_THRESHOLD


def score_relevance(query: str, chunk: str) -> float:
    """
    Ask the LLM to score how relevant chunk is to query (0.0–1.0).
    """
    system_prompt = (
        "You are a guardrail agent. Given a user query and a document chunk, "
        "you must output ONLY a floating point number between 0.0 and 1.0 "
        "indicating how relevant the chunk is to answering the query. "
        "0.0 means not relevant at all. 1.0 means perfectly relevant."
    )

    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Document chunk:\n{chunk}\n\n"
        "Relevance score (0.0-1.0):"
    )

    resp = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=5,
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()
    try:
        value = float(content)
    except ValueError:
        value = 0.0

    return max(0.0, min(1.0, value))


def filter_chunks(query: str, retrieved_docs: List[Dict]) -> List[Dict]:
    """
    For each retrieved document, ask the guardrail agent for a relevance score.
    Keep only those above GUARDRAIL_THRESHOLD.
    """
    filtered = []
    for doc in retrieved_docs:
        score = score_relevance(query, doc["text"])
        doc["guardrail_score"] = score
        if score >= GUARDRAIL_THRESHOLD:
            filtered.append(doc)

    return filtered
