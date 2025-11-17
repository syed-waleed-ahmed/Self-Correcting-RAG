from typing import List, Dict

from .llm_client import client
from .config import GENERATOR_MODEL


def build_context_snippet(chunks: List[Dict]) -> str:
    """Concatenate chunk texts into a single context string."""
    texts = [
        f"[Document {i+1} from {c['filename']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    return "\n\n".join(texts)


def generate_answer(
    query: str,
    context_chunks: List[Dict],
    previous_answer: str | None = None,
) -> str:
    """
    Use the generator model to produce an answer based on the context.
    If previous_answer is provided, ask the model to improve upon it.
    """
    context_text = build_context_snippet(context_chunks)

    if previous_answer:
        user_prompt = (
            f"User question:\n{query}\n\n"
            f"Relevant context:\n{context_text}\n\n"
            f"Previous answer that was judged low quality:\n{previous_answer}\n\n"
            "Please provide a corrected and improved answer that is strictly grounded "
            "in the context. If the context does not contain the answer, say you are "
            "not sure instead of guessing."
        )
    else:
        user_prompt = (
            f"User question:\n{query}\n\n"
            f"Relevant context:\n{context_text}\n\n"
            "Provide a helpful answer that is strictly grounded in the context above. "
            "If the context does not contain the information, say you are not sure."
        )

    resp = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful assistant that only uses the given context.",
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()
