"""Generator agent: produce an answer grounded strictly in the given context."""

from __future__ import annotations

from ..llm.client import ChatClient
from ..models import RetrievedChunk

_SYSTEM = (
    "You are a careful assistant that answers strictly using the provided context. "
    "Never invent facts. If the context does not contain the answer, say you are "
    "not sure instead of guessing."
)

MAX_FEEDBACK_CHARS = 1000


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, source-attributed context block."""
    return "\n\n".join(
        f"[Document {i + 1} | source: {rc.chunk.source}]\n{rc.chunk.text}"
        for i, rc in enumerate(chunks)
    )


class GeneratorAgent:
    def __init__(self, client: ChatClient, model: str, temperature: float = 0.2) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        previous_answer: str | None = None,
        feedback: str | None = None,
    ) -> str:
        """Generate an answer; on retries, revise ``previous_answer`` using ``feedback``.

        ``feedback`` is the evaluator's explanation of what was wrong with the
        previous answer. Without it the model would be asked to "improve" an
        answer with no idea what to change.
        """
        context = build_context(chunks)
        if previous_answer:
            critique = (feedback or "").strip()[:MAX_FEEDBACK_CHARS]
            critique_block = (
                f"Reviewer feedback on that answer:\n{critique}\n\n" if critique else ""
            )
            user = (
                f"User question:\n{query}\n\nRelevant context:\n{context}\n\n"
                f"A previous answer was judged low quality:\n{previous_answer}\n\n"
                f"{critique_block}"
                "Provide a corrected, improved answer grounded strictly in the context, "
                "addressing the feedback. If the context lacks the answer, say you are not sure."
            )
        else:
            user = (
                f"User question:\n{query}\n\nRelevant context:\n{context}\n\n"
                "Provide a helpful answer grounded strictly in the context above. "
                "If the context lacks the information, say you are not sure."
            )
        return self.client.complete(
            model=self.model,
            system=_SYSTEM,
            user=user,
            temperature=self.temperature,
        )
