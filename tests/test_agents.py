from __future__ import annotations

from scrag.agents._parsing import parse_json_object, parse_score
from scrag.agents.evaluator import EvaluatorAgent
from scrag.agents.guardrail import GuardrailAgent
from scrag.models import Chunk, RetrievedChunk

from .conftest import FakeChatClient


def _rc(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id="x", source="s.txt", text=text, position=0), score=score
    )


# --- parsing helpers -------------------------------------------------------
def test_parse_score_variants():
    assert parse_score("0.8") == 0.8
    assert parse_score("Relevance: 0.9 out of 1") == 0.9
    assert parse_score("nonsense") == 0.0
    assert parse_score("1.7") == 1.0  # clamped
    assert parse_score("-0.5") == 0.0  # clamped


def test_parse_json_object_tolerates_prose_and_fences():
    assert parse_json_object('{"score": 0.5}') == {"score": 0.5}
    assert parse_json_object('Here you go:\n```json\n{"score": 0.4}\n```') == {"score": 0.4}
    assert parse_json_object("not json at all") is None


# --- guardrail agent -------------------------------------------------------
def test_guardrail_filters_below_threshold():
    client = FakeChatClient(guardrail=["0.9", "0.1"])
    agent = GuardrailAgent(client, model="m", threshold=0.6, max_workers=1)
    kept = agent.filter("q", [_rc("relevant"), _rc("irrelevant")])
    assert [rc.chunk.text for rc in kept] == ["relevant"]
    # Scores are recorded on all chunks, even dropped ones is not required,
    # but kept ones must carry their guardrail score.
    assert kept[0].guardrail_score == 0.9


def test_guardrail_keeps_chunk_when_llm_fails():
    class BoomClient(FakeChatClient):
        def complete(self, **kwargs):
            from scrag.llm.client import LLMError

            raise LLMError("boom")

    agent = GuardrailAgent(BoomClient(), model="m", threshold=0.6, max_workers=1)
    kept = agent.filter("q", [_rc("x")])
    assert len(kept) == 1  # failure must not starve the generator of context


# --- evaluator agent -------------------------------------------------------
def test_evaluator_parses_score_and_explanation():
    client = FakeChatClient(evaluator='{"score": 0.82, "explanation": "grounded"}')
    agent = EvaluatorAgent(client, model="m")
    result = agent.evaluate("q", "answer", [_rc("ctx")])
    assert result.score == 0.82
    assert result.explanation == "grounded"


def test_evaluator_handles_unparseable_output():
    client = FakeChatClient(evaluator="totally not json")
    agent = EvaluatorAgent(client, model="m")
    result = agent.evaluate("q", "answer", [_rc("ctx")])
    assert result.score == 0.0
