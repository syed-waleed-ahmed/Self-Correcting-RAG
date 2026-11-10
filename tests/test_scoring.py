"""Regression tests for score parsing and agent failure classification (Phase 1)."""

from __future__ import annotations

import pytest

from scrag.agents._parsing import normalize_score
from scrag.agents.evaluator import MAX_EXPLANATION_CHARS, EvaluatorAgent
from scrag.agents.generator import GeneratorAgent
from scrag.agents.guardrail import GuardrailAgent
from scrag.errors import LLMError
from scrag.models import Chunk, RetrievedChunk

from .conftest import FakeChatClient


def _rc(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id="x", source="s.txt", text=text, position=0), score=0.5)


# --- score normalization (defect: "8/10" and "85%" used to clamp to 1.0) ----
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.8", 0.8),
        (".75", 0.75),
        ("85%", 0.85),
        ("Score: 85 %", 0.85),
        ("8/10", 0.8),
        ("7.5 / 10", 0.75),
        ("9 out of 10", 0.9),
        ("0.9 out of 1", 0.9),
        ("150%", 1.0),  # clamped
        ("12/10", 1.0),  # clamped
    ],
)
def test_normalize_score_formats(text, expected):
    assert normalize_score(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "no number here", "3/0", None])
def test_normalize_score_returns_none_when_no_valid_score(text):
    assert normalize_score(text) is None


# --- evaluator failure classification ----------------------------------------
def test_evaluator_llm_failure_is_not_a_low_score():
    agent = EvaluatorAgent(FakeChatClient(evaluator=LLMError("provider down")), model="m")
    result = agent.evaluate("q", "answer", [_rc("ctx")])
    assert result.status == "error"
    assert result.score is None
    assert not result.ok


def test_evaluator_missing_score_key_is_invalid():
    agent = EvaluatorAgent(FakeChatClient(evaluator='{"explanation": "no score"}'), model="m")
    result = agent.evaluate("q", "answer", [_rc("ctx")])
    assert result.status == "invalid"
    assert result.score is None


def test_evaluator_accepts_percentage_scores():
    agent = EvaluatorAgent(
        FakeChatClient(evaluator='{"score": "85%", "explanation": "mostly"}'), model="m"
    )
    result = agent.evaluate("q", "answer", [_rc("ctx")])
    assert result.ok
    assert result.score == pytest.approx(0.85)


def test_evaluator_truncates_long_explanations():
    body = '{"score": 0.5, "explanation": "' + "x" * 5000 + '"}'
    agent = EvaluatorAgent(FakeChatClient(evaluator=body), model="m")
    assert len(agent.evaluate("q", "a", [_rc("ctx")]).explanation) == MAX_EXPLANATION_CHARS


# --- guardrail fails open without fabricating scores --------------------------
def test_guardrail_failure_keeps_chunk_without_fake_score():
    agent = GuardrailAgent(
        FakeChatClient(guardrail=LLMError("boom")), model="m", threshold=0.6, max_workers=1
    )
    kept = agent.filter("q", [_rc("x")])
    assert len(kept) == 1
    assert kept[0].guardrail_score is None  # previously reported as 1.0


def test_guardrail_unparseable_output_keeps_chunk():
    agent = GuardrailAgent(
        FakeChatClient(guardrail="I cannot score this"), model="m", threshold=0.6, max_workers=1
    )
    kept = agent.filter("q", [_rc("x")])
    assert len(kept) == 1
    assert kept[0].guardrail_score is None


# --- generator receives evaluator feedback -------------------------------------
def test_generator_includes_evaluator_feedback_on_retry():
    client = FakeChatClient(generator="revised")
    GeneratorAgent(client, model="m").generate(
        "q", [_rc("ctx")], previous_answer="old answer", feedback="cite the source"
    )
    prompt = client.prompts_for("generator")[0]
    assert "old answer" in prompt
    assert "cite the source" in prompt


def test_generator_first_attempt_has_no_feedback_block():
    client = FakeChatClient(generator="first")
    GeneratorAgent(client, model="m").generate("q", [_rc("ctx")])
    assert "Reviewer feedback" not in client.prompts_for("generator")[0]


# --- reasoning-model output hygiene ----------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.8", "0.8"),
        ("<think>the chunk mentions 0.2 things</think>0.9", "0.9"),
        ("<THINK>\nmulti\nline\n</THINK>\n\nFinal answer.", "Final answer."),
        ("Answer first <think>truncated reasoning", "Answer first"),
    ],
)
def test_strip_reasoning(raw, expected):
    from scrag.llm.client import strip_reasoning

    assert strip_reasoning(raw) == expected


def test_guardrail_token_budget_is_configurable():
    client = FakeChatClient(guardrail="0.9")
    agent = GuardrailAgent(client, model="m", max_tokens=256, max_workers=1)
    assert agent.max_tokens == 256
    assert agent.filter("q", [_rc("x")])[0].guardrail_score == 0.9
