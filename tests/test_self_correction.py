"""Regression tests for the self-correction loop's semantics (Phase 1)."""

from __future__ import annotations

import pytest

from scrag.errors import LLMError

from .conftest import FakeChatClient
from .test_pipeline import _pipeline

QUERY = "what is a guardrail agent"


def test_retry_receives_evaluator_feedback(settings, sample_chunks):
    client = FakeChatClient(
        guardrail="0.9",
        generator=["vague answer", "precise answer"],
        evaluator=[
            '{"score": 0.4, "explanation": "does not mention the guardrail threshold"}',
            '{"score": 0.9, "explanation": "good"}',
        ],
    )
    result = _pipeline(settings, client, sample_chunks).run(QUERY)

    retry_prompt = client.prompts_for("generator")[1]
    assert "vague answer" in retry_prompt
    assert "does not mention the guardrail threshold" in retry_prompt
    assert result.answer == "precise answer"
    assert (result.attempts, result.best_attempt) == (2, 2)
    assert result.evaluation_status == "passed"


def test_evaluator_outage_stops_loop_and_reports_unavailable(settings, sample_chunks):
    client = FakeChatClient(
        guardrail="0.9",
        generator=["only answer", "should not be generated"],
        evaluator=LLMError("evaluator down"),
    )
    result = _pipeline(settings, client, sample_chunks).run(QUERY)

    assert result.answer == "only answer"
    assert result.evaluation_score is None  # not a fabricated 0.0
    assert result.evaluation_status == "unavailable"
    assert result.attempts == 1
    assert client.calls.count("generator") == 1  # no blind regeneration
    assert result.attempt_history[0].evaluation_status == "error"


def test_scored_attempt_preferred_over_unevaluated_retry(settings, sample_chunks):
    client = FakeChatClient(
        guardrail="0.9",
        generator=["first", "second"],
        evaluator=['{"score": 0.5, "explanation": "partial"}', LLMError("down")],
    )
    result = _pipeline(settings, client, sample_chunks).run(QUERY)

    assert result.answer == "first"
    assert result.evaluation_score == 0.5
    assert result.evaluation_status == "below_threshold"
    assert (result.attempts, result.best_attempt) == (2, 1)


def test_generator_failure_on_retry_returns_earlier_attempt(settings, sample_chunks):
    client = FakeChatClient(
        guardrail="0.9",
        generator=["first answer", LLMError("provider hiccup")],
        evaluator='{"score": 0.5, "explanation": "partial"}',
    )
    result = _pipeline(settings, client, sample_chunks).run(QUERY)

    assert result.answer == "first answer"
    assert result.attempts == 1


def test_generator_failure_on_first_attempt_propagates(settings, sample_chunks):
    client = FakeChatClient(guardrail="0.9", generator=LLMError("provider down"))
    with pytest.raises(LLMError):
        _pipeline(settings, client, sample_chunks).run(QUERY)


def test_never_exceeds_max_self_correct_steps(settings, sample_chunks):
    settings.max_self_correct_steps = 3
    client = FakeChatClient(
        guardrail="0.9",
        generator="always mediocre",
        evaluator='{"score": 0.2, "explanation": "weak"}',
    )
    result = _pipeline(settings, client, sample_chunks).run(QUERY)

    assert result.attempts == 3
    assert client.calls.count("generator") == 3
    assert result.evaluation_status == "below_threshold"


def test_empty_index_reports_skipped_without_score(settings):
    result = _pipeline(settings, FakeChatClient(), chunks=[]).run("anything")
    assert result.evaluation_status == "skipped"
    assert result.evaluation_score is None
