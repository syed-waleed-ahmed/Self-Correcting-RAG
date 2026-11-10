"""Phase 6: the answer-quality comparison harness (fakes only, no network)."""

from __future__ import annotations

import json

from scrag.container import build_container
from scrag.evaluation import GoldenItem, load_golden
from scrag.evaluation_compare import CONFIGS, TokenPacer, compare_answers, stratified
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeEmbedder
from .test_observability import MeteredChat

PLAN = json.dumps({"steps": [{"tool": "document_search", "input": {"query": "guardrail"}}]})


def _container(settings, sample_chunks, chat):
    embedder = FakeEmbedder()
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    store.add(embedder.embed([c.text for c in sample_chunks]), sample_chunks)
    return build_container(settings, embedder=embedder, store=store, chat_client=chat)


def test_every_configuration_answers_and_is_graded_by_the_same_judge(settings, sample_chunks):
    chat = MeteredChat(
        generator=[
            "The guardrail can filter off-topic context.",  # baseline
            "The guardrail can filter off-topic context.",  # hybrid
            "A guardrail agent can filter context.",  # self_correcting
            PLAN,  # agent: plan
            "It filters off-topic context [S1].",  # agent: writer
        ],
        evaluator='{"score": 0.9, "explanation": "supported"}',
    )
    container = _container(settings, sample_chunks, chat)
    item = GoldenItem(
        id="g1",
        question="What does a guardrail agent filter?",
        relevant_sources=("doc1.txt",),
        answer_contains=("filter", "off-topic"),
        kind="semantic",
    )
    report = compare_answers(container, [item], CONFIGS, pacer=TokenPacer(10**9))

    assert [r["config"] for r in report.per_question] == list(CONFIGS)
    by_config = {r["config"]: r for r in report.per_question}
    assert all("error" not in r for r in by_config.values())
    assert by_config["baseline"]["llm_calls"] == 1  # one generation, nothing else
    assert by_config["self_correcting"]["llm_calls"] >= 3  # guardrail + generate + evaluate
    assert by_config["agent"]["llm_calls"] >= 3  # plan + write + verify
    assert by_config["self_correcting"]["keyword_coverage"] == 0.5  # "off-topic" missing
    assert by_config["baseline"]["keyword_coverage"] == 1.0
    # The judge's tokens are measured apart from each method's own cost.
    assert report.judge_tokens == 4 * 15
    assert by_config["baseline"]["tokens"] == 15
    summary = report.summary["agent"]
    assert summary["answered"] == 1 and summary["mean_judge_score"] == 0.9
    assert summary["pass_rate"] == 1.0


def test_token_pacer_waits_for_the_trailing_minute():
    now = [0.0]
    slept: list[float] = []
    pacer = TokenPacer(
        1000,
        clock=lambda: now[0],
        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)),
    )
    pacer.wait(600)
    pacer.add(600)
    assert slept == []
    now[0] = 10.0
    pacer.wait(600)  # 600 + 600 > 1000: wait until the first spend leaves the window
    assert slept == [50.0] and now[0] == 60.0


def test_stratified_sample_covers_both_kinds_deterministically():
    items = load_golden("data/eval/golden.jsonl")
    sample = stratified(items, 8)
    assert len(sample) == 8 and len({i.id for i in sample}) == 8
    assert {i.kind for i in sample} == {"lexical", "semantic"}
    assert [i.kind for i in sample][:2] in (["semantic", "lexical"], ["lexical", "semantic"])
    assert stratified(items, 8) == sample
    assert stratified(items, 0) == items


def test_cli_compare_runs_end_to_end(settings, sample_chunks, monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from scrag import cli

    chat = MeteredChat(generator="The guardrail filters off-topic context.")
    container = _container(settings, sample_chunks, chat)
    container.store.save()  # `scrag eval` loads the persisted index
    monkeypatch.setattr(cli, "_bootstrap", lambda: settings)
    monkeypatch.setattr(cli, "build_container", lambda s: container)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "g1",
                "question": "What does a guardrail filter?",
                "relevant_sources": ["doc1.txt"],
                "answer_contains": ["filter"],
                "kind": "semantic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "eval",
            "--golden",
            str(golden),
            "--mode",
            "dense",
            "--compare",
            "1",
            "--configs",
            "baseline,hybrid",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "baseline" in result.output and "hybrid" in result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["comparison"]["summary"]) == {"baseline", "hybrid"}


def test_abstention_is_detected_and_scored_separately_from_keywords(settings, sample_chunks):
    from scrag.evaluation_compare import abstained

    assert abstained("I am not sure: the context does not mention it.")
    assert abstained("The provided context doesn't specify the year.")
    assert not abstained("The paper was published in 2009.")

    chat = MeteredChat(
        generator=["I am not sure; the context does not say.", "Canberra."],
        evaluator='{"score": 1.0, "explanation": "grounded"}',
    )
    container = _container(settings, sample_chunks, chat)
    item = GoldenItem(
        id="u1",
        question="What is the capital of Australia?",
        relevant_sources=(),
        kind="unanswerable",
        expect_abstain=True,
    )
    report = compare_answers(container, [item, item], ("baseline",), pacer=TokenPacer(10**9))
    rows = report.per_question
    assert [r["abstained"] for r in rows] == [True, False]
    assert all(r["keyword_coverage"] is None for r in rows)  # not applicable
    summary = report.summary["baseline"]
    assert summary["abstain_rate"] == 0.5 and summary["false_abstain_rate"] is None
    assert summary["keyword_coverage"] is None
