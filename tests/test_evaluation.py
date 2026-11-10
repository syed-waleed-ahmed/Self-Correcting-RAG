"""Golden-set evaluation harness (Phase 2)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scrag.evaluation import (
    evaluate_answers,
    evaluate_retrieval,
    hit_at_k,
    load_golden,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scrag.loaders import SUPPORTED_SUFFIXES
from scrag.models import Chunk, RetrievedChunk

from .conftest import FakeChatClient
from .test_pipeline import _pipeline

ROOT = Path(__file__).resolve().parent.parent


def test_metric_math():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b", "z"}, 3) == 0.5
    assert hit_at_k(ranked, {"c"}, 2) == 0.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))


class _ScriptedRetriever:
    mode = "scripted"

    def __init__(self, ranking: dict[str, list[str]]):
        self.ranking = ranking

    def retrieve(self, query, top_k=None):
        return [
            RetrievedChunk(chunk=Chunk(id=f"{s}{i}", source=s, text=s), score=1.0)
            for i, s in enumerate(self.ranking[query])
        ]


def test_evaluate_retrieval_dedupes_sources_per_question(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"id": "1", "question": "q1", "relevant_sources": ["a"], "kind": "lexical"}\n'
        '{"id": "2", "question": "q2", "relevant_sources": ["b"], "kind": "semantic"}\n',
        encoding="utf-8",
    )
    retriever = _ScriptedRetriever({"q1": ["a", "a", "c"], "q2": ["c", "c", "b"]})
    report = evaluate_retrieval(retriever, load_golden(golden), k=2)  # type: ignore[arg-type]
    rows = {r["id"]: r for r in report.per_question}
    assert rows["1"]["retrieved_sources"] == ["a", "c"]  # duplicate chunk source collapsed
    assert rows["2"]["rr"] == 0.5  # "b" is the 2nd unique source
    assert report.by_kind["lexical"]["mrr"] == 1.0
    assert report.mrr == 0.75


def test_load_golden_rejects_bad_lines(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="g.jsonl:1"):
        load_golden(golden)


def test_evaluate_answers_counts_corrections_and_coverage(settings, sample_chunks, tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"id": "1", "question": "what is a guardrail agent", "relevant_sources": ["x"],'
        ' "answer_contains": ["filter", "missing"]}\n',
        encoding="utf-8",
    )
    client = FakeChatClient(
        guardrail="0.9",
        generator=["draft", "a guardrail agent can filter context"],
        evaluator=['{"score": 0.4, "explanation": "vague"}', '{"score": 0.9, "explanation": "ok"}'],
    )
    report = evaluate_answers(_pipeline(settings, client, sample_chunks), load_golden(golden))
    assert report.corrected == 1
    assert report.keyword_coverage == 0.5
    assert report.status_counts == {"passed": 1}


def test_repository_golden_set_is_consistent():
    """Every golden label must point at a real, ingestable sample document."""
    items = load_golden(ROOT / "data" / "eval" / "golden.jsonl")
    docs = {p.name for p in (ROOT / "data" / "docs").iterdir() if p.suffix in SUPPORTED_SUFFIXES}
    assert len(items) >= 40
    assert len({i.id for i in items}) == len(items)
    for item in items:
        assert set(item.relevant_sources) <= docs, item.id
