"""Offline evaluation against a golden dataset.

Retrieval metrics are computed at the *source document* level: the top-k
retrieved chunks are mapped to their source files (first occurrence wins) and
compared with each question's ``relevant_sources``. These metrics need no LLM,
so they are free and deterministic for a given index.

Answer evaluation (optional) runs the full pipeline, so it spends LLM quota. It
reports the evaluator's verdicts and a simple keyword-coverage check against
``answer_contains``. Both are heuristics: keyword coverage is not correctness,
and the evaluator is an LLM judge.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from .pipeline import SelfCorrectingRAGPipeline
from .retriever import Retriever


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    relevant_sources: tuple[str, ...]
    answer_contains: tuple[str, ...] = ()
    kind: str = "unspecified"
    # True for questions the corpus cannot answer: the right answer is to say so.
    expect_abstain: bool = False


def load_golden(path: Path | str) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            items.append(
                GoldenItem(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    relevant_sources=tuple(raw["relevant_sources"]),
                    answer_contains=tuple(raw.get("answer_contains", ())),
                    kind=str(raw.get("kind", "unspecified")),
                    expect_abstain=bool(raw.get("expect_abstain", False)),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid golden item ({exc})") from exc
    if not items:
        raise ValueError(f"{path}: no golden items")
    return items


# --- metrics ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, source in enumerate(ranked, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ranked[:k]) if s in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def _unique_sources(sources: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for s in sources:
        seen.setdefault(s, None)
    return list(seen)


@dataclass
class RetrievalReport:
    mode: str
    k: int
    questions: int
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_latency_ms: float
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)


def evaluate_retrieval(retriever: Retriever, items: list[GoldenItem], k: int) -> RetrievalReport:
    rows: list[dict[str, Any]] = []
    for item in items:
        start = time.perf_counter()
        results = retriever.retrieve(item.question, top_k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        ranked = _unique_sources([rc.chunk.source for rc in results])
        relevant = set(item.relevant_sources)
        rows.append(
            {
                "id": item.id,
                "kind": item.kind,
                "question": item.question,
                "retrieved_sources": ranked,
                "relevant_sources": sorted(relevant),
                "recall": recall_at_k(ranked, relevant, k),
                "hit": hit_at_k(ranked, relevant, k),
                "rr": reciprocal_rank(ranked, relevant),
                "ndcg": ndcg_at_k(ranked, relevant, k),
                "latency_ms": round(latency_ms, 2),
            }
        )

    def summary(subset: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "questions": len(subset),
            "recall_at_k": round(mean(r["recall"] for r in subset), 4),
            "hit_rate_at_k": round(mean(r["hit"] for r in subset), 4),
            "mrr": round(mean(r["rr"] for r in subset), 4),
            "ndcg_at_k": round(mean(r["ndcg"] for r in subset), 4),
        }

    overall = summary(rows)
    kinds = sorted({r["kind"] for r in rows})
    return RetrievalReport(
        mode=getattr(retriever, "mode", type(retriever).__name__),
        k=k,
        questions=len(rows),
        recall_at_k=overall["recall_at_k"],
        hit_rate_at_k=overall["hit_rate_at_k"],
        mrr=overall["mrr"],
        ndcg_at_k=overall["ndcg_at_k"],
        mean_latency_ms=round(mean(r["latency_ms"] for r in rows), 2),
        by_kind={kind: summary([r for r in rows if r["kind"] == kind]) for kind in kinds},
        per_question=rows,
    )


@dataclass
class AnswerReport:
    questions: int
    status_counts: dict[str, int]
    mean_score: float | None
    mean_attempts: float
    corrected: int
    keyword_coverage: float
    mean_latency_ms: float
    errors: int
    per_question: list[dict[str, Any]] = field(default_factory=list)


def evaluate_answers(pipeline: SelfCorrectingRAGPipeline, items: list[GoldenItem]) -> AnswerReport:
    """Run the full pipeline per question (spends LLM quota)."""
    rows: list[dict[str, Any]] = []
    for item in items:
        start = time.perf_counter()
        try:
            result = pipeline.run(item.question)
        except Exception as exc:  # noqa: BLE001 - record and continue the run
            rows.append({"id": item.id, "error": type(exc).__name__})
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        answer = result.answer.lower()
        needles = [n.lower() for n in item.answer_contains]
        coverage = sum(n in answer for n in needles) / len(needles) if needles else 1.0
        first = result.attempt_history[0].score if result.attempt_history else None
        rows.append(
            {
                "id": item.id,
                "status": result.evaluation_status,
                "score": result.evaluation_score,
                "first_attempt_score": first,
                "attempts": result.attempts,
                "best_attempt": result.best_attempt,
                "keyword_coverage": coverage,
                "latency_ms": round(latency_ms, 1),
            }
        )

    ok = [r for r in rows if "error" not in r]
    scores = [r["score"] for r in ok if r["score"] is not None]
    counts: dict[str, int] = {}
    for r in ok:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return AnswerReport(
        questions=len(rows),
        status_counts=counts,
        mean_score=round(mean(scores), 4) if scores else None,
        mean_attempts=round(mean(r["attempts"] for r in ok), 3) if ok else 0.0,
        corrected=sum(1 for r in ok if r["best_attempt"] > 1),
        keyword_coverage=round(mean(r["keyword_coverage"] for r in ok), 4) if ok else 0.0,
        mean_latency_ms=round(mean(r["latency_ms"] for r in ok), 1) if ok else 0.0,
        errors=len(rows) - len(ok),
        per_question=rows,
    )


def to_jsonable(report: RetrievalReport | AnswerReport) -> dict[str, Any]:
    return asdict(report)
