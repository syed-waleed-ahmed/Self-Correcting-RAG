"""Answer-quality comparison: baseline vs hybrid vs self-correcting vs agent.

Every configuration answers the same golden questions with the same models:

* ``baseline``: dense retrieval, one generation, no guardrail or self-check.
* ``hybrid``: hybrid retrieval (BM25 + dense, RRF), one generation.
* ``self_correcting``: the deployed pipeline (hybrid, guardrail, evaluate and
  correct, best attempt returned).
* ``agent``: the bounded agent workflow (plan, tools, write, verify).

Each final answer is then graded once by the same independent judge call
(the configured evaluator model, against the context that configuration used),
and checked for the golden ``answer_contains`` keywords. The judge's tokens are
measured separately from the method's own cost.

Caveats (also stated wherever results are published):
* The judge is an LLM, and the same model family that self-corrects. The
  self-correcting and agent configurations select answers using that judge, so
  its score favours them (an optimisation target, not ground truth). Keyword
  coverage is the model-independent signal.
* Free-tier quotas keep samples small; small samples give indicative numbers.

This spends real LLM quota, paced to stay under a tokens-per-minute budget.
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any

from .errors import LLMRateLimitError, ScragError
from .evaluation import GoldenItem
from .logging_config import get_logger
from .models import RetrievedChunk
from .usage import UsageLedger, track_usage

if TYPE_CHECKING:
    from .container import Container

logger = get_logger(__name__)

CONFIGS = ("baseline", "hybrid", "self_correcting", "agent")

# How the generator is told to decline ("say you are not sure"), plus common
# paraphrases of "the context doesn't say".
_ABSTAIN = re.compile(
    r"not sure|i don['\u2019]t know|i do not know|cannot (?:answer|determine|find)|"
    r"can['\u2019]t (?:answer|determine|find)|could not find|couldn['\u2019]t find|"
    r"(?:does not|doesn['\u2019]t|do not|don['\u2019]t) "
    r"(?:contain|mention|specify|say|state|provide|include)|"
    r"no (?:relevant )?information|not (?:mentioned|provided|specified|stated|included|covered)",
    re.IGNORECASE,
)


def abstained(answer: str) -> bool:
    """Whether an answer declines (says the context doesn't contain the answer)."""
    return bool(_ABSTAIN.search(answer))


class TokenPacer:
    """Keep the trailing-minute token count under ``budget`` (provider TPM limits)."""

    def __init__(
        self,
        budget: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.budget = budget
        self._clock = clock
        self._sleep = sleep
        self._spent: deque[tuple[float, int]] = deque()

    def _used(self, now: float) -> int:
        while self._spent and now - self._spent[0][0] >= 60:
            self._spent.popleft()
        return sum(tokens for _, tokens in self._spent)

    def wait(self, reserve: int) -> None:
        """Block until ``reserve`` more tokens fit in the trailing minute."""
        while self._spent and self._used(self._clock()) + reserve > self.budget:
            oldest = self._spent[0][0]
            self._sleep(max(0.5, 60 - (self._clock() - oldest)))

    def add(self, tokens: int) -> None:
        self._spent.append((self._clock(), tokens))


def stratified(items: list[GoldenItem], n: int) -> list[GoldenItem]:
    """A deterministic sample of ``n`` items: evenly spaced within each question
    kind, kinds interleaved, so a small run still covers lexical and semantic."""
    if n <= 0 or n >= len(items):
        return list(items)
    by_kind: dict[str, list[GoldenItem]] = {}
    for item in items:
        by_kind.setdefault(item.kind, []).append(item)
    share = -(-n // len(by_kind))  # ceil
    picks = []
    for group in by_kind.values():
        step = len(group) / share
        picks.append(
            [group[min(len(group) - 1, int(i * step))] for i in range(min(share, len(group)))]
        )
    out: list[GoldenItem] = []
    for row in zip(*picks, strict=False):
        out.extend(row)
    return out[:n]


def _evidence_chunks(steps: list[Any]) -> list[RetrievedChunk]:
    from .orchestration.workflow import _evidence

    return [RetrievedChunk(chunk=chunk, score=1.0) for _, chunk in _evidence(steps)]


def _answer(container: Container, config: str, question: str, top_k: int) -> tuple[str, list[Any]]:
    """Answer with one configuration; returns the answer and the context it used."""
    from .container import build_retriever

    pipeline = container.pipeline
    assert pipeline is not None
    if config in ("baseline", "hybrid"):
        retriever = build_retriever(
            container.settings,
            container.embedder,
            container.store,
            mode="dense" if config == "baseline" else "hybrid",
        )
        chunks = retriever.retrieve(question, top_k)
        return pipeline.generator.generate(question, chunks), chunks
    if config == "self_correcting":
        result = pipeline.run(question, top_k=top_k)
        return result.answer, result.used_chunks
    if config == "agent":
        assert container.agent is not None
        run = container.agent.run(question)
        return run.answer or "", _evidence_chunks(run.steps)
    raise ValueError(f"unknown configuration {config!r}")


@dataclass
class ComparisonReport:
    questions: int
    configs: list[str]
    judge_model: str
    threshold: float
    top_k: int
    summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    judge_tokens: int = 0


def compare_answers(
    container: Container,
    items: list[GoldenItem],
    configs: tuple[str, ...] = CONFIGS,
    *,
    top_k: int = 3,
    tokens_per_minute: int = 6000,
    reserve_tokens: int = 3500,
    pacer: TokenPacer | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> ComparisonReport:
    pipeline = container.pipeline
    if pipeline is None:
        raise ScragError("answer comparison needs an LLM key")
    judge = pipeline.evaluator
    pacer = pacer or TokenPacer(tokens_per_minute)
    report = ComparisonReport(
        questions=len(items),
        configs=list(configs),
        judge_model=judge.model,
        threshold=pipeline.eval_threshold,
        top_k=top_k,
    )
    for item in items:
        for config in configs:
            row = _measure(container, judge, config, item, top_k, pacer, reserve_tokens, report)
            report.per_question.append(row)
            if on_row is not None:
                on_row(row)
    report.summary = _summarize(report.per_question, configs, report.threshold)
    return report


def _measure(
    container: Container,
    judge: Any,
    config: str,
    item: GoldenItem,
    top_k: int,
    pacer: TokenPacer,
    reserve_tokens: int,
    report: ComparisonReport,
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": item.id, "kind": item.kind, "config": config}
    for attempt in (1, 2):
        pacer.wait(reserve_tokens)
        method, grading = UsageLedger(), UsageLedger()
        try:
            start = time.perf_counter()
            with track_usage(method):
                answer, context = _answer(container, config, item.question, top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            with track_usage(grading):
                verdict = judge.evaluate(item.question, answer, context)
        except LLMRateLimitError:
            pacer.add(method.total_tokens + grading.total_tokens)
            logger.warning("Rate limited on %s/%s (attempt %d)", item.id, config, attempt)
            if attempt == 1:
                pacer.add(10**9)  # force a full minute's cooldown before retrying
                pacer.wait(0)
                continue
            return {**row, "error": "rate_limited"}
        except ScragError as exc:
            return {**row, "error": exc.code}
        pacer.add(method.total_tokens + grading.total_tokens)
        report.judge_tokens += grading.total_tokens
        text = answer.lower()
        needles = [n.lower() for n in item.answer_contains]
        return {
            **row,
            "expect_abstain": item.expect_abstain,
            "abstained": abstained(answer),
            "score": verdict.score,
            "judge_status": verdict.status,
            # Only answerable questions with labelled keywords have a coverage.
            "keyword_coverage": (sum(n in text for n in needles) / len(needles))
            if needles and not item.expect_abstain
            else None,
            "latency_ms": round(latency_ms, 1),
            "tokens": method.total_tokens,
            "llm_calls": method.calls,
        }
    return {**row, "error": "rate_limited"}


def _summarize(
    rows: list[dict[str, Any]], configs: tuple[str, ...], threshold: float
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for config in configs:
        ok = [r for r in rows if r["config"] == config and "error" not in r]
        scored = [r["score"] for r in ok if r["score"] is not None]

        def avg(key: str, subset: list[dict[str, Any]]) -> float | None:
            values = [r[key] for r in subset if r.get(key) is not None]
            return round(mean(values), 4) if values else None

        def rate(subset: list[dict[str, Any]]) -> float | None:
            return round(sum(r["abstained"] for r in subset) / len(subset), 4) if subset else None

        unanswerable = [r for r in ok if r.get("expect_abstain")]
        answerable = [r for r in ok if not r.get("expect_abstain")]

        summary[config] = {
            "answered": len(ok),
            "errors": sum(1 for r in rows if r["config"] == config and "error" in r),
            "mean_judge_score": round(mean(scored), 4) if scored else None,
            "pass_rate": round(sum(s >= threshold for s in scored) / len(scored), 4)
            if scored
            else None,
            "keyword_coverage": avg("keyword_coverage", answerable),
            "keyword_coverage_by_kind": {
                kind: avg("keyword_coverage", [r for r in answerable if r["kind"] == kind])
                for kind in sorted({r["kind"] for r in answerable})
            },
            # Unanswerable questions: share answered with "not in the context".
            "abstain_rate": rate(unanswerable),
            # Answerable questions wrongly declined.
            "false_abstain_rate": rate(answerable),
            "mean_latency_ms": avg("latency_ms", ok),
            "mean_tokens": avg("tokens", ok),
            "mean_llm_calls": avg("llm_calls", ok),
        }
    return summary


def report_to_json(report: ComparisonReport) -> dict[str, Any]:
    return asdict(report)
