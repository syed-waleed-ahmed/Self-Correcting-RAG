"""Paired bootstrap of hybrid minus dense retrieval, from `main.py eval --output` reports.

The averages in docs/evaluation.md say which mode scored higher; this says how
sure that is. Both modes answer the same 60 questions, so the comparison is
paired: resample the questions with replacement, take the mean per-question
difference each time, and report the 2.5th and 97.5th percentiles.

    python scripts/paired_bootstrap.py data/eval/reports/retrieval-k5.json

Standard library only, so it runs without the project installed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import fmean

RESAMPLES = 10_000
METRICS = ("recall", "hit", "rr", "ndcg")


def interval(deltas: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(fmean(rng.choices(deltas, k=n)) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def compare(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    dense = {row["id"]: row for row in report["retrieval_dense"]["per_question"]}
    hybrid = {row["id"]: row for row in report["retrieval_hybrid"]["per_question"]}
    ids = sorted(dense)
    print(f"{path.name}: {len(ids)} questions, k = {report['retrieval_dense']['k']}")
    for metric in METRICS:
        deltas = [hybrid[i][metric] - dense[i][metric] for i in ids]
        low, high = interval(deltas)
        better = sum(d > 1e-12 for d in deltas)
        worse = sum(d < -1e-12 for d in deltas)
        print(
            f"  {metric:<6} dense {fmean(dense[i][metric] for i in ids):.3f}"
            f"  hybrid {fmean(hybrid[i][metric] for i in ids):.3f}"
            f"  difference {fmean(deltas):+.3f}  95% CI [{low:+.3f}, {high:+.3f}]"
            f"  better {better}, worse {worse}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/paired_bootstrap.py REPORT.json [REPORT.json ...]")
    for argument in sys.argv[1:]:
        compare(Path(argument))
