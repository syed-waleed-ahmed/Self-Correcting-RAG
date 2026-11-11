"""Turn Locust ``--csv`` output into a Markdown table (for docs and CI summaries).

Usage::

    python loadtests/summarize.py out/capacity_stats.csv [out/capacity_failures.csv]

Stdlib only.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def _ms(value: str) -> str:
    try:
        return f"{float(value):,.0f}"
    except ValueError:
        return value or "-"


def stats_table(path: Path) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    lines = [
        "| Endpoint | Requests | Failures | Req/s | Median ms | p95 ms | p99 ms | Max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        name = row["Name"] if row["Name"] != "Aggregated" else "**All**"
        lines.append(
            f"| {name} | {int(row['Request Count']):,} | {int(row['Failure Count']):,}"
            f" | {float(row['Requests/s']):.1f} | {_ms(row['Median Response Time'])}"
            f" | {_ms(row['95%'])} | {_ms(row['99%'])} | {_ms(row['Max Response Time'])} |"
        )
    return "\n".join(lines)


def failures_table(path: Path) -> str:
    counts: Counter[tuple[str, str]] = Counter()
    for row in csv.DictReader(path.open(encoding="utf-8")):
        counts[(row["Name"], row["Error"][:80])] += int(row["Occurrences"])
    if not counts:
        return "No failed requests."
    lines = ["| Endpoint | Error | Count |", "| --- | --- | ---: |"]
    for (name, error), count in counts.most_common():
        lines.append(f"| {name} | {error} | {count:,} |")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    print(stats_table(Path(sys.argv[1])))
    if len(sys.argv) > 2:
        print()
        print(failures_table(Path(sys.argv[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
