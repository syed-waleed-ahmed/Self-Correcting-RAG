"""Run every quality gate CI runs, in the same order: one command, one exit code.

Usage (from the repository root, inside the project's virtual environment):

    python scripts/verify.py          # lint, format check, type check, tests
    python scripts/verify.py --fix    # apply ruff fixes and formatting first

All checks run even if an earlier one fails, so a single run reports everything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="apply ruff fixes and formatting")
    args = parser.parse_args()

    py = sys.executable
    steps: list[tuple[str, list[str]]] = []
    if args.fix:
        steps += [
            ("ruff fix", [py, "-m", "ruff", "check", "--fix", "."]),
            ("ruff format", [py, "-m", "ruff", "format", "."]),
        ]
    steps += [
        ("lint", [py, "-m", "ruff", "check", "."]),
        ("format", [py, "-m", "ruff", "format", "--check", "."]),
        ("types", [py, "-m", "mypy", "src", "scripts", "loadtests"]),
        ("tests", [py, "-m", "pytest", "-q"]),
    ]

    results: list[tuple[str, bool, float]] = []
    for name, cmd in steps:
        print(f"\n=== {name}: {' '.join(cmd[2:])}", flush=True)
        start = time.perf_counter()
        ok = subprocess.run(cmd, cwd=ROOT).returncode == 0
        results.append((name, ok, time.perf_counter() - start))

    print("\n=== summary")
    for name, ok, seconds in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<12} {seconds:5.1f}s")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
