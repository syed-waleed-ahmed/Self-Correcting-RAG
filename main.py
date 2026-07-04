"""Backward-compatible entry point.

Prefer installing the package (``pip install -e .``) and using the ``scrag``
command. This shim lets ``python main.py <command>`` work straight from a clone
by putting ``src/`` on the path and delegating to the Typer CLI.

Examples:
    python main.py ingest
    python main.py query "What is a guardrail agent?"
    python main.py serve
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scrag.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
