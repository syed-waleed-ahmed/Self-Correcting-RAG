"""The release version is written in several files; they must all agree.

0.5.0 shipped with ``CITATION.cff`` still at 0.4.0, so GitHub's "Cite this
repository" gave the old version. Bump every file below in a release PR.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import scrag

ROOT = Path(__file__).resolve().parents[1]


def _match(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    found = re.search(pattern, text, re.MULTILINE)
    assert found, f"no version found in {path}"
    return found.group(1)


def _declared_versions() -> dict[str, str]:
    lock = json.loads((ROOT / "web/package-lock.json").read_text(encoding="utf-8"))
    return {
        "pyproject.toml": _match("pyproject.toml", r'^version = "([^"]+)"'),
        "CITATION.cff": _match("CITATION.cff", r'^version: "?([^"\s]+)"?'),
        "web/package.json": json.loads((ROOT / "web/package.json").read_text(encoding="utf-8"))[
            "version"
        ],
        "web/package-lock.json": lock["version"],
        "web/package-lock.json (root package)": lock["packages"][""]["version"],
        # The newest released section, right after "Unreleased".
        "CHANGELOG.md": _match("CHANGELOG.md", r"^## \[(\d+\.\d+\.\d+)\]"),
    }


@pytest.mark.parametrize("source", list(_declared_versions()))
def test_version_matches_package(source: str) -> None:
    assert _declared_versions()[source] == scrag.__version__
