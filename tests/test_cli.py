"""CLI error handling: operator-friendly messages instead of tracebacks."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from scrag import cli
from scrag.config import get_settings


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    # Point the CLI at temp dirs and ignore any developer .env / API key.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCRAG_DOCS_DIR", str(tmp_path / "missing-docs"))
    monkeypatch.setenv("SCRAG_INDEX_DIR", str(tmp_path / "index"))
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "SCRAG_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_ingest_with_missing_docs_dir_fails_cleanly(isolated_env):
    result = CliRunner().invoke(cli.app, ["ingest"])
    assert result.exit_code == 1
    assert "Documents directory not found" in result.output
    assert "Traceback" not in result.output


def test_query_without_llm_key_fails_cleanly(isolated_env):
    result = CliRunner().invoke(cli.app, ["query", "hello"])
    assert result.exit_code == 1
    assert "No LLM key configured" in result.output


def test_settings_accept_field_names_from_code():
    """Regression: Settings(llm_api_key=...) used to be silently ignored."""
    from scrag.config import Settings

    assert Settings(_env_file=None, llm_api_key="from-code").llm_api_key == "from-code"
