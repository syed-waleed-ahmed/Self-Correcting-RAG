# Contributing

Thanks for your interest in improving Self-Correcting RAG! This guide covers the
local setup, the checks we run, and how to propose changes.

## Development setup

```bash
git clone https://github.com/syed-waleed-ahmed/Self-Correcting-RAG.git
cd Self-Correcting-RAG
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[local,dev]"
```

`.[dev]` installs the test and lint tooling; `.[local]` adds the sentence-
transformers + torch stack used by the real embedder. The test suite itself
needs neither torch nor an API key (it uses fakes), so `.[dev]` alone is enough
to run and pass the tests.

## Checks before opening a PR

All of these must pass - CI runs the same on Python 3.10, 3.11 and 3.12:

```bash
python scripts/verify.py        # ruff check, ruff format --check, mypy src, pytest
python scripts/verify.py --fix  # apply ruff fixes and formatting first
```

Please also:

- Add or update tests for any behaviour you change. The suite runs offline with
  a fake embedder and a scriptable fake LLM client (see `tests/conftest.py`) -
  new features should be testable the same way, without network calls.
- Keep functions small and typed. The package ships a `py.typed` marker; avoid
  introducing untyped public APIs.
- Never commit secrets. `.env` is git-ignored; document new settings in
  `.env.example` and the README configuration table instead.

## Coding conventions

- **Style / linting:** [Ruff](https://github.com/astral-sh/ruff) (config in
  `pyproject.toml`, line length 100).
- **Architecture:** depend on interfaces, not concretions. New backends should
  implement the relevant protocol/ABC (`Embedder`, `VectorStore`, `ChatClient`)
  and be wired in `scrag/container.py` - orchestration code must not change.
- **Config:** all tunables live in `scrag/config.py` (`Settings`) and are
  environment-driven; don't hard-code values in modules.
- **Errors:** raise a `ScragError` subclass (`scrag/errors.py`) from the core;
  the API layer maps it to its HTTP status and client-safe `public_message`.
  Internal detail belongs in the exception text, which is only logged.

## Pull request process

1. Branch from `main` (e.g. `feat/...`, `fix/...`, `docs/...`).
2. Make your change with tests and docs.
3. Ensure `python scripts/verify.py` is green.
4. Open a PR describing the motivation and the change. Update
   [CHANGELOG.md](CHANGELOG.md) under the "Unreleased" heading.

## Reporting bugs

Open an issue with the smallest reproduction you can: the query/config used, the
observed behaviour, and what you expected. Redact any API keys.
