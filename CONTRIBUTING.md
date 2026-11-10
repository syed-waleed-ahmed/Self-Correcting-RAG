# Contributing

Thanks for your interest in improving Self-Correcting RAG. This guide covers
local setup, the checks CI runs, and how to propose changes.

By taking part you agree to the [code of conduct](CODE_OF_CONDUCT.md).
Security problems go through [private reporting](SECURITY.md), never public
issues.

## Development setup

```bash
git clone https://github.com/syed-waleed-ahmed/Self-Correcting-RAG.git
cd Self-Correcting-RAG
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[onnx,docs,dev]"
cp .env.example .env                 # add GROQ_API_KEY to run real queries
```

The extras:

| Extra | Contents |
| --- | --- |
| `dev` | Test and lint tooling, plus the Postgres and JWT libraries the tests use |
| `onnx` | The lightweight fastembed embedder (what the Docker image uses) |
| `docs` | PDF and DOCX parsing |
| `local` | The torch-based sentence-transformers embedder (same vectors, more memory) |

The test suite needs no embedding model, no API key and no network: it uses
fakes. `.[dev]` alone is enough to run and pass it.

## Checks before opening a PR

CI runs the same checks on Python 3.10, 3.11 and 3.12:

```bash
python scripts/verify.py        # ruff check, ruff format --check, mypy src, pytest
python scripts/verify.py --fix  # apply ruff fixes and formatting first
```

Two groups of tests skip locally unless their dependency is present. CI always
runs both.
- **FAISS:** `pip install -e ".[faiss]"`.
- **Postgres** (sessions, uploads, history, jobs): needs a disposable Postgres
  with pgvector.

  ```bash
  docker run -d --rm -e POSTGRES_PASSWORD=pg -p 5432:5432 pgvector/pgvector:pg17
  export TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres
  pytest tests/test_persistence.py
  ```

  The tests truncate the `scrag` schema's tables, so never point
  `TEST_DATABASE_URL` at a database you care about.

For the web UI (`web/`, Next.js static export):

```bash
cd web
npm ci
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev   # with `scrag serve` running
npm run build && npm run lint                                # what CI runs
```

## What a good change looks like

- **Tests for changed behaviour.** They run offline, with the fake embedder and
  the scriptable fake LLM client in `tests/conftest.py`. Database behaviour
  gets a test in `tests/test_persistence.py`.
- **Small, typed functions.** The package ships a `py.typed` marker, so avoid
  untyped public APIs.
- **Interfaces, not concretions.** New backends implement the relevant
  protocol or ABC (`Embedder`, `VectorStore`, `ChatClient`) and are wired in
  `scrag/container.py`; orchestration code shouldn't need to change.
- **Configuration in one place.** Tunables live in `scrag/config.py`
  (`Settings`) and come from the environment. Document new ones in
  `.env.example` and the README configuration table.
- **Domain errors.** Raise a `ScragError` subclass from `scrag/errors.py`. The
  API maps it to an HTTP status and a client-safe message; internal detail
  stays in the exception text, which is only logged.
- **Schema changes as migrations.** Add a new numbered pair in
  `src/scrag/persistence/sql/` (`.up.sql` and `.down.sql`). Never edit an
  applied migration (ADR 0007).
- **Free-tier friendly.** The public demo runs entirely on free tiers in a
  512 MB container. Don't add a required paid service, and keep an eye on
  memory: the CI `container` job fails above 450 MB.
- **No secrets.** `.env` is git-ignored. Never commit keys, tokens or database
  URLs, including in tests, fixtures or screenshots.

Architecture decisions are recorded in [docs/decisions/](docs/decisions/). A
change that reverses one should add a new ADR explaining why.

## Pull request process

1. Branch from `main` (`feat/...`, `fix/...`, `docs/...`).
2. Make the change, with tests and docs.
3. Make sure `python scripts/verify.py` passes.
4. Open a PR. The template asks what changed, why, and how you tested it. Add
   an entry to [CHANGELOG.md](CHANGELOG.md) under "Unreleased".

Keep PRs focused: one concern per PR is much easier to review than a bundle.

## Reporting bugs and requesting features

Use the issue forms. They ask for the details that make a report actionable:
the smallest reproduction, the version or commit, and the `X-Request-ID` from
the response if you have one. Remove API keys and session tokens first.
