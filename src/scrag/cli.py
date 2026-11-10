"""Command-line interface (Typer): ingest, query, serve, info."""

from __future__ import annotations

from typing import NoReturn

import typer

from .config import Settings, get_settings
from .container import build_container
from .errors import ScragError
from .logging_config import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    help="Self-Correcting RAG — multi-agent retrieval-augmented generation.",
    no_args_is_help=True,
)
logger = get_logger(__name__)


def _bootstrap() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    return settings


def _fail(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def ingest(
    reset: bool = typer.Option(
        True, help="Rebuild the index from scratch (--no-reset upserts changed documents)."
    ),
) -> None:
    """Build the vector index from documents in the configured docs directory."""
    settings = _bootstrap()
    container = build_container(settings)
    if not reset:
        container.load_index()  # upsert against what is already persisted
    try:
        stats = container.run_ingest(reset=reset)
    except ScragError as exc:  # CLI users are operators: show the full detail
        _fail(f"Ingestion failed: {exc}")
    typer.echo(
        f"Ingested {stats.documents} documents into {stats.chunks} new chunks "
        f"(added {stats.added_documents}, updated {stats.updated_documents}, "
        f"unchanged {stats.skipped_documents}; index size: {stats.index_size})."
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Chunks to retrieve."),
) -> None:
    """Ask a question against the ingested corpus."""
    settings = _bootstrap()
    container = build_container(settings)
    if container.pipeline is None:
        _fail("No LLM key configured. Set GROQ_API_KEY (or SCRAG_LLM_API_KEY).")
    if not container.load_index() or len(container.store) == 0:
        _fail("Index is empty. Run `scrag ingest` first.")

    try:
        result = container.pipeline.run(question, top_k=top_k)
    except ScragError as exc:
        _fail(f"Query failed: {exc}")

    score = "n/a" if result.evaluation_score is None else f"{result.evaluation_score:.2f}"
    typer.secho("\n=== Answer ===", fg=typer.colors.GREEN, bold=True)
    typer.echo(result.answer)
    typer.secho("\n=== Evaluation ===", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"Score: {score} ({result.evaluation_status})  |  "
        f"Attempts: {result.attempts} (returned #{result.best_attempt})"
    )
    typer.echo(f"Explanation: {result.evaluation_explanation}")
    typer.secho("\n=== Context used ===", fg=typer.colors.CYAN, bold=True)
    for i, rc in enumerate(result.used_chunks, start=1):
        guard = f"{rc.guardrail_score:.2f}" if rc.guardrail_score is not None else "n/a"
        snippet = rc.chunk.text[:300] + ("..." if len(rc.chunk.text) > 300 else "")
        typer.echo(f"\n[{i}] {rc.chunk.source} (sim={rc.score:.2f}, guardrail={guard})\n{snippet}")


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host (defaults to config)."),
    port: int = typer.Option(None, help="Bind port (defaults to config)."),
    reload: bool = typer.Option(False, help="Auto-reload for development."),
) -> None:
    """Run the FastAPI service with uvicorn."""
    import uvicorn

    settings = _bootstrap()
    uvicorn.run(
        "scrag.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@app.command()
def info() -> None:
    """Show effective configuration and index status."""
    settings = _bootstrap()
    container = build_container(settings)
    container.load_index()
    typer.echo(f"Embedding model : {settings.embedding_model} (dim={settings.embedding_dim})")
    typer.echo(f"Vector backend  : {settings.vector_backend}")
    typer.echo(f"Index size      : {len(container.store)} chunks")
    typer.echo(f"LLM configured  : {settings.has_llm} ({settings.llm_base_url})")
    typer.echo(f"Generator model : {settings.generator_model}")
    typer.echo(f"Docs dir        : {settings.docs_dir}")
    typer.echo(f"Index dir       : {settings.index_dir}")


if __name__ == "__main__":
    app()
