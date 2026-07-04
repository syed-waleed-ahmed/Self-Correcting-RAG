"""Command-line interface (Typer): ingest, query, serve, info."""

from __future__ import annotations

import typer

from .config import get_settings
from .container import build_container
from .ingest import ingest as run_ingest
from .logging_config import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    help="Self-Correcting RAG — multi-agent retrieval-augmented generation.",
    no_args_is_help=True,
)
logger = get_logger(__name__)


def _bootstrap():  # type: ignore[no-untyped-def]
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    return settings


@app.command()
def ingest(
    reset: bool = typer.Option(True, help="Rebuild the index from scratch."),
) -> None:
    """Build the vector index from documents in the configured docs directory."""
    settings = _bootstrap()
    container = build_container(settings)
    stats = run_ingest(
        embedder=container.embedder,
        store=container.store,
        chunker=container.chunker,
        docs_dir=settings.docs_dir,
        batch_size=settings.embedding_batch_size,
        reset=reset,
    )
    typer.echo(
        f"Ingested {stats.documents} documents into {stats.chunks} chunks "
        f"(index size: {stats.index_size})."
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
        typer.secho(
            "No LLM key configured. Set GROQ_API_KEY (or SCRAG_LLM_API_KEY).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if not container.load_index() or len(container.store) == 0:
        typer.secho(
            "Index is empty. Run `scrag ingest` first.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    result = container.pipeline.run(question, top_k=top_k)

    typer.secho("\n=== Answer ===", fg=typer.colors.GREEN, bold=True)
    typer.echo(result.answer)
    typer.secho("\n=== Evaluation ===", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Score: {result.evaluation_score:.2f}  |  Attempts: {result.attempts}")
    typer.echo(f"Explanation: {result.evaluation_explanation}")
    typer.secho("\n=== Context used ===", fg=typer.colors.CYAN, bold=True)
    for i, rc in enumerate(result.used_chunks, start=1):
        guard = f"{rc.guardrail_score:.2f}" if rc.guardrail_score is not None else "n/a"
        snippet = rc.chunk.text[:300] + ("..." if len(rc.chunk.text) > 300 else "")
        typer.echo(
            f"\n[{i}] {rc.chunk.source} (sim={rc.score:.2f}, guardrail={guard})\n{snippet}"
        )


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
    typer.echo(f"Docs dir        : {settings.docs_dir}")
    typer.echo(f"Index dir       : {settings.index_dir}")


if __name__ == "__main__":
    app()
