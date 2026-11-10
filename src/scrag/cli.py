"""Command-line interface (Typer): ingest, query, serve, info."""

from __future__ import annotations

from pathlib import Path
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


@app.command()
def agent(
    task: str = typer.Argument(..., help="What the agent should do."),
    approve: list[str] = typer.Option(
        [], "--approve", help="Approve a sensitive tool for this run (repeatable)."
    ),
) -> None:
    """Run the bounded agent workflow: plan, use tools, write, verify."""
    settings = _bootstrap()
    container = build_container(settings)
    if container.agent is None:
        _fail("No LLM key configured. Set GROQ_API_KEY (or SCRAG_LLM_API_KEY).")
    container.load_index()
    result = container.agent.run(task, approved=frozenset(approve))

    typer.secho(f"\n=== Plan ({result.plan_source}) ===", fg=typer.colors.CYAN, bold=True)
    for i, step in enumerate(result.plan, 1):
        typer.echo(f"{i}. {step.tool} {step.input}  - {step.purpose}")
    if result.status == "awaiting_approval":
        names = ", ".join(sorted({s.tool for s in result.pending_approval}))
        _fail(f"\nNeeds human approval for: {names}. Re-run with --approve <tool>.")
    typer.secho("\n=== Tool calls ===", fg=typer.colors.CYAN, bold=True)
    for executed in result.steps:
        call = executed.call
        typer.echo(f"{executed.index}. {call.tool}: {call.status} ({call.latency_ms} ms)")
    typer.secho(f"\n=== Answer ({result.status}) ===", fg=typer.colors.GREEN, bold=True)
    typer.echo(result.answer or result.error or "")
    if result.verification:
        v = result.verification
        score = "n/a" if v.score is None else f"{v.score:.2f}"
        typer.echo(
            f"\nVerification: {score} ({'passed' if v.passed else v.status})  |  "
            f"attempts: {result.attempts}"
        )
    typer.echo(f"Usage: {result.usage}")


@app.command("eval")
def evaluate(
    golden: Path = typer.Option(Path("data/eval/golden.jsonl"), help="Golden dataset (JSONL)."),
    k: int = typer.Option(5, "--k", "-k", help="Cut-off for retrieval metrics."),
    mode: str = typer.Option("both", help="Retrieval mode: dense, hybrid or both."),
    answers: bool = typer.Option(
        False, help="Also run the full pipeline per question (spends LLM quota)."
    ),
    limit: int = typer.Option(0, help="Only use the first N questions (0 = all)."),
    output: Path | None = typer.Option(None, help="Write the full JSON report here."),
) -> None:
    """Measure retrieval (and optionally answer) quality on a golden dataset."""
    import json

    from .container import build_retriever
    from .evaluation import evaluate_answers, evaluate_retrieval, load_golden, to_jsonable

    settings = _bootstrap()
    container = build_container(settings)
    if not container.load_index() or len(container.store) == 0:
        _fail("Index is empty. Run `scrag ingest` first.")
    items = load_golden(golden)
    if limit > 0:
        items = items[:limit]
    container.embedder.embed(["warm-up"])  # keep one-time model loading out of latencies
    modes = ["dense", "hybrid"] if mode == "both" else [mode]
    if any(m not in {"dense", "hybrid"} for m in modes):
        _fail("--mode must be dense, hybrid or both.")

    report: dict[str, object] = {"golden": str(golden), "index_size": len(container.store)}
    typer.secho(f"\nRetrieval @k={k} over {len(items)} questions", bold=True)
    typer.echo(f"{'mode':<8} {'recall':>7} {'hit':>7} {'MRR':>7} {'nDCG':>7} {'ms':>7}")
    for m in modes:
        retriever = build_retriever(settings, container.embedder, container.store, mode=m)
        r = evaluate_retrieval(retriever, items, k)
        report[f"retrieval_{m}"] = to_jsonable(r)
        typer.echo(
            f"{m:<8} {r.recall_at_k:>7.3f} {r.hit_rate_at_k:>7.3f} {r.mrr:>7.3f} "
            f"{r.ndcg_at_k:>7.3f} {r.mean_latency_ms:>7.1f}"
        )

    if answers:
        if container.pipeline is None:
            _fail("--answers needs an LLM key (GROQ_API_KEY).")
        a = evaluate_answers(container.pipeline, items)
        report["answers"] = to_jsonable(a)
        typer.secho(f"\nAnswers ({settings.retrieval_mode} retrieval)", bold=True)
        typer.echo(f"statuses: {a.status_counts}  errors: {a.errors}")
        typer.echo(
            f"mean score: {a.mean_score}  mean attempts: {a.mean_attempts}  "
            f"corrected: {a.corrected}  keyword coverage: {a.keyword_coverage:.3f}  "
            f"mean latency: {a.mean_latency_ms:.0f} ms"
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"\nReport written to {output}")


if __name__ == "__main__":
    app()
