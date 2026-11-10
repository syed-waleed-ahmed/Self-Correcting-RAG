"""Command-line interface (Typer): ingest, query, serve, info, agent, eval, db."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

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


db_app = typer.Typer(
    help="Database schema migrations (needs SCRAG_DATABASE_URL).", no_args_is_help=True
)
app.add_typer(db_app, name="db")


def _db_connect() -> Any:
    settings = _bootstrap()
    if not settings.database_url:
        _fail("SCRAG_DATABASE_URL (or DATABASE_URL) is not set.")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        _fail('Needs the postgres extra: pip install "self-correcting-rag[postgres]"')
    return psycopg.connect(
        settings.database_url,
        prepare_threshold=None,
        row_factory=dict_row,
        connect_timeout=max(1, round(settings.db_connect_timeout_s)),
    )


@db_app.command("status")
def db_status() -> None:
    """Show the applied schema version and the latest available one."""
    from .persistence.migrations import current_version, load_migrations

    latest = max((m.version for m in load_migrations()), default=0)
    with _db_connect() as conn:
        version = current_version(conn)
    typer.echo(f"Schema version: {version} (latest available: {latest})")


@db_app.command("upgrade")
def db_upgrade(
    to: int | None = typer.Option(None, "--to", help="Target version (default: latest)."),
) -> None:
    """Apply pending migrations."""
    from .persistence.migrations import apply_migrations

    with _db_connect() as conn:
        applied = apply_migrations(conn, target=to)
    typer.echo(f"Applied: {applied}" if applied else "Already up to date.")


@db_app.command("downgrade")
def db_downgrade(
    to: int = typer.Option(..., "--to", help="Version to go back to (0 removes every table)."),
    yes: bool = typer.Option(False, "--yes", help="Confirm: downgrades delete data."),
) -> None:
    """Revert migrations newer than --to. Destructive: the dropped tables' data is lost."""
    from .persistence.migrations import rollback_migrations

    if not yes:
        _fail("Downgrading deletes data. Re-run with --yes to confirm.")
    with _db_connect() as conn:
        reverted = rollback_migrations(conn, target=to)
    typer.echo(f"Reverted: {reverted}" if reverted else "Nothing to revert.")


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
    compare: int = typer.Option(
        0,
        help="Compare baseline / hybrid / self_correcting / agent answers on a stratified "
        "sample of N questions (spends LLM quota; 0 = off).",
    ),
    configs: str = typer.Option(
        "baseline,hybrid,self_correcting,agent", help="Configurations for --compare."
    ),
    answer_top_k: int = typer.Option(3, help="Chunks per answer in --compare (demo uses 3)."),
    tpm: int = typer.Option(6000, help="Tokens-per-minute budget for --compare pacing."),
    retrieval: bool = typer.Option(
        True, "--retrieval/--no-retrieval", help="Also measure retrieval (needs labels)."
    ),
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
    if retrieval:
        typer.secho(f"\nRetrieval @k={k} over {len(items)} questions", bold=True)
        typer.echo(f"{'mode':<8} {'recall':>7} {'hit':>7} {'MRR':>7} {'nDCG':>7} {'ms':>7}")
    for m in modes if retrieval else []:
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

    if compare > 0:
        from .evaluation_compare import CONFIGS, compare_answers, report_to_json, stratified

        if container.pipeline is None or container.agent is None:
            _fail("--compare needs an LLM key (GROQ_API_KEY).")
        chosen = tuple(c.strip() for c in configs.split(",") if c.strip())
        if any(c not in CONFIGS for c in chosen):
            _fail(f"--configs must be a subset of {','.join(CONFIGS)}.")
        sample = stratified(items, compare)
        typer.secho(
            f"\nComparing {len(chosen)} configurations on {len(sample)} questions "
            f"(top_k={answer_top_k}, pacing at {tpm} tokens/min)",
            bold=True,
        )

        def progress(row: dict[str, Any]) -> None:
            if "error" in row:
                typer.echo(f"  {row['id']} {row['config']:<16} ERROR {row['error']}")
            else:
                score = "n/a" if row["score"] is None else f"{row['score']:.2f}"
                cover = row["keyword_coverage"]
                if row["expect_abstain"]:
                    outcome = "declined" if row["abstained"] else "ANSWERED"
                else:
                    outcome = "keywords n/a" if cover is None else f"keywords {cover:.2f}"
                typer.echo(
                    f"  {row['id']} {row['config']:<16} score {score}  {outcome:<13} "
                    f"{row['tokens']:>5} tok  {row['latency_ms']:>7.0f} ms"
                )

        comparison = compare_answers(
            container, sample, chosen, top_k=answer_top_k, tokens_per_minute=tpm, on_row=progress
        )
        report["comparison"] = report_to_json(comparison)
        typer.secho(f"\nAnswers (judge: {comparison.judge_model})", bold=True)
        typer.echo(
            f"{'config':<16} {'judge':>6} {'pass':>6} {'keywords':>9} {'abstain':>8} "
            f"{'tokens':>7} {'calls':>6} {'ms':>7} {'n':>3}"
        )

        def fmt(value: Any, spec: str) -> str:
            return "n/a" if value is None else format(value, spec)

        for name, row in comparison.summary.items():
            typer.echo(
                f"{name:<16} {fmt(row['mean_judge_score'], '.3f'):>6} "
                f"{fmt(row['pass_rate'], '.2f'):>6} {fmt(row['keyword_coverage'], '.3f'):>9} "
                f"{fmt(row['abstain_rate'], '.2f'):>8} "
                f"{fmt(row['mean_tokens'], '.0f'):>7} {fmt(row['mean_llm_calls'], '.1f'):>6} "
                f"{fmt(row['mean_latency_ms'], '.0f'):>7} {row['answered']:>3}"
            )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"\nReport written to {output}")


if __name__ == "__main__":
    app()
