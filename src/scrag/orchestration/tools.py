"""Typed, permissioned tool registry.

Every tool declares a pydantic input model, a permission level and a timeout.
:meth:`ToolRegistry.invoke` is the only way a workflow runs a tool, and it:

1. rejects unknown tools,
2. validates the input against the tool's schema,
3. refuses ``sensitive`` tools unless they were explicitly approved for this
   run (human-in-the-loop) and refuses tools that are disabled,
4. enforces the timeout,
5. returns a JSON-safe output and an audit record (:class:`ToolCall`) whether
   the call succeeded or not. Tool exceptions never escape.

There is deliberately no shell, filesystem, network or code-execution tool.
"""

from __future__ import annotations

import ast
import contextvars
import math
import operator
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..logging_config import get_logger
from ..observability import metrics, span

logger = get_logger(__name__)

Permission = Literal["safe", "sensitive"]
ToolStatus = Literal["ok", "invalid_input", "denied", "needs_approval", "timeout", "error"]


class ToolCall(BaseModel):
    """Audit record of one tool invocation."""

    tool: str
    input: dict[str, Any]
    status: ToolStatus
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[BaseModel], Any]
    permission: Permission = "safe"
    timeout_s: float = 10.0
    enabled: bool = True

    def spec(self) -> dict[str, Any]:
        """Discovery entry: what the planner (and API clients) may see."""
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "input_schema": self.input_model.model_json_schema(),
            "enabled": self.enabled,
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self.tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def invoke(
        self, name: str, raw_input: dict[str, Any], *, approved: frozenset[str] = frozenset()
    ) -> ToolCall:
        start = time.perf_counter()
        safe_input = raw_input if isinstance(raw_input, dict) else {"value": raw_input}

        def done(status: ToolStatus, *, output: Any = None, error: str | None = None) -> ToolCall:
            call = ToolCall(
                tool=name,
                input=safe_input,
                status=status,
                output=output,
                error=error,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
            logger.info("tool %s -> %s (%.1f ms)", name, status, call.latency_ms)
            # Tool names come from the planner (LLM output): never a free-form label.
            metrics.TOOL_CALLS.labels(name if name in self.tools else "unknown", status).inc()
            return call

        tool = self.tools.get(name)
        if tool is None:
            return done("denied", error=f"unknown tool {name!r}")
        if not tool.enabled:
            return done("denied", error=f"tool {name!r} is disabled here")
        if tool.permission == "sensitive" and name not in approved:
            return done("needs_approval", error=f"tool {name!r} requires explicit approval")
        try:
            args = tool.input_model.model_validate(safe_input)
        except ValidationError as exc:
            return done("invalid_input", error=_short_validation_error(exc))

        # Run in a worker so a hung tool can't stall the workflow past its timeout.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            with span(f"tool.{name}", scrag__tool=name, scrag__permission=tool.permission):
                ctx = contextvars.copy_context()  # inside the span, so the tool nests under it
                output = pool.submit(ctx.run, tool.run, args).result(timeout=tool.timeout_s)
        except FutureTimeout:
            return done("timeout", error=f"exceeded {tool.timeout_s:g}s")
        except Exception as exc:  # noqa: BLE001 - tool failures are recorded, never raised
            logger.warning("tool %s failed: %s", name, exc)
            return done("error", error=f"{type(exc).__name__}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return done("ok", output=output)


def _short_validation_error(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()]
    return "; ".join(parts)[:300]


# --------------------------------------------------------------------------------------
# Built-in tools
# --------------------------------------------------------------------------------------
class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="What to look up.")
    top_k: int = Field(default=3, ge=1, le=8)


def document_search_tool(retrieve: Callable[[str, int], list[Any]]) -> Tool:
    def run(args: BaseModel) -> list[dict[str, Any]]:
        assert isinstance(args, DocumentSearchInput)
        return [
            {"source": rc.chunk.source, "text": rc.chunk.text, "score": round(rc.score, 4)}
            for rc in retrieve(args.query, args.top_k)
        ]

    return Tool(
        name="document_search",
        description="Search the knowledge base and return the most relevant passages.",
        input_model=DocumentSearchInput,
        run=run,
        timeout_s=20.0,
    )


class CalculatorInput(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic only: numbers, + - * / // % **, parentheses, "
        "and sqrt, log, log10, exp, abs, round, min, max, pi, e.",
    )


_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic by walking a whitelisted AST (never ``eval``)."""

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            result = _BIN_OPS[type(node.op)](left, right)
            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("result too large")
            return result
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
            and len(node.args) <= 4
        ):
            return _FUNCS[node.func.id](*(walk(a) for a in node.args))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    value = walk(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is not finite")
    return value


def calculator_tool() -> Tool:
    def run(args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, CalculatorInput)
        try:
            return {"expression": args.expression, "result": safe_eval(args.expression)}
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as exc:
            return {"expression": args.expression, "error": f"{type(exc).__name__}: {exc}"}

    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression exactly (no code execution).",
        input_model=CalculatorInput,
        run=run,
        timeout_s=2.0,
    )


class ReindexInput(BaseModel):
    reset: bool = Field(default=False, description="Rebuild from scratch instead of upserting.")


def reindex_tool(run_ingest: Callable[[bool], Any], *, enabled: bool = True) -> Tool:
    def run(args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, ReindexInput)
        stats = run_ingest(args.reset)
        return {
            "documents": stats.documents,
            "chunks": stats.chunks,
            "index_size": stats.index_size,
        }

    return Tool(
        name="reindex_documents",
        description="Re-index the server's document directory. Changes server state: "
        "requires explicit human approval.",
        input_model=ReindexInput,
        run=run,
        permission="sensitive",
        timeout_s=120.0,
        enabled=enabled,
    )
