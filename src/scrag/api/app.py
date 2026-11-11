"""FastAPI application factory.

``create_app`` builds the app, wires a :class:`Container` (real or injected) via
the lifespan handler, registers routes and installs the error handling layer.

Error contract: every non-validation error response has the body
``{"error": <code>, "detail": <client-safe message>, "request_id": <id>}`` and
an ``X-Request-ID`` header. Internal details (paths, provider responses, stack
traces) are logged under that request ID and never returned to the client.
Request validation errors keep FastAPI's standard 422 shape.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..config import Settings, get_settings
from ..container import Container, build_container
from ..demo import RateLimiter
from ..errors import RateLimitedError, ScragError
from ..logging_config import configure_logging, get_logger, request_id_var
from ..observability import configure_tracing, metrics, set_attributes, shutdown_tracing, span
from .deps import client_address
from .routes import router
from .workspace import router as workspace_router

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# Multipart framing (boundaries, part headers) on top of the file itself.
_MULTIPART_OVERHEAD_BYTES = 16 * 1024
# Accept a caller-supplied ID only if it is short and log-safe.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

_HTTP_ERROR_CODES = {
    400: "bad_request",
    422: "validation_error",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    411: "length_required",
    413: "payload_too_large",
    429: "rate_limited",
    503: "service_unavailable",
}


def error_body(code: str, detail: object) -> dict[str, object]:
    return {"error": code, "detail": detail, "request_id": request_id_var.get()}


class RequestContextMiddleware:
    """Assign a request ID, expose it in a header, and contain unhandled errors.

    Implemented as plain ASGI (not ``BaseHTTPMiddleware``) so it can guarantee a
    sanitized 500 response, with the request ID, for any unexpected exception.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = next(
            (v.decode("latin-1") for k, v in scope["headers"] if k == b"x-request-id"), ""
        )
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        response_started = False

        async def send_with_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.exception("Unhandled error while serving %s", scope.get("path"))
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content=error_body("internal_error", "An internal error occurred."),
            )
            await response(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


def route_template(scope: Scope) -> str:
    """The matched route as a template, e.g. ``/api/v1/documents/{document_id}``.

    Rebuilt from the request path and the matched path parameters, because an
    included router's route object doesn't carry its mount prefix. Requests that
    matched no route share the label ``unmatched``; middleware that answers
    before routing can name itself in ``scope["scrag.route"]``.
    """
    if (label := scope.get("scrag.route")) is not None:
        return str(label)
    if scope.get("route") is None:
        return "unmatched"
    path: str = scope.get("path", "")
    params = {str(value): name for name, value in (scope.get("path_params") or {}).items()}
    if not params:
        return path
    return "/".join("{" + params[seg] + "}" if seg in params else seg for seg in path.split("/"))


class MetricsMiddleware:
    """Count and time every HTTP request by route template, and open its span.

    Sits outside RequestContextMiddleware, so it records the final status code
    (including sanitized 500s) and can tag the span with the request ID. Routes
    are FastAPI path templates; unmatched paths share one ``unmatched`` label, so
    scanners can't inflate the metric cardinality.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        start = time.perf_counter()
        status = 500
        request_id: str | None = None

        async def send_and_observe(message: Message) -> None:
            nonlocal status, request_id
            if message["type"] == "http.response.start":
                status = message["status"]
                for key, value in message.get("headers", []):
                    if key.lower() == b"x-request-id":
                        request_id = value.decode("latin-1")
            await send(message)

        with span(f"HTTP {method}", http__request__method=method) as current:
            try:
                await self.app(scope, receive, send_and_observe)
            finally:
                route = route_template(scope)
                metrics.HTTP_REQUESTS.labels(method, route, str(status)).inc()
                metrics.HTTP_DURATION.labels(method, route).observe(time.perf_counter() - start)
                if current is not None:
                    current.update_name(f"HTTP {method} {route}")
                    set_attributes(
                        current,
                        http__route=route,
                        http__response__status_code=status,
                        scrag__request_id=request_id,
                    )


class FloodGuardMiddleware:
    """A cheap per-client cap on all requests, checked before routing (demo mode).

    The demo rate limit covers only the expensive endpoints (search, query,
    agent runs, session creation, uploads). Everything else, including the
    database-backed reads of sessions, documents, jobs and history, had no
    per-client limit. Measured: a client flooding ``GET /demo`` was served 900
    requests a second; with this guard it got 417 in three minutes, and other
    visitors' p95 latency went back to about baseline (docs/load-testing.md).
    Health checks are exempt, so a platform's liveness probe is never throttled.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        forwarded_ip_header: str | None,
        forwarded_ip_hops: int,
        exempt_paths: frozenset[str] = frozenset({"/health", "/api/v1/health"}),
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.header = forwarded_ip_header
        self.hops = forwarded_ip_hops
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] not in self.exempt_paths:
            try:
                self.limiter.check(client_address(scope, self.header, self.hops))
            except RateLimitedError as exc:
                scope["scrag.route"] = "flood_guard"  # its own metrics label
                response = JSONResponse(
                    status_code=429,
                    content=error_body(exc.code, exc.public_message),
                    headers=exc.headers,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class BodyLimitMiddleware:
    """Cap every request body before the app buffers it.

    Uploads (``POST .../documents``) may be up to ``upload_max_bytes`` and must
    declare a ``Content-Length`` (browsers always do for form uploads), because
    Starlette spools a multipart body to disk while parsing it, before any route
    code runs. Every other body is capped at ``max_bytes``: JSON requests are
    tiny, and without a cap a client could make the server buffer megabytes
    before validation rejects them. A declared length over the cap is refused
    at once; a body without one (chunked) is counted as it arrives and cut off.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        upload_max_bytes: int,
        upload_path_suffix: str = "/documents",
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.upload_max_bytes = upload_max_bytes
        self.upload_path_suffix = upload_path_suffix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        upload = scope["method"] == "POST" and scope["path"].rstrip("/").endswith(
            self.upload_path_suffix
        )
        limit = self.upload_max_bytes if upload else self.max_bytes
        declared = next((v for k, v in scope["headers"] if k == b"content-length"), None)
        status, code, detail = 0, "", ""
        if upload and (declared is None or not declared.isdigit()):
            status, code, detail = 411, "length_required", "Uploads need a Content-Length."
        elif declared is not None and declared.isdigit() and int(declared) > limit:
            status, code = 413, "upload_too_large" if upload else "payload_too_large"
            detail = (
                "The uploaded file is too large." if upload else "The request body is too large."
            )
        if status:
            response = JSONResponse(status_code=status, content=error_body(code, detail))
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # FastAPI re-raises HTTPExceptions from body parsing, so this
                    # becomes an ordinary 413 error response.
                    raise StarletteHTTPException(413, "The request body is too large.")
            return message

        await self.app(scope, limited_receive, send)


# Hardening headers on every API response. Nothing here is rendered by a browser
# except the HTML API docs, which load Swagger UI from a CDN and so keep their
# own policy; everything else gets a deny-all Content-Security-Policy.
_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "cross-origin-opener-policy": "same-origin",
}
_API_CSP = "default-src 'none'; frame-ancestors 'none'"


class SecurityHeadersMiddleware:
    """Add hardening headers, and ``Cache-Control: no-store`` unless a route set one.

    API responses can carry session tokens (``POST /sessions``) or a visitor's
    documents and history, so no shared cache may keep them.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
                headers.setdefault("cache-control", "no-store")
                if not headers.get("content-type", "").startswith("text/html"):
                    headers.setdefault("content-security-policy", _API_CSP)
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Args:
        settings: Configuration; falls back to the cached process settings.
        container: Pre-built dependency container. When provided (e.g. in tests),
            the lifespan handler reuses it instead of constructing the real,
            model-loading container.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    metrics.BUILD_INFO.labels(__version__).set(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        tracing = settings.otel_exporter != "none" and configure_tracing(
            settings.otel_exporter, service_name=settings.otel_service_name, version=__version__
        )
        c = app.state.container = container or build_container(settings)
        loaded = c.load_index()
        if c.persistence is not None:
            c.persistence.start()  # never blocks on the database (see Persistence.start)
        if settings.warm_up_on_start:
            # In the background, so the server answers health checks (and the UI's
            # wake-up poll) at once. A request that needs the model meanwhile waits
            # on the embedder's load lock instead of loading a second copy.
            threading.Thread(target=c.warm_up, name="scrag-warm-up", daemon=True).start()
        logger.info(
            "Startup complete (index loaded=%s, size=%d, llm=%s, persistence=%s)",
            loaded,
            len(c.store),
            settings.has_llm,
            c.persistence is not None,
        )
        try:
            yield
        finally:
            if c.persistence is not None:
                c.persistence.stop()
            if tracing:
                shutdown_tracing()

    app = FastAPI(
        title="Self-Correcting RAG",
        version=__version__,
        summary="Multi-agent, self-correcting Retrieval-Augmented Generation API.",
        lifespan=lifespan,
    )

    # Order matters: the last middleware added is the outermost. CORS wraps the
    # request-context layer so even sanitized 500s carry CORS headers, the flood
    # guard and body limit run inside it so their errors carry a request ID (the
    # guard first: it is the cheapest way to refuse a request), security headers
    # wrap the request context so sanitized 500s get them too, and metrics wrap
    # both so they see the final status.
    app.add_middleware(
        BodyLimitMiddleware,
        max_bytes=settings.max_request_bytes,
        upload_max_bytes=settings.upload_max_bytes + _MULTIPART_OVERHEAD_BYTES,
    )
    if settings.demo_mode:  # outside the body limit: a flood is refused before anything else
        app.add_middleware(
            FloodGuardMiddleware,
            limiter=RateLimiter(settings.demo_flood_limit_per_minute, settings.demo_flood_burst),
            forwarded_ip_header=settings.forwarded_ip_header,
            forwarded_ip_hops=settings.forwarded_ip_hops,
        )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.exception_handler(ScragError)
    async def _scrag_error_handler(request: Request, exc: ScragError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.error("%s on %s: %s", exc.code, request.url.path, exc)
        else:
            logger.info("%s on %s: %s", exc.code, request.url.path, exc)
        return JSONResponse(
            status_code=exc.http_status,
            content=error_body(exc.code, exc.public_message),
            headers=exc.headers or None,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, exc.detail),
            headers=getattr(exc, "headers", None),
        )

    # Versioned API, plus the original unversioned paths for existing clients
    # (hidden from the OpenAPI schema to avoid listing every route twice).
    app.include_router(router, prefix="/api/v1")
    app.include_router(router, include_in_schema=False)
    app.include_router(workspace_router, prefix="/api/v1")  # new routes: versioned only

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """The bare API URL (linked from the README) opens the interactive docs."""
        return RedirectResponse("/docs", status_code=307)

    return app
