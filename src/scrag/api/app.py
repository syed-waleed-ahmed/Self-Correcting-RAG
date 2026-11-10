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
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..config import Settings, get_settings
from ..container import Container, build_container
from ..errors import ScragError
from ..logging_config import configure_logging, get_logger, request_id_var
from ..observability import configure_tracing, metrics, set_attributes, shutdown_tracing, span
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
    matched no route share the label ``unmatched``.
    """
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


class UploadLimitMiddleware:
    """Reject oversized uploads before the multipart body is read.

    Starlette spools a multipart upload to disk while parsing it, before any
    route code runs, so the size limit has to be enforced here. Uploads must
    declare a ``Content-Length`` (browsers always do for form uploads); the
    server enforces that the body matches it.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int, path_suffix: str = "/documents") -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_suffix = path_suffix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/").endswith(self.path_suffix)
        ):
            declared = next((v for k, v in scope["headers"] if k == b"content-length"), None)
            status, code, detail = 0, "", ""
            if declared is None or not declared.isdigit():
                status, code, detail = 411, "length_required", "Uploads need a Content-Length."
            elif int(declared) > self.max_bytes:
                status, code, detail = 413, "upload_too_large", "The uploaded file is too large."
            if status:
                response = JSONResponse(status_code=status, content=error_body(code, detail))
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


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
    # request-context layer so even sanitized 500s carry CORS headers, the upload
    # limit runs inside it so its errors carry a request ID, and metrics wrap the
    # request context so they see the final status.
    app.add_middleware(
        UploadLimitMiddleware,
        max_bytes=settings.upload_max_bytes + _MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(RequestContextMiddleware)
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
    return app
