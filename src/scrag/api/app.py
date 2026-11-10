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
from .routes import router

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# Accept a caller-supplied ID only if it is short and log-safe.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

_HTTP_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.container = container or build_container(settings)
        loaded = app.state.container.load_index()
        logger.info(
            "Startup complete (index loaded=%s, size=%d, llm=%s)",
            loaded,
            len(app.state.container.store),
            settings.has_llm,
        )
        yield

    app = FastAPI(
        title="Self-Correcting RAG",
        version=__version__,
        summary="Multi-agent, self-correcting Retrieval-Augmented Generation API.",
        lifespan=lifespan,
    )

    # Order matters: the last middleware added is the outermost. CORS wraps the
    # request-context layer so even sanitized 500s carry CORS headers.
    app.add_middleware(RequestContextMiddleware)
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
            status_code=exc.http_status, content=error_body(exc.code, exc.public_message)
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, exc.detail),
            headers=getattr(exc, "headers", None),
        )

    app.include_router(router)
    return app
