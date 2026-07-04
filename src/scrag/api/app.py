"""FastAPI application factory.

``create_app`` builds the app, wires a :class:`Container` (real or injected) via
the lifespan handler, registers routes and installs error handlers that turn
domain errors into clean JSON responses.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings, get_settings
from ..container import Container, build_container
from ..llm.client import LLMError
from ..logging_config import configure_logging, get_logger
from .routes import router

logger = get_logger(__name__)


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
    async def lifespan(app: FastAPI):
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMError)
    async def _llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(
            status_code=502, content={"error": "llm_error", "detail": str(exc)}
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400, content={"error": "bad_request", "detail": str(exc)}
        )

    app.include_router(router)
    return app
