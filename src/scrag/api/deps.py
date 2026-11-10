"""Request dependencies shared by the routers: auth, rate limits and sessions."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from ..container import Container
from ..errors import ForbiddenError, PersistenceDisabledError, SessionRequiredError

if TYPE_CHECKING:
    from ..persistence import Persistence, SessionRecord

SESSION_HEADER = "x-session-token"


def get_container(request: Request) -> Container:
    return request.app.state.container


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def operator_status(request: Request) -> bool | None:
    """Did the caller prove operator rights?

    True/False when operator auth is configured (a static ``api_auth_token``
    and/or JWT admins); None when nothing is configured. The static token is
    accepted as ``X-API-Key`` or ``Authorization: Bearer`` and compared in
    constant time; a bearer JWT counts if it verifies and its ``sub`` is an
    admin subject.
    """
    c = get_container(request)
    expected = c.settings.api_auth_token
    verifier = c.jwt_verifier
    if not expected and verifier is None:
        return None
    bearer = _bearer(request)
    if expected:
        for candidate in (request.headers.get("x-api-key"), bearer):
            if candidate and secrets.compare_digest(candidate.encode(), expected.encode()):
                return True
    if verifier is not None and bearer:
        claims = verifier.verify(bearer)
        if claims is not None and claims.get("sub") in c.settings.admin_subjects:
            return True
    return False


def require_auth(request: Request) -> None:
    """Reject a missing or wrong credential when operator auth is configured."""
    if operator_status(request) is False:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_auth_unless_demo(request: Request) -> None:
    """Public read endpoints on the demo; token-protected otherwise.

    In demo mode, /search, /query and /agent/run are open to anonymous visitors
    (guarded by the rate limit and daily budget); operator auth only protects
    state-changing endpoints such as /ingest.
    """
    if get_container(request).settings.demo_mode:
        return
    require_auth(request)


def require_operator(request: Request) -> None:
    """Operator-only endpoints. On the demo they need a proven operator, always."""
    status = operator_status(request)
    if status is True:
        return
    if status is False:
        require_auth(request)  # raises 401
    if get_container(request).settings.demo_mode:
        raise ForbiddenError("operator endpoint called without operator credentials on the demo")


def client_key(request: Request) -> str:
    """Rate-limit key: the client IP, read from a trusted proxy header if configured."""
    settings = get_container(request).settings
    header = settings.forwarded_ip_header
    if header:
        value = request.headers.get(header, "")
        hops = [part.strip() for part in value.split(",") if part.strip()]
        if len(hops) >= settings.forwarded_ip_hops:
            return hops[-settings.forwarded_ip_hops]
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Spend one token of the caller's demo rate limit (no-op outside demo mode)."""
    c = get_container(request)
    if c.rate_limiter is not None:
        c.rate_limiter.check(client_key(request))


def get_persistence(request: Request) -> Persistence:
    persistence = get_container(request).persistence
    if persistence is None:
        raise PersistenceDisabledError()
    return persistence


def optional_session(request: Request) -> SessionRecord | None:
    """The caller's session if an X-Session-Token is sent; 401 if it is invalid or expired."""
    token = request.headers.get(SESSION_HEADER)
    if not token:
        return None
    session = get_persistence(request).sessions.authenticate(token.strip())
    if session is None:
        raise SessionRequiredError("invalid or expired session token")
    return session


def require_session(request: Request) -> SessionRecord:
    get_persistence(request)  # 503 persistence_disabled beats 401: no token could help
    session = optional_session(request)
    if session is None:
        raise SessionRequiredError("missing session token")
    return session
