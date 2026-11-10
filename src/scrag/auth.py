"""Bearer-JWT verification for the operator/admin (e.g. Supabase Auth).

Two key sources:

* **JWKS** (asymmetric: ES256/RS256/EdDSA), e.g. Supabase's
  ``https://<project>.supabase.co/auth/v1/.well-known/jwks.json``. Keys are
  fetched lazily and cached.
* **Shared secret** (HS256), for legacy Supabase projects.

The accepted algorithms come from the configuration, never from the token's own
header, so a token can't downgrade itself (e.g. an HS256 token "signed" with a
public key). ``exp`` and ``sub`` are required; ``aud`` and, if configured,
``iss`` are checked.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .logging_config import get_logger

logger = get_logger(__name__)

ASYMMETRIC_ALGORITHMS = ["ES256", "RS256", "EdDSA"]


class JWTVerifier:
    def __init__(
        self,
        *,
        jwks_url: str | None = None,
        secret: str | None = None,
        issuer: str | None = None,
        audience: str | None = "authenticated",
        leeway_s: int = 30,
        jwks_client: Any = None,
    ) -> None:
        if not (jwks_url or secret or jwks_client):
            raise ValueError("JWTVerifier needs a JWKS URL or a shared secret")
        import jwt  # PyJWT, from the `auth` extra

        self._jwt = jwt
        self._jwks = jwks_client or (
            jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600, timeout=5)
            if jwks_url
            else None
        )
        self._secret = secret
        self.issuer = issuer
        self.audience = audience
        self.leeway_s = leeway_s

    def verify(self, token: str) -> dict[str, Any] | None:
        """The token's claims if it is valid, else None (the reason is logged)."""
        jwt = self._jwt
        try:
            if self._jwks is not None:
                key: Any = self._jwks.get_signing_key_from_jwt(token).key
                algorithms = ASYMMETRIC_ALGORITHMS
            else:
                key, algorithms = self._secret, ["HS256"]
            claims: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=algorithms,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_s,
                options={"require": ["exp", "sub"], "verify_aud": self.audience is not None},
            )
        except jwt.PyJWTError as exc:
            logger.info("Rejected bearer JWT: %s", type(exc).__name__)
            return None
        return claims


def build_verifier(settings: Settings) -> JWTVerifier | None:
    """A verifier when JWT admin auth is configured (keys and at least one admin), else None."""
    if not settings.admin_subjects or not (settings.auth_jwks_url or settings.auth_jwt_secret):
        return None
    return JWTVerifier(
        jwks_url=settings.auth_jwks_url,
        secret=settings.auth_jwt_secret,
        issuer=settings.auth_issuer,
        audience=settings.auth_audience or None,
    )
