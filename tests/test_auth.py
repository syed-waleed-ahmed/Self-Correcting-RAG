"""Bearer-JWT admin authentication (e.g. Supabase Auth) for operator endpoints."""

from __future__ import annotations

import hashlib
import hmac
import time
import types

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from jwt.utils import base64url_encode

from scrag.api import create_app
from scrag.auth import JWTVerifier, build_verifier
from scrag.container import build_container
from scrag.vectorstore.numpy_store import NumpyVectorStore

from .conftest import FAKE_DIM, FakeChatClient, FakeEmbedder

SECRET = "test-jwt-secret-that-is-long-enough-for-hs256"
ADMIN = "11111111-2222-3333-4444-555555555555"


def _token(sub=ADMIN, *, key=SECRET, alg="HS256", **claims):
    payload = {"sub": sub, "aud": "authenticated", "exp": int(time.time()) + 300, **claims}
    return jwt.encode(payload, key, algorithm=alg)


@pytest.fixture
def admin_app(settings, tmp_path):
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "a.txt").write_text("rag retrieval generation", encoding="utf-8")
    settings.demo_mode = True
    settings.auth_jwt_secret = SECRET
    settings.admin_subjects = [ADMIN]
    store = NumpyVectorStore(dim=FAKE_DIM, path=settings.index_dir)
    container = build_container(
        settings, embedder=FakeEmbedder(), store=store, chat_client=FakeChatClient()
    )
    return TestClient(create_app(settings, container=container))


def _ingest(client, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/api/v1/ingest", json={"reset": True}, headers=headers)


def test_admin_jwt_grants_operator_rights_on_the_demo(admin_app):
    with admin_app as client:
        assert _ingest(client, _token()).status_code == 200


@pytest.mark.parametrize(
    "token",
    [
        _token(sub="someone-else"),  # valid user, not an admin
        _token(exp=int(time.time()) - 3600),  # expired
        _token(aud="anon"),  # wrong audience
        _token(key="a-different-secret-that-is-also-long-enough"),  # forged
        "not-a-jwt",
    ],
)
def test_invalid_or_non_admin_jwts_are_rejected(admin_app, token):
    with admin_app as client:
        response = _ingest(client, token)
    assert response.status_code == 401


def test_missing_credentials_are_rejected(admin_app):
    with admin_app as client:
        assert _ingest(client).status_code == 401


def test_issuer_is_checked_when_configured():
    verifier = JWTVerifier(secret=SECRET, issuer="https://x.supabase.co/auth/v1")
    assert verifier.verify(_token(iss="https://x.supabase.co/auth/v1"))["sub"] == ADMIN
    assert verifier.verify(_token(iss="https://evil.example/auth/v1")) is None
    assert verifier.verify(_token()) is None  # iss missing


def test_jwks_verification_rejects_algorithm_confusion():
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key()
    jwks = types.SimpleNamespace(
        get_signing_key_from_jwt=lambda token: types.SimpleNamespace(key=public)
    )
    verifier = JWTVerifier(jwks_client=jwks)
    assert verifier.verify(_token(key=private, alg="ES256"))["sub"] == ADMIN

    # The classic attack: an HS256 token whose HMAC key is the (public) PEM key.
    # PyJWT refuses to create one, so it is assembled by hand.
    pem = public.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header = base64url_encode(b'{"alg":"HS256","typ":"JWT"}')
    body = base64url_encode(f'{{"sub":"{ADMIN}","aud":"authenticated","exp":9999999999}}'.encode())
    signature = base64url_encode(hmac.new(pem, header + b"." + body, hashlib.sha256).digest())
    confused = (header + b"." + body + b"." + signature).decode()
    assert verifier.verify(confused) is None


def test_verifier_is_only_built_when_admins_and_keys_are_configured(settings):
    assert build_verifier(settings) is None
    settings.auth_jwt_secret = SECRET
    assert build_verifier(settings) is None  # no admin subjects: nobody could use it
    settings.admin_subjects = [ADMIN]
    assert isinstance(build_verifier(settings), JWTVerifier)
