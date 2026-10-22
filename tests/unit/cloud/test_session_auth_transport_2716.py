"""Credential headers on a reused pooled session, checked at a real HTTP transport (#2716).

A local aiohttp server records the headers it actually receives. Only two boundaries
are replaced: the JWT validator (a signature check) and the backend API-key check.
The credential transition uses the public auth API: authenticate with a bearer
token, then ``set_api_key`` and ``authenticate``. The request is made on the same
session object that carried the first credential.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

import traigent.cloud.client as _cloud_client_module
from traigent.cloud.auth import AuthCredentials, AuthManager, AuthMode
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import TraigentCloudClient

pytestmark = pytest.mark.backend_online

KEY_B = "tg_" + "b" * 61  # pragma: allowlist secret
JWT_A = "header.payload.signature"


class _Recorder:
    """Records the method, path and headers of every request the server receives."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def app(self) -> web.Application:
        app = web.Application()

        async def handle(request: web.Request) -> web.Response:
            self.requests.append(
                {
                    "method": request.method,
                    "path": request.path,
                    "authorization": request.headers.getall("Authorization", []),
                    "x_api_key": request.headers.getall("X-API-Key", []),
                }
            )
            return web.json_response({"ok": True})

        app.router.add_route("*", "/{tail:.*}", handle)
        return app


@pytest.fixture(autouse=True)
def _real_client_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cloud_client_module, "AuthManager", AuthManager)


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in os.environ if n.startswith("TRAIGENT_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")


@pytest_asyncio.fixture
async def server():
    recorder = _Recorder()
    async with TestServer(recorder.app()) as test_server:
        yield recorder, f"http://{test_server.host}:{test_server.port}"


def _jwt_validator_accepts() -> Any:
    """Boundary: the JWT signature check. Everything after it is the real path."""
    validator = MagicMock()
    validator.validate_token.return_value = SimpleNamespace(
        valid=True, warnings=[], expires_at=None
    )
    return patch(
        "traigent.security.jwt_validator.get_secure_jwt_validator",
        return_value=validator,
    )


async def _accept_api_key(self, api_key):  # noqa: ARG001 - backend key check boundary
    return None


def _backend_accepts_api_keys() -> Any:
    return patch.object(
        AuthManager, "_validate_api_key_with_backend", new=_accept_api_key
    )


async def _jwt_first_then_api_key_b(client: Any, recorder: _Recorder, request) -> None:
    """Real transition on one client: JWT A serves a request, then API key B takes over.

    ``request`` performs one request on the client's reused session. The first call
    runs under JWT A and must reach the server as bearer-only. The second runs after
    ``set_api_key(KEY_B)`` and must reach it as key-only.
    """
    auth = client.auth
    with _jwt_validator_accepts():
        assert (
            await auth.authenticate(
                AuthCredentials(mode=AuthMode.JWT_TOKEN, jwt_token=JWT_A)
            )
        ).success
    assert await request() is True
    first = recorder.requests[-1]
    assert first["authorization"] == [f"Bearer {JWT_A}"]
    assert first["x_api_key"] == []

    auth.set_api_key(KEY_B)
    with _backend_accepts_api_keys():
        assert (await auth.authenticate()).success


@pytest.mark.asyncio
async def test_cloud_client_reused_session_sends_only_api_key_b(server):
    recorder, base_url = server
    client = TraigentCloudClient(base_url=base_url)

    async def delete():
        return await client.delete_session("session-1")

    await _jwt_first_then_api_key_b(client, recorder, delete)
    session_under_jwt = client._aio_session
    assert session_under_jwt is not None

    assert await delete() is True

    assert client._aio_session is session_under_jwt  # same pooled session
    sent = recorder.requests[-1]
    assert sent["method"] == "DELETE"
    assert sent["x_api_key"] == [KEY_B]
    assert sent["authorization"] == []


@pytest.mark.asyncio
async def test_backend_pool_reused_session_sends_only_api_key_b(server):
    recorder, base_url = server
    backend = BackendIntegratedClient(base_url=base_url)

    async def delete():
        return await backend.delete_session("session-1", cascade=True)

    await _jwt_first_then_api_key_b(backend, recorder, delete)
    pool_under_jwt = await backend._ensure_session()

    assert await delete() is True

    assert await backend._ensure_session() is pool_under_jwt  # same pool
    sent = recorder.requests[-1]
    assert sent["method"] == "DELETE"
    assert sent["x_api_key"] == [KEY_B]
    assert sent["authorization"] == []
