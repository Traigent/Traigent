"""Credentials are sent per request, never frozen into aiohttp session defaults (#2716).

aiohttp merges the session defaults into every request. When the session was created
with the sign-in headers, a credential change (JWT A -> API key B) left
``Authorization: Bearer A`` on the wire next to ``X-API-Key: B``. These tests drive the
real ``TraigentCloudClient`` session paths and capture what is actually sent. Only the
aiohttp transport and the credential source (``auth.get_headers``) are replaced.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import traigent.cloud.client as _cloud_client_module
from traigent.cloud._aiohttp_compat import aiohttp
from traigent.cloud.auth import AuthManager
from traigent.cloud.client import TraigentCloudClient

pytestmark = pytest.mark.backend_online

URL = "https://api.test"
KEY_B = "tg_" + "b" * 61
JWT_A = "header.payload.signature"
_PUBLIC_ADDRINFO = [(2, 1, 6, "", ("93.184.216.34", 0))]  # AF_INET, SOCK_STREAM
_CREDENTIAL_HEADERS = {"authorization", "x-api-key"}


class _Response:
    status = 200

    def __init__(self, body: Any) -> None:
        self._body = body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        return self._body

    async def text(self) -> str:
        return ""


class _Wire:
    """Records every request; ``headers`` are the session defaults aiohttp merges in."""

    sent: list[dict[str, Any]] = []

    def __init__(self, *args: Any, headers: Any = None, **kwargs: Any) -> None:
        self.headers = dict(headers or {})
        self.closed = False

    async def __aenter__(self) -> _Wire:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        self.closed = True

    def _record(self, method: str, url: Any, kwargs: dict[str, Any]) -> _Response:
        _Wire.sent.append(
            {
                "method": method,
                "url": str(url),
                "defaults": dict(self.headers),
                "request_headers": dict(kwargs.get("headers") or {}),
            }
        )
        return _Response({})

    def post(self, url: Any, *args: Any, **kwargs: Any) -> _Response:
        return self._record("POST", url, kwargs)

    def get(self, url: Any, *args: Any, **kwargs: Any) -> _Response:
        return self._record("GET", url, kwargs)

    def delete(self, url: Any, *args: Any, **kwargs: Any) -> _Response:
        return self._record("DELETE", url, kwargs)


def _effective(record: dict[str, Any]) -> dict[str, str]:
    """Headers aiohttp puts on the wire: session defaults overridden per request."""
    request_names = {n.lower() for n in record["request_headers"]}
    merged = {
        k: v for k, v in record["defaults"].items() if k.lower() not in request_names
    }
    merged.update(record["request_headers"])
    return {k.lower(): v for k, v in merged.items()}


@pytest.fixture(autouse=True)
def _real_client_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the genuine AuthManager in the client module, whatever ran before."""
    monkeypatch.setattr(_cloud_client_module, "AuthManager", AuthManager)


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every TRAIGENT_* variable so ambient credentials cannot decide the headers."""
    for name in [n for n in os.environ if n.startswith("TRAIGENT_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    _Wire.sent = []
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setattr(aiohttp, "ClientSession", _Wire)
    return _Wire.sent


def _cloud_client() -> TraigentCloudClient:
    with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
        return TraigentCloudClient(api_key=KEY_B, base_url=URL)


def _credential_source(client: TraigentCloudClient, state: dict[str, Any]) -> None:
    """Make ``auth.get_headers`` return whatever the credential state currently is."""

    async def _get_headers(target: str = "both") -> dict[str, str]:
        return dict(state["headers"])

    client.auth.get_headers = AsyncMock(side_effect=_get_headers)  # type: ignore[method-assign]


def _defaults_credentials(session: Any) -> set[str]:
    return {
        k.lower() for k in dict(session.headers) if k.lower() in _CREDENTIAL_HEADERS
    }


@pytest.mark.asyncio
async def test_jwt_then_api_key_sends_only_current_credential(wire):
    """(a) Authenticate with JWT A, switch to API key B, request: no stale Authorization."""
    client = _cloud_client()
    state: dict[str, Any] = {"headers": {"Authorization": f"Bearer {JWT_A}"}}
    _credential_source(client, state)
    await client._ensure_session()

    state["headers"] = {"X-API-Key": KEY_B}
    await client.cancel_agent_optimization("opt-1")

    sent = _effective(wire[-1])
    assert sent.get("x-api-key") == KEY_B
    assert "authorization" not in sent


@pytest.mark.asyncio
async def test_session_defaults_hold_no_credentials_after_enter(wire):
    """(b1) ``__aenter__`` builds the session with no credential defaults."""
    client = _cloud_client()
    state: dict[str, Any] = {"headers": {"Authorization": f"Bearer {JWT_A}"}}
    _credential_source(client, state)

    await client.__aenter__()
    try:
        assert _defaults_credentials(client._aio_session) == set()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_defaults_hold_no_credentials_after_ensure_session(wire):
    """(b2) ``_ensure_session`` builds the session with no credential defaults."""
    client = _cloud_client()
    state: dict[str, Any] = {"headers": {"X-API-Key": KEY_B}}
    _credential_source(client, state)

    session = await client._ensure_session()
    assert _defaults_credentials(session) == set()


@pytest.mark.asyncio
async def test_delete_session_sends_current_credential_after_switch(wire):
    """(c) A path that used to rely on the session defaults sends the current key."""
    client = _cloud_client()
    state: dict[str, Any] = {"headers": {"Authorization": f"Bearer {JWT_A}"}}
    _credential_source(client, state)
    await client._ensure_session()

    state["headers"] = {"X-API-Key": KEY_B}
    assert await client.delete_session("session-1") is True

    record = wire[-1]
    assert record["method"] == "DELETE"
    sent = _effective(record)
    assert sent.get("x-api-key") == KEY_B
    assert "authorization" not in sent
