"""Content-less session creation, checked at the wire.

Real BackendIntegratedClient and AuthManager objects. Only the backend key validation
and the aiohttp transport are replaced. Each test records what is actually sent (the
session defaults, the per-request headers, the JSON payload) and checks the ref cache
selection and writes against it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud import content_less as cl
from traigent.cloud._aiohttp_compat import aiohttp
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.auth import AuthManager
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest

pytestmark = pytest.mark.backend_online

URL = "https://api.test"
#: The API base the clients resolve ``URL`` to (the cache scope is keyed on it).
API_BASE = "https://api.test/api/v1"
KEY_A = "tg_" + "a" * 61
KEY_B = "tg_" + "b" * 61
JWT_A = "Bearer header.payload.signature"
AGENT_REF = "agentref_" + "ab" * 16
DATASET_REF = "datasetref_" + "cd" * 16
EVALUATOR_REF = "evaluatorref_" + "ef" * 16
_PUBLIC_ADDRINFO = [(2, 1, 6, "", ("93.184.216.34", 0))]  # AF_INET, SOCK_STREAM


def _body() -> dict[str, Any]:
    return {
        "session_id": "s1",
        "status": "active",
        "metadata": {"experiment_id": "e", "experiment_run_id": "r"},
        "tenant_id": "tenant_" + "1" * 32,
        "project_id": "project_" + "2" * 32,
        "capabilities": ["content_less_v1"],
        "agent_ref": AGENT_REF,
        "dataset_ref": DATASET_REF,
        "evaluator_ref": EVALUATOR_REF,
    }


def _request(**overrides: Any) -> SessionCreationRequest:
    base: dict[str, Any] = {
        "function_name": "pkg.mod.fn (abc123)",
        "agent_key": "agent-x",
        "configuration_space": {"temperature": [0.1, 0.9]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 2, "name": "evalset-x"},
        "metadata": {
            "function_name": "pkg/mod.py_fn",
            "function_display_name": "display",
            "function_slug": "slug_abc123",
            "evaluation_set": "evalset-x",
            "function_module": "pkg.mod",
        },
        "dataset_id": "dataset-x",
        "evaluator_id": "evaluator-x",
        "max_trials": 3,
        "content_less": cl.ContentLessOptions(enabled=True),
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


class _Response:
    status = 201

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._body

    async def text(self) -> str:
        return ""


class _Wire:
    """Records every POST; the session defaults are what aiohttp merges in."""

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

    def post(self, url: Any, *args: Any, **kwargs: Any) -> _Response:
        _Wire.sent.append(
            {
                "url": str(url),
                "defaults": dict(self.headers),
                "request_headers": dict(kwargs.get("headers") or {}),
                "json": kwargs.get("json"),
            }
        )
        return _Response(_body())


def _effective(record: dict[str, Any]) -> dict[str, str]:
    """What aiohttp puts on the wire: session defaults, overridden per request."""
    merged = {
        k: v
        for k, v in record["defaults"].items()
        if k.lower() not in {n.lower() for n in record["request_headers"]}
    }
    merged.update(record["request_headers"])
    return merged


async def _accept_key(self, api_key):  # noqa: ARG001 - backend validation boundary
    return None


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    _Wire.sent = []
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setattr(aiohttp, "ClientSession", _Wire)
    monkeypatch.setattr(
        "traigent.labels.transport.send_session_labels", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", _accept_key)
    return _Wire.sent


def _client(api_key: str) -> BackendIntegratedClient:
    with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
        return BackendIntegratedClient(api_key=api_key, base_url=URL)


def _cloud_client(api_key: str) -> TraigentCloudClient:
    with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
        return TraigentCloudClient(api_key=api_key, base_url=URL)


def _scope_index(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "content_less_refs.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["scope_index"])


def _principal(key: str) -> str:
    principal = cl.principal_for_headers(API_BASE, {"X-API-Key": key})
    assert principal is not None
    return principal


@pytest.mark.asyncio
async def test_single_key_refs_come_from_that_keys_scope(wire, tmp_path):
    client = _client(KEY_A)
    ops = ApiOperations(client)

    await ops.create_traigent_session_via_api(_request())
    first = wire[-1]
    assert _effective(first)["X-API-Key"] == KEY_A
    assert "agent_ref" not in first["json"]
    assert set(_scope_index(tmp_path)) == {_principal(KEY_A)}

    await ops.create_traigent_session_via_api(_request())
    second = wire[-1]
    assert _effective(second)["X-API-Key"] == KEY_A
    assert second["json"]["agent_ref"] == AGENT_REF
    assert set(_scope_index(tmp_path)) == {_principal(KEY_A)}


@pytest.mark.asyncio
async def test_cleared_key_then_env_key_sends_b_and_no_a_refs(
    wire, tmp_path, monkeypatch
):
    client = _client(KEY_A)
    ops = ApiOperations(client)
    await ops.create_traigent_session_via_api(_request())
    assert set(_scope_index(tmp_path)) == {_principal(KEY_A)}

    client.auth_manager.auth.set_api_key(None)
    monkeypatch.setenv("TRAIGENT_API_KEY", KEY_B)
    await ops.create_traigent_session_via_api(_request())

    sent = wire[-1]
    assert _effective(sent)["X-API-Key"] == KEY_B
    assert not {"agent_ref", "dataset_ref", "evaluator_ref"} & set(sent["json"])
    assert set(_scope_index(tmp_path)) == {_principal(KEY_A), _principal(KEY_B)}


@pytest.mark.asyncio
async def test_session_default_jwt_with_api_key_is_never_cached(
    wire, tmp_path, monkeypatch
):
    """Client path: an Authorization session default rides along with X-API-Key B."""
    client = _cloud_client(KEY_B)
    client.auth.set_api_key(KEY_B)
    client._aio_session = _Wire(headers={"Authorization": JWT_A})

    loads = []
    original_load = cl.RefCache._load
    monkeypatch.setattr(
        cl.RefCache, "_load", lambda self: loads.append(1) or original_load(self)
    )

    await client.create_optimization_session(_request())

    sent = wire[-1]
    effective = _effective(sent)
    assert effective["X-API-Key"] == KEY_B
    assert effective["Authorization"] == JWT_A
    assert "agent_ref" not in sent["json"]
    assert loads == []
    assert _scope_index(tmp_path) == {}
