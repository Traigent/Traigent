"""Follow-ups: cross-origin redirects drop X-Project-Id; 422 maps on every write path."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from aiohttp import web

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.labels.transport_gate import gate_session

pytestmark = pytest.mark.backend_online

BODY_422 = (
    '{"success":false,"error_code":"PROJECT_CONTEXT_REQUIRED_FOR_WRITE",'
    '"message":"customer text must not appear"}'
)
SENSITIVE = {
    "X-Project-Id": "proj_1",
    "X-API-Key": "k",  # pragma: allowlist secret
    "Authorization": "Bearer t",
}


async def _start(routes: list[web.RouteDef]):
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.fixture
async def servers():
    seen: dict[str, dict[str, str]] = {}

    async def target(request: web.Request) -> web.Response:
        seen["target"] = dict(request.headers)
        return web.json_response({"ok": True})

    async def same_final(request: web.Request) -> web.Response:
        seen["same"] = dict(request.headers)
        return web.json_response({"ok": True})

    runner_b, base_b = await _start(
        [web.post("/final", target), web.get("/final", target)]
    )

    async def cross(request: web.Request) -> web.Response:
        raise web.HTTPFound(f"{base_b}/final")

    async def same(request: web.Request) -> web.Response:
        raise web.HTTPFound("/same-final")

    runner_a, base_a = await _start(
        [
            web.post("/cross", cross),
            web.post("/same", same),
            web.get("/same-final", same_final),
        ]
    )
    yield base_a, seen
    await runner_a.cleanup()
    await runner_b.cleanup()


@pytest.mark.asyncio
async def test_cross_origin_redirect_drops_project_and_credentials(servers):
    base_a, seen = servers
    session = gate_session(aiohttp.ClientSession(), origins=[base_a])
    async with session:
        async with session.post(
            f"{base_a}/cross", json={"a": 1}, headers=dict(SENSITIVE)
        ) as response:
            assert response.status == 200
    lowered = {k.lower() for k in seen["target"]}
    assert "x-project-id" not in lowered
    assert "x-api-key" not in lowered
    assert "authorization" not in lowered


@pytest.mark.asyncio
async def test_same_origin_redirect_keeps_project_header(servers):
    base_a, seen = servers
    session = gate_session(aiohttp.ClientSession(), origins=[base_a])
    async with session:
        async with session.post(
            f"{base_a}/same", json={"a": 1}, headers=dict(SENSITIVE)
        ) as response:
            assert response.status == 200
    assert seen["same"]["X-Project-Id"] == "proj_1"
    assert seen["same"]["X-API-Key"] == "k"  # pragma: allowlist secret


def _fake_http(status: int, body: str):
    response = AsyncMock()
    response.status = status
    response.text = AsyncMock(return_value=body)
    response.json = AsyncMock(return_value={})
    response.headers = {}
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.post = Mock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _cloud_client(session) -> TraigentCloudClient:
    client = TraigentCloudClient(
        api_key="tg_" + "a" * 61,  # pragma: allowlist secret
        base_url="http://localhost:5000",
    )
    client._ensure_session = AsyncMock(return_value=None)
    client._get_headers = AsyncMock(
        return_value={"X-API-Key": "k"}
    )  # pragma: allowlist secret
    client._aio_session = session
    return client


CLOUD_WRITES = {
    "next_trial": lambda c: c.get_next_trial("s-1"),
    "results": lambda c: c.submit_trial_result("s-1", "t-1", {"accuracy": 1.0}, 1.0),
    "finalize": lambda c: c.finalize_optimization("s-1"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("write", list(CLOUD_WRITES))
async def test_direct_cloud_client_422_maps_to_clear_error(write):
    client = _cloud_client(_fake_http(422, BODY_422))
    with pytest.raises(CloudServiceError) as exc:
        await CLOUD_WRITES[write](client)
    assert "TRAIGENT_PROJECT_ID" in str(exc.value)
    assert "customer text" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("write", list(CLOUD_WRITES))
async def test_direct_cloud_client_other_422_stays_generic(write):
    client = _cloud_client(_fake_http(422, '{"error_code":"validation_error"}'))
    with pytest.raises(CloudServiceError) as exc:
        await CLOUD_WRITES[write](client)
    assert "TRAIGENT_PROJECT_ID" not in str(exc.value)


@pytest.mark.asyncio
async def test_hybrid_finalize_422_maps_to_clear_error():
    with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
        client = BackendIntegratedClient(
            api_key="tg_" + "a" * 61,  # pragma: allowlist secret
            base_url="http://localhost:5000",
        )
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"X-API-Key": "k"}  # pragma: allowlist secret
    )
    client._ensure_session = AsyncMock(return_value=_fake_http(422, BODY_422))
    with pytest.raises(CloudServiceError) as exc:
        await client._session_ops.finalize_hybrid_session("s-1")
    assert "TRAIGENT_PROJECT_ID" in str(exc.value)
