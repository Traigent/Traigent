"""Finalization sends a terminal cause over the actual local HTTP transport."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from traigent.cloud.session_operations import SessionOperations

pytestmark = pytest.mark.backend_online


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["cost_limit", "error", "timeout", None])
async def test_stop_reason_uses_top_level_finalize_field(monkeypatch, stop_reason):
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    payloads = []

    async def finalize(request):
        payloads.append(await request.json())
        return web.json_response({"status": "completed"})

    app = web.Application()
    app.router.add_post("/sessions/local-stop-control/finalize", finalize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        client = SimpleNamespace(
            auth_manager=SimpleNamespace(augment_headers=AsyncMock(return_value={})),
            backend_config=SimpleNamespace(api_base_url=f"http://127.0.0.1:{port}"),
        )
        operations = SessionOperations(client)
        result = await operations._finalize_session_via_api(
            "local-stop-control",
            "run-local-control",
            **({"stop_reason": stop_reason} if stop_reason is not None else {}),
        )
        assert result == {"status": "completed"}
        expected = {
            "reason": "sdk_explicit_finalization",
            "experiment_run_id": "run-local-control",
        }
        if stop_reason is not None:
            expected["stop_reason"] = stop_reason
        assert payloads == [expected]
    finally:
        await runner.cleanup()
