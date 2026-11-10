"""MCPServerConfig timeout / retry options must take effect (#3067)."""

from __future__ import annotations

import asyncio

import pytest

from traigent.cloud import production_mcp_client as pmc
from traigent.cloud.production_mcp_client import (
    MCPServerConfig,
    ProductionMCPClient,
    get_production_mcp_client,
    set_production_mcp_client,
)
from traigent.utils.retry import NetworkError

pytestmark = pytest.mark.backend_online


def _client(**cfg) -> ProductionMCPClient:
    return ProductionMCPClient(
        MCPServerConfig(server_path="python", **cfg), enable_fallback=False
    )


def test_retry_handler_uses_server_config() -> None:
    client = _client(max_retries=0, retry_delay=0.25)
    assert client._retry_handler.config.max_attempts == 1
    assert client._retry_handler.config.initial_delay == 0.25
    assert _client(max_retries=4)._retry_handler.config.max_attempts == 5


@pytest.mark.asyncio
async def test_session_call_is_bounded_by_timeout() -> None:
    client = _client(timeout=0.05, max_retries=0)

    class _Hung:
        async def call_tool(self, *_a):
            await asyncio.sleep(5)

    client._session = _Hung()
    client._connected = True

    async def _connected() -> bool:
        return True

    client.is_connected = _connected  # type: ignore[method-assign]
    with pytest.raises(NetworkError):
        await client._execute_tool_call("t", {}, "op")


def test_global_client_rejects_conflicting_args() -> None:
    original = pmc._production_client
    try:
        pmc._production_client = None
        first = get_production_mcp_client("python", ["a"], timeout=5.0)
        assert get_production_mcp_client() is first
        assert get_production_mcp_client(timeout=5.0) is first
        with pytest.raises(pmc.ValidationException):
            get_production_mcp_client("other")
        with pytest.raises(pmc.ValidationException):
            get_production_mcp_client(timeout=9.0)
    finally:
        set_production_mcp_client(original)  # type: ignore[arg-type]
        pmc._production_client = original
