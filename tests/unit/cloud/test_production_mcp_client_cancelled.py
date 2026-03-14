"""CancelledError regression tests for ProductionMCPClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from traigent.cloud.production_mcp_client import MCPServerConfig, ProductionMCPClient


@pytest.mark.asyncio
async def test_call_tool_reraises_cancelled_error() -> None:
    """call_tool should not convert task cancellation into an error response."""
    client = ProductionMCPClient(
        MCPServerConfig(server_path="python"),
        enable_fallback=False,
    )
    client._retry_handler.execute_async = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await client.call_tool("test_tool", {"arg": "value"})
