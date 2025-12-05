"""Validation tests for ProductionMCPClient inputs."""

import pytest

from traigent.cloud.production_mcp_client import MCPServerConfig, ProductionMCPClient
from traigent.utils.exceptions import ValidationError as ValidationException


def test_mcp_server_config_rejects_invalid_values():
    with pytest.raises(ValidationException):
        MCPServerConfig(server_path="")

    with pytest.raises(ValidationException):
        MCPServerConfig(server_path="python", timeout=0)

    with pytest.raises(ValidationException):
        MCPServerConfig(server_path="python", max_retries=-1)

    with pytest.raises(ValidationException):
        MCPServerConfig(server_path="python", server_args=["", "arg"])


@pytest.mark.asyncio
async def test_call_tool_validates_inputs(monkeypatch):
    config = MCPServerConfig(server_path="python")
    client = ProductionMCPClient(config)

    response = await client.call_tool("", {})
    assert response.success is False
    assert "tool_name" in (response.error_message or "")

    response = await client.call_tool("create_agent", 123)  # type: ignore[arg-type]
    assert response.success is False
    assert response.request_id is not None
