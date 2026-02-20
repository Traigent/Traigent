"""Unit tests for MCP transport implementation."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.hybrid.mcp_transport import CONFIG_SPACE_URI, HEALTH_URI, MCPTransport
from traigent.hybrid.protocol import (
    HybridEvaluateRequest,
    HybridExecuteRequest,
    ServiceCapabilities,
)
from traigent.hybrid.transport import TransportConnectionError, TransportError


@dataclass
class MockMCPResponse:
    """Mock response from MCP client."""

    success: bool
    data: dict | None = None
    error_message: str | None = None


class TestMCPTransportInit:
    """Tests for MCPTransport initialization."""

    def test_init_with_client(self) -> None:
        """Test initialization with existing MCP client."""
        mock_client = MagicMock()
        transport = MCPTransport(mcp_client=mock_client)

        assert transport._client is mock_client
        assert transport._owns_client is False
        assert transport._mcp_config is None
        assert transport._closed is False

    def test_init_with_config(self) -> None:
        """Test initialization with MCP config."""
        mock_config = MagicMock()
        transport = MCPTransport(mcp_config=mock_config)

        assert transport._client is None
        assert transport._owns_client is True
        assert transport._mcp_config is mock_config
        assert transport._closed is False

    def test_init_requires_client_or_config(self) -> None:
        """Test that init requires either client or config."""
        with pytest.raises(ValueError, match="Must provide either"):
            MCPTransport()

    def test_init_with_both_prefers_client(self) -> None:
        """Test that init prefers client when both provided."""
        mock_client = MagicMock()
        mock_config = MagicMock()
        transport = MCPTransport(mcp_client=mock_client, mcp_config=mock_config)

        assert transport._client is mock_client
        assert transport._owns_client is False


class TestMCPTransportGetClient:
    """Tests for MCP client creation."""

    @pytest.mark.asyncio
    async def test_get_client_returns_existing(self) -> None:
        """Test _get_client returns existing client."""
        mock_client = MagicMock()
        transport = MCPTransport(mcp_client=mock_client)

        result = await transport._get_client()
        assert result is mock_client

    @pytest.mark.asyncio
    async def test_get_client_creates_from_config(self) -> None:
        """Test _get_client creates client from config."""
        # Skip if MCP import fails due to pydantic version conflicts
        try:
            from traigent.cloud import production_mcp_client  # noqa: F401
        except (ImportError, KeyError):
            pytest.skip("MCP client not available due to import errors")

        mock_config = MagicMock()
        transport = MCPTransport(mcp_config=mock_config)

        with patch(
            "traigent.cloud.production_mcp_client.ProductionMCPClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            result = await transport._get_client()

            mock_cls.assert_called_once_with(mock_config)
            assert result is mock_client
            assert transport._owns_client is True

    @pytest.mark.asyncio
    async def test_get_client_raises_without_config(self) -> None:
        """Test _get_client raises error if config needed but missing."""
        # Skip if MCP import fails due to pydantic version conflicts
        try:
            from traigent.cloud import production_mcp_client  # noqa: F401
        except (ImportError, KeyError):
            pytest.skip("MCP client not available due to import errors")

        # Create transport with client, then clear it
        mock_client = MagicMock()
        transport = MCPTransport(mcp_client=mock_client)
        transport._client = None  # Simulate client was cleared
        transport._mcp_config = None  # Ensure no config

        with pytest.raises(ValueError, match="MCP config required"):
            await transport._get_client()


class TestMCPTransportReadResource:
    """Tests for MCP read_resource operations."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_read_resource_success(self, transport: MCPTransport) -> None:
        """Test successful resource read."""
        mock_response = MockMCPResponse(
            success=True,
            data={"content": '{"version": "1.0", "key": "value"}'},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        result = await transport._read_resource("traigent://test")

        assert result == {"version": "1.0", "key": "value"}
        transport._client.read_resource.assert_called_once_with("traigent://test")

    @pytest.mark.asyncio
    async def test_read_resource_failure(self, transport: MCPTransport) -> None:
        """Test resource read failure."""
        mock_response = MockMCPResponse(
            success=False,
            data={"error": "not found"},
            error_message="Resource not found",
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        with pytest.raises(TransportError) as exc_info:
            await transport._read_resource("traigent://test")

        assert "MCP read_resource failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_resource_no_content(self, transport: MCPTransport) -> None:
        """Test resource read with no content."""
        mock_response = MockMCPResponse(
            success=True,
            data={},  # No content field
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        with pytest.raises(TransportError) as exc_info:
            await transport._read_resource("traigent://test")

        assert "No content returned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_resource_invalid_json(self, transport: MCPTransport) -> None:
        """Test resource read with invalid JSON content."""
        mock_response = MockMCPResponse(
            success=True,
            data={"content": "not valid json {"},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        with pytest.raises(TransportError) as exc_info:
            await transport._read_resource("traigent://test")

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_resource_connection_error(
        self, transport: MCPTransport
    ) -> None:
        """Test resource read with connection error."""
        transport._client.read_resource = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(TransportConnectionError) as exc_info:
            await transport._read_resource("traigent://test")

        assert "MCP read_resource failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_resource_null_data(self, transport: MCPTransport) -> None:
        """Test resource read with null data."""
        mock_response = MockMCPResponse(
            success=True,
            data=None,
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        with pytest.raises(TransportError) as exc_info:
            await transport._read_resource("traigent://test")

        assert "No content returned" in str(exc_info.value)


class TestMCPTransportCallTool:
    """Tests for MCP call_tool operations."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_call_tool_success(self, transport: MCPTransport) -> None:
        """Test successful tool call."""
        mock_response = MockMCPResponse(
            success=True,
            data={"result": "success", "value": 42},
        )
        transport._client.call_tool = AsyncMock(return_value=mock_response)

        result = await transport._call_tool("test_tool", {"arg": "value"})

        assert result == {"result": "success", "value": 42}
        # Verify call_tool was called with correct args (including generated UUID)
        call_args = transport._client.call_tool.call_args
        assert call_args[0][0] == "test_tool"
        assert call_args[0][1] == {"arg": "value"}

    @pytest.mark.asyncio
    async def test_call_tool_failure(self, transport: MCPTransport) -> None:
        """Test tool call failure."""
        mock_response = MockMCPResponse(
            success=False,
            data={"error": "tool error"},
            error_message="Tool execution failed",
        )
        transport._client.call_tool = AsyncMock(return_value=mock_response)

        with pytest.raises(TransportError) as exc_info:
            await transport._call_tool("test_tool", {})

        assert "MCP call_tool(test_tool) failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_empty_data(self, transport: MCPTransport) -> None:
        """Test tool call with empty/null data returns empty dict."""
        mock_response = MockMCPResponse(
            success=True,
            data=None,
        )
        transport._client.call_tool = AsyncMock(return_value=mock_response)

        result = await transport._call_tool("test_tool", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_call_tool_connection_error(self, transport: MCPTransport) -> None:
        """Test tool call with connection error."""
        transport._client.call_tool = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        with pytest.raises(TransportConnectionError) as exc_info:
            await transport._call_tool("test_tool", {})

        assert "MCP call_tool(test_tool) failed" in str(exc_info.value)


class TestMCPTransportCapabilities:
    """Tests for capabilities method."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_capabilities_success(self, transport: MCPTransport) -> None:
        """Test fetching capabilities."""
        caps_data = {
            "version": "1.0",
            "supports_evaluate": True,
            "supports_keep_alive": True,
            "max_batch_size": 50,
        }
        mock_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        caps = await transport.capabilities()

        assert isinstance(caps, ServiceCapabilities)
        assert caps.version == "1.0"
        assert caps.supports_evaluate is True
        assert caps.supports_keep_alive is True
        assert caps.max_batch_size == 50

    @pytest.mark.asyncio
    async def test_capabilities_caching(self, transport: MCPTransport) -> None:
        """Test that capabilities are cached."""
        caps_data = {"version": "1.0"}
        mock_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        # First call
        await transport.capabilities()
        # Second call should use cache
        await transport.capabilities()

        # Should only call read_resource once
        assert transport._client.read_resource.call_count == 1

    @pytest.mark.asyncio
    async def test_capabilities_fallback_on_error(
        self, transport: MCPTransport
    ) -> None:
        """Test capabilities returns defaults on error."""
        mock_response = MockMCPResponse(
            success=False,
            error_message="Resource not found",
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        caps = await transport.capabilities()

        assert caps.version == "1.0"
        assert caps.supports_evaluate is True  # Default
        assert caps.supports_keep_alive is False  # Default


class TestMCPTransportDiscoverConfigSpace:
    """Tests for discover_config_space method."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_discover_config_space(self, transport: MCPTransport) -> None:
        """Test config space discovery."""
        config_space = {
            "schema_version": "0.9",
            "tunable_id": "test_agent",
            "tvars": [
                {"name": "model", "type": "enum", "domain": {"values": ["a", "b"]}},
            ],
        }
        mock_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(config_space)},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        result = await transport.discover_config_space()

        assert result.tunable_id == "test_agent"
        assert len(result.tvars) == 1
        transport._client.read_resource.assert_called_with(CONFIG_SPACE_URI)


class TestMCPTransportExecute:
    """Tests for execute method."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_execute(self, transport: MCPTransport) -> None:
        """Test execute method."""
        execute_response = {
            "request_id": "req-123",
            "execution_id": "exec-456",
            "status": "completed",
            "outputs": [{"input_id": "1", "output": {"result": "test"}}],
            "operational_metrics": {"total_cost_usd": 0.001},
        }
        mock_response = MockMCPResponse(
            success=True,
            data=execute_response,
        )
        transport._client.call_tool = AsyncMock(return_value=mock_response)

        request = HybridExecuteRequest(
            tunable_id="test_agent",
            config={"model": "fast"},
            inputs=[{"input_id": "1", "data": {}}],
        )
        response = await transport.execute(request)

        assert response.status == "completed"
        assert len(response.outputs) == 1
        # Verify call_tool was called with "execute"
        call_args = transport._client.call_tool.call_args
        assert call_args[0][0] == "execute"


class TestMCPTransportEvaluate:
    """Tests for evaluate method."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_evaluate(self, transport: MCPTransport) -> None:
        """Test evaluate method."""
        # Setup capabilities to support evaluate
        caps_data = {"version": "1.0", "supports_evaluate": True}
        caps_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )

        evaluate_response = {
            "request_id": "req-123",
            "status": "completed",
            "results": [{"input_id": "1", "metrics": {"accuracy": 0.95}}],
            "aggregate_metrics": {"accuracy": {"mean": 0.95}},
        }
        eval_response = MockMCPResponse(
            success=True,
            data=evaluate_response,
        )

        transport._client.read_resource = AsyncMock(return_value=caps_response)
        transport._client.call_tool = AsyncMock(return_value=eval_response)

        request = HybridEvaluateRequest(
            tunable_id="test_agent",
            evaluations=[{"input_id": "1", "output": {}, "target": {}}],
        )
        response = await transport.evaluate(request)

        assert response.status == "completed"
        assert len(response.results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_not_supported(self, transport: MCPTransport) -> None:
        """Test evaluate raises error when not supported."""
        caps_data = {"version": "1.0", "supports_evaluate": False}
        mock_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        request = HybridEvaluateRequest(
            tunable_id="test_agent",
            evaluations=[],
        )

        with pytest.raises(NotImplementedError):
            await transport.evaluate(request)


class TestMCPTransportHealthCheck:
    """Tests for health_check method."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_health_check(self, transport: MCPTransport) -> None:
        """Test health check method."""
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
        }
        mock_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(health_data)},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        health = await transport.health_check()

        assert health.status == "healthy"
        assert health.version == "1.0.0"
        transport._client.read_resource.assert_called_with(HEALTH_URI)


class TestMCPTransportKeepAlive:
    """Tests for keep_alive method."""

    @pytest.fixture
    def transport(self) -> MCPTransport:
        """Create transport with mock client."""
        mock_client = AsyncMock()
        return MCPTransport(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_keep_alive(self, transport: MCPTransport) -> None:
        """Test keep-alive method."""
        caps_data = {"version": "1.0", "supports_keep_alive": True}
        caps_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )

        keep_alive_response = MockMCPResponse(
            success=True,
            data={"alive": True, "session_id": "session-123"},
        )

        transport._client.read_resource = AsyncMock(return_value=caps_response)
        transport._client.call_tool = AsyncMock(return_value=keep_alive_response)

        alive = await transport.keep_alive("session-123")

        assert alive is True

    @pytest.mark.asyncio
    async def test_keep_alive_expired(self, transport: MCPTransport) -> None:
        """Test keep-alive with expired session."""
        caps_data = {"version": "1.0", "supports_keep_alive": True}
        caps_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )

        keep_alive_response = MockMCPResponse(
            success=True,
            data={"alive": False, "reason": "expired"},
        )

        transport._client.read_resource = AsyncMock(return_value=caps_response)
        transport._client.call_tool = AsyncMock(return_value=keep_alive_response)

        alive = await transport.keep_alive("session-123")

        assert alive is False

    @pytest.mark.asyncio
    async def test_keep_alive_not_supported(self, transport: MCPTransport) -> None:
        """Test keep-alive when not supported."""
        caps_data = {"version": "1.0", "supports_keep_alive": False}
        mock_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )
        transport._client.read_resource = AsyncMock(return_value=mock_response)

        with pytest.raises(NotImplementedError):
            await transport.keep_alive("session-123")

    @pytest.mark.asyncio
    async def test_keep_alive_session_not_found(self, transport: MCPTransport) -> None:
        """Test keep-alive returns False when session not found."""
        caps_data = {"version": "1.0", "supports_keep_alive": True}
        caps_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )

        transport._client.read_resource = AsyncMock(return_value=caps_response)
        transport._client.call_tool = AsyncMock(
            side_effect=TransportError("Session not found")
        )

        alive = await transport.keep_alive("session-123")

        assert alive is False

    @pytest.mark.asyncio
    async def test_keep_alive_with_status_alive(self, transport: MCPTransport) -> None:
        """Keep-alive with status='alive' returns True (not backward-compat path)."""
        caps_data = {"version": "1.0", "supports_keep_alive": True}
        caps_response = MockMCPResponse(
            success=True,
            data={"content": json.dumps(caps_data)},
        )

        keep_alive_response = MockMCPResponse(
            success=True,
            data={"status": "alive", "session_id": "session-123"},
        )

        transport._client.read_resource = AsyncMock(return_value=caps_response)
        transport._client.call_tool = AsyncMock(return_value=keep_alive_response)

        alive = await transport.keep_alive("session-123")

        assert alive is True


class TestMCPTransportClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_owned_client(self) -> None:
        """Test closing transport with owned client."""
        mock_config = MagicMock()
        transport = MCPTransport(mcp_config=mock_config)

        # Manually set up a mock client to simulate owned client scenario
        mock_client = AsyncMock()
        transport._client = mock_client
        transport._owns_client = True

        await transport.close()

        mock_client.disconnect.assert_called_once()
        assert transport._closed is True
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_close_external_client(self) -> None:
        """Test closing transport with external client doesn't disconnect."""
        mock_client = AsyncMock()
        transport = MCPTransport(mcp_client=mock_client)

        await transport.close()

        # Should NOT call disconnect since we don't own the client
        mock_client.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        """Test that close is idempotent."""
        mock_config = MagicMock()
        transport = MCPTransport(mcp_config=mock_config)

        # Manually set up a mock client
        mock_client = AsyncMock()
        transport._client = mock_client
        transport._owns_client = True

        await transport.close()
        await transport.close()  # Second close should be safe

        # Only one disconnect call
        assert mock_client.disconnect.call_count == 1

    @pytest.mark.asyncio
    async def test_close_handles_disconnect_error(self) -> None:
        """Test close handles disconnect errors gracefully."""
        mock_config = MagicMock()
        transport = MCPTransport(mcp_config=mock_config)

        # Manually set up a mock client that fails on disconnect
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock(side_effect=Exception("Disconnect error"))
        transport._client = mock_client
        transport._owns_client = True

        # Should not raise, just log warning
        await transport.close()

        assert transport._closed is True


class TestMCPTransportContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        mock_client = AsyncMock()

        # Create transport with external client (simpler for testing)
        async with MCPTransport(mcp_client=mock_client) as transport:
            assert transport._closed is False

        # After context exit, close was called (but no disconnect for external client)
        # Transport is still marked as not closed since we don't own the client

    @pytest.mark.asyncio
    async def test_context_manager_with_owned_client(self) -> None:
        """Test async context manager with owned client."""
        mock_config = MagicMock()
        transport = MCPTransport(mcp_config=mock_config)

        # Manually set up a mock client
        mock_client = AsyncMock()
        transport._client = mock_client
        transport._owns_client = True

        async with transport:
            assert transport._closed is False

        # After context exit, should be closed
        assert transport._closed is True
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        """Test context manager returns transport instance."""
        mock_client = MagicMock()

        async with MCPTransport(mcp_client=mock_client) as transport:
            assert isinstance(transport, MCPTransport)
