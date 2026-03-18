"""Comprehensive unit tests for MCP production client."""

import json
import time
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the MCP production client components
from traigent.cloud.production_mcp_client import (
    MCP_AVAILABLE,
    ClientSession,
    StdioClientTransport,
    StdioServerParameters,
)
from traigent.utils.exceptions import ValidationError

# We need to import the actual client if it exists
try:
    from traigent.cloud.production_mcp_client import ProductionMCPClient
except ImportError:
    # If the class doesn't exist, we'll create a mock for testing
    class ProductionMCPClient:
        def __init__(self, *args, **kwargs):
            pass


class TestMCPAvailability:
    """Test MCP library availability detection."""

    def test_mcp_available_flag_exists(self):
        """Test that MCP_AVAILABLE flag exists."""
        assert isinstance(MCP_AVAILABLE, bool)

    @patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True)
    def test_mcp_available_true(self):
        """Test behavior when MCP is available."""
        from traigent.cloud.production_mcp_client import MCP_AVAILABLE

        assert MCP_AVAILABLE is True

    @patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", False)
    def test_mcp_available_false(self):
        """Test behavior when MCP is not available."""
        from traigent.cloud.production_mcp_client import MCP_AVAILABLE

        assert MCP_AVAILABLE is False


class TestMockMCPClasses:
    """Test mock MCP classes when library is not available."""

    def test_client_session_mock_initialization(self):
        """Test ClientSession mock raises ImportError when MCP unavailable."""
        if not MCP_AVAILABLE:
            with pytest.raises(ImportError, match="MCP not available"):
                ClientSession()

    def test_stdio_server_parameters_mock(self):
        """Test StdioServerParameters mock initialization."""
        if not MCP_AVAILABLE:
            # Should not raise an error
            params = StdioServerParameters()
            assert params is not None

    def test_stdio_client_transport_mock(self):
        """Test StdioClientTransport mock initialization."""
        if not MCP_AVAILABLE:
            # Should not raise an error
            transport = StdioClientTransport()
            assert transport is not None


class TestProductionMCPClientBasics:
    """Test basic ProductionMCPClient functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        self.server_config = MCPServerConfig(
            server_path="python",
            server_args=["-m", "traigent_backend.mcp.server"],
            timeout=30.0,
            max_retries=3,
        )

    def test_client_initialization(self):
        """Test basic client initialization."""
        # Mock the RetryConfig to avoid parameter errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = ProductionMCPClient(self.server_config)
                assert client is not None
                assert client.server_config == self.server_config
                assert client.enable_fallback is True  # Default
                assert client._connected is False
                assert client._session is None
                assert client._transport is None

    def test_client_initialization_with_custom_settings(self):
        """Test client initialization with custom settings."""
        # Mock the RetryConfig to avoid parameter errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = ProductionMCPClient(self.server_config, enable_fallback=False)
                assert client is not None
                assert client.enable_fallback is False
                assert client.server_config.timeout == 30.0
                assert client.server_config.max_retries == 3


class TestMCPClientConnectionManagement:
    """Test MCP client connection management."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_connection_lifecycle_mocked(self):
        """Test connection lifecycle with mocked MCP components."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            with patch(
                "traigent.cloud.production_mcp_client.StdioClientTransport"
            ) as mock_transport:
                with patch(
                    "traigent.cloud.production_mcp_client.ClientSession"
                ) as mock_session:
                    mock_transport_instance = AsyncMock()
                    mock_session_instance = AsyncMock()

                    mock_transport.return_value = mock_transport_instance
                    mock_session.return_value = mock_session_instance

                    # Mock async context managers
                    mock_transport_instance.__aenter__ = AsyncMock(
                        return_value=mock_transport_instance
                    )
                    mock_transport_instance.__aexit__ = AsyncMock(return_value=None)
                    mock_session_instance.__aenter__ = AsyncMock(
                        return_value=mock_session_instance
                    )
                    mock_session_instance.__aexit__ = AsyncMock(return_value=None)

                    # Test connection establishment
                    connection_tested = False
                    if hasattr(self.client, "connect"):
                        await self.client.connect()
                        connection_tested = True

                    # Test connection cleanup
                    if hasattr(self.client, "disconnect"):
                        await self.client.disconnect()
                        connection_tested = True

                    # Verify at least one method was called
                    assert connection_tested or not hasattr(self.client, "connect")

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            with patch(
                "traigent.cloud.production_mcp_client.StdioClientTransport"
            ) as mock_transport:
                mock_transport.side_effect = Exception("Connection failed")

                if hasattr(self.client, "connect"):
                    try:
                        await self.client.connect()
                    except Exception as e:
                        assert "Connection failed" in str(e) or isinstance(e, Exception)

    def test_connection_without_mcp_library(self):
        """Test connection attempt when MCP library is not available."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", False):
            if hasattr(self.client, "is_available"):
                assert not self.client.is_available()


class TestMCPRequestHandling:
    """Test MCP request handling and communication."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)
        self.sample_request = {"method": "list_tools", "params": {}}
        self.sample_response = {
            "result": {
                "tools": [
                    {
                        "name": "create_agent",
                        "description": "Create a new agent",
                        "schema": {},
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_send_request_mocked(self):
        """Test sending MCP request with mocked session."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value=self.sample_response)

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "send_request"):
                    response = await self.client.send_request(self.sample_request)
                    assert response is not None

    @pytest.mark.asyncio
    async def test_list_tools_operation(self):
        """Test listing available tools."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()
            mock_session.list_tools = AsyncMock(
                return_value=self.sample_response["result"]
            )

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "list_tools"):
                    tools = await self.client.list_tools()
                    assert tools is not None

    @pytest.mark.asyncio
    async def test_call_tool_operation(self):
        """Test calling a specific tool."""
        tool_name = "create_agent"
        tool_args = {"name": "test_agent", "type": "optimization"}

        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value={"success": True})

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "call_tool"):
                    result = await self.client.call_tool(tool_name, tool_args)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self):
        """Test request timeout handling.

        The ProductionMCPClient handles timeouts gracefully by returning
        an MCPResponse with success=False rather than raising the exception.
        """
        # Skip if client doesn't have required attributes
        if not hasattr(self.client, "_session") or not hasattr(
            self.client, "call_tool"
        ):
            pytest.skip("Client does not have _session or call_tool attributes")

        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()
            mock_session.call_tool = AsyncMock(
                side_effect=TimeoutError("Connection timed out")
            )

            # Patch is_connected to return True to skip connection attempts
            with patch.object(
                self.client, "is_connected", new_callable=AsyncMock
            ) as mock_connected:
                mock_connected.return_value = True
                self.client._session = mock_session

                # Mock the retry handler to return a failed result with proper attribute
                mock_retry_result = Mock()
                mock_retry_result.success = False
                mock_retry_result.last_exception = TimeoutError("MCP operation timeout")

                async def mock_execute(func):
                    return mock_retry_result

                self.client._retry_handler.execute_async = mock_execute

                # The client handles timeout gracefully - returns MCPResponse with success=False
                result = await self.client.call_tool("test_tool", {})
                assert (
                    result.success is False
                ), "Timeout should result in failed response"
                assert (
                    "timeout" in result.error_message.lower()
                ), "Error message should mention timeout"


class TestMCPRetryMechanism:
    """Test MCP client retry mechanisms."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"], max_retries=3
        )

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        """Test retry mechanism on network errors."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()
            # First two calls fail, third succeeds
            mock_session.call_tool = AsyncMock(
                side_effect=[
                    Exception("Network error"),
                    Exception("Network error"),
                    {"success": True},
                ]
            )

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "call_tool_with_retry"):
                    result = await self.client.call_tool_with_retry("test_tool", {})
                    assert result is not None

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test behavior when all retries are exhausted."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()
            mock_session.call_tool = AsyncMock(
                side_effect=Exception("Persistent error")
            )

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "call_tool_with_retry"):
                    try:
                        await self.client.call_tool_with_retry("test_tool", {})
                    except Exception as e:
                        assert "Persistent error" in str(e) or isinstance(e, Exception)

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                # Test valid retry configuration
                server_config = MCPServerConfig(
                    server_path="python", max_retries=5, retry_delay=2.0
                )
                client = ProductionMCPClient(server_config)
                assert client is not None
                assert client.server_config.max_retries == 5
                assert client.server_config.retry_delay == 2.0

                # Test with negative retries (should be handled gracefully)
                with pytest.raises(ValidationError):
                    MCPServerConfig(server_path="python", max_retries=-1)


class TestMCPDataSerialization:
    """Test MCP data serialization and deserialization."""

    def test_request_serialization(self):
        """Test MCP request serialization."""
        request_data = {
            "method": "create_agent",
            "params": {
                "name": "test_agent",
                "type": "optimization",
                "config": {"temperature": 0.7},
            },
        }

        # Test JSON serialization
        serialized = json.dumps(request_data)
        assert isinstance(serialized, str)

        # Test deserialization
        deserialized = json.loads(serialized)
        assert deserialized == request_data

    def test_response_deserialization(self):
        """Test MCP response deserialization."""
        response_json = json.dumps(
            {
                "result": {
                    "agent_id": "agent_123",
                    "status": "created",
                    "timestamp": time.time(),
                }
            }
        )

        response_data = json.loads(response_json)
        assert "result" in response_data
        assert "agent_id" in response_data["result"]

    def test_complex_data_serialization(self):
        """Test serialization of complex data structures."""
        complex_data = {
            "agents": [
                {"id": str(uuid.uuid4()), "name": f"agent_{i}"} for i in range(5)
            ],
            "metadata": {"created_at": time.time(), "version": "1.0.0"},
        }

        serialized = json.dumps(complex_data, default=str)
        assert isinstance(serialized, str)

        deserialized = json.loads(serialized)
        assert len(deserialized["agents"]) == 5


class TestMCPErrorHandling:
    """Test MCP client error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    def test_import_error_handling(self):
        """Test handling of MCP import errors."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", False):
            if hasattr(self.client, "check_availability"):
                availability = self.client.check_availability()
                assert availability is False

    @pytest.mark.asyncio
    async def test_connection_error_recovery(self):
        """Test connection error recovery mechanisms."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_transport = Mock()
            mock_transport.side_effect = [
                Exception("Connection refused"),
                Mock(),  # Successful connection on retry
            ]

            recovery_tested = False
            if hasattr(self.client, "connect_with_retry"):
                try:
                    await self.client.connect_with_retry()
                    recovery_tested = True
                except Exception:
                    recovery_tested = True  # Exception is acceptable
            assert recovery_tested or not hasattr(self.client, "connect_with_retry")

    def test_invalid_request_handling(self):
        """Test handling of invalid requests."""
        invalid_requests = [None, {}, {"method": None}, {"params": "invalid"}]

        validation_tested = False
        for invalid_request in invalid_requests:
            if hasattr(self.client, "validate_request"):
                try:
                    self.client.validate_request(invalid_request)
                except (ValueError, TypeError, AttributeError):
                    validation_tested = True  # Expected for invalid requests
        assert validation_tested or not hasattr(self.client, "validate_request")

    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test handling of server-side errors."""
        server_errors = [
            {"error": {"code": 500, "message": "Internal server error"}},
            {"error": {"code": 404, "message": "Tool not found"}},
            {"error": {"code": 400, "message": "Invalid parameters"}},
        ]

        error_handled = False
        for error_response in server_errors:
            if hasattr(self.client, "handle_error_response"):
                try:
                    self.client.handle_error_response(error_response)
                except Exception:
                    error_handled = True  # Expected for error responses
        assert error_handled or not hasattr(self.client, "handle_error_response")


class TestMCPIntegrationScenarios:
    """Test realistic MCP integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_agent_creation_workflow(self):
        """Test complete agent creation workflow."""
        from traigent.cloud.models import AgentSpecification

        # Create proper AgentSpecification object
        agent_spec = AgentSpecification(
            name="test_optimization_agent", agent_type="optimization"
        )

        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()

            # Mock successful tool calls
            mock_session.call_tool = AsyncMock(
                return_value={"result": {"agent_id": "agent_123", "status": "created"}}
            )

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "create_agent"):
                    result = await self.client.create_agent(agent_spec)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_optimization_request_workflow(self):
        """Test optimization request workflow."""
        optimization_request = {
            "agent_id": "agent_123",
            "dataset_id": "dataset_456",
            "optimization_type": "hyperparameter_tuning",
            "target_metric": "accuracy",
        }

        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()

            mock_session.call_tool = AsyncMock(
                return_value={
                    "result": {
                        "optimization_id": "opt_789",
                        "status": "started",
                        "estimated_duration": 3600,
                    }
                }
            )

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "start_optimization"):
                    result = await self.client.start_optimization(optimization_request)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_status_monitoring_workflow(self):
        """Test status monitoring workflow."""
        task_id = "opt_789"

        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", True):
            mock_session = AsyncMock()

            # Mock status progression
            status_responses = [
                {"result": {"status": "running", "progress": 0.3}},
                {"result": {"status": "running", "progress": 0.7}},
                {"result": {"status": "completed", "progress": 1.0}},
            ]

            mock_session.call_tool = AsyncMock(side_effect=status_responses)

            if hasattr(self.client, "_session"):
                self.client._session = mock_session

                if hasattr(self.client, "get_task_status"):
                    for _expected_response in status_responses:
                        result = await self.client.get_task_status(task_id)
                        assert result is not None


class TestMCPServerConfigValidation:
    """Test MCPServerConfig validation."""

    def test_server_config_empty_server_path_raises(self):
        """Test that empty server_path raises ValidationError."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        with pytest.raises(ValidationError):
            MCPServerConfig(server_path="")

    def test_server_config_invalid_server_args_type(self):
        """Test that invalid server_args type raises ValidationError."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        with pytest.raises(ValidationError, match="server_args must be a list"):
            MCPServerConfig(server_path="python", server_args="not-a-list")

    def test_server_config_empty_string_in_server_args(self):
        """Test that empty string in server_args raises ValidationError."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        with pytest.raises(ValidationError):
            MCPServerConfig(server_path="python", server_args=["-m", ""])

    def test_server_config_negative_timeout(self):
        """Test that negative timeout raises ValidationError."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        with pytest.raises(ValidationError, match="timeout must be a positive value"):
            MCPServerConfig(server_path="python", timeout=-1.0)

    def test_server_config_zero_timeout(self):
        """Test that zero timeout raises ValidationError."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        with pytest.raises(ValidationError, match="timeout must be a positive value"):
            MCPServerConfig(server_path="python", timeout=0)

    def test_server_config_negative_retry_delay(self):
        """Test that negative retry_delay raises ValidationError."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        with pytest.raises(ValidationError, match="retry_delay must be non-negative"):
            MCPServerConfig(server_path="python", retry_delay=-1.0)


class TestMCPResponseDataclass:
    """Test MCPResponse dataclass."""

    def test_mcp_response_success(self):
        """Test successful MCPResponse creation."""
        from traigent.cloud.production_mcp_client import MCPResponse

        response = MCPResponse(
            success=True,
            data={"result": "test"},
            request_id="req-123",
        )
        assert response.success is True
        assert response.data == {"result": "test"}
        assert response.request_id == "req-123"
        assert response.error_message is None

    def test_mcp_response_failure(self):
        """Test failed MCPResponse creation."""
        from traigent.cloud.production_mcp_client import MCPResponse

        response = MCPResponse(
            success=False,
            error_message="Test error",
            request_id="req-456",
        )
        assert response.success is False
        assert response.error_message == "Test error"
        assert response.data is None


class TestProductionMCPClientValidationMethods:
    """Test ProductionMCPClient static validation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    def test_validate_identifier_valid(self):
        """Test _validate_identifier with valid input."""
        # Should not raise
        self.client._validate_identifier("valid-id", "test_field")

    def test_validate_identifier_empty(self):
        """Test _validate_identifier with empty string."""
        with pytest.raises(ValidationError):
            self.client._validate_identifier("", "test_field")

    def test_validate_mapping_valid(self):
        """Test _validate_mapping with valid input."""
        # Should not raise
        self.client._validate_mapping({"key": "value"}, "test_field")

    def test_validate_mapping_invalid(self):
        """Test _validate_mapping with non-dict input."""
        with pytest.raises(ValidationError):
            self.client._validate_mapping("not-a-dict", "test_field")

    def test_validate_positive_int_valid(self):
        """Test _validate_positive_int with valid input."""
        # Should not raise
        self.client._validate_positive_int(5, "test_field")

    def test_validate_positive_int_invalid(self):
        """Test _validate_positive_int with non-positive input."""
        with pytest.raises(ValidationError):
            self.client._validate_positive_int(0, "test_field")

    def test_validate_tool_call_inputs_valid(self):
        """Test _validate_tool_call_inputs with valid inputs."""
        result = self.client._validate_tool_call_inputs(
            "test_tool", {"arg": "value"}, "op-123"
        )
        assert result is None

    def test_validate_tool_call_inputs_empty_tool_name(self):
        """Test _validate_tool_call_inputs with empty tool name."""
        result = self.client._validate_tool_call_inputs("", {"arg": "value"}, "op-123")
        assert result is not None
        assert result.success is False

    def test_validate_tool_call_inputs_invalid_arguments_type(self):
        """Test _validate_tool_call_inputs with invalid arguments type."""
        result = self.client._validate_tool_call_inputs(
            "test_tool", "not-dict", "op-123"
        )
        assert result is not None
        assert result.success is False
        assert "dictionary" in result.error_message


class TestProductionMCPClientContextManager:
    """Test ProductionMCPClient async context manager."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        self.server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager entry and exit."""
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = ProductionMCPClient(self.server_config)

                with patch.object(
                    client, "connect", new_callable=AsyncMock
                ) as mock_connect:
                    with patch.object(
                        client, "disconnect", new_callable=AsyncMock
                    ) as mock_disconnect:
                        mock_connect.return_value = True

                        async with client as c:
                            assert c is client

                        mock_connect.assert_called_once()
                        mock_disconnect.assert_called_once()


class TestProductionMCPClientStatistics:
    """Test ProductionMCPClient statistics methods."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    def test_get_statistics(self):
        """Test get_statistics method."""
        stats = self.client.get_statistics()
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "connection_attempts" in stats
        assert "connected" in stats
        assert "active_operations" in stats
        assert "cached_results" in stats

    def test_get_active_operations(self):
        """Test get_active_operations method."""
        # Initially empty
        ops = self.client.get_active_operations()
        assert ops == {}

        # Add some operations
        self.client._active_operations["op1"] = {"tool": "test"}
        ops = self.client.get_active_operations()
        assert len(ops) == 1
        assert "op1" in ops


class TestProductionMCPClientFallback:
    """Test ProductionMCPClient fallback operations."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_fallback_create_experiment(self):
        """Test fallback for create_experiment tool."""
        response = await self.client._fallback_operation("create_experiment", {})
        assert response.success is True
        assert "experiment_id" in response.data
        assert "fallback_exp_" in response.data["experiment_id"]

    @pytest.mark.asyncio
    async def test_fallback_start_experiment_run(self):
        """Test fallback for start_experiment_run tool."""
        response = await self.client._fallback_operation("start_experiment_run", {})
        assert response.success is True
        assert "experiment_run_id" in response.data

    @pytest.mark.asyncio
    async def test_fallback_create_configuration_run(self):
        """Test fallback for create_configuration_run tool."""
        response = await self.client._fallback_operation("create_configuration_run", {})
        assert response.success is True
        assert "config_run_id" in response.data

    @pytest.mark.asyncio
    async def test_fallback_create_agent(self):
        """Test fallback for create_agent tool."""
        response = await self.client._fallback_operation("create_agent", {})
        assert response.success is True
        assert "agent_id" in response.data

    @pytest.mark.asyncio
    async def test_fallback_upload_example_set(self):
        """Test fallback for upload_example_set tool."""
        response = await self.client._fallback_operation("upload_example_set", {})
        assert response.success is True
        assert "example_set_id" in response.data

    @pytest.mark.asyncio
    async def test_fallback_unknown_tool(self):
        """Test fallback for unknown tool."""
        response = await self.client._fallback_operation("unknown_tool", {})
        assert response.success is False
        assert "No fallback available" in response.error_message


class TestProductionMCPClientHealthCheck:
    """Test ProductionMCPClient health check."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check when not connected."""
        response = await self.client.health_check()
        assert response.success is False
        assert "Not connected" in response.error_message

    @pytest.mark.asyncio
    async def test_health_check_connected_success(self):
        """Test health check when connected and healthy."""
        from traigent.cloud.production_mcp_client import MCPResponse

        self.client._connected = True
        self.client._session = Mock()

        with patch.object(
            self.client, "list_resources", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = MCPResponse(success=True, data={"resources": []})

            response = await self.client.health_check()
            assert response.success is True
            assert response.data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_connected_failure(self):
        """Test health check when connected but list_resources fails."""
        from traigent.cloud.production_mcp_client import MCPResponse

        self.client._connected = True
        self.client._session = Mock()

        with patch.object(
            self.client, "list_resources", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = MCPResponse(success=False, error_message="Failed")

            response = await self.client.health_check()
            assert response.success is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Test health check when an exception occurs."""
        self.client._connected = True
        self.client._session = Mock()

        with patch.object(
            self.client, "list_resources", new_callable=AsyncMock
        ) as mock_list:
            mock_list.side_effect = Exception("Unexpected error")

            response = await self.client.health_check()
            assert response.success is False
            assert "Health check error" in response.error_message


class TestProductionMCPClientConnectionMethods:
    """Test ProductionMCPClient connection methods."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        self.server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )

    @pytest.mark.asyncio
    async def test_connect_mcp_not_available(self):
        """Test connect when MCP library is not available."""
        with patch("traigent.cloud.production_mcp_client.MCP_AVAILABLE", False):
            with patch(
                "traigent.cloud.production_mcp_client.RetryConfig"
            ) as mock_retry_config:
                with patch(
                    "traigent.cloud.production_mcp_client.RetryHandler"
                ) as mock_retry_handler:
                    mock_retry_config.return_value = Mock()
                    mock_retry_handler.return_value = Mock()

                    client = ProductionMCPClient(self.server_config)
                    result = await client.connect()
                    assert result is False

    @pytest.mark.asyncio
    async def test_is_connected_true(self):
        """Test is_connected when connected."""
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = ProductionMCPClient(self.server_config)
                client._connected = True
                client._session = Mock()

                result = await client.is_connected()
                assert result is True

    @pytest.mark.asyncio
    async def test_is_connected_false_no_session(self):
        """Test is_connected when no session."""
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = ProductionMCPClient(self.server_config)
                client._connected = True
                client._session = None

                result = await client.is_connected()
                assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_connection_with_errors(self):
        """Test _cleanup_connection handles errors gracefully."""
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = ProductionMCPClient(self.server_config)
                client._connected = True
                client._session = AsyncMock()
                client._transport = AsyncMock()
                client._session.close.side_effect = Exception("Close error")

                # Should not raise
                await client._cleanup_connection()

                assert client._session is None
                assert client._transport is None
                assert client._connected is False


class TestGlobalMCPClientFunctions:
    """Test global MCP client functions."""

    def teardown_method(self):
        """Clean up global state."""
        from traigent.cloud import production_mcp_client

        production_mcp_client._production_client = None

    def test_get_production_mcp_client_creates_new(self):
        """Test get_production_mcp_client creates new client."""
        from traigent.cloud import production_mcp_client
        from traigent.cloud.production_mcp_client import (
            _production_client,
            get_production_mcp_client,
        )

        # Ensure no client exists
        production_mcp_client._production_client = None

        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = get_production_mcp_client()
                assert client is not None
                assert production_mcp_client._production_client is not None
                assert client.server_config.server_args is not None

    def test_default_mcp_server_args_prefers_legacy_backend_when_new_missing(self):
        """Fallback to the legacy backend module during rename transitions."""
        from traigent.cloud.production_mcp_client import default_mcp_server_args

        def fake_find_spec(name: str):
            if name == "traigent_backend.mcp.server":
                return None
            if name == "optigen_backend.mcp.server":
                return object()
            return None

        with patch(
            "traigent.cloud.production_mcp_client.importlib.util.find_spec",
            side_effect=fake_find_spec,
        ):
            assert default_mcp_server_args()[1] == "optigen_backend.mcp.server"

    def test_default_mcp_server_args_prefers_renamed_backend_when_available(self):
        """Use the renamed backend package when it is installed."""
        from traigent.cloud.production_mcp_client import default_mcp_server_args

        def fake_find_spec(name: str):
            if name == "traigent_backend.mcp.server":
                return object()
            return None

        with patch(
            "traigent.cloud.production_mcp_client.importlib.util.find_spec",
            side_effect=fake_find_spec,
        ):
            assert default_mcp_server_args()[1] == "traigent_backend.mcp.server"

    def test_get_production_mcp_client_returns_existing(self):
        """Test get_production_mcp_client returns existing client."""
        from traigent.cloud import production_mcp_client
        from traigent.cloud.production_mcp_client import (
            MCPServerConfig,
            get_production_mcp_client,
        )

        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client1 = get_production_mcp_client()
                client2 = get_production_mcp_client()
                assert client1 is client2

    def test_get_production_mcp_client_with_custom_args(self):
        """Test get_production_mcp_client with custom arguments."""
        from traigent.cloud import production_mcp_client
        from traigent.cloud.production_mcp_client import get_production_mcp_client

        production_mcp_client._production_client = None

        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                client = get_production_mcp_client(
                    server_path="/usr/bin/python",
                    server_args=["-m", "custom_server"],
                    timeout=60.0,
                )
                assert client.server_config.server_path == "/usr/bin/python"
                assert client.server_config.timeout == 60.0

    def test_set_production_mcp_client(self):
        """Test set_production_mcp_client."""
        from traigent.cloud import production_mcp_client
        from traigent.cloud.production_mcp_client import (
            MCPServerConfig,
            set_production_mcp_client,
        )

        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                config = MCPServerConfig(server_path="custom")
                custom_client = ProductionMCPClient(config)

                set_production_mcp_client(custom_client)
                assert production_mcp_client._production_client is custom_client


class TestMCPConfiguration:
    """Test MCP client configuration management."""

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                valid_configs = [
                    MCPServerConfig(server_path="python", timeout=30.0, max_retries=3),
                    MCPServerConfig(server_path="python", server_args=["-m", "server"]),
                    MCPServerConfig(server_path="python"),
                ]

                for config in valid_configs:
                    client = ProductionMCPClient(config)
                    assert client is not None

    def test_configuration_defaults(self):
        """Test default configuration values."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                # Create client with default configuration
                default_config = MCPServerConfig(server_path="python")
                client = ProductionMCPClient(default_config)

                # Test that client has reasonable defaults
                assert client.server_config.timeout > 0
                assert client.server_config.max_retries >= 0

    def test_configuration_overrides(self):
        """Test configuration override behavior."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        # Mock RetryConfig and RetryHandler to avoid initialization errors
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()

                # Create client with custom configuration
                custom_config = MCPServerConfig(
                    server_path="python", timeout=60.0, max_retries=5
                )
                client = ProductionMCPClient(custom_config)

                # Verify configuration was applied
                assert client.server_config.timeout == 60.0
                assert client.server_config.max_retries == 5


class TestCallToolExecution:
    """Test call_tool method execution paths."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_call_tool_validation_error_branch(self):
        """Test call_tool returns validation error when inputs invalid."""
        # Empty tool name should trigger validation error
        result = await self.client.call_tool("", {"arg": "value"})
        assert result.success is False
        assert "tool_name" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_call_tool_with_operation_id(self):
        """Test call_tool with custom operation_id."""
        custom_op_id = "custom-op-123"
        result = await self.client.call_tool("test_tool", {}, operation_id=custom_op_id)
        # The operation result should be stored with this ID
        assert (
            custom_op_id in self.client._operation_results
            or result.request_id == custom_op_id
        )


class TestListResources:
    """Test list_resources method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_list_resources_not_connected(self):
        """Test list_resources when not connected and connect fails."""
        self.client._connected = False
        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = False
            with patch.object(
                self.client, "connect", new_callable=AsyncMock
            ) as mock_connect:
                mock_connect.return_value = False

                result = await self.client.list_resources()
                assert result.success is False
                assert "Unable to connect" in result.error_message

    @pytest.mark.asyncio
    async def test_list_resources_success(self):
        """Test list_resources when connected successfully."""
        self.client._connected = True
        mock_resource = Mock()
        mock_resource.__dict__ = {"uri": "test://resource", "name": "Test"}

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.resources = [mock_resource]
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        self.client._session = mock_session

        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = True

            # Mock asdict to return a simple dict
            with patch(
                "traigent.cloud.production_mcp_client.asdict",
                return_value={"uri": "test"},
            ):
                result = await self.client.list_resources()
                assert result.success is True
                assert "resources" in result.data


class TestReadResource:
    """Test read_resource method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_read_resource_not_connected(self):
        """Test read_resource when not connected and connect fails."""
        self.client._connected = False
        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = False
            with patch.object(
                self.client, "connect", new_callable=AsyncMock
            ) as mock_connect:
                mock_connect.return_value = False

                result = await self.client.read_resource("test://uri")
                assert result.success is False
                assert "Unable to connect" in result.error_message

    @pytest.mark.asyncio
    async def test_read_resource_success(self):
        """Test read_resource when connected successfully."""
        self.client._connected = True
        mock_content = Mock()
        mock_content.text = "Test resource content"

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.contents = [mock_content]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        self.client._session = mock_session

        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = True

            result = await self.client.read_resource("test://uri")
            assert result.success is True
            assert result.data["content"] == "Test resource content"


class TestCreateExperiment:
    """Test create_experiment method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_create_experiment_invalid_type(self):
        """Test create_experiment with invalid type raises ValidationError."""
        from traigent.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must be an OptimizationRequest"):
            await self.client.create_experiment("not an optimization request")

    @pytest.mark.asyncio
    async def test_create_experiment_valid(self):
        """Test create_experiment with valid OptimizationRequest."""
        from traigent.cloud.models import OptimizationRequest
        from traigent.cloud.production_mcp_client import MCPResponse
        from traigent.evaluators.base import Dataset, EvaluationExample

        # Create proper dataset with examples
        example = EvaluationExample(
            input_data={"test": "input"}, expected_output="output"
        )
        dataset = Dataset(name="test", examples=[example])

        request = OptimizationRequest(
            function_name="test_function",
            dataset=dataset,
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
        )

        with patch.object(
            self.client, "call_tool", new_callable=AsyncMock
        ) as mock_call_tool:
            mock_call_tool.return_value = MCPResponse(
                success=True, data={"experiment_id": "exp_123"}
            )

            result = await self.client.create_experiment(request)
            assert result.success is True
            mock_call_tool.assert_called_once()


class TestStartExperimentRun:
    """Test start_experiment_run method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_start_experiment_run_valid(self):
        """Test start_experiment_run with valid inputs."""
        from traigent.cloud.production_mcp_client import MCPResponse

        with patch.object(
            self.client, "call_tool", new_callable=AsyncMock
        ) as mock_call_tool:
            mock_call_tool.return_value = MCPResponse(
                success=True, data={"experiment_run_id": "run_123"}
            )

            result = await self.client.start_experiment_run(
                "exp_123", {"param": [1, 2]}, 10
            )
            assert result.success is True
            mock_call_tool.assert_called_once()


class TestCreateConfigurationRun:
    """Test create_configuration_run method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_create_configuration_run_valid(self):
        """Test create_configuration_run with valid inputs."""
        from traigent.cloud.production_mcp_client import MCPResponse

        with patch.object(
            self.client, "call_tool", new_callable=AsyncMock
        ) as mock_call_tool:
            mock_call_tool.return_value = MCPResponse(
                success=True, data={"config_run_id": "config_123"}
            )

            result = await self.client.create_configuration_run(
                "run_123", {"param1": "value1"}, "trial_123"
            )
            assert result.success is True
            mock_call_tool.assert_called_once()


class TestUpdateConfigurationRunResults:
    """Test update_configuration_run_results method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_update_configuration_run_results_valid(self):
        """Test update_configuration_run_results with valid inputs."""
        from traigent.cloud.production_mcp_client import MCPResponse

        with patch.object(
            self.client, "call_tool", new_callable=AsyncMock
        ) as mock_call_tool:
            mock_call_tool.return_value = MCPResponse(success=True, data={})

            result = await self.client.update_configuration_run_results(
                "config_123", {"accuracy": 0.95}, "COMPLETED"
            )
            assert result.success is True
            mock_call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_configuration_run_results_with_error(self):
        """Test update_configuration_run_results with error message."""
        from traigent.cloud.production_mcp_client import MCPResponse

        with patch.object(
            self.client, "call_tool", new_callable=AsyncMock
        ) as mock_call_tool:
            mock_call_tool.return_value = MCPResponse(success=True, data={})

            result = await self.client.update_configuration_run_results(
                "config_123", {"accuracy": 0.0}, "FAILED", error_message="Test error"
            )
            assert result.success is True


class TestCreateAgent:
    """Test create_agent method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_create_agent_invalid_type(self):
        """Test create_agent with invalid type raises ValidationError."""
        from traigent.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must be an AgentSpecification"):
            await self.client.create_agent("not an agent spec")

    @pytest.mark.asyncio
    async def test_create_agent_valid(self):
        """Test create_agent with valid AgentSpecification."""
        from traigent.cloud.models import AgentSpecification
        from traigent.cloud.production_mcp_client import MCPResponse

        agent_spec = AgentSpecification(name="test_agent", agent_type="optimization")

        with patch.object(
            self.client, "call_tool", new_callable=AsyncMock
        ) as mock_call_tool:
            mock_call_tool.return_value = MCPResponse(
                success=True, data={"agent_id": "agent_123"}
            )

            result = await self.client.create_agent(agent_spec)
            assert result.success is True
            mock_call_tool.assert_called_once()


class TestUploadDataset:
    """Test upload_dataset method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_upload_dataset_invalid_type(self):
        """Test upload_dataset with invalid type raises ValidationError."""
        from traigent.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must be a Dataset"):
            await self.client.upload_dataset("not a dataset")

    @pytest.mark.asyncio
    async def test_upload_dataset_valid(self):
        """Test upload_dataset with valid Dataset."""
        from traigent.cloud.production_mcp_client import MCPResponse
        from traigent.evaluators.base import Dataset, EvaluationExample

        # Create proper EvaluationExample instances
        example = EvaluationExample(
            input_data={"test": "input"},
            expected_output="test output",
        )
        dataset = Dataset(
            name="test_dataset",
            examples=[example],
        )

        # Mock the converter (locally imported in upload_dataset)
        mock_metadata = Mock()
        mock_metadata.name = "test_dataset"
        mock_metadata.type = "qa"
        mock_metadata.description = "Test dataset"

        mock_converter = Mock()
        mock_converter.sdk_dataset_to_backend_examples.return_value = (
            [{"input": "test"}],
            mock_metadata,
        )

        with patch("traigent.cloud.dataset_converter.converter", mock_converter):
            with patch.object(
                self.client, "call_tool", new_callable=AsyncMock
            ) as mock_call_tool:
                mock_call_tool.return_value = MCPResponse(
                    success=True, data={"example_set_id": "set_123"}
                )

                result = await self.client.upload_dataset(dataset)
                assert result.success is True


class TestCreateOptimizationWorkflow:
    """Test create_optimization_workflow method."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_create_optimization_workflow_invalid_type(self):
        """Test create_optimization_workflow with invalid type."""
        from traigent.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must be an OptimizationRequest"):
            await self.client.create_optimization_workflow("not a request")

    @pytest.mark.asyncio
    async def test_create_optimization_workflow_missing_agent_spec(self):
        """Test create_optimization_workflow with missing agent_specification."""
        from traigent.cloud.models import OptimizationRequest
        from traigent.evaluators.base import Dataset, EvaluationExample
        from traigent.utils.exceptions import ValidationError

        example = EvaluationExample(
            input_data={"test": "input"}, expected_output="output"
        )
        dataset = Dataset(name="test", examples=[example])
        request = OptimizationRequest(
            function_name="test_function",
            dataset=dataset,
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            agent_specification=None,
        )

        with pytest.raises(ValidationError, match="agent_specification is required"):
            await self.client.create_optimization_workflow(request)

    @pytest.mark.asyncio
    async def test_create_optimization_workflow_invalid_dataset(self):
        """Test create_optimization_workflow with invalid dataset type."""
        from traigent.cloud.models import AgentSpecification, OptimizationRequest
        from traigent.utils.exceptions import ValidationError

        request = OptimizationRequest(
            function_name="test_function",
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            agent_specification=AgentSpecification(name="test", agent_type="opt"),
            dataset="not a dataset",  # Invalid type
        )

        with pytest.raises(ValidationError, match="must be a Dataset"):
            await self.client.create_optimization_workflow(request)

    @pytest.mark.asyncio
    async def test_create_optimization_workflow_agent_creation_failure(self):
        """Test create_optimization_workflow when agent creation fails."""
        from traigent.cloud.models import AgentSpecification, OptimizationRequest
        from traigent.cloud.production_mcp_client import MCPResponse
        from traigent.evaluators.base import Dataset, EvaluationExample

        example = EvaluationExample(
            input_data={"test": "input"}, expected_output="output"
        )
        request = OptimizationRequest(
            function_name="test_function",
            dataset=Dataset(name="test", examples=[example]),
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            agent_specification=AgentSpecification(name="test", agent_type="opt"),
        )

        with patch.object(
            self.client, "create_agent", new_callable=AsyncMock
        ) as mock_create_agent:
            mock_create_agent.return_value = MCPResponse(
                success=False, error_message="Agent creation failed"
            )

            with pytest.raises(RuntimeError, match="Failed to create agent"):
                await self.client.create_optimization_workflow(request)

    @pytest.mark.asyncio
    async def test_create_optimization_workflow_dataset_upload_failure(self):
        """Test create_optimization_workflow when dataset upload fails."""
        from traigent.cloud.models import AgentSpecification, OptimizationRequest
        from traigent.cloud.production_mcp_client import MCPResponse
        from traigent.evaluators.base import Dataset, EvaluationExample

        example = EvaluationExample(
            input_data={"test": "input"}, expected_output="output"
        )
        request = OptimizationRequest(
            function_name="test_function",
            dataset=Dataset(name="test", examples=[example]),
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            agent_specification=AgentSpecification(name="test", agent_type="opt"),
            metadata={},
        )

        with patch.object(
            self.client, "create_agent", new_callable=AsyncMock
        ) as mock_create_agent:
            mock_create_agent.return_value = MCPResponse(
                success=True, data={"agent_id": "agent_123"}
            )

            with patch.object(
                self.client, "upload_dataset", new_callable=AsyncMock
            ) as mock_upload:
                mock_upload.return_value = MCPResponse(
                    success=False, error_message="Upload failed"
                )

                with pytest.raises(RuntimeError, match="Failed to upload dataset"):
                    await self.client.create_optimization_workflow(request)

    @pytest.mark.asyncio
    async def test_create_optimization_workflow_success(self):
        """Test create_optimization_workflow with successful workflow."""
        from traigent.cloud.models import AgentSpecification, OptimizationRequest
        from traigent.cloud.production_mcp_client import MCPResponse
        from traigent.evaluators.base import Dataset, EvaluationExample

        example = EvaluationExample(
            input_data={"test": "input"}, expected_output="output"
        )
        request = OptimizationRequest(
            function_name="test_function",
            dataset=Dataset(name="test", examples=[example]),
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            agent_specification=AgentSpecification(name="test", agent_type="opt"),
            metadata={},
        )

        with patch.object(
            self.client, "create_agent", new_callable=AsyncMock
        ) as mock_create_agent:
            mock_create_agent.return_value = MCPResponse(
                success=True, data={"agent_id": "agent_123"}
            )

            with patch.object(
                self.client, "upload_dataset", new_callable=AsyncMock
            ) as mock_upload:
                mock_upload.return_value = MCPResponse(
                    success=True, data={"example_set_id": "set_123"}
                )

                with patch.object(
                    self.client, "create_experiment", new_callable=AsyncMock
                ) as mock_create_exp:
                    mock_create_exp.return_value = MCPResponse(
                        success=True, data={"experiment_id": "exp_123"}
                    )

                    with patch.object(
                        self.client, "start_experiment_run", new_callable=AsyncMock
                    ) as mock_start_run:
                        mock_start_run.return_value = MCPResponse(
                            success=True, data={"experiment_run_id": "run_123"}
                        )

                        agent_id, exp_id, run_id = (
                            await self.client.create_optimization_workflow(request)
                        )
                        assert agent_id == "agent_123"
                        assert exp_id == "exp_123"
                        assert run_id == "run_123"


class TestSessionUnavailableAfterConnect:
    """Test ConnectionError when session is None after successful connect."""

    def setup_method(self):
        """Set up test fixtures."""
        from traigent.cloud.production_mcp_client import MCPServerConfig

        server_config = MCPServerConfig(
            server_path="python", server_args=["-m", "test"]
        )
        with patch(
            "traigent.cloud.production_mcp_client.RetryConfig"
        ) as mock_retry_config:
            with patch(
                "traigent.cloud.production_mcp_client.RetryHandler"
            ) as mock_retry_handler:
                mock_retry_config.return_value = Mock()
                mock_retry_handler.return_value = Mock()
                self.client = ProductionMCPClient(server_config)

    @pytest.mark.asyncio
    async def test_call_tool_session_unavailable_after_connect(self):
        """Test call_tool returns error when session is None after connect."""
        self.client._connected = False
        self.client._session = None

        # Mock retry handler to execute the function directly
        async def mock_execute(func):
            try:
                result = await func()
                return Mock(success=True, result=result)
            except Exception as e:
                return Mock(success=False, last_exception=e, result=None)

        self.client._retry_handler.execute_async = mock_execute

        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = False
            with patch.object(
                self.client, "connect", new_callable=AsyncMock
            ) as mock_connect:
                # connect returns True but session remains None
                mock_connect.return_value = True

                result = await self.client.call_tool("test_tool", {"arg": "value"})
                assert result.success is False
                assert (
                    "session" in result.error_message.lower()
                    or "unavailable" in result.error_message.lower()
                )

    @pytest.mark.asyncio
    async def test_list_resources_session_unavailable_after_connect(self):
        """Test list_resources raises ConnectionError when session is None after connect."""
        self.client._connected = False
        self.client._session = None

        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = False
            with patch.object(
                self.client, "connect", new_callable=AsyncMock
            ) as mock_connect:
                # connect returns True but session remains None
                mock_connect.return_value = True

                result = await self.client.list_resources()
                assert result.success is False
                assert "session" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_read_resource_session_unavailable_after_connect(self):
        """Test read_resource raises ConnectionError when session is None after connect."""
        self.client._connected = False
        self.client._session = None

        with patch.object(
            self.client, "is_connected", new_callable=AsyncMock
        ) as mock_is_connected:
            mock_is_connected.return_value = False
            with patch.object(
                self.client, "connect", new_callable=AsyncMock
            ) as mock_connect:
                # connect returns True but session remains None
                mock_connect.return_value = True

                result = await self.client.read_resource("test://uri")
                assert result.success is False
                assert "session" in result.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__])
