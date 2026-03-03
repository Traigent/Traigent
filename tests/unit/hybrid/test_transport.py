"""Unit tests for hybrid mode transport layer."""

from unittest.mock import MagicMock, patch

import pytest

from traigent.hybrid.transport import (
    HybridTransport,
    TransportAuthError,
    TransportConnectionError,
    TransportError,
    TransportRateLimitError,
    TransportServerError,
    TransportTimeoutError,
    create_transport,
)


class TestTransportErrors:
    """Tests for transport error classes."""

    def test_transport_error(self) -> None:
        """Test base transport error."""
        error = TransportError(
            "Test error",
            status_code=500,
            response_body="error body",
        )
        assert str(error) == "Test error"
        assert error.status_code == 500
        assert error.response_body == "error body"

    def test_transport_connection_error(self) -> None:
        """Test connection error."""
        original = Exception("connection refused")
        error = TransportConnectionError("Failed to connect", cause=original)
        assert error.cause is original

    def test_transport_timeout_error(self) -> None:
        """Test timeout error."""
        error = TransportTimeoutError("Request timed out")
        assert "timed out" in str(error)

    def test_transport_auth_error(self) -> None:
        """Test authentication error."""
        error = TransportAuthError("Unauthorized", status_code=401)
        assert error.status_code == 401

    def test_transport_rate_limit_error(self) -> None:
        """Test rate limit error with retry_after."""
        error = TransportRateLimitError(
            "Rate limit exceeded",
            retry_after=60.0,
            response_body="too many requests",
        )
        assert error.status_code == 429
        assert error.retry_after == 60.0
        assert error.response_body == "too many requests"

    def test_transport_rate_limit_error_without_retry_after(self) -> None:
        """Test rate limit error without retry_after."""
        error = TransportRateLimitError("Rate limit exceeded")
        assert error.status_code == 429
        assert error.retry_after is None

    def test_transport_server_error(self) -> None:
        """Test server error (5xx)."""
        error = TransportServerError(
            "Internal server error",
            status_code=500,
            response_body="server crashed",
        )
        assert error.status_code == 500
        assert error.response_body == "server crashed"

    def test_transport_server_error_503(self) -> None:
        """Test service unavailable error."""
        error = TransportServerError(
            "Service unavailable",
            status_code=503,
        )
        assert error.status_code == 503


class TestCreateTransport:
    """Tests for create_transport factory function."""

    def test_create_http_transport(self) -> None:
        """Test creating HTTP transport."""
        with patch("traigent.hybrid.http_transport.HTTPTransport") as mock_http:
            mock_http.return_value = MagicMock()
            _ = create_transport(
                transport_type="http",
                base_url="http://localhost:8080",
                auth_header="Bearer token",
                timeout=60.0,
            )

            mock_http.assert_called_once_with(
                base_url="http://localhost:8080",
                auth_header="Bearer token",
                timeout=60.0,
                max_connections=10,
                require_http2=False,
            )

    def test_create_http_transport_with_require_http2(self) -> None:
        """Test creating HTTP transport with strict HTTP/2 requirement."""
        with patch("traigent.hybrid.http_transport.HTTPTransport") as mock_http:
            mock_http.return_value = MagicMock()
            _ = create_transport(
                transport_type="http",
                base_url="https://api.example.com",
                require_http2=True,
            )

            mock_http.assert_called_once_with(
                base_url="https://api.example.com",
                auth_header=None,
                timeout=300.0,
                max_connections=10,
                require_http2=True,
            )

    def test_create_mcp_transport(self) -> None:
        """Test creating MCP transport."""
        mock_client = MagicMock()
        with patch("traigent.hybrid.mcp_transport.MCPTransport") as mock_mcp:
            mock_mcp.return_value = MagicMock()
            _ = create_transport(
                transport_type="mcp",
                mcp_client=mock_client,
            )

            mock_mcp.assert_called_once_with(
                mcp_client=mock_client,
                mcp_config=None,
            )

    def test_auto_detect_http(self) -> None:
        """Test auto-detection selects HTTP when base_url provided."""
        with patch("traigent.hybrid.http_transport.HTTPTransport") as mock_http:
            mock_http.return_value = MagicMock()
            _ = create_transport(
                transport_type="auto",
                base_url="http://localhost:8080",
            )

            mock_http.assert_called_once()

    def test_auto_detect_mcp(self) -> None:
        """Test auto-detection selects MCP when mcp_client provided."""
        mock_client = MagicMock()
        with patch("traigent.hybrid.mcp_transport.MCPTransport") as mock_mcp:
            mock_mcp.return_value = MagicMock()
            _ = create_transport(
                transport_type="auto",
                mcp_client=mock_client,
            )

            mock_mcp.assert_called_once()

    def test_auto_detect_raises_without_options(self) -> None:
        """Test auto-detection raises error when no options provided."""
        with pytest.raises(ValueError, match="Must specify"):
            create_transport(transport_type="auto")

    def test_http_requires_base_url(self) -> None:
        """Test HTTP transport requires base_url."""
        with pytest.raises(ValueError, match="base_url is required"):
            create_transport(transport_type="http")

    def test_unknown_transport_type(self) -> None:
        """Test unknown transport type raises error."""
        with pytest.raises(ValueError, match="Unknown transport type"):
            create_transport(transport_type="invalid")  # type: ignore


class TestHybridTransportProtocol:
    """Tests for HybridTransport protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that HybridTransport is runtime checkable."""

        # The protocol should be marked as runtime_checkable
        # which allows isinstance() checks
        assert hasattr(HybridTransport, "__protocol_attrs__") or hasattr(
            HybridTransport, "_is_runtime_protocol"
        )

    def test_mock_transport_implements_protocol(self) -> None:
        """Test that a mock implementing all methods passes protocol check."""

        class MockTransport:
            async def capabilities(self):
                pass

            async def discover_config_space(self, *, tunable_id=None):
                pass

            async def execute(self, request):
                pass

            async def evaluate(self, request):
                pass

            async def benchmarks(self, tunable_id=None):
                pass

            async def health_check(self):
                pass

            async def keep_alive(self, session_id):
                pass

            async def close(self):
                pass

        transport = MockTransport()
        assert isinstance(transport, HybridTransport)
