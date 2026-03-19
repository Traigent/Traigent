"""Unit tests for HTTP transport implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from traigent.hybrid.http_transport import HTTPTransport
from traigent.hybrid.protocol import (
    HybridExecuteRequest,
    HybridExecuteResponse,
    ServiceCapabilities,
)
from traigent.hybrid.transport import (
    TransportAuthError,
    TransportConnectionError,
    TransportError,
    TransportRateLimitError,
    TransportServerError,
    TransportTimeoutError,
)


class TestHTTPTransportInit:
    """Tests for HTTPTransport initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        transport = HTTPTransport(base_url="http://localhost:8080")
        assert transport.base_url == "http://localhost:8080"
        assert transport.timeout == 300.0
        assert transport.max_connections == 10
        assert transport._auth_header is None
        assert transport.require_http2 is False

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        transport = HTTPTransport(
            base_url="http://example.com",
            timeout=60.0,
            max_connections=5,
            auth_header="Bearer token123",
            require_http2=False,
        )
        assert transport.base_url == "http://example.com"
        assert transport.timeout == 60.0
        assert transport.max_connections == 5
        assert transport._auth_header == "Bearer token123"

    def test_init_require_http2_requires_https(self) -> None:
        """Strict HTTP/2 mode requires an HTTPS base URL."""
        with pytest.raises(ValueError, match="https:// base_url"):
            HTTPTransport(base_url="http://localhost:8080", require_http2=True)

    def test_init_with_require_http2_on_https(self) -> None:
        """Strict HTTP/2 mode accepts HTTPS endpoints."""
        transport = HTTPTransport(
            base_url="https://api.example.com",
            require_http2=True,
        )
        assert transport.base_url == "https://api.example.com"
        assert transport.require_http2 is True

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        transport = HTTPTransport(base_url="http://localhost:8080/")
        assert transport.base_url == "http://localhost:8080"


class TestHTTPTransportErrorHandling:
    """Tests for HTTP transport error handling."""

    @pytest.fixture
    def transport(self) -> HTTPTransport:
        """Create transport for testing."""
        return HTTPTransport(base_url="http://localhost:8080")

    @pytest.mark.asyncio
    async def test_connection_error(self, transport: HTTPTransport) -> None:
        """Test handling of connection errors."""
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportConnectionError) as exc_info:
                await transport._request("GET", "/test")
            assert "Failed to connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error(self, transport: HTTPTransport) -> None:
        """Test handling of timeout errors."""
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportTimeoutError) as exc_info:
                await transport._request("GET", "/test")
            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_auth_error_401(self, transport: HTTPTransport) -> None:
        """Test handling of 401 authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportAuthError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_error_403(self, transport: HTTPTransport) -> None:
        """Test handling of 403 authorization error."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportAuthError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rate_limit_error_429(self, transport: HTTPTransport) -> None:
        """Test handling of 429 rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_response.headers = {"Retry-After": "60"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportRateLimitError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 429
            assert exc_info.value.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_rate_limit_error_without_retry_after(
        self, transport: HTTPTransport
    ) -> None:
        """Test handling of 429 without Retry-After header."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportRateLimitError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_timeout_error_408(self, transport: HTTPTransport) -> None:
        """Test handling of HTTP 408 timeout responses."""
        mock_response = MagicMock()
        mock_response.status_code = 408
        mock_response.text = "Request Timeout"
        mock_response.http_version = "HTTP/2"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportTimeoutError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_require_http2_rejects_http11_response(self) -> None:
        """Strict HTTP/2 mode rejects HTTP/1.1 responses."""
        transport = HTTPTransport(
            base_url="https://api.example.com",
            require_http2=True,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.http_version = "HTTP/1.1"
        mock_response.text = "{}"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportError, match="HTTP/2 is required"):
                await transport._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_server_error_500(self, transport: HTTPTransport) -> None:
        """Test handling of 500 server error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportServerError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_server_error_503(self, transport: HTTPTransport) -> None:
        """Test handling of 503 service unavailable error."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportServerError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_client_error_400(self, transport: HTTPTransport) -> None:
        """Test handling of 400 client error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.reason_phrase = "Bad Request"
        mock_response.text = "Invalid request"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportError) as exc_info:
                await transport._request("GET", "/test")
            assert exc_info.value.status_code == 400


class TestHTTPTransportMethods:
    """Tests for HTTP transport API methods."""

    @pytest.fixture
    def transport(self) -> HTTPTransport:
        """Create transport for testing."""
        return HTTPTransport(base_url="http://localhost:8080")

    @pytest.mark.asyncio
    async def test_capabilities(self, transport: HTTPTransport) -> None:
        """Test fetching service capabilities."""
        mock_data = {
            "version": "1.0",
            "supports_evaluate": True,
            "supports_keep_alive": False,
            "max_batch_size": 50,
        }

        with patch.object(
            transport, "_request", new_callable=AsyncMock, return_value=mock_data
        ):
            caps = await transport.capabilities()

        assert isinstance(caps, ServiceCapabilities)
        assert caps.version == "1.0"
        assert caps.supports_evaluate is True
        assert caps.max_batch_size == 50

    @pytest.mark.asyncio
    async def test_capabilities_caching(self, transport: HTTPTransport) -> None:
        """Test that capabilities are cached."""
        mock_data = {"version": "1.0"}

        mock_request = AsyncMock(return_value=mock_data)
        with patch.object(transport, "_request", mock_request):
            # First call
            await transport.capabilities()
            # Second call should use cache
            await transport.capabilities()

        # Should only call _request once
        assert mock_request.call_count == 1

    @pytest.mark.asyncio
    async def test_execute(self, transport: HTTPTransport) -> None:
        """Test execute method."""
        mock_response = {
            "request_id": "req-123",
            "execution_id": "exec-456",
            "status": "completed",
            "outputs": [{"example_id": "1", "output": {"result": "test"}}],
            "operational_metrics": {"total_cost_usd": 0.001},
        }

        with patch.object(
            transport, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            request = HybridExecuteRequest(
                tunable_id="test_agent",
                config={"model": "fast"},
                examples=[{"example_id": "1", "data": {}}],
            )
            response = await transport.execute(request)

        assert isinstance(response, HybridExecuteResponse)
        assert response.status == "completed"
        assert len(response.outputs) == 1

    @pytest.mark.asyncio
    async def test_health_check(self, transport: HTTPTransport) -> None:
        """Test health check method."""
        mock_response = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
        }

        with patch.object(
            transport, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            health = await transport.health_check()

        assert health.status == "healthy"
        assert health.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_keep_alive(self, transport: HTTPTransport) -> None:
        """Test keep-alive method."""
        mock_caps = ServiceCapabilities(version="1.0", supports_keep_alive=True)
        mock_response = {"status": "alive", "session_id": "session-123"}

        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(
                transport,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            alive = await transport.keep_alive("session-123")

        assert alive is True

    @pytest.mark.asyncio
    async def test_keep_alive_expired(self, transport: HTTPTransport) -> None:
        """Test keep-alive with expired session."""
        mock_caps = ServiceCapabilities(version="1.0", supports_keep_alive=True)
        mock_response = {"status": "expired", "reason": "expired"}

        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(
                transport,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            alive = await transport.keep_alive("session-123")

        assert alive is False

    @pytest.mark.asyncio
    async def test_keep_alive_backward_compatible_alive_field(
        self, transport: HTTPTransport
    ) -> None:
        """Legacy wrappers returning {'alive': bool} are still supported."""
        mock_caps = ServiceCapabilities(version="1.0", supports_keep_alive=True)
        mock_response = {"alive": True, "session_id": "session-123"}

        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(
                transport,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            alive = await transport.keep_alive("session-123")

        assert alive is True

    @pytest.mark.asyncio
    async def test_keep_alive_not_supported(self, transport: HTTPTransport) -> None:
        """Test keep-alive when not supported."""
        mock_caps = ServiceCapabilities(version="1.0", supports_keep_alive=False)

        with patch.object(
            transport, "capabilities", new_callable=AsyncMock, return_value=mock_caps
        ):
            with pytest.raises(NotImplementedError):
                await transport.keep_alive("session-123")

    @pytest.mark.asyncio
    async def test_keep_alive_no_status_or_alive_returns_false(
        self, transport: HTTPTransport
    ) -> None:
        """Keep-alive with response missing both 'status' and 'alive' returns False."""
        mock_caps = ServiceCapabilities(version="1.0", supports_keep_alive=True)
        mock_response = {"session_id": "session-123", "info": "no relevant key"}

        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(
                transport,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            alive = await transport.keep_alive("session-123")

        assert alive is False

    @pytest.mark.asyncio
    async def test_close(self, transport: HTTPTransport) -> None:
        """Test closing transport."""
        # Initialize client
        mock_client = AsyncMock()
        transport._client = mock_client
        transport._closed = False

        await transport.close()

        assert transport._closed is True
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, transport: HTTPTransport) -> None:
        """Test that close is idempotent."""
        mock_client = AsyncMock()
        transport._client = mock_client
        transport._closed = False

        await transport.close()
        await transport.close()  # Should not raise

        assert transport._closed is True


class TestHTTPTransportClientCreation:
    """Tests for HTTP client creation."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self) -> None:
        """Test _get_client creates httpx client."""
        transport = HTTPTransport(base_url="http://localhost:8080")

        with patch("traigent.hybrid.http_transport.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client = await transport._get_client()

            assert client is mock_client
            assert transport._client is mock_client
            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_with_auth_header(self) -> None:
        """Test _get_client includes auth header."""
        transport = HTTPTransport(
            base_url="http://localhost:8080",
            auth_header="Bearer token123",
        )

        with patch("traigent.hybrid.http_transport.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            await transport._get_client()

            # Verify call includes auth header
            call_kwargs = mock_cls.call_args.kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_get_client_recreates_after_close(self) -> None:
        """Test _get_client recreates client after close."""
        transport = HTTPTransport(base_url="http://localhost:8080")

        with patch("traigent.hybrid.http_transport.httpx.AsyncClient") as mock_cls:
            first_mock = AsyncMock()
            second_mock = AsyncMock()
            mock_cls.side_effect = [first_mock, second_mock]

            first_client = await transport._get_client()
            await transport.close()

            # Should create a new client
            second_client = await transport._get_client()

            assert first_client is first_mock
            assert second_client is second_mock
            assert transport._closed is False

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self) -> None:
        """Test _get_client reuses existing client."""
        transport = HTTPTransport(base_url="http://localhost:8080")

        with patch("traigent.hybrid.http_transport.httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            first_client = await transport._get_client()
            second_client = await transport._get_client()

            assert first_client is second_client
            assert mock_cls.call_count == 1  # Only created once


class TestHTTPTransportRequestOptions:
    """Tests for HTTP request options."""

    @pytest.fixture
    def transport(self) -> HTTPTransport:
        """Create transport for testing."""
        return HTTPTransport(base_url="http://localhost:8080", timeout=30.0)

    @pytest.mark.asyncio
    async def test_request_with_timeout_override(
        self, transport: HTTPTransport
    ) -> None:
        """Test _request with timeout override."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            result = await transport._request("GET", "/test", timeout_override=60.0)

        assert result == {"result": "ok"}
        # Verify timeout was passed
        call_args = mock_client.request.call_args
        assert call_args.kwargs["timeout"].read == 60.0

    @pytest.mark.asyncio
    async def test_request_with_json_data(self, transport: HTTPTransport) -> None:
        """Test _request with JSON body."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            await transport._request("POST", "/test", json_data={"key": "value"})

        call_args = mock_client.request.call_args
        assert call_args.kwargs["json"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_rate_limit_with_invalid_retry_after(
        self, transport: HTTPTransport
    ) -> None:
        """Test rate limit error with invalid Retry-After header."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_response.headers = {"Retry-After": "not-a-number"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportRateLimitError) as exc_info:
                await transport._request("GET", "/test")

        # Invalid Retry-After should result in None
        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_http_status_error(self, transport: HTTPTransport) -> None:
        """Test handling of httpx.HTTPStatusError."""
        mock_response = MagicMock()
        mock_response.status_code = 502

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Gateway",
                request=MagicMock(),
                response=mock_response,
            )
        )

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportError) as exc_info:
                await transport._request("GET", "/test")

        assert "HTTP error" in str(exc_info.value)
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_unexpected_error(self, transport: HTTPTransport) -> None:
        """Test handling of unexpected errors."""
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportError) as exc_info:
                await transport._request("GET", "/test")

        assert "Unexpected error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transport_error_passthrough(self, transport: HTTPTransport) -> None:
        """Test TransportError exceptions are passed through."""
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=TransportAuthError("Auth failed", status_code=401)
        )

        with patch.object(transport, "_get_client", return_value=mock_client):
            with pytest.raises(TransportAuthError) as exc_info:
                await transport._request("GET", "/test")

        assert exc_info.value.status_code == 401


class TestHTTPTransportAdditionalMethods:
    """Tests for additional HTTP transport methods."""

    @pytest.fixture
    def transport(self) -> HTTPTransport:
        """Create transport for testing."""
        return HTTPTransport(base_url="http://localhost:8080")

    @pytest.mark.asyncio
    async def test_capabilities_fallback_on_error(
        self, transport: HTTPTransport
    ) -> None:
        """Test capabilities returns defaults on error."""
        with patch.object(
            transport,
            "_request",
            new_callable=AsyncMock,
            side_effect=TransportError("Connection failed"),
        ):
            caps = await transport.capabilities()

        assert caps.version == "1.0"
        assert caps.supports_evaluate is True  # Default

    @pytest.mark.asyncio
    async def test_discover_config_space(self, transport: HTTPTransport) -> None:
        """Test discover_config_space method."""
        mock_data = {
            "schema_version": "0.9",
            "tunable_id": "test_agent",
            "tvars": [
                {
                    "name": "model",
                    "type": "enum",
                    "domain": {"values": ["gpt-4", "claude-3"]},
                }
            ],
        }

        with patch.object(
            transport, "_request", new_callable=AsyncMock, return_value=mock_data
        ):
            config_space = await transport.discover_config_space()

        assert config_space.tunable_id == "test_agent"
        assert len(config_space.tvars) == 1

    @pytest.mark.asyncio
    async def test_evaluate(self, transport: HTTPTransport) -> None:
        """Test evaluate method."""
        from traigent.hybrid.protocol import HybridEvaluateRequest

        mock_caps = ServiceCapabilities(version="1.0", supports_evaluate=True)
        mock_response = {
            "request_id": "req-123",
            "status": "completed",
            "results": [{"example_id": "1", "metrics": {"accuracy": 0.95}}],
            "aggregate_metrics": {"accuracy": {"mean": 0.95}},
        }

        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(
                transport,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            request = HybridEvaluateRequest(
                tunable_id="test_agent",
                evaluations=[{"example_id": "1", "output": {}, "target": {}}],
            )
            response = await transport.evaluate(request)

        assert response.status == "completed"
        assert len(response.results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_not_supported(self, transport: HTTPTransport) -> None:
        """Test evaluate raises error when not supported."""
        from traigent.hybrid.protocol import HybridEvaluateRequest

        mock_caps = ServiceCapabilities(version="1.0", supports_evaluate=False)

        with patch.object(
            transport,
            "capabilities",
            new_callable=AsyncMock,
            return_value=mock_caps,
        ):
            request = HybridEvaluateRequest(
                tunable_id="test_agent",
                evaluations=[],
            )
            with pytest.raises(NotImplementedError):
                await transport.evaluate(request)

    @pytest.mark.asyncio
    async def test_keep_alive_session_not_found(self, transport: HTTPTransport) -> None:
        """Test keep-alive returns False on 404."""
        mock_caps = ServiceCapabilities(version="1.0", supports_keep_alive=True)

        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(
                transport,
                "_request",
                new_callable=AsyncMock,
                side_effect=TransportError("Not found", status_code=404),
            ),
        ):
            alive = await transport.keep_alive("session-123")

        assert alive is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, transport: HTTPTransport) -> None:
        """Test execute uses request timeout."""
        mock_response = {
            "request_id": "req-123",
            "execution_id": "exec-456",
            "status": "completed",
            "outputs": [],
            "operational_metrics": {},
        }

        mock_request = AsyncMock(return_value=mock_response)
        with patch.object(transport, "_request", mock_request):
            request = HybridExecuteRequest(
                tunable_id="test_agent",
                config={},
                examples=[],
                timeout_ms=60000,  # 60 seconds
            )
            await transport.execute(request)

        # Verify timeout_override was passed
        call_args = mock_request.call_args
        assert call_args.kwargs.get("timeout_override") == 60.0

    @pytest.mark.asyncio
    async def test_evaluate_with_timeout(self, transport: HTTPTransport) -> None:
        """Test evaluate uses request timeout when provided."""
        from traigent.hybrid.protocol import HybridEvaluateRequest

        mock_caps = ServiceCapabilities(version="1.0", supports_evaluate=True)
        mock_response = {
            "request_id": "req-123",
            "status": "completed",
            "results": [],
            "aggregate_metrics": {},
        }

        mock_request = AsyncMock(return_value=mock_response)
        with (
            patch.object(
                transport,
                "capabilities",
                new_callable=AsyncMock,
                return_value=mock_caps,
            ),
            patch.object(transport, "_request", mock_request),
        ):
            request = HybridEvaluateRequest(
                tunable_id="test_agent",
                evaluations=[{"example_id": "1", "output": {}, "target": {}}],
                timeout_ms=45000,
            )
            await transport.evaluate(request)

        call_args = mock_request.call_args
        assert call_args.kwargs.get("timeout_override") == 45.0


class TestHTTPTransportBenchmarks:
    """Tests for benchmarks() endpoint."""

    @pytest.fixture
    def transport(self) -> HTTPTransport:
        """Create transport for testing."""
        return HTTPTransport(base_url="http://localhost:8080")

    @pytest.mark.asyncio
    async def test_benchmarks_happy_path(self, transport: HTTPTransport) -> None:
        """Test benchmarks returns all benchmarks with example IDs."""
        mock_data = {
            "benchmarks": [
                {
                    "benchmark_id": "bench_001",
                    "tunable_ids": ["child-age-agent-a"],
                    "example_ids": ["case_001", "case_002", "case_003"],
                    "name": "Test Benchmark",
                }
            ],
            "benchmarks_revision": None,
        }

        mock_request = AsyncMock(return_value=mock_data)
        with patch.object(transport, "_request", mock_request):
            resp = await transport.benchmarks("child-age-agent-a")

        assert len(resp.benchmarks) == 1
        assert resp.benchmarks[0].benchmark_id == "bench_001"
        assert resp.benchmarks[0].tunable_ids == ["child-age-agent-a"]
        assert resp.benchmarks[0].example_ids == ["case_001", "case_002", "case_003"]
        assert resp.benchmarks_revision is None

        # Verify correct path and params
        mock_request.assert_called_once_with(
            "GET",
            "/traigent/v1/benchmarks",
            params={"tunable_id": "child-age-agent-a"},
        )

    @pytest.mark.asyncio
    async def test_benchmarks_no_tunable_filter(self, transport: HTTPTransport) -> None:
        """Test benchmarks without tunable_id filter."""
        mock_data = {
            "benchmarks": [
                {
                    "benchmark_id": "bench_001",
                    "tunable_ids": ["agent-x"],
                    "example_ids": ["q006", "q007"],
                    "name": "Bench A",
                },
                {
                    "benchmark_id": "bench_002",
                    "tunable_ids": ["agent-y"],
                    "example_ids": ["q008"],
                    "name": "Bench B",
                },
            ],
            "benchmarks_revision": "rev_abc",
        }

        mock_request = AsyncMock(return_value=mock_data)
        with patch.object(transport, "_request", mock_request):
            resp = await transport.benchmarks()

        assert len(resp.benchmarks) == 2
        assert resp.benchmarks_revision == "rev_abc"

        # Verify no params when tunable_id is None
        mock_request.assert_called_once_with(
            "GET", "/traigent/v1/benchmarks", params=None
        )

    @pytest.mark.asyncio
    async def test_benchmarks_empty_response(self, transport: HTTPTransport) -> None:
        """Test benchmarks with empty response."""
        mock_data = {
            "benchmarks": [],
            "benchmarks_revision": None,
        }

        mock_request = AsyncMock(return_value=mock_data)
        with patch.object(transport, "_request", mock_request):
            resp = await transport.benchmarks("t")

        assert resp.benchmarks == []
        assert resp.benchmarks_revision is None

    @pytest.mark.asyncio
    async def test_benchmarks_unknown_tunable_raises(
        self, transport: HTTPTransport
    ) -> None:
        """Test benchmarks propagates 404 as TransportError."""
        with patch.object(
            transport,
            "_request",
            new_callable=AsyncMock,
            side_effect=TransportError("Not found", status_code=404),
        ):
            with pytest.raises(TransportError) as exc_info:
                await transport.benchmarks("bogus")
            assert exc_info.value.status_code == 404


class TestHTTPTransportContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        with patch("traigent.hybrid.http_transport.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async with HTTPTransport(base_url="http://localhost:8080") as transport:
                # Force client creation
                await transport._get_client()
                assert transport._closed is False

            # After context exit, should be closed
            assert transport._closed is True
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        """Test context manager returns transport instance."""
        async with HTTPTransport(base_url="http://localhost:8080") as transport:
            assert isinstance(transport, HTTPTransport)
