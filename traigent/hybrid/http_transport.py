"""HTTP transport implementation for Hybrid API mode.

Provides HTTP/REST communication with external agentic services
following the Traigent hybrid API protocol.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION HTTP-TRANSPORT

from __future__ import annotations

from typing import Any

import httpx

from traigent.hybrid.protocol import (
    ConfigSpaceResponse,
    HealthCheckResponse,
    HybridEvaluateRequest,
    HybridEvaluateResponse,
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
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class HTTPTransport:
    """HTTP transport for external agentic API endpoints.

    Implements the HybridTransport protocol using httpx AsyncClient
    with HTTP/2 support and connection pooling.

    Attributes:
        base_url: Base URL for the external service
        timeout: Request timeout in seconds
        max_connections: Maximum concurrent connections
    """

    # API endpoint paths (relative to base_url)
    CAPABILITIES_PATH = "/traigent/v1/capabilities"
    CONFIG_SPACE_PATH = "/traigent/v1/config-space"
    EXECUTE_PATH = "/traigent/v1/execute"
    EVALUATE_PATH = "/traigent/v1/evaluate"
    HEALTH_PATH = "/traigent/v1/health"
    KEEP_ALIVE_PATH = "/traigent/v1/keep-alive"

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        max_connections: int = 10,
        auth_header: str | None = None,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            base_url: Base URL for the external service (e.g., "http://agent:8080")
            timeout: Request timeout in seconds (default 300)
            max_connections: Maximum concurrent HTTP connections (default 10)
            auth_header: Optional Authorization header value
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_connections = max_connections
        self._auth_header = auth_header
        self._client: httpx.AsyncClient | None = None
        self._capabilities: ServiceCapabilities | None = None
        self._closed = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with connection pooling."""
        if self._client is None or self._closed:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Traigent-SDK/1.0",
            }
            if self._auth_header:
                headers["Authorization"] = self._auth_header

            # Configure connection pooling
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_connections,
            )

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
                limits=limits,
                http2=True,  # Enable HTTP/2 for better performance
            )
            self._closed = False

        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path (relative to base_url)
            json_data: Optional JSON body
            timeout_override: Optional timeout override for this request

        Returns:
            Parsed JSON response.

        Raises:
            TransportConnectionError: If connection fails
            TransportTimeoutError: If request times out
            TransportAuthError: If authentication fails
            TransportError: For other HTTP errors
        """
        client = await self._get_client()

        try:
            timeout = (
                httpx.Timeout(timeout_override)
                if timeout_override
                else httpx.Timeout(self.timeout)
            )

            response = await client.request(
                method,
                path,
                json=json_data,
                timeout=timeout,
            )

            # Handle HTTP errors
            if response.status_code == 401:
                raise TransportAuthError(
                    "Authentication failed",
                    status_code=401,
                    response_body=response.text,
                )
            elif response.status_code == 403:
                raise TransportAuthError(
                    "Authorization denied",
                    status_code=403,
                    response_body=response.text,
                )
            elif response.status_code == 429:
                # Parse Retry-After header if present
                retry_after_str = response.headers.get("Retry-After")
                retry_after: float | None = None
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass  # Could be HTTP-date format, ignore for now
                raise TransportRateLimitError(
                    "Rate limit exceeded",
                    retry_after=retry_after,
                    response_body=response.text,
                )
            elif response.status_code >= 500:
                raise TransportServerError(
                    f"Server error: HTTP {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            elif response.status_code >= 400:
                raise TransportError(
                    f"HTTP {response.status_code}: {response.reason_phrase}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            result: dict[str, Any] = response.json()
            return result

        except httpx.ConnectError as e:
            raise TransportConnectionError(
                f"Failed to connect to {self.base_url}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise TransportTimeoutError(
                f"Request timed out after {self.timeout}s",
                cause=e,
            ) from e
        except httpx.HTTPStatusError as e:
            raise TransportError(
                f"HTTP error: {e}",
                status_code=e.response.status_code if e.response else None,
                cause=e,
            ) from e
        except Exception as e:
            if isinstance(e, TransportError):
                raise
            raise TransportError(
                f"Unexpected error: {e}",
                cause=e,
            ) from e

    async def capabilities(self) -> ServiceCapabilities:
        """Fetch service capabilities via handshake.

        Caches the result for subsequent calls.

        Returns:
            ServiceCapabilities with version and feature flags.

        Raises:
            TransportError: If handshake fails.
        """
        if self._capabilities is not None:
            return self._capabilities

        try:
            data = await self._request("GET", self.CAPABILITIES_PATH)
            self._capabilities = ServiceCapabilities.from_dict(data)
            logger.debug(
                "Service capabilities: version=%s, evaluate=%s, keep_alive=%s",
                self._capabilities.version,
                self._capabilities.supports_evaluate,
                self._capabilities.supports_keep_alive,
            )
            return self._capabilities
        except TransportError:
            # If capabilities endpoint not available, return defaults
            logger.warning("Capabilities endpoint not available, using defaults")
            self._capabilities = ServiceCapabilities(version="1.0")
            return self._capabilities

    async def discover_config_space(self) -> ConfigSpaceResponse:
        """Fetch TVAR definitions from external service.

        Returns:
            ConfigSpaceResponse with TVARs and constraints.

        Raises:
            TransportError: If discovery fails.
        """
        data = await self._request("GET", self.CONFIG_SPACE_PATH)
        return ConfigSpaceResponse.from_dict(data)

    async def execute(
        self,
        request: HybridExecuteRequest,
    ) -> HybridExecuteResponse:
        """Execute agent with config on inputs.

        Args:
            request: Execution request with config and inputs.

        Returns:
            HybridExecuteResponse with outputs and metrics.

        Raises:
            TransportError: If execution fails.
        """
        # Use request timeout if specified
        timeout_s = request.timeout_ms / 1000.0 if request.timeout_ms > 0 else None

        data = await self._request(
            "POST",
            self.EXECUTE_PATH,
            json_data=request.to_dict(),
            timeout_override=timeout_s,
        )
        return HybridExecuteResponse.from_dict(data)

    async def evaluate(
        self,
        request: HybridEvaluateRequest,
    ) -> HybridEvaluateResponse:
        """Score outputs against targets.

        Args:
            request: Evaluation request with outputs and targets.

        Returns:
            HybridEvaluateResponse with per-example and aggregate metrics.

        Raises:
            TransportError: If evaluation fails.
            NotImplementedError: If evaluate not supported.
        """
        # Check capabilities first
        caps = await self.capabilities()
        if not caps.supports_evaluate:
            raise NotImplementedError(
                "External service does not support separate evaluate endpoint"
            )

        data = await self._request(
            "POST",
            self.EVALUATE_PATH,
            json_data=request.to_dict(),
        )
        return HybridEvaluateResponse.from_dict(data)

    async def health_check(self) -> HealthCheckResponse:
        """Check service health.

        Returns:
            HealthCheckResponse with status and details.

        Raises:
            TransportError: If health check fails.
        """
        data = await self._request("GET", self.HEALTH_PATH)
        return HealthCheckResponse.from_dict(data)

    async def keep_alive(self, session_id: str) -> bool:
        """Signal ongoing session for stateful agents.

        Args:
            session_id: Active session identifier.

        Returns:
            True if session is still alive, False if expired.

        Raises:
            TransportError: If keep-alive fails.
            NotImplementedError: If keep-alive not supported.
        """
        caps = await self.capabilities()
        if not caps.supports_keep_alive:
            raise NotImplementedError("External service does not support keep-alive")

        try:
            data = await self._request(
                "POST",
                self.KEEP_ALIVE_PATH,
                json_data={"session_id": session_id},
            )
            alive: bool = data.get("alive", False)
            return alive
        except TransportError as e:
            # Session expired or invalid
            if e.status_code == 404:
                return False
            raise

    async def close(self) -> None:
        """Cleanup HTTP client resources.

        Safe to call multiple times.
        """
        if self._client is not None and not self._closed:
            await self._client.aclose()
            self._closed = True
            self._client = None
            self._capabilities = None

    async def __aenter__(self) -> HTTPTransport:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
