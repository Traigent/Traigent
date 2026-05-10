"""Transport abstraction for Hybrid API mode.

Provides a unified interface for external service communication
supporting both HTTP REST and MCP transports.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION TRANSPORT-ABSTRACTION

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from traigent.hybrid.protocol import (
    BenchmarksResponse,
    ConfigSpaceResponse,
    HealthCheckResponse,
    HybridEvaluateRequest,
    HybridEvaluateResponse,
    HybridExecuteRequest,
    HybridExecuteResponse,
    ServiceCapabilities,
)
from traigent.utils.url_security import validate_outbound_url

if TYPE_CHECKING:
    from traigent.cloud.production_mcp_client import (
        MCPServerConfig,
        ProductionMCPClient,
    )


@runtime_checkable
class HybridTransport(Protocol):
    """Protocol for hybrid mode transport layer.

    Supports both HTTP and MCP transports with unified interface.
    All methods are async to support non-blocking I/O.
    """

    async def capabilities(self) -> ServiceCapabilities:
        """Handshake to discover service features.

        Returns:
            ServiceCapabilities with version, feature flags, and limits.

        Raises:
            TransportError: If handshake fails.
        """
        ...

    async def discover_config_space(
        self, *, tunable_id: str | None = None
    ) -> ConfigSpaceResponse:
        """Fetch TVAR definitions from external service.

        Args:
            tunable_id: Optional tunable ID to fetch config space for.
                When provided, the server returns the config space for
                that specific tunable. When omitted, the default is returned.

        Returns:
            ConfigSpaceResponse with TVARs and constraints.

        Raises:
            TransportError: If discovery fails.
        """
        ...

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
        ...

    async def evaluate(
        self,
        request: HybridEvaluateRequest,
    ) -> HybridEvaluateResponse:
        """Score outputs against targets.

        Only available if capabilities().supports_evaluate is True.

        Args:
            request: Evaluation request with outputs and targets.

        Returns:
            HybridEvaluateResponse with per-example and aggregate metrics.

        Raises:
            TransportError: If evaluation fails.
            NotImplementedError: If evaluate not supported.
        """
        ...

    async def benchmarks(
        self,
        tunable_id: str | None = None,
    ) -> BenchmarksResponse:
        """Discover available benchmarks and their example IDs.

        Args:
            tunable_id: Optional filter — only return benchmarks linked to this tunable.

        Returns:
            BenchmarksResponse with benchmark entries and example IDs.

        Raises:
            TransportError: If discovery fails or tunable_id is unknown.
        """
        ...

    async def health_check(self) -> HealthCheckResponse:
        """Check service health.

        Returns:
            HealthCheckResponse with status and details.

        Raises:
            TransportError: If health check fails.
        """
        ...

    async def keep_alive(self, session_id: str) -> bool:
        """Signal ongoing session for stateful agents.

        Only available if capabilities().supports_keep_alive is True.

        Args:
            session_id: Active session identifier.

        Returns:
            True if session is still alive, False if expired.

        Raises:
            TransportError: If keep-alive fails.
            NotImplementedError: If keep-alive not supported.
        """
        ...

    async def close(self) -> None:
        """Cleanup transport resources.

        Should be called when transport is no longer needed.
        Safe to call multiple times.
        """
        ...


class TransportError(Exception):
    """Base exception for transport layer errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.cause = cause


class TransportConnectionError(TransportError):
    """Connection to external service failed."""

    pass


class TransportTimeoutError(TransportError):
    """Request to external service timed out."""

    pass


class TransportAuthError(TransportError):
    """Authentication with external service failed."""

    pass


class TransportRateLimitError(TransportError):
    """Rate limit exceeded (HTTP 429).

    Attributes:
        retry_after: Suggested retry delay in seconds (from Retry-After header).
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        status_code: int = 429,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.retry_after = retry_after


class TransportServerError(TransportError):
    """Server-side error (HTTP 5xx)."""

    pass


def create_transport(
    transport_type: Literal["http", "mcp", "auto"] = "auto",
    *,
    # HTTP options
    base_url: str | None = None,
    auth_header: str | None = None,
    timeout: float = 300.0,
    max_connections: int = 10,
    require_http2: bool = False,
    # MCP options
    mcp_client: ProductionMCPClient | None = None,
    mcp_config: MCPServerConfig | None = None,
) -> HybridTransport:
    """Create transport based on type or auto-detect.

    Args:
        transport_type: Transport type to use:
            - "http": Use HTTP transport (requires base_url)
            - "mcp": Use MCP transport (requires mcp_client or mcp_config)
            - "auto": Auto-detect based on provided options

        base_url: Base URL for HTTP transport (e.g., "http://agent:8080")
        auth_header: Optional Authorization header value for HTTP
        timeout: Request timeout in seconds (default 300)
        max_connections: Maximum concurrent HTTP connections (default 10)
        require_http2: Enforce HTTPS + HTTP/2 responses for HTTP transport

        mcp_client: Existing ProductionMCPClient instance
        mcp_config: MCPServerConfig for creating new MCP client

    Returns:
        HybridTransport instance.

    Raises:
        ValueError: If required options not provided for transport type.
        ImportError: If MCP dependencies not available.
    """
    # Auto-detect transport type
    if transport_type == "auto":
        if mcp_client is not None or mcp_config is not None:
            transport_type = "mcp"
        elif base_url is not None:
            transport_type = "http"
        else:
            raise ValueError(
                "Must specify base_url for HTTP or mcp_client/mcp_config for MCP"
            )

    if transport_type == "http":
        if base_url is None:
            raise ValueError("base_url is required for HTTP transport")
        safe_base_url = validate_outbound_url(
            base_url,
            purpose="hybrid HTTP base_url",
            allow_private_hosts=True,
        )

        from traigent.hybrid.http_transport import HTTPTransport

        return cast(
            HybridTransport,
            HTTPTransport(
                base_url=safe_base_url,
                auth_header=auth_header,
                timeout=timeout,
                max_connections=max_connections,
                require_http2=require_http2,
            ),
        )

    elif transport_type == "mcp":
        from traigent.hybrid.mcp_transport import MCPTransport

        return cast(
            HybridTransport,
            MCPTransport(
                mcp_client=mcp_client,
                mcp_config=mcp_config,
            ),
        )

    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
