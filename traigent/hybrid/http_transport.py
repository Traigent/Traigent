"""HTTP transport implementation for Hybrid API mode.

Provides HTTP/REST communication with external agentic services
following the Traigent hybrid API protocol.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION HTTP-TRANSPORT

from __future__ import annotations

import asyncio
from typing import Any

import httpx

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
    DATASETS_PATH = "/traigent/v1/datasets"
    BENCHMARKS_PATH = "/traigent/v1/benchmarks"
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
        require_http2: bool = False,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            base_url: Base URL for the external service (e.g., "http://agent:8080")
            timeout: Request timeout in seconds (default 300)
            max_connections: Maximum concurrent HTTP connections (default 10)
            auth_header: Optional Authorization header value
            require_http2: Enforce HTTPS + HTTP/2 responses (strict mode)
        """
        self.base_url = base_url.rstrip("/")
        if require_http2 and not self.base_url.startswith("https://"):
            raise ValueError("require_http2=True requires an https:// base_url")
        self.timeout = timeout
        self.max_connections = max_connections
        self._auth_header = auth_header
        self.require_http2 = require_http2
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._capabilities: ServiceCapabilities | None = None
        self._closed = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with connection pooling."""
        if self._client is not None and not self._closed:
            return self._client

        async with self._client_lock:
            if self._client is None or self._closed:
                headers: dict[str, str] = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Traigent-SDK/1.0",
                }
                if self._auth_header:
                    headers["Authorization"] = self._auth_header
                    headers["x-api-key"] = self._auth_header

                # Configure connection pooling
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_connections,
                )

                # Use HTTP/2 if h2 package is available, fall back to HTTP/1.1
                try:
                    import h2  # noqa: F401

                    use_http2 = True
                except ImportError:
                    use_http2 = False
                    logger.debug("h2 package not installed, using HTTP/1.1")

                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=httpx.Timeout(self.timeout),
                    limits=limits,
                    http2=use_http2,
                )
                self._closed = False

        return self._client

    @staticmethod
    def _parse_retry_after(headers: httpx.Headers) -> float | None:
        """Extract Retry-After value from response headers.

        Returns:
            Parsed float seconds, or None if absent/unparseable.
        """
        raw = headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None  # Could be HTTP-date format, ignore for now

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise the appropriate TransportError for non-2xx responses.

        Args:
            response: The HTTP response to inspect.
        """
        code = response.status_code

        if code == 401:
            raise TransportAuthError(
                "Authentication failed",
                status_code=401,
                response_body=response.text,
            )
        if code == 403:
            raise TransportAuthError(
                "Authorization denied",
                status_code=403,
                response_body=response.text,
            )
        if code == 429:
            raise TransportRateLimitError(
                "Rate limit exceeded",
                retry_after=self._parse_retry_after(response.headers),
                response_body=response.text,
            )
        if code in (408, 504):
            raise TransportTimeoutError(
                f"Request timed out: HTTP {code}",
                status_code=code,
                response_body=response.text,
            )
        if code >= 500:
            raise TransportServerError(
                f"Server error: HTTP {code}",
                status_code=code,
                response_body=response.text,
            )
        if code >= 400:
            raise TransportError(
                f"HTTP {code}: {response.reason_phrase}",
                status_code=code,
                response_body=response.text,
            )

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        timeout_override: float | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path (relative to base_url)
            json_data: Optional JSON body
            timeout_override: Optional timeout override for this request
            params: Optional query parameters

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
                params=params,
                timeout=timeout,
            )

            if self.require_http2 and response.http_version != "HTTP/2":
                raise TransportError(
                    (
                        "HTTP/2 is required, but upstream responded with "
                        f"{response.http_version}"
                    ),
                    status_code=426,
                    response_body=response.text,
                )

            self._raise_for_status(response)

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
            # Missing capabilities must not claim support for optional endpoints.
            logger.warning("Capabilities endpoint not available, using safe defaults")
            self._capabilities = ServiceCapabilities(version="1.0")
            return self._capabilities

    async def discover_config_space(
        self, *, tunable_id: str | None = None
    ) -> ConfigSpaceResponse:
        """Fetch TVAR definitions from external service.

        Args:
            tunable_id: Optional tunable ID to fetch config space for.

        Returns:
            ConfigSpaceResponse with TVARs and constraints.

        Raises:
            TransportError: If discovery fails.
        """
        params: dict[str, str] | None = None
        if tunable_id is not None:
            params = {"tunable_id": tunable_id}
        data = await self._request("GET", self.CONFIG_SPACE_PATH, params=params)
        return ConfigSpaceResponse.from_dict(data)

    async def benchmarks(
        self,
        tunable_id: str | None = None,
    ) -> BenchmarksResponse:
        """Discover available datasets and their example IDs.

        Args:
            tunable_id: Optional filter — only return datasets linked to this tunable.

        Returns:
            BenchmarksResponse with dataset entries and example IDs.

        Raises:
            TransportError: If discovery fails or tunable_id is unknown.
        """
        params: dict[str, str] | None = None
        if tunable_id is not None:
            params = {"tunable_id": tunable_id}
        try:
            data = await self._request("GET", self.DATASETS_PATH, params=params)
        except TransportError as exc:
            if exc.status_code != 404:
                raise
            data = await self._request("GET", self.BENCHMARKS_PATH, params=params)
        return BenchmarksResponse.from_dict(data)

    @staticmethod
    def _build_legacy_payload(
        request: HybridExecuteRequest,
    ) -> dict[str, Any]:
        """Build legacy execute payload using ``inputs``/``input_id`` keys.

        Some deployed services still require the old shape.

        Args:
            request: The original execute request.

        Returns:
            Dict payload with ``inputs`` instead of ``examples``.
        """
        legacy_inputs: list[dict[str, Any]] = []
        for item in request.examples:
            if not isinstance(item, dict):
                legacy_inputs.append({"input_id": str(item)})
                continue

            example_id = item.get("example_id")
            payload_data = item.get("data")

            input_id = example_id
            if isinstance(payload_data, dict):
                input_id = (
                    payload_data.get("input_id")
                    or payload_data.get("example_id")
                    or example_id
                )

            legacy_inputs.append({"input_id": str(input_id or "")})

        payload: dict[str, Any] = {
            "request_id": request.request_id,
            "tunable_id": request.tunable_id,
            "config": request.config,
            "inputs": legacy_inputs,
        }
        if request.session_id is not None:
            payload["session_id"] = request.session_id
        if request.timeout_ms is not None:
            payload["timeout_ms"] = request.timeout_ms
        return payload

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

        request_payload = request.to_dict()
        try:
            data = await self._request(
                "POST",
                self.EXECUTE_PATH,
                json_data=request_payload,
                timeout_override=timeout_s,
            )
        except TransportError as exc:
            # Legacy compatibility: some deployed services still require
            # `inputs` with `input_id` instead of `examples` with `example_id`.
            response_body = (exc.response_body or "").lower()
            is_legacy_shape_error = (
                exc.status_code == 400
                and "missing required fields" in response_body
                and "inputs" in response_body
            )
            if not is_legacy_shape_error:
                raise

            logger.info(
                "Execute request falling back to legacy payload format "
                "(inputs/input_id) for compatibility"
            )
            data = await self._request(
                "POST",
                self.EXECUTE_PATH,
                json_data=self._build_legacy_payload(request),
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

        timeout_s = (
            request.timeout_ms / 1000.0
            if request.timeout_ms is not None and request.timeout_ms > 0
            else None
        )
        data = await self._request(
            "POST",
            self.EVALUATE_PATH,
            json_data=request.to_dict(),
            timeout_override=timeout_s,
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
            status = data.get("status")
            if isinstance(status, str):
                return status.lower() == "alive"

            # Backward compatibility with older wrapper servers
            if "alive" in data:
                return bool(data.get("alive"))

            return False
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
