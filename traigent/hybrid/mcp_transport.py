"""MCP transport implementation for Hybrid API mode.

Provides MCP (Model Context Protocol) communication with external agentic
services using the existing ProductionMCPClient infrastructure.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION MCP-TRANSPORT

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

from traigent.hybrid.protocol import (
    ConfigSpaceResponse,
    HealthCheckResponse,
    HybridEvaluateRequest,
    HybridEvaluateResponse,
    HybridExecuteRequest,
    HybridExecuteResponse,
    InputsResponse,
    ServiceCapabilities,
)
from traigent.hybrid.transport import TransportConnectionError, TransportError
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.cloud.production_mcp_client import (
        MCPServerConfig,
        ProductionMCPClient,
    )

logger = get_logger(__name__)


# Canonical MCP resource URIs for Traigent protocol
CONFIG_SPACE_URI = "traigent://config-space"
CAPABILITIES_URI = "traigent://capabilities"
HEALTH_URI = "traigent://health"


class MCPTransport:
    """MCP transport using existing ProductionMCPClient infrastructure.

    Maps the HybridTransport protocol to MCP tool calls and resource reads.

    MCP Protocol Mapping:
        - discover_config_space() -> read_resource("traigent://config-space")
        - capabilities() -> read_resource("traigent://capabilities")
        - execute() -> call_tool("execute", {...})
        - evaluate() -> call_tool("evaluate", {...})
        - health_check() -> read_resource("traigent://health")
        - keep_alive() -> call_tool("keep_alive", {...})
    """

    def __init__(
        self,
        mcp_client: ProductionMCPClient | None = None,
        mcp_config: MCPServerConfig | None = None,
    ) -> None:
        """Initialize MCP transport.

        Args:
            mcp_client: Existing ProductionMCPClient instance to reuse.
                If not provided, a new client will be created from mcp_config.
            mcp_config: MCPServerConfig for creating new MCP client.
                Required if mcp_client is not provided.

        Raises:
            ValueError: If neither mcp_client nor mcp_config is provided.
            ImportError: If MCP dependencies are not available.
        """
        if mcp_client is None and mcp_config is None:
            raise ValueError(
                "Must provide either mcp_client or mcp_config for MCP transport"
            )

        self._owns_client = mcp_client is None
        self._client = mcp_client
        self._client_lock = asyncio.Lock()
        self._mcp_config = mcp_config
        self._capabilities: ServiceCapabilities | None = None
        self._closed = False

    async def _get_client(self) -> ProductionMCPClient:
        """Get or create the MCP client."""
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is None:
                from traigent.cloud.production_mcp_client import ProductionMCPClient

                if self._mcp_config is None:
                    raise ValueError("MCP config required to create client")
                self._client = ProductionMCPClient(self._mcp_config)
                self._owns_client = True

        return self._client

    async def _read_resource(self, uri: str) -> dict[str, Any]:
        """Read MCP resource and parse as JSON.

        Args:
            uri: Resource URI to read.

        Returns:
            Parsed JSON content.

        Raises:
            TransportError: If read fails or content is invalid.
        """
        client = await self._get_client()

        try:
            response = await client.read_resource(uri)

            if not response.success:
                raise TransportError(
                    f"MCP read_resource failed: {response.error_message}",
                    response_body=str(response.data),
                )

            content = response.data.get("content") if response.data else None
            if content is None:
                raise TransportError(f"No content returned for resource: {uri}")

            result: dict[str, Any] = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            raise TransportError(
                f"Invalid JSON in resource {uri}: {e}",
                cause=e,
            ) from e
        except Exception as e:
            if isinstance(e, TransportError):
                raise
            raise TransportConnectionError(
                f"MCP read_resource failed: {e}",
                cause=e,
            ) from e

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call MCP tool and return result.

        Args:
            tool_name: Name of the MCP tool to call.
            arguments: Tool arguments dictionary.

        Returns:
            Tool result data.

        Raises:
            TransportError: If tool call fails.
        """
        client = await self._get_client()
        operation_id = str(uuid.uuid4())

        try:
            response = await client.call_tool(tool_name, arguments, operation_id)

            if not response.success:
                raise TransportError(
                    f"MCP call_tool({tool_name}) failed: {response.error_message}",
                    response_body=str(response.data),
                )

            return response.data or {}

        except Exception as e:
            if isinstance(e, TransportError):
                raise
            raise TransportConnectionError(
                f"MCP call_tool({tool_name}) failed: {e}",
                cause=e,
            ) from e

    async def capabilities(self) -> ServiceCapabilities:
        """Fetch service capabilities via MCP resource.

        Reads the traigent://capabilities resource.
        Caches the result for subsequent calls.

        Returns:
            ServiceCapabilities with version and feature flags.

        Raises:
            TransportError: If capabilities fetch fails.
        """
        if self._capabilities is not None:
            return self._capabilities

        try:
            data = await self._read_resource(CAPABILITIES_URI)
            self._capabilities = ServiceCapabilities.from_dict(data)
            logger.debug(
                "MCP service capabilities: version=%s, evaluate=%s, keep_alive=%s",
                self._capabilities.version,
                self._capabilities.supports_evaluate,
                self._capabilities.supports_keep_alive,
            )
            return self._capabilities
        except TransportError:
            # If capabilities resource not available, return defaults
            logger.warning("Capabilities resource not available, using defaults")
            self._capabilities = ServiceCapabilities(version="1.0")
            return self._capabilities

    async def discover_config_space(
        self, *, tunable_id: str | None = None
    ) -> ConfigSpaceResponse:
        """Fetch TVAR definitions via MCP resource.

        Reads the traigent://config-space resource, optionally filtered
        by tunable_id.

        Args:
            tunable_id: Optional tunable ID to fetch config space for.

        Returns:
            ConfigSpaceResponse with TVARs and constraints.

        Raises:
            TransportError: If discovery fails.
        """
        uri = CONFIG_SPACE_URI
        if tunable_id is not None:
            uri = f"{CONFIG_SPACE_URI}?tunable_id={tunable_id}"
        data = await self._read_resource(uri)
        return ConfigSpaceResponse.from_dict(data)

    async def inputs(
        self,
        tunable_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> InputsResponse:
        """Discover available input IDs via MCP.

        Not yet supported over MCP transport.

        Raises:
            NotImplementedError: Always — MCP input discovery is not yet implemented.
        """
        raise NotImplementedError(
            "Input ID discovery is not yet supported over MCP transport. "
            "Use HTTP transport or provide input IDs explicitly."
        )

    async def execute(
        self,
        request: HybridExecuteRequest,
    ) -> HybridExecuteResponse:
        """Execute agent via MCP tool call.

        Calls the "execute" MCP tool with request data.

        Args:
            request: Execution request with config and inputs.

        Returns:
            HybridExecuteResponse with outputs and metrics.

        Raises:
            TransportError: If execution fails.
        """
        arguments = request.to_dict()
        data = await self._call_tool("execute", arguments)
        return HybridExecuteResponse.from_dict(data)

    async def evaluate(
        self,
        request: HybridEvaluateRequest,
    ) -> HybridEvaluateResponse:
        """Score outputs via MCP tool call.

        Calls the "evaluate" MCP tool with request data.

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
                "External MCP service does not support separate evaluate tool"
            )

        arguments = request.to_dict()
        data = await self._call_tool("evaluate", arguments)
        return HybridEvaluateResponse.from_dict(data)

    async def health_check(self) -> HealthCheckResponse:
        """Check service health via MCP resource.

        Reads the traigent://health resource.

        Returns:
            HealthCheckResponse with status and details.

        Raises:
            TransportError: If health check fails.
        """
        data = await self._read_resource(HEALTH_URI)
        return HealthCheckResponse.from_dict(data)

    async def keep_alive(self, session_id: str) -> bool:
        """Signal ongoing session via MCP tool call.

        Calls the "keep_alive" MCP tool.

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
            raise NotImplementedError(
                "External MCP service does not support keep-alive"
            )

        try:
            data = await self._call_tool("keep_alive", {"session_id": session_id})
            status = data.get("status")
            if isinstance(status, str):
                return status.lower() == "alive"

            # Backward compatibility with older integrations.
            return bool(data.get("alive", False))
        except TransportError as e:
            # Session expired or invalid
            if "not found" in str(e).lower():
                return False
            raise

    async def close(self) -> None:
        """Cleanup MCP client resources.

        Only closes the client if this transport owns it
        (i.e., it was created from mcp_config, not passed in).
        Safe to call multiple times.
        """
        if self._client is not None and self._owns_client and not self._closed:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting MCP client: {e}")
            finally:
                self._closed = True
                self._client = None
                self._capabilities = None

    async def __aenter__(self) -> MCPTransport:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
