"""Hybrid API mode for external agentic service optimization.

This package provides the infrastructure for optimizing external services
via a standardized protocol, supporting both HTTP REST and MCP transports.

Usage:
    from traigent.hybrid import create_transport, HybridTransport
    from traigent.hybrid.protocol import (
        HybridExecuteRequest,
        HybridExecuteResponse,
        ServiceCapabilities,
    )

    # Create HTTP transport
    transport = create_transport(
        transport_type="http",
        base_url="http://agent-service:8080",
    )

    # Discover configuration space
    config_space = await transport.discover_config_space()

    # Execute with configuration
    response = await transport.execute(
        HybridExecuteRequest(
            tunable_id="my_agent",
            config={"temperature": 0.7},
            examples=[{"example_id": "1", "data": {...}}],
        )
    )
"""

# Traceability: HYBRID-MODE-OPTIMIZATION

from traigent.hybrid.discovery import (
    ConfigSpaceDiscovery,
    merge_config_spaces,
    normalize_tvar_to_config_space,
    validate_config_against_tvars,
)
from traigent.hybrid.lifecycle import AgentLifecycleManager, SessionInfo
from traigent.hybrid.protocol import (
    BatchOptions,
    BenchmarkEntry,
    BenchmarksResponse,
    ConfigSpaceResponse,
    EvaluationKwargDefinition,
    HealthCheckResponse,
    HybridEvaluateRequest,
    HybridEvaluateResponse,
    HybridExecuteRequest,
    HybridExecuteResponse,
    ServiceCapabilities,
    TVARDefinition,
)
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

__all__ = [
    # Transport
    "HybridTransport",
    "create_transport",
    # Exceptions
    "TransportError",
    "TransportConnectionError",
    "TransportTimeoutError",
    "TransportAuthError",
    "TransportRateLimitError",
    "TransportServerError",
    # Protocol DTOs
    "BatchOptions",
    "HybridExecuteRequest",
    "HybridExecuteResponse",
    "HybridEvaluateRequest",
    "HybridEvaluateResponse",
    "ServiceCapabilities",
    "TVARDefinition",
    "EvaluationKwargDefinition",
    "ConfigSpaceResponse",
    "BenchmarkEntry",
    "BenchmarksResponse",
    "HealthCheckResponse",
    # Lifecycle
    "AgentLifecycleManager",
    "SessionInfo",
    # Discovery
    "ConfigSpaceDiscovery",
    "merge_config_spaces",
    "normalize_tvar_to_config_space",
    "validate_config_against_tvars",
]
