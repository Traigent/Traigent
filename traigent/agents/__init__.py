"""Agent execution framework for Traigent SDK.

This module provides the infrastructure for executing AI agents
with optimized configurations.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-AGENTS REQ-AGNT-013 REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.utils.logging import get_logger

from .config_mapper import (
    ConfigurationMapper,
    ParameterMapping,
    PlatformMapping,
    apply_config_to_agent,
    get_mapping_platforms,
    get_supported_platforms,
    register_platform_mapping,
    validate_config_compatibility,
)
from .executor import AgentExecutionResult, AgentExecutor

# Import from the platforms.py file in current directory
try:
    from .platforms import (
        LangChainAgentExecutor,
        OpenAIAgentExecutor,
        PlatformRegistry,
        get_executor_for_platform,
    )

    _platforms_available = True
except ImportError:
    # Fallback if platforms.py is not available
    logger = get_logger(__name__)
    logger.debug("Platform executors not available - platforms.py import failed")
    LangChainAgentExecutor = None  # type: ignore[assignment,misc]
    OpenAIAgentExecutor = None  # type: ignore[assignment,misc]
    PlatformRegistry = None  # type: ignore[assignment,misc]
    get_executor_for_platform = None  # type: ignore[assignment,misc]
    _platforms_available = False

# Build __all__ dynamically to exclude None exports
__all__ = [
    "AgentExecutor",
    "AgentExecutionResult",
    "ConfigurationMapper",
    "ParameterMapping",
    "PlatformMapping",
    "apply_config_to_agent",
    "validate_config_compatibility",
    "register_platform_mapping",
    "get_supported_platforms",
    "get_mapping_platforms",
]

# Only include platform-specific exports if platforms are available
if _platforms_available:
    __all__.extend(
        [
            "LangChainAgentExecutor",
            "OpenAIAgentExecutor",
            "PlatformRegistry",
            "get_executor_for_platform",
        ]
    )
