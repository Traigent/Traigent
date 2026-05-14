"""Base agent executor for Traigent SDK.

This module defines the base interface for executing AI agents
with different configurations and platforms.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-AGENTS REQ-AGNT-013

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict, cast

from traigent.utils.exceptions import AgentExecutionError
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.cloud.models import AgentSpecification

logger = get_logger(__name__)


class PlatformConfigValidationResult(TypedDict, total=False):
    """Result payload for platform configuration validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]


class CostEstimate(TypedDict, total=False):
    """Result payload for cost estimation."""

    estimated_cost: float
    estimated_input_cost: float
    estimated_output_cost: float
    estimated_tokens: int
    confidence: float


@dataclass
class AgentExecutionResult:
    """Result from agent execution."""

    output: Any
    duration: float
    tokens_used: int | None = None
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AgentExecutor(ABC):
    """Base class for agent executors.

    This abstract class defines the interface for executing agents
    with different platforms and configurations.

    Supports async context manager protocol for automatic cleanup:
        async with executor:
            await executor.execute(...)
    """

    def __init__(self, platform_config: dict[str, Any] | None = None) -> None:
        """Initialize agent executor.

        Args:
            platform_config: Platform-specific configuration
        """
        self.platform_config = platform_config or {}
        self._initialized = False
        self._cleanup_done = False

    async def initialize(self) -> None:
        """Initialize the executor with platform-specific setup."""
        if self._initialized:
            return

        await self._platform_initialize()
        self._initialized = True

    @abstractmethod
    async def _platform_initialize(self) -> None:
        """Platform-specific initialization."""
        raise NotImplementedError

    async def execute(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> AgentExecutionResult:
        """Execute an agent with given input and configuration.

        Args:
            agent_spec: Agent specification
            input_data: Input data for the agent
            config_overrides: Configuration overrides to apply

        Returns:
            AgentExecutionResult with output and metadata. On failure,
            the result will contain error information in the error field
            and metadata will include error details.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()

        try:
            # Validate agent specification
            self._validate_agent_spec(agent_spec)

            # Merge configurations
            effective_config = self._merge_configurations(
                agent_spec.model_parameters or {}, config_overrides
            )

            # Platform-specific execution
            result = await self._execute_agent(agent_spec, input_data, effective_config)

            # Calculate duration
            duration = time.perf_counter() - start_time

            # Create execution result
            return AgentExecutionResult(
                output=result.get("output"),
                duration=duration,
                tokens_used=result.get("tokens_used"),
                cost=result.get("cost"),
                metadata=result.get("metadata", {}),
                error=None,
            )

        except asyncio.CancelledError:
            # Re-raise cancellation to support cooperative cancellation
            raise
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Agent execution failed: {e}")

            return AgentExecutionResult(
                output=None,
                duration=duration,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )

    @abstractmethod
    async def _execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Platform-specific agent execution.

        Args:
            agent_spec: Agent specification
            input_data: Input data
            config: Effective configuration

        Returns:
            Dictionary with execution results
        """
        raise NotImplementedError

    def _validate_agent_spec(self, agent_spec: AgentSpecification) -> None:
        """Validate agent specification for the platform.

        Args:
            agent_spec: Agent specification to validate

        Raises:
            AgentExecutionError: If specification is invalid
        """
        if not agent_spec.prompt_template:
            raise AgentExecutionError("Agent must have a prompt template")

        if not agent_spec.agent_platform:
            raise AgentExecutionError("Agent must specify a platform")

        # Platform-specific validation
        self._validate_platform_spec(agent_spec)

    @abstractmethod
    def _validate_platform_spec(self, agent_spec: AgentSpecification) -> None:
        """Platform-specific validation."""
        raise NotImplementedError

    def _merge_configurations(
        self, base_config: dict[str, Any], overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge base configuration with overrides.

        Args:
            base_config: Base configuration from agent spec
            overrides: Configuration overrides

        Returns:
            Merged configuration
        """
        if not overrides:
            return base_config.copy()

        merged = base_config.copy()

        # Deep merge for nested configurations
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        return merged

    async def batch_execute(
        self,
        agent_spec: AgentSpecification,
        input_batch: list[dict[str, Any]],
        config_overrides: dict[str, Any] | None = None,
        max_concurrent: int = 5,
    ) -> list[AgentExecutionResult]:
        """Execute agent on a batch of inputs.

        Args:
            agent_spec: Agent specification
            input_batch: List of input data
            config_overrides: Configuration overrides
            max_concurrent: Maximum concurrent executions

        Returns:
            List of execution results
        """
        try:
            concurrency_limit = int(max_concurrent)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_concurrent must be a positive integer") from exc

        if concurrency_limit <= 0:
            raise ValueError("max_concurrent must be a positive integer")

        semaphore = asyncio.Semaphore(concurrency_limit)

        async def execute_with_semaphore(input_data: dict[str, Any]):
            async with semaphore:
                return await self.execute(agent_spec, input_data, config_overrides)

        tasks = [execute_with_semaphore(input_data) for input_data in input_batch]
        return cast(list[AgentExecutionResult], await asyncio.gather(*tasks))

    def get_platform_info(self) -> dict[str, Any]:
        """Get information about the platform and executor.

        Returns:
            Platform information dictionary
        """
        return {
            "platform": self.__class__.__name__,
            "initialized": self._initialized,
            "capabilities": self._get_platform_capabilities(),
            "config": self.platform_config,
        }

    @abstractmethod
    def _get_platform_capabilities(self) -> list[str]:
        """Get list of platform capabilities."""
        raise NotImplementedError

    async def estimate_cost(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> CostEstimate:
        """Estimate execution cost for given input.

        Args:
            agent_spec: Agent specification
            input_data: Input data
            config_overrides: Configuration overrides

        Returns:
            Cost estimation dictionary
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement cost estimation"
        )

    async def validate_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration for the platform.

        Args:
            config: Configuration to validate

        Returns:
            Validation results
        """
        errors = []
        warnings = []

        # Platform-specific validation
        platform_validation = await self._validate_platform_config(config)
        errors.extend(platform_validation.get("errors", []))
        warnings.extend(platform_validation.get("warnings", []))

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    @abstractmethod
    async def _validate_platform_config(
        self, config: dict[str, Any]
    ) -> PlatformConfigValidationResult:
        """Platform-specific configuration validation."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Cleanup resources and shutdown executor gracefully.

        This method ensures all resources are properly released, including:
        - Platform-specific resources (API clients, connections, etc.)
        - Running tasks (with cancellation)
        - Cached state

        Safe to call multiple times - subsequent calls are no-ops.
        """
        if self._cleanup_done:
            logger.debug("Cleanup already performed, skipping")
            return

        logger.info(f"Cleaning up {self.__class__.__name__}")

        try:
            # Platform-specific cleanup
            await self._platform_cleanup()

            # Mark as not initialized to prevent further usage
            self._initialized = False
            self._cleanup_done = True

            logger.info(f"Cleanup completed for {self.__class__.__name__}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Mark as done even if cleanup fails to prevent retries
            self._cleanup_done = True
            raise

    async def _platform_cleanup(self) -> None:
        """Platform-specific cleanup implementation.

        Override this method to cleanup platform-specific resources like:
        - API clients
        - Network connections
        - Background tasks
        - Temporary files or caches

        Default implementation is a no-op.
        """
        # Default: no cleanup needed
        await asyncio.sleep(0)  # Make it properly async

    async def __aenter__(self) -> AgentExecutor:
        """Enter async context manager - initialize executor.

        Returns:
            Self for use in async with statement
        """
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager - cleanup resources.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        await self.cleanup()
