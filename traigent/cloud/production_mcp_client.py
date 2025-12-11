"""Production MCP Client for OptiGen Backend Integration.

This module provides a production MCP client that connects to OptiGen Backend
MCP server, enabling real integration with backend services for optimization
and agent management.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, cast

# Optional dependencies
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import StdioClientTransport

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

    # Mock MCP classes for type hints
    class ClientSession:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("MCP not available") from None

    class StdioServerParameters:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            pass

    class StdioClientTransport:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            pass


from traigent.evaluators.base import Dataset
from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.retry import (
    NetworkError,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
)
from traigent.utils.validation import CoreValidators, validate_or_raise

from .backend_bridges import bridge
from .models import (
    AgentSpecification,
    OptimizationRequest,
)

logger = get_logger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for MCP server connection."""

    server_path: str
    server_args: list[str] | None = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """Validate server configuration parameters."""
        validate_or_raise(
            CoreValidators.validate_string_non_empty(self.server_path, "server_path")
        )

        if self.server_args is not None:
            if not isinstance(self.server_args, list):
                raise ValidationException(
                    "server_args must be a list of non-empty strings"
                )
            for idx, arg in enumerate(self.server_args):
                validate_or_raise(
                    CoreValidators.validate_string_non_empty(arg, f"server_args[{idx}]")
                )

        if self.timeout <= 0:
            raise ValidationException("timeout must be a positive value")

        if self.max_retries < 0:
            raise ValidationException("max_retries must be greater than or equal to 0")

        if self.retry_delay < 0:
            raise ValidationException("retry_delay must be non-negative")


@dataclass
class MCPResponse:
    """Response from MCP server operations."""

    success: bool
    data: dict[str, Any] | None = None
    error_message: str | None = None
    request_id: str | None = None


class ProductionMCPClient:
    """Production MCP client for OptiGen Backend integration.

    This client provides real integration with OptiGen Backend MCP server,
    enabling experiment management, agent operations, and optimization workflows.
    """

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> None:
        """Validate that identifier is a non-empty string."""
        validate_or_raise(CoreValidators.validate_string_non_empty(value, field_name))

    @staticmethod
    def _validate_mapping(mapping: dict[str, Any], field_name: str) -> None:
        """Validate that the given mapping is a dictionary."""
        validate_or_raise(CoreValidators.validate_dict(mapping, field_name))

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Validate that an integer value is positive."""
        validate_or_raise(CoreValidators.validate_positive_int(value, field_name))

    def _validate_tool_call_inputs(
        self, tool_name: str, arguments: dict[str, Any], operation_id: str
    ) -> MCPResponse | None:
        """Validate inputs for call_tool and return error response if invalid."""
        try:
            self._validate_identifier(tool_name, "tool_name")
        except ValidationException as exc:
            error_msg = str(exc)
            logger.error(error_msg)
            return MCPResponse(
                success=False, error_message=error_msg, request_id=operation_id
            )

        if not isinstance(arguments, dict):
            error_msg = "arguments must be a dictionary"
            logger.error(error_msg)
            return MCPResponse(
                success=False, error_message=error_msg, request_id=operation_id
            )
        return None

    def __init__(
        self, server_config: MCPServerConfig, enable_fallback: bool = True
    ) -> None:
        """Initialize production MCP client.

        Args:
            server_config: MCP server connection configuration
            enable_fallback: Enable fallback to local operations
        """
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, client will use fallback mode only")

        self.server_config = server_config
        self.enable_fallback = enable_fallback
        if self.server_config.server_args:
            self.server_config.server_args = [
                arg.strip() for arg in self.server_config.server_args
            ]

        # Connection management
        self._session: ClientSession | None = None
        self._transport: StdioClientTransport | None = None
        self._connected = False
        self._connection_lock = asyncio.Lock()

        # Operation tracking
        self._active_operations: dict[str, dict[str, Any]] = {}
        self._operation_results: dict[str, MCPResponse] = {}

        # Configure retry handler for MCP operations
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            exponential_base=2.0,
            jitter=True,
            retry_on_timeout=True,
            retry_on_connection_error=True,
        )
        self._retry_handler = RetryHandler(retry_config)

        # Statistics
        self._stats: dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # Connection Management

    async def connect(self) -> bool:
        """Connect to MCP server.

        Returns:
            True if connection successful
        """
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, using fallback mode")
            return False

        async with self._connection_lock:
            if self._connected:
                return True

            self._stats["connection_attempts"] += 1

            try:
                # Create server parameters
                server_params = StdioServerParameters(
                    command=self.server_config.server_path,
                    args=self.server_config.server_args or [],
                    env=None,
                )

                # Create transport and session
                self._transport = StdioClientTransport(server_params)
                self._session = ClientSession(self._transport)

                # Initialize connection
                await self._session.initialize()

                self._connected = True
                self._stats["successful_connections"] += 1

                logger.info(
                    f"Connected to MCP server: {self.server_config.server_path}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                await self._cleanup_connection()
                return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        async with self._connection_lock:
            if self._connected:
                await self._cleanup_connection()
                logger.info("Disconnected from MCP server")

    async def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected and self._session is not None

    async def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        try:
            if self._session:
                await self._session.close()
            if self._transport:
                await self._transport.close()
        except Exception as e:
            logger.warning(f"Error during connection cleanup: {e}")
        finally:
            self._session = None
            self._transport = None
            self._connected = False

    # Core MCP Operations

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], operation_id: str | None = None
    ) -> MCPResponse:
        """Call MCP tool on backend server with retry handling.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments
            operation_id: Optional operation identifier

        Returns:
            MCP response with operation results
        """
        operation_id = operation_id or str(uuid.uuid4())
        validation_error = self._validate_tool_call_inputs(
            tool_name, arguments, operation_id
        )
        if validation_error:
            self._operation_results[operation_id] = validation_error
            return validation_error

        # Track operation
        self._active_operations[operation_id] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "start_time": time.time(),
        }

        try:

            async def execute_tool_call():
                """Internal function for tool execution with proper error handling."""
                if not await self.is_connected():
                    if not await self.connect():
                        raise NetworkError("Unable to connect to MCP server") from None

                assert self._session is not None, "Connected but session is None"
                self._stats["total_requests"] += 1

                # Call tool via MCP
                try:
                    result = await self._session.call_tool(tool_name, arguments)

                    response = MCPResponse(
                        success=True,
                        data=result.content[0].text if result.content else None,
                        request_id=operation_id,
                    )

                    self._stats["successful_requests"] += 1
                    logger.debug(f"MCP tool call successful: {tool_name}")
                    return response

                except ConnectionError as e:
                    # Connection errors should be retryable
                    raise NetworkError(f"MCP connection error: {e}") from None
                except TimeoutError as e:
                    # Timeout errors should be retryable
                    raise NetworkError(f"MCP operation timeout: {e}") from None
                except Exception as e:
                    # Other exceptions may not be retryable
                    logger.error(f"MCP tool call failed: {tool_name} - {e}")
                    raise

            # Use retry handler for robust tool execution
            try:
                result = await self._retry_handler.execute_async(execute_tool_call)

                if result.success:
                    response = result.result
                    self._operation_results[operation_id] = response
                    return cast(MCPResponse, response)
                else:
                    # All retry attempts failed
                    raise result.last_exception or Exception(
                        "Tool call failed after retries"
                    )

            except Exception as e:
                self._stats["failed_requests"] += 1

                response = MCPResponse(
                    success=False, error_message=str(e), request_id=operation_id
                )

                if self.enable_fallback:
                    # Attempt fallback operation
                    fallback_response = await self._fallback_operation(
                        tool_name, arguments
                    )
                    if fallback_response.success:
                        return fallback_response

                self._operation_results[operation_id] = response
                return response

        finally:
            # Clean up operation tracking
            self._active_operations.pop(operation_id, None)

    async def list_resources(self) -> MCPResponse:
        """List available MCP resources.

        Returns:
            MCP response with available resources
        """
        try:
            if not await self.is_connected():
                if not await self.connect():
                    raise ConnectionError("Unable to connect to MCP server") from None

            assert self._session is not None, "Connected but session is None"
            resources = await self._session.list_resources()

            return MCPResponse(
                success=True,
                data={"resources": [asdict(r) for r in resources.resources]},
            )

        except Exception as e:
            logger.error(f"Failed to list MCP resources: {e}")
            return MCPResponse(success=False, error_message=str(e))

    async def read_resource(self, uri: str) -> MCPResponse:
        """Read MCP resource.

        Args:
            uri: Resource URI

        Returns:
            MCP response with resource content
        """
        try:
            if not await self.is_connected():
                if not await self.connect():
                    raise ConnectionError("Unable to connect to MCP server") from None

            assert self._session is not None, "Connected but session is None"
            resource = await self._session.read_resource(uri)

            return MCPResponse(
                success=True,
                data={
                    "content": resource.contents[0].text if resource.contents else None
                },
            )

        except Exception as e:
            logger.error(f"Failed to read MCP resource {uri}: {e}")
            return MCPResponse(success=False, error_message=str(e))

    # Backend Integration Operations

    async def create_experiment(
        self, optimization_request: OptimizationRequest
    ) -> MCPResponse:
        """Create experiment via MCP backend.

        Args:
            optimization_request: SDK optimization request

        Returns:
            MCP response with experiment details
        """
        if not isinstance(optimization_request, OptimizationRequest):
            raise ValidationException(
                "optimization_request must be an OptimizationRequest instance"
            )
        self._validate_identifier(
            optimization_request.function_name, "optimization_request.function_name"
        )
        if optimization_request.metadata is not None:
            self._validate_mapping(
                optimization_request.metadata, "optimization_request.metadata"
            )

        # Convert to backend format
        backend_request = bridge.optimization_request_to_backend(optimization_request)

        # Prepare tool arguments
        arguments = {
            "name": backend_request.name,
            "description": backend_request.description,
            "agent_id": (
                backend_request.agent_data.get("agent_id")
                if hasattr(backend_request, "agent_data")
                else None
            ),
            "example_set_id": (
                backend_request.example_set_data.get("example_set_id")
                if hasattr(backend_request, "example_set_data")
                else None
            ),
            "measures": backend_request.measures,
            "metadata": backend_request.metadata,
        }

        return await self.call_tool("create_experiment", arguments)

    async def start_experiment_run(
        self,
        experiment_id: str,
        configuration_space: dict[str, Any],
        max_trials: int = 50,
    ) -> MCPResponse:
        """Start experiment run via MCP backend.

        Args:
            experiment_id: Backend experiment ID
            configuration_space: Parameter search space
            max_trials: Maximum trials

        Returns:
            MCP response with experiment run details
        """
        self._validate_identifier(experiment_id, "experiment_id")
        self._validate_mapping(configuration_space, "configuration_space")
        self._validate_positive_int(max_trials, "max_trials")

        arguments = {
            "experiment_id": experiment_id,
            "subset_parameters": list(configuration_space.keys()),
            "subset_size": max_trials,
            "metadata": {
                "configuration_space": configuration_space,
                "max_trials": max_trials,
                "created_from": "sdk_optimization",
            },
        }

        return await self.call_tool("start_experiment_run", arguments)

    async def create_configuration_run(
        self, experiment_run_id: str, config: dict[str, Any], trial_id: str
    ) -> MCPResponse:
        """Create configuration run via MCP backend.

        Args:
            experiment_run_id: Backend experiment run ID
            config: Trial configuration
            trial_id: SDK trial ID

        Returns:
            MCP response with configuration run details
        """
        self._validate_identifier(experiment_run_id, "experiment_run_id")
        self._validate_identifier(trial_id, "trial_id")
        self._validate_mapping(config, "config")

        arguments = {
            "experiment_run_id": experiment_run_id,
            "configuration": config,
            "metadata": {"sdk_trial_id": trial_id, "created_from": "sdk_trial"},
        }

        return await self.call_tool("create_configuration_run", arguments)

    async def update_configuration_run_results(
        self,
        config_run_id: str,
        metrics: dict[str, float],
        status: str,
        error_message: str | None = None,
    ) -> MCPResponse:
        """Update configuration run results via MCP backend.

        Args:
            config_run_id: Backend configuration run ID
            metrics: Trial metrics
            status: Trial status
            error_message: Optional error message

        Returns:
            MCP response confirming update
        """
        self._validate_identifier(config_run_id, "config_run_id")
        self._validate_mapping(metrics, "metrics")
        self._validate_identifier(status, "status")

        arguments = {
            "config_run_id": config_run_id,
            "status": status,
            "results": metrics,
            "error_message": error_message,
            "completed_at": time.time(),
        }

        return await self.call_tool("update_configuration_run", arguments)

    async def create_agent(self, agent_spec: AgentSpecification) -> MCPResponse:
        """Create agent via MCP backend.

        Args:
            agent_spec: SDK agent specification

        Returns:
            MCP response with agent details
        """
        if not isinstance(agent_spec, AgentSpecification):
            raise ValidationException(
                "agent_spec must be an AgentSpecification instance"
            )

        # Convert to backend format
        backend_agent = bridge.agent_specification_to_backend(agent_spec)

        arguments = {
            "name": backend_agent.get("name"),
            "agent_type": backend_agent.get("agent_type"),
            "platform": backend_agent.get("platform"),
            "prompt_template": backend_agent.get("prompt_template"),
            "model_parameters": backend_agent.get("model_parameters"),
            "metadata": backend_agent.get("metadata", {}),
        }

        return await self.call_tool("create_agent", arguments)

    async def upload_dataset(
        self, dataset: Dataset, agent_id: str | None = None, privacy_mode: bool = False
    ) -> MCPResponse:
        """Upload dataset via MCP backend.

        Args:
            dataset: SDK dataset
            agent_id: Optional agent ID
            privacy_mode: Enable privacy-preserving upload

        Returns:
            MCP response with dataset/example set details
        """
        if not isinstance(dataset, Dataset):
            raise ValidationException("dataset must be a Dataset instance")
        if agent_id is not None:
            self._validate_identifier(agent_id, "agent_id")

        from .dataset_converter import converter

        # Convert dataset
        examples, metadata = converter.sdk_dataset_to_backend_examples(
            dataset, privacy_mode
        )

        arguments = {
            "name": metadata.name,
            "type": metadata.type,
            "description": metadata.description,
            "agent_id": agent_id,
            "examples": examples,
            "privacy_mode": privacy_mode,
        }

        return await self.call_tool("upload_example_set", arguments)

    # High-Level Workflow Operations

    async def create_optimization_workflow(
        self, optimization_request: OptimizationRequest
    ) -> tuple[str, str, str]:
        """Create complete optimization workflow via MCP backend.

        This creates agent, example set, experiment, and experiment run
        in a single coordinated workflow.

        Args:
            optimization_request: SDK optimization request

        Returns:
            Tuple of (agent_id, experiment_id, experiment_run_id)

        Raises:
            RuntimeError: If workflow creation fails
        """
        if not isinstance(optimization_request, OptimizationRequest):
            raise ValidationException(
                "optimization_request must be an OptimizationRequest instance"
            )
        if optimization_request.agent_specification is None:
            raise ValidationException(
                "optimization_request.agent_specification is required"
            )
        if not isinstance(optimization_request.dataset, Dataset):
            raise ValidationException("optimization_request.dataset must be a Dataset")

        try:
            # Step 1: Create agent if needed
            agent_response = await self.create_agent(
                optimization_request.agent_specification
            )
            if not agent_response.success:
                raise RuntimeError(
                    f"Failed to create agent: {agent_response.error_message}"
                )

            agent_data = (
                json.loads(agent_response.data)
                if isinstance(agent_response.data, str)
                else agent_response.data
            )
            agent_id = (agent_data or {}).get("agent_id")

            # Step 2: Upload dataset
            dataset_response = await self.upload_dataset(
                optimization_request.dataset,
                agent_id,
                optimization_request.metadata.get("privacy_mode", False),
            )
            if not dataset_response.success:
                raise RuntimeError(
                    f"Failed to upload dataset: {dataset_response.error_message}"
                )

            dataset_data = (
                json.loads(dataset_response.data)
                if isinstance(dataset_response.data, str)
                else dataset_response.data
            )
            example_set_id = (dataset_data or {}).get("example_set_id")

            # Step 3: Create experiment
            # Update optimization request with backend IDs
            optimization_request.agent_id = agent_id  # type: ignore[attr-defined]
            optimization_request.example_set_id = example_set_id  # type: ignore[attr-defined]

            experiment_response = await self.create_experiment(optimization_request)
            if not experiment_response.success:
                raise RuntimeError(
                    f"Failed to create experiment: {experiment_response.error_message}"
                )

            experiment_data = (
                json.loads(experiment_response.data)
                if isinstance(experiment_response.data, str)
                else experiment_response.data
            )
            experiment_id = (experiment_data or {}).get("experiment_id")
            if not experiment_id:
                raise RuntimeError("Experiment ID not returned from creation response")

            # Step 4: Start experiment run
            run_response = await self.start_experiment_run(
                str(experiment_id),
                optimization_request.configuration_space,
                optimization_request.max_trials,
            )
            if not run_response.success:
                raise RuntimeError(
                    f"Failed to start experiment run: {run_response.error_message}"
                )

            run_data = (
                json.loads(run_response.data)
                if isinstance(run_response.data, str)
                else run_response.data
            )
            experiment_run_id = (run_data or {}).get("experiment_run_id")

            logger.info(
                f"Created optimization workflow: {agent_id} -> {experiment_id} -> {experiment_run_id}"
            )
            return str(agent_id or ""), str(experiment_id), str(experiment_run_id or "")

        except Exception as e:
            logger.error(f"Failed to create optimization workflow: {e}")
            raise RuntimeError(f"Workflow creation failed: {e}") from None

    # Fallback Operations

    async def _fallback_operation(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPResponse:
        """Fallback operation when MCP server unavailable.

        Args:
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Fallback response
        """
        logger.info(f"Using fallback for MCP tool: {tool_name}")

        # Simple fallback responses for testing
        fallback_responses = {
            "create_experiment": {
                "experiment_id": f"fallback_exp_{int(time.time())}",
                "status": "created",
            },
            "start_experiment_run": {
                "experiment_run_id": f"fallback_run_{int(time.time())}",
                "status": "started",
            },
            "create_configuration_run": {
                "config_run_id": f"fallback_config_{int(time.time())}",
                "status": "created",
            },
            "create_agent": {
                "agent_id": f"fallback_agent_{int(time.time())}",
                "status": "created",
            },
            "upload_example_set": {
                "example_set_id": f"fallback_dataset_{int(time.time())}",
                "status": "uploaded",
            },
        }

        if tool_name in fallback_responses:
            return MCPResponse(success=True, data=fallback_responses[tool_name])
        else:
            return MCPResponse(
                success=False,
                error_message=f"No fallback available for tool: {tool_name}",
            )

    # Utility Methods

    def get_statistics(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "active_operations": len(self._active_operations),
            "cached_results": len(self._operation_results),
        }

    def get_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get currently active operations."""
        return self._active_operations.copy()

    async def health_check(self) -> MCPResponse:
        """Perform health check on MCP connection."""
        try:
            if not await self.is_connected():
                return MCPResponse(
                    success=False, error_message="Not connected to MCP server"
                )

            # Try to list resources as a simple health check
            resources_response = await self.list_resources()

            if resources_response.success:
                return MCPResponse(
                    success=True,
                    data={"status": "healthy", "resources_available": True},
                )
            else:
                return MCPResponse(
                    success=False,
                    error_message="Health check failed: unable to list resources",
                )

        except Exception as e:
            return MCPResponse(success=False, error_message=f"Health check error: {e}")


# Global production client instance
_production_client: ProductionMCPClient | None = None


def get_production_mcp_client(
    server_path: str = "python", server_args: list[str] | None = None, **kwargs
) -> ProductionMCPClient:
    """Get or create global production MCP client.

    Args:
        server_path: Path to MCP server executable
        server_args: Server arguments
        **kwargs: Additional client configuration

    Returns:
        ProductionMCPClient instance
    """
    global _production_client

    if _production_client is None:
        # Default server args for OptiGen Backend MCP
        if server_args is None:
            server_args = [
                "-m",
                "optigen_backend.mcp.server",
                "--host",
                "localhost",
                "--port",
                "5000",
            ]

        server_config = MCPServerConfig(
            server_path=server_path, server_args=server_args, **kwargs
        )

        _production_client = ProductionMCPClient(server_config)

    return _production_client


def set_production_mcp_client(client: ProductionMCPClient) -> None:
    """Set global production MCP client.

    Args:
        client: ProductionMCPClient instance
    """
    global _production_client
    _production_client = client
