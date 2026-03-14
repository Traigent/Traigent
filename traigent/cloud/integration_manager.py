"""Integration Manager for Traigent SDK and Traigent Backend.

This module provides a unified interface for managing the integration between
Traigent SDK and Traigent Backend, coordinating all the integration components
including model bridges, session lifecycle management, dataset conversion,
and MCP client operations.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

from .backend_bridges import SessionExperimentMapping
from .backend_client import BackendClientConfig, get_backend_client
from .dataset_converter import converter
from .models import (
    OptimizationRequest,
    OptimizationSession,
    OptimizationSessionStatus,
    TrialResultSubmission,
    TrialSuggestion,
)
from .production_mcp_client import get_production_mcp_client
from .sessions import SessionLifecycleManager

lifecycle_manager = SessionLifecycleManager()

logger = get_logger(__name__)

# Error message for uninitialized backend client
_BACKEND_CLIENT_NOT_INITIALIZED = "Backend client not initialized"


class IntegrationMode(Enum):
    """Integration execution modes."""

    EDGE_ANALYTICS = "edge_analytics"  # Client-side execution with local orchestration
    PRIVACY = (
        "privacy"  # Client-side execution, backend orchestration, privacy-preserving
    )
    STANDARD = (
        "standard"  # Client-side execution, backend orchestration, full data sharing
    )
    CLOUD = "cloud"  # Full cloud execution


@dataclass
class IntegrationConfig:
    """Configuration for SDK-Backend integration."""

    mode: IntegrationMode = IntegrationMode.PRIVACY
    backend_base_url: str | None = (
        None  # Will be set from BackendConfig if not provided
    )
    mcp_server_path: str = "python"
    mcp_server_args: list[str] | None = None
    enable_session_sync: bool = True
    session_sync_interval: float = 5.0
    enable_fallback: bool = True
    max_retries: int = 3
    timeout: float = 30.0

    def __post_init__(self) -> None:
        """Set backend URL from centralized config if not provided."""
        if self.backend_base_url is None:
            from traigent.config.backend_config import BackendConfig

            self.backend_base_url = BackendConfig.get_backend_url()


@dataclass
class IntegrationResult:
    """Result from integration operations."""

    success: bool
    session_id: str | None = None
    experiment_id: str | None = None
    experiment_run_id: str | None = None
    agent_id: str | None = None
    example_set_id: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IntegrationManager:
    """Unified manager for SDK-Backend integration.

    This class coordinates all integration components and provides a simple
    interface for setting up and managing optimization workflows that span
    both Traigent SDK and Traigent Backend.
    """

    def __init__(self, config: IntegrationConfig | None = None) -> None:
        """Initialize integration manager.

        Args:
            config: Integration configuration
        """
        self.config = config or IntegrationConfig()

        # Initialize components
        self._mcp_client: Any | None = None
        self._backend_client: Any | None = None
        self._initialized = False

        # State tracking
        self._state_lock = threading.Lock()
        self._active_integrations: dict[str, dict[str, Any]] = {}
        self._integration_stats: dict[str, Any] = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "active_sessions": 0,
        }

    async def initialize(self) -> bool:
        """Initialize integration manager and all components.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            logger.info("Initializing Traigent SDK-Backend integration manager")

            # Initialize MCP client
            if self.config.mcp_server_args is None:
                self.config.mcp_server_args = [
                    "-m",
                    "optigen_backend.mcp.server",
                    "--host",
                    "localhost",
                    "--port",
                    "5000",
                ]

            self._mcp_client = get_production_mcp_client(
                server_path=self.config.mcp_server_path,
                server_args=self.config.mcp_server_args,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

            # Initialize backend client
            backend_config = BackendClientConfig(
                backend_base_url=self.config.backend_base_url,
                enable_session_sync=self.config.enable_session_sync,
                session_sync_interval=self.config.session_sync_interval,
            )

            self._backend_client = get_backend_client(
                backend_config=backend_config,
                enable_fallback=self.config.enable_fallback,
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
            )

            self._initialized = True
            logger.info("Integration manager initialized successfully")
            return True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize integration manager: {e}")
            return False

    async def __aenter__(self) -> IntegrationManager:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

    # High-Level Integration Workflows

    async def start_optimization_integration(
        self,
        optimization_request: OptimizationRequest,
        mode: IntegrationMode | None = None,
    ) -> IntegrationResult:
        """Start integrated optimization workflow.

        Args:
            optimization_request: SDK optimization request
            mode: Optional integration mode override

        Returns:
            IntegrationResult with workflow details
        """
        if not self._initialized:
            await self.initialize()

        mode = mode or self.config.mode
        integration_id = f"integration_{int(time.time())}"

        try:
            logger.info(
                f"Starting {mode.value} optimization integration: {integration_id}"
            )

            with self._state_lock:
                self._integration_stats["total_integrations"] += 1

            if mode == IntegrationMode.PRIVACY:
                result = await self._start_privacy_integration(
                    optimization_request, integration_id
                )
            elif mode == IntegrationMode.EDGE_ANALYTICS:
                result = await self._start_edge_analytics_integration(
                    optimization_request, integration_id
                )
            elif mode == IntegrationMode.STANDARD:
                result = await self._start_standard_integration(
                    optimization_request, integration_id
                )
            elif mode == IntegrationMode.CLOUD:
                result = await self._start_cloud_integration(
                    optimization_request, integration_id
                )
            else:
                raise ValueError(f"Unsupported integration mode: {mode}") from None

            with self._state_lock:
                if result.success:
                    self._integration_stats["successful_integrations"] += 1
                    self._integration_stats["active_sessions"] += 1
                    self._active_integrations[integration_id] = {
                        "mode": mode.value,
                        "optimization_request": optimization_request,
                        "result": result,
                        "start_time": time.time(),
                    }
                else:
                    self._integration_stats["failed_integrations"] += 1

            return result

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to start optimization integration: {e}")
            with self._state_lock:
                self._integration_stats["failed_integrations"] += 1

            return IntegrationResult(success=False, error_message=str(e))

    async def _start_edge_analytics_integration(
        self, optimization_request: OptimizationRequest, integration_id: str
    ) -> IntegrationResult:
        """Start Edge Analytics integration with client-side orchestration.

        Args:
            optimization_request: SDK optimization request
            integration_id: Integration identifier

        Returns:
            IntegrationResult with Edge Analytics workflow details
        """
        logger.info(f"Starting edge_analytics integration: {integration_id}")

        try:
            # Edge Analytics mode: client-side orchestration with optional metrics submission
            # Using simple optimizers (grid, random, bayesian) locally

            session_id = f"edge_analytics_{integration_id}"

            return IntegrationResult(
                success=True,
                session_id=session_id,
                metadata={
                    "mode": "edge_analytics",
                    "integration_id": integration_id,
                    "orchestration": "client-side",
                },
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Local integration failed: {e}")
            return IntegrationResult(success=False, error_message=str(e))

    async def _start_privacy_integration(
        self, optimization_request: OptimizationRequest, integration_id: str
    ) -> IntegrationResult:
        """Start privacy-preserving integration.

        Args:
            optimization_request: SDK optimization request
            integration_id: Integration identifier

        Returns:
            IntegrationResult with privacy workflow details
        """
        if self._mcp_client is None:
            raise RuntimeError("MCP client not initialized")
        if self._backend_client is None:
            raise RuntimeError(_BACKEND_CLIENT_NOT_INITIALIZED)

        try:
            # Step 1: Generate agent specification from function if needed
            if not optimization_request.agent_specification:
                from ..agents.specification_generator import generator

                optimization_request.agent_specification = (
                    generator.from_function_signature(
                        optimization_request.function_name,
                        f"{optimization_request.function_name}(input: str) -> str",
                        configuration_space=optimization_request.configuration_space,
                        objectives=optimization_request.objectives,
                    )
                )

            # Step 2: Create privacy metadata (no actual data)
            privacy_metadata = converter.create_privacy_metadata(
                optimization_request.dataset, include_sample=True, sample_size=3
            )

            # Step 3: Create backend experiment tracking via MCP
            (
                agent_id,
                experiment_id,
                experiment_run_id,
            ) = await self._mcp_client.create_optimization_workflow(
                optimization_request
            )

            # Step 4: Create cloud session for configuration suggestions
            (
                session_id,
                _,
                _,
            ) = await self._backend_client.create_privacy_optimization_session(
                function_name=optimization_request.function_name,
                configuration_space=optimization_request.configuration_space,
                objectives=optimization_request.objectives,
                dataset_metadata=privacy_metadata,
                max_trials=optimization_request.max_trials,
            )

            # Step 5: Register with session lifecycle manager
            session = OptimizationSession(
                session_id=session_id,
                function_name=optimization_request.function_name,
                configuration_space=optimization_request.configuration_space,
                objectives=optimization_request.objectives,
                max_trials=optimization_request.max_trials,
                status=OptimizationSessionStatus.CREATED,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

            mapping = SessionExperimentMapping(
                session_id=session_id,
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
                function_name=optimization_request.function_name,
                configuration_space=optimization_request.configuration_space,
                objectives=optimization_request.objectives,
                trial_mappings={},
            )

            lifecycle_manager.register_session(session, mapping)
            lifecycle_manager.start_session(session_id)

            return IntegrationResult(
                success=True,
                session_id=session_id,
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
                agent_id=agent_id,
                metadata={
                    "mode": "privacy",
                    "privacy_metadata": privacy_metadata,
                    "integration_id": integration_id,
                },
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Privacy-first integration failed: {e}")
            return IntegrationResult(success=False, error_message=str(e))

    async def _start_cloud_integration(
        self, optimization_request: OptimizationRequest, integration_id: str
    ) -> IntegrationResult:
        """Start cloud SaaS integration (Model 2).

        Args:
            optimization_request: SDK optimization request
            integration_id: Integration identifier

        Returns:
            IntegrationResult with cloud workflow details
        """
        if self._mcp_client is None:
            raise RuntimeError("MCP client not initialized")
        if self._backend_client is None:
            raise RuntimeError(_BACKEND_CLIENT_NOT_INITIALIZED)

        try:
            # Step 1: Generate agent specification if needed
            if not optimization_request.agent_specification:
                from ..agents.specification_generator import generator

                optimization_request.agent_specification = (
                    generator.from_function_signature(
                        optimization_request.function_name,
                        f"{optimization_request.function_name}(input: str) -> str",
                        configuration_space=optimization_request.configuration_space,
                        objectives=optimization_request.objectives,
                    )
                )

            # Step 2: Create full backend workflow via MCP
            (
                agent_id,
                experiment_id,
                experiment_run_id,
            ) = await self._mcp_client.create_optimization_workflow(
                optimization_request
            )

            # Step 3: Start cloud agent optimization
            optimization_response = await self._backend_client.start_agent_optimization(
                agent_spec=optimization_request.agent_specification,
                dataset=optimization_request.dataset,
                configuration_space=optimization_request.configuration_space,
                objectives=optimization_request.objectives,
                max_trials=optimization_request.max_trials,
            )

            return IntegrationResult(
                success=True,
                session_id=optimization_response.session_id,
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
                agent_id=agent_id,
                metadata={
                    "mode": "cloud",
                    "optimization_id": optimization_response.optimization_id,
                    "estimated_cost": optimization_response.estimated_cost,
                    "estimated_duration": optimization_response.estimated_duration,
                    "integration_id": integration_id,
                },
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Cloud SaaS integration failed: {e}")
            return IntegrationResult(success=False, error_message=str(e))

    async def _start_standard_integration(
        self, optimization_request: OptimizationRequest, integration_id: str
    ) -> IntegrationResult:
        """Start standard integration with full data sharing.

        Args:
            optimization_request: SDK optimization request
            integration_id: Integration identifier

        Returns:
            IntegrationResult with hybrid workflow details
        """
        # Determine mode based on request characteristics
        privacy_indicators = [
            optimization_request.metadata.get("privacy_mode", False),
            optimization_request.metadata.get("sensitive_data", False),
            len(optimization_request.dataset.examples)
            > 1000,  # Large datasets prefer privacy
        ]

        if any(privacy_indicators):
            logger.info(f"Hybrid mode choosing privacy-first for {integration_id}")
            return await self._start_privacy_integration(
                optimization_request, integration_id
            )
        else:
            logger.info(f"Hybrid mode choosing cloud SaaS for {integration_id}")
            return await self._start_cloud_integration(
                optimization_request, integration_id
            )

    # Trial Management

    async def get_next_trial(
        self,
        session_id: str,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> TrialSuggestion | None:
        """Get next trial suggestion from integrated workflow.

        Args:
            session_id: Session ID
            previous_results: Previous trial results

        Returns:
            Next trial suggestion
        """
        if self._backend_client is None:
            raise RuntimeError(_BACKEND_CLIENT_NOT_INITIALIZED)

        try:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(session_id, "session_id")
            )
        except ValidationException as exc:
            logger.error(f"Invalid session_id provided: {exc}")
            return None

        try:
            # Get session state
            session_state = lifecycle_manager.get_session_state(session_id)
            if not session_state:
                logger.warning(f"Session {session_id} not found")
                return None

            # Get trial suggestion based on integration mode
            integration = self._get_integration_for_session(session_id)
            if not integration:
                logger.warning(f"No integration found for session {session_id}")
                return None

            mode = integration["mode"]

            if mode == "private":
                suggestion = await self._backend_client.get_next_privacy_trial(
                    session_id, previous_results
                )
            else:
                # For cloud SaaS, trials are managed by the cloud service
                suggestion = None
                logger.info("Cloud SaaS mode: trials managed by cloud service")

            # Note: Trial registration is handled differently in the refactored lifecycle manager
            # The refactored manager tracks trials when results are submitted
            if suggestion:
                logger.debug(
                    f"Generated trial suggestion {suggestion.trial_id} for session {session_id}"
                )

            return cast(TrialSuggestion | None, suggestion)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to get next trial for session {session_id}: {e}")
            return None

    async def submit_trial_results(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        duration: float,
        error_message: str | None = None,
    ) -> bool:
        """Submit trial results to integrated workflow.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            config: Configuration that was tested
            metrics: Trial metrics
            duration: Execution duration
            error_message: Optional error message

        Returns:
            True if submission successful
        """
        if self._backend_client is None:
            raise RuntimeError(_BACKEND_CLIENT_NOT_INITIALIZED)

        try:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(session_id, "session_id")
            )
            validate_or_raise(
                CoreValidators.validate_string_non_empty(trial_id, "trial_id")
            )
        except ValidationException as exc:
            logger.error(f"Invalid identifier provided: {exc}")
            return False

        try:
            # Note: Trial lifecycle is handled differently in the refactored manager
            # The refactored manager tracks trial outcomes when results are submitted to backend
            logger.debug(
                f"Processing trial results for {trial_id}: metrics={metrics}, duration={duration}"
            )

            # Submit to backend client
            integration = self._get_integration_for_session(session_id)
            if integration and integration["mode"] == "private":
                success = await self._backend_client.submit_privacy_trial_results(
                    session_id, trial_id, config, metrics, duration, error_message
                )
                return cast(bool, success)

            return True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to submit trial results: {e}")
            return False

    # Session Management

    async def finalize_session(
        self, session_id: str, final_results: dict[str, Any] | None = None
    ) -> bool:
        """Finalize optimization session.

        Args:
            session_id: Session ID
            final_results: Optional final results

        Returns:
            True if finalization successful
        """
        if self._backend_client is None:
            raise RuntimeError(_BACKEND_CLIENT_NOT_INITIALIZED)

        try:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(session_id, "session_id")
            )
        except ValidationException as exc:
            logger.error(f"Invalid session_id provided: {exc}")
            return False

        try:
            # Complete session in lifecycle manager
            lifecycle_manager.complete_session(session_id, final_results)

            # Finalize with backend client
            await self._backend_client.finalize_session(session_id, True)

            # Update integration tracking
            integration_id = self._get_integration_id_for_session(session_id)
            if integration_id:
                with self._state_lock:
                    if integration_id in self._active_integrations:
                        del self._active_integrations[integration_id]
                        self._integration_stats["active_sessions"] -= 1

            logger.info(f"Finalized session {session_id}")
            return True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to finalize session {session_id}: {e}")
            return False

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel optimization session.

        Args:
            session_id: Session ID

        Returns:
            True if cancellation successful
        """
        try:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(session_id, "session_id")
            )
        except ValidationException as exc:
            logger.error(f"Invalid session_id provided: {exc}")
            return False

        try:
            # Cancel in lifecycle manager
            lifecycle_manager.cancel_session(session_id)

            # Update integration tracking
            integration_id = self._get_integration_id_for_session(session_id)
            if integration_id:
                with self._state_lock:
                    if integration_id in self._active_integrations:
                        del self._active_integrations[integration_id]
                        self._integration_stats["active_sessions"] -= 1

            logger.info(f"Cancelled session {session_id}")
            return True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel session {session_id}: {e}")
            return False

    # Utility Methods

    def _get_integration_for_session(self, session_id: str) -> dict[str, Any] | None:
        """Get integration details for session."""
        with self._state_lock:
            for integration in self._active_integrations.values():
                if integration["result"].session_id == session_id:
                    return integration
        return None

    def _get_integration_id_for_session(self, session_id: str) -> str | None:
        """Get integration ID for session."""
        with self._state_lock:
            for integration_id, integration in self._active_integrations.items():
                if integration["result"].session_id == session_id:
                    return integration_id
        return None

    def get_integration_statistics(self) -> dict[str, Any]:
        """Get integration statistics."""
        with self._state_lock:
            stats_snapshot = dict(self._integration_stats)
            active_count = len(self._active_integrations)
        return {
            **stats_snapshot,
            "mcp_stats": self._mcp_client.get_statistics() if self._mcp_client else {},
            "lifecycle_stats": lifecycle_manager.get_statistics(),
            "active_integrations": active_count,
        }

    def get_active_integrations(self) -> dict[str, dict[str, Any]]:
        """Get active integrations."""
        with self._state_lock:
            return self._active_integrations.copy()

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health_status: dict[str, Any] = {
            "integration_manager": "healthy",
            "initialized": self._initialized,
            "components": {},
        }

        try:
            # Check MCP client
            if self._mcp_client:
                mcp_health = await self._mcp_client.health_check()
                health_status["components"]["mcp_client"] = (
                    "healthy" if mcp_health.success else "unhealthy"
                )
            else:
                health_status["components"]["mcp_client"] = "not_initialized"

            # Check backend client
            if self._backend_client:
                health_status["components"]["backend_client"] = "healthy"
            else:
                health_status["components"]["backend_client"] = "not_initialized"

            # Check lifecycle manager
            health_status["components"]["lifecycle_manager"] = "healthy"

        except asyncio.CancelledError:
            raise
        except Exception as e:
            health_status["integration_manager"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    async def cleanup(self) -> None:
        """Clean up integration manager resources."""
        try:
            # Cleanup MCP client
            if self._mcp_client:
                await self._mcp_client.disconnect()

            # Cleanup backend client
            if self._backend_client:
                # Backend client cleanup handled by context manager
                pass

            self._initialized = False
            logger.info("Integration manager cleanup completed")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during integration manager cleanup: {e}")


# Global integration manager instance
_integration_manager: IntegrationManager | None = None


def get_integration_manager(
    config: IntegrationConfig | None = None,
) -> IntegrationManager:
    """Get or create global integration manager.

    Args:
        config: Optional integration configuration

    Returns:
        IntegrationManager instance
    """
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager(config)
    return _integration_manager


def set_integration_manager(manager: IntegrationManager) -> None:
    """Set global integration manager.

    Args:
        manager: IntegrationManager instance
    """
    global _integration_manager
    _integration_manager = manager
