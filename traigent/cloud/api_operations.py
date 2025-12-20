"""Backend API operations for TraiGent Cloud Client.

This module contains methods for interacting with the backend API endpoints,
including session creation, status updates, and configuration run management.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from traigent.cloud.client import CloudServiceError
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    DatasetSubsetIndices,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationSessionStatus,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
    TrialSuggestion,
)
from traigent.config.backend_config import BackendConfig
from traigent.utils.env_config import is_mock_mode
from traigent.utils.logging import get_logger

# Optional aiohttp dependency handling
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

    class _AiohttpPlaceholder:
        """Minimal placeholder to satisfy type references when aiohttp missing."""

        class ClientConnectorError(RuntimeError):
            pass

        class ClientError(RuntimeError):
            pass

        class ClientTimeout:  # pragma: no cover - placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("aiohttp is not installed")

        class TCPConnector:  # pragma: no cover - placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("aiohttp is not installed")

        class ClientSession:  # pragma: no cover - placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("aiohttp is not installed")

    aiohttp = _AiohttpPlaceholder()

# Track whether we've already warned about backend unavailability (per session)
_backend_unavailable_warned: bool = False

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

logger = get_logger(__name__)


class ApiOperations:
    """Handles backend API operations."""

    def __init__(self, client: "BackendIntegratedClient"):
        """Initialize API operations handler.

        Args:
            client: Parent BackendIntegratedClient instance
        """
        self.client = client

    def validate_and_sanitize_url(self, url: str) -> str:
        """Validate and sanitize URL to prevent injection attacks.

        Args:
            url: URL to validate

        Returns:
            Sanitized URL

        Raises:
            ValueError: If URL is invalid or uses unsafe scheme
        """
        if not url:
            raise ValueError("URL cannot be empty") from None

        # Parse URL
        parsed = urlparse(url)

        # Validate scheme
        allowed_schemes = {"http", "https"}
        if parsed.scheme not in allowed_schemes:
            raise ValueError(
                f"URL scheme must be one of {allowed_schemes}, got {parsed.scheme}"
            )

        # Validate host
        if not parsed.netloc:
            raise ValueError("URL must have a valid host")

        # Check for localhost/private IPs only in development
        if parsed.hostname:
            hostname = parsed.hostname.lower()
            # Allow localhost for development but provide helpful context
            if hostname in ["localhost", "127.0.0.1", "::1"]:
                logger.info(f"🔧 Using local backend URL: {url}")
                logger.info("💡 TraiGent is configured for local development mode")
            # Block other private IP ranges
            elif (
                hostname.startswith("192.168.")
                or hostname.startswith("10.")
                or hostname.startswith("172.")
            ):
                logger.warning(f"Using private IP address: {hostname}")

        # Reconstruct clean URL without fragments or unnecessary components
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url += f"?{parsed.query}"

        return clean_url.rstrip("/")

    def map_to_backend_status(self, status: str) -> str:
        """Map TraiGent status values to backend-expected values.

        TraiGent uses lowercase (pending, completed, failed)
        Backend expects uppercase (CREATED, ACTIVE, COMPLETED, FAILED, PRUNED)

        Note: PRUNED is a success case (early stopping for efficiency).
        CANCELLED indicates the trial was abandoned before execution.
        """
        status_mapping = {
            "pending": "ACTIVE",
            "running": "ACTIVE",
            "in_progress": "ACTIVE",
            "completed": "COMPLETED",
            "failed": "FAILED",
            "not_started": "CREATED",
            # Pruned is early stopping (success case)
            "pruned": "PRUNED",
            # Cancelled means trial didn't execute
            "cancelled": "CANCELLED",
            # Also handle if already uppercase
            "PENDING": "ACTIVE",
            "IN_PROGRESS": "ACTIVE",
            "COMPLETED": "COMPLETED",
            "FAILED": "FAILED",
            "CREATED": "CREATED",
            "ACTIVE": "ACTIVE",
            "PRUNED": "PRUNED",
            "CANCELLED": "CANCELLED",
        }
        return status_mapping.get(status, status.upper())

    def sanitize_error_message(self, error_message: str | None) -> str | None:
        """Sanitize error message for transmission.

        Args:
            error_message: Raw error message

        Returns:
            Sanitized error message or None
        """
        if not error_message:
            return None

        # Remove sensitive information patterns
        import re

        # Remove potential secrets/tokens
        sanitized = re.sub(
            r"(api[_-]?key|token|secret|password)[\s=:]+[\w-]+",
            "[REDACTED]",
            error_message,
            flags=re.IGNORECASE,
        )

        # Remove file paths that might contain user info
        sanitized = re.sub(r"/home/[\w/.-]+", "[PATH]", sanitized)
        sanitized = re.sub(r"/Users/[\w/.-]+", "[PATH]", sanitized)

        # Truncate if too long
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated]"

        return sanitized

    async def create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ) -> tuple[str, str, str]:
        """Create a TraiGent optimization session using the new session endpoints.

        Returns:
            Tuple of (session_id, experiment_id, experiment_run_id)
        """
        # Skip cloud session creation in mock mode - run fully offline
        if is_mock_mode():
            logger.debug("Mock mode: using local session IDs")
            return (
                f"mock_session_{int(time.time())}",
                f"mock_exp_{int(time.time())}",
                f"mock_run_{int(time.time())}",
            )

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, using fallback IDs")
            return (
                f"session_{int(time.time())}",
                f"exp_{int(time.time())}",
                f"run_{int(time.time())}",
            )

        try:
            max_trials_value = self._resolve_max_trials(session_request)
            session_payload = self._build_session_payload(
                session_request, max_trials_value
            )
            connector = self._build_connector()
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )
            return await self._post_session_creation(
                session_payload, headers, connector
            )
        except aiohttp.ClientConnectorError as e:
            self._handle_connector_error(e)
        except aiohttp.ClientError as e:
            self._handle_client_error(e)
        except Exception as e:
            self._handle_generic_session_exception(e)
        raise CloudServiceError("Unexpected session creation failure")

    def _resolve_max_trials(self, session_request: SessionCreationRequest) -> int:
        """Resolve max trials with logging when falling back to default."""

        value = session_request.max_trials
        if value is None:
            logger.warning("max_trials is None in session_request, using default 10")
            return 10
        return value

    def _build_session_payload(
        self, session_request: SessionCreationRequest, max_trials: int
    ) -> dict[str, Any]:
        """Build the payload sent to the cloud session creation endpoint."""

        metadata = session_request.metadata or {}
        evaluation_set = metadata.get("evaluation_set", "default")

        return {
            "problem_statement": session_request.function_name,
            "dataset": {
                "examples": [],  # Privacy mode - no actual data sent
                "metadata": session_request.dataset_metadata,
            },
            "search_space": session_request.configuration_space,
            "optimization_config": {
                "algorithm": "grid",
                "max_trials": max_trials,
                "optimization_goal": (
                    session_request.objectives[0]
                    if session_request.objectives
                    else "maximize"
                ),
            },
            "metadata": {
                "function_name": session_request.function_name,
                "evaluation_set": evaluation_set,
                **metadata,
            },
        }

    def _build_connector(self) -> Any | None:
        """Create an aiohttp connector, disabling SSL for localhost usage."""

        backend_url = self.client.backend_config.backend_base_url
        if backend_url and ("localhost" in backend_url or "127.0.0.1" in backend_url):
            return cast(Any, aiohttp).TCPConnector(ssl=False)
        return None

    async def _post_session_creation(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        connector: Any | None,
    ) -> tuple[str, str, str]:
        """Send the session creation request and handle responses."""

        async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
            api_base = (
                self.client.backend_config.api_base_url
                or BackendConfig.get_backend_api_url()
            )
            url = f"{api_base}/sessions"
            timeout = cast(Any, aiohttp).ClientTimeout(total=30)

            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status == 201:
                    return await self._parse_session_response(response)

                error_msg = await response.text()
                self._handle_session_error(response.status, error_msg)

        raise CloudServiceError("Unexpected session creation failure")

    async def _parse_session_response(self, response: Any) -> tuple[str, str, str]:
        """Parse the JSON success payload returned by the session endpoint."""

        result = await response.json()
        session_id = result.get("session_id")
        experiment_id = result.get("metadata", {}).get("experiment_id", session_id)
        experiment_run_id = result.get("metadata", {}).get(
            "experiment_run_id", session_id
        )

        logger.info(
            f"✅ Created TraiGent session: {session_id} "
            f"(exp: {experiment_id}, run: {experiment_run_id})"
        )
        return session_id, experiment_id, experiment_run_id

    def _handle_session_error(self, status_code: int, error_msg: str) -> None:
        """Log and raise appropriate errors for non-success HTTP responses."""

        if status_code == 500:
            logger.warning("⚡ Cloud backend returned server error (HTTP 500)")
            logger.info(
                "💡 The backend service may be starting up or temporarily unavailable"
            )
            logger.info("   TraiGent will fall back to local optimization")
            raise CloudServiceError(
                "Cloud backend temporarily unavailable (server error) - using local optimization"
            )

        if status_code == 503:
            logger.warning("⚡ Cloud backend service unavailable (HTTP 503)")
            logger.info(
                "💡 The backend service is temporarily overloaded or down for maintenance"
            )
            logger.info("   TraiGent will fall back to local optimization")
            raise CloudServiceError(
                "Cloud backend service unavailable - using local optimization"
            )

        if status_code in {502, 504}:
            logger.warning(f"⚡ Cloud backend gateway error (HTTP {status_code})")
            logger.info("💡 There's a temporary network issue reaching the backend")
            logger.info("   TraiGent will fall back to local optimization")
            raise CloudServiceError(
                "Cloud backend gateway error - using local optimization"
            )

        logger.error(
            f"Failed to create TraiGent session: {status_code} - {error_msg[:200]}"
        )
        raise CloudServiceError(
            f"Failed to create session via /api/v1/sessions endpoint: {status_code} - {error_msg[:200]}"
        )

    def _handle_connector_error(self, error: aiohttp.ClientConnectorError) -> None:
        """Handle aiohttp connector errors with contextual logging."""
        global _backend_unavailable_warned

        # Skip warnings entirely in mock mode
        if is_mock_mode():
            logger.debug(f"Backend connection failed (mock mode): {error}")
        # Only show full warning once per session to reduce log noise
        elif not _backend_unavailable_warned:
            logger.warning(f"⚡ Cloud backend unavailable (connection failed): {error}")
            if "localhost" in str(error) or "127.0.0.1" in str(error):
                logger.info(
                    "💡 This is normal for local development - backend tracking unavailable"
                )
                logger.info("   Results will be saved to local storage only")
            _backend_unavailable_warned = True
        else:
            logger.debug(f"Backend connection failed: {error}")

        raise CloudServiceError(
            "Backend unavailable for tracking - optimization will continue with local storage only"
        ) from error

    def _handle_client_error(self, error: aiohttp.ClientError) -> None:
        """Handle generic aiohttp client errors."""

        logger.warning(f"⚡ Network error connecting to cloud backend: {error}")
        raise CloudServiceError(f"Network error: {error}") from None

    def _handle_generic_session_exception(self, error: Exception) -> None:
        """Handle unexpected exceptions raised during session creation."""

        error_msg = str(error)
        if "500" in error_msg or "Internal Server Error" in error_msg:
            logger.warning("⚡ Cloud backend returned server error (HTTP 500)")
            logger.info(
                "💡 This typically means the backend service is starting up or temporarily unavailable"
            )
            logger.info("   Results will be saved to local storage only")
            raise CloudServiceError(
                "Backend temporarily unavailable (server error) - using local storage for tracking"
            ) from error

        if "Connection refused" in error_msg or "ConnectionRefusedError" in error_msg:
            logger.warning("⚡ Cloud backend connection refused")
            logger.info("💡 The backend service may not be running or accessible")
            logger.info("   TraiGent will continue with local optimization")
            raise CloudServiceError(
                "Cloud backend service not accessible - using local optimization"
            ) from error

        logger.error(f"Error creating TraiGent session: {error}")
        raise error

    async def update_config_run_status(self, config_run_id: str, status: str) -> bool:
        """Update configuration run status in the backend.

        Args:
            config_run_id: Configuration run ID (same as trial_id)
            status: Status to set (COMPLETED, FAILED, etc.)

        Returns:
            True if successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            return False

        try:
            # Create connector without SSL for localhost
            connector = None
            backend_url = self.client.backend_config.backend_base_url
            if backend_url and (
                "localhost" in backend_url or "127.0.0.1" in backend_url
            ):
                connector = cast(Any, aiohttp).TCPConnector(ssl=False)

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/configuration-runs/{config_run_id}/status"
                status_data = {"status": status}

                async with session.put(
                    url,
                    json=status_data,
                    headers=headers,
                    timeout=cast(Any, aiohttp).ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 204]:
                        logger.debug(
                            f"Updated configuration run {config_run_id} status to {status}"
                        )
                        return True
                    else:
                        error_msg = await response.text()
                        logger.debug(
                            f"Failed to update config run status: {response.status} - {error_msg[:100]}"
                        )
                        return False

        except Exception as e:
            logger.debug(f"Error updating config run status: {e}")
            return False

    async def update_config_run_measures(
        self,
        config_run_id: str,
        metrics: dict[str, float],
        execution_time: float | None = None,
    ) -> bool:
        """Update configuration run measures in the backend.

        Args:
            config_run_id: Configuration run ID (same as trial_id)
            metrics: Metrics to convert to measures
            execution_time: Optional execution time in seconds

        Returns:
            True if successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE or not metrics:
            return False

        try:
            # Convert metrics to backend measures format per OptiGen schema
            # Map TraiGent metrics to OptiGen standard measure IDs
            mapped_metrics = {}

            # Helper function to ensure value is a valid number
            def ensure_numeric(value):
                """Ensure value is a valid number, converting if necessary."""
                if value is None:
                    return 0.0
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            # Standard OptiGen measure mappings
            if "accuracy" in metrics and metrics["accuracy"] is not None:
                mapped_metrics["accuracy"] = ensure_numeric(metrics["accuracy"])
            if "score" in metrics and metrics["score"] is not None:
                mapped_metrics["score"] = ensure_numeric(metrics["score"])
            elif (
                "accuracy" in metrics and metrics["accuracy"] is not None
            ):  # Use accuracy as score if no explicit score
                mapped_metrics["score"] = ensure_numeric(metrics["accuracy"])

            # Include other standard measures if present
            for key in [
                "faithfulness",
                "relevance",
                "latency",
                "cost",
                "context_precision",
                "context_recall",
            ]:
                if key in metrics and metrics[key] is not None:
                    mapped_metrics[key] = ensure_numeric(metrics[key])

            # Include any other metrics as-is, ensuring they are numeric
            for key, value in metrics.items():
                if key not in mapped_metrics and value is not None:
                    mapped_metrics[key] = ensure_numeric(value)

            # Build measures data in the correct OptiGen format
            measures_data = {"measures": {"metrics": mapped_metrics, "metadata": {}}}

            # Add execution time if provided
            if execution_time is not None:
                measures_data["measures"]["metadata"]["execution_time"] = float(
                    execution_time
                )

            # Create connector without SSL for localhost
            connector = None
            backend_url = self.client.backend_config.backend_base_url
            if backend_url and (
                "localhost" in backend_url or "127.0.0.1" in backend_url
            ):
                connector = cast(Any, aiohttp).TCPConnector(ssl=False)

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/configuration-runs/{config_run_id}/measures"

                async with session.put(
                    url,
                    json=measures_data,
                    headers=headers,
                    timeout=cast(Any, aiohttp).ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.debug(
                            f"Updated configuration run {config_run_id} with {len(mapped_metrics)} measures"
                        )
                        return True
                    else:
                        error_msg = await response.text()
                        logger.debug(
                            f"Failed to update config run measures: {response.status} - {error_msg[:100]}"
                        )
                        return False

        except Exception as e:
            logger.debug(f"Error updating config run measures: {e}")
            return False

    async def update_experiment_run_status_on_completion(
        self, experiment_run_id: str, status: str
    ) -> None:
        """Update experiment run status when session is finalized.

        Args:
            experiment_run_id: Experiment run ID
            status: Status to set (COMPLETED, FAILED, etc.)
        """
        if not AIOHTTP_AVAILABLE:
            logger.debug("aiohttp not available, skipping experiment run status update")
            return

        try:
            # Map SDK status to backend status using the standard mapping
            backend_status = self.map_to_backend_status(status)

            # Valid experiment run statuses that the backend should accept
            # Note: PRUNED and CANCELLED are valid trial outcomes that the backend
            # needs to support. If backend shows these as "UNKNOWN", the backend
            # needs to be updated to recognize these statuses.
            valid_statuses = [
                "COMPLETED",
                "FAILED",
                "RUNNING",
                "NOT_STARTED",
                "ACTIVE",
                "CREATED",
                "PRUNED",
                "CANCELLED",
            ]
            if backend_status not in valid_statuses:
                logger.warning(
                    f"Unexpected status '{status}' (mapped to '{backend_status}'), "
                    f"using COMPLETED as default"
                )
                backend_status = "COMPLETED"

            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            # Create connector without SSL for localhost
            connector = None
            backend_url = self.client.backend_config.backend_base_url
            if backend_url and (
                "localhost" in backend_url or "127.0.0.1" in backend_url
            ):
                connector = cast(Any, aiohttp).TCPConnector(ssl=False)

            async with cast(Any, aiohttp).ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/experiment-runs/runs/{experiment_run_id}"

                async with session.put(
                    url,
                    json={"status": backend_status},
                    headers=headers,
                    timeout=cast(Any, aiohttp).ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 204]:
                        logger.info(
                            f"✅ Updated experiment run {experiment_run_id} to status {backend_status}"
                        )
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Failed to update experiment run status (HTTP {response.status}): {error_text}"
                        )
        except Exception as e:
            logger.warning(f"Error updating experiment run status: {e}")

    # Cloud API Methods

    async def create_cloud_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Create cloud session for optimization.

        Mock implementation for now - would connect to actual cloud API.
        """
        logger.debug("Creating cloud session (mock implementation)")

        # Mock response
        return SessionCreationResponse(
            session_id=f"cloud_session_{int(time.time())}",
            status=OptimizationSessionStatus.CREATED,
            optimization_strategy=request.optimization_strategy or {},
            metadata={
                "created_at": time.time(),
                "billing_tier": request.billing_tier,
            },
        )

    async def get_cloud_trial_suggestion(
        self, request: NextTrialRequest
    ) -> NextTrialResponse:
        """Get next trial suggestion from cloud optimizer.

        Mock implementation for now - would connect to actual cloud API.
        """
        logger.debug("Getting cloud trial suggestion (mock implementation)")

        # Mock response - no suggestion means optimization is complete
        suggestion = TrialSuggestion(
            trial_id=f"trial_{int(time.time())}",
            session_id=request.session_id,
            trial_number=1,
            config={"param": 0},
            dataset_subset=DatasetSubsetIndices(
                indices=[0],
                selection_strategy="mock",
                confidence_level=0.9,
                estimated_representativeness=0.9,
            ),
            exploration_type="exploration",
        )

        return NextTrialResponse(
            suggestion=suggestion,
            should_continue=True,
            session_status=OptimizationSessionStatus.ACTIVE,
            metadata={
                "mock": True,
                "session_id": request.session_id,
            },
        )

    async def submit_cloud_trial_results(
        self, submission: TrialResultSubmission
    ) -> None:
        """Submit trial results to cloud service.

        Mock implementation for now - would connect to actual cloud API.
        """
        logger.debug(f"Submitting cloud trial results for {submission.trial_id} (mock)")
        # Mock implementation - just log
        pass

    async def submit_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Submit agent for cloud optimization.

        Mock implementation for now - would connect to actual cloud API.
        """
        agent_name = getattr(request.agent_spec, "name", "unknown")
        logger.debug(f"Submitting agent optimization for {agent_name} (mock)")

        # Mock response
        return AgentOptimizationResponse(
            session_id=f"agent_session_{int(time.time())}",
            optimization_id=f"opt_{int(time.time())}",
            status="started",
            estimated_duration=300.0,
            next_steps=["await_results"],
        )

    async def execute_cloud_agent(
        self, request: AgentExecutionRequest
    ) -> AgentExecutionResponse:
        """Execute agent in cloud.

        Mock implementation for now - would connect to actual cloud API.
        """
        agent_name = getattr(request.agent_spec, "name", "unknown")
        logger.debug(f"Executing cloud agent {agent_name} (mock)")

        # Mock response
        return AgentExecutionResponse(
            output="Mock agent response",
            duration=1.5,
            tokens_used=50,
            cost=0.001,
            metadata={
                "status": "COMPLETED",
                "session_id": getattr(request.agent_spec, "id", "mock_session"),
            },
        )

    # Deprecated methods that should not be used

    async def create_backend_experiment_tracking(
        self, session_request: SessionCreationRequest
    ) -> tuple[str, str]:
        """DEPRECATED: This method should not be used anymore.

        The SDK should only use session endpoints.
        This method is kept for backward compatibility but should be removed.
        """
        logger.error(
            "⚠️ DEPRECATED: create_backend_experiment_tracking called. SDK should only use session endpoints!"
        )
        raise NotImplementedError(
            "SDK must use session endpoints only. Direct experiment creation is not allowed."
        )

    async def create_backend_agent_experiment(
        self,
        agent_spec: Any,
        dataset: Any,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
    ) -> tuple[str, str]:
        """DEPRECATED: This method should not be used anymore.

        The SDK should only use session endpoints.
        Backend handles agent and experiment creation internally.
        """
        logger.error(
            "⚠️ DEPRECATED: create_backend_agent_experiment called. SDK should only use session endpoints!"
        )
        raise NotImplementedError(
            "SDK must use session endpoints only. Direct agent/experiment creation is not allowed."
        )
