"""Enhanced Traigent Cloud Client with Backend Integration (Refactored).

This is the refactored version of backend_client.py using modular sub-components
for better maintainability and adherence to software engineering principles.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Security FUNC-CLOUD-HYBRID FUNC-AGENTS FUNC-SECURITY REQ-CLOUD-009 REQ-AGNT-013 REQ-SEC-010

import asyncio
import os
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

# Import and re-export BackendClientConfig for backward compatibility
from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.backend_bridges import SDKBackendBridge, SessionExperimentMapping
from traigent.cloud.backend_bridges import bridge as _bridge
from traigent.cloud.backend_components import (
    BackendAuthManager,
    BackendClientConfig,
    BackendSessionManager,
    BackendTrialManager,
)
from traigent.cloud.client import CloudServiceError
from traigent.cloud.cloud_operations import CloudOperations
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentOptimizationRequest,
    AgentOptimizationResponse,
    AgentSpecification,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSession,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
    TrialSuggestion,
)

# Import refactored sub-modules
from traigent.cloud.privacy_operations import PrivacyOperations
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.subset_selection import SmartSubsetSelector
from traigent.cloud.trial_operations import TrialOperations
from traigent.config.backend_config import BackendConfig
from traigent.evaluators.base import Dataset
from traigent.security.session_manager import SessionManager as SecuritySessionManager
from traigent.utils.logging import get_logger

# Type alias for aiohttp session, falling back to Any at runtime if unavailable
if TYPE_CHECKING:
    from aiohttp import ClientSession as AioClientSession
else:
    AioClientSession = Any  # type: ignore[assignment]

# Import local storage for fallback when backend is unavailable
LOCAL_STORAGE_AVAILABLE = False
try:
    from traigent.storage.local_storage import (
        LocalStorageManager as LocalStorageBackendManager,
    )

    LOCAL_STORAGE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency for offline mode
    LOCAL_STORAGE_AVAILABLE = False

    class LocalStorageBackendManager:  # type: ignore[no-redef]
        """Fallback stub when local storage support is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "LocalStorageManager is unavailable. Install optional storage dependencies."
            ) from None

        def add_trial_result(
            self,
            session_id: str,
            config: dict[str, Any],
            score: float | None,
            metadata: dict[str, Any] | None = None,
            error: str | None = None,
        ) -> None:
            raise ImportError(
                "Local storage functionality requires optional dependencies."
            ) from None


logger = get_logger(__name__)


def _session_is_closed(session: Any) -> bool:
    """Return True when the aiohttp session reports being closed."""
    closed_flag = getattr(session, "closed", False)
    if isinstance(closed_flag, bool):
        return closed_flag
    return False


# Export public API
__all__ = [
    "BackendClientConfig",
    "BackendIntegratedClient",
]

# Backwards compatibility: expose shared bridge instance at this module scope so
# existing tests patching ``traigent.cloud.backend_client.bridge`` continue to
# work after the refactor.
bridge = _bridge


class BackendIntegratedClient:
    """Enhanced cloud client with Traigent Backend integration (Refactored).

    This client supports both execution models:
    - Model 1: Privacy-first optimization with client-side execution
    - Model 2: Cloud SaaS optimization with backend agent execution

    The refactored version delegates specific responsibilities to sub-modules:
    - validators: Data validation functions
    - privacy_operations: Privacy-first optimization operations
    - cloud_operations: Cloud SaaS optimization operations
    - session_operations: Session lifecycle management
    - trial_operations: Trial management and result submission
    - api_operations: Backend API integration methods
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        backend_config: BackendClientConfig | None = None,
        enable_fallback: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
        enable_rate_limiting: bool = True,
        rate_limit_calls: int = 100,
        rate_limit_period: float = 60.0,
        local_storage_path: str | None = None,
    ) -> None:
        """Initialize backend integrated client.

        Args:
            api_key: Traigent Cloud API key
            base_url: Cloud service base URL
            backend_config: Backend integration configuration
            enable_fallback: Fall back to local optimization if cloud fails
            max_retries: Maximum retry attempts for requests
            timeout: Request timeout in seconds
            enable_rate_limiting: Enable rate limiting
            rate_limit_calls: Maximum calls within rate limit period
            rate_limit_period: Rate limit period in seconds
        """
        if isinstance(api_key, BackendClientConfig) and backend_config is None:
            backend_config = api_key
            api_key = None
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, client will use fallback mode only")

        # Initialize API operations helper
        self._api_ops = ApiOperations(self)

        self.backend_config = backend_config or BackendClientConfig()

        effective_base_url = base_url or self.backend_config.backend_base_url
        if effective_base_url is None:
            effective_base_url = BackendConfig.get_backend_url()

        # Validate and sanitize URLs
        self.base_url = self._api_ops.validate_and_sanitize_url(effective_base_url)

        backend_base_url = self.backend_config.backend_base_url or self.base_url
        self.backend_config.backend_base_url = self._api_ops.validate_and_sanitize_url(
            backend_base_url
        )

        api_base_candidate = (
            self.backend_config.api_base_url or BackendConfig.get_backend_api_url()
        )
        self.api_base_url = self._api_ops.validate_and_sanitize_url(api_base_candidate)
        self.backend_config.api_base_url = self.api_base_url

        self.enable_fallback = enable_fallback
        self.enable_rate_limiting = enable_rate_limiting
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self._rate_limit_timestamps: list[float] = []

        # Initialize extracted components
        effective_rate_limit = rate_limit_calls if enable_rate_limiting else sys.maxsize
        self.auth_manager: BackendAuthManager = BackendAuthManager(
            api_key,
            effective_rate_limit,
            rate_limit_period,
        )
        self.auth = self.auth_manager.auth
        self.session_manager = BackendSessionManager(
            self.auth_manager, self.backend_config
        )
        self.trial_manager = BackendTrialManager(self.auth_manager, self.backend_config)
        self.max_retries = min(max_retries, 10)  # Cap max retries for safety
        self.timeout = min(timeout, 300.0)  # Cap timeout at 5 minutes

        # Initialize session
        self._session: AioClientSession | None = None
        self._session_lock: asyncio.Lock = asyncio.Lock()

        # Initialize remaining components
        self.subset_selector = SmartSubsetSelector()

        # Request tracking for security
        self._request_nonces: set[str] = set()
        self._max_nonces = 10000  # Prevent unbounded growth

        # Initialize local storage for fallback when backend is unavailable
        self._local_storage_root = self._resolve_local_storage_path(local_storage_path)
        self.local_storage: LocalStorageBackendManager | None = None
        if LOCAL_STORAGE_AVAILABLE and enable_fallback:
            try:
                self.local_storage = LocalStorageBackendManager(
                    str(self._local_storage_root)
                )
                logger.info(
                    "Local storage fallback initialized at %s",
                    self._local_storage_root,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize local storage fallback: {e}")

        # Session management
        self._session = None
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._active_sessions_lock = Lock()

        # Memory bounds to prevent unbounded growth
        self._max_active_sessions = 100  # Maximum concurrent sessions

        # Security session manager for token issuance and validation
        redis_url = os.getenv("TRAIGENT_SESSION_REDIS_URL")
        session_ttl = int(os.getenv("TRAIGENT_SESSION_TTL", "3600"))
        max_sessions_per_user = int(os.getenv("TRAIGENT_MAX_SESSIONS_PER_USER", "10"))
        self._security_sessions = SecuritySessionManager(
            redis_url=redis_url,
            session_ttl=session_ttl,
            max_sessions_per_user=max_sessions_per_user,
        )
        self._session_credentials: dict[str, str] = {}

        # Initialize operation handlers (delegating to sub-modules)
        self._privacy_ops = PrivacyOperations(self)
        self._cloud_ops = CloudOperations(self)
        self._session_ops = SessionOperations(self)
        self._trial_ops = TrialOperations(self)

    @property
    def session_bridge(self) -> "SDKBackendBridge":
        """Return the shared session bridge (exposed for test patching)."""

        return bridge

    def _resolve_local_storage_path(self, override: str | None) -> Path:
        """Determine the root path for local fallback storage."""

        candidates: list[str | None] = [
            override,
            os.getenv("TRAIGENT_RESULTS_FOLDER"),
        ]

        for candidate in candidates:
            if candidate:
                return Path(candidate).expanduser().resolve()

        return (Path.home() / ".traigent").expanduser().resolve()

    # === Rate Limiting & Security ===

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting.

        Raises:
            CloudServiceError: If rate limit is exceeded
        """
        if not self.enable_rate_limiting:
            return

        current_time = time.time()

        # Clean old timestamps
        self._rate_limit_timestamps = [
            t
            for t in self._rate_limit_timestamps
            if current_time - t < self.rate_limit_period
        ]

        # Check if rate limit exceeded
        if len(self._rate_limit_timestamps) >= self.rate_limit_calls:
            wait_time = self.rate_limit_period - (
                current_time - self._rate_limit_timestamps[0]
            )
            raise CloudServiceError(
                f"Rate limit exceeded. Please wait {wait_time:.1f} seconds before making another request."
            )

        # Add current timestamp
        self._rate_limit_timestamps.append(current_time)

    def _generate_request_nonce(self) -> str:
        """Generate a unique nonce for request tracking.

        Returns:
            Unique nonce string
        """
        # Clean old nonces if too many
        if len(self._request_nonces) >= self._max_nonces:
            # Keep only recent half
            to_remove = len(self._request_nonces) // 2
            for _ in range(to_remove):
                self._request_nonces.pop()

        nonce = secrets.token_urlsafe(32)
        self._request_nonces.add(nonce)
        return nonce

    def _register_security_session(
        self,
        session_id: str,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> str:
        """Register session with security manager and return issued token."""
        try:
            _, session_token = self._security_sessions.create_session(
                user_id=user_id or "anonymous",
                metadata=metadata,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(
                "Security session registration failed for %s: %s", session_id, exc
            )
            session_token = secrets.token_urlsafe(32)

        self._session_credentials[session_id] = session_token
        return session_token

    def _revoke_security_session(self, session_id: str) -> None:
        """Revoke a session from the security manager and clear cached token."""
        try:
            self._security_sessions.revoke_session(session_id)
        except Exception as exc:
            logger.debug("Security session revoke failed for %s: %s", session_id, exc)
        finally:
            self._session_credentials.pop(session_id, None)

    def get_session_token(self, session_id: str) -> str | None:
        """Return the cached session token for the given session, if available."""
        return self._session_credentials.get(session_id)

    def validate_session_token(self, session_id: str, token: str) -> bool:
        """Validate a session token using the security manager."""
        try:
            return (
                self._security_sessions.validate_session(session_id, token) is not None
            )
        except Exception as exc:
            logger.debug(
                "Security session validation failed for %s: %s", session_id, exc
            )
            return False

    # === Context Manager Support ===

    async def __aenter__(self) -> "BackendIntegratedClient":
        """Async context manager entry."""
        if AIOHTTP_AVAILABLE:
            # Try to get headers, but don't fail if authentication is not available
            try:
                headers = await self.auth_manager.auth.get_headers()
            except Exception as e:
                # In Edge Analytics mode or when no API key is available, continue without auth headers
                logger.debug(f"No authentication available for context manager: {e}")
                headers = {}

            # Add security headers
            headers.update(
                {
                    "X-Request-ID": self._generate_request_nonce(),
                    "X-Client-Version": "2.0.0",
                }
            )
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self._reset_http_session("context-exit")

    async def _ensure_session(self) -> AioClientSession:
        """Ensure session exists with current auth headers."""

        existing = self._session
        if existing is not None and not _session_is_closed(existing):
            return cast(AioClientSession, existing)

        async with self._session_lock:
            existing = self._session
            if existing is not None and not _session_is_closed(existing):
                return cast(AioClientSession, existing)

            if not AIOHTTP_AVAILABLE:
                raise CloudServiceError(
                    "Client session not initialized and aiohttp not available"
                )

            # Try to get headers, but don't fail if authentication is not available
            try:
                headers = await self.auth_manager.auth.get_headers()
            except Exception as exc:
                logger.warning("Could not get auth headers: %s", exc)
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Traigent-SDK/1.0",
                }
            else:
                headers.setdefault("Content-Type", "application/json")

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers,
            )

            return cast(AioClientSession, self._session)

    async def _reset_http_session(self, reason: str | None = None) -> None:
        """Close and discard the shared aiohttp session."""

        if not self._session:
            return

        session = self._session
        self._session = None
        try:
            if _session_is_closed(session):
                return
            await session.close()
            # On shutdown/context-exit paths, give the event loop time to run
            # the connector's cleanup callbacks (especially for HTTPS/SSL).
            # Without this, asyncio.run() may tear down the loop before the
            # underlying TCP transport is fully closed, producing the
            # "Unclosed client session" ResourceWarning.
            # Skip the delay on retry/error paths to avoid adding 250ms
            # on every transient reset.
            if reason in ("shutdown", "context-exit"):
                await asyncio.sleep(0.25)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.debug(
                "Error closing backend HTTP session%s: %s",
                f" ({reason})" if reason else "",
                exc,
            )

    async def close(self) -> None:
        """Close any active HTTP session to avoid resource leaks."""

        await self._reset_http_session("shutdown")

    # === Delegated Operations ===
    # The following methods delegate to the refactored sub-modules

    # Privacy Operations
    async def create_privacy_optimization_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        dataset_metadata: dict[str, Any],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Create privacy-first optimization session.
        Delegates to privacy_operations module."""
        return await self._privacy_ops.create_privacy_optimization_session(
            function_name,
            configuration_space,
            objectives,
            dataset_metadata,
            max_trials,
            user_id,
        )

    async def _deprecated_create_privacy_optimization_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        dataset_metadata: dict[str, Any],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Compatibility shim for legacy tests expecting the deprecated method name."""

        logger.warning(
            "_deprecated_create_privacy_optimization_session is deprecated; use create_privacy_optimization_session"
        )
        return await self.create_privacy_optimization_session(
            function_name,
            configuration_space,
            objectives,
            dataset_metadata,
            max_trials,
            user_id,
        )

    async def get_next_privacy_trial(
        self,
        session_id: str,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> TrialSuggestion | None:
        """Get next trial suggestion for privacy-first optimization.
        Delegates to privacy_operations module."""
        return await self._privacy_ops.get_next_privacy_trial(
            session_id, previous_results
        )

    async def submit_privacy_trial_results(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        duration: float,
        error_message: str | None = None,
    ) -> bool:
        """Submit trial results for privacy-first optimization.
        Delegates to privacy_operations module."""
        return await self._privacy_ops.submit_privacy_trial_results(
            session_id, trial_id, config, metrics, duration, error_message
        )

    # Cloud Operations
    async def start_agent_optimization(
        self,
        agent_spec: AgentSpecification,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> AgentOptimizationResponse:
        """Start cloud SaaS optimization with full agent execution.
        Delegates to cloud_operations module."""
        return await self._cloud_ops.start_agent_optimization(
            agent_spec, dataset, configuration_space, objectives, max_trials, user_id
        )

    async def execute_agent(
        self,
        agent_spec: AgentSpecification,
        input_data: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
    ) -> AgentExecutionResponse:
        """Execute agent with specified configuration.
        Delegates to cloud_operations module."""
        return await self._cloud_ops.execute_agent(
            agent_spec, input_data, config_overrides
        )

    # Session Operations
    def create_session(
        self,
        function_name: str,
        search_space: dict[str, Any],
        optimization_goal: str = "maximize",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous wrapper for creating a session.
        Delegates to session_operations module."""
        return self._session_ops.create_session(
            function_name, search_space, optimization_goal, metadata
        )

    async def create_hybrid_session(
        self,
        problem_statement: str,
        search_space: dict[str, Any],
        optimization_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        """Create a hybrid execution session.
        Delegates to session_operations module."""
        return await self._session_ops.create_hybrid_session(
            problem_statement, search_space, optimization_config, metadata
        )

    async def get_hybrid_session_status(self, session_id: str) -> dict[str, Any]:
        """Get status of a hybrid session.
        Delegates to session_operations module."""
        return await self._session_ops.get_hybrid_session_status(session_id)

    async def finalize_hybrid_session(self, session_id: str) -> dict[str, Any]:
        """Finalize hybrid session and get results.
        Delegates to session_operations module."""
        return await self._session_ops.finalize_hybrid_session(session_id)

    async def finalize_session(
        self, session_id: str, include_full_history: bool = False
    ) -> OptimizationFinalizationResponse:
        """Finalize optimization session and get results.
        Delegates to session_operations module."""
        return await self._session_ops.finalize_session(
            session_id, include_full_history
        )

    async def delete_session(self, session_id: str, cascade: bool = True) -> bool:
        """Delete optimization session and optionally related data.
        Delegates to session_operations module."""
        return await self._session_ops.delete_session(session_id, cascade)

    def finalize_session_sync(
        self, session_id: str, include_full_history: bool = False
    ) -> OptimizationFinalizationResponse | None:
        """Synchronous wrapper for finalize_session.
        Delegates to session_operations module."""
        return self._session_ops.finalize_session_sync(session_id, include_full_history)

    def delete_session_sync(self, session_id: str, cascade: bool = True) -> bool:
        """Synchronous wrapper for delete_session.
        Delegates to session_operations module."""
        return self._session_ops.delete_session_sync(session_id, cascade)

    def get_active_sessions(self) -> dict[str, OptimizationSession]:
        """Get all active optimization sessions.
        Delegates to session_operations module."""
        return self._session_ops.get_active_sessions()

    def get_session_mapping(self, session_id: str) -> SessionExperimentMapping | None:
        """Get session to experiment mapping.
        Delegates to session_operations module."""
        return self._session_ops.get_session_mapping(session_id)

    # Trial Operations
    async def register_trial_start(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
    ) -> bool:
        """Register a trial start with the backend.
        Delegates to trial_operations module."""
        return await self._trial_ops.register_trial_start(session_id, trial_id, config)

    def register_trial_start_sync(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
    ) -> bool:
        """Synchronous wrapper for register_trial_start.
        Delegates to trial_operations module."""
        return self._trial_ops.register_trial_start_sync(session_id, trial_id, config)

    def _generate_trial_id(
        self,
        session_id: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate deterministic trial IDs compatible with legacy behaviour."""

        if metadata and isinstance(metadata.get("trial_id"), str):
            return cast(str, metadata["trial_id"])

        from traigent.utils.hashing import generate_trial_hash

        return generate_trial_hash(session_id, config, "")

    async def _submit_trial_result_via_session(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        status: str,
        error_message: str | None = None,
        execution_mode: str | None = None,
    ) -> bool:
        """Submit trial results via the Traigent session endpoint.
        Delegates to trial_operations module."""
        return await self._trial_ops.submit_trial_result_via_session(
            session_id, trial_id, config, metrics, status, error_message, execution_mode
        )

    async def _submit_summary_stats(
        self,
        session_id: str,
        trial_id: str,
        config: dict[str, Any],
        summary_stats: dict[str, Any],
        status: str = "completed",
    ) -> bool:
        """Submit summary statistics for privacy-preserving mode.
        Delegates to trial_operations module."""
        return await self._trial_ops.submit_summary_stats(
            session_id, trial_id, config, summary_stats, status
        )

    async def update_trial_weighted_scores(
        self,
        trial_id: str,
        weighted_score: float,
        normalization_info: dict[str, Any] | None = None,
        objective_weights: dict[str, float] | None = None,
    ) -> bool:
        """Update configuration run with weighted multi-objective scores.
        Delegates to trial_operations module."""
        return await self._trial_ops.update_trial_weighted_scores(
            trial_id, weighted_score, normalization_info, objective_weights
        )

    def submit_result(
        self,
        session_id: str,
        config: dict[str, Any],
        score: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Compatibility shim for legacy synchronous result submission."""

        logger.debug("submit_result called for session=%s score=%s", session_id, score)

        session = self._active_sessions.get(session_id)
        if session:
            session.completed_trials += 1
            session.updated_at = datetime.now(UTC)

        if self.enable_fallback and self.local_storage:
            try:
                self.local_storage.add_trial_result(
                    session_id=session_id,
                    config=config,
                    score=score,
                    metadata=metadata or {},
                )
            except Exception as exc:
                logger.debug("Local storage trial result fallback failed: %s", exc)

    # API Operations
    def _map_to_backend_status(self, status: str) -> str:
        """Map Traigent status values to backend-expected values.
        Delegates to api_operations module."""
        return self._api_ops.map_to_backend_status(status)

    def _normalize_execution_mode(self, execution_mode: str | None) -> str:
        """Translate SDK execution modes to backend-supported values."""
        if not execution_mode:
            return "standard"

        normalized = execution_mode.strip().lower()
        mode_aliases = {
            "edge_analytics": "local",
            "edge": "local",
            "local": "local",
            "privacy": "privacy",
            "private": "privacy",
            "hybrid": "local",
            "standard": "standard",
            "cloud": "cloud",
            "saas": "cloud",
        }
        return mode_aliases.get(normalized, "standard")

    def _sanitize_error_message(self, error_message: str | None) -> str | None:
        """Sanitize error message for transmission.
        Delegates to api_operations module."""
        return self._api_ops.sanitize_error_message(error_message)

    async def _create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ) -> tuple[str, str, str]:
        """Create a Traigent optimization session using the new session endpoints.
        Delegates to api_operations module."""
        return await self._api_ops.create_traigent_session_via_api(session_request)

    async def _update_config_run_status(self, config_run_id: str, status: str) -> bool:
        """Update configuration run status in the backend.
        Delegates to api_operations module."""
        return await self._api_ops.update_config_run_status(config_run_id, status)

    async def _update_config_run_measures(
        self,
        config_run_id: str,
        metrics: dict[str, float],
        execution_time: float | None = None,
    ) -> bool:
        """Update configuration run measures in the backend.
        Delegates to api_operations module."""
        return await self._api_ops.update_config_run_measures(
            config_run_id, metrics, execution_time
        )

    async def _update_experiment_run_status_on_completion(
        self, experiment_run_id: str, status: str
    ) -> None:
        """Update experiment run status when session is finalized.
        Delegates to api_operations module."""
        await self._api_ops.update_experiment_run_status_on_completion(
            experiment_run_id, status
        )

    # Cloud API methods (mock implementations in api_operations)
    async def _create_cloud_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Create cloud session for optimization.
        Delegates to api_operations module."""
        return await self._api_ops.create_cloud_session(request)

    async def _get_cloud_trial_suggestion(
        self, request: NextTrialRequest
    ) -> NextTrialResponse:
        """Get next trial suggestion from cloud optimizer.
        Delegates to api_operations module."""
        return await self._api_ops.get_cloud_trial_suggestion(request)

    async def _submit_cloud_trial_results(
        self, submission: TrialResultSubmission
    ) -> None:
        """Submit trial results to cloud service.
        Delegates to api_operations module."""
        await self._api_ops.submit_cloud_trial_results(submission)

    async def _submit_agent_optimization(
        self, request: AgentOptimizationRequest
    ) -> AgentOptimizationResponse:
        """Submit agent for cloud optimization.
        Delegates to api_operations module."""
        return await self._api_ops.submit_agent_optimization(request)

    async def _execute_cloud_agent(
        self, request: AgentExecutionRequest
    ) -> AgentExecutionResponse:
        """Execute agent in cloud.
        Delegates to api_operations module."""
        return await self._api_ops.execute_cloud_agent(request)

    # Deprecated methods (delegated to api_operations for proper error handling)
    async def _create_backend_experiment_tracking(
        self, session_request: SessionCreationRequest
    ) -> tuple[str, str]:
        """DEPRECATED: Use session endpoints only."""
        return await self._api_ops.create_backend_experiment_tracking(session_request)

    async def _create_backend_experiment_via_api(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Historical method retained solely for backward compatibility tests."""

        logger.error(
            "⚠️ DEPRECATED: _create_backend_experiment_via_api called. SDK should only use session endpoints!"
        )
        raise NotImplementedError(
            "SDK must use session endpoints only. Direct experiment creation is not allowed."
        )

    async def _create_backend_experiment_run_via_api(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Historical method retained solely for backward compatibility tests."""

        logger.error(
            "⚠️ DEPRECATED: _create_backend_experiment_run_via_api called. SDK should only use session endpoints!"
        )
        raise NotImplementedError(
            "SDK must use session endpoints only. Direct experiment run creation is not allowed."
        )

    async def _create_backend_config_run(self, *args: Any, **kwargs: Any) -> None:
        """Historical method retained solely for backward compatibility tests."""

        logger.error(
            "⚠️ DEPRECATED: _create_backend_config_run called. SDK should only use session endpoints!"
        )
        raise NotImplementedError(
            "SDK must use session endpoints only. Direct configuration run creation is not allowed."
        )

    async def _update_backend_config_run_results(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Historical method retained solely for backward compatibility tests."""

        logger.error(
            "⚠️ DEPRECATED: _update_backend_config_run_results called. SDK should only use session endpoints!"
        )
        raise NotImplementedError(
            "SDK must use session endpoints only. Direct configuration run updates are not allowed."
        )

    async def _create_backend_agent_experiment(
        self,
        agent_spec: AgentSpecification,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
    ) -> tuple[str, str]:
        """DEPRECATED: Use session endpoints only."""
        return await self._api_ops.create_backend_agent_experiment(
            agent_spec, dataset, configuration_space, objectives, max_trials
        )


_backend_client: BackendIntegratedClient | None = None


def get_backend_client(**kwargs) -> BackendIntegratedClient:
    """Return a shared ``BackendIntegratedClient`` instance.

    The first invocation creates a client; subsequent calls reuse it unless
    tests reset the module-level `_backend_client`.
    """

    global _backend_client
    if _backend_client is None:
        _backend_client = BackendIntegratedClient(**kwargs)
    return _backend_client
