"""Session management operations for Traigent Cloud Client.

This module handles session creation, lifecycle management, and hybrid mode
operations for both privacy-preserving and cloud-based optimization.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import asyncio
import concurrent.futures
import inspect
import os
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn, cast

from traigent.cloud.auth import AuthenticationError
from traigent.cloud.backend_bridges import SessionExperimentMapping
from traigent.cloud.client import CloudServiceError
from traigent.cloud.models import (
    OptimizationFinalizationResponse,
    OptimizationSession,
    OptimizationSessionStatus,
    SessionCreationRequest,
)
from traigent.cloud.session_types import (
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.config.backend_config import BackendConfig
from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

# Optional aiohttp dependency handling
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

logger = get_logger(__name__)
_SESSION_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _get_session_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return a shared executor for running async work from sync contexts."""
    global _SESSION_EXECUTOR
    if _SESSION_EXECUTOR is None:
        _SESSION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return _SESSION_EXECUTOR


class SessionOperations:
    """Handles session management operations."""

    def __init__(self, client: "BackendIntegratedClient"):
        """Initialize session operations handler.

        Args:
            client: Parent BackendIntegratedClient instance
        """
        self.client = client

    @staticmethod
    def _validate_non_empty_string(value: str, field_name: str) -> None:
        """Validate that a string value is non-empty."""
        validate_or_raise(CoreValidators.validate_string_non_empty(value, field_name))

    @staticmethod
    def _validate_mapping(value: dict[str, Any], field_name: str) -> None:
        """Validate that a value is a dictionary."""
        validate_or_raise(CoreValidators.validate_dict(value, field_name))

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Validate that an integer is positive."""
        validate_or_raise(CoreValidators.validate_positive_int(value, field_name))

    async def _reset_client_session(self, reason: str) -> None:
        closer = getattr(self.client, "_reset_http_session", None)
        if not closer:
            return

        try:
            result = closer(reason=reason)
        except TypeError:
            result = closer(reason)

        if inspect.isawaitable(result):
            try:
                await result
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to reset HTTP session after %s: %s", reason, exc)

    def _describe_backend(self) -> str:
        """Return sanitized backend connection context for logging."""

        backend_url = (
            self.client.backend_config.backend_base_url
            or BackendConfig.get_backend_url()
        )
        env = os.getenv("TRAIGENT_ENV", "production")
        api_key_preview = None
        auth = getattr(self.client, "auth_manager", None)
        if auth and hasattr(auth, "auth") and hasattr(auth.auth, "get_api_key_preview"):
            try:
                api_key_preview = auth.auth.get_api_key_preview()
            except Exception:  # pragma: no cover - defensive
                api_key_preview = None

        api_key_display = api_key_preview or "not configured"
        return f"backend_url={backend_url}, env={env}, api_key={api_key_display}"

    @staticmethod
    def _summarize_actor(info: dict[str, Any] | None) -> str:
        """Return concise information about the current caller."""

        if not info:
            return "unknown principal"

        parts: list[str] = []
        if info.get("owner_user_id"):
            parts.append(f"user '{info['owner_user_id']}'")
        if info.get("owner_api_key_id"):
            parts.append(f"api-key '{info['owner_api_key_id']}'")
        if info.get("owner_api_key_preview"):
            parts.append(f"token {info['owner_api_key_preview']}")
        if info.get("credential_source"):
            parts.append(f"source={info['credential_source']}")

        return ", ".join(parts) if parts else "unknown principal"

    def _raise_ownership_error(
        self, session_id: str, action: str, status: int, error_text: str
    ) -> NoReturn:
        """Raise a CloudServiceError with ownership remediation guidance."""

        fingerprint: dict[str, Any] | None = None
        auth = getattr(self.client, "auth_manager", None)
        auth_core = getattr(auth, "auth", None) if auth else None
        get_fingerprint = getattr(auth_core, "get_owner_fingerprint", None)
        if callable(get_fingerprint):
            try:
                fingerprint = get_fingerprint() or {}
            except Exception:  # pragma: no cover - defensive
                fingerprint = {}

        summary = self._summarize_actor(fingerprint)
        excerpt = self._first_error_line(error_text)

        message = (
            f"{action} for session '{session_id}' failed: HTTP {status} Forbidden. "
            "Session ownership enforcement requires the caller to match the session owner or hold admin permissions. "
            f"Calling credentials: {summary}. Re-authenticate with the owning token or recreate the session with the current credentials."
        )

        if excerpt:
            message = f"{message} Backend response: {excerpt}"

        raise CloudServiceError(message)

    @staticmethod
    def _first_error_line(error_text: str | None) -> str:
        """Return a safe, truncated first line for backend error messages."""

        if not error_text:
            return ""

        lines = error_text.strip().splitlines()
        if not lines:
            return ""

        excerpt = lines[0].strip()
        if len(excerpt) > 200:
            return f"{excerpt[:197]}..."
        return excerpt

    def _create_local_fallback_session(
        self,
        function_name: str,
        search_space: dict[str, Any],
        optimization_goal: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Create a local-only session after backend failure.

        Returns:
            Session ID for local tracking.
        """
        if self.client.local_storage:
            try:
                optimization_config = {
                    "search_space": search_space,
                    "optimization_goal": optimization_goal,
                    "baseline_config": None,
                }
                fallback_metadata = dict(metadata or {})
                fallback_metadata.update(
                    {
                        "execution_mode": "local_fallback",
                        "backend_fallback": True,
                        "created_with_version": "traigent-local-fallback-1.0.0",
                    }
                )
                session_id = self.client.local_storage.create_session(
                    function_name=function_name,
                    optimization_config=optimization_config,
                    metadata=fallback_metadata,
                )
                self.client._register_security_session(
                    session_id,
                    (metadata or {}).get("user_id"),
                    {
                        "function_name": function_name,
                        "fallback": True,
                        "optimization_goal": optimization_goal,
                    },
                )
                logger.debug("Created local fallback session: %s", session_id)
                return session_id
            except Exception as storage_e:
                logger.debug("Local storage fallback failed: %s", storage_e)

        fallback_id = f"local_session_{uuid.uuid4()}"
        self.client._register_security_session(
            fallback_id,
            (metadata or {}).get("user_id"),
            {
                "function_name": function_name,
                "fallback": True,
                "optimization_goal": optimization_goal,
                "storage": "ephemeral",
            },
        )
        logger.debug("Using ephemeral session ID: %s", fallback_id)
        return fallback_id

    def create_session(
        self,
        function_name: str,
        search_space: dict[str, Any],
        optimization_goal: str = "maximize",
        metadata: dict[str, Any] | None = None,
    ) -> SessionCreationResult:
        """Create a session with backend metadata submission.

        Returns a structured result indicating whether the backend session was
        created successfully or fell back to local storage.

        Args:
            function_name: Name of the function being optimized
            search_space: Configuration search space (metadata only)
            optimization_goal: Optimization goal (minimize/maximize)
            metadata: Additional metadata

        Returns:
            SessionCreationResult with session_id and backend status
        """
        self._validate_non_empty_string(function_name, "function_name")
        self._validate_mapping(search_space, "search_space")
        self._validate_non_empty_string(optimization_goal, "optimization_goal")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationException("metadata must be a dictionary if provided")

        async def _create_session_async() -> SessionCreationResult:
            # Preflight: check if API key exists before attempting any HTTP call
            has_key = self.client.auth_manager.has_api_key()
            if not has_key:
                logger.debug(
                    "No API key configured — skipping backend session creation"
                )
                fallback_id = self._create_local_fallback_session(
                    function_name, search_space, optimization_goal, metadata
                )
                return SessionCreationResult.fallback(
                    session_id=fallback_id,
                    reason=SessionCreationFailureReason.NO_API_KEY,
                    detail="No API key configured",
                )

            max_trials_from_metadata = metadata.get("max_trials") if metadata else None
            if max_trials_from_metadata is None:
                logger.debug("max_trials not found in metadata, using default 10")
                max_trials_from_metadata = 10
            else:
                try:
                    self._validate_positive_int(
                        int(max_trials_from_metadata), "metadata.max_trials"
                    )
                    max_trials_from_metadata = int(max_trials_from_metadata)
                except (ValueError, ValidationException) as exc:
                    raise ValidationException(
                        f"metadata.max_trials must be a positive integer: {exc}"
                    ) from exc
                logger.info(
                    f"Using max_trials from metadata: {max_trials_from_metadata}"
                )

            session_request = SessionCreationRequest(
                function_name=function_name,
                configuration_space=search_space,
                objectives=[optimization_goal],
                dataset_metadata={
                    "size": metadata.get("dataset_size", 0) if metadata else 0,
                    "privacy_mode": True,
                },
                max_trials=max_trials_from_metadata,
                optimization_strategy={"mode": "local_execution"},
                user_id=None,  # Privacy preserving
                billing_tier="privacy",
                metadata=metadata or {},
            )

            try:
                logger.info("Creating session via Traigent session endpoints")
                (
                    session_id,
                    experiment_id,
                    experiment_run_id,
                ) = await self.client._create_traigent_session_via_api(session_request)

                self.client.session_bridge.create_session_mapping(
                    session_id=session_id,
                    experiment_id=experiment_id,
                    experiment_run_id=experiment_run_id,
                    function_name=function_name,
                    configuration_space=search_space,
                    objectives=[optimization_goal],
                )

                self.client._register_security_session(
                    session_id,
                    None,
                    {
                        "function_name": function_name,
                        "objectives": [optimization_goal],
                        "experiment_id": experiment_id,
                        "experiment_run_id": experiment_run_id,
                    },
                )

                with self.client._active_sessions_lock:
                    if (
                        len(self.client._active_sessions)
                        >= self.client._max_active_sessions
                    ):
                        oldest_session_id = None
                        oldest_time = datetime.max.replace(tzinfo=UTC)

                        for sid, sess in list(self.client._active_sessions.items()):
                            if (
                                sess.status
                                in [
                                    OptimizationSessionStatus.COMPLETED,
                                    OptimizationSessionStatus.FAILED,
                                ]
                                and sess.created_at < oldest_time
                            ):
                                oldest_time = sess.created_at
                                oldest_session_id = sid

                        if oldest_session_id is None and self.client._active_sessions:
                            oldest_session_id = min(
                                self.client._active_sessions.keys(),
                                key=lambda s: self.client._active_sessions[
                                    s
                                ].created_at,
                            )

                        if oldest_session_id:
                            del self.client._active_sessions[oldest_session_id]
                            logger.debug(
                                "Removed session %s to stay within active session limit",
                                oldest_session_id,
                            )

                    self.client._active_sessions[session_id] = OptimizationSession(
                        session_id=session_id,
                        function_name=function_name,
                        configuration_space=search_space,
                        objectives=[optimization_goal],
                        max_trials=max_trials_from_metadata,
                        status=OptimizationSessionStatus.ACTIVE,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                        metadata={
                            "mode": "session_api",
                            "experiment_id": experiment_id,
                            "experiment_run_id": experiment_run_id,
                            **(metadata or {}),
                        },
                    )

                logger.debug(
                    "Created backend tracking: session=%s, experiment=%s, run=%s",
                    session_id,
                    experiment_id,
                    experiment_run_id,
                )
                return SessionCreationResult.connected(session_id=session_id)

            except ValidationException:
                raise  # User input errors must propagate

            except AuthenticationError as e:
                await self._reset_client_session("create_session auth_failure")
                logger.debug("Backend auth failed for '%s': %s", function_name, e)
                fallback_id = self._create_local_fallback_session(
                    function_name, search_space, optimization_goal, metadata
                )
                return SessionCreationResult.fallback(
                    session_id=fallback_id,
                    reason=SessionCreationFailureReason.AUTH,
                    detail=str(e)[:200],
                )

            except (TimeoutError, CloudServiceError, OSError) as e:
                await self._reset_client_session("create_session fallback")
                logger.debug("Backend unavailable for '%s': %s", function_name, e)
                fallback_id = self._create_local_fallback_session(
                    function_name, search_space, optimization_goal, metadata
                )
                detail = str(e).split("\n", 1)[0][:200]
                return SessionCreationResult.fallback(
                    session_id=fallback_id,
                    reason=SessionCreationFailureReason.SESSION_FAILED,
                    detail=detail,
                )

        # Run async method in sync context
        try:
            try:
                asyncio.get_running_loop()

                def _run_in_new_loop() -> SessionCreationResult:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(_create_session_async())
                    finally:
                        new_loop.close()

                executor = _get_session_executor()
                future = executor.submit(_run_in_new_loop)
                return future.result(timeout=60)

            except RuntimeError:
                return asyncio.run(_create_session_async())

        except ValidationException:
            raise  # User input errors must propagate

        except AuthenticationError as exc:
            logger.debug(
                "Backend auth failed for '%s': %s",
                function_name,
                exc,
            )
            fallback_id = self._create_local_fallback_session(
                function_name, search_space, optimization_goal, metadata
            )
            return SessionCreationResult.fallback(
                session_id=fallback_id,
                reason=SessionCreationFailureReason.AUTH,
                detail=str(exc)[:200],
            )

        except (TimeoutError, CloudServiceError, OSError) as exc:
            logger.debug(
                "Error in create_session for function '%s': %s",
                function_name,
                exc,
            )
            fallback_id = self._create_local_fallback_session(
                function_name, search_space, optimization_goal, metadata
            )
            detail = str(exc).split("\n", 1)[0][:200]
            return SessionCreationResult.fallback(
                session_id=fallback_id,
                reason=SessionCreationFailureReason.SESSION_FAILED,
                detail=detail,
            )

        except Exception as exc:
            # Last-resort fallback for unexpected errors (e.g. from
            # session_bridge or event loop machinery).  Ensures SDK
            # never crashes during session creation.
            logger.debug(
                "Unexpected error in create_session for function '%s': %s",
                function_name,
                exc,
            )
            fallback_id = self._create_local_fallback_session(
                function_name, search_space, optimization_goal, metadata
            )
            return SessionCreationResult.fallback(
                session_id=fallback_id,
                reason=SessionCreationFailureReason.SESSION_FAILED,
                detail=str(exc)[:200],
            )

    async def create_hybrid_session(
        self,
        problem_statement: str,
        search_space: dict[str, Any],
        optimization_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        """Create a hybrid execution session.

        In hybrid mode, the client executes locally but submits metrics
        directly to the optimizer for orchestration.

        Args:
            problem_statement: Problem description
            search_space: Configuration search space
            optimization_config: Optimization settings
            metadata: Additional metadata

        Returns:
            Tuple of (session_id, token, optimizer_endpoint)
        """
        self._validate_non_empty_string(problem_statement, "problem_statement")
        self._validate_mapping(search_space, "search_space")
        self._validate_mapping(optimization_config, "optimization_config")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationException("metadata must be a dictionary if provided")
        if not AIOHTTP_AVAILABLE:
            raise CloudServiceError("aiohttp not available for hybrid sessions")

        logger.info("Creating hybrid execution session")

        session = await self.client._ensure_session()

        try:
            # Make request to backend for hybrid session
            api_base = (
                self.client.backend_config.api_base_url
                or BackendConfig.get_backend_api_url()
            )
            async with session.post(
                f"{api_base}/hybrid/sessions",
                json={
                    "problem_statement": problem_statement,
                    "search_space": search_space,
                    "optimization_config": optimization_config,
                    "metadata": metadata,
                },
                headers=await self.client.auth_manager.auth.get_headers(),
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    await self._reset_client_session(
                        "create_hybrid_session non-201 response"
                    )
                    raise CloudServiceError(
                        f"Failed to create hybrid session: {error_text}"
                    )

                data = await response.json()

                # Store session info
                session_id = data["session_id"]
                with self.client._active_sessions_lock:
                    self.client._active_sessions[session_id] = OptimizationSession(
                        session_id=session_id,
                        function_name=problem_statement,
                        configuration_space=search_space,
                        objectives=optimization_config.get("objectives", ["maximize"]),
                        max_trials=optimization_config.get("max_trials", 50),
                        status=OptimizationSessionStatus.ACTIVE,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                        metadata={
                            "mode": "hybrid",
                            "optimizer_endpoint": data.get("optimizer_endpoint", ""),
                            "token": data.get("token", ""),
                            **(metadata or {}),
                        },
                    )

                return session_id, data["token"], data["optimizer_endpoint"]

        except aiohttp.ClientError as e:
            await self._reset_client_session("create_hybrid_session network error")
            logger.error(f"Network error creating hybrid session: {e}")
            raise CloudServiceError(f"Network error: {e}") from None

    async def get_hybrid_session_status(self, session_id: str) -> dict[str, Any]:
        """Get status of a hybrid session.

        Args:
            session_id: Hybrid session ID

        Returns:
            Session status information
        """
        self._validate_non_empty_string(session_id, "session_id")
        if not AIOHTTP_AVAILABLE:
            raise CloudServiceError("aiohttp not available")

        session = await self.client._ensure_session()

        try:
            api_base = (
                self.client.backend_config.api_base_url
                or BackendConfig.get_backend_api_url()
            )
            async with session.get(
                f"{api_base}/hybrid/sessions/{session_id}/status",
                headers=await self.client.auth_manager.auth.get_headers(),
            ) as response:
                if response.status == 200:
                    return cast(dict[str, Any], await response.json())

                error_text = await response.text()
                if response.status == 403:
                    self._raise_ownership_error(
                        session_id,
                        "Fetching hybrid session status",
                        response.status,
                        error_text,
                    )

                await self._reset_client_session(
                    "get_hybrid_session_status unexpected response"
                )
                raise CloudServiceError(f"Failed to get session status: {error_text}")

        except aiohttp.ClientError as e:
            await self._reset_client_session("get_hybrid_session_status network error")
            logger.error(f"Network error getting session status: {e}")
            raise CloudServiceError(f"Network error: {e}") from None

    async def finalize_hybrid_session(self, session_id: str) -> dict[str, Any]:
        """Finalize hybrid session and get results.

        Args:
            session_id: Hybrid session ID

        Returns:
            Final session results
        """
        self._validate_non_empty_string(session_id, "session_id")
        if not AIOHTTP_AVAILABLE:
            raise CloudServiceError("aiohttp not available")

        session = await self.client._ensure_session()

        try:
            api_base = (
                self.client.backend_config.api_base_url
                or BackendConfig.get_backend_api_url()
            )
            async with session.post(
                f"{api_base}/hybrid/sessions/{session_id}/finalize",
                headers=await self.client.auth_manager.auth.get_headers(),
            ) as response:
                if response.status == 200:
                    # Remove from active sessions
                    with self.client._active_sessions_lock:
                        self.client._active_sessions.pop(session_id, None)
                    self.client._revoke_security_session(session_id)

                    return cast(dict[str, Any], await response.json())

                error_text = await response.text()
                if response.status == 403:
                    self._raise_ownership_error(
                        session_id,
                        "Finalizing hybrid session",
                        response.status,
                        error_text,
                    )

                await self._reset_client_session(
                    "finalize_hybrid_session unexpected response"
                )
                raise CloudServiceError(f"Failed to finalize session: {error_text}")

        except aiohttp.ClientError as e:
            await self._reset_client_session("finalize_hybrid_session network error")
            logger.error(f"Network error finalizing session: {e}")
            raise CloudServiceError(f"Network error: {e}") from None

    async def finalize_session(
        self, session_id: str, include_full_history: bool = False
    ) -> OptimizationFinalizationResponse:
        """Finalize optimization session and get results.

        This method is optional as the backend auto-finalizes sessions when all trials complete.
        Use this for early termination or to add custom finalization metadata.

        Args:
            session_id: Session ID to finalize
            include_full_history: Whether to include full trial history in response

        Returns:
            OptimizationFinalizationResponse with session summary
        """
        self._validate_non_empty_string(session_id, "session_id")
        logger.info(f"Finalizing optimization session {session_id}")

        # Get session mapping for experiment_run_id
        mapping = self.client.session_bridge.get_session_mapping(session_id)

        # Try to finalize via backend API endpoint (POST /sessions/{id}/finalize)
        finalized_via_api = False
        if mapping:
            try:
                finalized_via_api = await self._finalize_session_via_api(
                    session_id, mapping.experiment_run_id
                )
            except CloudServiceError:
                raise
            except Exception as e:
                logger.debug(f"Backend finalization API not available: {e}")

        # If backend finalization not available, the session may already be auto-finalized
        # Just log this - no need to fail
        if not finalized_via_api:
            logger.info(
                f"Session {session_id} finalization handled by backend auto-finalization or already complete"
            )

        # Mark session as completed and remove from active list
        with self.client._active_sessions_lock:
            session = self.client._active_sessions.pop(session_id, None)
        completed_trials = 0
        if session:
            session.status = OptimizationSessionStatus.COMPLETED
            session.updated_at = datetime.now(UTC)
            session.metadata["completed_at"] = (
                time.time()
            )  # Store in metadata since no dedicated attr
            completed_trials = getattr(session, "completed_trials", 0)

        # Revoke security session
        self.client._revoke_security_session(session_id)

        # Create finalization response (simplified version)
        response = OptimizationFinalizationResponse(
            session_id=session_id,
            best_config={},
            best_metrics={},
            total_trials=completed_trials,
            successful_trials=completed_trials,
            total_duration=0.0,
            cost_savings=0.0,
            convergence_history=[],
            full_history=[] if include_full_history else None,
            metadata={
                "finalized_at": time.time(),
                "experiment_run_id": mapping.experiment_run_id if mapping else None,
                "finalized_via_api": finalized_via_api,
            },
        )

        return response

    async def _finalize_session_via_api(
        self, session_id: str, experiment_run_id: str
    ) -> bool:
        """Call backend session finalization endpoint.

        Args:
            session_id: Session ID to finalize
            experiment_run_id: Associated experiment run ID

        Returns:
            True if successfully finalized via API, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.debug("aiohttp not available, skipping API finalization")
            return False

        try:
            # Prepare headers with API key
            headers = await self.client.auth_manager.augment_headers(
                {"Content-Type": "application/json"}
            )

            connector = None

            async with aiohttp.ClientSession(connector=connector) as session:
                api_base = (
                    self.client.backend_config.api_base_url
                    or BackendConfig.get_backend_api_url()
                )
                url = f"{api_base}/sessions/{session_id}/finalize"

                async with session.post(
                    url,
                    json={
                        "reason": "sdk_explicit_finalization",
                        "experiment_run_id": experiment_run_id,
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(
                            f"✅ Finalized session {session_id} via backend API"
                        )
                        return True
                    elif response.status == 403:
                        error_text = await response.text()
                        self._raise_ownership_error(
                            session_id,
                            "Finalizing session via backend API",
                            response.status,
                            error_text,
                        )
                        raise AssertionError(
                            "unreachable"
                        )  # _raise_ownership_error never returns
                    else:
                        error_text = await response.text()
                        logger.debug(
                            f"Backend finalization endpoint returned HTTP {response.status}: {error_text}"
                        )
                        return False
        except Exception as e:
            logger.debug(f"Error calling backend finalization API: {e}")
            return False

    async def delete_session(self, session_id: str, cascade: bool = True) -> bool:
        """Delete optimization session and optionally related data.

        Args:
            session_id: Session ID to delete
            cascade: If True, also delete related experiment data

        Returns:
            True if deletion successful
        """
        self._validate_non_empty_string(session_id, "session_id")
        logger.info(f"Deleting session {session_id} (cascade={cascade})")

        # Remove from active sessions
        with self.client._active_sessions_lock:
            deleted = self.client._active_sessions.pop(session_id, None) is not None

        # Revoke security session
        self.client._revoke_security_session(session_id)

        if cascade:
            # Remove session mapping
            mapping = self.client.session_bridge.get_session_mapping(session_id)
            if mapping:
                # Could optionally delete backend experiment data here
                # For now just remove the mapping
                self.client.session_bridge._session_mappings.pop(session_id, None)
                logger.debug(f"Removed session mapping for {session_id}")

        return deleted

    def finalize_session_sync(
        self, session_id: str, include_full_history: bool = False
    ) -> OptimizationFinalizationResponse | None:
        """Synchronous wrapper for finalize_session."""
        import concurrent.futures

        self._validate_non_empty_string(session_id, "session_id")

        def _run_in_new_loop() -> OptimizationFinalizationResponse | None:
            """Run the async operation in a fresh event loop on this thread."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self.finalize_session(session_id, include_full_history)
                )
            finally:
                new_loop.close()

        try:
            # Check if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, use new one
                loop = None

            if loop:
                # We're in an async context - run in a separate thread to avoid deadlock
                # Cannot use run_coroutine_threadsafe + future.result() from the same
                # thread as the event loop - that causes deadlock
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_new_loop)
                    return future.result(timeout=60)
            else:
                # No async context, run directly
                return _run_in_new_loop()
        except Exception as e:
            logger.warning(f"Failed to finalize session synchronously: {e}")
            return None

    def delete_session_sync(self, session_id: str, cascade: bool = True) -> bool:
        """Synchronous wrapper for delete_session.

        This method safely handles being called from both sync and async contexts
        by using a ThreadPoolExecutor when a running event loop is detected.
        """
        import concurrent.futures

        self._validate_non_empty_string(session_id, "session_id")

        def _run_in_new_loop() -> bool:
            """Run the async operation in a fresh event loop on this thread."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self.delete_session(session_id, cascade=cascade)
                )
            finally:
                new_loop.close()

        try:
            # Check if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, use new one
                loop = None

            if loop:
                # We're in an async context - run in a separate thread to avoid deadlock
                # Cannot use run_coroutine_threadsafe + future.result() from the same
                # thread as the event loop - that causes deadlock
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_new_loop)
                    return future.result(timeout=60)
            else:
                # No async context, run directly
                return _run_in_new_loop()
        except Exception as e:
            logger.warning(f"Failed to delete session synchronously: {e}")
            return False

    def get_active_sessions(self) -> dict[str, OptimizationSession]:
        """Get all active optimization sessions."""
        with self.client._active_sessions_lock:
            return self.client._active_sessions.copy()

    def get_session_mapping(self, session_id: str) -> SessionExperimentMapping | None:
        """Get session to experiment mapping."""
        return self.client.session_bridge.get_session_mapping(session_id)
