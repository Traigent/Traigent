"""Session management operations for Traigent Cloud Client.

This module handles session creation, lifecycle management, and hybrid mode
operations for both privacy-preserving and cloud-based optimization.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import asyncio
import concurrent.futures
import inspect
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn, cast

from traigent.cloud.auth import AuthenticationError
from traigent.cloud.backend_bridges import SessionExperimentMapping
from traigent.cloud.client import CloudServiceError, SessionContractError
from traigent.cloud.models import (
    OptimizationFinalizationResponse,
    OptimizationSession,
    OptimizationSessionStatus,
    SessionCreationRequest,
)
from traigent.cloud.session_types import (
    SessionCreationFailureDetail,
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.config.backend_config import BackendConfig
from traigent.utils.env_config import resolve_environment_label
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


def _get_session_creation_failure_detail(
    exc: Exception,
) -> SessionCreationFailureDetail | None:
    detail = getattr(exc, "session_creation_failure", None)
    if isinstance(detail, SessionCreationFailureDetail):
        return detail
    return None


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
        env = resolve_environment_label(default="production")
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

    def _persist_connected_session_locally(
        self,
        *,
        session_id: str,
        function_name: str,
        search_space: dict[str, Any],
        optimization_goal: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Mirror a connected backend session into local_storage (#1279).

        On the connected hybrid path the session is registered only in the
        in-memory ``_active_sessions``; ``local_storage`` never learns of it, so
        ``_persist_trial_locally`` -> ``add_trial_result`` raises "session not
        found" and the advertised "run locally, sync to backend" safety net does
        not exist there. Creating a local session keyed to the backend
        ``session_id`` gives every connected trial a durable local record, so a
        completed (paid-for) trial survives a mid-run remote-submit failure.

        Best-effort: a local-storage failure must never break the connected run,
        but it is logged at WARNING — the safety net silently not existing is the
        exact failure mode #1279 is about.
        """
        local_storage = getattr(self.client, "local_storage", None)
        if not local_storage:
            return
        try:
            optimization_config = {
                "search_space": search_space,
                "optimization_goal": optimization_goal,
                "baseline_config": None,
            }
            connected_metadata = dict(metadata or {})
            connected_metadata.update(
                {
                    "execution_mode": "connected",
                    "backend_session": True,
                    "created_with_version": "traigent-connected-mirror-1.0.0",
                }
            )
            local_storage.create_session(
                function_name=function_name,
                optimization_config=optimization_config,
                metadata=connected_metadata,
                session_id=session_id,
            )
            logger.debug("Created local mirror for connected session: %s", session_id)
        except Exception as storage_e:
            logger.warning(
                "Could not create local mirror for connected session %s; trials "
                "on this run have no local-persistence safety net: %s",
                session_id,
                storage_e,
            )

    def create_session(
        self,
        function_name: str,
        search_space: dict[str, Any],
        optimization_goal: str = "maximize",
        metadata: dict[str, Any] | None = None,
        objectives: list[Any] | None = None,
        promotion_policy: dict[str, Any] | None = None,
        tvl_governance: dict[str, Any] | None = None,
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

        # Phase 8 (review round 2): the local-availability fallback is for
        # NON-governed sessions only. A governed/strict session that cannot
        # be created on the backend fails LOUD — quietly degrading to a
        # local-only run would mask laundering refusals, contract
        # misconfiguration, and old-BE typed rejection as success.
        governed = bool(promotion_policy or tvl_governance)

        def _must_fail_loud(exc: Exception) -> bool:
            return governed or isinstance(exc, SessionContractError)

        async def _create_session_async() -> SessionCreationResult:
            # Preflight: check if API key exists before attempting any HTTP call
            has_key = self.client.auth_manager.has_api_key()
            if not has_key:
                if governed:
                    # ADJUDICATED (review round 3): no API key is an explicit
                    # local-only configuration, not a cloud failure — strict
                    # enforcement runs fully locally and no backend record
                    # exists that could launder strict mode. Stay local, but
                    # make the governance/no-cloud mismatch VISIBLE.
                    logger.warning(
                        "Governed session '%s' declared promotion_policy/"
                        "tvl_governance but no API key is configured — "
                        "running local-only (strict enforcement stays local; "
                        "no backend session or certified record is created)",
                        function_name,
                    )
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
                # Typed contract: objectives are METRIC names/objects. The
                # legacy [optimization_goal] placeholder survives only as the
                # fallback when the caller supplied none (legacy shape keeps
                # reading objectives[0] as its optimization_goal).
                objectives=list(objectives) if objectives else [optimization_goal],
                dataset_metadata={
                    "size": metadata.get("dataset_size", 0) if metadata else 0,
                    "privacy_mode": True,
                },
                max_trials=max_trials_from_metadata,
                promotion_policy=promotion_policy,
                tvl_governance=tvl_governance,
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
                                key=lambda s: (
                                    self.client._active_sessions[s].created_at
                                ),
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

                # #1279: also mirror the connected session into local_storage
                # keyed to the backend session_id, so _persist_trial_locally has
                # a durable local record on connected runs too. Without this the
                # only record of a completed (paid-for) trial is the breaker-gated
                # remote submit; if that fails mid-run the trial is lost silently.
                self._persist_connected_session_locally(
                    session_id=session_id,
                    function_name=function_name,
                    search_space=search_space,
                    optimization_goal=optimization_goal,
                    metadata=metadata,
                )
                return SessionCreationResult.connected(session_id=session_id)

            except ValidationException:
                raise  # User input errors must propagate

            except AuthenticationError as e:
                await self._reset_client_session("create_session auth_failure")
                if _must_fail_loud(e):
                    raise
                logger.debug("Backend auth failed for '%s': %s", function_name, e)
                failure_response = _get_session_creation_failure_detail(e)
                fallback_id = self._create_local_fallback_session(
                    function_name, search_space, optimization_goal, metadata
                )
                return SessionCreationResult.fallback(
                    session_id=fallback_id,
                    reason=SessionCreationFailureReason.AUTH,
                    detail=(
                        failure_response.one_line_summary()
                        if failure_response
                        else str(e)[:200]
                    ),
                    failure_response=failure_response,
                )

            except (TimeoutError, CloudServiceError, OSError) as e:
                await self._reset_client_session("create_session fallback")
                if _must_fail_loud(e):
                    raise
                logger.debug("Backend unavailable for '%s': %s", function_name, e)
                failure_response = _get_session_creation_failure_detail(e)
                fallback_id = self._create_local_fallback_session(
                    function_name, search_space, optimization_goal, metadata
                )
                detail = (
                    failure_response.one_line_summary()
                    if failure_response
                    else str(e).split("\n", 1)[0][:200]
                )
                return SessionCreationResult.fallback(
                    session_id=fallback_id,
                    reason=SessionCreationFailureReason.SESSION_FAILED,
                    detail=detail,
                    failure_response=failure_response,
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
            if _must_fail_loud(exc):
                raise
            logger.debug(
                "Backend auth failed for '%s': %s",
                function_name,
                exc,
            )
            failure_response = _get_session_creation_failure_detail(exc)
            fallback_id = self._create_local_fallback_session(
                function_name, search_space, optimization_goal, metadata
            )
            return SessionCreationResult.fallback(
                session_id=fallback_id,
                reason=SessionCreationFailureReason.AUTH,
                detail=(
                    failure_response.one_line_summary()
                    if failure_response
                    else str(exc)[:200]
                ),
                failure_response=failure_response,
            )

        except (TimeoutError, CloudServiceError, OSError) as exc:
            if _must_fail_loud(exc):
                raise
            logger.debug(
                "Error in create_session for function '%s': %s",
                function_name,
                exc,
            )
            failure_response = _get_session_creation_failure_detail(exc)
            fallback_id = self._create_local_fallback_session(
                function_name, search_space, optimization_goal, metadata
            )
            detail = (
                failure_response.one_line_summary()
                if failure_response
                else str(exc).split("\n", 1)[0][:200]
            )
            return SessionCreationResult.fallback(
                session_id=fallback_id,
                reason=SessionCreationFailureReason.SESSION_FAILED,
                detail=detail,
                failure_response=failure_response,
            )

        except Exception as exc:
            # Last-resort fallback for unexpected errors (e.g. from
            # session_bridge or event loop machinery).  Ensures SDK
            # never crashes during session creation — except for
            # governed/contract failures, which must stay loud.
            if _must_fail_loud(exc):
                raise
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
        self,
        session_id: str,
        include_full_history: bool = False,
        certified_selection: dict[str, Any] | None = None,
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

        # Get session mapping for experiment_run_id, or reconstruct it from the
        # active-session metadata if live tracking had succeeded but the in-memory
        # bridge entry was lost (e.g. after a partial restart or lost-reference
        # race). Finalization is a one-shot remote call — losing it silently on
        # a recoverable-metadata path would leave the session flagged as never
        # finalized.
        mapping = self.client.session_bridge.get_session_mapping(session_id)
        if mapping is None:
            with self.client._active_sessions_lock:
                active_session = self.client._active_sessions.get(session_id)
            if active_session is not None:
                metadata = dict(getattr(active_session, "metadata", {}) or {})
                experiment_id = metadata.get("experiment_id")
                experiment_run_id = metadata.get("experiment_run_id")
                if experiment_id and experiment_run_id:
                    mapping = self.client.session_bridge.create_session_mapping(
                        session_id=session_id,
                        experiment_id=str(experiment_id),
                        experiment_run_id=str(experiment_run_id),
                        function_name=str(
                            getattr(active_session, "function_name", "unknown_function")
                        ),
                        configuration_space=dict(
                            getattr(active_session, "configuration_space", {}) or {}
                        ),
                        objectives=list(
                            getattr(active_session, "objectives", []) or []
                        ),
                    )
                    logger.info(
                        "Recovered session mapping for %s from active session metadata",
                        session_id,
                    )

        # Try to finalize via backend API endpoint (POST /sessions/{id}/finalize).
        # backend_payload is dict on 2xx (possibly empty if the backend gave us
        # no summary), or None when the endpoint was unavailable/failed. See #890.
        backend_payload: dict[str, Any] | None = None
        if mapping:
            try:
                backend_payload = await self._finalize_session_via_api(
                    session_id,
                    mapping.experiment_run_id,
                    certified_selection=certified_selection,
                )
            except CloudServiceError:
                raise
            except Exception as e:
                logger.debug(f"Backend finalization API not available: {e}")

        finalized_via_api = backend_payload is not None

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

        # Build finalization response. When the backend returned a summary
        # payload, preserve its fields verbatim. Otherwise mark the summary
        # as unavailable so callers don't treat empty best_config/best_metrics
        # as an authoritative "empty optimization" result (per the tracked fix).
        backend_summary = backend_payload or {}

        backend_best_config = backend_summary.get("best_config")
        backend_best_metrics = backend_summary.get("best_metrics")
        backend_total_trials = backend_summary.get("total_trials")
        backend_successful_trials = backend_summary.get("successful_trials")
        backend_total_duration = backend_summary.get("total_duration")
        backend_cost_savings = backend_summary.get("cost_savings")
        backend_stop_reason = backend_summary.get("stop_reason")
        backend_convergence_history = backend_summary.get("convergence_history")
        backend_full_history = backend_summary.get("full_history")

        summary_fields: list[str] = []

        def _is_real_number(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        response_best_config = (
            backend_best_config if isinstance(backend_best_config, dict) else {}
        )
        if isinstance(backend_best_config, dict):
            summary_fields.append("best_config")

        response_best_metrics = (
            backend_best_metrics if isinstance(backend_best_metrics, dict) else {}
        )
        if isinstance(backend_best_metrics, dict):
            summary_fields.append("best_metrics")

        response_total_trials = (
            int(cast(int | float, backend_total_trials))
            if _is_real_number(backend_total_trials)
            else completed_trials
        )
        if _is_real_number(backend_total_trials):
            summary_fields.append("total_trials")

        response_successful_trials = (
            int(cast(int | float, backend_successful_trials))
            if _is_real_number(backend_successful_trials)
            else completed_trials
        )
        if _is_real_number(backend_successful_trials):
            summary_fields.append("successful_trials")

        response_total_duration = (
            float(cast(int | float, backend_total_duration))
            if _is_real_number(backend_total_duration)
            else 0.0
        )
        if _is_real_number(backend_total_duration):
            summary_fields.append("total_duration")

        response_cost_savings = (
            float(cast(int | float, backend_cost_savings))
            if _is_real_number(backend_cost_savings)
            else 0.0
        )
        if _is_real_number(backend_cost_savings):
            summary_fields.append("cost_savings")

        response_stop_reason = (
            str(backend_stop_reason) if isinstance(backend_stop_reason, str) else None
        )
        if isinstance(backend_stop_reason, str):
            summary_fields.append("stop_reason")

        response_convergence_history = (
            backend_convergence_history
            if isinstance(backend_convergence_history, list)
            else []
        )
        if isinstance(backend_convergence_history, list):
            summary_fields.append("convergence_history")

        response_full_history = (
            backend_full_history
            if include_full_history and isinstance(backend_full_history, list)
            else ([] if include_full_history else None)
        )
        if include_full_history and isinstance(backend_full_history, list):
            summary_fields.append("full_history")

        # summary_available=True only when at least one documented backend
        # summary field was type-valid and preserved in the SDK response. A
        # legacy or malformed payload must not mark a synthetic fallback value
        # as backend-authoritative. Callers that require specific fields should
        # inspect metadata["summary_fields"].
        summary_available = bool(summary_fields)

        response = OptimizationFinalizationResponse(
            session_id=session_id,
            best_config=response_best_config,
            best_metrics=response_best_metrics,
            total_trials=response_total_trials,
            successful_trials=response_successful_trials,
            total_duration=response_total_duration,
            cost_savings=response_cost_savings,
            stop_reason=response_stop_reason,
            convergence_history=response_convergence_history,
            full_history=response_full_history,
            metadata={
                "finalized_at": time.time(),
                "experiment_run_id": mapping.experiment_run_id if mapping else None,
                "finalized_via_api": finalized_via_api,
                # summary_available=True iff at least one documented backend
                # summary field was type-valid and preserved. It does not
                # imply any specific field is present; inspect summary_fields
                # before treating best_config/best_metrics as authoritative.
                "summary_available": summary_available,
                "summary_fields": summary_fields,
            },
        )

        return response

    async def _finalize_session_via_api(
        self,
        session_id: str,
        experiment_run_id: str,
        certified_selection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Call backend session finalization endpoint.

        Args:
            session_id: Session ID to finalize
            experiment_run_id: Associated experiment run ID

        Returns:
            Parsed backend response payload (may be ``{}`` if the backend
            returned 2xx with an empty body) when finalization succeeded;
            ``None`` when the endpoint was unavailable, returned a non-2xx
            status, or failed locally. Per the tracked fix, callers use this
            return shape to distinguish "backend gave us a summary" from
            "backend just accepted the finalize call with no payload".
        """
        if not AIOHTTP_AVAILABLE:
            logger.debug("aiohttp not available, skipping API finalization")
            return None

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

                finalize_body: dict[str, Any] = {
                    "reason": "sdk_explicit_finalization",
                    "experiment_run_id": experiment_run_id,
                }
                if certified_selection is not None:
                    # Phase 8: the client-attested, content-free certified-
                    # selection report — TOP-LEVEL key only (the backend
                    # rejects metadata-tunneled reports).
                    finalize_body["certified_selection"] = certified_selection
                async with session.post(
                    url,
                    json=finalize_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(
                            f"✅ Finalized session {session_id} via backend API"
                        )
                        try:
                            payload = await response.json()
                            if not isinstance(payload, dict):
                                logger.debug(
                                    "Backend finalize returned non-dict body: %r",
                                    type(payload).__name__,
                                )
                                return {}
                            return payload
                        except Exception:
                            # Backend returned 2xx but body wasn't JSON or
                            # was empty — finalize succeeded but no summary
                            # payload to preserve.
                            return {}
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
                        return None
        except Exception as e:
            logger.debug(f"Error calling backend finalization API: {e}")
            return None

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
        self,
        session_id: str,
        include_full_history: bool = False,
        certified_selection: dict[str, Any] | None = None,
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
                    self.finalize_session(
                        session_id,
                        include_full_history,
                        certified_selection=certified_selection,
                    )
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
