"""Privacy-first optimization operations for TraiGent Cloud Client.

This module handles privacy-preserving optimization operations where data
remains local and only metrics are transmitted to the cloud.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from traigent.cloud.client import CloudServiceError
from traigent.cloud.models import (
    NextTrialRequest,
    OptimizationSession,
    OptimizationSessionStatus,
    SessionCreationRequest,
    TrialResultSubmission,
    TrialStatus,
    TrialSuggestion,
)
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient

logger = get_logger(__name__)


class PrivacyOperations:
    """Handles privacy-first optimization operations."""

    def __init__(self, client: "BackendIntegratedClient"):
        """Initialize privacy operations handler.

        Args:
            client: Parent BackendIntegratedClient instance
        """
        self.client = client

    async def create_privacy_optimization_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        dataset_metadata: dict[str, Any],
        max_trials: int = 50,
        user_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Create privacy-first optimization session with backend experiment tracking.

        This creates both a cloud session for configuration suggestions and a backend
        experiment for tracking, while keeping sensitive data local.

        Args:
            function_name: Name of function being optimized
            configuration_space: Parameter search space
            objectives: Optimization objectives
            dataset_metadata: Dataset metadata (no actual data)
            max_trials: Maximum optimization trials
            user_id: Optional user identifier

        Returns:
            Tuple of (session_id, experiment_id, experiment_run_id)
        """
        # Input validation
        if not function_name or not isinstance(function_name, str):
            raise ValueError("function_name must be a non-empty string")
        if not configuration_space or not isinstance(configuration_space, dict):
            raise ValueError("configuration_space must be a non-empty dictionary")
        if not objectives or not isinstance(objectives, list):
            raise ValueError("objectives must be a non-empty list")
        if max_trials < 1 or max_trials > 10000:
            raise ValueError("max_trials must be between 1 and 10000")

        # Sanitize function name to prevent injection
        function_name = function_name[:100].replace("\n", " ").replace("\r", " ")

        # Check rate limit
        await self.client._check_rate_limit()

        logger.info(f"Creating privacy-first optimization session for {function_name}")

        try:
            # Create cloud session for configuration suggestions
            session_request = SessionCreationRequest(
                function_name=function_name,
                configuration_space=configuration_space,
                objectives=objectives,
                dataset_metadata=dataset_metadata,
                max_trials=max_trials,
                user_id=user_id,
                billing_tier="privacy",  # Special tier for privacy mode
            )

            session_response = await self.client._create_cloud_session(session_request)
            session_id = session_response.session_id

            # Always use session endpoints for tracking
            logger.info("🔄 Creating session via TraiGent session endpoints...")
            (
                session_id,
                experiment_id,
                experiment_run_id,
            ) = await self.client._create_traigent_session_via_api(session_request)
            logger.info(
                f"✅ Successfully created session via session endpoints: {session_id}"
            )

            # Create session mapping
            self.client.session_bridge.create_session_mapping(
                session_id=session_id,
                experiment_id=experiment_id,
                experiment_run_id=experiment_run_id,
                function_name=function_name,
                configuration_space=configuration_space,
                objectives=objectives,
            )

            # Register session with security manager and issue validation token
            self.client._register_security_session(
                session_id,
                user_id,
                {
                    "function_name": function_name,
                    "objectives": objectives,
                    "experiment_id": experiment_id,
                    "experiment_run_id": experiment_run_id,
                },
            )

            # Store active session
            session = OptimizationSession(
                session_id=session_id,
                function_name=function_name,
                configuration_space=configuration_space,
                objectives=objectives,
                max_trials=max_trials,
                status=OptimizationSessionStatus.ACTIVE,
                created_at=session_response.metadata.get("created_at", time.time()),
                updated_at=datetime.now(timezone.utc),
                optimization_strategy=session_response.optimization_strategy,
            )

            with self.client._active_sessions_lock:
                if (
                    len(self.client._active_sessions)
                    >= self.client._max_active_sessions
                ):
                    # Find and remove the oldest completed session
                    oldest_session_id = None
                    oldest_time: datetime = datetime.max

                    for sid, sess in list(self.client._active_sessions.items()):
                        if sess.status in [
                            OptimizationSessionStatus.COMPLETED,
                            OptimizationSessionStatus.FAILED,
                        ]:
                            if sess.created_at < oldest_time:
                                oldest_time = sess.created_at
                                oldest_session_id = sid

                    # If no completed sessions, remove the oldest active one
                    if oldest_session_id is None and self.client._active_sessions:
                        oldest_session_id = min(
                            self.client._active_sessions.keys(),
                            key=lambda s: self.client._active_sessions[s].created_at,
                        )

                    if oldest_session_id:
                        del self.client._active_sessions[oldest_session_id]
                        logger.debug(
                            "Removed session %s to stay within active session limit",
                            oldest_session_id,
                        )

                self.client._active_sessions[session_id] = session

            logger.info(
                f"Created privacy session: {session_id} -> {experiment_id}/{experiment_run_id}"
            )
            return session_id, experiment_id, experiment_run_id

        except Exception as e:
            logger.error(f"Failed to create privacy optimization session: {e}")
            raise CloudServiceError(f"Failed to create session: {e}") from None

    async def get_next_privacy_trial(
        self,
        session_id: str,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> TrialSuggestion | None:
        """Get next trial suggestion for privacy-first optimization.

        Args:
            session_id: Session ID
            previous_results: Previous trial results (metrics only, no data)

        Returns:
            Next trial suggestion with configuration and dataset subset indices
        """
        logger.debug(f"Getting next trial for privacy session {session_id}")

        try:
            # Get suggestion from cloud (no sensitive data transmitted)
            request = NextTrialRequest(
                session_id=session_id, previous_results=previous_results
            )

            response = await self.client._get_cloud_trial_suggestion(request)

            if response.suggestion:
                # Backend handles configuration run creation internally
                # Just track the trial mapping for our records
                mapping = self.client.session_bridge.get_session_mapping(session_id)
                if mapping:
                    self.client.session_bridge.add_trial_mapping(
                        session_id,
                        response.suggestion.trial_id,
                        response.suggestion.trial_id,  # Trial ID is the config run ID
                    )

                # Update session status
                with self.client._active_sessions_lock:
                    session = self.client._active_sessions.get(session_id)
                    if session:
                        session.completed_trials += 1
                        session.updated_at = datetime.now(timezone.utc)

            return response.suggestion

        except Exception as e:
            logger.error(f"Failed to get next trial for session {session_id}: {e}")
            return None

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

        Args:
            session_id: Session ID
            trial_id: Trial ID
            config: Configuration that was tested
            metrics: Evaluation metrics (no sensitive data)
            duration: Trial execution duration
            error_message: Optional error message

        Returns:
            True if submission successful
        """
        # Input validation
        if not session_id or not isinstance(session_id, str):
            logger.error("Invalid session_id provided")
            return False
        if not trial_id or not isinstance(trial_id, str):
            logger.error("Invalid trial_id provided")
            return False
        if not isinstance(config, dict):
            logger.error("Config must be a dictionary")
            return False
        if not isinstance(metrics, dict):
            logger.error("Metrics must be a dictionary")
            return False
        if not isinstance(duration, (int, float)) or duration < 0:
            logger.error("Duration must be a non-negative number")
            return False

        # Validate session exists and has valid token
        with self.client._active_sessions_lock:
            session_exists = session_id in self.client._active_sessions

        if not session_exists:
            logger.warning(f"Session {session_id} not found in active sessions")
            # Don't fail, but log for debugging

        # Check rate limit
        try:
            await self.client._check_rate_limit()
        except CloudServiceError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return False

        logger.debug(f"Submitting privacy trial results: {session_id}/{trial_id}")

        try:
            # Submit results to cloud (metrics only)
            submission = TrialResultSubmission(
                session_id=session_id,
                trial_id=trial_id,
                metrics=metrics,
                duration=duration,
                status=TrialStatus.FAILED if error_message else TrialStatus.COMPLETED,
                error_message=error_message,
            )

            await self.client._submit_cloud_trial_results(submission)

            # Submit via session endpoint only - no fallback to config run endpoint
            session_submitted = await self.client._submit_trial_result_via_session(
                session_id,
                trial_id,
                config,  # Pass config for backend hash generation
                metrics,
                "COMPLETED" if not error_message else "FAILED",
                error_message,
            )

            if not session_submitted:
                logger.error(
                    f"Failed to submit trial results for session {session_id}, trial {trial_id}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to submit trial results: {e}")
            return False
