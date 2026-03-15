"""Privacy-safe Edge Analytics insights.

This module collects aggregated, non-sensitive usage statistics in Edge Analytics mode
to help users understand the value proposition of upgrading to cloud services.
All data sent is privacy-safe and contains no sensitive information.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Performance FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import asyncio
import logging
import uuid
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from traigent.api.types import OptimizationStatus
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.storage.local_storage import LocalStorageManager

# Import constant with fallback for base-only installs (cloud module not available)
try:
    from traigent.cloud.auth import MIN_TOKEN_LENGTH
except ModuleNotFoundError:
    MIN_TOKEN_LENGTH = 20  # Default minimum acceptable token length

# Note: "Unclosed client session" warnings from background analytics tasks
# are suppressed locally in _submit_analytics_background() using catch_warnings()
# to avoid process-wide suppression that could hide real leaks elsewhere.

logger = logging.getLogger(__name__)

# Default analytics endpoint (deprecated - will use backend client instead)
DEFAULT_ANALYTICS_ENDPOINT = "http://localhost:5000/v1/local-usage"
_BACKGROUND_ANALYTICS_TASKS: set[asyncio.Task[Any]] = set()


def _register_background_task(task: asyncio.Task[Any]) -> None:
    """Keep fire-and-forget analytics tasks alive and surface failures."""

    def _on_done(fut: asyncio.Task[Any]) -> None:
        _BACKGROUND_ANALYTICS_TASKS.discard(fut)
        if fut.cancelled():
            return
        try:
            fut.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Background analytics task failed: %s", exc)

    _BACKGROUND_ANALYTICS_TASKS.add(task)
    task.add_done_callback(_on_done)


class LocalAnalytics:
    """Handles privacy-safe analytics collection for Edge Analytics mode."""

    def __init__(self, config: TraigentConfig) -> None:
        """Initialize analytics collector.

        Args:
            config: Traigent configuration with analytics settings
        """
        self.config = config
        storage_path_option = config.get_local_storage_path()
        storage_path_str = storage_path_option or str(Path.home() / ".traigent")
        self.storage = LocalStorageManager(storage_path_str)
        self.analytics_endpoint = (
            config.analytics_endpoint or DEFAULT_ANALYTICS_ENDPOINT
        )
        self.enabled = config.enable_usage_analytics and (
            config.execution_mode == ExecutionMode.EDGE_ANALYTICS.value
        )

        # Get or create anonymous user ID
        self.user_id = self._get_or_create_user_id()

    def _get_or_create_user_id(self) -> str:
        """Get or create an anonymous user identifier."""
        if self.config.anonymous_user_id:
            return self.config.anonymous_user_id

        # Check for existing ID in storage
        analytics_file = Path(self.storage.storage_path) / ".analytics_id"
        if analytics_file.exists():
            try:
                return analytics_file.read_text().strip()
            except Exception as e:
                logger.debug(f"Could not read analytics ID from storage: {e}")

        # Create new anonymous ID
        user_id = str(uuid.uuid4())
        try:
            analytics_file.write_text(user_id)
        except Exception as e:
            logger.debug(f"Could not save analytics ID: {e}")

        return user_id

    def collect_usage_stats(self) -> dict[str, Any]:
        """Collect privacy-safe aggregated usage statistics.

        Returns:
            Dictionary containing aggregated, non-sensitive stats
        """
        if not self.enabled:
            return {}

        try:
            sessions = self.storage.list_sessions()
            completed_sessions = [
                s for s in sessions if s.status == OptimizationStatus.COMPLETED.value
            ]
            failed_sessions = [
                s for s in sessions if s.status == OptimizationStatus.FAILED.value
            ]

            # Calculate aggregated metrics
            total_trials = sum(
                s.completed_trials for s in sessions
            )  # Use completed_trials as total
            total_completed_trials = sum(s.completed_trials for s in sessions)

            # Function complexity metrics (no actual function names)
            function_count = len({s.function_name for s in sessions})
            avg_trials_per_session = total_trials / len(sessions) if sessions else 0

            # Time-based aggregations
            now = datetime.now(UTC)
            recent_sessions = [
                s
                for s in sessions
                if datetime.fromisoformat(s.created_at) > now - timedelta(days=30)
            ]

            # Configuration space complexity (aggregated)
            config_space_sizes: list[int] = []
            for session in completed_sessions:
                if (
                    hasattr(session, "optimization_config")
                    and session.optimization_config
                ):
                    config_space = session.optimization_config.get(
                        "configuration_space", {}
                    )
                    if isinstance(config_space, dict):
                        config_space_sizes.append(len(config_space))

            avg_config_space_size = (
                sum(config_space_sizes) / len(config_space_sizes)
                if config_space_sizes
                else 0
            )

            # Performance improvement metrics (aggregated)
            improvements: list[float] = []
            for session in completed_sessions:
                summary = self.storage.get_session_summary(session.session_id)
                if isinstance(summary, dict):
                    value = summary.get("improvement")
                    if isinstance(value, (int, float)):
                        improvements.append(float(value))

            avg_improvement = (
                sum(improvements) / len(improvements) if improvements else None
            )

            return {
                # Basic usage metrics
                "total_sessions": len(sessions),
                "completed_sessions": len(completed_sessions),
                "failed_sessions": len(failed_sessions),
                "total_trials": total_trials,
                "total_completed_trials": total_completed_trials,
                "unique_functions_optimized": function_count,
                # Complexity and performance metrics
                "avg_trials_per_session": round(avg_trials_per_session, 2),
                "avg_config_space_size": round(avg_config_space_size, 2),
                "avg_improvement_percent": (
                    round(avg_improvement * 100, 2) if avg_improvement else None
                ),
                # Time-based metrics
                "sessions_last_30_days": len(recent_sessions),
                "days_since_first_use": self._get_days_since_first_use(),
                # Version and environment (non-sensitive)
                "sdk_version": "1.1.0",  # Should be dynamic from version
                "execution_mode": self.config.execution_mode,
                "timestamp": datetime.now(UTC).isoformat(),
                # Anonymous user tracking
                "anonymous_user_id": self.user_id,
                # No sensitive data included:
                # - No function names, parameters, or values
                # - No file paths or user information
                # - No actual optimization results
                # - No API keys or credentials
            }

        except Exception as e:
            logger.debug(f"Error collecting usage stats: {e}")
            return {}

    def _get_days_since_first_use(self) -> int:
        """Calculate days since first optimization session."""
        try:
            sessions = self.storage.list_sessions()
            if not sessions:
                return 0

            earliest_session = min(
                sessions, key=lambda s: datetime.fromisoformat(s.created_at)
            )

            first_use = datetime.fromisoformat(earliest_session.created_at)
            return (datetime.now(UTC) - first_use).days

        except Exception as e:
            logger.debug(f"Could not calculate days active (defaulting to 0): {e}")
            return 0

    async def submit_usage_stats(self, force: bool = False) -> dict[str, Any]:
        """Submit usage statistics using existing backend session infrastructure.

        Creates a lightweight analytics session with only aggregated, non-sensitive data.

        Args:
            force: Force submission even if not due yet

        Returns:
            Result dictionary with success status and details
        """
        if not self.enabled:
            return {"success": False, "reason": "Analytics disabled"}

        # Check if submission is due (daily by default)
        if not force and not self._is_submission_due():
            return {"success": False, "reason": "Submission not due yet"}

        stats = self.collect_usage_stats()
        if not stats:
            return {"success": False, "reason": "No stats to submit"}

        try:
            # Import backend client lazily to avoid circular imports
            from traigent.cloud.backend_client import get_backend_client

            # Get API key from environment
            from traigent.config.backend_config import BackendConfig

            api_key = BackendConfig.get_api_key()
            if not api_key:
                logger.debug("No Traigent API key found, analytics submission skipped")
                return {"success": False, "reason": "No API key available"}

            if len(api_key) < MIN_TOKEN_LENGTH:
                logger.debug(
                    "API key too short for analytics submission (length=%d), skipping",
                    len(api_key),
                )
                return {"success": False, "reason": "API key invalid"}

            backend_client = get_backend_client(
                api_key=api_key,
                base_url=BackendConfig.get_backend_url(),
                enable_fallback=False,  # We're explicitly sending analytics
                timeout=5.0,  # Short timeout for analytics - don't block program exit
            )

            try:
                await backend_client.__aenter__()
                # Create a lightweight analytics session
                # Using the hybrid session method for analytics submission
                (
                    session_id,
                    token,
                    endpoint,
                ) = await backend_client.create_hybrid_session(
                    problem_statement="Local usage analytics data collection",
                    search_space={
                        "type": "analytics",
                        "mode": "local_usage_stats",
                        "version": stats.get("sdk_version", "1.1.0"),
                    },
                    optimization_config={
                        "objectives": ["track_usage"],  # Placeholder objective
                        "max_trials": 1,  # Single analytics submission
                    },
                    metadata={
                        "user_id": self.user_id,
                        "dataset_metadata": {
                            "size": 0,  # No actual dataset
                            "type": "analytics_metadata",
                            "sessions": stats.get("total_sessions", 0),
                            "trials": stats.get("total_trials", 0),
                        },
                    },
                )

                # Submit the aggregated stats as trial results
                success = await backend_client.submit_privacy_trial_results(
                    session_id=session_id,
                    trial_id=f"analytics_{datetime.now(UTC).isoformat()}",
                    config={},
                    metrics={
                        "total_sessions": float(stats.get("total_sessions", 0)),
                        "completed_sessions": float(stats.get("completed_sessions", 0)),
                        "total_trials": float(stats.get("total_trials", 0)),
                        "avg_trials_per_session": stats.get(
                            "avg_trials_per_session", 0.0
                        ),
                        "avg_config_space_size": stats.get(
                            "avg_config_space_size", 0.0
                        ),
                        "avg_improvement_percent": stats.get(
                            "avg_improvement_percent", 0.0
                        )
                        or 0.0,
                        "sessions_last_30_days": float(
                            stats.get("sessions_last_30_days", 0)
                        ),
                        "days_since_first_use": float(
                            stats.get("days_since_first_use", 0)
                        ),
                    },
                    duration=1.0,  # Analytics collection time
                )

                if success:
                    self._update_last_submission()
                    return {
                        "success": True,
                        "stats_submitted": len(stats),
                        "anonymous_id": self.user_id,
                        "session_id": session_id,
                    }
                else:
                    return {
                        "success": False,
                        "reason": "Failed to submit trial results",
                    }
            finally:
                # Ensure session is closed even on cancellation
                try:
                    await backend_client.__aexit__(None, None, None)
                except Exception:
                    pass  # Best effort cleanup

        except TimeoutError:
            return {"success": False, "reason": "Request timeout"}
        except ImportError:
            logger.debug("Backend client not available for analytics submission")
            return {"success": False, "reason": "Backend client not available"}
        except Exception as e:
            logger.debug(f"Analytics submission failed: {e}")
            return {"success": False, "reason": str(e)}

    def _is_submission_due(self) -> bool:
        """Check if analytics submission is due."""
        try:
            last_submission_file = Path(self.storage.storage_path) / ".last_analytics"
            if not last_submission_file.exists():
                return True

            last_submission = datetime.fromisoformat(
                last_submission_file.read_text().strip()
            )

            # Ensure timezone-aware comparison
            if last_submission.tzinfo is None:
                last_submission = last_submission.replace(tzinfo=UTC)

            # Submit once per day
            return datetime.now(UTC) - last_submission > timedelta(days=1)

        except Exception as e:
            logger.debug(f"Could not check submission status (assuming due): {e}")
            return True  # If we can't determine, assume it's due

    def _update_last_submission(self) -> None:
        """Update the timestamp of last analytics submission."""
        try:
            last_submission_file = Path(self.storage.storage_path) / ".last_analytics"
            last_submission_file.write_text(datetime.now(UTC).isoformat())
        except Exception as e:
            logger.debug(f"Could not update last submission time: {e}")

    def get_cloud_incentive_data(self) -> dict[str, Any]:
        """Generate data for cloud adoption incentives.

        Returns:
            Dictionary with cloud adoption messaging and value props
        """
        stats = self.collect_usage_stats()
        if not stats:
            return {}

        # Calculate value propositions based on actual usage
        total_sessions = stats.get("total_sessions", 0)
        avg_trials = stats.get("avg_trials_per_session", 0)
        avg_config_space = stats.get("avg_config_space_size", 0)

        # Estimate time investment
        estimated_time_per_trial = 2  # minutes (conservative estimate)
        total_time_invested = stats.get("total_trials", 0) * estimated_time_per_trial

        # Cloud benefits messaging based on usage patterns
        benefits = {
            "algorithm_upgrade": {
                "current": "Random/Grid search only",
                "cloud": "Bayesian optimization (3-5x faster)",
                "value": f"Could save ~{int(total_time_invested * 0.6)} minutes",
            },
            "trial_limit": {
                "current": f"20 trial limit (avg: {avg_trials:.1f} trials)",
                "cloud": "Unlimited trials",
                "value": "Explore full optimization potential",
            },
            "analytics": {
                "current": "Basic local summaries",
                "cloud": "Advanced web dashboard with insights",
                "value": f"Cross-optimization insights from {total_sessions} sessions",
            },
        }

        # Advanced benefits for power users
        if total_sessions >= 10:
            benefits["team_collaboration"] = {
                "current": "Individual optimization only",
                "cloud": "Team sharing and collaboration",
                "value": f"Share {total_sessions} optimizations with team",
            }

        if avg_config_space >= 5:
            benefits["complex_optimization"] = {
                "current": "Limited to simple parameter spaces",
                "cloud": "Multi-objective and constraint optimization",
                "value": f"Handle complex {avg_config_space:.0f}-parameter spaces",
            }

        return {
            "usage_summary": stats,
            "cloud_benefits": benefits,
            "personalized_message": self._generate_personalized_message(stats),
            "upgrade_urgency": self._calculate_upgrade_urgency(stats),
        }

    def _generate_personalized_message(self, stats: dict[str, Any]) -> str:
        """Generate personalized upgrade message based on usage."""
        total_sessions = stats.get("total_sessions", 0)
        total_trials = stats.get("total_trials", 0)
        avg_improvement = stats.get("avg_improvement_percent")

        if total_sessions >= 20:
            return f"🏆 Power user detected! You've completed {total_sessions} optimizations. Cloud features would supercharge your workflow."
        elif total_trials >= 100:
            return f"📈 You've run {total_trials} trials - cloud Bayesian optimization could reduce this by 60-80%."
        elif avg_improvement and avg_improvement > 20:
            return f"🎯 Great results! {avg_improvement:.1f}% average improvement. Cloud features can help you achieve even more."
        elif total_sessions >= 5:
            return f"✨ You're getting good results with {total_sessions} optimizations. Ready for advanced algorithms?"
        else:
            return "🚀 Great start! Cloud features unlock advanced optimization algorithms and unlimited trials."

    def _calculate_upgrade_urgency(self, stats: dict[str, Any]) -> str:
        """Calculate upgrade urgency based on usage patterns."""
        total_sessions = stats.get("total_sessions", 0)
        avg_trials = stats.get("avg_trials_per_session", 0)
        days_used = stats.get("days_since_first_use", 0)

        # High urgency indicators
        if avg_trials >= 15:  # Hitting trial limits
            return "high"
        elif total_sessions >= 15:  # Heavy usage
            return "high"
        elif total_sessions >= 5 and days_used >= 7:  # Regular usage over time
            return "medium"
        elif total_sessions >= 3:  # Some usage
            return "low"
        else:
            return "none"


# Convenience function for easy integration
def collect_and_submit_analytics(config: TraigentConfig) -> None:
    """Convenience function to collect and submit analytics asynchronously.

    This function is designed to work from both sync and async contexts.
    It runs analytics submission in the background without blocking.
    """
    if not config.enable_usage_analytics or config.execution_mode not in {
        ExecutionMode.EDGE_ANALYTICS.value,
    }:
        return

    try:
        analytics = LocalAnalytics(config)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()

            # We're in an async context, but we can't await here since this is a sync function
            # Instead, we'll schedule the coroutine to run soon without creating an unawaited task
            async def _submit_wrapper() -> None:
                # Suppress "Unclosed client session" warnings locally for this
                # fire-and-forget task that may be cancelled on program exit
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Unclosed client session",
                        category=ResourceWarning,
                    )
                    try:
                        # Use timeout to ensure we don't block program exit
                        await asyncio.wait_for(
                            analytics.submit_usage_stats(), timeout=10.0
                        )
                        logger.debug("Analytics submission completed")
                    except TimeoutError:
                        logger.debug("Analytics submission timed out")
                    except asyncio.CancelledError:
                        # Task was cancelled (e.g., program exiting) - expected behavior
                        logger.debug("Analytics submission cancelled")
                        raise  # Re-raise to properly propagate cancellation
                    except Exception as e:
                        logger.debug(f"Analytics submission failed: {e}")

            task = asyncio.ensure_future(_submit_wrapper())
            _register_background_task(task)
        except RuntimeError:
            # No running loop, we're in a sync context
            # Create a new event loop in a thread to avoid blocking
            import threading

            def _run_in_thread() -> None:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(analytics.submit_usage_stats())
                    loop.close()
                    logger.debug("Analytics submission completed")
                except Exception as e:
                    logger.debug(f"Analytics submission failed: {e}")

            # Run in a daemon thread so it doesn't block program exit
            thread = threading.Thread(target=_run_in_thread, daemon=True)
            thread.start()

    except Exception as e:
        logger.debug(f"Background analytics submission failed: {e}")


# Integration with existing incentive system
def get_enhanced_cloud_incentives(config: TraigentConfig) -> dict[str, Any]:
    """Get enhanced cloud incentives with real usage data."""
    analytics = LocalAnalytics(config)
    return analytics.get_cloud_incentive_data()
