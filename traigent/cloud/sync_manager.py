"""
Local-to-cloud sync manager for Traigent optimization sessions.
Handles migration of local data to Traigent backend when users upgrade.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

import os
from datetime import UTC, datetime
from typing import Any

import requests  # Always needed for synchronous operations

try:
    import aiohttp  # noqa: F401 - Import check only

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..config.backend_config import BackendConfig
from ..config.types import TraigentConfig
from ..storage.local_storage import LocalStorageManager, OptimizationSession
from ..utils.exceptions import TraigentStorageError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SyncManager:
    """
    Manages synchronization of local optimization data to Traigent backend.

    Converts local session data to Traigent experiment/experiment_run format
    and handles the upload process with progress tracking.
    """

    def __init__(self, config: TraigentConfig, api_key: str | None = None) -> None:
        """
        Initialize sync manager.

        Args:
            config: TraigentConfig with local storage settings
            api_key: API key for cloud service authentication
        """
        self.config = config
        self.storage = LocalStorageManager(config.get_local_storage_path())
        self.api_key = api_key
        self.base_url = BackendConfig.get_backend_api_url().rstrip("/")

        # Setup HTTP client - always create a requests session for sync operations
        # Include both X-API-Key and Authorization for backward compatibility
        self.headers = (
            {"X-API-Key": api_key, "Authorization": f"Bearer {api_key}"}
            if api_key
            else {}
        )
        # Always create requests session for synchronous operations
        self._session = requests.Session()
        if api_key:
            self._session.headers.update(self.headers)
        self._request_timeout = self._resolve_request_timeout()

    @property
    def session(self):
        """Public access to the HTTP session for testing."""
        return self._session

    def _resolve_request_timeout(self) -> float:
        """Resolve timeout (seconds) applied to blocking HTTP calls."""
        default_timeout = 15.0

        env_timeout = os.getenv("TRAIGENT_SYNC_HTTP_TIMEOUT")
        custom_timeout = None
        custom_params = getattr(self.config, "custom_params", None)
        if isinstance(custom_params, dict):
            for key in (
                "sync_request_timeout",
                "sync_request_timeout_seconds",
                "request_timeout",
            ):
                if custom_params.get(key) is not None:
                    custom_timeout = custom_params[key]
                    break

        sources: list[tuple[str, Any]] = [
            ("environment variable TRAIGENT_SYNC_HTTP_TIMEOUT", env_timeout),
            ("TraigentConfig.custom_params", custom_timeout),
        ]

        for source_name, raw_value in sources:
            if raw_value is None:
                continue
            try:
                timeout_value = float(raw_value)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid %s value for sync HTTP timeout: %r. "
                    "Falling back to default %s seconds.",
                    source_name,
                    raw_value,
                    default_timeout,
                )
                continue

            if timeout_value <= 0:
                logger.warning(
                    "%s provided non-positive sync HTTP timeout (%s). "
                    "Using default %s seconds.",
                    source_name,
                    timeout_value,
                    default_timeout,
                )
                continue

            return timeout_value

        return default_timeout

    def get_sync_status(self) -> dict[str, Any]:
        """Get status of local sessions and sync eligibility."""
        sessions = self.storage.list_sessions()
        completed_sessions = [s for s in sessions if s.status == "completed"]

        total_trials = sum(s.completed_trials for s in sessions)

        sync_status = {
            "total_sessions": len(sessions),
            "completed_sessions": len(completed_sessions),
            "total_trials": total_trials,
            "sync_eligible": len(completed_sessions),
            "estimated_cloud_value": self._estimate_cloud_value(completed_sessions),
            "storage_info": self.storage.get_storage_info(),
        }

        return sync_status

    def _estimate_cloud_value(
        self, sessions: list[OptimizationSession]
    ) -> dict[str, Any]:
        """Estimate the value proposition of syncing to cloud."""
        if not sessions:
            return {}

        avg_trials = sum(s.completed_trials for s in sessions) / len(sessions)
        total_time_invested = len(sessions) * 15  # Estimate 15 min per optimization

        return {
            "sessions_to_sync": len(sessions),
            "average_trials_per_session": avg_trials,
            "estimated_time_invested_hours": total_time_invested / 60,
            "cloud_benefits": {
                "advanced_algorithms": "3-5x faster convergence",
                "web_dashboard": "Visual analytics and progress tracking",
                "team_collaboration": "Share optimizations across team",
                "unlimited_trials": f"vs. current {int(avg_trials)} trial average",
            },
        }

    def convert_session_to_traigent_format(
        self, session: OptimizationSession
    ) -> dict[str, Any]:
        """
        Convert local session to Traigent experiment format.

        Args:
            session: Local optimization session

        Returns:
            Dict in Traigent experiment/experiment_run format
        """
        # Generate placeholder IDs for required fields
        experiment_id = f"local_import_{session.session_id}"
        agent_id = f"local_agent_{session.function_name}"
        benchmark_id = f"local_benchmark_{session.function_name}"
        model_params_id = f"edge_analytics_model_params_{session.session_id}"

        # Create minimal agent definition
        agent_data = {
            "id": agent_id,
            "name": f"Local Agent: {session.function_name}",
            "agent_type": "custom",
            "description": f"Imported from local optimization of {session.function_name}",
            "source": "local_import",
            "created_at": session.created_at,
        }

        # Create minimal benchmark definition
        benchmark_data = {
            "id": benchmark_id,
            "name": f"Local Benchmark: {session.function_name}",
            "label": f"{session.function_name}_eval",
            "description": f"Evaluation dataset for {session.function_name}",
            "type": "custom",
            "examples_count": session.completed_trials,  # Approximate
            "source": "local_import",
        }

        # Create model parameters
        opt_config = session.optimization_config or {}
        model_params_data = {
            "id": model_params_id,
            "model_id": (opt_config.get("search_space", {}) or {}).get(
                "model", "unknown"
            ),
            "source": "local_import",
        }

        # Create experiment
        experiment_data = {
            "id": experiment_id,
            "name": f"Local Import: {session.function_name}",
            "description": "Imported optimization session from Edge Analytics mode",
            "agent_id": agent_id,
            "benchmark_id": benchmark_id,
            "model_parameters_id": model_params_id,
            "configurations": opt_config.get("search_space", {}),
            "measures": ["score", "accuracy"],  # Default measures
            "status": session.status,
            "source": "local_import",
            "metadata": {
                "original_session_id": session.session_id,
                "import_timestamp": datetime.now(UTC).isoformat(),
                "edge_analytics_version": "1.0.0",
            },
        }

        # Create experiment run with trials
        experiment_run_data = {
            "id": f"{experiment_id}_run",
            "experiment_id": experiment_id,
            "experiment_data": experiment_data,
            "status": session.status,
            "results": self._convert_trials_to_results(session.trials or []),
            "metadata": {
                "total_trials": session.completed_trials,
                "best_score": session.best_score,
                "best_config": session.best_config,
                "baseline_score": session.baseline_score,
            },
        }

        return {
            "agent": agent_data,
            "benchmark": benchmark_data,
            "model_parameters": model_params_data,
            "experiment": experiment_data,
            "experiment_run": experiment_run_data,
        }

    def _convert_trials_to_results(self, trials: list[Any]) -> list[dict[str, Any]]:
        """Convert local trials to Traigent configuration_run format."""
        results = []

        for trial in trials:
            result = {
                "id": f"config_run_{trial.trial_id}",
                "trial_id": trial.trial_id,
                "experiment_parameters": trial.config,
                "measures": {
                    "score": trial.score,
                    "accuracy": trial.score,  # Map score to accuracy for compatibility
                },
                "timestamp": trial.timestamp,
                "status": "completed" if trial.error is None else "failed",
            }

            if trial.error:
                result["error"] = trial.error

            if trial.metadata:
                result["metadata"] = trial.metadata

            results.append(result)

        return results

    def sync_session_to_cloud(
        self, session_id: str, dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Sync a single session to cloud.

        Args:
            session_id: Local session ID to sync
            dry_run: If True, only validate data without uploading

        Returns:
            Sync result with status and details
        """
        session = self.storage.load_session(session_id)
        if not session:
            raise TraigentStorageError(f"Session {session_id} not found") from None

        # Convert to Traigent format
        traigent_data = self.convert_session_to_traigent_format(session)

        sync_result = {
            "session_id": session_id,
            "function_name": session.function_name,
            "status": "success",
            "dry_run": dry_run,
            "data_converted": True,
            "cloud_experiment_id": traigent_data["experiment"]["id"],
            "trials_converted": len(traigent_data["experiment_run"]["results"]),
            "errors": [],
        }

        if dry_run:
            sync_result["preview"] = {
                "experiment_name": traigent_data["experiment"]["name"],
                "agent_name": traigent_data["agent"]["name"],
                "benchmark_name": traigent_data["benchmark"]["name"],
                "trial_count": len(traigent_data["experiment_run"]["results"]),
                "best_score": session.best_score,
            }
            return sync_result

        # Actually sync to cloud
        if not self.api_key:
            sync_result["status"] = "error"
            sync_result["errors"].append("No API key provided for cloud sync")
            return sync_result

        try:
            # Track success/failure of each step
            successful_steps = 0
            total_steps = 4

            # Step 1: Create or verify agent
            agent_result = self._sync_agent(traigent_data["agent"])
            if not agent_result["success"]:
                sync_result["errors"].append(
                    f"Agent sync failed: {agent_result['error']}"
                )
            else:
                successful_steps += 1

            # Step 2: Create or verify benchmark
            benchmark_result = self._sync_benchmark(traigent_data["benchmark"])
            if not benchmark_result["success"]:
                sync_result["errors"].append(
                    f"Benchmark sync failed: {benchmark_result['error']}"
                )
            else:
                successful_steps += 1

            # Step 3: Create experiment
            experiment_result = self._sync_experiment(traigent_data["experiment"])
            if not experiment_result["success"]:
                sync_result["errors"].append(
                    f"Experiment sync failed: {experiment_result['error']}"
                )
            else:
                successful_steps += 1

            # Step 4: Create experiment run with results
            run_result = self._sync_experiment_run(traigent_data["experiment_run"])
            if not run_result["success"]:
                sync_result["errors"].append(
                    f"Experiment run sync failed: {run_result['error']}"
                )
            else:
                successful_steps += 1

            # Determine status based on success rate
            if successful_steps == 0:
                sync_result["status"] = "error"  # Complete failure
            elif successful_steps == total_steps:
                sync_result["status"] = "success"
                sync_result["cloud_url"] = (
                    f"{self.base_url}/experiments/{traigent_data['experiment']['id']}"
                )
            else:
                sync_result["status"] = "partial"  # Some steps succeeded

        except Exception as e:
            # Override any partial status with error for complete failures
            sync_result["status"] = "error"
            sync_result["errors"].append(f"Sync failed: {str(e)}")
            logger.error(f"Failed to sync session {session_id}: {e}")

        return sync_result

    def sync_all_sessions(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Sync all eligible local sessions to cloud.

        Args:
            dry_run: If True, only validate data without uploading

        Returns:
            Overall sync result
        """
        sessions = self.storage.list_sessions()
        completed_sessions = [s for s in sessions if s.status == "completed"]

        session_results: list[dict[str, Any]] = []
        synced_successfully = 0
        sync_errors = 0
        overall_status = "success"

        for session in completed_sessions:
            try:
                result = self.sync_session_to_cloud(session.session_id, dry_run)
                session_results.append(result)

                if result["status"] == "success":
                    synced_successfully += 1
                else:
                    sync_errors += 1

            except Exception as e:
                sync_errors += 1
                session_results.append(
                    {
                        "session_id": session.session_id,
                        "status": "error",
                        "errors": [str(e)],
                    }
                )

        if sync_errors > 0:
            overall_status = "partial" if synced_successfully > 0 else "failed"

        return {
            "total_sessions": len(sessions),
            "eligible_sessions": len(completed_sessions),
            "synced_successfully": synced_successfully,
            "sync_errors": sync_errors,
            "dry_run": dry_run,
            "session_results": session_results,
            "overall_status": overall_status,
        }

    def _sync_agent(self, agent_data: dict[str, Any]) -> dict[str, Any]:
        """Sync agent data to cloud."""
        try:
            response = self._session.post(
                f"{self.base_url}/agents",
                json=agent_data,
                timeout=self._request_timeout,
            )
            if response.status_code in [200, 201, 409]:  # 409 = already exists
                return {"success": True, "agent_id": agent_data["id"]}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_benchmark(self, benchmark_data: dict[str, Any]) -> dict[str, Any]:
        """Sync benchmark data to cloud."""
        try:
            response = self._session.post(
                f"{self.base_url}/benchmarks",
                json=benchmark_data,
                timeout=self._request_timeout,
            )
            if response.status_code in [200, 201, 409]:
                return {"success": True, "benchmark_id": benchmark_data["id"]}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_experiment(self, experiment_data: dict[str, Any]) -> dict[str, Any]:
        """Sync experiment data to cloud."""
        try:
            response = self._session.post(
                f"{self.base_url}/experiments",
                json=experiment_data,
                timeout=self._request_timeout,
            )
            if response.status_code in [200, 201, 409]:
                return {"success": True, "experiment_id": experiment_data["id"]}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_experiment_run(self, run_data: dict[str, Any]) -> dict[str, Any]:
        """Sync experiment run data to cloud."""
        try:
            response = self._session.post(
                f"{self.base_url}/experiment-runs",
                json=run_data,
                timeout=self._request_timeout,
            )
            if response.status_code in [200, 201]:
                return {"success": True, "run_id": run_data["id"]}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_cloud_analytics_preview(self) -> dict[str, Any]:
        """Get preview of analytics available in cloud after sync."""
        sessions = self.storage.list_sessions()
        completed_sessions = [s for s in sessions if s.status == "completed"]

        if not completed_sessions:
            return {"message": "No completed sessions to analyze"}

        # Calculate potential insights
        total_trials = sum(s.completed_trials for s in completed_sessions)
        avg_improvement: list[float] = []

        for session in completed_sessions:
            summary = self.storage.get_session_summary(session.session_id)
            improvement = summary.get("improvement") if summary else None
            if improvement is not None:
                avg_improvement.append(float(improvement))

        analytics_preview = {
            "sessions_ready_for_sync": len(completed_sessions),
            "total_optimization_trials": total_trials,
            "functions_optimized": len({s.function_name for s in completed_sessions}),
            "average_improvement": (
                sum(avg_improvement) / len(avg_improvement) if avg_improvement else None
            ),
            "cloud_analytics_available": {
                "cross_function_insights": "Compare optimization patterns across functions",
                "trend_analysis": "Track optimization performance over time",
                "parameter_impact_analysis": "Understand which parameters drive performance",
                "team_leaderboards": "See top-performing optimizations across your team",
                "automated_alerts": "Get notified when optimizations complete or fail",
                "export_capabilities": "Export results to CSV, PDF, or integrate with BI tools",
            },
            "estimated_dashboard_value": {
                "time_saved_reviewing_results": f"{len(completed_sessions) * 5} minutes",
                "insights_gained": f"Cross-function patterns from {total_trials} trials",
                "collaboration_potential": f"Share {len(completed_sessions)} optimizations with team",
            },
        }

        return analytics_preview

    def cleanup_after_sync(
        self, session_ids: list[str], keep_backup: bool = True
    ) -> dict[str, Any]:
        """
        Clean up local sessions after successful sync.

        Args:
            session_ids: List of session IDs to clean up
            keep_backup: If True, create backup before deletion

        Returns:
            Cleanup result
        """
        sessions_backed_up = 0
        sessions_deleted = 0
        errors: list[str] = []

        if keep_backup:
            backup_dir = (
                self.storage.storage_path
                / "backups"
                / f"sync_backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir.mkdir(parents=True, exist_ok=True)

            for session_id in session_ids:
                try:
                    backup_success = self.storage.export_session(
                        session_id, str(backup_dir / f"{session_id}.json"), "json"
                    )
                    if backup_success:
                        sessions_backed_up += 1
                except Exception as e:
                    errors.append(f"Backup failed for {session_id}: {e}")

        # Delete sessions
        for session_id in session_ids:
            try:
                if self.storage.delete_session(session_id):
                    sessions_deleted += 1
            except Exception as e:
                errors.append(f"Deletion failed for {session_id}: {e}")

        return {
            "sessions_processed": len(session_ids),
            "sessions_backed_up": sessions_backed_up,
            "sessions_deleted": sessions_deleted,
            "errors": errors,
        }
