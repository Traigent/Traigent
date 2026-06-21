"""
Local-to-backend sync manager for Traigent optimization sessions.
Handles migration of local data to Traigent backend when users upgrade.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlencode

import requests  # Always needed for synchronous operations

from traigent.cloud.client import raise_if_cloud_egress_disabled

try:
    import aiohttp  # noqa: F401 - Import check only

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..config.backend_config import BackendConfig, get_no_credentials_hint
from ..config.types import TraigentConfig
from ..storage.local_storage import (
    TRIAL_COST_FIELDS,
    LocalStorageManager,
    OptimizationSession,
    extract_trial_cost_fields,
)
from ..utils.exceptions import TraigentStorageError
from ..utils.logging import get_logger

logger = get_logger(__name__)

VALID_SYNC_AGENT_TYPE_IDS = frozenset(
    {"chat", "classification", "completion", "qa", "retrieval", "tool", "function"}
)
DEFAULT_SYNC_AGENT_TYPE_ID = "completion"
_BACKEND_NAME_DISALLOWED = re.compile(r"[^a-zA-Z0-9 _-]+")
_BACKEND_NAME_SPACES = re.compile(r"\s+")


def build_experiment_url(
    base_url: str,
    experiment_id: str,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> str:
    """Build the portal URL for an experiment.

    Single source of truth for experiment URL construction —
    used by both SyncManager and the orchestrator.
    """
    url = f"{base_url.rstrip('/')}/experiments/view/{quote(str(experiment_id), safe='')}"
    query_params = {
        key: normalized
        for key, value in (("project_id", project_id), ("tenant_id", tenant_id))
        if value is not None and (normalized := str(value).strip())
    }
    if not query_params:
        return url
    return f"{url}?{urlencode(query_params, quote_via=quote)}"


def sanitize_backend_name(value: str, fallback: str = "Local Dataset") -> str:
    """Normalize generated sync names to the backend name pattern."""

    sanitized = _BACKEND_NAME_DISALLOWED.sub(" ", value)
    sanitized = _BACKEND_NAME_SPACES.sub(" ", sanitized).strip()
    return sanitized or fallback


class SyncManager:
    """
    Manages synchronization of local optimization data to Traigent backend.

    Converts local session data to Traigent experiment/experiment_run format
    and handles the upload process with progress tracking.
    """

    def __init__(
        self,
        config: TraigentConfig,
        api_key: str | None = None,
        no_egress: bool = False,
    ) -> None:
        """
        Initialize sync manager.

        Args:
            config: TraigentConfig with local storage settings
            api_key: API key for backend authentication
            no_egress: Runtime policy flag that forbids backend transport
        """
        self.config = config
        self.storage = LocalStorageManager(config.get_local_storage_path())
        self.api_key = api_key
        self.no_egress = bool(no_egress)
        self.base_url = BackendConfig.get_cloud_api_url().rstrip("/")

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

    def _raise_if_backend_egress_disabled(self, operation: str) -> None:
        """Fail closed before local-to-backend sync transport."""

        raise_if_cloud_egress_disabled(operation, no_egress=self.no_egress)

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

    @staticmethod
    def _session_sync_status(session: OptimizationSession) -> str:
        """Classify a local session's cloud-sync status from its sync_state."""
        state = session.sync_state or {}
        status = state.get("status")
        if status in {"synced", "partial", "failed"}:
            return str(status)
        return "unsynced"

    @staticmethod
    def _is_finished_session(session: OptimizationSession) -> bool:
        """Return True for sessions that are definitively finished.

        A session is finished when its status is "completed" OR when it is
        stuck at "pending" but has completed trials and a terminal stop reason
        stored in its metadata (edge case: #1344 — old SDK wrote trials but
        forgot to flip the status flag).
        """
        if session.status == "completed":
            return True
        if session.status in ("failed", "cancelled"):
            return False
        # Fallback guard for sessions left at "pending" by an older SDK
        # version that recorded trials but never called finalize_session.
        # Any canonical StopReason value signals the optimization is done.
        if session.completed_trials > 0:
            stop_reason = (session.metadata or {}).get("stop_reason")
            if stop_reason in (
                # All canonical StopReason values from traigent/api/types.py
                "max_trials_reached",
                "max_samples_reached",
                "timeout",
                "cost_limit",
                "metric_limit",
                "optimizer",
                "plateau",
                "convergence",
                "user_cancelled",
                "condition",
                "error",
                "vendor_error",
                "network_error",
                # Legacy alias retained for backward-compat with pre-1.0 sessions
                "budget_exhausted",
            ):
                return True
        return False

    def get_sync_status(self) -> dict[str, Any]:
        """Get status of local sessions and cloud-sync state.

        Reports how many completed runs are already synced vs. still pending,
        wandb-style, by reading each session's persisted ``sync_state``.
        """
        sessions = self.storage.list_sessions()
        # Include sessions that are definitively finished even if stuck at
        # "pending" due to the pre-fix #1344 status-persistence bug.
        completed_sessions = [s for s in sessions if self._is_finished_session(s)]

        total_trials = sum(s.completed_trials for s in sessions)

        # Per-status counts across completed (sync-eligible) sessions.
        counts = {"unsynced": 0, "synced": 0, "partial": 0, "failed": 0}
        for session in completed_sessions:
            counts[self._session_sync_status(session)] += 1
        # Anything not already fully synced is eligible to (re)sync.
        pending = counts["unsynced"] + counts["partial"] + counts["failed"]

        sync_status = {
            "total_sessions": len(sessions),
            "completed_sessions": len(completed_sessions),
            "total_trials": total_trials,
            "synced": counts["synced"],
            "unsynced": counts["unsynced"],
            "partial": counts["partial"],
            "failed": counts["failed"],
            # Backwards-compatible field: count still pending an upload.
            "sync_eligible": pending,
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
        opt_config = session.optimization_config or {}
        search_space = opt_config.get("search_space", {})
        if not isinstance(search_space, dict):
            search_space = {}
        # The experiment-run create validator requires a top-level
        # `infrastructure` key inside `configurations` (presence-checked); the
        # canonical shape is {infrastructure, parameters}.
        configurations = {"infrastructure": {}, "parameters": search_space}

        configuration_runs = self._convert_trials_to_configuration_runs(
            session.trials or []
        )
        measures = self._derive_experiment_measures(configuration_runs)

        # Create minimal agent definition accepted by POST /agents.
        agent_data = {
            "name": f"Local Agent: {session.function_name}",
            "agent_type_id": DEFAULT_SYNC_AGENT_TYPE_ID,
        }

        # The backend benchmark route is the dataset source for experiments.
        benchmark_data = {
            "name": sanitize_backend_name(f"Local Dataset {session.function_name}"),
            "type": "input-output",
            "label": sanitize_backend_name(
                f"{session.function_name} eval", fallback="Local Eval"
            ),
        }

        # agent_id and dataset_id are filled with returned backend IDs at sync time.
        experiment_data = {
            "name": f"Local Import: {session.function_name}",
            "measures": measures,
            "configurations": configurations,
            "status": "COMPLETED",
        }

        # experiment_data is filled with returned backend IDs at sync time.
        experiment_run_data = {
            "measures": measures,
            "configurations": configurations,
        }

        return {
            "agent": agent_data,
            "benchmark": benchmark_data,
            "experiment": experiment_data,
            "experiment_run": experiment_run_data,
            "configuration_runs": configuration_runs,
        }

    def _convert_trials_to_configuration_runs(
        self, trials: list[Any]
    ) -> list[dict[str, Any]]:
        """Reshape local trial results into configuration-run create payloads."""
        configuration_runs: list[dict[str, Any]] = []

        for result in self._convert_trials_to_results(trials):
            measures = self._configuration_run_measures(result)
            configuration_runs.append(
                {
                    "experiment_parameters": result.get("experiment_parameters", {}),
                    "measures": measures,
                    "status": "COMPLETED",
                }
            )

        return configuration_runs

    def _configuration_run_measures(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build measures for the backend configuration-run schema."""
        measures: dict[str, Any] = {}

        metadata = result.get("metadata")
        if isinstance(metadata, Mapping):
            for key, value in metadata.items():
                if self._is_numeric(value):
                    measures[key] = value

        for key, value in (result.get("measures") or {}).items():
            if key in TRIAL_COST_FIELDS:
                if self._is_numeric(value):
                    measures[key] = float(value)
            elif value is not None:
                measures[key] = value

        return measures

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Return True for JSON numeric values while excluding booleans."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _derive_experiment_measures(
        configuration_runs: list[dict[str, Any]],
    ) -> list[str]:
        """Derive the experiment-level measure list from trial payloads."""
        measures: list[str] = []
        for configuration_run in configuration_runs:
            for measure_name in configuration_run.get("measures", {}):
                if measure_name not in measures:
                    measures.append(measure_name)

        return measures or ["score"]

    def _convert_trials_to_results(self, trials: list[Any]) -> list[dict[str, Any]]:
        """Convert local trials to Traigent configuration_run format."""
        results = []

        for trial in trials:
            trial_cost_payload = dict(trial.metadata or {})
            for field in TRIAL_COST_FIELDS:
                field_value = getattr(trial, field, None)
                if field_value is not None:
                    trial_cost_payload[field] = field_value
            cost_measures = {
                field: value
                for field, value in extract_trial_cost_fields(
                    trial_cost_payload
                ).items()
                if value is not None
            }

            # Build measures from real per-objective metrics first so that
            # user-provided values (e.g. "accuracy", "latency") are never
            # clobbered by the composite score alias.  Per-objective metrics
            # are stored by the evaluator under metadata["all_metrics"].  Cost
            # fields come last so they also don't overwrite objective metrics
            # that happen to share a name with a cost field.
            all_metrics: dict[str, Any] = (trial.metadata or {}).get(
                "all_metrics"
            ) or {}
            trial_metrics = {
                k: v for k, v in all_metrics.items() if self._is_numeric(v)
            }
            result = {
                "id": f"config_run_{trial.trial_id}",
                "trial_id": trial.trial_id,
                "experiment_parameters": trial.config,
                "measures": {
                    **trial_metrics,
                    "score": trial.score,
                    **cost_measures,
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

    @staticmethod
    def _compute_payload_hash(traigent_data: dict[str, Any]) -> str:
        """Stable fingerprint of a session's syncable content.

        Used as the idempotency key: re-syncing an unchanged session is a no-op
        (so we never create duplicate cloud experiments), while a changed
        session is detected and re-synced.
        """
        canonical = json.dumps(traigent_data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def sync_session_to_cloud(
        self, session_id: str, dry_run: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """
        Sync a single session to cloud.

        Idempotent: a session already synced with identical content is skipped
        (status ``already_synced``) so re-running ``traigent sync`` never creates
        duplicate cloud experiments. Pass ``force=True`` to re-upload anyway.

        Args:
            session_id: Local session ID to sync
            dry_run: If True, only validate data without uploading
            force: If True, re-upload even when an unchanged sync already exists

        Returns:
            Sync result with status and details
        """
        # Serialize per-session sync so two concurrent `traigent sync` processes
        # can't both pass the "unsynced" check and create duplicate cloud
        # resources. We re-read state inside the lock so we observe another
        # process's just-written marker.
        with self.storage.acquire_lock(f"sync_{session_id}"):
            session = self.storage.load_session(session_id)
            if not session:
                raise TraigentStorageError(f"Session {session_id} not found") from None

            # Convert to Traigent format
            traigent_data = self.convert_session_to_traigent_format(session)
            payload_hash = self._compute_payload_hash(traigent_data)
            prior_state = dict(session.sync_state or {})
            already_synced = (
                prior_state.get("status") == "synced"
                and prior_state.get("payload_hash") == payload_hash
            )

            sync_result: dict[str, Any] = {
                "session_id": session_id,
                "function_name": session.function_name,
                "status": "success",
                "dry_run": dry_run,
                "data_converted": True,
                "cloud_experiment_id": prior_state.get("cloud_experiment_id"),
                "trials_converted": len(traigent_data["configuration_runs"]),
                "payload_hash": payload_hash,
                "errors": [],
            }

            # Idempotency: an unchanged, already-synced session is a no-op.
            if already_synced and not force:
                sync_result["status"] = "already_synced"
                sync_result["cloud_url"] = prior_state.get("cloud_url")
                sync_result["cloud_experiment_id"] = prior_state.get(
                    "cloud_experiment_id"
                )
                return sync_result

            if dry_run:
                # Preserve the legacy "success" status for an un-synced
                # validation run; only flag the already-synced case distinctly.
                sync_result["status"] = (
                    "already_synced" if already_synced else "success"
                )
                sync_result["preview"] = {
                    "experiment_name": traigent_data["experiment"]["name"],
                    "agent_name": traigent_data["agent"]["name"],
                    "benchmark_name": traigent_data["benchmark"]["name"],
                    "trial_count": len(traigent_data["configuration_runs"]),
                    "best_score": session.best_score,
                    "already_synced": already_synced,
                }
                return sync_result

            # Actually sync to backend.
            if not self.api_key:
                msg = (
                    "No API key provided for backend sync. " + get_no_credentials_hint()
                )
                logger.warning(msg)
                sync_result["status"] = "error"
                sync_result["errors"].append(msg)
                return sync_result

            # Resume (reuse cloud ids) only when re-trying the SAME content
            # after a partial/failed attempt — that is what prevents duplicate
            # experiments. A changed run, or --force, starts a fresh experiment.
            resume = (
                prior_state.get("payload_hash") == payload_hash
                and prior_state.get("status") in {"partial", "failed"}
                and not force
            )
            try:
                self._upload_session(
                    traigent_data, sync_result, prior_state, resume=resume
                )
            except Exception as e:
                sync_result["status"] = "error"
                sync_result["errors"].append(f"Sync failed: {str(e)}")
                logger.error(f"Failed to sync session {session_id}: {e}")

            # Record the outcome on EVERY non-trivial exit (success, partial, or
            # error) so a retry can reuse the cloud ids a prior attempt created
            # instead of minting a fresh, duplicate experiment.
            self._record_sync_state(session_id, sync_result, payload_hash)
            return sync_result

    def _upload_session(
        self,
        traigent_data: dict[str, Any],
        sync_result: dict[str, Any],
        prior_state: dict[str, Any],
        *,
        resume: bool,
    ) -> None:
        """Upload a converted session to the backend resource endpoints.

        When ``resume`` is True, reuses any cloud ids a prior attempt already
        created so a retry resumes rather than minting a duplicate experiment.
        Mutates ``sync_result`` in place and returns early on a failed step.
        """
        reuse = resume
        configuration_runs = traigent_data["configuration_runs"]
        total_steps = 4 + len(configuration_runs)
        successful_steps = 0
        project_id = prior_state.get("project_id") if reuse else None
        tenant_id = prior_state.get("tenant_id") if reuse else None
        sync_result["project_id"] = project_id
        sync_result["tenant_id"] = tenant_id

        # Agent + benchmark dedup by name; reuse a saved id when present.
        agent_id = prior_state.get("cloud_agent_id") if reuse else None
        if not agent_id:
            agent_result = self._sync_agent(traigent_data["agent"])
            if not agent_result["success"]:
                sync_result["status"] = "error"
                sync_result["errors"].append(
                    f"Agent sync failed: {agent_result['error']}"
                )
                return
            agent_id = agent_result["agent_id"]
        sync_result["cloud_agent_id"] = agent_id
        successful_steps += 1

        benchmark_id = prior_state.get("cloud_benchmark_id") if reuse else None
        if not benchmark_id:
            benchmark_result = self._sync_benchmark(traigent_data["benchmark"])
            if not benchmark_result["success"]:
                sync_result["status"] = "partial"
                sync_result["errors"].append(
                    f"Benchmark sync failed: {benchmark_result['error']}"
                )
                return
            benchmark_id = benchmark_result["benchmark_id"]
        sync_result["cloud_benchmark_id"] = benchmark_id
        successful_steps += 1

        # The experiment is NOT name-deduped by the backend, so reusing a saved
        # id is what prevents duplicate experiments on a retry (BLOCKER fix).
        experiment_id = prior_state.get("cloud_experiment_id") if reuse else None
        if not experiment_id:
            experiment_payload = self._build_experiment_payload(
                traigent_data["experiment"], agent_id, benchmark_id
            )
            experiment_result = self._sync_experiment(experiment_payload)
            if not experiment_result["success"]:
                sync_result["status"] = "partial"
                sync_result["errors"].append(
                    f"Experiment sync failed: {experiment_result['error']}"
                )
                return
            experiment_id = experiment_result["experiment_id"]
            project_id = experiment_result.get("project_id")
            tenant_id = experiment_result.get("tenant_id")
        sync_result["cloud_experiment_id"] = experiment_id
        sync_result["project_id"] = project_id
        sync_result["tenant_id"] = tenant_id
        successful_steps += 1

        experiment_run_id = (
            prior_state.get("cloud_experiment_run_id") if reuse else None
        )
        if not experiment_run_id:
            run_payload = self._build_experiment_run_payload(
                traigent_data["experiment_run"], agent_id, benchmark_id
            )
            run_result = self._sync_experiment_run(experiment_id, run_payload)
            if not run_result["success"]:
                sync_result["status"] = "partial"
                sync_result["errors"].append(
                    f"Experiment run sync failed: {run_result['error']}"
                )
                return
            experiment_run_id = run_result["experiment_run_id"]
        sync_result["cloud_experiment_run_id"] = experiment_run_id
        successful_steps += 1

        configuration_result = self._sync_configuration_runs(
            experiment_run_id, configuration_runs
        )
        successful_steps += configuration_result["synced"]
        if not configuration_result["success"]:
            sync_result["errors"].extend(
                f"Configuration run sync failed: {error}"
                for error in configuration_result["errors"]
            )

        if successful_steps == 0:
            sync_result["status"] = "error"  # Complete failure
        elif successful_steps == total_steps:
            sync_result["status"] = "success"
            sync_result["cloud_url"] = build_experiment_url(
                BackendConfig.get_cloud_backend_url(),
                experiment_id,
                project_id=project_id,
                tenant_id=tenant_id,
            )
        else:
            sync_result["status"] = "partial"  # Some steps succeeded

    def _record_sync_state(
        self, session_id: str, sync_result: dict[str, Any], payload_hash: str
    ) -> None:
        """Persist the outcome of a sync attempt onto the local session.

        Records a durable ``synced`` marker (with the cloud ids + payload hash)
        so a later ``traigent sync`` is idempotent, and a ``partial``/``failed``
        marker otherwise so status reporting reflects reality. Best-effort:
        never let a bookkeeping failure mask the sync result.
        """
        status = sync_result.get("status")
        marker_status = {
            "success": "synced",
            "partial": "partial",
            "error": "failed",
        }.get(str(status), "failed")

        patch: dict[str, Any] = {
            "status": marker_status,
            "source": "offline_sync",
            "payload_hash": payload_hash,
            # Persist all cloud ids so a retry of a partial sync reuses them
            # instead of creating duplicate resources.
            "cloud_agent_id": sync_result.get("cloud_agent_id"),
            "cloud_benchmark_id": sync_result.get("cloud_benchmark_id"),
            "cloud_experiment_id": sync_result.get("cloud_experiment_id"),
            "cloud_experiment_run_id": sync_result.get("cloud_experiment_run_id"),
            "project_id": sync_result.get("project_id"),
            "tenant_id": sync_result.get("tenant_id"),
            "cloud_url": sync_result.get("cloud_url"),
            "synced_at": datetime.now(UTC).isoformat(),
            "last_error": "; ".join(sync_result.get("errors", [])) or None,
        }
        prior_attempts = 0
        existing = self.storage.load_session(session_id)
        prior_sync_state = getattr(existing, "sync_state", None)
        if isinstance(prior_sync_state, dict):
            try:
                prior_attempts = int(prior_sync_state.get("attempts", 0) or 0)
            except (TypeError, ValueError):
                prior_attempts = 0
        patch["attempts"] = prior_attempts + 1

        try:
            self.storage.update_sync_state(session_id, patch)
        except Exception as exc:  # pragma: no cover - bookkeeping must not break sync
            logger.warning(
                "Failed to persist sync_state for session %s: %s", session_id, exc
            )

    def sync_all_sessions(
        self, dry_run: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """
        Sync all eligible local sessions to cloud.

        Args:
            dry_run: If True, only validate data without uploading
            force: If True, re-upload even runs already synced unchanged

        Returns:
            Overall sync result
        """
        sessions = self.storage.list_sessions()
        # Include sessions that are definitively finished even if stuck at
        # "pending" due to the pre-fix #1344 status-persistence bug.
        completed_sessions = [s for s in sessions if self._is_finished_session(s)]

        session_results: list[dict[str, Any]] = []
        synced_successfully = 0
        skipped = 0
        sync_errors = 0
        overall_status = "success"

        for session in completed_sessions:
            try:
                result = self.sync_session_to_cloud(
                    session.session_id, dry_run, force=force
                )
                session_results.append(result)

                status = result.get("status")
                if status == "success":
                    synced_successfully += 1
                elif status in ("already_synced", "ready"):
                    # Idempotent no-op: an unchanged, already-synced run.
                    skipped += 1
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
            "skipped": skipped,
            "sync_errors": sync_errors,
            "dry_run": dry_run,
            "session_results": session_results,
            "overall_status": overall_status,
        }

    def _sync_agent(self, agent_data: dict[str, Any]) -> dict[str, Any]:
        """Create or reuse the local-import agent by stable name."""
        self._raise_if_backend_egress_disabled("sync agent")
        try:
            existing_agent = self._find_existing_by_name(
                "agents", agent_data["name"], "agents"
            )
            if existing_agent:
                agent_id = self._extract_resource_id(existing_agent)
                if agent_id:
                    return {"success": True, "agent_id": agent_id, "reused": True}

            response = self._session.post(
                f"{self.base_url}/agents",
                json=agent_data,
                timeout=self._request_timeout,
            )

            if response.status_code == 429:
                existing_agent = self._find_existing_by_name(
                    "agents", agent_data["name"], "agents"
                )
                if existing_agent:
                    agent_id = self._extract_resource_id(existing_agent)
                    if agent_id:
                        return {
                            "success": True,
                            "agent_id": agent_id,
                            "reused": True,
                        }

            if response.status_code in [200, 201]:
                agent_id = self._extract_response_id(response)
                if not agent_id:
                    return {
                        "success": False,
                        "error": "Agent create response did not include id",
                    }
                return {"success": True, "agent_id": agent_id, "reused": False}

            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_benchmark(self, benchmark_data: dict[str, Any]) -> dict[str, Any]:
        """Create or reuse the benchmark by stable name."""
        self._raise_if_backend_egress_disabled("sync benchmark")
        try:
            existing_benchmark = self._find_existing_by_name(
                "datasets", benchmark_data["name"], "datasets"
            )
            if existing_benchmark:
                benchmark_id = self._extract_resource_id(existing_benchmark)
                if benchmark_id:
                    return {
                        "success": True,
                        "dataset_id": benchmark_id,
                        "benchmark_id": benchmark_id,
                        "reused": True,
                    }

            response = self._session.post(
                f"{self.base_url}/datasets",
                json=benchmark_data,
                timeout=self._request_timeout,
            )

            if response.status_code in [409, 500]:
                existing_benchmark = self._find_existing_by_name(
                    "datasets", benchmark_data["name"], "datasets"
                )
                if existing_benchmark:
                    benchmark_id = self._extract_resource_id(existing_benchmark)
                    if benchmark_id:
                        return {
                            "success": True,
                            "dataset_id": benchmark_id,
                            "benchmark_id": benchmark_id,
                            "reused": True,
                        }

            if response.status_code in [200, 201]:
                benchmark_id = self._extract_response_id(response)
                if not benchmark_id:
                    return {
                        "success": False,
                        "error": "Benchmark create response did not include id",
                    }
                return {
                    "success": True,
                    "dataset_id": benchmark_id,
                    "benchmark_id": benchmark_id,
                    "reused": False,
                }

            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_experiment(self, experiment_data: dict[str, Any]) -> dict[str, Any]:
        """Sync experiment data to cloud."""
        self._raise_if_backend_egress_disabled("sync experiment")
        try:
            response = self._session.post(
                f"{self.base_url}/experiments",
                json=experiment_data,
                timeout=self._request_timeout,
            )
            if response.status_code in [200, 201]:
                payload = self._response_json(response)
                experiment_id = (
                    self._extract_resource_id(payload)
                    if isinstance(payload, Mapping)
                    else None
                )
                if not experiment_id:
                    return {
                        "success": False,
                        "error": "Experiment create response did not include id",
                    }
                context = (
                    self._extract_experiment_context(payload)
                    if isinstance(payload, Mapping)
                    else {}
                )
                return {"success": True, "experiment_id": experiment_id, **context}
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_experiment_run(
        self, experiment_id: str, run_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Sync experiment run data to cloud."""
        self._raise_if_backend_egress_disabled("sync experiment run")
        try:
            if not experiment_id:
                return {
                    "success": False,
                    "error": "Experiment run sync requires experiment_id",
                }

            response = self._session.post(
                f"{self.base_url}/experiment-runs/{experiment_id}/runs",
                json=run_data,
                timeout=self._request_timeout,
            )
            if response.status_code in [200, 201]:
                experiment_run_id = self._extract_response_id(response)
                if not experiment_run_id:
                    return {
                        "success": False,
                        "error": "Experiment-run create response did not include id",
                    }
                return {
                    "success": True,
                    "run_id": experiment_run_id,
                    "experiment_run_id": experiment_run_id,
                }
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sync_configuration_runs(
        self, experiment_run_id: str, trials: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create the cost-bearing configuration rows for an experiment run."""
        self._raise_if_backend_egress_disabled("sync configuration runs")
        errors: list[str] = []
        configuration_run_ids: list[str] = []
        synced = 0

        for index, trial in enumerate(trials, start=1):
            try:
                response = self._session.post(
                    f"{self.base_url}/configuration-runs/runs/"
                    f"{experiment_run_id}/configurations",
                    json=trial,
                    timeout=self._request_timeout,
                )
            except Exception as exc:
                errors.append(f"trial {index}: {exc}")
                continue

            if response.status_code == 201:
                synced += 1
                configuration_run_id = self._extract_response_id(response)
                if configuration_run_id:
                    configuration_run_ids.append(configuration_run_id)
                continue

            errors.append(
                f"trial {index}: HTTP {response.status_code}: {response.text}"
            )

        return {
            "success": not errors,
            "synced": synced,
            "errors": errors,
            "configuration_run_ids": configuration_run_ids,
        }

    @staticmethod
    def _build_experiment_payload(
        experiment_data: dict[str, Any], agent_id: str, benchmark_id: str
    ) -> dict[str, Any]:
        """Fill returned backend IDs into the experiment create payload."""
        return {
            **experiment_data,
            "agent_id": agent_id,
            "dataset_id": benchmark_id,
        }

    @staticmethod
    def _build_experiment_run_payload(
        run_data: dict[str, Any], agent_id: str, benchmark_id: str
    ) -> dict[str, Any]:
        """Build the closed-schema experiment-run payload."""
        return {
            "experiment_data": {
                "agent_id": agent_id,
                "benchmark_id": benchmark_id,
                "measures": run_data["measures"],
                "configurations": run_data["configurations"],
            }
        }

    def _find_existing_by_name(
        self, resource_path: str, name: str, collection_key: str
    ) -> dict[str, Any] | None:
        """Find an existing backend resource by exact name."""
        self._raise_if_backend_egress_disabled("sync lookup")
        response = self._session.get(
            f"{self.base_url}/{resource_path}",
            params={"name": name},
            timeout=self._request_timeout,
        )
        if response.status_code != 200:
            return None

        payload = self._response_json(response)
        for item in self._iter_response_items(payload, collection_key):
            if item.get("name") == name:
                return item

        return None

    def _iter_response_items(
        self, payload: Any, collection_key: str
    ) -> Iterable[dict[str, Any]]:
        """Yield resource dicts from common list-response envelope shapes."""
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return

        if not isinstance(payload, dict):
            return

        if "name" in payload and self._extract_resource_id(payload):
            yield payload

        for key in (collection_key, "items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(value, dict):
                yield from self._iter_response_items(value, collection_key)

    @staticmethod
    def _response_json(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _extract_resource_id(payload: Mapping[str, Any]) -> str | None:
        for key in (
            "id",
            "agent_id",
            "benchmark_id",
            "dataset_id",
            "experiment_id",
            "experiment_run_id",
            "run_id",
        ):
            value = payload.get(key)
            if value is not None:
                return str(value)
        # Backend responses wrap the resource in the standard envelope
        # {success, data: {id, ...}, message}; unwrap one level.
        data = payload.get("data")
        if isinstance(data, Mapping):
            return SyncManager._extract_resource_id(data)
        return None

    def _extract_response_id(self, response: requests.Response) -> str | None:
        payload = self._response_json(response)
        if isinstance(payload, Mapping):
            return self._extract_resource_id(payload)
        return None

    @staticmethod
    def _extract_context_value(payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is not None:
            normalized = str(value).strip()
            return normalized or None

        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key)
            if value is not None:
                normalized = str(value).strip()
                return normalized or None

        data = payload.get("data")
        if isinstance(data, Mapping):
            return SyncManager._extract_context_value(data, key)
        return None

    @staticmethod
    def _extract_experiment_context(payload: Mapping[str, Any]) -> dict[str, str]:
        context: dict[str, str] = {}
        for key in ("project_id", "tenant_id"):
            value = SyncManager._extract_context_value(payload, key)
            if value:
                context[key] = value
        return context

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

        # When backups are requested, only sessions whose backup succeeded are
        # eligible for deletion — never delete a run we failed to back up.
        deletable: list[str] = list(session_ids)
        if keep_backup:
            backup_dir = (
                self.storage.storage_path
                / "backups"
                / f"sync_backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir.mkdir(parents=True, exist_ok=True)

            backed_up: list[str] = []
            for session_id in session_ids:
                try:
                    backup_success = self.storage.export_session(
                        session_id, str(backup_dir / f"{session_id}.json"), "json"
                    )
                    if backup_success:
                        sessions_backed_up += 1
                        backed_up.append(session_id)
                    else:
                        errors.append(
                            f"Backup failed for {session_id}: export returned False; "
                            "skipping deletion"
                        )
                except Exception as e:
                    errors.append(
                        f"Backup failed for {session_id}: {e}; skipping deletion"
                    )
            deletable = backed_up

        # Delete sessions (only those safely backed up when backups are on).
        for session_id in deletable:
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
