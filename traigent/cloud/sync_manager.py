"""
Local-to-backend sync manager for Traigent optimization sessions.
Handles migration of local data to Traigent backend when users upgrade.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlencode

import requests  # Always needed for synchronous operations

from traigent.cloud.api_operations import _typed_configuration_space
from traigent.cloud.client import raise_if_cloud_egress_disabled
from traigent.cloud.url_security import validate_cloud_base_url

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

_BACKEND_NAME_DISALLOWED = re.compile(r"[^a-zA-Z0-9 _-]+")
_BACKEND_NAME_SPACES = re.compile(r"\s+")


def build_experiment_url(
    base_url: str,
    experiment_id: str,
    *,
    run_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> str:
    """Build the portal URL for an experiment.

    Single source of truth for experiment URL construction —
    used by both SyncManager and the orchestrator.
    """
    url = (
        f"{base_url.rstrip('/')}/experiments/view/{quote(str(experiment_id), safe='')}"
    )
    query_params = {
        key: normalized
        for key, value in (
            ("run_id", run_id),
            ("project_id", project_id),
            ("tenant_id", tenant_id),
        )
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
        backend_api_url = BackendConfig.get_cloud_api_url().rstrip("/")
        self.base_url = validate_cloud_base_url(backend_api_url, purpose="sync request")

        # Setup HTTP client - always create a requests session for sync operations
        # X-API-Key only — Authorization: Bearer is reserved for JWTs.
        self.headers = {"X-API-Key": api_key} if api_key else {}
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
        Convert a local session to the content-free typed-session payload.

        Offline sync imports historical runs through the same content-free
        typed-session endpoints the live SDK uses (``POST /sessions`` ->
        per-trial ``POST /sessions/{id}/results`` -> ``POST
        /sessions/{id}/finalize``).  The session is created with
        ``tracking_mode=native_local`` and binds NO benchmark, so the backend's
        empty-dataset guard hits its documented no-dataset pass-through and a run
        whose server-side dataset would have zero examples imports cleanly.  No
        prompt/output content egresses: the dataset is represented only by a
        content-free label (name + size).

        Args:
            session: Local optimization session

        Returns:
            Dict with a ``session_create`` payload plus ``configuration_runs``.
        """
        opt_config = session.optimization_config or {}
        search_space = opt_config.get("search_space", {})
        if not isinstance(search_space, dict):
            search_space = {}

        configuration_runs = self._convert_trials_to_configuration_runs(
            session.trials or []
        )
        objectives = self._derive_objectives(opt_config, configuration_runs)

        # Content-free dataset label: name + size only, never example content.
        # The typed path validates dataset_metadata.size as a positive int
        # whenever the key is present, so an unknown/zero local size is coerced
        # to 1 — exactly like the live SDK builder
        # (api_operations._build_typed_session_payload). The empty-dataset
        # pass-through comes from binding NO benchmark, not from the size field.
        dataset_size = opt_config.get("dataset_size")
        if (
            not isinstance(dataset_size, int)
            or isinstance(dataset_size, bool)
            or dataset_size <= 0
        ):
            dataset_size = 1
        dataset_name = sanitize_backend_name(f"Local Dataset {session.function_name}")
        evaluation_set = opt_config.get("evaluation_set") or "default"

        session_create = {
            "function_name": session.function_name,
            "configuration_space": _typed_configuration_space(search_space),
            "objectives": objectives,
            "dataset_metadata": {
                "size": dataset_size,
                "name": dataset_name,
                "privacy_mode": True,
            },
            "max_trials": max(len(configuration_runs), 1),
            # native_local lets the backend materialize each trial directly from
            # the submitted config (historical backfill, no /next-trial round
            # trip, no Optuna suggestion). It binds no benchmark, so the
            # EMPTY_DATASET guard never fires (empty-dataset sync fix).
            "optimization_strategy": {
                "algorithm": "optuna",
                "tracking_mode": "native_local",
            },
            "metadata": {
                "function_name": session.function_name,
                "evaluation_set": evaluation_set,
                "source": "offline_sync",
            },
        }

        return {
            "session_create": session_create,
            "configuration_runs": configuration_runs,
        }

    @staticmethod
    def _derive_objectives(
        opt_config: dict[str, Any], configuration_runs: list[dict[str, Any]]
    ) -> list[str]:
        """Derive objective names for the typed session create payload.

        Prefer the objectives recorded on the local optimization config; fall
        back to the measure names present on the trials (``["score"]`` at
        minimum). Content-free: only measure/objective *names* are emitted.
        """
        raw_objectives = opt_config.get("objectives")
        if isinstance(raw_objectives, (list, tuple)):
            names = [str(name) for name in raw_objectives if name]
            if names:
                return names
        return SyncManager._derive_experiment_measures(configuration_runs)

    def _convert_trials_to_configuration_runs(
        self, trials: list[Any]
    ) -> list[dict[str, Any]]:
        """Reshape local trial results into per-trial result payloads.

        Each entry carries ``trial_id`` (native_local requires it so the
        backend can create the trial from the submitted ``config``),
        ``experiment_parameters`` (the trial config), content-free numeric
        ``measures``, and the trial's REAL terminal status (``COMPLETED`` or
        ``FAILED`` — never masked to COMPLETED; the session results endpoint
        accepts failed trials).
        """
        configuration_runs: list[dict[str, Any]] = []

        for result in self._convert_trials_to_results(trials):
            measures = self._configuration_run_measures(result)
            experiment_parameters = result.get("experiment_parameters") or {}
            if not experiment_parameters:
                # native_local requires a non-empty config to materialize the
                # trial; fall back to a config recorded in the trial metadata
                # (if any) before giving up.
                metadata = result.get("metadata")
                if isinstance(metadata, Mapping):
                    metadata_config = metadata.get("config")
                    if isinstance(metadata_config, Mapping) and metadata_config:
                        experiment_parameters = dict(metadata_config)
            status = "FAILED" if result.get("status") == "failed" else "COMPLETED"
            configuration_runs.append(
                {
                    "trial_id": result.get("trial_id"),
                    "experiment_parameters": experiment_parameters,
                    "measures": measures,
                    "status": status,
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

    @staticmethod
    def _config_run_key(index: int) -> str:
        """Stable key identifying a configuration-run within a session's batch.

        Configuration runs are derived deterministically from
        ``session.trials`` in order (see
        ``_convert_trials_to_configuration_runs``), so the zero-based position
        in that list is a stable identity across sync attempts of the SAME
        content. We record which config-runs already uploaded under this key so
        a resume never re-POSTs them — the backend does NOT dedup config-run
        creates (backend issue #1330), so the SDK must be idempotent itself.
        """
        return f"cfg_{index}"

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
                # Dry-run validates the content-free typed-session payload the
                # real sync would submit (create session -> per-trial results ->
                # finalize). The session binds no benchmark, so an empty
                # server-side dataset never blocks the import; a completed local
                # session always predicts success.
                session_create = traigent_data["session_create"]
                dataset_metadata = session_create["dataset_metadata"]
                trial_count = len(traigent_data["configuration_runs"])

                if already_synced:
                    sync_result["status"] = "already_synced"
                else:
                    sync_result["status"] = "success"
                sync_result["preview"] = {
                    "function_name": session_create["function_name"],
                    "dataset_name": dataset_metadata.get("name"),
                    "dataset_size": dataset_metadata.get("size"),
                    "trial_count": trial_count,
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
                    session_id,
                    traigent_data,
                    sync_result,
                    prior_state,
                    resume=resume,
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
        session_id: str,
        traigent_data: dict[str, Any],
        sync_result: dict[str, Any],
        prior_state: dict[str, Any],
        *,
        resume: bool,
    ) -> None:
        """Import a converted session through the content-free session endpoints.

        Three steps: create a native_local typed session (no benchmark, so the
        backend's EMPTY_DATASET guard never fires), submit one result
        per trial, then finalize. When ``resume`` is True, reuses any cloud ids
        a prior attempt already created so a retry resumes rather than minting a
        duplicate session. Mutates ``sync_result`` in place and returns early on
        a failed step.
        """
        reuse = resume
        configuration_runs = traigent_data["configuration_runs"]
        total_steps = 2 + len(configuration_runs)
        successful_steps = 0
        project_id = prior_state.get("project_id") if reuse else None
        tenant_id = prior_state.get("tenant_id") if reuse else None
        sync_result["project_id"] = project_id
        sync_result["tenant_id"] = tenant_id

        # Step 1: create the typed session. The session is NOT name-deduped by
        # the backend, so reusing a saved id on resume is what prevents a
        # duplicate session/experiment.
        session_cloud_id = prior_state.get("cloud_session_id") if reuse else None
        experiment_id = prior_state.get("cloud_experiment_id") if reuse else None
        experiment_run_id = (
            prior_state.get("cloud_experiment_run_id") if reuse else None
        )
        if not session_cloud_id:
            create_result = self._sync_create_session(traigent_data["session_create"])
            if not create_result["success"]:
                sync_result["status"] = "error"
                sync_result["errors"].append(
                    f"Session create failed: {create_result['error']}"
                )
                return
            session_cloud_id = create_result["session_id"]
            experiment_id = create_result["experiment_id"]
            experiment_run_id = create_result["experiment_run_id"]
            project_id = create_result.get("project_id")
            tenant_id = create_result.get("tenant_id")
        sync_result["cloud_session_id"] = session_cloud_id
        sync_result["cloud_experiment_id"] = experiment_id
        sync_result["cloud_experiment_run_id"] = experiment_run_id
        sync_result["project_id"] = project_id
        sync_result["tenant_id"] = tenant_id
        successful_steps += 1

        # Step 2: submit each completed trial result.
        # Resume idempotency: on a retry of the SAME content we must NOT re-POST
        # trial results a prior attempt already submitted — the backend does not
        # dedup other creates, so a re-POST silently duplicates rows. We persist
        # each result as synced (keyed by its stable position) immediately after
        # its POST succeeds, and on resume we skip the ones already recorded.
        already_synced_keys: set[str] = set()
        if reuse:
            prior_trials = prior_state.get("trials")
            if isinstance(prior_trials, dict):
                already_synced_keys = {
                    key
                    for key, record in prior_trials.items()
                    if isinstance(record, dict) and record.get("status") == "synced"
                }

        def _record_config_run_synced(
            key: str, configuration_run_id: str | None
        ) -> None:
            """Persist one result as synced right after its POST succeeds.

            Crash-safe: written incrementally (not batched at the end) so a
            crash/failure mid-batch still leaves a durable marker that lets the
            next resume skip the already-uploaded rows.
            """
            try:
                self.storage.update_sync_state(
                    session_id,
                    {},
                    trial_updates={
                        key: {
                            "status": "synced",
                            "cloud_configuration_run_id": configuration_run_id,
                        }
                    },
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - bookkeeping must not break sync
                logger.warning(
                    "Failed to persist result progress (%s) for session %s: %s",
                    key,
                    session_id,
                    exc,
                )

        configuration_result = self._sync_session_results(
            session_cloud_id,
            configuration_runs,
            already_synced_keys=already_synced_keys,
            on_synced=_record_config_run_synced,
        )
        # Both freshly-synced AND already-synced (skipped) results count as done,
        # so a fully-resumed session can still reach total_steps and be finalized.
        successful_steps += configuration_result["synced"]
        successful_steps += configuration_result["skipped"]
        if not configuration_result["success"]:
            sync_result["errors"].extend(
                f"Result sync failed: {error}"
                for error in configuration_result["errors"]
            )

        # Step 3: finalize the session now that all results are submitted. We
        # finalize only on a clean upload (all results synced); a partial upload
        # stays open so a retry can resume and finish it.
        if configuration_result["success"]:
            finalize_result = self._sync_finalize_session(
                session_cloud_id, experiment_run_id
            )
            finalization_status = finalize_result.get("classification")
            if finalization_status:
                sync_result["finalization_status"] = finalization_status
            current_status = finalize_result.get("current_status")
            if current_status:
                sync_result["finalization_current_status"] = current_status
            if finalize_result.get("skipped"):
                sync_result.setdefault("warnings", []).append(
                    f"Session finalization skipped: {finalize_result['message']}"
                )
                successful_steps += 1
            elif finalize_result["success"]:
                successful_steps += 1
            else:
                sync_result["errors"].append(
                    f"Session finalization failed: {finalize_result['error']}"
                )
                # Successful result uploads but failed finalization is partial.
                if successful_steps == total_steps - 1:
                    sync_result["status"] = "partial"
                    return

        if successful_steps == 0:
            sync_result["status"] = "error"  # Complete failure
        elif successful_steps == total_steps and not sync_result.get("errors"):
            sync_result["status"] = "success"
            sync_result["cloud_url"] = build_experiment_url(
                BackendConfig.get_cloud_web_url(),
                experiment_id,
                run_id=experiment_run_id,
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
            # instead of creating a duplicate session/experiment.
            "cloud_session_id": sync_result.get("cloud_session_id"),
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

    def _sync_create_session(self, session_create: dict[str, Any]) -> dict[str, Any]:
        """Create the content-free native_local typed session.

        POSTs the typed contract to ``/sessions`` (content-free: function name,
        typed configuration space, objective names, and a dataset *label* only).
        Because the session binds no benchmark, the backend's EMPTY_DATASET
        guard hits its no-dataset pass-through, so a run whose server-side
        dataset would have zero examples imports cleanly. Parses the response
        like ``api_operations._parse_session_response`` (experiment_id /
        experiment_run_id fall back to session_id when absent).
        """
        self._raise_if_backend_egress_disabled("sync session create")
        try:
            response = self._session.post(
                f"{self.base_url}/sessions",
                json=session_create,
                timeout=self._request_timeout,
            )
            if response.status_code not in (200, 201):
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
            payload = self._response_json(response)
            if not isinstance(payload, Mapping):
                return {
                    "success": False,
                    "error": "Session create response was not a JSON object",
                }
            session_id = payload.get("session_id")
            if not session_id:
                return {
                    "success": False,
                    "error": "Session create response did not include session_id",
                }
            session_id = str(session_id)
            metadata = payload.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            experiment_id = str(metadata.get("experiment_id") or session_id)
            experiment_run_id = str(metadata.get("experiment_run_id") or session_id)
            project_id = self._optional_context_id(
                payload.get("project_id") or metadata.get("project_id")
            )
            tenant_id = self._optional_context_id(
                payload.get("tenant_id") or metadata.get("tenant_id")
            )
            return {
                "success": True,
                "session_id": session_id,
                "experiment_id": experiment_id,
                "experiment_run_id": experiment_run_id,
                "project_id": project_id,
                "tenant_id": tenant_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _optional_context_id(value: Any) -> str | None:
        """Normalize an owning-context id (project/tenant) to a non-empty str."""
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _sync_session_results(
        self,
        session_id: str,
        configuration_runs: list[dict[str, Any]],
        *,
        already_synced_keys: set[str] | None = None,
        on_synced: Callable[[str, str | None], None] | None = None,
    ) -> dict[str, Any]:
        """Submit one trial result per configuration run.

        Idempotent across resumes: each result carries a stable key
        (``_config_run_key`` of its zero-based position). Runs whose key is in
        ``already_synced_keys`` are SKIPPED (not re-POSTed) -- the backend does
        not dedup result creates, so re-POSTing would duplicate rows.
        ``on_synced(key, result_id)`` is invoked immediately after each
        successful POST so progress is persisted incrementally (crash-safe),
        not only at the end of the batch. The per-trial payload is
        ``{trial_id, config, status, metrics}`` -- ``config`` is REQUIRED (and
        must be non-empty) in native_local because the backend materializes the
        trial from it; ``status`` is the trial's real terminal status
        (COMPLETED or FAILED).
        """
        self._raise_if_backend_egress_disabled("sync session results")
        already_synced_keys = already_synced_keys or set()
        errors: list[str] = []
        configuration_run_ids: list[str] = []
        synced = 0
        skipped = 0

        for index, configuration_run in enumerate(configuration_runs):
            key = self._config_run_key(index)
            # Skip results a prior attempt already uploaded so a resume never
            # creates duplicate rows.
            if key in already_synced_keys:
                skipped += 1
                continue

            # 1-based ordinal for human-readable error messages.
            ordinal = index + 1
            trial_config = configuration_run.get("experiment_parameters") or {}
            if not trial_config:
                # native_local rejects empty configs ("config is required for
                # native_local trial submissions"); surface a clear SDK-side
                # error instead of the raw backend message.
                errors.append(
                    f"trial {ordinal}: trial has an empty config, which the "
                    "content-free session import cannot submit (native_local "
                    "materializes each trial from its config); this trial was "
                    "skipped"
                )
                continue
            result_payload = {
                "trial_id": configuration_run.get("trial_id"),
                "config": trial_config,
                "status": configuration_run.get("status") or "COMPLETED",
                "metrics": configuration_run.get("measures", {}),
            }
            try:
                response = self._session.post(
                    f"{self.base_url}/sessions/{session_id}/results",
                    json=result_payload,
                    timeout=self._request_timeout,
                )
            except Exception as exc:
                errors.append(f"trial {ordinal}: {exc}")
                continue

            if response.status_code in (200, 201):
                synced += 1
                result_id = self._extract_response_id(response)
                if result_id:
                    configuration_run_ids.append(result_id)
                # Persist this result as synced RIGHT NOW (before the next POST)
                # so a crash/failure on a later row leaves a durable marker the
                # next resume can skip past.
                if on_synced is not None:
                    on_synced(key, result_id)
                continue

            errors.append(
                f"trial {ordinal}: HTTP {response.status_code}: {response.text}"
            )

        return {
            "success": not errors,
            "synced": synced,
            "skipped": skipped,
            "errors": errors,
            "configuration_run_ids": configuration_run_ids,
        }

    def _sync_finalize_session(
        self, session_id: str, experiment_run_id: str | None
    ) -> dict[str, Any]:
        """Finalize the typed session after all results are submitted.

        POSTs ``/sessions/{id}/finalize`` with a content-free reason and the
        experiment_run_id (no certified_selection -- this is a content-free
        historical import). Treats an already-finalized session as an
        idempotent success.
        """
        self._raise_if_backend_egress_disabled("finalize session")
        try:
            response = self._session.post(
                f"{self.base_url}/sessions/{session_id}/finalize",
                json={
                    "reason": "offline_sync_finalization",
                    "experiment_run_id": experiment_run_id,
                },
                timeout=self._request_timeout,
            )
            if response.status_code in (200, 201):
                return {"success": True, "classification": "completed"}
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
