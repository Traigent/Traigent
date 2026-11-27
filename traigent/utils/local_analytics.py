"""Privacy-safe Edge Analytics insights.

This module collects aggregated, non-sensitive usage statistics in Edge Analytics mode
to help users understand the value proposition of upgrading to cloud services.
All data sent is privacy-safe and contains no sensitive information.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Performance FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import uuid
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from traigent._version import get_version
from traigent.api.types import OptimizationStatus
from traigent.config.backend_config import (
    DEFAULT_CLOUD_URL,
    EndpointNotConfiguredError,
    endpoint_missing_message,
    is_strict_endpoint_mode,
)
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.relative_change import relative_change
from traigent.utils.send_policy import (
    _TelemetryAuthSuppressed,
    _usage_analytics_submission,
    may_send_telemetry,
)

# Import constant with fallback for base-only installs (cloud module not available)
try:
    from traigent.cloud.auth import MIN_TOKEN_LENGTH
except ModuleNotFoundError:
    MIN_TOKEN_LENGTH = 20  # Default minimum acceptable token length

# Note: "Unclosed client session" warnings from background analytics tasks
# are suppressed locally in _submit_analytics_background() using catch_warnings()
# to avoid process-wide suppression that could hide real leaks elsewhere.

logger = logging.getLogger(__name__)

# Wall-clock cap for the session scan done on the submit path. The scan runs in
# a worker thread so the orchestrator's wait_for can fire, and this budget makes
# the thread itself stop shortly after instead of reading every session file.
_SUBMIT_SCAN_BUDGET_SECONDS = 3.0
_FIRST_USE_FILENAME = ".first_use"
# Local session ids start with %Y%m%d_%H%M%S_%f (LocalStorageManager.create_session).
_SESSION_ID_TIME_FORMAT = "%Y%m%d_%H%M%S_%f"
_SESSION_ID_TIME_LENGTH = 22
# Extra wait for the collector thread beyond its own budget before giving up.
_SUBMIT_SCAN_MARGIN_SECONDS = 1.0
# Last collector thread; a stalled read must not be joined by new ones.
_COLLECTOR_LOCK = threading.Lock()
_COLLECTOR_THREAD: threading.Thread | None = None
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _reset_collector_after_fork() -> None:
    global _COLLECTOR_LOCK, _COLLECTOR_THREAD
    _COLLECTOR_LOCK = threading.Lock()
    _COLLECTOR_THREAD = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_collector_after_fork)

# Default analytics endpoint (deprecated - will use backend client instead)
DEFAULT_ANALYTICS_ENDPOINT = f"{DEFAULT_CLOUD_URL}/v1/local-usage"
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


def _average_config_space_size(completed_sessions: list[Any]) -> float:
    # Configuration space complexity (aggregated)
    config_space_sizes: list[int] = []
    for session in completed_sessions:
        if hasattr(session, "optimization_config") and session.optimization_config:
            config_space = session.optimization_config.get("configuration_space", {})
            if isinstance(config_space, dict):
                config_space_sizes.append(len(config_space))

    avg_config_space_size = (
        sum(config_space_sizes) / len(config_space_sizes) if config_space_sizes else 0
    )

    return avg_config_space_size


def _analytics_api_key_error(api_key: str | None) -> str | None:
    if not api_key:
        logger.debug("No Traigent API key found, analytics submission skipped")
        return "No API key available"

    if len(api_key) < MIN_TOKEN_LENGTH:
        logger.debug(
            "API key too short for analytics submission (length=%d), skipping",
            len(api_key),
        )
        return "API key invalid"

    return None


def usage_analytics_allowed(config: TraigentConfig) -> bool:
    """Single gate in front of every usage-analytics send (#3184).

    Refuses when ``enable_usage_analytics`` is False, backend egress is disabled
    (``no_egress``, ``offline=True`` or an offline policy / ``TRAIGENT_OFFLINE_MODE``), or the
    documented ``TRAIGENT_DISABLE_TELEMETRY`` opt-out is set. The default of
    ``enable_usage_analytics`` is unchanged.
    """
    from traigent.core.execution_policy_runtime import backend_egress_disabled

    if not config.enable_usage_analytics:
        return False
    if not may_send_telemetry() or getattr(config, "offline", False) is True:
        return False
    return not backend_egress_disabled(config)


async def _close_analytics_backend_client(
    backend_client_module: Any, backend_client: Any
) -> None:
    try:
        await backend_client.__aexit__(None, None, None)
    except Exception:
        pass  # Best effort cleanup
    finally:
        if getattr(backend_client_module, "_backend_client", None) is backend_client:
            backend_client_module._backend_client = None


def _analytics_trial_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_sessions": float(stats.get("total_sessions", 0)),
        "completed_sessions": float(stats.get("completed_sessions", 0)),
        "total_trials": float(stats.get("total_trials", 0)),
        "avg_trials_per_session": stats.get("avg_trials_per_session", 0.0),
        "avg_config_space_size": stats.get("avg_config_space_size", 0.0),
        "avg_improvement_percent": stats.get("avg_improvement_percent", 0.0) or 0.0,
        "sessions_last_30_days": float(stats.get("sessions_last_30_days", 0)),
        "days_since_first_use": float(stats.get("days_since_first_use", 0)),
    }


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
        # Strict-endpoint mode: never default to the vendor analytics host.
        self.analytics_endpoint = config.analytics_endpoint or (
            None if is_strict_endpoint_mode() else DEFAULT_ANALYTICS_ENDPOINT
        )
        self.enabled = config.enable_usage_analytics and (
            config.execution_mode == ExecutionMode.LOCAL.value
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

    @staticmethod
    def _session_id_time(name: str) -> datetime | None:
        """Creation time encoded in a local session id, or None if not parseable."""
        try:
            return datetime.strptime(
                name[:_SESSION_ID_TIME_LENGTH], _SESSION_ID_TIME_FORMAT
            ).replace(tzinfo=UTC)
        except ValueError:
            return None

    def _scan_sessions(self, time_budget: float) -> dict[str, Any]:
        """Budgeted scan: every step checks the deadline, so the caller's thread
        ends within ``time_budget`` plus at most one file read.

        Names are enumerated incrementally (``os.scandir``), then sessions are
        read newest-first (ids start with their creation time).
        """
        deadline = time.monotonic() + time_budget
        sessions_dir = Path(self.storage.storage_path) / "sessions"
        names: list[str] = []
        enumerated_fully = True
        try:
            with os.scandir(sessions_dir) as entries:
                for entry in entries:
                    if time.monotonic() > deadline:
                        enumerated_fully = False
                        break
                    if entry.name.endswith(".json"):
                        names.append(entry.name[: -len(".json")])
        except OSError:
            enumerated_fully = False
        # Timestamp ids newest-first, then ids that do not encode a time.
        timed_pairs: list[tuple[datetime, str]] = []
        untimed: list[str] = []
        parsed_fully = True
        for index, name in enumerate(names):
            if index % 256 == 0 and time.monotonic() > deadline:
                parsed_fully = False
                untimed.extend(names[index:])
                break
            stamp = self._session_id_time(name)
            if stamp is None:
                untimed.append(name)
            else:
                timed_pairs.append((stamp, name))
        timed_pairs.sort(reverse=True)
        untimed.sort()
        all_parsed = not untimed and parsed_fully
        if not parsed_fully:
            enumerated_fully = False
        names = [n for _, n in timed_pairs] + untimed
        sessions: list[Any] = []
        read_fully = enumerated_fully
        for name in names:
            if time.monotonic() > deadline:
                read_fully = False
                break
            session = self.storage.load_session(name)
            if session:
                sessions.append(session)
        return {
            "sessions": sessions,
            "names": names,
            "enumerated_fully": enumerated_fully,
            "read_fully": read_fully,
            "all_parsed": all_parsed,
            "stamps": {name: stamp for stamp, name in timed_pairs},
        }

    def _session_improvement(self, session: Any) -> Any:
        """Improvement of the session already in memory (no second file read);
        the same computation as ``LocalStorageManager.get_session_summary``."""
        if not session.trials:
            return None
        return relative_change(
            session.baseline_score,
            session.best_score,
            self.storage._session_summary_orientation(session),
        )

    def collect_usage_stats(self, time_budget: float | None = None) -> dict[str, Any]:
        """Collect privacy-safe aggregated usage statistics.

        Args:
            time_budget: Optional cap in seconds on the scan. When the cap is
                hit, ``total_sessions`` is the file count (if the listing
                itself was cut short: the number of names seen) and
                ``days_since_first_use`` comes from the session ids. Every
                other aggregate (completed/failed counts, trials, functions,
                config-space and improvement averages) is computed from the
                sessions read in time: ids that encode a timestamp
                newest-first, then ids that do not. ``sessions_last_30_days``
                is exact whenever the recent sessions fit in the budget.

        Returns:
            Dictionary containing aggregated, non-sensitive stats
        """
        if not self.enabled:
            return {}

        try:
            first_use_candidates: list[datetime] | None = None
            persist_first_use = True
            truncated_total: int | None = None
            if time_budget is None:
                sessions = self.storage.list_sessions()
            else:
                scan = self._scan_sessions(time_budget)
                sessions = scan["sessions"]
                persist_first_use = scan["enumerated_fully"] and (
                    scan["read_fully"] or scan["all_parsed"]
                )
                if not scan["read_fully"]:
                    truncated_total = len(scan["names"])
                stamps: dict[str, datetime] = scan["stamps"]
                first_use_candidates = list(stamps.values())
                # Ids that do not encode a time fall back to the sample.
                first_use_candidates += [
                    datetime.fromisoformat(x.created_at)
                    for x in sessions
                    if x.session_id not in stamps
                ]
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

            avg_config_space_size = _average_config_space_size(completed_sessions)

            # Performance improvement metrics (aggregated)
            improvements: list[float] = []
            for session in completed_sessions:
                value = self._session_improvement(session)
                if isinstance(value, (int, float)):
                    improvements.append(float(value))

            avg_improvement = (
                sum(improvements) / len(improvements) if improvements else None
            )

            return {
                # Basic usage metrics
                "total_sessions": (
                    truncated_total if truncated_total is not None else len(sessions)
                ),
                "completed_sessions": len(completed_sessions),
                "failed_sessions": len(failed_sessions),
                "total_trials": total_trials,
                "total_completed_trials": total_completed_trials,
                "unique_functions_optimized": function_count,
                # Complexity and performance metrics
                "avg_trials_per_session": round(avg_trials_per_session, 2),
                "avg_config_space_size": round(avg_config_space_size, 2),
                "avg_improvement_percent": (
                    round(avg_improvement * 100, 2)
                    if avg_improvement is not None
                    else None
                ),
                # Time-based metrics
                "sessions_last_30_days": len(recent_sessions),
                "days_since_first_use": self._get_days_since_first_use(
                    first_use_candidates, persist=persist_first_use
                ),
                # Version and environment (non-sensitive)
                "sdk_version": get_version(),
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

    def _first_use_file(self) -> Path:
        return Path(self.storage.storage_path) / _FIRST_USE_FILENAME

    def _write_first_use(self, first_use: datetime) -> None:
        """Atomic write: unique temp file (O_EXCL, O_NOFOLLOW) then os.replace;
        never through a symlink; a concurrent writer is tolerated."""
        target = self._first_use_file()
        if target.is_symlink():
            logger.debug("Refusing to write first-use time through a symlink")
            return
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(first_use.isoformat())
            os.replace(tmp, target)
        except Exception as e:
            logger.debug(f"Could not save first-use time: {e}")
            try:
                tmp.unlink()
            except OSError:
                pass

    def _read_first_use(self) -> str:
        if not _O_NOFOLLOW and os.path.islink(self._first_use_file()):
            raise OSError("first-use file is a symlink")
        fd = os.open(self._first_use_file(), os.O_RDONLY | _O_NOFOLLOW)
        with os.fdopen(fd) as handle:
            return handle.read().strip()

    def _get_days_since_first_use(
        self,
        first_use_candidates: list[datetime] | None = None,
        persist: bool = True,
    ) -> int:
        """Calculate days since first optimization session.

        Reads a stored first-use timestamp (next to ``.analytics_id``). Only when
        it is missing is it derived once, from ``first_use_candidates`` if given
        or else a full scan, and then stored so later calls never scan.
        """
        try:
            try:
                first_use = datetime.fromisoformat(self._read_first_use())
                if first_use.tzinfo is None:
                    first_use = first_use.replace(tzinfo=UTC)
                return max((datetime.now(UTC) - first_use).days, 0)
            except (OSError, ValueError):
                pass

            if first_use_candidates is None:
                first_use_candidates = [
                    datetime.fromisoformat(s.created_at)
                    for s in self.storage.list_sessions()
                ]
            if not first_use_candidates:
                return 0

            first_use = min(first_use_candidates)
            if first_use.tzinfo is None:
                first_use = first_use.replace(tzinfo=UTC)
            if persist:
                self._write_first_use(first_use)
            return (datetime.now(UTC) - first_use).days

        except Exception as e:
            logger.debug(f"Could not calculate days active (defaulting to 0): {e}")
            return 0

    def _submission_skip_result(self, force: bool) -> dict[str, Any] | None:
        # Re-evaluate the gate at send time: the environment can change after init.
        if not self.enabled or not usage_analytics_allowed(self.config):
            return {"success": False, "reason": "Analytics disabled"}

        # Check if submission is due (daily by default)
        if not force and not self._is_submission_due():
            return {"success": False, "reason": "Submission not due yet"}
        return None

    def _credentials_refusal(self) -> tuple[dict[str, Any] | None, str | None]:
        """The checks develop runs after collecting stats and before building
        the client: API-key read, ``_analytics_api_key_error``, then
        ``may_send_telemetry`` (origin/develop local_analytics.py 327-332).

        Returns ``(refusal_result, api_key)``.
        """
        try:
            from traigent.config.backend_config import BackendConfig

            api_key = BackendConfig.get_api_key()
        except ImportError:
            logger.debug("Backend client not available for analytics submission")
            return {"success": False, "reason": "Backend client not available"}, None
        api_key_error = _analytics_api_key_error(api_key)
        if api_key_error:
            return {"success": False, "reason": api_key_error}, None
        if not may_send_telemetry():
            return {"success": False, "reason": "Analytics disabled"}, None
        return None, api_key

    async def _collect_stats_off_loop(self, budget: float) -> dict[str, Any] | None:
        """Run collection on a dedicated daemon thread (not the default executor,
        which ``asyncio.run`` joins at shutdown). Returns None if it overran
        ``budget`` plus a margin; the abandoned daemon thread cannot block exit.
        """
        global _COLLECTOR_THREAD
        result: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()

        def work() -> None:
            try:
                outcome = self.collect_usage_stats(budget)
            except BaseException as exc:  # hand any failure back to the caller
                try:
                    result.set_exception(exc)
                except concurrent.futures.InvalidStateError:
                    pass
                return
            try:
                result.set_result(outcome)
            except concurrent.futures.InvalidStateError:
                pass  # the caller already gave up

        thread = threading.Thread(target=work, name="traigent-usage-stats", daemon=True)
        with _COLLECTOR_LOCK:
            current = _COLLECTOR_THREAD
            alive = current is not None and current.is_alive()
            if not alive:
                _COLLECTOR_THREAD = thread
        if alive:
            logger.debug("Previous usage stats collection still running; skipping")
            return None
        thread.start()
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(result),
                timeout=budget + _SUBMIT_SCAN_MARGIN_SECONDS,
            )
        except TimeoutError:
            logger.debug("Usage stats collection timed out; skipping this submission")
            return None

    async def submit_usage_stats(self, force: bool = False) -> dict[str, Any]:
        """Submit usage statistics using existing backend session infrastructure.

        Creates a lightweight analytics session with only aggregated, non-sensitive data.

        Args:
            force: Force submission even if not due yet

        Returns:
            Result dictionary with success status and details
        """
        # Develop's pre-collect check (origin/develop 312), unchanged.
        skipped = self._submission_skip_result(force)
        if skipped is not None:
            return skipped

        # Only to avoid a scan whose send would be refused anyway; the
        # after-collect run below is the authoritative one.
        refusal, _ = self._credentials_refusal()
        if refusal is not None:
            return refusal

        stats = await self._collect_stats_off_loop(_SUBMIT_SCAN_BUDGET_SECONDS)
        if stats is None:
            return {"success": False, "reason": "Request timeout"}
        if not stats:
            return {"success": False, "reason": "No stats to submit"}

        try:
            # Import backend client lazily to avoid circular imports
            from traigent.cloud import backend_client as backend_client_module
            from traigent.config.backend_config import BackendConfig

            refusal, api_key = self._credentials_refusal()
            if refusal is not None:
                return refusal
            backend_client = backend_client_module.get_backend_client(
                api_key=api_key,
                base_url=BackendConfig.get_backend_url(),
                enable_fallback=False,  # We're explicitly sending analytics
                timeout=5.0,  # Short timeout for analytics - don't block program exit
            )

            if not may_send_telemetry():
                return {"success": False, "reason": "Analytics disabled"}
            submission = _usage_analytics_submission.set(True)
            try:
                await backend_client.__aenter__()
                if not may_send_telemetry():
                    return {"success": False, "reason": "Analytics disabled"}
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
                        "version": stats.get("sdk_version") or get_version(),
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

                if not may_send_telemetry():
                    return {"success": False, "reason": "Analytics disabled"}
                # Submit the aggregated stats as trial results
                success = await backend_client.submit_privacy_trial_results(
                    session_id=session_id,
                    trial_id=f"analytics_{datetime.now(UTC).isoformat()}",
                    config={},
                    metrics=_analytics_trial_metrics(stats),
                    duration=1.0,  # Analytics collection time
                )

                if not success and not may_send_telemetry():
                    return {"success": False, "reason": "Analytics disabled"}
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
                _usage_analytics_submission.reset(submission)
                # Ensure session is closed even on cancellation
                await _close_analytics_backend_client(
                    backend_client_module, backend_client
                )

        except _TelemetryAuthSuppressed:
            return {"success": False, "reason": "Analytics disabled"}
        except TimeoutError:
            return {"success": False, "reason": "Request timeout"}
        except ImportError:
            logger.debug("Backend client not available for analytics submission")
            return {"success": False, "reason": "Backend client not available"}
        except EndpointNotConfiguredError:
            logger.debug(endpoint_missing_message("analytics submission"))
            return {"success": False, "reason": "No backend endpoint configured"}
        except Exception as e:
            if not may_send_telemetry():
                return {"success": False, "reason": "Analytics disabled"}
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
    from traigent.utils.env_config import is_mock_llm

    if not usage_analytics_allowed(config) or config.execution_mode not in {
        ExecutionMode.LOCAL.value,
    }:
        return

    if is_mock_llm():
        logger.debug("Skipping usage analytics submission in mock LLM mode")
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

            task = asyncio.create_task(_submit_wrapper(), name="analytics_submit")
            _register_background_task(task)
        except RuntimeError:
            # No running loop, we're in a sync context
            # Create a new event loop in a thread to avoid blocking
            import threading

            def _run_in_thread() -> None:
                try:
                    asyncio.run(analytics.submit_usage_stats())
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
