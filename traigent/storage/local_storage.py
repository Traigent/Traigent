"""
Local storage manager for Traigent optimization sessions.
Inspired by DeepEval's approach to local JSON file storage.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Performance FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

from ..api.types import OptimizationStatus
from ..utils.exceptions import TraigentStorageError
from ..utils.function_identity import sanitize_identifier
from ..utils.logging import get_logger
from ..utils.objectives import is_minimization_objective
from ..utils.secure_path import PathTraversalError, safe_open, validate_path

logger = get_logger(__name__)

TRIAL_COST_FIELDS = ("cost", "total_cost", "input_cost", "output_cost")


def _coerce_optional_cost(value: Any) -> float | None:
    """Coerce a cost-like value to float without accepting booleans."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_trial_cost_fields(
    metadata: Mapping[str, Any | None] | None,
) -> dict[str, float | None]:
    """Extract canonical trial cost fields from stored trial metadata.

    Local runs have historically carried cost in metadata rather
    than first-class storage fields. Keep sync backward-compatible by accepting
    top-level metrics and the older total_example_cost alias.
    """
    costs: dict[str, float | None] = dict.fromkeys(TRIAL_COST_FIELDS)

    if not isinstance(metadata, Mapping):
        return costs

    sources: list[Mapping[str, Any | None]] = [metadata]
    for nested_key in ("all_metrics", "metrics"):
        nested_value = metadata.get(nested_key)
        if isinstance(nested_value, Mapping):
            sources.append(nested_value)

    evaluation_result = metadata.get("evaluation_result")
    if isinstance(evaluation_result, Mapping):
        sources.append(evaluation_result)
        evaluation_metrics = evaluation_result.get("metrics")
        if isinstance(evaluation_metrics, Mapping):
            sources.append(evaluation_metrics)

    for source in sources:
        for field in TRIAL_COST_FIELDS:
            if costs[field] is None:
                costs[field] = _coerce_optional_cost(source.get(field))

        if costs["total_cost"] is None:
            costs["total_cost"] = _coerce_optional_cost(
                source.get("total_example_cost")
            )

    if costs["total_cost"] is None and costs["cost"] is not None:
        costs["total_cost"] = costs["cost"]
    if costs["cost"] is None and costs["total_cost"] is not None:
        costs["cost"] = costs["total_cost"]
    if costs["total_cost"] is None:
        input_cost = costs["input_cost"]
        output_cost = costs["output_cost"]
        if input_cost is not None and output_cost is not None:
            costs["total_cost"] = input_cost + output_cost
            costs["cost"] = costs["total_cost"]

    return costs


@dataclass
class TrialResult:
    """Individual trial result in an optimization session."""

    trial_id: int
    config: dict[str, Any]
    score: float | None
    timestamp: str
    metadata: dict[str, Any | None] | None = None
    error: str | None = None
    cost: float | None = None
    total_cost: float | None = None
    input_cost: float | None = None
    output_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class SessionSummary:
    """The two session fields usage counters need (see ``session_summaries``)."""

    status: str
    completed_trials: int


@dataclass
class OptimizationSession:
    """Complete optimization session data."""

    session_id: str
    function_name: str
    created_at: str
    updated_at: str
    status: str  # not_started, pending, running, completed, failed, cancelled
    # Number of trials RECORDED on this session. Invariant: ``total_trials ==
    # len(trials)``. Maintained by ``add_trial_result`` and re-asserted by
    # ``finalize_session``; both deserializers — ``load_session`` and
    # ``OptimizationSession.from_dict`` — reconcile it in memory so records
    # written before #2032 (which froze it at the 0 set at construction, next
    # to a non-zero ``completed_trials``) report a truthful count. Any future
    # deserializer belongs on that list too: a reader that skips the
    # reconciliation is how the invariant comes back apart.
    #
    # It is deliberately NOT the ``max_trials`` budget, and must never be
    # populated from ``metadata["max_trials"]``. Copying a budget in would make a
    # 24-config exhaustive grid persist ``total_trials: 10, completed_trials: 24``
    # and the CLI print "24/10" — a recorded count and a budget are different
    # quantities, so the rule stands regardless of how the budget is derived.
    #
    # The original reason given here was narrower and is now out of date: it said
    # ``metadata["max_trials"]`` carries a *fabricated* 10 for a run that set no
    # budget, so no guard could tell a user's real 10 from the SDK's default.
    # #2049 removed that substitution, so the metadata value is now absent when
    # no budget was set. The prohibition is unchanged — it never depended on the
    # fabrication, only on the two quantities meaning different things.
    total_trials: int
    completed_trials: int
    best_config: dict[str, Any | None] | None = None
    best_score: float | None = None
    baseline_score: float | None = None
    trials: list[TrialResult] | None = None
    optimization_config: dict[str, Any | None] | None = None
    metadata: dict[str, Any | None] | None = None
    # Cloud-sync provenance & progress (Sync v2). Absent on never-synced
    # sessions; populated by the syncer so re-sync is idempotent/resumable and
    # `traigent sync` can report status. See traigent/cloud/session_sync.py.
    sync_state: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.trials is None:
            self.trials: list[Any] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationSession":
        """Create from dictionary."""
        # Convert trial data back to TrialResult objects
        trials = []
        for trial_data in data.get("trials", []):
            if isinstance(trial_data, dict):
                trials.append(TrialResult.from_dict(trial_data))
            else:
                trials.append(trial_data)

        data["trials"] = trials
        session = cls(**data)
        session._reconcile_total_trials(str(session.session_id))
        return session

    def _reconcile_total_trials(self, label: str) -> None:
        """Restore ``total_trials == len(trials)`` on a just-deserialized session.

        Shared by both deserializers (this dataclass's ``from_dict`` and
        ``LocalStorageManager.load_session``) rather than written out twice: a
        record written before #2032 carries ``total_trials: 0`` alongside a
        non-zero trial list, which makes callers print a zero total and makes
        any completion ratio divide by zero. The count is definitionally
        derivable from the trials just read, so recomputing it recovers a fact
        rather than inventing a default. No write here — these are read paths;
        the file self-heals on its next save.
        """
        recorded_trials = len(self.trials or [])
        if self.total_trials != recorded_trials:
            # the session label is a (possibly externally supplied) session id: never logged
            logger.debug("Reconciled total_trials of a session (#2032)")
            self.total_trials = recorded_trials


def _dataclass_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that ``cls`` (a dataclass) accepts.

    Lets a session file written by a newer SDK — with fields this version
    doesn't know about — still load instead of raising ``TypeError`` from an
    unexpected keyword argument.
    """
    allowed = {f.name for f in dataclass_fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


class LocalStorageManager:
    """
    Manages local storage for Traigent optimization sessions.
    Similar to DeepEval's DEEPEVAL_RESULTS_FOLDER approach.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """
        Initialize local storage manager.

        Args:
            storage_path: Custom storage path. If None, uses environment variable
                         TRAIGENT_RESULTS_FOLDER or default ~/.traigent/
        """
        self.storage_path = self._get_storage_path(storage_path)
        self._ensure_directories()

    def _get_storage_path(self, custom_path: str | None) -> Path:
        """Get storage path from custom path, env var, or default."""
        if custom_path:
            return Path(custom_path).expanduser().resolve()

        # Check environment variable (like DeepEval's DEEPEVAL_RESULTS_FOLDER)
        env_path = os.getenv("TRAIGENT_RESULTS_FOLDER")
        if env_path:
            return Path(env_path).expanduser().resolve()

        # Default to ~/.traigent/
        return Path.home() / ".traigent"

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            (self.storage_path / "sessions").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "cache").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "cache" / "model_responses").mkdir(
                parents=True, exist_ok=True
            )
            logger.debug("Initialized storage directories")
        except Exception as exc:
            # OS error text quotes the (customer-chosen) storage path
            raise TraigentStorageError(
                f"Failed to create storage directories ({type(exc).__name__})"
            ) from None

    def _is_process_alive(self, pid: int) -> bool:
        """Best-effort check whether a process is alive."""
        if pid <= 0:
            return False
        return bool(psutil.pid_exists(pid))

    def _read_lock_pid(self, lock_path: Path) -> int | None:
        """Read owning PID from lock file, if present."""
        try:
            # nosec - lock_path is computed internally by LocalStorage from
            # self.storage_path and a fixed lock filename; never user-controlled.
            with open(lock_path, encoding="utf-8") as lock_file:
                raw_value = lock_file.read().strip()
        except OSError:
            return None

        if not raw_value:
            return None
        try:
            return int(raw_value)
        except ValueError:
            return None

    def _try_cleanup_stale_lock(self, lock_path: Path) -> bool:
        """Remove stale lock when owner PID is no longer running."""
        owner_pid = self._read_lock_pid(lock_path)
        if owner_pid is None or self._is_process_alive(owner_pid):
            return False

        try:
            os.unlink(str(lock_path))
            # lock names can embed a session id (sync_manager: "sync_<id>"): not logged
            logger.warning("Removed a stale lock owned by a dead process")
            return True
        except FileNotFoundError:
            # Another process removed it before us.
            return True
        except OSError:
            return False

    @contextmanager
    def acquire_lock(self, lock_name: str, timeout: float = 5.0):
        """
        Acquire inter-process lock using OS primitives (cross-platform).

        Args:
            lock_name: Name of the lock (e.g., "dedup_func_dataset")
            timeout: Maximum time to wait for lock acquisition

        Yields:
            None when lock is acquired

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        safe_lock_name = sanitize_identifier(lock_name)
        lock_dir = self.storage_path / ".locks"
        lock_dir.mkdir(exist_ok=True, parents=True)
        lock_path = validate_path(
            lock_dir / f"{safe_lock_name}.lock",
            self.storage_path,
            must_exist=False,
        )

        start_time = time.time()
        backoff = 0.01

        while True:
            try:
                # Atomic file creation works on all platforms
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                logger.debug("Acquired lock")
                try:
                    yield  # Lock acquired
                finally:
                    os.close(fd)
                    try:
                        os.unlink(str(lock_path))  # Clean up lock file
                        logger.debug("Released lock")
                    except OSError:
                        pass  # Another process may have cleaned it
                break
            except FileExistsError:
                # Lock is held by another process
                if self._try_cleanup_stale_lock(lock_path):
                    continue
                if time.time() - start_time > timeout:
                    # the lock name can embed a session id: kept off the message
                    raise TimeoutError(
                        "Could not acquire lock before the timeout"
                    ) from None
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.5)  # Exponential backoff with cap

    def create_session(
        self,
        function_name: str,
        optimization_config: dict[str, Any | None] | None = None,
        metadata: dict[str, Any | None] | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Create a new optimization session.

        Args:
            function_name: Name of the function being optimized
            optimization_config: Configuration for the optimization
            metadata: Additional metadata
            session_id: Optional externally-supplied id (e.g. a backend
                session id, so connected hybrid runs can durably persist
                trials under the same key the backend uses — see #1279). When
                omitted, a local id is minted. If a session with this id
                already exists it is returned unchanged (idempotent — existing
                trials are never clobbered).

        Returns:
            session_id: Unique identifier for the session
        """
        timestamp = datetime.now(UTC)
        if session_id is not None:
            existing = self.load_session(session_id)
            if existing is not None:
                return session_id
        else:
            safe_function = sanitize_identifier(function_name)
            session_id = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{safe_function}_{uuid4().hex[:8]}"

        session = OptimizationSession(
            session_id=session_id,
            function_name=function_name,
            created_at=timestamp.isoformat(),
            updated_at=timestamp.isoformat(),
            status=OptimizationStatus.PENDING.value,
            total_trials=0,
            completed_trials=0,
            optimization_config=optimization_config,
            metadata=metadata or {},
        )

        self._save_session(session)
        logger.info("Created optimization session")
        return session_id

    def update_session_status(self, session_id: str, status: str) -> None:
        """Update session status."""
        session = self.load_session(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now(UTC).isoformat()
            self._save_session(session)
            logger.debug("Updated session status")

    def add_trial_result(
        self,
        session_id: str,
        config: dict[str, Any],
        score: float | None,
        metadata: dict[str, Any | None] | None = None,
        error: str | None = None,
        cost: float | None = None,
        total_cost: float | None = None,
        input_cost: float | None = None,
        output_cost: float | None = None,
    ) -> None:
        """
        Add a trial result to the session.

        Args:
            session_id: Session identifier
            config: Configuration that was tested
            score: Score achieved with this configuration (may be None when unavailable)
            metadata: Additional trial metadata
            error: Error message if trial failed
            cost: Optional total trial cost alias in USD
            total_cost: Optional total trial cost in USD
            input_cost: Optional input-token cost in USD
            output_cost: Optional output-token cost in USD
        """
        session = self.load_session(session_id)
        if not session:
            raise TraigentStorageError(
                "Session <id> not found", details={"session_id": session_id}
            ) from None

        if session.trials is None:
            session.trials = []

        extracted_costs = extract_trial_cost_fields(metadata)
        explicit_costs = {
            "cost": cost,
            "total_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
        }
        for field, value in explicit_costs.items():
            coerced_value = _coerce_optional_cost(value)
            if coerced_value is not None:
                extracted_costs[field] = coerced_value

        if (
            extracted_costs["total_cost"] is None
            and extracted_costs["cost"] is not None
        ):
            extracted_costs["total_cost"] = extracted_costs["cost"]
        if (
            extracted_costs["cost"] is None
            and extracted_costs["total_cost"] is not None
        ):
            extracted_costs["cost"] = extracted_costs["total_cost"]

        trial_result = TrialResult(
            trial_id=len(session.trials) + 1,
            config=config,
            score=score,
            timestamp=datetime.now(UTC).isoformat(),
            metadata=metadata,
            error=error,
            cost=extracted_costs["cost"],
            total_cost=extracted_costs["total_cost"],
            input_cost=extracted_costs["input_cost"],
            output_cost=extracted_costs["output_cost"],
        )

        session.trials.append(trial_result)
        session.completed_trials = len(session.trials)
        # total_trials counts recorded trials, so it moves with the trial list
        # rather than staying at the 0 stamped at construction (#2032).
        session.total_trials = len(session.trials)
        session.updated_at = datetime.now(UTC).isoformat()

        # Update best score if this is better
        if score is not None:
            primary_objective = self._resolve_primary_objective_name(
                session.optimization_config
            )
            orientation = self._resolve_primary_objective_orientation(
                session.optimization_config, primary_objective
            )
            if session.best_score is None:
                is_better = True
            elif orientation == "band":
                low, high = self._resolve_primary_objective_band(
                    session.optimization_config, primary_objective
                )

                def distance_from_band(value: float) -> float:
                    if low <= value <= high:
                        return 0.0
                    return min(abs(value - low), abs(value - high))

                is_better = distance_from_band(score) < distance_from_band(
                    session.best_score
                )
            else:
                is_minimize = is_minimization_objective(
                    primary_objective, orientation=orientation
                )
                if is_minimize:
                    is_better = score < session.best_score
                else:
                    is_better = score > session.best_score
            if is_better:
                session.best_score = score
                session.best_config = config.copy()

        self._save_session(session)
        logger.debug("Added trial result to session")

    @staticmethod
    def _extract_objective_name_candidate(candidate: Any) -> str | None:
        """Extract an objective name from a string or mapping candidate."""
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, dict):
            name = candidate.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    @classmethod
    def _extract_primary_name_from_list(cls, value: Any) -> str | None:
        """Extract the first valid objective name from a list-like config field."""
        if not (isinstance(value, list) and value):
            return None
        return cls._extract_objective_name_candidate(value[0])

    @classmethod
    def _extract_primary_name_from_schema(
        cls, optimization_config: dict[str, Any | None]
    ) -> str | None:
        """Extract the primary objective name from objective_schema when present."""
        objective_schema = optimization_config.get("objective_schema")
        if not isinstance(objective_schema, dict):
            return None
        return cls._extract_primary_name_from_list(objective_schema.get("objectives"))

    @classmethod
    def _resolve_primary_objective_name(
        cls,
        optimization_config: dict[str, Any | None] | None,
    ) -> str:
        """Resolve primary objective name from stored optimization config."""
        if not isinstance(optimization_config, dict):
            return "score"

        name = cls._extract_primary_name_from_list(
            optimization_config.get("objectives")
        )
        if name:
            return name

        name = cls._extract_primary_name_from_schema(optimization_config)
        if name:
            return name

        return "score"

    @staticmethod
    def _find_objective_definition(
        schema: Any, primary_objective: str
    ) -> dict[str, Any] | None:
        """First schema objective definition named ``primary_objective``."""
        definitions = schema.get("objectives") if isinstance(schema, dict) else None
        if not isinstance(definitions, list):
            return None
        for definition in definitions:
            if (
                isinstance(definition, dict)
                and definition.get("name") == primary_objective
            ):
                return definition
        return None

    def _resolve_primary_objective_orientation(
        self,
        optimization_config: dict[str, Any | None] | None,
        primary_objective: str,
    ) -> str | None:
        if not isinstance(optimization_config, dict):
            return "maximize" if primary_objective == "score" else None
        schema = optimization_config.get("objective_schema")
        definition = LocalStorageManager._find_objective_definition(
            schema, primary_objective
        )
        if definition is not None:
            orientation = definition.get("orientation")
            return str(orientation) if orientation is not None else None
        orientations = optimization_config.get("objective_orientations")
        if isinstance(orientations, dict) and primary_objective in orientations:
            return str(orientations[primary_objective])
        has_declared_objective = bool(optimization_config.get("objectives")) or bool(
            isinstance(schema, dict) and schema.get("objectives")
        )
        if primary_objective == "score" and not has_declared_objective:
            # A session created without objective metadata uses the storage
            # protocol's scalar score contract, which explicitly maximizes.
            # This is not metric-name inference for a user-defined objective.
            return "maximize"
        return None

    @staticmethod
    def _valid_band_target(target: Any) -> tuple[float, float] | None:
        """Return ``(low, high)`` when ``target`` is a finite, ordered pair."""
        if not (isinstance(target, list) and len(target) == 2):
            return None
        low, high = target
        # Type-check both bounds before any finiteness check: math.isfinite
        # raises OverflowError on an int too large for a float.
        for bound in (low, high):
            if not isinstance(bound, (int, float)) or isinstance(bound, bool):
                return None
        if not (math.isfinite(low) and math.isfinite(high)):
            return None
        if low > high:
            return None
        return float(low), float(high)

    @staticmethod
    def _resolve_primary_objective_band(
        optimization_config: dict[str, Any | None] | None,
        primary_objective: str,
    ) -> tuple[float, float]:
        schema = (
            optimization_config.get("objective_schema")
            if isinstance(optimization_config, dict)
            else None
        )
        definitions = schema.get("objectives") if isinstance(schema, dict) else None
        for definition in definitions if isinstance(definitions, list) else ():
            if not (
                isinstance(definition, dict)
                and definition.get("name") == primary_objective
            ):
                continue
            band = definition.get("band")
            target = band.get("target") if isinstance(band, dict) else None
            resolved = LocalStorageManager._valid_band_target(target)
            if resolved is not None:
                return resolved
        # the objective name is stored caller config: not quoted
        raise ValueError("Banded objective requires a complete target band")

    def finalize_session(
        self,
        session_id: str,
        status: str | None = None,
        *,
        best_config: dict[str, Any] | None = None,
        best_score: float | None = None,
        apply_best: bool = False,
    ) -> OptimizationSession | None:
        """
        Finalize an optimization session.

        Args:
            session_id: Session identifier
            status: Final status (completed, failed, cancelled)
            best_config: The terminal selector's winner. When given, it
                replaces the incrementally tracked best so the stored session
                names the same best as the returned result (tie-break and
                eligibility rules are applied once, by the selector;
                Traigent#2660).
            best_score: Score of ``best_config``.
            apply_best: When True the stored best is REPLACED by
                ``best_config``/``best_score`` even if they are empty, i.e. a
                final selection with no winner clears a stale incremental best.

        Returns:
            The finalized session data or None if not found
        """
        session = self.load_session(session_id)
        if not session:
            logger.error("Session to finalize not found")
            return None

        session.status = status or OptimizationStatus.COMPLETED.value
        if apply_best:
            session.best_config = dict(best_config) if best_config else None
            session.best_score = best_score if best_config else None
        elif best_config is not None:
            session.best_config = dict(best_config)
            session.best_score = best_score
        # Re-assert the recorded-trial invariant at the end of the run: an
        # early-stopped run records fewer trials than it planned, and the count
        # that persists is the one that actually happened (#2032).
        session.total_trials = len(session.trials or [])
        session.updated_at = datetime.now(UTC).isoformat()
        self._save_session(session)

        logger.info("Finalized session")
        return session

    def update_sync_state(
        self,
        session_id: str,
        patch: dict[str, Any],
        *,
        trial_updates: dict[str, dict[str, Any]] | None = None,
    ) -> OptimizationSession | None:
        """Merge a patch into a session's ``sync_state`` and persist atomically.

        ``patch`` is shallow-merged into the top level of ``sync_state``.
        ``trial_updates`` (keyed by the local trial id as a string) is merged
        into ``sync_state["trials"]`` so per-trial upload progress accumulates
        across resumable sync attempts. Returns the updated session, or None if
        the session is missing.
        """
        session = self.load_session(session_id)
        if not session:
            logger.warning("Cannot update sync_state: session not found")
            return None

        state: dict[str, Any] = dict(session.sync_state or {})
        state.update(patch)
        if trial_updates:
            trials_state: dict[str, Any] = dict(state.get("trials") or {})
            for trial_key, trial_patch in trial_updates.items():
                merged = dict(trials_state.get(trial_key) or {})
                merged.update(trial_patch)
                trials_state[trial_key] = merged
            state["trials"] = trials_state

        session.sync_state = state
        session.updated_at = datetime.now(UTC).isoformat()
        self._save_session(session)
        return session

    def load_session(self, session_id: str) -> OptimizationSession | None:
        """Load a session from storage."""
        session_file = validate_path(
            self.storage_path / "sessions" / f"{session_id}.json",
            self.storage_path,
            must_exist=False,
        )

        if not session_file.exists():
            return None

        try:
            with safe_open(session_file, self.storage_path, mode="r") as f:
                data = json.load(f)

            # Convert trial data back to TrialResult objects. Drop unknown keys
            # so a file written by a newer SDK (with extra fields) still loads.
            trials = []
            for trial_data in data.get("trials", []):
                trials.append(TrialResult(**_dataclass_kwargs(TrialResult, trial_data)))

            data["trials"] = trials

            # Handle backward compatibility for missing timestamp fields
            from datetime import datetime

            current_time = datetime.now(UTC).isoformat()

            # Provide default values for missing required fields
            if "created_at" not in data:
                data["created_at"] = current_time
            if "updated_at" not in data:
                data["updated_at"] = current_time

            session = OptimizationSession(
                **_dataclass_kwargs(OptimizationSession, data)
            )

            # Reconcile the recorded-trial count in memory (#2032). Shared with
            # OptimizationSession.from_dict so the two deserializers cannot
            # disagree about the session they just read.
            session._reconcile_total_trials(session_id)

            return session

        except Exception as e:
            # the exception text can quote the session file's full path
            logger.error("Failed to load session (%s)", type(e).__name__)
            return None

    def list_sessions(self, status: str | None = None) -> list[OptimizationSession]:
        """
        List all sessions, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of sessions matching criteria
        """
        sessions: list[OptimizationSession] = []
        sessions_dir = self.storage_path / "sessions"

        if not sessions_dir.exists():
            return sessions

        for session_file in sessions_dir.glob("*.json"):
            session = self.load_session(session_file.stem)
            if session and (status is None or session.status == status):
                sessions.append(session)

        # Sort by creation time, newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    _SESSION_INDEX_NAME = "session_summary_index.json"

    def session_summaries(self) -> list[SessionSummary]:
        """Return ``status`` / ``completed_trials`` for every stored session.

        Equivalent to reading those two fields from :meth:`list_sessions`, but
        a session file is parsed only when its ``(mtime_ns, size)`` differs
        from the cached entry in ``cache/session_summary_index.json``. Callers
        that only count sessions (the post-run incentive hint, #2457) would
        otherwise re-read every stored session after every run: O(history)
        work that grew to a ~1 min stall on a large ``~/.traigent``.
        """
        sessions_dir = self.storage_path / "sessions"
        if not sessions_dir.exists():
            return []
        # ``cache`` may be a symlink to somewhere else: confine the index (and
        # its temp file) to the storage root like session writes are, and run
        # uncached rather than follow the link out.
        index_path: Path | None
        try:
            index_path = validate_path(
                self.storage_path / "cache" / self._SESSION_INDEX_NAME,
                self.storage_path,
                must_exist=False,
            )
            # A symlinked index file could point at a stored session (in-root,
            # so containment passes) and the atomic replace would clobber it.
            if (
                index_path.is_symlink()
                or (self.storage_path / "cache" / self._SESSION_INDEX_NAME).is_symlink()
            ):
                index_path = None
        except (PathTraversalError, OSError, ValueError):
            index_path = None
        index: dict[str, Any] = {}
        try:
            if index_path is None:
                raise OSError("session summary index outside storage root")
            with open(index_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                index = loaded
        except (OSError, ValueError):
            index = {}

        fresh: dict[str, list[Any]] = {}
        for session_file in sessions_dir.glob("*.json"):
            try:
                stat = session_file.stat()
            except OSError:
                continue
            session_id = session_file.stem
            cached = index.get(session_id)
            if (
                isinstance(cached, list)
                and len(cached) == 4
                and type(cached[0]) is int
                and type(cached[1]) is int
                and isinstance(cached[2], str)
                and cached[2] in {status.value for status in OptimizationStatus}
                and type(cached[3]) is int
                and cached[3] >= 0
                and cached[0] == stat.st_mtime_ns
                and cached[1] == stat.st_size
            ):
                fresh[session_id] = cached
                continue
            session = self.load_session(session_id)
            if session is None:
                continue
            fresh[session_id] = [
                stat.st_mtime_ns,
                stat.st_size,
                session.status,
                int(session.completed_trials or 0),
            ]

        if index_path is not None and fresh != index:
            try:
                tmp_path = validate_path(
                    index_path.with_name(f".{index_path.name}.{uuid4().hex}.tmp"),
                    self.storage_path,
                    must_exist=False,
                )
                try:
                    with open(tmp_path, "x") as f:
                        json.dump(fresh, f)
                    os.replace(tmp_path, index_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
            except (PathTraversalError, OSError) as e:
                logger.debug(
                    "Failed to write session summary index (%s)", type(e).__name__
                )

        return [
            SessionSummary(status=entry[2], completed_trials=entry[3])
            for entry in fresh.values()
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from storage."""
        session_file = validate_path(
            self.storage_path / "sessions" / f"{session_id}.json",
            self.storage_path,
            must_exist=False,
        )

        if session_file.exists():
            try:
                session_file.unlink()
                logger.info("Deleted session")
                return True
            except Exception as e:
                logger.error("Failed to delete session (%s)", type(e).__name__)
                return False

        return False

    def export_session(
        self, session_id: str, export_path: str, format: str = "json"
    ) -> bool:
        """
        Export a session to a specified location.
        Similar to DeepEval's save_as functionality.

        Args:
            session_id: Session to export
            export_path: Destination path
            format: Export format (currently only 'json')

        Returns:
            Success status
        """
        if format != "json":
            logger.error("Unsupported export format")
            return False

        session = self.load_session(session_id)
        if not session:
            raise TraigentStorageError(
                "Session <id> not found", details={"session_id": session_id}
            ) from None

        try:
            path_obj = Path(export_path).expanduser()
            base_dir = (
                path_obj.parent if path_obj.is_absolute() else Path.cwd().resolve()
            )
            path_obj = validate_path(path_obj, base_dir, must_exist=False)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Convert session to dict for JSON serialization
            session_data = asdict(session)

            # Atomic write: write to temp file, then rename
            temp_path = validate_path(
                path_obj.with_suffix(f"{path_obj.suffix}.tmp.{os.getpid()}"),
                path_obj.parent,
                must_exist=False,
            )
            try:
                with safe_open(temp_path, path_obj.parent, mode="w") as f:
                    json.dump(session_data, f, indent=2, default=str)
                temp_path.replace(path_obj)  # Atomic rename on POSIX
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            logger.info("Exported session")
            return True

        except Exception as e:
            logger.error("Failed to export session (%s)", type(e).__name__)
            return False

    def get_session_summary(self, session_id: str) -> dict[str, Any | None] | None:
        """Get a summary of session statistics.

        The emitted ``total_trials`` is the recorded-trial count (``==
        len(trials)``), not a planned budget — see ``OptimizationSession``.
        It is truthful for pre-#2032 records because ``load_session``
        reconciles the field before this method reads it.
        """
        session = self.load_session(session_id)
        if not session:
            return None

        if not session.trials:
            return {
                "session_id": session_id,
                "function_name": session.function_name,
                "status": session.status,
                "total_trials": session.total_trials,
                "completed_trials": session.completed_trials,
                "successful_trials": 0,
                "best_score": None,
                "baseline_score": session.baseline_score,
                "improvement": None,
                "avg_score": None,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }

        scores = [
            trial.score
            for trial in session.trials
            if trial.error is None and trial.score is not None
        ]
        successful_trials = len(
            [trial for trial in session.trials if trial.error is None]
        )
        improvement = None

        if session.baseline_score and session.best_score:
            improvement = (
                session.best_score - session.baseline_score
            ) / session.baseline_score

        return {
            "session_id": session_id,
            "function_name": session.function_name,
            "status": session.status,
            "total_trials": session.total_trials,
            "completed_trials": session.completed_trials,
            "successful_trials": successful_trials,
            "best_score": session.best_score,
            "baseline_score": session.baseline_score,
            "improvement": improvement,
            "avg_score": sum(scores) / len(scores) if scores else None,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    def _save_session(self, session: OptimizationSession) -> None:
        """Save session to JSON file atomically.

        Uses temp file + rename pattern to prevent data corruption
        if the process crashes during write.
        """
        session_file = validate_path(
            self.storage_path / "sessions" / f"{session.session_id}.json",
            self.storage_path,
            must_exist=False,
        )

        try:
            # Convert to dict for JSON serialization
            session_data = asdict(session)
            session_data = self._make_json_serializable(session_data)

            # Atomic write: write to temp file, then rename
            temp_path = validate_path(
                session_file.with_suffix(f".json.tmp.{os.getpid()}"),
                session_file.parent,
                must_exist=False,
            )
            try:
                with safe_open(temp_path, session_file.parent, mode="w") as f:
                    json.dump(session_data, f, indent=2)
                temp_path.replace(session_file)  # Atomic rename on POSIX
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            # the exception text can quote the session file's full path
            raise TraigentStorageError(
                f"Failed to save session ({type(e).__name__})",
                details={"session_id": session.session_id},
            ) from None

    def _make_json_serializable(self, value: Any) -> Any:
        """Recursively convert a value to something JSON can encode."""
        from traigent.utils.numpy_compat import convert_numpy_value, is_numpy_type

        if is_numpy_type(value):
            return convert_numpy_value(value)
        if isinstance(value, dict):
            return {k: self._make_json_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._make_json_serializable(v) for v in value]
        if isinstance(value, tuple):
            return [self._make_json_serializable(v) for v in value]
        if isinstance(value, set):
            return [self._make_json_serializable(v) for v in value]
        return value

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Clean up sessions older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of sessions deleted
        """
        cutoff_date = datetime.now(UTC).timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0

        for session in self.list_sessions():
            session_date = datetime.fromisoformat(session.created_at).timestamp()
            if session_date < cutoff_date:
                if self.delete_session(session.session_id):
                    deleted_count += 1

        logger.info("Cleaned up %d old sessions", deleted_count)
        return deleted_count

    def find_cached_result(
        self, session_id: str, config: dict[str, Any]
    ) -> TrialResult | None:
        """
        Find a cached trial result for the given configuration.

        Args:
            session_id: Session identifier
            config: Configuration to look for

        Returns:
            Cached trial result if found, None otherwise
        """
        try:
            session = self.load_session(session_id)
            if not session:
                return None

            # Look for existing trial with same config
            if session.trials:
                for trial in session.trials:
                    if trial.error is None and self.compute_config_hash(
                        trial.config
                    ) == self.compute_config_hash(config):
                        logger.info("🎯 Cache hit! Found cached result for config")
                        return trial

            return None

        except Exception as e:
            logger.error("Error checking cache (%s)", type(e).__name__)
            return None

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about storage usage."""
        sessions = self.list_sessions()
        total_trials = sum(session.completed_trials for session in sessions)

        storage_size = 0
        if self.storage_path.exists():
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    storage_size += file_path.stat().st_size

        return {
            "storage_path": str(self.storage_path),
            "total_sessions": len(sessions),
            "total_trials": total_trials,
            "storage_size_bytes": storage_size,
            "storage_size_mb": (
                max(0.01, round(storage_size / (1024 * 1024), 2))
                if len(sessions) > 0
                else 0.01
            ),
        }

    def _normalize_config(self, value: Any) -> Any:
        """Normalize config values for consistent hashing.

        Args:
            value: Value to normalize (can be nested dict/list)

        Returns:
            Normalized value with floats rounded and dicts sorted
        """
        from traigent.utils.numpy_compat import convert_numpy_value, is_numpy_type

        if is_numpy_type(value):
            return self._normalize_config(convert_numpy_value(value))
        if isinstance(value, float):
            # Round floats to 8 decimal places for consistency
            return round(value, 8)
        elif isinstance(value, dict):
            # Recursively normalize and sort dict keys
            return {k: self._normalize_config(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            # Recursively normalize list items
            return [self._normalize_config(item) for item in value]
        elif isinstance(value, tuple):
            # Convert tuples to lists for consistent JSON serialization
            return [self._normalize_config(item) for item in value]
        else:
            # Return other types as-is
            return value

    def compute_config_hash(
        self, config: dict[str, Any], keys: list[str] | None = None
    ) -> str:
        """Compute stable hash for configuration deduplication.

        Args:
            config: Configuration dictionary to hash
            keys: Optional list of user config keys to include in the hash

        Returns:
            12-character hash string
        """
        # Dedup identity intentionally excludes underscore-namespaced values.
        # Those are realization/runtime metadata (for example ``_optuna*``,
        # ``_traigent*``, and ``__subset_indices__``), not user choices.
        user_config = {
            key: value
            for key, value in config.items()
            if isinstance(key, str) and not key.startswith("_")
        }

        # Filter to specified keys if provided.
        if keys:
            filtered_config = {
                key: user_config[key]
                for key in sorted(keys)
                if isinstance(key, str)
                and not key.startswith("_")
                and key in user_config
            }
        else:
            filtered_config = user_config

        # Normalize the config for consistent hashing
        normalized_config = self._normalize_config(filtered_config)

        # Serialize with consistent ordering
        config_str = json.dumps(
            normalized_config, sort_keys=True, separators=(",", ":")
        )

        # Generate short hash for efficiency
        return sha256(config_str.encode("utf-8")).hexdigest()[:12]

    def is_config_seen(
        self,
        function_name: str,
        dataset_name: str,
        config: dict[str, Any],
        keys: list[str] | None = None,
    ) -> bool:
        """Check if configuration has been evaluated in any previous session.

        Args:
            function_name: Name of the function being optimized
            dataset_name: Name of the dataset being used
            config: Configuration to check
            keys: Optional list of config keys to use for comparison

        Returns:
            True if configuration has been seen and successfully evaluated before
        """
        base_keys = self._dedup_base_keys(config, keys)

        try:
            # List all sessions for this function
            sessions = self.list_sessions()

            for session_info in sessions:
                if session_info.function_name != function_name:
                    continue

                # Load the actual session data
                session = self.load_session(session_info.session_id)
                if not session:
                    continue

                if self._session_dataset_name(session) != dataset_name:
                    continue

                if self._session_has_matching_trial(session, config, base_keys):
                    return True

        except Exception as e:
            logger.warning(
                "Error during config deduplication check (%s)", type(e).__name__
            )
            # Best effort - don't fail optimization due to dedup errors

        return False

    @staticmethod
    def _user_config_keys(names: Iterable[Any]) -> set[str]:
        """Names that are user choices (string, not underscore-namespaced)."""
        return {
            key for key in names if isinstance(key, str) and not key.startswith("_")
        }

    @classmethod
    def _dedup_base_keys(
        cls, config: dict[str, Any], keys: list[str] | None
    ) -> set[str] | None:
        user_config_keys = cls._user_config_keys(config)
        if not keys:
            return None
        return cls._user_config_keys(keys) | user_config_keys

    @staticmethod
    def _session_dataset_name(session: Any) -> Any:
        """Dataset a session ran on (prefer evaluation_set metadata)."""
        metadata = session.metadata
        session_dataset = metadata.get("evaluation_set") if metadata else None
        if not session_dataset:
            session_dataset = metadata.get("dataset_name") if metadata else None
        return session_dataset

    def _session_has_matching_trial(
        self, session: Any, config: dict[str, Any], base_keys: set[str] | None
    ) -> bool:
        """True when a successful trial of ``session`` has the same user config."""
        for trial in session.trials or ():
            # Check if trial was successful (no error)
            if trial.error is not None:
                continue
            # Compare the complete user-facing identity on both
            # sides. Unioning realized user keys preserves spec
            # awareness when a config space later expands or
            # narrows, while ignoring runtime-only metadata.
            if base_keys is None:
                compare_keys = None
            else:
                compare_keys = sorted(base_keys | self._user_config_keys(trial.config))
            target_hash = self.compute_config_hash(config, compare_keys)
            trial_hash = self.compute_config_hash(trial.config, compare_keys)
            if trial_hash == target_hash:
                return True
        return False

    def find_cached_result_across_sessions(
        self,
        function_name: str,
        dataset_name: str,
        config: dict[str, Any],
        keys: list[str] | None = None,
    ) -> TrialResult | None:
        """Find a cached result across all stored sessions.

        This helper scans historical sessions for a matching configuration that
        completed successfully. It is used by higher-level policies that attempt
        to reuse prior work before launching new trials.

        Args:
            function_name: Name of the function being optimized
            dataset_name: Name of the dataset being used
            config: Configuration to find result for
            keys: Optional list of config keys to use for comparison

        Returns:
            Cached trial result if found, None otherwise
        """

        target_hash = self.compute_config_hash(config, keys)

        try:
            for session_info in self.list_sessions():
                if session_info.function_name != function_name:
                    continue

                # Ensure dataset matches before attempting reuse
                session = self.load_session(session_info.session_id)
                if not session:
                    continue

                session_dataset = (
                    session.metadata.get("evaluation_set") if session.metadata else None
                )
                if not session_dataset:
                    session_dataset = (
                        session.metadata.get("dataset_name")
                        if session.metadata
                        else None
                    )

                if session_dataset != dataset_name:
                    continue

                cached = self.find_cached_result(session_info.session_id, config)
                if cached is None:
                    continue

                trial_hash = self.compute_config_hash(cached.config, keys)
                if trial_hash == target_hash:
                    return cached

        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning(
                "Error during cross-session cache lookup (%s)", type(exc).__name__
            )

        return None
