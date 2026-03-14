"""
Local storage manager for Traigent optimization sessions.
Inspired by DeepEval's approach to local JSON file storage.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Performance FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..api.types import OptimizationStatus
from ..utils.exceptions import TraigentStorageError
from ..utils.function_identity import sanitize_identifier
from ..utils.logging import get_logger
from ..utils.objectives import is_minimization_objective
from ..utils.secure_path import safe_open, validate_path

logger = get_logger(__name__)


@dataclass
class TrialResult:
    """Individual trial result in an optimization session."""

    trial_id: int
    config: dict[str, Any]
    score: float | None
    timestamp: str
    metadata: dict[str, Any | None] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OptimizationSession:
    """Complete optimization session data."""

    session_id: str
    function_name: str
    created_at: str
    updated_at: str
    status: str  # not_started, pending, running, completed, failed, cancelled
    total_trials: int
    completed_trials: int
    best_config: dict[str, Any | None] | None = None
    best_score: float | None = None
    baseline_score: float | None = None
    trials: list[TrialResult] | None = None
    optimization_config: dict[str, Any | None] | None = None
    metadata: dict[str, Any | None] | None = None

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
        return cls(**data)


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
            logger.debug(f"Initialized storage directories at {self.storage_path}")
        except Exception as e:
            raise TraigentStorageError(
                f"Failed to create storage directories: {e}"
            ) from e

    def _is_process_alive(self, pid: int) -> bool:
        """Best-effort check whether a process is alive."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _read_lock_pid(self, lock_path: Path) -> int | None:
        """Read owning PID from lock file, if present."""
        try:
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

    def _try_cleanup_stale_lock(self, lock_path: Path, lock_name: str) -> bool:
        """Remove stale lock when owner PID is no longer running."""
        owner_pid = self._read_lock_pid(lock_path)
        if owner_pid is None or self._is_process_alive(owner_pid):
            return False

        try:
            os.unlink(str(lock_path))
            logger.warning(
                "Removed stale lock '%s' owned by dead pid %s", lock_name, owner_pid
            )
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
                logger.debug(f"Acquired lock: {lock_name}")
                try:
                    yield  # Lock acquired
                finally:
                    os.close(fd)
                    try:
                        os.unlink(str(lock_path))  # Clean up lock file
                        logger.debug(f"Released lock: {lock_name}")
                    except OSError:
                        pass  # Another process may have cleaned it
                break
            except FileExistsError:
                # Lock is held by another process
                if self._try_cleanup_stale_lock(lock_path, lock_name):
                    continue
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock '{lock_name}' after {timeout}s"
                    ) from None
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.5)  # Exponential backoff with cap

    def create_session(
        self,
        function_name: str,
        optimization_config: dict[str, Any | None] | None = None,
        metadata: dict[str, Any | None] | None = None,
    ) -> str:
        """
        Create a new optimization session.

        Args:
            function_name: Name of the function being optimized
            optimization_config: Configuration for the optimization
            metadata: Additional metadata

        Returns:
            session_id: Unique identifier for the session
        """
        timestamp = datetime.now(UTC)
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
        logger.info(f"Created optimization session: {session_id}")
        return session_id

    def update_session_status(self, session_id: str, status: str) -> None:
        """Update session status."""
        session = self.load_session(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now(UTC).isoformat()
            self._save_session(session)
            logger.debug(f"Updated session {session_id} status to {status}")

    def add_trial_result(
        self,
        session_id: str,
        config: dict[str, Any],
        score: float | None,
        metadata: dict[str, Any | None] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Add a trial result to the session.

        Args:
            session_id: Session identifier
            config: Configuration that was tested
            score: Score achieved with this configuration (may be None when unavailable)
            metadata: Additional trial metadata
            error: Error message if trial failed
        """
        session = self.load_session(session_id)
        if not session:
            raise TraigentStorageError(f"Session {session_id} not found") from None

        if session.trials is None:
            session.trials = []

        trial_result = TrialResult(
            trial_id=len(session.trials) + 1,
            config=config,
            score=score,
            timestamp=datetime.now(UTC).isoformat(),
            metadata=metadata,
            error=error,
        )

        session.trials.append(trial_result)
        session.completed_trials = len(session.trials)
        session.updated_at = datetime.now(UTC).isoformat()

        # Update best score if this is better
        if score is not None:
            primary_objective = self._resolve_primary_objective_name(
                session.optimization_config
            )
            is_minimize = is_minimization_objective(primary_objective)
            if session.best_score is None:
                is_better = True
            elif is_minimize:
                is_better = score < session.best_score
            else:
                is_better = score > session.best_score
            if is_better:
                session.best_score = score
                session.best_config = config.copy()

        self._save_session(session)
        logger.debug(f"Added trial result to session {session_id}: score={score}")

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

    def finalize_session(
        self, session_id: str, status: str | None = None
    ) -> OptimizationSession | None:
        """
        Finalize an optimization session.

        Args:
            session_id: Session identifier
            status: Final status (completed, failed, cancelled)

        Returns:
            The finalized session data or None if not found
        """
        session = self.load_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None

        session.status = status or OptimizationStatus.COMPLETED.value
        session.updated_at = datetime.now(UTC).isoformat()
        self._save_session(session)

        logger.info(f"Finalized session {session_id} with status {status}")
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

            # Convert trial data back to TrialResult objects
            trials = []
            for trial_data in data.get("trials", []):
                trials.append(TrialResult(**trial_data))

            data["trials"] = trials

            # Handle backward compatibility for missing timestamp fields
            from datetime import datetime

            current_time = datetime.now(UTC).isoformat()

            # Provide default values for missing required fields
            if "created_at" not in data:
                data["created_at"] = current_time
            if "updated_at" not in data:
                data["updated_at"] = current_time

            return OptimizationSession(**data)

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
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
                logger.info(f"Deleted session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
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
            logger.error(f"Unsupported export format: {format}")
            return False

        session = self.load_session(session_id)
        if not session:
            raise TraigentStorageError(f"Session {session_id} not found") from None

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

            logger.info(f"Exported session {session_id} to {path_obj}")
            return True

        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return False

    def get_session_summary(self, session_id: str) -> dict[str, Any | None] | None:
        """Get a summary of session statistics."""
        session = self.load_session(session_id)
        if not session:
            return None

        if not session.trials:
            return {
                "session_id": session_id,
                "status": session.status,
                "trials": 0,
                "best_score": None,
                "improvement": None,
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
            raise TraigentStorageError(
                f"Failed to save session {session.session_id}: {e}"
            ) from e

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

        logger.info(f"Cleaned up {deleted_count} old sessions (older than {days} days)")
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
                    if trial.config == config and trial.error is None:
                        logger.info(
                            f"🎯 Cache hit! Found cached result for config: {config}"
                        )
                        return trial

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
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
            keys: Optional list of keys to include in hash (for config_space alignment)

        Returns:
            12-character hash string
        """
        # Filter to specified keys if provided
        if keys:
            filtered_config = {k: config.get(k) for k in sorted(keys) if k in config}
        else:
            filtered_config = config

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
        target_hash = self.compute_config_hash(config, keys)

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

                # Check dataset match (prefer evaluation_set metadata)
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

                # Check trials for matching config
                if session.trials:
                    for trial in session.trials:
                        # Check if trial was successful (no error)
                        if trial.error is None:
                            trial_hash = self.compute_config_hash(trial.config, keys)
                            if trial_hash == target_hash:
                                return True

        except Exception as e:
            logger.warning(f"Error during config deduplication check: {e}")
            # Best effort - don't fail optimization due to dedup errors

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
            logger.warning("Error during cross-session cache lookup: %s", exc)

        return None
