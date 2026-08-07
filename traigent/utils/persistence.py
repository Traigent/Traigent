"""File-based persistence system for Traigent optimization results."""

# Traceability: CONC-Layer-Data CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ..api.types import (
    OptimizationResult,
    TrialResult,
)
from ..utils.function_identity import sanitize_identifier

# Imported by full dotted path, never via `traigent.utils` — that barrel is
# eager and `traigent.api.types` imports from it, so re-exporting the manifest
# module there would cycle (see its module docstring).
from ..utils.optimization_result_persistence import (
    RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    decode_result,
    decode_trial_error,
    decode_trial_score,
    encode_result_fields,
    encode_trial_error,
    verify_envelope_version,
)
from ..utils.secure_path import safe_open, validate_path

logger = logging.getLogger(__name__)

# File name constants
METADATA_FILE = "metadata.json"
TRIALS_JSON_FILE = "trials.json.gz"
TRIALS_PKL_FILE = "trials.pkl.gz"
TRIALS_SUMMARY_FILE = "trials_summary.json"

_SAFE_PICKLE_GLOBALS = frozenset(
    {
        ("_codecs", "encode"),
        ("__builtin__", "complex"),
        ("__builtin__", "frozenset"),
        ("__builtin__", "set"),
        ("builtins", "complex"),
        ("builtins", "frozenset"),
        ("builtins", "set"),
        ("datetime", "datetime"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("traigent.api.types", "OptimizationStatus"),
        ("traigent.api.types", "Trial"),
        ("traigent.api.types", "TrialError"),
        ("traigent.api.types", "TrialResult"),
        ("traigent.api.types", "TrialStatus"),
        ("traigent.core.types", "OptimizationStatus"),
        ("traigent.core.types", "Trial"),
        ("traigent.core.types", "TrialResult"),
        ("traigent.core.types", "TrialStatus"),
        ("traigent.optimizers.results", "OptimizationResult"),
        ("traigent.optimizers.results", "Trial"),
    }
)


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler for legacy Traigent trial files with pinned safe globals."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _SAFE_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"Attempted to unpickle unsafe global: {module}.{name}"
            )
        return super().find_class(module, name)


def _safe_json_value(value: Any) -> Any:
    """Convert a value to a JSON-safe type, handling nested structures recursively.

    This handles:
    - Objects with to_dict() methods (EvaluationResult, ExampleResult, etc.)
    - datetime objects (converted to ISO format strings)
    - Nested dicts and lists (recursively processed)
    - Unknown types (converted to string representation as fallback)
    """
    if value is None:
        return None

    # Handle objects with to_dict() method
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()

    # Handle datetime objects
    if isinstance(value, datetime):
        return value.isoformat()

    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: _safe_json_value(v) for k, v in value.items()}

    # Handle lists recursively
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]

    # Handle tuples (convert to list)
    if isinstance(value, tuple):
        return [_safe_json_value(item) for item in value]

    # Handle sets (convert to list)
    if isinstance(value, set):
        return [_safe_json_value(item) for item in value]

    # Primitives that are already JSON-safe
    if isinstance(value, (str, int, float, bool)):
        return value

    # Handle numpy types (must come after primitives check)
    # numpy scalars have item() method to convert to Python native types
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    # Fallback: convert to string representation
    try:
        # Try to see if it's JSON serializable as-is
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        # Last resort: string representation
        return str(value)


def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Serialize metadata dict, converting objects with to_dict() methods.

    This handles EvaluationResult, ExampleResult, and other dataclass objects
    that need to be converted to dicts for JSON serialization. Recursively
    processes nested dicts and lists.
    """
    return cast(dict[str, Any], _safe_json_value(metadata))


def _rehydrate_evaluation_result(value: dict) -> Any:
    """Rehydrate an EvaluationResult from its dict representation."""
    from ..evaluators.base import EvaluationResult

    return EvaluationResult.from_dict(value)


def _rehydrate_example_results(value: list) -> list:
    """Rehydrate a list of ExampleResult from their dict representations."""
    from ..api.types import ExampleResult

    return [
        (
            ExampleResult.from_dict(item)
            if isinstance(item, dict) and "example_id" in item
            else item
        )
        for item in value
    ]


def _rehydrate_value(key: str, value: Any) -> Any:
    """Rehydrate a single value based on its key and type."""
    if key == "evaluation_result" and isinstance(value, dict):
        return _rehydrate_evaluation_result(value)
    if key == "example_results" and isinstance(value, list):
        return _rehydrate_example_results(value)
    if isinstance(value, dict):
        return _rehydrate_metadata(value)
    if isinstance(value, list):
        return [
            _rehydrate_metadata(item) if isinstance(item, dict) else item
            for item in value
        ]
    return value


def _rehydrate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Rehydrate metadata dict, converting dicts back to typed objects where possible.

    This reconstructs EvaluationResult and ExampleResult objects from their
    dict representations when loading from JSON.
    """
    if not metadata:
        return metadata

    return {key: _rehydrate_value(key, value) for key, value in metadata.items()}


class PersistenceManager:
    """Manages saving and loading optimization results to/from disk."""

    def __init__(self, base_dir: str | Path = ".traigent") -> None:
        """Initialize persistence manager.

        Args:
            base_dir: Base directory for storing optimization results
        """
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(exist_ok=True)

    def _resolve_path(self, path: Path, must_exist: bool = False) -> Path:
        return cast(Path, validate_path(path, self.base_dir, must_exist=must_exist))

    def _atomic_write_json(self, file_path: Path, data: Any) -> None:
        """Write JSON data atomically using temp file + rename pattern.

        This prevents data corruption if the process crashes during write.

        Args:
            file_path: Target file path
            data: Data to serialize as JSON
        """
        validated_path = self._resolve_path(file_path)
        temp_path = validated_path.with_suffix(
            f"{validated_path.suffix}.tmp.{os.getpid()}"
        )
        temp_path = self._resolve_path(temp_path)
        try:
            with safe_open(temp_path, self.base_dir, mode="w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(validated_path)  # Atomic rename on POSIX
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _atomic_write_gzip_json(self, file_path: Path, data: Any) -> None:
        """Write gzipped JSON data atomically.

        Args:
            file_path: Target file path
            data: Data to serialize as JSON
        """
        validated_path = self._resolve_path(file_path)
        temp_path = validated_path.with_suffix(
            f"{validated_path.suffix}.tmp.{os.getpid()}"
        )
        temp_path = self._resolve_path(temp_path)
        try:
            with gzip.open(temp_path, "wt") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(validated_path)  # Atomic rename on POSIX
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _atomic_write_gzip_pickle(self, file_path: Path, data: Any) -> None:
        """Write gzipped pickle data atomically.

        Args:
            file_path: Target file path
            data: Data to pickle
        """
        validated_path = self._resolve_path(file_path)
        temp_path = validated_path.with_suffix(
            f"{validated_path.suffix}.tmp.{os.getpid()}"
        )
        try:
            with gzip.open(temp_path, "wb") as f:
                pickle.dump(data, f)
            temp_path.replace(validated_path)  # Atomic rename on POSIX
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def save_result(self, result: OptimizationResult, name: str | None = None) -> str:
        """Save optimization result to disk.

        Args:
            result: Optimization result to save
            name: Optional name for the saved result

        Returns:
            Path to saved file
        """
        if name is None:
            # Generate name from metadata or generic name
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            func_slug = result.metadata.get("function_slug") or sanitize_identifier(
                str(result.metadata.get("function_name", "optimization"))
            )
            name = f"{func_slug}_{timestamp}"

        # Create subdirectory for this optimization
        result_dir = self._resolve_path(self.base_dir / name)
        result_dir.mkdir(exist_ok=True)

        # Save metadata as JSON
        metadata = {
            "function_identifier": result.metadata.get("function_name", "unknown"),
            "function_name": result.metadata.get(
                "function_display_name",
                result.metadata.get("function_name", "unknown"),
            ),
            "algorithm": result.algorithm,
            "objectives": _safe_json_value(result.objectives),
            "configuration_space": _safe_json_value(
                result.metadata.get("configuration_space", {})
            ),
            "best_score": _safe_json_value(result.best_score),
            "best_config": _safe_json_value(result.best_config),
            "preset_selection": _safe_json_value(
                result.preset_selection.to_dict()
                if result.preset_selection is not None
                else None
            ),
            "success_rate": _safe_json_value(result.success_rate),
            "duration": result.duration,
            "convergence_info": _safe_json_value(result.convergence_info),
            "created_at": datetime.now(UTC).isoformat(),
            "total_trials": len(result.trials),
            "successful_trials": len(result.successful_trials),
            # #1854 hardening: the aggregated winner's replicate identity lives
            # in session_summary["winning_trial_ids"]; without persisting it a
            # save->load round-trip silently degrades best_metrics to the
            # full-config-equality fallback (sol post-ship finding).
            "session_summary": _safe_json_value(result.metadata.get("session_summary")),
        }

        # #2031: the 15 curated keys above are the human/CLI-readable summary
        # (list_results sorts on created_at, can_resume matches on
        # function_name + configuration_space), and they are kept exactly as
        # they were. They are NOT a faithful record of the dataclass: they never
        # carried optimization_id, status, source, stop_reason, total_cost,
        # warnings, ... so load_result had nothing to restore them from. The
        # full restorable field set is written alongside them, under its own
        # schema version.
        #
        # This method has always accepted any object shaped like a result — it
        # read ~10 attributes — so a duck-typed one must not start raising here.
        # But "not an OptimizationResult" is not the same as "has nothing worth
        # recording": an object that does carry optimization_id, timestamp and
        # the rest encodes exactly as faithfully as the real dataclass, and
        # falling back to the curated artifact for it would throw away fields
        # this writer is holding. So the encode is attempted for every result and
        # only a genuine encoding failure downgrades the artifact.
        #
        # A real OptimizationResult is not given that escape hatch: it cannot
        # legitimately fail to encode (a dataclass always carries every
        # attribute), so a failure there means a type-violating field value —
        # e.g. `status="bogus"` — and silently writing a lossy artifact for it
        # would hide a caller bug behind a log line.
        try:
            result_fields = encode_result_fields(result)
        except (TypeError, ValueError) as exc:
            if isinstance(result, OptimizationResult):
                raise
            # Required fields (`optimization_id`, `timestamp`) have no declared
            # default to fall back to, and `status` may be an arbitrary string
            # rather than an OptimizationStatus. Rather than fabricate those
            # values or reject the caller, such a result gets exactly the
            # pre-#2031 curated artifact it has always got; load_result reads it
            # back through its `legacy_format="persistence"` branch.
            logger.warning(
                "Saving a %s rather than an OptimizationResult, and it cannot "
                "be encoded as a full result record (%s): writing the pre-#2031 "
                "summary artifact only. Fields such as optimization_id, status, "
                "source and total_cost will not survive load_result.",
                type(result).__name__,
                exc,
            )
        else:
            # Sanitizing is a *separate* failure class from encoding, and it is
            # deliberately not fatal even for a real OptimizationResult. The
            # strict re-raise above is justified by a field whose declared type
            # was violated; this step walks the free-form containers — above
            # all `metadata`, declared `dict[str, Any]`, which the encode now
            # carries in full where the pre-#2031 artifact only ever held a
            # curated four keys of it. Any object a user or plugin put in there
            # is in scope, and `_safe_json_value` calls `to_dict()` on anything
            # that has one, so an arbitrary caller exception can arrive here.
            # Letting it out would lose the whole artifact — the run is simply
            # not persisted — where the pre-#2031 writer wrote the curated one.
            # So this degrades to exactly that artifact, loudly. `Exception`,
            # not `(TypeError, ValueError)`: a third-party `to_dict()` raises
            # whatever it likes, and the point is that none of it destroys the
            # save.
            try:
                encoded_fields = _safe_json_value(result_fields)
            except Exception as exc:  # noqa: BLE001 - see comment above
                logger.warning(
                    "Could not convert the full result record of %s to a "
                    "JSON-safe form (%s), most likely because result.metadata "
                    "holds an object that cannot be serialized: writing the "
                    "pre-#2031 summary artifact only. Fields such as "
                    "optimization_id, status, source and total_cost will not "
                    "survive load_result.",
                    type(result).__name__,
                    exc,
                )
            else:
                metadata[SCHEMA_VERSION_KEY] = RESULT_SCHEMA_VERSION
                metadata["result_fields"] = encoded_fields

        self._atomic_write_json(result_dir / METADATA_FILE, metadata)

        # Save trials as compressed JSON (secure and portable)
        trials_data = []
        for trial in result.trials:
            # Serialize metadata, converting objects with to_dict() methods
            raw_metadata = trial.metadata if hasattr(trial, "metadata") else {}
            serialized_metadata = _serialize_metadata(raw_metadata)

            trial_dict = {
                # Persist the real id so the winning_trial_ids stamp (#1854)
                # still matches after a round-trip (load already reads it).
                # getattr: save tolerates duck-typed trials (CLI tests use
                # bare stand-ins), matching the hasattr guards below.
                "trial_id": getattr(trial, "trial_id", None),
                "config": _safe_json_value(trial.config),
                "metrics": _safe_json_value(
                    trial.metrics if hasattr(trial, "metrics") else {}
                ),
                "duration": trial.duration if hasattr(trial, "duration") else 0.0,
                "status": trial.status if hasattr(trial, "status") else "unknown",
                "timestamp": (
                    trial.timestamp.isoformat()
                    if hasattr(trial, "timestamp") and trial.timestamp
                    else None
                ),
                # load_result has always read this key; nothing ever wrote it,
                # so a failed trial's diagnosis was dropped by the round trip
                # while its FAILED status survived. getattr for the same reason
                # as trial_id above.
                "error_message": getattr(trial, "error_message", None),
                "metadata": serialized_metadata,
                # #2047: this format wrote neither key, so `error` and `score`
                # were lost at WRITE time here (the config_state format lost
                # them at read time instead). Without them a reloaded run
                # cannot tell a crashed trial from a badly-scoring one.
                # getattr for the same duck-typing reason as trial_id above.
                "error": encode_trial_error(getattr(trial, "error", None)),
                "score": getattr(trial, "score", None),
            }
            trials_data.append(trial_dict)

        self._atomic_write_gzip_json(result_dir / TRIALS_JSON_FILE, trials_data)

        # Also save as pickle for backward compatibility (will be deprecated)
        self._atomic_write_gzip_pickle(result_dir / TRIALS_PKL_FILE, result.trials)

        # Save successful trials summary as JSON for easy reading
        trials_summary = []
        for trial in result.successful_trials[:50]:  # Limit to first 50 for readability
            trials_summary.append(
                {
                    "config": _safe_json_value(trial.config),
                    "metrics": _safe_json_value(trial.metrics),
                    "duration": trial.duration,
                    "status": trial.status,
                }
            )

        self._atomic_write_json(result_dir / TRIALS_SUMMARY_FILE, trials_summary)

        return str(result_dir)

    def load_result(self, name: str) -> OptimizationResult:
        """Load optimization result from disk.

        Args:
            name: Name of the saved result

        Returns:
            Loaded optimization result

        Raises:
            FileNotFoundError: If result doesn't exist
            ValueError: If result data is corrupted
        """
        result_dir = self._resolve_path(self.base_dir / name)

        if not result_dir.exists():
            raise FileNotFoundError(
                f"Optimization result '{name}' not found in {self.base_dir}"
            )

        # Load metadata
        metadata_file = self._resolve_path(result_dir / METADATA_FILE)
        if not metadata_file.exists():
            raise ValueError(f"Metadata file missing for result '{name}'")

        with safe_open(metadata_file, self.base_dir, mode="r") as f:
            metadata = json.load(f)

        # Load trials using JSON instead of pickle for security
        trials_file = self._resolve_path(result_dir / TRIALS_JSON_FILE)
        pkl_file = self._resolve_path(result_dir / TRIALS_PKL_FILE)

        # Try JSON first (secure), fall back to pickle with warning
        trials: list[TrialResult]
        if trials_file.exists():
            validated_trials_file = self._resolve_path(
                result_dir / TRIALS_JSON_FILE, must_exist=True
            )
            with gzip.open(validated_trials_file, "rt") as f:
                trials_data = json.load(f)
                # Reconstruct TrialResult objects from JSON
                from ..api.types import TrialStatus

                trials = []
                for i, t in enumerate(trials_data):
                    # Rehydrate metadata objects if present
                    raw_metadata = t.get("metadata", {})
                    rehydrated_metadata = _rehydrate_metadata(raw_metadata)

                    trial = TrialResult(
                        trial_id=t.get("trial_id") or f"trial_{i}",
                        config=t["config"],
                        metrics=t.get("metrics", {}),
                        status=TrialStatus(t.get("status", "completed")),
                        duration=t.get("duration", 0.0),
                        timestamp=(
                            datetime.fromisoformat(t["timestamp"])
                            if t.get("timestamp")
                            else datetime.now(UTC)
                        ),
                        error_message=t.get("error_message"),
                        metadata=rehydrated_metadata,
                        # #2047. Absent in every artifact written before this
                        # format started emitting them, which decodes to None —
                        # the same value those trials already had. This
                        # loader's deliberate tolerance for missing keys
                        # (above) is preserved; only a PRESENT-but-corrupt
                        # payload raises.
                        error=decode_trial_error(
                            t.get("error"), artifact_name=str(validated_trials_file)
                        ),
                        score=decode_trial_score(
                            t.get("score"), artifact_name=str(validated_trials_file)
                        ),
                    )
                    trials.append(trial)
        elif pkl_file.exists():
            logger.warning(
                f"Loading legacy pickle file for '{name}' - consider re-saving in JSON format"
            )
            validated_pkl_file = self._resolve_path(
                result_dir / TRIALS_PKL_FILE, must_exist=True
            )
            with gzip.open(validated_pkl_file, "rb") as fp:
                trials = cast(list[TrialResult], RestrictedUnpickler(fp).load())
        else:
            raise ValueError(f"Trials file missing for result '{name}'")

        # Validate required metadata fields before reconstruction so a
        # truncated / hand-edited / legacy metadata.json surfaces as the
        # documented ValueError ("If result data is corrupted") rather than a
        # bare KeyError that callers catching ValueError would miss (#1962).
        required_keys = (
            "best_config",
            "best_score",
            "duration",
            "convergence_info",
            "objectives",
            "algorithm",
            "created_at",
            "function_name",
            "configuration_space",
        )
        for key in required_keys:
            if key not in metadata:
                raise ValueError(
                    f"Corrupted metadata for result '{name}': missing '{key}'"
                )

        # Reconstruct optimization result through the #2031 manifest. The old
        # hand-written constructor named 11 of the dataclass's 27 fields and
        # fabricated three of them (a "loaded_<name>" id, a COMPLETED status,
        # and the save time as the run timestamp).
        result_fields = metadata.get("result_fields")
        # The version is stamped twice — on the envelope and on the payload —
        # and only the payload's copy is decoded below. Checking that they agree
        # is what stops an envelope declaring a version this build cannot read
        # from being decoded anyway at whatever version its payload claims.
        verify_envelope_version(metadata, result_fields, artifact_name=name)
        if isinstance(result_fields, dict):
            # The decoded `metadata` is authoritative and is returned exactly as
            # the run recorded it. The curated keys around it are a *derived*
            # summary written for list_results / can_resume — several of them
            # (function_name, configuration_space) are copies with defaults
            # substituted at save time. Merging them back in would let a
            # `"function_name": "unknown"` or a `"configuration_space": {}`
            # placeholder override the authoritative persisted field, and would
            # break the verbatim metadata round trip pinned by #2026: a result
            # saved with `metadata == {}` would come back non-empty.
            return decode_result(result_fields, trials=trials, artifact_name=name)

        # Pre-#2031 artifact: the curated keys are all there ever was, so the
        # loader rebuilds `metadata` from them (see `_legacy_view`) — there is
        # no authoritative inner record to defer to.
        return decode_result(
            metadata,
            trials=trials,
            legacy_format="persistence",
            artifact_name=name,
        )

    def list_results(self) -> list[dict[str, Any]]:
        """List all saved optimization results.

        Returns:
            List of result metadata dictionaries (the curated summary keys; the
            full #2031 ``result_fields`` payload and its schema-version stamp
            are dropped — they are for ``load_result``, and a listing must not
            carry a second copy of every result's metadata, metrics and
            best_config).
        """
        results = []

        for result_dir in self.base_dir.iterdir():
            if result_dir.is_dir():
                metadata_file = self._resolve_path(result_dir / METADATA_FILE)
                if metadata_file.exists():
                    try:
                        with safe_open(metadata_file, self.base_dir, mode="r") as f:
                            metadata = json.load(f)
                        metadata.pop("result_fields", None)
                        # … and the version stamp that describes it: it says
                        # nothing about the curated summary keys, and a renderer
                        # that dumps every key of a listing entry would show it
                        # as if it were one of the result's own values.
                        metadata.pop(SCHEMA_VERSION_KEY, None)
                        metadata["name"] = result_dir.name
                        results.append(metadata)
                    except (json.JSONDecodeError, FileNotFoundError):
                        continue

        # Sort by creation time, newest first
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results

    def delete_result(self, name: str) -> bool:
        """Delete a saved optimization result.

        Args:
            name: Name of the result to delete

        Returns:
            True if deleted successfully, False if not found
        """
        result_dir = self.base_dir / name

        if not result_dir.exists():
            return False

        # Delete all files in the directory
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                file_path.unlink()

        # Remove the directory
        result_dir.rmdir()
        return True

    def get_result_hash(self, result: OptimizationResult) -> str:
        """Generate hash for optimization result to detect duplicates.

        Args:
            result: Optimization result

        Returns:
            SHA256 hash of result key properties
        """
        function_name = getattr(result, "function_name", None)
        if not function_name:
            metadata = result.metadata or {}
            function_name = metadata.get("function_name") or metadata.get(
                "function_identifier", "unknown"
            )

        configuration_space = getattr(result, "configuration_space", None)
        if not configuration_space:
            configuration_space = (result.metadata or {}).get("configuration_space", {})

        if isinstance(configuration_space, dict):
            configuration_space_repr = dict(sorted(configuration_space.items()))
        else:
            configuration_space_repr = configuration_space

        # Create deterministic representation
        key_data = {
            "function_name": function_name,
            "algorithm": result.algorithm,
            "objectives": sorted(result.objectives),
            "configuration_space": configuration_space_repr,
            "trial_count": len(result.trials),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:12]


class ResumableOptimization:
    """Enables resuming optimization from saved state."""

    def __init__(self, persistence_manager: PersistenceManager) -> None:
        """Initialize resumable optimization.

        Args:
            persistence_manager: Persistence manager instance
        """
        self.persistence_manager = persistence_manager

    def save_checkpoint(self, result: OptimizationResult, checkpoint_name: str) -> str:
        """Save optimization checkpoint.

        Args:
            result: Current optimization result
            checkpoint_name: Name for the checkpoint

        Returns:
            Path to saved checkpoint
        """
        return self.persistence_manager.save_result(
            result, f"checkpoint_{checkpoint_name}"
        )

    def load_checkpoint(self, checkpoint_name: str) -> OptimizationResult:
        """Load optimization checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint

        Returns:
            Loaded optimization result
        """
        checkpoint_key = f"checkpoint_{checkpoint_name}"
        return self.persistence_manager.load_result(checkpoint_key)

    def can_resume(
        self, function_name: str, configuration_space: dict[str, Any]
    ) -> str | None:
        """Check if optimization can be resumed.

        Args:
            function_name: Name of the function being optimized
            configuration_space: Configuration space

        Returns:
            Name of resumable checkpoint, or None if not found
        """
        results = self.persistence_manager.list_results()

        for result_info in results:
            if (
                result_info["function_name"] == function_name
                and result_info["configuration_space"] == configuration_space
                and result_info["name"].startswith("checkpoint_")
            ):
                return cast(str | None, result_info["name"])

        return None
