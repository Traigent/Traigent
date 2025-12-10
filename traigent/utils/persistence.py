"""File-based persistence system for TraiGent optimization results."""

# Traceability: CONC-Layer-Data CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ..api.types import OptimizationResult, OptimizationStatus, TrialResult
from ..utils.function_identity import sanitize_identifier

logger = logging.getLogger(__name__)


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


def _rehydrate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Rehydrate metadata dict, converting dicts back to typed objects where possible.

    This reconstructs EvaluationResult and ExampleResult objects from their
    dict representations when loading from JSON.
    """
    if not metadata:
        return metadata

    result: dict[str, Any] = {}

    for key, value in metadata.items():
        if key == "evaluation_result" and isinstance(value, dict):
            # Rehydrate EvaluationResult
            from ..evaluators.base import EvaluationResult

            result[key] = EvaluationResult.from_dict(value)
        elif key == "example_results" and isinstance(value, list):
            # Rehydrate list of ExampleResult
            from ..api.types import ExampleResult

            result[key] = [
                (
                    ExampleResult.from_dict(item)
                    if isinstance(item, dict) and "example_id" in item
                    else item
                )
                for item in value
            ]
        elif isinstance(value, dict):
            # Recursively process nested dicts
            result[key] = _rehydrate_metadata(value)
        elif isinstance(value, list):
            # Process lists that might contain rehydratable objects
            result[key] = [
                _rehydrate_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


class PersistenceManager:
    """Manages saving and loading optimization results to/from disk."""

    def __init__(self, base_dir: str | Path = ".traigent") -> None:
        """Initialize persistence manager.

        Args:
            base_dir: Base directory for storing optimization results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

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
        result_dir = self.base_dir / name
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
            "success_rate": _safe_json_value(result.success_rate),
            "duration": result.duration,
            "convergence_info": _safe_json_value(result.convergence_info),
            "created_at": datetime.now(UTC).isoformat(),
            "total_trials": len(result.trials),
            "successful_trials": len(result.successful_trials),
        }

        with open(result_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save trials as compressed JSON (secure and portable)
        trials_data = []
        for trial in result.trials:
            # Serialize metadata, converting objects with to_dict() methods
            raw_metadata = trial.metadata if hasattr(trial, "metadata") else {}
            serialized_metadata = _serialize_metadata(raw_metadata)

            trial_dict = {
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
                "metadata": serialized_metadata,
            }
            trials_data.append(trial_dict)

        with gzip.open(result_dir / "trials.json.gz", "wt") as f:
            json.dump(trials_data, f, indent=2)

        # Also save as pickle for backward compatibility (will be deprecated)
        with gzip.open(result_dir / "trials.pkl.gz", "wb") as f:
            pickle.dump(result.trials, f)

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

        with open(result_dir / "trials_summary.json", "w") as f:
            json.dump(trials_summary, f, indent=2)

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
        result_dir = self.base_dir / name

        if not result_dir.exists():
            raise FileNotFoundError(
                f"Optimization result '{name}' not found in {self.base_dir}"
            )

        # Load metadata
        metadata_file = result_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Metadata file missing for result '{name}'")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Load trials using JSON instead of pickle for security
        trials_file = result_dir / "trials.json.gz"
        pkl_file = result_dir / "trials.pkl.gz"

        # Try JSON first (secure), fall back to pickle with warning
        trials: list[TrialResult]
        if trials_file.exists():
            with gzip.open(trials_file, "rt") as f:
                trials_data = json.load(f)
                # Reconstruct TrialResult objects from JSON
                from ..api.types import TrialStatus

                trials = []
                for i, t in enumerate(trials_data):
                    # Rehydrate metadata objects if present
                    raw_metadata = t.get("metadata", {})
                    rehydrated_metadata = _rehydrate_metadata(raw_metadata)

                    trial = TrialResult(
                        trial_id=t.get("trial_id", f"trial_{i}"),
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
                    )
                    trials.append(trial)
        elif pkl_file.exists():
            logger.warning(
                f"Loading legacy pickle file for '{name}' - consider re-saving in JSON format"
            )
            with gzip.open(pkl_file, "rb") as fp:
                # Only load pickle from trusted sources with restricted imports
                import pickle

                class RestrictedUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Only allow safe classes
                        ALLOWED_MODULES = {
                            "traigent.core.types",
                            "traigent.optimizers.results",
                            "__builtin__",
                            "builtins",
                            "collections",
                            "datetime",
                        }
                        if module not in ALLOWED_MODULES:
                            raise pickle.UnpicklingError(
                                f"Attempted to unpickle unsafe module: {module}"
                            )
                        return super().find_class(module, name)

                trials = cast(list[TrialResult], RestrictedUnpickler(fp).load())
        else:
            raise ValueError(f"Trials file missing for result '{name}'")

        # Reconstruct optimization result
        result = OptimizationResult(
            trials=trials,
            best_config=metadata["best_config"],
            best_score=metadata["best_score"],
            optimization_id=f"loaded_{name}",
            duration=metadata["duration"],
            convergence_info=metadata["convergence_info"],
            status=OptimizationStatus.COMPLETED,
            objectives=metadata["objectives"],
            algorithm=metadata["algorithm"],
            timestamp=datetime.fromisoformat(metadata["created_at"]),
            metadata={
                "function_name": metadata["function_name"],
                "configuration_space": metadata["configuration_space"],
            },
        )

        return result

    def list_results(self) -> list[dict[str, Any]]:
        """List all saved optimization results.

        Returns:
            List of result metadata dictionaries
        """
        results = []

        for result_dir in self.base_dir.iterdir():
            if result_dir.is_dir():
                metadata_file = result_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
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
