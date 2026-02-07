"""Configuration state management for optimized functions.

Manages the lifecycle of optimization configuration: loading, saving,
applying, and exporting best configurations. Extracted from
OptimizedFunction to reduce class complexity.
"""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.utils.exceptions import ConfigurationError, OptimizationStateError
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import (
    PathTraversalError,
    safe_open,
    validate_path,
    validate_user_path,
)

logger = get_logger(__name__)


class OptimizationState(Enum):
    """Lifecycle state of an OptimizedFunction.

    States:
        UNOPTIMIZED: Before any optimization has been run.
        OPTIMIZING: During an active optimization run.
        OPTIMIZED: After optimization has completed successfully.
        ERROR: Optimization failed.
    """

    UNOPTIMIZED = auto()
    OPTIMIZING = auto()
    OPTIMIZED = auto()
    ERROR = auto()


class ConfigStateManager:
    """Manages optimization config state: results, history, load/save/export.

    This class is owned by OptimizedFunction and manages all config persistence
    and lifecycle state. It uses a callback to re-wrap the function when
    configuration changes.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        default_config: dict[str, Any],
        local_storage_path: str | None,
        configuration_space: dict[str, Any] | None,
        auto_load_best: bool,
        load_from: str | None,
        setup_wrapper_callback: Callable[[], None],
    ) -> None:
        """Initialize config state manager.

        Args:
            func: The original user function
            default_config: Default configuration to use
            local_storage_path: Path for local optimization logs
            configuration_space: Config space definition
            auto_load_best: Whether to auto-load best config on init
            load_from: Explicit path to load config from
            setup_wrapper_callback: Callback to re-wrap function with new config
        """
        self.func = func
        self.default_config = default_config
        self.local_storage_path = local_storage_path
        self.configuration_space = configuration_space
        self._auto_load_best = auto_load_best
        self._load_from = load_from
        self._setup_wrapper_callback = setup_wrapper_callback

        # Core state
        self._state = OptimizationState.UNOPTIMIZED
        self._state_lock = threading.RLock()
        self._optimization_results: OptimizationResult | None = None
        self._optimization_history: list[OptimizationResult] = []
        self._current_config: dict[str, Any] = default_config.copy()
        self._best_config: dict[str, Any] | None = None

    # -- Properties --------------------------------------------------------

    @property
    def state(self) -> OptimizationState:
        """Get the current lifecycle state."""
        return self._state

    @state.setter
    def state(self, value: OptimizationState) -> None:
        self._state = value

    @property
    def state_lock(self) -> threading.RLock:
        """Access the state lock for atomic operations."""
        return self._state_lock

    @property
    def optimization_results(self) -> OptimizationResult | None:
        return self._optimization_results

    @optimization_results.setter
    def optimization_results(self, value: OptimizationResult | None) -> None:
        self._optimization_results = value

    @property
    def optimization_history(self) -> list[OptimizationResult]:
        return self._optimization_history

    @property
    def current_config(self) -> dict[str, Any]:
        """Get the configuration this function uses when called.

        Raises:
            OptimizationStateError: If accessed during an active optimization.
        """
        with self._state_lock:
            if self._state == OptimizationState.OPTIMIZING:
                raise OptimizationStateError(
                    "Cannot access current_config during an active optimization. "
                    "Use traigent.get_config() to access the current trial's "
                    "configuration within your optimized function.",
                    current_state=self._state.name,
                    expected_states=["UNOPTIMIZED", "OPTIMIZED", "ERROR"],
                )
            return self._current_config.copy()

    @property
    def current_config_raw(self) -> dict[str, Any]:
        """Direct access to current config without lock/state check (internal use)."""
        return self._current_config

    @current_config_raw.setter
    def current_config_raw(self, value: dict[str, Any]) -> None:
        self._current_config = value

    @property
    def best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization."""
        return self._best_config.copy() if self._best_config else None

    @best_config.setter
    def best_config(self, value: dict[str, Any] | None) -> None:
        self._best_config = value

    # -- Query methods -----------------------------------------------------

    def get_best_config(self) -> dict[str, Any] | None:
        """Get the best configuration found during optimization."""
        if self._optimization_results:
            best: dict[str, Any] = self._optimization_results.best_config
            return best
        return None

    def get_optimization_results(self) -> OptimizationResult | None:
        """Get the latest optimization results."""
        return self._optimization_results

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of all optimization runs."""
        return self._optimization_history.copy()

    def is_optimization_complete(self) -> bool:
        """Check if optimization has been completed."""
        return self._optimization_results is not None

    # -- State mutation methods --------------------------------------------

    def reset_optimization(self) -> None:
        """Reset optimization state and restore default configuration."""
        self._optimization_results = None
        self._optimization_history = []
        self._current_config = self.default_config.copy()
        self._best_config = None
        self._state = OptimizationState.UNOPTIMIZED
        self._setup_wrapper_callback()
        logger.info(f"Reset optimization state for {self.func.__name__}")

    def set_config(self, config: dict[str, Any]) -> None:
        """Set current configuration manually."""
        with self._state_lock:
            self._current_config = config.copy()
            self._setup_wrapper_callback()
        logger.debug(f"Set configuration for {self.func.__name__}: {config}")

    def apply_best_config(
        self,
        results: OptimizationResult | None = None,
        *,
        get_wrapped_func: Callable[[], Any] | None = None,
        set_wrapped_func: Callable[[Any], None] | None = None,
    ) -> bool:
        """Apply best configuration from optimization results.

        Args:
            results: OptimizationResult to use (defaults to latest optimization)
            get_wrapped_func: Getter for the wrapped function (for rollback)
            set_wrapped_func: Setter for the wrapped function (for rollback)

        Returns:
            True if configuration applied successfully

        Raises:
            ConfigurationError: If no optimization results are available
        """
        if results is None:
            results = self._optimization_results

        if not results or not results.best_config:
            raise ConfigurationError(
                "No optimization results available to apply. "
                "Please run optimization first using .optimize()"
            )

        with self._state_lock:
            old_config = self._current_config.copy()
            old_best = self._best_config.copy() if self._best_config else None
            old_wrapped_func = get_wrapped_func() if get_wrapped_func else None
            try:
                self._current_config = results.best_config.copy()
                self._best_config = results.best_config.copy()
                self._setup_wrapper_callback()
            except Exception:
                self._current_config = old_config
                self._best_config = old_best
                if set_wrapped_func and old_wrapped_func is not None:
                    set_wrapped_func(old_wrapped_func)
                raise

        logger.info(
            f"Applied best config for {self.func.__name__}: {results.best_config} "
            f"(previous: {old_config})"
        )

        return True

    # -- Persistence methods -----------------------------------------------

    def save_optimization_results(self, path: str) -> None:
        """Save optimization results to file.

        Raises:
            ConfigurationError: If no optimization results to save
        """
        if not self._optimization_results:
            raise ConfigurationError("No optimization results to save")

        from dataclasses import asdict

        result_dict = asdict(self._optimization_results)
        output_path = Path(path).expanduser()
        base_dir = (
            output_path.parent if output_path.is_absolute() else Path.cwd().resolve()
        )
        output_path = validate_path(output_path, base_dir, must_exist=False)
        with safe_open(output_path, base_dir, mode="w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Saved optimization results to {output_path}")

    def load_optimization_results(self, path: str) -> None:
        """Load optimization results from file.

        Raises:
            ConfigurationError: If results cannot be loaded
        """
        try:
            input_path = Path(path).expanduser()
            base_dir = (
                input_path.parent if input_path.is_absolute() else Path.cwd().resolve()
            )
            input_path = validate_path(input_path, base_dir, must_exist=True)
            with safe_open(input_path, base_dir, mode="r", encoding="utf-8") as f:
                result_dict = json.load(f)

            from traigent.api.types import TrialResult, TrialStatus

            trials = []
            for trial_data in result_dict.get("trials", []):
                trial = TrialResult(
                    trial_id=trial_data["trial_id"],
                    config=trial_data["config"],
                    metrics=trial_data["metrics"],
                    status=TrialStatus(trial_data["status"]),
                    duration=trial_data["duration"],
                    timestamp=datetime.fromisoformat(trial_data["timestamp"]),
                    error_message=trial_data.get("error_message"),
                    metadata=trial_data.get("metadata", {}),
                )
                trials.append(trial)

            self._optimization_results = OptimizationResult(
                trials=trials,
                best_config=result_dict["best_config"],
                best_score=result_dict["best_score"],
                optimization_id=result_dict["optimization_id"],
                duration=result_dict["duration"],
                convergence_info=result_dict["convergence_info"],
                status=OptimizationStatus(result_dict["status"]),
                objectives=result_dict["objectives"],
                algorithm=result_dict["algorithm"],
                timestamp=datetime.fromisoformat(result_dict["timestamp"]),
                metadata=result_dict.get("metadata", {}),
            )
            self._optimization_history.append(self._optimization_results)

            if self._optimization_results.best_config:
                self._current_config = self._optimization_results.best_config.copy()
                self._best_config = self._optimization_results.best_config.copy()
                self._setup_wrapper_callback()

            if self._optimization_results.status == OptimizationStatus.COMPLETED:
                self._state = OptimizationState.OPTIMIZED
            elif self._optimization_results.status == OptimizationStatus.FAILED:
                self._state = OptimizationState.ERROR
            else:
                self._state = OptimizationState.OPTIMIZED

            logger.info(f"Loaded optimization results from {path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to load optimization results: {e}") from e

    # -- Auto-load methods -------------------------------------------------

    def maybe_auto_load_config(self) -> None:
        """Auto-load configuration if requested via decorator parameters or env var."""
        config_path: str | None = None

        if self._load_from:
            config_path = self._load_from
            logger.debug(f"Using explicit load_from path: {config_path}")

        if not config_path:
            env_path = os.environ.get("TRAIGENT_CONFIG_PATH")
            if env_path:
                config_path = env_path
                logger.debug(f"Using TRAIGENT_CONFIG_PATH: {config_path}")

        if not config_path and self._auto_load_best:
            config_path = self._find_latest_config_path()
            if config_path:
                logger.debug(f"Auto-found latest config: {config_path}")

        if config_path:
            try:
                loaded_config = self._load_config_from_path(config_path)
                if loaded_config:
                    self._current_config = loaded_config.copy()
                    self._best_config = loaded_config.copy()
                    self._setup_wrapper_callback()
                    logger.info(
                        f"Auto-loaded config for {self.func.__name__} from {config_path}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to auto-load config from {config_path}: {e}. "
                    "Function will use default configuration."
                )

    def _find_latest_config_path(self) -> str | None:
        """Find the most recent best_config file in optimization logs."""
        log_dirs = [
            Path.cwd() / ".traigent" / "optimization_logs",
            Path(os.environ.get("TRAIGENT_RESULTS_FOLDER", Path.home() / ".traigent"))
            / "optimization_logs",
        ]

        if self.local_storage_path:
            log_dirs.insert(0, Path(self.local_storage_path) / "optimization_logs")

        func_name = getattr(self.func, "__name__", "unknown")

        for log_dir in log_dirs:
            experiments_dir = log_dir / "experiments" / func_name / "runs"
            if not experiments_dir.exists():
                continue

            run_dirs = sorted(
                [d for d in experiments_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )

            for run_dir in run_dirs:
                for config_file in [
                    run_dir / "artifacts" / "best_config_v2.json",
                    run_dir / "artifacts" / "best_config.json",
                ]:
                    if config_file.exists():
                        return str(config_file)

        return None

    @staticmethod
    def _load_config_from_path(path: str) -> dict[str, Any] | None:
        """Load configuration from a JSON file."""
        try:
            config_path = validate_user_path(path, for_write=False)
            if not config_path.exists():
                logger.warning(f"Config file not found: {path}")
                return None

            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)

            if "config" in data:
                return dict(data["config"])
            elif "best_config" in data:
                return dict(data["best_config"])
            elif isinstance(data, dict) and not any(
                k in data for k in ["trials", "metrics", "metadata"]
            ):
                return dict(data)
            else:
                logger.warning(
                    f"Unrecognized config format in {path}. "
                    "Expected 'config' or 'best_config' key, or direct config dict."
                )
                return None

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file {path}: {e}")
            return None
        except PathTraversalError as e:
            logger.warning(f"Security: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error loading config from {path}: {e}")
            return None

    # -- Export methods ----------------------------------------------------

    def export_config(
        self,
        path: str | Path,
        *,
        format: str = "slim",  # noqa: A002
        include_metadata: bool = True,
    ) -> Path:
        """Export the best configuration to a file."""
        if not self._best_config:
            raise ConfigurationError(
                "No best configuration available to export. "
                "Please run optimization first using .optimize() or load a config."
            )

        output_path: Path = validate_user_path(path, for_write=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "slim":
            export_data = self._create_slim_export(include_metadata)
        elif format == "full":
            export_data = self._create_full_export(include_metadata)
        else:
            raise ConfigurationError(
                f"Unknown export format: {format}. Use 'slim' or 'full'."
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported config for {self.func.__name__} to {output_path}")
        return output_path

    def _create_slim_export(self, include_metadata: bool) -> dict[str, Any]:
        """Create a slim export suitable for git and deployment."""
        from traigent import __version__

        export: dict[str, Any] = {
            "config": dict(self._best_config) if self._best_config else {}
        }

        if include_metadata:
            export["function_name"] = getattr(self.func, "__name__", "unknown")
            export["exported_at"] = datetime.now().isoformat()
            export["traigent_version"] = __version__

            if self._optimization_results and self._optimization_results.best_metrics:
                export["metrics"] = {
                    k: v
                    for k, v in self._optimization_results.best_metrics.items()
                    if not k.startswith("_")
                }

        return export

    def _create_full_export(self, include_metadata: bool) -> dict[str, Any]:
        """Create a full export including trial history."""
        export = self._create_slim_export(include_metadata)

        if self._optimization_results:
            export["trials"] = [
                {
                    "trial_id": t.trial_id,
                    "config": t.config,
                    "metrics": t.metrics,
                    "status": t.status if hasattr(t, "status") else "completed",
                }
                for t in (self._optimization_results.trials or [])
            ]

            export["optimization"] = {
                "total_trials": len(self._optimization_results.trials),
                "stop_reason": self._optimization_results.stop_reason,
                "duration_seconds": getattr(
                    self._optimization_results, "duration_seconds", None
                ),
            }

        if self.configuration_space:
            if hasattr(self.configuration_space, "to_dict"):
                export["configuration_space"] = self.configuration_space.to_dict()
            else:
                export["configuration_space"] = dict(self.configuration_space)

        return export
