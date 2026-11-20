"""Weights & Biases (wandb) integration for Traigent experiment tracking."""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from numbers import Number
from pathlib import Path
from typing import Any, cast

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

    # Mock wandb for type hints
    class wandb:  # type: ignore[no-redef]
        @staticmethod
        def init(*args, **kwargs) -> None:
            """Stand-in used when wandb is not installed: intentionally a no-op."""

        @staticmethod
        def log(*args, **kwargs) -> None:
            """Stand-in used when wandb is not installed: intentionally a no-op."""

        @staticmethod
        def finish(*args, **kwargs) -> None:
            """Stand-in used when wandb is not installed: intentionally a no-op."""

        @staticmethod
        def save(*args, **kwargs) -> None:
            """Stand-in used when wandb is not installed: intentionally a no-op."""


from ...api.types import OptimizationResult, TrialResult
from ...core.objective_directions import resolve_objective_orientation
from ...utils.callbacks import OptimizationCallback
from ...utils.discrete_domains import (
    discrete_cardinality_for_config_param,
    discrete_values_for_config_param,
)
from ...utils.logging import get_logger
from ._offline_guard import offline_guarded

logger = get_logger(__name__)

#: Run-name prefix when the optimized function's name is not available.
_DEFAULT_RUN_LABEL = "traigent"


class TraigentWandBTracker:
    """Weights & Biases experiment tracker for Traigent optimizations."""

    @offline_guarded("W&B tracking")
    def __init__(
        self,
        project: str = "traigent-optimization",
        entity: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        auto_log: bool = True,
    ) -> None:
        """Initialize W&B tracker.

        Args:
            project: W&B project name
            entity: W&B entity (team/user)
            tags: Tags for the run
            notes: Notes for the run
            auto_log: Automatically log optimization results
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Please install it with: pip install wandb"
            )

        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.notes = notes
        self.auto_log = auto_log
        self.current_run = None

    @offline_guarded("W&B tracking")
    def start_optimization_run(
        self,
        function_name: str,
        objectives: list[str],
        configuration_space: dict[str, Any],
        run_name: str | None = None,
        additional_tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Start W&B run for optimization.

        Args:
            function_name: Name of function being optimized
            objectives: Optimization objectives
            configuration_space: Parameter search space
            run_name: Optional run name
            additional_tags: Additional tags for this run
            config: Additional configuration to log

        Returns:
            W&B run ID
        """
        if not run_name:
            timestamp = int(time.time())
            run_name = f"{function_name}_optimization_{timestamp}"

        # Combine tags
        all_tags = self.tags + (additional_tags or [])
        all_tags.extend(["traigent", "optimization", function_name])

        # Prepare config
        run_config = {
            "traigent": {
                "function_name": function_name,
                "objectives": objectives,
                "configuration_space": configuration_space,
                "num_objectives": len(objectives),
                "config_space_size": len(configuration_space),
            }
        }

        if config:
            run_config.update(config)

        # Initialize run
        self.current_run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            tags=all_tags,
            notes=self.notes,
            config=run_config,
            reinit=True,
        )

        # Log configuration space as artifact
        config_file = "configuration_space.json"
        with open(config_file, "w") as f:
            json.dump(configuration_space, f, indent=2)

        try:
            wandb.save(config_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to save configuration space artifact: %s", exc, exc_info=False
            )
        Path(config_file).unlink(missing_ok=True)

        run = self.current_run
        if run is None:
            raise RuntimeError("W&B run initialization failed")
        logger.info(f"Started W&B run: {run.id}")
        return cast(str, run.id)

    @offline_guarded("W&B tracking")
    def log_trial(
        self, trial: TrialResult, trial_number: int, step: int | None = None
    ) -> None:
        """Log individual trial to W&B.

        Args:
            trial: Trial result to log
            trial_number: Trial number in optimization
            step: Optional step number for metrics
        """
        if not self.current_run:
            logger.warning("No active W&B run. Call start_optimization_run first.")
            return

        if step is None:
            step = trial_number

        trial_id = trial.trial_id
        status_value = trial.status.value
        duration_value = float(trial.duration)

        config_values = self._mapping_or_warn(
            trial_id, getattr(trial, "config", None), "config", "config"
        )
        metric_values = self._mapping_or_warn(
            trial_id, getattr(trial, "metrics", None), "metrics", "metric"
        )

        # Prepare trial data for logging
        trial_data = {
            f"trial_{trial_number}/status": status_value,
            f"trial_{trial_number}/duration": duration_value,
        }

        for param_name, param_value in config_values.items():
            trial_data[f"trial_{trial_number}/config/{param_name}"] = (
                self._loggable_param(param_value)
            )

        for metric_name, metric_value in metric_values.items():
            if isinstance(metric_value, Number):
                trial_data[f"trial_{trial_number}/metrics/{metric_name}"] = metric_value
                trial_data[f"metrics/{metric_name}"] = metric_value
            else:
                logger.warning(
                    "Skipping non-numeric metric '%s' for trial %s",
                    metric_name,
                    trial.trial_id,
                )

        try:
            wandb.log(trial_data, step=step)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log trial data to W&B: %s", exc, exc_info=False)

        # Create trial artifact
        sanitized_config = {
            name: self._loggable_param(value) for name, value in config_values.items()
        }
        sanitized_metrics = {
            name: value
            for name, value in metric_values.items()
            if isinstance(value, Number)
        }

        trial_artifact_data = {
            "trial_id": trial_id,
            "trial_number": trial_number,
            "config": sanitized_config,
            "metrics": sanitized_metrics,
            "duration": duration_value,
            "status": status_value,
            "timestamp": self._timestamp_iso(trial),
        }

        # Save trial as JSON artifact. Coerce trial_number through int() so
        # the formatted filename can never carry path-traversal segments even
        # if a caller violates the int type contract. The coercion is inside
        # the try block so a bad caller falls into the same graceful-failure
        # path as I/O / serialisation errors instead of crashing log_trial().
        artifact_written = False
        trial_file: str | None = None
        try:
            trial_file = f"trial_{int(trial_number)}.json"
            with open(trial_file, "w", encoding="utf-8") as f:
                json.dump(trial_artifact_data, f, indent=2)
                artifact_written = True
        except (OSError, TypeError, ValueError) as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write trial artifact for %s: %s",
                trial_id,
                exc,
                exc_info=False,
            )

        if artifact_written and trial_file is not None:
            try:
                wandb.save(trial_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to save trial artifact to W&B: %s", exc, exc_info=False
                )
        if trial_file is not None:
            Path(trial_file).unlink(missing_ok=True)

    @staticmethod
    def _mapping_or_warn(
        trial_id: Any, raw: Any, field_name: str, logging_noun: str
    ) -> dict[str, Any]:
        """Return ``raw`` as a dict, warning when it is a non-empty non-mapping."""
        if isinstance(raw, Mapping):
            return dict(raw)
        if raw not in (None, {}):
            logger.warning(
                "Trial %s %s is not a mapping; skipping %s logging",
                trial_id,
                field_name,
                logging_noun,
            )
        return {}

    @staticmethod
    def _loggable_param(value: Any) -> Any:
        if isinstance(value, (str, Number, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _timestamp_iso(trial: Any) -> str | None:
        timestamp_attr = getattr(trial, "timestamp", None)
        if timestamp_attr is None:
            return None
        iso_method = getattr(timestamp_attr, "isoformat", None)
        if not callable(iso_method):
            return None
        try:
            return cast("str | None", iso_method())
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _add_trial_statistics(
        summary_data: dict[str, Any], result: OptimizationResult
    ) -> None:
        successful_trials = [
            t
            for t in result.trials
            if hasattr(t, "status") and t.status.value == "completed"
        ]
        summary_data["optimization/successful_trials"] = len(successful_trials)
        summary_data["optimization/success_rate"] = len(successful_trials) / len(
            result.trials
        )

        # Calculate metric statistics
        if not (successful_trials and result.best_metrics):
            return
        for metric_name in result.best_metrics.keys():
            metric_values = [
                t.metrics.get(metric_name, 0)
                for t in successful_trials
                if hasattr(t, "metrics") and metric_name in t.metrics
            ]
            if not metric_values:
                continue
            mean_val = sum(metric_values) / len(metric_values)
            std_val = (
                sum((x - mean_val) ** 2 for x in metric_values) / len(metric_values)
            ) ** 0.5
            summary_data[f"statistics/{metric_name}_mean"] = mean_val
            summary_data[f"statistics/{metric_name}_std"] = std_val
            summary_data[f"statistics/{metric_name}_min"] = min(metric_values)
            summary_data[f"statistics/{metric_name}_max"] = max(metric_values)

    @offline_guarded("W&B tracking")
    def log_optimization_result(
        self, result: OptimizationResult, dataset_path: str | None = None
    ) -> None:
        """Log complete optimization result to W&B.

        Args:
            result: Optimization result to log
            dataset_path: Optional path to evaluation dataset
        """
        if not self.current_run:
            logger.warning("No active W&B run. Call start_optimization_run first.")
            return

        # Log summary metrics
        summary_data = {
            "optimization/total_trials": len(result.trials),
            "optimization/duration": result.duration,
            "optimization/status": result.status.value,
        }

        # Log best configuration and metrics
        if result.best_config:
            for param_name, param_value in result.best_config.items():
                summary_data[f"best_config/{param_name}"] = param_value

        if result.best_metrics:
            for metric_name, metric_value in result.best_metrics.items():
                summary_data[f"best_metrics/{metric_name}"] = metric_value

        # Calculate trial statistics
        if result.trials:
            self._add_trial_statistics(summary_data, result)

        # Log metadata
        if result.metadata:
            for key, value in result.metadata.items():
                summary_data[f"metadata/{key}"] = value

        # Log summary
        wandb.log(summary_data)

        # Save complete result as artifact
        result_data = {
            "best_config": result.best_config,
            "best_metrics": result.best_metrics,
            "total_trials": len(result.trials),
            "duration": result.duration,
            "status": result.status.value,
            "metadata": result.metadata,
            "trial_summary": [
                {
                    "trial_id": t.trial_id,
                    "config": t.config,
                    "metrics": t.metrics,
                    "duration": t.duration,
                    "status": t.status.value,
                }
                for t in result.trials
            ],
        }

        result_file = "optimization_result.json"
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        wandb.save(result_file)
        Path(result_file).unlink(missing_ok=True)

        # Log dataset if provided
        if dataset_path and Path(dataset_path).exists():
            wandb.save(dataset_path)

        logger.info(f"Logged optimization result to W&B run: {self.current_run.id}")

    @offline_guarded("W&B tracking")
    def end_optimization_run(self) -> None:
        """End current W&B run."""
        if self.current_run:
            wandb.finish()
            logger.info(f"Ended W&B run: {self.current_run.id}")
            self.current_run = None

    @offline_guarded("W&B tracking")
    def log_hyperparameter_sweep(
        self, sweep_config: dict[str, Any], function_name: str
    ) -> str:
        """Create W&B sweep for hyperparameter optimization.

        Args:
            sweep_config: W&B sweep configuration
            function_name: Name of function being optimized

        Returns:
            Sweep ID
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not available for sweep")

        # Add Traigent metadata to sweep config
        sweep_config.setdefault("metadata", {})
        sweep_config["metadata"]["traigent_function"] = function_name
        sweep_config["metadata"]["framework"] = "traigent"

        sweep_id = wandb.sweep(sweep_config, project=self.project, entity=self.entity)

        logger.info(f"Created W&B sweep: {sweep_id}")
        return cast(str, sweep_id)


class WandBOptimizationCallback(OptimizationCallback):
    """Callback for automatic W&B logging during optimization."""

    def __init__(self, tracker: TraigentWandBTracker) -> None:
        """Initialize callback with W&B tracker.

        Args:
            tracker: TraigentWandBTracker instance
        """
        self.tracker = tracker
        self.trial_count = 0
        self._function_name: str | None = None

    def on_run_settings(self, settings: dict[str, Any]) -> None:
        """Remember the optimized function's name for the W&B run name."""
        name = settings.get("function_name")
        if isinstance(name, str) and name:
            self._function_name = name

    def on_optimization_start(  # type: ignore[override]
        self,
        config_space: dict[str, Any] | None = None,
        objectives: list[str] | None = None,
        algorithm: str | None = None,
        *,
        function_name: str | None = None,
        configuration_space: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when optimization starts.

        Accepts the ``OptimizationCallback`` signature
        ``(config_space, objectives, algorithm)``. The legacy keyword form
        ``function_name=..., configuration_space=...`` still works.
        """
        if isinstance(config_space, str):
            # Legacy positional order (function_name, objectives, config_space),
            # possibly mixed with keywords.
            function_name = function_name or config_space
            config_space = algorithm if isinstance(algorithm, dict) else None
        space = config_space if config_space is not None else configuration_space
        name = function_name or self._function_name or _DEFAULT_RUN_LABEL
        self.tracker.start_optimization_run(
            function_name=name,
            objectives=list(objectives or []),
            configuration_space=dict(space or {}),
            additional_tags=kwargs.get("wandb_tags", []),
            config=kwargs.get("wandb_config", {}),
        )

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts (nothing to log until it completes)."""
        return None

    def on_trial_complete(  # type: ignore[override]
        self, trial: TrialResult, progress: Any = None
    ) -> None:
        """Called when trial completes."""
        self.trial_count += 1
        self.tracker.log_trial(trial, self.trial_count)

    def on_optimization_complete(
        self, result: OptimizationResult, dataset_path: str | None = None
    ) -> None:
        """Called when optimization completes."""
        self.tracker.log_optimization_result(result, dataset_path)
        self.tracker.end_optimization_run()


# Convenience aliases for backward compatibility (defined later)
WandBIntegration = TraigentWandBTracker


# Convenience functions
def create_wandb_tracker(
    project: str = "traigent-optimization",
    entity: str | None = None,
    tags: list[str] | None = None,
    *,
    notes: str | None = None,
    auto_log: bool = True,
) -> TraigentWandBTracker:
    """Create W&B tracker for Traigent.

    Args:
        project: W&B project name
        entity: W&B entity (team/user)
        tags: Tags for runs

    Returns:
        Configured W&B tracker
    """
    return TraigentWandBTracker(project, entity, tags, notes=notes, auto_log=auto_log)


def enable_wandb_autolog(
    project: str = "traigent-optimization",
    entity: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> WandBOptimizationCallback:
    """Enable automatic W&B logging for Traigent optimizations.

    Args:
        project: W&B project name
        entity: W&B entity (team/user)
        tags: Tags for runs

    Returns:
        W&B callback for use with @traigent.optimize
    """
    tracker = create_wandb_tracker(project, entity, tags, notes=notes)
    return WandBOptimizationCallback(tracker)


def log_traigent_optimization(
    result: OptimizationResult,
    function_name: str,
    objectives: list[str],
    configuration_space: dict[str, Any],
    dataset_path: str | None = None,
    project: str = "traigent-optimization",
    entity: str | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> str:
    """Log Traigent optimization result to W&B.

    Args:
        result: Optimization result
        function_name: Name of optimized function
        objectives: Optimization objectives
        configuration_space: Parameter search space
        dataset_path: Path to evaluation dataset
        project: W&B project name
        entity: W&B entity
        run_name: Run name
        tags: Tags for the run

    Returns:
        W&B run ID
    """
    tracker = create_wandb_tracker(project, entity, tags, notes=notes)

    run_id = tracker.start_optimization_run(
        function_name=function_name,
        objectives=objectives,
        configuration_space=configuration_space,
        run_name=run_name,
    )

    # Log all trials
    for i, trial in enumerate(result.trials):
        tracker.log_trial(trial, i + 1)

    # Log final result
    tracker.log_optimization_result(result, dataset_path)
    tracker.end_optimization_run()

    return run_id


_WANDB_MAX_GRID_VALUES = 10_000


def _is_unit_or_no_step(param_range: Mapping[str, Any]) -> bool:
    step = param_range.get("step")
    if step is None:
        return True
    is_int = str(param_range.get("type", "")).lower() in {"int", "integer"}
    return is_int and step == 1


def _wandb_int_or_float_bounds(param_range: dict[str, Any]) -> dict[str, Any]:
    """Plain min/max domain for a typed numeric range dict without a step."""
    bounds: dict[str, Any] = {
        "min": param_range["low"],
        "max": param_range["high"],
    }
    range_type = str(param_range.get("type", "")).lower()
    if range_type in {"int", "integer"}:
        bounds["distribution"] = "int_uniform"
    elif range_type == "float":
        # Explicit float domain: integer-valued bounds must not be
        # inferred as int_uniform by W&B (#3211).
        bounds = {
            "min": float(param_range["low"]),
            "max": float(param_range["high"]),
            "distribution": "uniform",
        }
    return bounds


def _wandb_stepped_values(
    param_name: str, param_range: dict[str, Any]
) -> dict[str, Any]:
    # Stepped range: export the exact grid so the sweep never leaves it.
    # Bound the size arithmetically before materializing the grid.
    count = discrete_cardinality_for_config_param(param_range) or 0
    if count < 1 or count > _WANDB_MAX_GRID_VALUES:
        raise ValueError(
            f"Parameter '{param_name}' has a stepped range with "
            f"{count} values; W&B export supports "
            f"1..{_WANDB_MAX_GRID_VALUES}"
        )
    grid = discrete_values_for_config_param(param_range) or ()
    return {"values": list(grid)}


def _wandb_parameter_spec(param_name: str, param_range: Any) -> dict[str, Any]:
    if isinstance(param_range, list):
        # Categorical parameter
        return {"values": param_range}
    if isinstance(param_range, tuple) and len(param_range) == 2:
        # Continuous parameter
        min_val, max_val = param_range
        return {"min": min_val, "max": max_val}
    if (
        isinstance(param_range, dict)
        and "low" in param_range
        and "high" in param_range
        and not param_range.get("log")
        and _is_unit_or_no_step(param_range)
    ):
        # Typed numeric range dict without a step, or an integer range
        # with step 1 (e.g. IntRange): a plain min/max domain
        return _wandb_int_or_float_bounds(param_range)
    if (
        isinstance(param_range, dict)
        and "low" in param_range
        and "high" in param_range
        and param_range.get("step") is not None
        and not param_range.get("log")
    ):
        return _wandb_stepped_values(param_name, param_range)
    # Single value - treat as constant
    return {"value": param_range}


def create_wandb_sweep_config(
    function_name: str,
    configuration_space: dict[str, Any],
    objectives: list[str],
    method: str = "bayes",
    orientations: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create W&B sweep configuration from Traigent config space.

    Args:
        function_name: Name of function to optimize
        configuration_space: Traigent configuration space
        objectives: Optimization objectives
        method: Sweep method ('bayes', 'grid', 'random')
        orientations: Optional ``{objective: "maximize" | "minimize"}``
            declarations. The sweep goal follows the primary objective's
            direction: SDK-owned names such as ``cost`` and ``latency``
            minimize, ``accuracy`` maximizes, and a custom name must be
            declared here.

    Returns:
        W&B sweep configuration

    Raises:
        ValueError: If the primary objective has no declared or SDK-owned
            direction (a wrong goal would steer the sweep to the worst config).
    """
    primary_objective = objectives[0]
    goal = resolve_objective_orientation(
        primary_objective, (orientations or {}).get(primary_objective)
    )
    # Convert Traigent config space to W&B format
    wandb_parameters: dict[str, dict[str, Any]] = {}

    for param_name, param_range in configuration_space.items():
        wandb_parameters[param_name] = _wandb_parameter_spec(param_name, param_range)

    # Create sweep config
    sweep_config = {
        "method": method,
        "parameters": wandb_parameters,
        "metric": {
            "name": f"best_metrics/{primary_objective}",  # Primary objective
            "goal": goal,
        },
        "name": f"{function_name}_sweep",
        "description": f"Traigent optimization sweep for {function_name}",
    }

    return sweep_config


# Define remaining aliases after functions are defined
def init_wandb_run(
    function_name: str,
    objectives: list[str] | Sequence[str],
    configuration_space: dict[str, Any] | Mapping[str, Any],
    *,
    run_name: str | None = None,
    project: str = "traigent-optimization",
    entity: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    auto_log: bool = True,
    additional_tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Initialize a W&B run using the provided configuration."""
    if not isinstance(function_name, str) or not function_name.strip():
        raise ValueError("function_name must be a non-empty string")

    if isinstance(objectives, Sequence) and not isinstance(objectives, (str, bytes)):
        objective_list = [str(obj) for obj in objectives]
    else:
        raise TypeError("objectives must be a sequence of strings")

    if not objective_list:
        raise ValueError("objectives must contain at least one item")

    if isinstance(configuration_space, Mapping):
        config_space_dict = dict(configuration_space)
    elif isinstance(configuration_space, dict):
        config_space_dict = configuration_space
    else:
        raise TypeError("configuration_space must be a mapping of search parameters")

    tracker = create_wandb_tracker(
        project=project,
        entity=entity,
        tags=tags,
        notes=notes,
        auto_log=auto_log,
    )

    return tracker.start_optimization_run(
        function_name=function_name,
        objectives=objective_list,
        configuration_space=config_space_dict,
        run_name=run_name,
        additional_tags=additional_tags,
        config=config,
    )


log_optimization_to_wandb = log_traigent_optimization


def log_trial_to_wandb(tracker, trial, trial_num):
    return tracker.log_trial(trial, trial_num)


def log_final_results_to_wandb(tracker, result):
    return tracker.log_optimization_result(result)


create_optimization_report = create_wandb_sweep_config
