"""Production readiness features for Haystack pipeline optimization.

This module provides CLI interface, configuration file support, production-hardened
apply function, TVL export, experiment history export, and CI/CD integration.

Example usage:
    from traigent.integrations.haystack import (
        load_optimization_config,
        apply_config_production,
        export_tuned_config,
        load_tuned_config,
    )

    # Load config from YAML
    config = load_optimization_config("optimization.yaml")

    # Apply with validation
    apply_config_production(pipeline, best_config, validate=True, backup=True)

    # Export as TVL
    export_tuned_config(result, "optimized.tvl")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from traigent.integrations.haystack.execution import apply_config
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigMismatchError(Exception):
    """Raised when configuration doesn't match pipeline structure."""

    pass


class ConfigValidationError(Exception):
    """Raised when configuration file is invalid."""

    pass


@dataclass
class OptimizationConfig:
    """Configuration for optimization run.

    Loaded from YAML/JSON configuration files.

    Attributes:
        pipeline_module: Module path to pipeline (e.g., "myapp.pipeline:pipe").
        search_space: Path to search space TVL file or inline definition.
        targets: List of optimization targets.
        constraints: List of constraints.
        strategy: Optimization strategy.
        n_trials: Number of trials.
        n_parallel: Number of parallel evaluations.
        timeout_seconds: Optional time budget.
        checkpoint_path: Path for checkpointing.
        artifact_path: Path for saving artifacts.
        eval_dataset_path: Path to evaluation dataset.
        random_seed: Random seed for reproducibility.
    """

    pipeline_module: str = ""
    search_space: str | dict[str, Any] = ""
    targets: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = "bayesian"
    n_trials: int = 50
    n_parallel: int = 1
    timeout_seconds: float | None = None
    checkpoint_path: str | None = None
    artifact_path: str | None = None
    eval_dataset_path: str | None = None
    random_seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationConfig:
        """Create config from dictionary."""
        return cls(
            pipeline_module=data.get("pipeline_module", ""),
            search_space=data.get("search_space", ""),
            targets=data.get("targets", []),
            constraints=data.get("constraints", []),
            strategy=data.get("strategy", "bayesian"),
            n_trials=data.get("n_trials", 50),
            n_parallel=data.get("n_parallel", 1),
            timeout_seconds=data.get("timeout_seconds"),
            checkpoint_path=data.get("checkpoint_path"),
            artifact_path=data.get("artifact_path"),
            eval_dataset_path=data.get("eval_dataset_path"),
            random_seed=data.get("random_seed"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_module": self.pipeline_module,
            "search_space": self.search_space,
            "targets": self.targets,
            "constraints": self.constraints,
            "strategy": self.strategy,
            "n_trials": self.n_trials,
            "n_parallel": self.n_parallel,
            "timeout_seconds": self.timeout_seconds,
            "checkpoint_path": self.checkpoint_path,
            "artifact_path": self.artifact_path,
            "eval_dataset_path": self.eval_dataset_path,
            "random_seed": self.random_seed,
        }

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not self.pipeline_module:
            errors.append("pipeline_module is required")

        if not self.search_space:
            errors.append("search_space is required (path or inline)")

        if not self.targets:
            errors.append("At least one target is required")

        for i, target in enumerate(self.targets):
            if "name" not in target and "metric_name" not in target:
                errors.append(f"Target {i}: name/metric_name is required")
            if "direction" not in target:
                errors.append(f"Target {i}: direction is required")

        valid_strategies = [
            "bayesian",
            "tpe",
            "evolutionary",
            "nsga2",
            "random",
            "grid",
        ]
        if self.strategy not in valid_strategies:
            errors.append(
                f"Invalid strategy: {self.strategy}. "
                f"Valid: {', '.join(valid_strategies)}"
            )

        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")

        if self.n_parallel < 1:
            errors.append("n_parallel must be >= 1")

        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be > 0")

        return errors


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    """Load optimization configuration from YAML or JSON file.

    Args:
        path: Path to configuration file.

    Returns:
        Loaded OptimizationConfig.

    Raises:
        ConfigValidationError: If file is invalid or has validation errors.
    """
    path = Path(path)

    if not path.exists():
        raise ConfigValidationError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                # Try YAML first, then JSON
                content = f.read()
                try:
                    data = yaml.safe_load(content)
                except yaml.YAMLError:
                    data = json.loads(content)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigValidationError(f"Failed to parse config file: {e}")

    if not isinstance(data, dict):
        raise ConfigValidationError("Config must be a dictionary/object")

    config = OptimizationConfig.from_dict(data)

    # Validate
    errors = config.validate()
    if errors:
        raise ConfigValidationError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return config


def save_optimization_config(config: OptimizationConfig, path: str | Path) -> None:
    """Save optimization configuration to YAML or JSON file.

    Args:
        config: Configuration to save.
        path: Output path.
    """
    path = Path(path)

    data = config.to_dict()

    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(data, f, indent=2, default=str)


@dataclass
class ApplyBackup:
    """Backup of original pipeline configuration for rollback."""

    original_values: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def restore(self, pipeline: Any) -> None:
        """Restore original values to pipeline."""
        for param_path, value in self.original_values.items():
            parts = param_path.split(".")
            if len(parts) < 2:
                continue

            component_name = parts[0]
            param_name = ".".join(parts[1:])

            component = pipeline.get_component(component_name)
            if component is not None and hasattr(component, param_name):
                setattr(component, param_name, value)


def apply_config_production(
    pipeline: Any,
    config: dict[str, Any],
    validate: bool = True,
    backup: bool = False,
) -> ApplyBackup | None:
    """Apply configuration to pipeline with production-grade validation.

    Args:
        pipeline: Haystack Pipeline object.
        config: Configuration dict mapping parameter paths to values.
        validate: Whether to validate config matches pipeline structure.
        backup: Whether to create backup for rollback.

    Returns:
        ApplyBackup object if backup=True, otherwise None.

    Raises:
        ConfigMismatchError: If config doesn't match pipeline structure.
        TypeError: If parameter type doesn't match.
    """
    backup_obj = None

    if validate:
        # Validate all config keys exist in pipeline
        for param_path in config.keys():
            parts = param_path.split(".")
            if len(parts) < 2:
                raise ConfigMismatchError(
                    f"Invalid parameter path: {param_path}. "
                    f"Expected format: component.parameter"
                )

            component_name = parts[0]
            param_name = ".".join(parts[1:])

            component = pipeline.get_component(component_name)
            if component is None:
                raise ConfigMismatchError(
                    f"Component not found in pipeline: {component_name}"
                )

            if not hasattr(component, param_name):
                raise ConfigMismatchError(
                    f"Parameter not found: {param_name} on {component_name}"
                )

            # Type validation: check if new value type is compatible
            current_value = getattr(component, param_name)
            new_value = config[param_path]
            if current_value is not None and new_value is not None:
                current_type = type(current_value)
                new_type = type(new_value)
                # Allow numeric type coercion (int -> float, float -> int)
                numeric_types = (int, float)
                if not (
                    isinstance(new_value, current_type)
                    or (
                        isinstance(current_value, numeric_types)
                        and isinstance(new_value, numeric_types)
                    )
                ):
                    raise TypeError(
                        f"Type mismatch for {param_path}: "
                        f"expected {current_type.__name__}, got {new_type.__name__}"
                    )

    if backup:
        # Store original values
        original_values = {}
        for param_path in config.keys():
            parts = param_path.split(".")
            if len(parts) >= 2:
                component_name = parts[0]
                param_name = ".".join(parts[1:])
                component = pipeline.get_component(component_name)
                if component is not None and hasattr(component, param_name):
                    original_values[param_path] = getattr(component, param_name)

        backup_obj = ApplyBackup(original_values=original_values)

    # Apply using existing function
    apply_config(pipeline, config)

    logger.info(f"Applied {len(config)} configuration values to pipeline")

    return backup_obj


def rollback_config(pipeline: Any, backup: ApplyBackup) -> None:
    """Rollback pipeline to previous configuration.

    Args:
        pipeline: Haystack Pipeline object.
        backup: Backup object from apply_config_production.
    """
    backup.restore(pipeline)
    logger.info(
        f"Rolled back {len(backup.original_values)} values "
        f"(backup from {backup.timestamp})"
    )


@dataclass
class TunedConfig:
    """Tuned configuration with metadata.

    Exported as Tuned Config TVL format.
    """

    version: str = "1.0"
    pipeline_name: str = ""
    framework: str = "haystack"
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    constraints_satisfied: bool = True
    optimization_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "pipeline_name": self.pipeline_name,
            "framework": self.framework,
            "config": self.config,
            "metrics": self.metrics,
            "constraints_satisfied": self.constraints_satisfied,
            "optimization_metadata": self.optimization_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunedConfig:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            pipeline_name=data.get("pipeline_name", ""),
            framework=data.get("framework", "haystack"),
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            constraints_satisfied=data.get("constraints_satisfied", True),
            optimization_metadata=data.get("optimization_metadata", {}),
        )


def export_tuned_config(
    result: Any,
    output_path: str | Path,
    pipeline_name: str = "",
    include_history: bool = False,
) -> TunedConfig:
    """Export optimized configuration as Tuned Config TVL file.

    Args:
        result: OptimizationResult object.
        output_path: Path for output file.
        pipeline_name: Optional pipeline name for metadata.
        include_history: Whether to include full history in metadata.

    Returns:
        TunedConfig object.
    """
    output_path = Path(output_path)

    # Build metadata
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_trials": getattr(result, "total_trials", 0),
        "duration_seconds": getattr(result, "duration", 0),
    }

    # Add strategy if available
    if hasattr(result, "strategy"):
        metadata["strategy"] = result.strategy

    # Add warnings if present
    if hasattr(result, "warnings") and result.warnings:
        metadata["warnings"] = result.warnings

    # Determine constraints_satisfied from result
    # Check if result has explicit constraints_satisfied attribute, otherwise infer
    best_config = getattr(result, "best_config", {}) or {}
    if hasattr(result, "constraints_satisfied"):
        constraints_satisfied = result.constraints_satisfied
    elif not best_config:
        # No best config means constraints likely not satisfied
        constraints_satisfied = False
    else:
        # Default to True if we have a best config and no explicit info
        constraints_satisfied = True

    # Create tuned config
    tuned = TunedConfig(
        pipeline_name=pipeline_name,
        config=best_config,
        metrics=getattr(result, "best_metrics", {}) or {},
        constraints_satisfied=constraints_satisfied,
        optimization_metadata=metadata,
    )

    # Include history if requested
    if include_history and hasattr(result, "history"):
        tuned.optimization_metadata["history_summary"] = {
            "total_runs": len(result.history),
            "successful_runs": sum(
                1
                for t in result.history
                if hasattr(t, "is_successful") and t.is_successful
            ),
        }

    # Save to file
    data = tuned.to_dict()

    with open(output_path, "w") as f:
        if output_path.suffix in (".yaml", ".yml", ".tvl"):
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(data, f, indent=2, default=str)

    logger.info(f"Exported tuned config to {output_path}")

    return tuned


def load_tuned_config(path: str | Path) -> TunedConfig:
    """Load tuned configuration from TVL file.

    Args:
        path: Path to TVL file.

    Returns:
        TunedConfig object.
    """
    path = Path(path)

    with open(path) as f:
        if path.suffix in (".yaml", ".yml", ".tvl"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return TunedConfig.from_dict(data)


def export_experiment_history(
    result: Any,
    output_path: str | Path,
    format: str = "json",
) -> None:
    """Export full experiment history.

    Args:
        result: OptimizationResult object.
        output_path: Path for output file.
        format: 'json' or 'csv'.
    """
    output_path = Path(output_path)

    if format == "json":
        data = {
            "schema_version": "1.0",
            "exported_at": datetime.now(UTC).isoformat(),
            "summary": {
                "total_trials": getattr(result, "total_trials", 0),
                "duration_seconds": getattr(result, "duration", 0),
                "best_config": getattr(result, "best_config", {}),
                "best_metrics": getattr(result, "best_metrics", {}),
            },
            "trials": [],
        }

        # Add trial history
        if hasattr(result, "history"):
            for trial in result.history:
                trial_data = {
                    "trial_id": getattr(trial, "trial_id", ""),
                    "config": getattr(trial, "config", {}),
                    "metrics": getattr(trial, "metrics", {}),
                    "is_successful": getattr(trial, "is_successful", True),
                    "constraints_satisfied": getattr(
                        trial, "constraints_satisfied", True
                    ),
                    "duration": getattr(trial, "duration", 0),
                }
                data["trials"].append(trial_data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    elif format == "csv":
        import csv

        # Collect all metric keys
        metric_keys: set[str] = set()
        config_keys: set[str] = set()

        if hasattr(result, "history"):
            for trial in result.history:
                if hasattr(trial, "metrics"):
                    metric_keys.update(trial.metrics.keys())
                if hasattr(trial, "config"):
                    config_keys.update(trial.config.keys())

        sorted_metric_keys = sorted(metric_keys)
        sorted_config_keys = sorted(config_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["trial_id", "is_successful", "constraints_satisfied", "duration"]
            header.extend(f"config_{k}" for k in sorted_config_keys)
            header.extend(f"metric_{k}" for k in sorted_metric_keys)
            writer.writerow(header)

            # Rows
            if hasattr(result, "history"):
                for trial in result.history:
                    row = [
                        getattr(trial, "trial_id", ""),
                        getattr(trial, "is_successful", True),
                        getattr(trial, "constraints_satisfied", True),
                        getattr(trial, "duration", 0),
                    ]

                    config = getattr(trial, "config", {})
                    for k in sorted_config_keys:
                        row.append(config.get(k, ""))

                    metrics = getattr(trial, "metrics", {})
                    for k in sorted_metric_keys:
                        row.append(metrics.get(k, ""))

                    writer.writerow(row)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")

    logger.info(f"Exported experiment history to {output_path}")


def save_artifacts(
    result: Any,
    artifact_path: str | Path,
    pipeline_name: str = "",
) -> dict[str, Path]:
    """Save all optimization artifacts to directory.

    Args:
        result: OptimizationResult object.
        artifact_path: Directory path for artifacts.
        pipeline_name: Optional pipeline name for metadata.

    Returns:
        Dict mapping artifact names to paths.
    """
    artifact_path = Path(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Save best config as TVL
    config_path = artifact_path / "best_config.tvl"
    export_tuned_config(result, config_path, pipeline_name=pipeline_name)
    artifacts["best_config"] = config_path

    # Save experiment history as JSON
    history_path = artifact_path / "experiment_history.json"
    export_experiment_history(result, history_path, format="json")
    artifacts["experiment_history"] = history_path

    # Save summary
    summary_path = artifact_path / "summary.json"
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "pipeline_name": pipeline_name,
        "total_trials": getattr(result, "total_trials", 0),
        "duration_seconds": getattr(result, "duration", 0),
        "best_config": getattr(result, "best_config", {}),
        "best_metrics": getattr(result, "best_metrics", {}),
        "pareto_count": len(getattr(result, "pareto_configs", None) or []),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    artifacts["summary"] = summary_path

    logger.info(f"Saved {len(artifacts)} artifacts to {artifact_path}")

    return artifacts


def load_artifacts(artifact_path: str | Path) -> dict[str, Any]:
    """Load optimization artifacts from directory.

    Args:
        artifact_path: Directory containing artifacts.

    Returns:
        Dict with loaded artifacts.
    """
    artifact_path = Path(artifact_path)

    artifacts = {}

    # Load best config
    config_path = artifact_path / "best_config.tvl"
    if config_path.exists():
        artifacts["best_config"] = load_tuned_config(config_path)

    # Load summary
    summary_path = artifact_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            artifacts["summary"] = json.load(f)

    # Load history
    history_path = artifact_path / "experiment_history.json"
    if history_path.exists():
        with open(history_path) as f:
            artifacts["experiment_history"] = json.load(f)

    return artifacts


@dataclass
class CLIResult:
    """Result from CLI optimization run.

    Structured for CI/CD integration.
    """

    status: str  # "success", "failure", "no_valid_configs"
    exit_code: int
    best_config: dict[str, Any] | None = None
    best_score: float | None = None
    n_trials_completed: int = 0
    constraints_satisfied: bool = False
    best_unconstrained_config: dict[str, Any] | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "status": self.status,
                "exit_code": self.exit_code,
                "best_config": self.best_config,
                "best_score": self.best_score,
                "n_trials_completed": self.n_trials_completed,
                "constraints_satisfied": self.constraints_satisfied,
                "best_unconstrained_config": self.best_unconstrained_config,
                "error_message": self.error_message,
                "warnings": self.warnings,
            },
            indent=2,
            default=str,
        )


def create_cli_result(result: Any) -> CLIResult:
    """Create CLI result from OptimizationResult.

    Args:
        result: OptimizationResult object.

    Returns:
        CLIResult for CI/CD output.
    """
    best_config = getattr(result, "best_config", None)
    best_metrics = getattr(result, "best_metrics", {})
    warnings = getattr(result, "warnings", [])
    total_trials = getattr(result, "total_trials", 0)

    # Determine status
    if best_config is None:
        return CLIResult(
            status="no_valid_configs",
            exit_code=1,
            n_trials_completed=total_trials,
            warnings=warnings,
        )

    # Get primary score
    best_score = None
    if best_metrics:
        # Use first metric as primary
        best_score = next(iter(best_metrics.values()), None)

    return CLIResult(
        status="success",
        exit_code=0,
        best_config=best_config,
        best_score=best_score,
        n_trials_completed=total_trials,
        constraints_satisfied=True,
        warnings=warnings,
    )
