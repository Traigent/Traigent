"""Advanced optimization for Haystack pipelines.

This module provides Bayesian optimization, NSGA-II evolutionary optimization,
multi-objective optimization, Pareto frontier computation, and parallel execution
for Haystack pipeline optimization.

Example usage:
    from traigent.integrations.haystack import (
        HaystackOptimizer,
        OptimizationTarget,
        HaystackEvaluator,
    )

    optimizer = HaystackOptimizer(
        evaluator=evaluator,
        config_space={"generator.temperature": (0.0, 1.0)},
        targets=[
            OptimizationTarget("accuracy", "maximize"),
            OptimizationTarget("total_cost", "minimize"),
        ],
        strategy="bayesian",  # or "evolutionary", "tpe", "random"
        n_trials=50,
    )

    result = await optimizer.optimize()
    print(result.best_config)
    print(result.pareto_configs)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.integrations.haystack.evaluator import HaystackEvaluator

logger = get_logger(__name__)


class OptimizationDirection(str, Enum):
    """Direction for optimization objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class OptimizationTarget:
    """Defines an optimization objective.

    Attributes:
        metric_name: Name of the metric to optimize.
        direction: Whether to maximize or minimize the metric.
        weight: Optional weight for weighted multi-objective optimization.
    """

    metric_name: str
    direction: OptimizationDirection | str = OptimizationDirection.MAXIMIZE
    weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.direction, str):
            self.direction = OptimizationDirection(self.direction.lower())


@dataclass
class TrialResult:
    """Result of a single optimization trial.

    Attributes:
        trial_id: Unique identifier for the trial.
        config: Configuration used for this trial.
        metrics: Metric values from evaluation.
        constraints_satisfied: Whether constraints were satisfied.
        duration: Time taken for evaluation.
        timestamp: When the trial was completed.
    """

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    constraints_satisfied: bool = True
    duration: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_successful(self) -> bool:
        """Check if trial completed successfully."""
        return len(self.metrics) > 0


@dataclass
class OptimizationResult:
    """Result of an optimization run.

    Attributes:
        best_config: Best configuration found.
        best_metrics: Metrics for the best configuration.
        history: All trial results in chronological order.
        pareto_configs: Pareto-optimal configurations.
        pareto_metrics: Metrics for Pareto-optimal configurations.
        total_trials: Total number of trials run.
        duration: Total optimization time.
        warnings: Any warnings generated during optimization.
    """

    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    history: list[TrialResult] = field(default_factory=list)
    pareto_configs: list[dict[str, Any]] = field(default_factory=list)
    pareto_metrics: list[dict[str, float]] = field(default_factory=list)
    total_trials: int = 0
    duration: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def ranked_runs(self) -> list[TrialResult]:
        """Get trials ranked by primary objective (best first)."""
        if not self.history:
            return []

        # Find the first successful trial to determine primary objective
        successful = [t for t in self.history if t.is_successful]
        if not successful:
            return []

        # Get primary objective from first trial's metrics
        primary_metric = list(successful[0].metrics.keys())[0] if successful else None
        if not primary_metric:
            return self.history

        # Sort by primary metric (descending by default for maximize)
        return sorted(
            successful,
            key=lambda t: t.metrics.get(primary_metric, float("-inf")),
            reverse=True,
        )


def compute_pareto_frontier(
    trials: list[TrialResult],
    targets: list[OptimizationTarget],
    require_constraints: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Compute Pareto-optimal configurations from trial results.

    A configuration is Pareto-optimal if no other configuration is better
    in all objectives.

    Args:
        trials: List of trial results.
        targets: Optimization targets with directions.
        require_constraints: Only include constraint-satisfying configs.

    Returns:
        Tuple of (pareto_configs, pareto_metrics).
    """
    if not trials or not targets:
        return [], []

    # Filter by constraints if required
    candidates = trials
    if require_constraints:
        candidates = [t for t in trials if t.constraints_satisfied and t.is_successful]

    if not candidates:
        return [], []

    # Extract objective values with direction signs
    # For minimize: keep original values (lower is better)
    # For maximize: negate values (so lower is still better)
    objective_names = [t.metric_name for t in targets]
    directions = [t.direction for t in targets]

    points = []
    for trial in candidates:
        point = []
        has_all_metrics = True
        for name, direction in zip(objective_names, directions, strict=True):
            value = trial.metrics.get(name)
            if value is None:
                # Skip trials with missing metrics - they can't be on Pareto frontier
                has_all_metrics = False
                break
            if direction == OptimizationDirection.MAXIMIZE:
                value = -value  # Negate so lower is still better
            point.append(value)
        if has_all_metrics:
            points.append((point, trial))

    # Find Pareto-optimal points
    pareto_indices = []
    for i, (point_i, _) in enumerate(points):
        is_dominated = False
        for j, (point_j, _) in enumerate(points):
            if i == j:
                continue
            # Check if point_j dominates point_i
            # point_j dominates if it's <= in all objectives and < in at least one
            if all(
                pj <= pi for pj, pi in zip(point_j, point_i, strict=True)
            ) and any(pj < pi for pj, pi in zip(point_j, point_i, strict=True)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)

    # Extract Pareto configs and metrics
    pareto_configs = []
    pareto_metrics = []
    for idx in pareto_indices:
        _, trial = points[idx]
        pareto_configs.append(trial.config)
        pareto_metrics.append(trial.metrics)

    return pareto_configs, pareto_metrics


def rank_by_metric(
    trials: list[TrialResult],
    metric_name: str,
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
    require_constraints: bool = True,
) -> list[TrialResult]:
    """Rank trials by a specific metric.

    Args:
        trials: List of trial results.
        metric_name: Metric to rank by.
        direction: Optimization direction.
        require_constraints: Only include constraint-satisfying configs.

    Returns:
        Sorted list of trials (best first).
    """
    candidates = trials
    if require_constraints:
        candidates = [t for t in trials if t.constraints_satisfied and t.is_successful]
    else:
        candidates = [t for t in trials if t.is_successful]

    if not candidates:
        return []

    reverse = direction == OptimizationDirection.MAXIMIZE
    return sorted(
        candidates,
        key=lambda t: t.metrics.get(
            metric_name, float("-inf") if reverse else float("inf")
        ),
        reverse=reverse,
    )


class HaystackOptimizer:
    """Advanced optimizer for Haystack pipelines.

    Supports Bayesian optimization, NSGA-II evolutionary optimization,
    multi-objective optimization, and parallel execution.

    Attributes:
        evaluator: HaystackEvaluator for pipeline evaluation.
        config_space: Configuration space to search.
        targets: Optimization targets.
        strategy: Optimization strategy.
        n_trials: Maximum number of trials.
    """

    def __init__(
        self,
        evaluator: HaystackEvaluator,
        config_space: dict[str, Any],
        targets: list[OptimizationTarget] | None = None,
        strategy: str = "bayesian",
        n_trials: int = 50,
        n_parallel: int = 1,
        timeout_seconds: float | None = None,
        checkpoint_path: str | None = None,
        random_seed: int | None = None,
        constraints: list[Any] | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            evaluator: HaystackEvaluator instance.
            config_space: Dict mapping parameter names to search ranges.
                Use tuples (low, high) for continuous params,
                lists for categorical params.
            targets: List of optimization targets. Defaults to accuracy maximize.
            strategy: Optimization strategy - "bayesian", "evolutionary",
                "tpe", "random", "grid".
            n_trials: Maximum number of trials.
            n_parallel: Number of parallel evaluations.
            timeout_seconds: Optional time budget.
            checkpoint_path: Path for checkpointing.
            random_seed: Random seed for reproducibility.
            constraints: Constraints from evaluator (auto-detected if None).
        """
        self.evaluator = evaluator
        self.config_space = config_space
        self.targets = targets or [OptimizationTarget("accuracy", "maximize")]
        self.strategy = strategy.lower()
        self.n_trials = n_trials
        self.n_parallel = n_parallel
        self.timeout_seconds = timeout_seconds
        self.checkpoint_path = checkpoint_path
        self.random_seed = random_seed

        # Use constraints from evaluator if not provided
        if constraints is None and hasattr(evaluator, "constraints"):
            self.constraints = evaluator.constraints
        else:
            self.constraints = constraints or []

        # Internal state
        self._optimizer: Any = None
        self._history: list[TrialResult] = []
        self._start_time: float | None = None
        self._trial_counter = 0

    def _create_optimizer(self) -> Any:
        """Create the underlying optimizer based on strategy."""
        objectives = [t.metric_name for t in self.targets]
        directions = [
            "maximize" if t.direction == OptimizationDirection.MAXIMIZE else "minimize"
            for t in self.targets
        ]

        if self.strategy in ("bayesian", "tpe"):
            try:
                from traigent.optimizers.optuna_optimizer import OptunaTPEOptimizer

                return OptunaTPEOptimizer(
                    config_space=self.config_space,
                    objectives=objectives,
                    max_trials=self.n_trials,
                    directions=directions,
                )
            except ImportError:
                # Fallback to native Bayesian optimizer
                from traigent.optimizers.bayesian import BayesianOptimizer

                return BayesianOptimizer(
                    config_space=self.config_space,
                    objectives=objectives,
                    random_seed=self.random_seed,
                )

        elif self.strategy in ("evolutionary", "nsga2", "nsga-ii"):
            from traigent.optimizers.optuna_optimizer import OptunaNSGAIIOptimizer

            return OptunaNSGAIIOptimizer(
                config_space=self.config_space,
                objectives=objectives,
                max_trials=self.n_trials,
                directions=directions,
            )

        elif self.strategy == "random":
            try:
                from traigent.optimizers.optuna_optimizer import OptunaRandomOptimizer

                return OptunaRandomOptimizer(
                    config_space=self.config_space,
                    objectives=objectives,
                    max_trials=self.n_trials,
                    directions=directions,
                )
            except ImportError:
                from traigent.optimizers.random import RandomSearchOptimizer

                return RandomSearchOptimizer(
                    config_space=self.config_space,
                    objectives=objectives,
                )

        elif self.strategy == "grid":
            try:
                from traigent.optimizers.optuna_optimizer import OptunaGridOptimizer

                return OptunaGridOptimizer(
                    config_space=self.config_space,
                    objectives=objectives,
                    max_trials=self.n_trials,
                    directions=directions,
                )
            except ImportError:
                from traigent.optimizers.grid import GridSearchOptimizer

                return GridSearchOptimizer(
                    config_space=self.config_space,
                    objectives=objectives,
                )

        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Valid strategies: bayesian, tpe, evolutionary, nsga2, random, grid"
            )

    def _should_stop(self) -> bool:
        """Check if optimization should stop."""
        # Check trial limit
        if self._trial_counter >= self.n_trials:
            return True

        # Check time budget
        if self.timeout_seconds and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed >= self.timeout_seconds:
                logger.info(
                    f"Time budget exhausted after {elapsed:.1f}s "
                    f"({self._trial_counter} trials)"
                )
                return True

        return False

    async def _evaluate_config(
        self,
        config: dict[str, Any],
    ) -> TrialResult:
        """Evaluate a single configuration."""
        start_time = time.time()
        self._trial_counter += 1
        trial_id = f"trial_{self._trial_counter}"

        try:
            # Run evaluation
            result = await self.evaluator.evaluate(
                func=self.evaluator.pipeline.run,
                config=config,
                dataset=self.evaluator.get_core_dataset(),
            )

            duration = time.time() - start_time

            # Extract metrics
            metrics = dict(result.aggregated_metrics)

            # Check constraints
            constraints_satisfied = metrics.get("constraints_satisfied", True)

            trial_result = TrialResult(
                trial_id=trial_id,
                config=config,
                metrics=metrics,
                constraints_satisfied=constraints_satisfied,
                duration=duration,
            )

            logger.info(
                f"Trial {trial_id}: metrics={metrics}, "
                f"constraints_satisfied={constraints_satisfied}, "
                f"duration={duration:.2f}s"
            )

            return trial_result

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            return TrialResult(
                trial_id=trial_id,
                config=config,
                metrics={},
                constraints_satisfied=False,
                duration=time.time() - start_time,
            )

    async def _run_parallel_trials(
        self,
        configs: list[dict[str, Any]],
    ) -> list[TrialResult]:
        """Run multiple trials in parallel."""
        tasks = [self._evaluate_config(config) for config in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        trial_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel trial failed: {result}")
                continue
            trial_results.append(result)

        return trial_results

    async def optimize(
        self,
        progress_callback: Callable[[int, TrialResult], None] | None = None,
    ) -> OptimizationResult:
        """Run the optimization.

        Args:
            progress_callback: Optional callback called after each trial.

        Returns:
            OptimizationResult with best config and Pareto frontier.
        """
        self._start_time = time.time()
        self._trial_counter = 0
        self._history = []

        # Create optimizer
        self._optimizer = self._create_optimizer()

        logger.info(
            f"Starting {self.strategy} optimization with "
            f"n_trials={self.n_trials}, n_parallel={self.n_parallel}"
        )

        warnings = []

        # Convert history to format expected by optimizer
        def _to_optimizer_history() -> list[Any]:
            from traigent.api.types import TrialResult as CoreTrialResult
            from traigent.api.types import TrialStatus

            return [
                CoreTrialResult(
                    trial_id=t.trial_id,
                    config=t.config,
                    metrics=t.metrics,
                    status=(
                        TrialStatus.COMPLETED if t.is_successful else TrialStatus.FAILED
                    ),
                    duration=t.duration,
                    timestamp=t.timestamp,
                )
                for t in self._history
            ]

        # Main optimization loop
        while not self._should_stop():
            try:
                # Get next configurations
                if self.n_parallel > 1:
                    # Generate multiple configs for parallel execution
                    configs = []
                    for _ in range(
                        min(self.n_parallel, self.n_trials - self._trial_counter)
                    ):
                        if self._should_stop():
                            break
                        config = self._optimizer.suggest_next_trial(
                            _to_optimizer_history()
                        )
                        # Remove internal tracking keys
                        config = {
                            k: v for k, v in config.items() if not k.startswith("_")
                        }
                        configs.append(config)

                    if not configs:
                        break

                    # Run trials in parallel
                    results = await self._run_parallel_trials(configs)
                    self._history.extend(results)

                    if progress_callback:
                        for result in results:
                            progress_callback(self._trial_counter, result)

                else:
                    # Sequential execution
                    config = self._optimizer.suggest_next_trial(_to_optimizer_history())
                    # Remove internal tracking keys
                    config = {k: v for k, v in config.items() if not k.startswith("_")}

                    result = await self._evaluate_config(config)
                    self._history.append(result)

                    if progress_callback:
                        progress_callback(self._trial_counter, result)

            except Exception as e:
                logger.error(f"Optimization error: {e}")
                warnings.append(f"Error during optimization: {e}")
                break

        duration = time.time() - self._start_time

        # Compute Pareto frontier
        pareto_configs, pareto_metrics = compute_pareto_frontier(
            self._history,
            self.targets,
            require_constraints=bool(self.constraints),
        )

        # Find best config by primary objective
        primary_target = self.targets[0] if self.targets else None
        best_config = None
        best_metrics: dict[str, float] = {}

        if primary_target:
            ranked = rank_by_metric(
                self._history,
                primary_target.metric_name,
                primary_target.direction,
                require_constraints=bool(self.constraints),
            )
            if ranked:
                best_config = ranked[0].config
                best_metrics = ranked[0].metrics

        # Generate warnings
        if not pareto_configs and self._history:
            if self.constraints:
                warnings.append(
                    "No configurations satisfied all constraints. "
                    "Consider relaxing constraints or expanding search space."
                )

        logger.info(
            f"Optimization completed: {self._trial_counter} trials, "
            f"{len(pareto_configs)} Pareto-optimal configs, "
            f"duration={duration:.1f}s"
        )

        return OptimizationResult(
            best_config=best_config,
            best_metrics=best_metrics,
            history=self._history,
            pareto_configs=pareto_configs,
            pareto_metrics=pareto_metrics,
            total_trials=self._trial_counter,
            duration=duration,
            warnings=warnings,
        )

    async def warm_start(
        self,
        previous_results: list[TrialResult],
    ) -> None:
        """Warm-start optimization with previous results.

        Args:
            previous_results: Trial results from previous optimization.
        """
        self._history = list(previous_results)
        self._trial_counter = len(previous_results)
        logger.info(f"Warm-started with {len(previous_results)} previous trials")


def get_hyperparameter_importance(
    trials: list[TrialResult],
    target_metric: str,
) -> dict[str, float]:
    """Compute hyperparameter importance using fANOVA-style analysis.

    This is a simplified importance analysis based on correlation
    between parameter values and target metric.

    Args:
        trials: List of trial results.
        target_metric: Metric to analyze importance for.

    Returns:
        Dict mapping parameter names to importance scores (0-1).
    """
    if not trials:
        return {}

    # Get successful trials with the target metric
    successful = [t for t in trials if t.is_successful and target_metric in t.metrics]

    if len(successful) < 5:
        logger.warning(
            f"Not enough successful trials ({len(successful)}) for importance analysis"
        )
        return {}

    # Collect parameter values and metric values
    param_names = set()
    for trial in successful:
        param_names.update(trial.config.keys())

    metric_values = [t.metrics[target_metric] for t in successful]
    metric_mean = sum(metric_values) / len(metric_values)
    metric_var = sum((v - metric_mean) ** 2 for v in metric_values) / len(metric_values)

    if metric_var == 0:
        return dict.fromkeys(param_names, 0.0)

    importance = {}

    for param in param_names:
        # Get parameter values
        param_values = []
        corresponding_metrics = []

        for trial in successful:
            if param in trial.config:
                param_values.append(trial.config[param])
                corresponding_metrics.append(trial.metrics[target_metric])

        if len(param_values) < 3:
            importance[param] = 0.0
            continue

        # Group by unique parameter values and compute variance reduction
        unique_values = set(
            v if not isinstance(v, (list, dict)) else str(v) for v in param_values
        )

        if len(unique_values) == 1:
            importance[param] = 0.0
            continue

        # Compute between-group variance
        groups: dict[Any, list[float]] = {}
        for val, metric in zip(param_values, corresponding_metrics):
            key = val if not isinstance(val, (list, dict)) else str(val)
            if key not in groups:
                groups[key] = []
            groups[key].append(metric)

        # Between-group variance
        group_means = [sum(g) / len(g) for g in groups.values()]
        overall_mean = sum(corresponding_metrics) / len(corresponding_metrics)
        between_var = sum(
            len(g) * (m - overall_mean) ** 2
            for g, m in zip(groups.values(), group_means)
        ) / len(corresponding_metrics)

        # Importance is ratio of between-group variance to total variance
        importance[param] = (
            min(1.0, between_var / metric_var) if metric_var > 0 else 0.0
        )

    # Normalize to sum to 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


def export_optimization_history(
    result: OptimizationResult,
    format: str = "json",
) -> str | dict[str, Any]:
    """Export optimization history to various formats.

    Args:
        result: OptimizationResult to export.
        format: Export format - "json", "csv", or "dict".

    Returns:
        Exported data as string or dict.
    """
    import json

    history_data = []
    for trial in result.history:
        entry = {
            "trial_id": trial.trial_id,
            "config": trial.config,
            "metrics": trial.metrics,
            "constraints_satisfied": trial.constraints_satisfied,
            "duration": trial.duration,
            "timestamp": trial.timestamp.isoformat(),
        }
        history_data.append(entry)

    export_data = {
        "best_config": result.best_config,
        "best_metrics": result.best_metrics,
        "total_trials": result.total_trials,
        "duration": result.duration,
        "pareto_count": len(result.pareto_configs),
        "pareto_configs": result.pareto_configs,
        "pareto_metrics": result.pareto_metrics,
        "warnings": result.warnings,
        "history": history_data,
    }

    if format == "json":
        return json.dumps(export_data, indent=2, default=str)
    elif format == "csv":
        # CSV format for history only
        if not history_data:
            return "trial_id,config,metrics,constraints_satisfied,duration\n"

        lines = ["trial_id,config,metrics,constraints_satisfied,duration"]
        for entry in history_data:
            line = (
                f"{entry['trial_id']},"
                f"\"{json.dumps(entry['config'])}\","
                f"\"{json.dumps(entry['metrics'])}\","
                f"{entry['constraints_satisfied']},"
                f"{entry['duration']}"
            )
            lines.append(line)
        return "\n".join(lines)
    elif format == "dict":
        return export_data
    else:
        raise ValueError(f"Unknown format: {format}. Valid: json, csv, dict")
