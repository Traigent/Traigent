"""Static helper functions for OptimizationOrchestrator.

This module contains pure functions extracted from OptimizationOrchestrator
to improve testability and reduce class complexity.

Functions:
- validate_constructor_arguments: Input validation for optimizer, evaluator, etc.
- validate_dataset: Ensure dataset is present and non-empty
- normalize_parallel_trials: Validate and normalize parallel_trials parameter
- prepare_objectives: Prepare objectives list and create default schema
- allocate_parallel_ceilings: Allocate sample budget across parallel trials
- extract_optuna_trial_id: Extract Optuna trial ID from config or use provided ID
- prepare_evaluation_config: Filter out internal Optuna keys from config
- constraint_requires_metrics: Check if a constraint function requires metrics argument
- enforce_constraints: Enforce constraint functions on config/metrics
- extract_cost_from_results: Extract cost/examples from evaluation results
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, cast

from traigent.core.objectives import ObjectiveSchema, create_default_objectives
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import TVLConstraintError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def validate_constructor_arguments(
    optimizer: BaseOptimizer,
    evaluator: BaseEvaluator,
    max_trials: int | None = None,
    max_total_examples: int | None = None,
    timeout: float | None = None,
) -> None:
    """Validate constructor arguments for type and value constraints.

    Args:
        optimizer: Optimizer instance (must be BaseOptimizer subclass)
        evaluator: Evaluator instance (must be BaseEvaluator subclass)
        max_trials: Maximum number of trials (None for unlimited, must be non-negative)
        max_total_examples: Maximum examples to evaluate (None for unlimited)
        timeout: Timeout in seconds (None for no timeout, must be non-negative)

    Raises:
        TypeError: If optimizer or evaluator are not instances of their base classes
        ValueError: If max_trials, max_total_examples, or timeout are negative
    """
    if not isinstance(optimizer, BaseOptimizer):
        raise TypeError("optimizer must be an instance of BaseOptimizer")
    if not isinstance(evaluator, BaseEvaluator):
        raise TypeError("evaluator must be an instance of BaseEvaluator")
    if max_trials is not None and max_trials < 0:
        raise ValueError("max_trials must be non-negative")
    if max_total_examples is not None and max_total_examples < 0:
        raise ValueError("max_total_examples must be non-negative")
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be non-negative")


def validate_dataset(dataset: Dataset) -> None:
    """Ensure dataset is present and non-empty before optimization.

    Args:
        dataset: The evaluation dataset to validate

    Raises:
        ValueError: If dataset is None or empty
    """
    if dataset is None:
        raise ValueError("Dataset cannot be None") from None

    if not dataset or len(dataset) == 0:
        raise ValueError("Dataset cannot be empty")


def normalize_parallel_trials(parallel_trials: int | None) -> int:
    """Validate and normalize the parallel_trials parameter.

    Args:
        parallel_trials: Number of trials to run in parallel (None defaults to 1)

    Returns:
        Normalized parallel_trials value (always >= 1)

    Raises:
        ValueError: If parallel_trials is not a positive integer
    """
    if parallel_trials is None:
        return 1

    if isinstance(parallel_trials, bool) or not isinstance(parallel_trials, int):
        raise ValueError("parallel_trials must be a positive integer")

    if parallel_trials <= 0:
        raise ValueError("parallel_trials must be a positive integer")

    return parallel_trials


def prepare_objectives(
    objectives: Sequence[str | None] | None,
    objective_schema: ObjectiveSchema | None,
) -> tuple[list[str], ObjectiveSchema | None]:
    """Prepare objectives list and create default schema if needed.

    Args:
        objectives: List of objective names (None values filtered out, defaults to ['accuracy'])
        objective_schema: Optional pre-constructed ObjectiveSchema

    Returns:
        Tuple of (prepared_objectives_list, objective_schema)
        - prepared_objectives_list: List of objective names with None values filtered
        - objective_schema: Provided schema or newly created default schema (may be None if creation fails)
    """
    prepared = [obj for obj in (objectives or ["accuracy"]) if obj]
    schema = objective_schema
    if schema is None and prepared:
        try:
            schema = create_default_objectives([str(name) for name in prepared])
        except Exception as exc:
            logger.debug(
                "Failed to construct default objective schema: %s",
                exc,
            )
    return prepared, schema


def _redistribute_leftover(
    allocations: list[int],
    dataset_sizes: list[int],
    leftover: int,
) -> None:
    """Redistribute leftover budget to trials with remaining capacity.

    Modifies allocations in-place.

    Args:
        allocations: Current allocation list (modified in-place)
        dataset_sizes: Maximum sizes for each trial
        leftover: Remaining budget to distribute
    """
    for idx, size in enumerate(dataset_sizes):
        if leftover <= 0:
            break
        available_room = size - allocations[idx]
        if available_room <= 0:
            continue
        increment = min(available_room, leftover)
        allocations[idx] += increment
        leftover -= increment


def allocate_parallel_ceilings(
    dataset_sizes: list[int],
    remaining_budget: int,
) -> list[int | None]:
    """Allocate sample budget across parallel trials.

    This is a safety mechanism to ensure that when spinning up N parallel
    trials with X remaining budget, each trial gets at most X/N samples.
    This prevents any single trial from consuming more than its fair share
    of the budget when running in parallel.

    The function distributes the remaining sample budget evenly across trials
    in the current batch, respecting individual dataset sizes. Any leftover
    budget (from trials with smaller datasets) is redistributed to trials
    that have room for more samples.

    Args:
        dataset_sizes: List of dataset sizes for each trial in the batch
        remaining_budget: Total remaining sample budget to allocate

    Returns:
        List of ceiling values for each trial (0 if budget exhausted)

    Example:
        With 3 parallel trials and 300 remaining budget:
        - Each trial gets at most 100 samples (300 / 3)
        - If a trial's dataset has only 50 samples, it gets 50
        - The leftover 50 is redistributed to other trials
    """
    if remaining_budget <= 0:
        return cast(list[int | None], [0 for _ in dataset_sizes])

    if not dataset_sizes:
        return []

    batch_size = len(dataset_sizes)

    # Safety: each trial in the batch gets at most remaining_budget / batch_size
    # This ensures parallel trials don't exceed the total budget
    base = remaining_budget // batch_size
    remainder = remaining_budget % batch_size
    allocations: list[int] = []

    for idx, size in enumerate(dataset_sizes):
        allocation = base + (1 if idx < remainder else 0)
        allocations.append(min(size, allocation))

    leftover = max(remaining_budget - sum(allocations), 0)
    if leftover > 0:
        _redistribute_leftover(allocations, dataset_sizes, leftover)

    return cast(list[int | None], allocations)


def extract_optuna_trial_id(
    config: dict[str, Any], optuna_trial_id: int | None
) -> int | None:
    """Extract Optuna trial ID from config or use provided ID.

    Preserves existing fallback behavior: if caller already passed an ID, keep it;
    otherwise extract _optuna_trial_id from config dict.

    Args:
        config: Configuration that may contain _optuna_trial_id
        optuna_trial_id: Optuna trial ID passed by caller (takes precedence)

    Returns:
        Optuna trial ID if available, None otherwise
    """
    if optuna_trial_id is not None:
        return optuna_trial_id

    if isinstance(config, dict):
        return config.get("_optuna_trial_id")

    return None


def prepare_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare evaluation config by filtering out internal Optuna keys.

    Args:
        config: Raw configuration including potential Optuna internal keys

    Returns:
        Filtered configuration for evaluation
    """
    if isinstance(config, dict):
        return {
            key: value for key, value in config.items() if not key.startswith("_optuna")
        }
    return config


def constraint_requires_metrics(constraint: Callable[..., Any]) -> bool:
    """Check if a constraint function requires a metrics argument.

    Constraints can optionally receive metrics as a second argument. This function
    determines if a constraint needs metrics by checking:
    1. The __tvl_constraint__ metadata for explicit 'requires_metrics' flag
    2. The function signature for number of parameters

    Args:
        constraint: Constraint function to inspect

    Returns:
        True if the constraint requires metrics, False otherwise
    """
    metadata = getattr(constraint, "__tvl_constraint__", None)
    if isinstance(metadata, dict) and "requires_metrics" in metadata:
        return bool(metadata["requires_metrics"])
    try:
        signature = inspect.signature(constraint)
    except (TypeError, ValueError):
        return False
    return len(signature.parameters) > 1


def enforce_constraints(
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
    constraints: list[Callable[[dict[str, Any], dict[str, Any] | None], bool]],
    stage: str,
) -> None:
    """Enforce constraint functions on configuration and metrics.

    Iterates through all constraints and raises TVLConstraintError if any
    constraint returns False or raises an exception.

    Args:
        config: Configuration to validate
        metrics: Optional metrics dict (may be required by some constraints)
        constraints: List of constraint functions to evaluate
        stage: Stage name for error messages (e.g., 'pre', 'post')

    Raises:
        TVLConstraintError: If a constraint fails or returns False
    """
    if not constraints:
        return

    payload_metrics = metrics or {}
    for constraint in constraints:
        requires_metrics = constraint_requires_metrics(constraint)
        try:
            if requires_metrics:
                passed = constraint(config, payload_metrics)
            else:
                passed = cast(Callable[[dict[str, Any]], bool], constraint)(config)
        except Exception as exc:
            metadata = getattr(constraint, "__tvl_constraint__", {}) or {}
            identifier = metadata.get("id") or getattr(
                constraint, "__name__", "constraint"
            )
            raise TVLConstraintError(
                f"Constraint '{identifier}' failed during {stage}: {exc}",
                details={"constraint": identifier, "stage": stage},
            ) from exc
        if not passed:
            metadata = getattr(constraint, "__tvl_constraint__", {}) or {}
            identifier = metadata.get("id") or getattr(
                constraint, "__name__", "constraint"
            )
            message = metadata.get("message") or f"Constraint '{identifier}' failed"
            raise TVLConstraintError(
                message,
                details={"constraint": identifier, "stage": stage},
            )


def pre_trial_validate_config(
    config: dict[str, Any],
    constraints: list[Callable[[dict[str, Any]], bool]],
) -> bool:
    """Validate a config against pre-trial constraints before execution.

    Returns True if the config satisfies all constraints, False otherwise.
    This prevents wasting budget on configurations that violate structural
    constraints by checking them before expensive trial execution.

    Args:
        config: Configuration dict to validate.
        constraints: Pre-evaluation constraint callables.

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    if not constraints:
        return True
    for constraint in constraints:
        try:
            if not constraint(config):
                return False
        except Exception:
            return False
    return True


def extract_cost_from_results(
    eval_result: Any,
    progress_state: dict[str, Any] | None,
    trial_id: str,
) -> tuple[int | None, float | None]:
    """Extract examples attempted and total cost from evaluation results.

    Args:
        eval_result: Evaluation result object
        progress_state: Progress tracking state
        trial_id: Trial identifier for logging

    Returns:
        Tuple of (examples_attempted, total_cost)
    """
    examples_attempted = None
    total_cost = None

    if progress_state:
        examples_attempted = progress_state.get("evaluated")
        total_cost = progress_state.get("total_cost")

    # Extract from example_results if available
    if hasattr(eval_result, "example_results") and eval_result.example_results:
        logger.debug(
            f"Trial {trial_id}: preserved {len(eval_result.example_results)} example results"
        )
        if examples_attempted is None:
            examples_attempted = len(eval_result.example_results)

        total_cost_examples = 0.0
        has_cost_samples = False
        for example in eval_result.example_results:
            metrics = getattr(example, "metrics", {}) or {}
            cost_value = None
            for key in ("total_cost", "cost", "example_cost"):
                if metrics.get(key) is not None:
                    cost_value = metrics[key]
                    break
            if cost_value is not None:
                try:
                    total_cost_examples += float(cost_value)
                    has_cost_samples = True
                except (TypeError, ValueError):
                    logger.debug(
                        "Unable to parse example cost %s for trial %s",
                        cost_value,
                        trial_id,
                    )

        if has_cost_samples:
            total_cost = total_cost_examples

    # Fallback to aggregated metrics
    if total_cost is None and hasattr(eval_result, "aggregated_metrics"):
        agg_metrics = eval_result.aggregated_metrics or {}
        total_cost = agg_metrics.get("total_cost")

    # Prefer evaluator-reported totals when available to avoid double counting
    result_total = getattr(eval_result, "total_examples", None)
    if isinstance(result_total, int) and result_total > 0:
        if examples_attempted is None or examples_attempted > result_total:
            examples_attempted = result_total

    return examples_attempted, total_cost
