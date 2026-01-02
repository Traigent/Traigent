"""Scenario builder functions for common test patterns.

These builders provide convenient factory functions for creating TestScenario
instances with sensible defaults for common testing patterns.
"""

from __future__ import annotations

from typing import Any

from .scenario import (
    ConstraintSpec,
    EvaluatorSpec,
    ExpectedOutcome,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)


def basic_scenario(
    name: str,
    *,
    injection_mode: str = "context",
    execution_mode: str = "edge_analytics",
    config_space: dict[str, Any] | None = None,
    max_trials: int = 3,
    **overrides: Any,
) -> TestScenario:
    """Build a basic working scenario with sensible defaults.

    Args:
        name: Unique scenario name
        injection_mode: Configuration injection mode
        execution_mode: Execution mode
        config_space: Configuration space (default: model + temperature)
        max_trials: Maximum trials to run
        **overrides: Additional TestScenario field overrides

    Returns:
        Configured TestScenario instance
    """
    if config_space is None:
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

    return TestScenario(
        name=name,
        description=f"Basic {injection_mode} injection with {execution_mode} execution",
        injection_mode=injection_mode,  # type: ignore[arg-type]
        execution_mode=execution_mode,  # type: ignore[arg-type]
        config_space=config_space,
        max_trials=max_trials,
        **overrides,
    )


def multi_objective_scenario(
    name: str,
    objectives: list[ObjectiveSpec],
    *,
    config_space: dict[str, Any] | None = None,
    max_trials: int = 5,
    **overrides: Any,
) -> TestScenario:
    """Build a multi-objective optimization scenario.

    Args:
        name: Unique scenario name
        objectives: List of ObjectiveSpec instances
        config_space: Configuration space
        max_trials: Maximum trials to run
        **overrides: Additional TestScenario field overrides

    Returns:
        Configured TestScenario instance
    """
    if config_space is None:
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.1, 0.5, 0.9],
        }

    objective_names = [obj.name for obj in objectives]

    return TestScenario(
        name=name,
        description=f"Multi-objective with {len(objectives)} objectives: {objective_names}",
        objectives=objectives,
        config_space=config_space,
        max_trials=max_trials,
        expected=ExpectedResult(required_metrics=objective_names),
        **overrides,
    )


def constrained_scenario(
    name: str,
    constraints: list[ConstraintSpec],
    *,
    config_space: dict[str, Any] | None = None,
    max_trials: int = 5,
    **overrides: Any,
) -> TestScenario:
    """Build a scenario with constraints.

    Args:
        name: Unique scenario name
        constraints: List of ConstraintSpec instances
        config_space: Configuration space
        max_trials: Maximum trials to run
        **overrides: Additional TestScenario field overrides

    Returns:
        Configured TestScenario instance
    """
    if config_space is None:
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.1, 0.5, 0.9],
        }

    constraint_names = [c.name for c in constraints]

    return TestScenario(
        name=name,
        description=f"Constrained optimization with: {constraint_names}",
        constraints=constraints,
        config_space=config_space,
        max_trials=max_trials,
        **overrides,
    )


def failure_scenario(
    name: str,
    expected_error: type[Exception],
    *,
    error_message: str | None = None,
    config_space: dict[str, Any] | None = None,
    **overrides: Any,
) -> TestScenario:
    """Build a scenario expected to fail with a specific error.

    Args:
        name: Unique scenario name
        expected_error: Expected exception type
        error_message: Substring expected in error message
        config_space: Configuration space
        **overrides: Additional TestScenario field overrides

    Returns:
        Configured TestScenario instance
    """
    if config_space is None:
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

    return TestScenario(
        name=name,
        description=f"Expected failure: {expected_error.__name__}",
        config_space=config_space,
        expected=ExpectedResult(
            outcome=ExpectedOutcome.FAILURE,
            error_type=expected_error,
            error_message_contains=error_message,
        ),
        **overrides,
    )


def evaluator_scenario(
    name: str,
    evaluator_type: str,
    *,
    evaluator_fn: Any = None,
    scoring_fn: Any = None,
    metric_fns: dict[str, Any] | None = None,
    should_fail: bool = False,
    **overrides: Any,
) -> TestScenario:
    """Build a scenario with custom evaluator configuration.

    Args:
        name: Unique scenario name
        evaluator_type: Type of evaluator
        evaluator_fn: Custom evaluator function
        scoring_fn: Scoring function
        metric_fns: Metric functions dictionary
        should_fail: Whether evaluator is expected to fail
        **overrides: Additional TestScenario field overrides

    Returns:
        Configured TestScenario instance
    """
    evaluator = EvaluatorSpec(
        type=evaluator_type,  # type: ignore[arg-type]
        evaluator_fn=evaluator_fn,
        scoring_fn=scoring_fn,
        metric_fns=metric_fns,
        should_fail=should_fail,
    )

    return TestScenario(
        name=name,
        description=f"Custom evaluator: {evaluator_type}",
        evaluator=evaluator,
        **overrides,
    )


def config_space_scenario(
    name: str,
    config_space: dict[str, Any],
    *,
    description: str | None = None,
    **overrides: Any,
) -> TestScenario:
    """Build a scenario focused on testing a specific config space.

    Args:
        name: Unique scenario name
        config_space: Configuration space to test
        description: Optional description
        **overrides: Additional TestScenario field overrides

    Returns:
        Configured TestScenario instance
    """
    param_types = []
    for key, value in config_space.items():
        if isinstance(value, tuple):
            param_types.append(f"{key}:continuous")
        elif isinstance(value, list):
            param_types.append(f"{key}:categorical")

    return TestScenario(
        name=name,
        description=description or f"Config space: {', '.join(param_types)}",
        config_space=config_space,
        **overrides,
    )
