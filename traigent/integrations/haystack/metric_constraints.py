"""Metric constraints for Haystack pipeline evaluations.

This module provides constraint classes for validating evaluation metrics
(cost, latency, quality) against thresholds after pipeline execution.

Example usage:
    from traigent.integrations.haystack.metric_constraints import (
        MetricConstraint,
        check_constraints,
        cost_constraint,
        latency_constraint,
    )

    # Define constraints
    constraints = [
        cost_constraint(max_cost=0.05),
        latency_constraint(p95_ms=3000),
        MetricConstraint("accuracy", ">=", 0.8),
    ]

    # Check against metrics
    result = check_constraints(constraints, metrics)
    print(f"All satisfied: {result.all_satisfied}")
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Mapping of operator strings to functions
OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class MetricConstraint:
    """Constraint that validates a metric against a threshold.

    MetricConstraint is used for post-evaluation constraint checking,
    validating that evaluation metrics (cost, latency, quality scores)
    meet specified thresholds.

    Attributes:
        metric_name: Name of the metric to check (e.g., "total_cost").
        op: Comparison operator as string (<=, <, >=, >, ==, !=).
        threshold: Value to compare against.
        name: Optional display name for the constraint.

    Example:
        >>> constraint = MetricConstraint("latency_p95_ms", "<=", 500)
        >>> constraint.check({"latency_p95_ms": 300})
        True
        >>> constraint.check({"latency_p95_ms": 600})
        False
    """

    metric_name: str
    op: str
    threshold: float | int
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate operator."""
        if self.op not in OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.op}'. "
                f"Valid operators: {list(OPERATORS.keys())}"
            )
        if self.name is None:
            self.name = f"{self.metric_name} {self.op} {self.threshold}"

    def check(self, metrics: dict[str, Any]) -> bool:
        """Check if metrics satisfy this constraint.

        Args:
            metrics: Dict of metric name -> value.

        Returns:
            True if constraint is satisfied, False otherwise.
        """
        if self.metric_name not in metrics:
            logger.debug(
                f"Metric '{self.metric_name}' not found in metrics, "
                f"treating as unsatisfied"
            )
            return False

        value = metrics[self.metric_name]
        if value is None:
            return False

        try:
            op_func = OPERATORS[self.op]
            return op_func(value, self.threshold)
        except (TypeError, ValueError) as e:
            logger.debug(f"Constraint check failed: {e}")
            return False

    def get_violation_message(self, metrics: dict[str, Any]) -> str:
        """Get a message describing the constraint violation.

        Args:
            metrics: Dict of metric name -> value.

        Returns:
            Human-readable violation description.
        """
        value = metrics.get(self.metric_name, "missing")
        return (
            f"Constraint '{self.name}' violated: "
            f"{self.metric_name}={value}, expected {self.op} {self.threshold}"
        )


@dataclass
class ConstraintViolation:
    """Details of a constraint violation.

    Attributes:
        constraint: The constraint that was violated.
        actual_value: The actual metric value.
        message: Human-readable violation message.
    """

    constraint: MetricConstraint
    actual_value: Any
    message: str


@dataclass
class ConstraintCheckResult:
    """Result of checking multiple constraints.

    Attributes:
        all_satisfied: True if all constraints are satisfied.
        violations: List of constraint violations.
        satisfied_count: Number of satisfied constraints.
        total_count: Total number of constraints checked.
    """

    all_satisfied: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    satisfied_count: int = 0
    total_count: int = 0

    @property
    def violation_messages(self) -> list[str]:
        """Get list of violation messages."""
        return [v.message for v in self.violations]


def check_constraints(
    constraints: list[MetricConstraint],
    metrics: dict[str, Any],
) -> ConstraintCheckResult:
    """Check multiple constraints against metrics.

    Args:
        constraints: List of constraints to check.
        metrics: Dict of metric name -> value.

    Returns:
        ConstraintCheckResult with detailed results.
    """
    if not constraints:
        return ConstraintCheckResult(
            all_satisfied=True,
            satisfied_count=0,
            total_count=0,
        )

    violations = []
    satisfied = 0

    for constraint in constraints:
        if constraint.check(metrics):
            satisfied += 1
        else:
            actual_value = metrics.get(constraint.metric_name)
            violation = ConstraintViolation(
                constraint=constraint,
                actual_value=actual_value,
                message=constraint.get_violation_message(metrics),
            )
            violations.append(violation)

    return ConstraintCheckResult(
        all_satisfied=len(violations) == 0,
        violations=violations,
        satisfied_count=satisfied,
        total_count=len(constraints),
    )


# Convenience functions for common constraints


def cost_constraint(
    max_cost: float,
    metric_name: str = "total_cost",
) -> MetricConstraint:
    """Create a cost constraint.

    Args:
        max_cost: Maximum allowed cost in USD.
        metric_name: Name of the cost metric (default: "total_cost").

    Returns:
        MetricConstraint for cost limit.

    Example:
        >>> constraint = cost_constraint(max_cost=0.05)
        >>> constraint.check({"total_cost": 0.03})
        True
    """
    return MetricConstraint(
        metric_name=metric_name,
        op="<=",
        threshold=max_cost,
        name=f"max_cost_{max_cost}",
    )


def latency_constraint(
    p50_ms: float | None = None,
    p95_ms: float | None = None,
    p99_ms: float | None = None,
    mean_ms: float | None = None,
    max_ms: float | None = None,
) -> list[MetricConstraint]:
    """Create latency constraints.

    Creates constraints for specified latency percentiles. At least one
    percentile must be specified.

    Args:
        p50_ms: Maximum p50 latency in milliseconds.
        p95_ms: Maximum p95 latency in milliseconds.
        p99_ms: Maximum p99 latency in milliseconds.
        mean_ms: Maximum mean latency in milliseconds.
        max_ms: Maximum max latency in milliseconds.

    Returns:
        List of MetricConstraint for latency limits.

    Example:
        >>> constraints = latency_constraint(p95_ms=500, p99_ms=1000)
        >>> len(constraints)
        2
    """
    constraints = []

    if p50_ms is not None:
        constraints.append(
            MetricConstraint(
                metric_name="latency_p50_ms",
                op="<=",
                threshold=p50_ms,
                name=f"max_p50_{p50_ms}ms",
            )
        )

    if p95_ms is not None:
        constraints.append(
            MetricConstraint(
                metric_name="latency_p95_ms",
                op="<=",
                threshold=p95_ms,
                name=f"max_p95_{p95_ms}ms",
            )
        )

    if p99_ms is not None:
        constraints.append(
            MetricConstraint(
                metric_name="latency_p99_ms",
                op="<=",
                threshold=p99_ms,
                name=f"max_p99_{p99_ms}ms",
            )
        )

    if mean_ms is not None:
        constraints.append(
            MetricConstraint(
                metric_name="latency_mean_ms",
                op="<=",
                threshold=mean_ms,
                name=f"max_mean_{mean_ms}ms",
            )
        )

    if max_ms is not None:
        constraints.append(
            MetricConstraint(
                metric_name="latency_max_ms",
                op="<=",
                threshold=max_ms,
                name=f"max_latency_{max_ms}ms",
            )
        )

    if not constraints:
        raise ValueError("At least one latency threshold must be specified")

    return constraints


def quality_constraint(
    metric_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[MetricConstraint]:
    """Create quality metric constraints.

    Args:
        metric_name: Name of the quality metric.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).

    Returns:
        List of MetricConstraint for quality limits.

    Example:
        >>> constraints = quality_constraint("accuracy", min_value=0.8)
        >>> len(constraints)
        1
    """
    constraints = []

    if min_value is not None:
        constraints.append(
            MetricConstraint(
                metric_name=metric_name,
                op=">=",
                threshold=min_value,
                name=f"min_{metric_name}_{min_value}",
            )
        )

    if max_value is not None:
        constraints.append(
            MetricConstraint(
                metric_name=metric_name,
                op="<=",
                threshold=max_value,
                name=f"max_{metric_name}_{max_value}",
            )
        )

    if not constraints:
        raise ValueError("At least one of min_value or max_value must be specified")

    return constraints


# Result filtering functions


def filter_by_constraints(
    results: list[Any],
    constraints_key: str = "constraints_satisfied",
) -> list[Any]:
    """Filter evaluation results to only those satisfying constraints.

    Filters a list of EvaluationResult objects (or similar) to return only
    those where constraints_satisfied is True.

    Args:
        results: List of evaluation results with aggregated_metrics.
        constraints_key: Key in aggregated_metrics for constraint status.
            Defaults to "constraints_satisfied".

    Returns:
        List of results where constraints are satisfied.

    Example:
        >>> results = [result1, result2, result3]  # Mixed constraint status
        >>> satisfying = filter_by_constraints(results)
        >>> all(r.aggregated_metrics["constraints_satisfied"] for r in satisfying)
        True
    """
    satisfying = []

    for result in results:
        metrics = getattr(result, "aggregated_metrics", None)
        if metrics is None:
            # Try legacy .metrics attribute
            metrics = getattr(result, "metrics", {})

        if metrics.get(constraints_key, False):
            satisfying.append(result)

    return satisfying


def get_best_satisfying(
    results: list[Any],
    metric: str = "accuracy",
    maximize: bool = True,
    constraints_key: str = "constraints_satisfied",
) -> Any | None:
    """Get the best result that satisfies all constraints.

    Finds the highest (or lowest) scoring result among those that satisfy
    all constraints.

    Args:
        results: List of evaluation results with aggregated_metrics.
        metric: Name of the metric to optimize. Defaults to "accuracy".
        maximize: If True, find highest value. If False, find lowest.
            Defaults to True.
        constraints_key: Key in aggregated_metrics for constraint status.
            Defaults to "constraints_satisfied".

    Returns:
        The best satisfying result, or None if no results satisfy constraints.

    Example:
        >>> results = [result1, result2, result3]  # Various scores and constraints
        >>> best = get_best_satisfying(results, metric="accuracy")
        >>> best.aggregated_metrics["constraints_satisfied"]
        True
    """
    satisfying = filter_by_constraints(results, constraints_key)

    if not satisfying:
        logger.debug("No results satisfy constraints")
        return None

    def get_metric_value(result: Any) -> float:
        """Extract metric value from result."""
        metrics = getattr(result, "aggregated_metrics", None)
        if metrics is None:
            metrics = getattr(result, "metrics", {})

        value = metrics.get(metric)
        if value is None:
            # Return worst possible value for sorting
            return float("-inf") if maximize else float("inf")
        return float(value)

    # Sort by metric and return best
    sorted_results = sorted(satisfying, key=get_metric_value, reverse=maximize)
    return sorted_results[0]
