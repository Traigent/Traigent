"""Test scenario specification dataclasses for optimizer validation.

This module provides declarative data structures for specifying test scenarios,
including inputs, expected outputs, and failure injection settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .trace_expectations import TraceExpectations


class ExpectedOutcome(Enum):
    """Expected outcome of a test scenario."""

    SUCCESS = "success"  # Optimization completes successfully
    FAILURE = "failure"  # Optimization fails with exception
    PARTIAL = "partial"  # Some trials succeed, some fail


@dataclass
class ObjectiveSpec:
    """Specification for an optimization objective.

    Attributes:
        name: Objective name (e.g., "accuracy", "cost")
        orientation: Whether to maximize, minimize, or target a band
        weight: Weight for multi-objective optimization (default 1.0)
        bounds: Optional (min, max) bounds for the objective value
    """

    name: str
    orientation: Literal["maximize", "minimize", "band"] = "maximize"
    weight: float = 1.0
    bounds: tuple[float, float] | None = None


@dataclass
class ConstraintSpec:
    """Specification for a constraint function.

    Attributes:
        name: Human-readable name for the constraint
        constraint_fn: The constraint function (config) -> bool or (config, metrics) -> bool
        requires_metrics: Whether the constraint needs metrics (post-trial)
        expected_rejections: Expected number of configs to be rejected (for validation)
    """

    name: str
    constraint_fn: Callable[..., bool]
    requires_metrics: bool = False
    expected_rejections: int = 0


@dataclass
class EvaluatorSpec:
    """Specification for custom evaluation behavior.

    Attributes:
        type: Type of evaluator ("default", "custom", "scoring_function", "metric_functions")
        evaluator_fn: Custom evaluator function (for type="custom")
        scoring_fn: Scoring function (for type="scoring_function")
        metric_fns: Dict of metric functions (for type="metric_functions")
        should_fail: Whether the evaluator is expected to fail
        fail_on_example: Fail on specific example index (0-based)
    """

    type: Literal["default", "custom", "scoring_function", "metric_functions"] = (
        "default"
    )
    evaluator_fn: Callable[..., Any] | None = None
    scoring_fn: Callable[..., float] | None = None
    metric_fns: dict[str, Callable[..., float]] | None = None
    should_fail: bool = False
    fail_on_example: int | None = None


@dataclass
class ExpectedResult:
    """Expected outcomes for a test scenario.

    Attributes:
        outcome: Expected outcome type (success, failure, partial)
        min_trials: Minimum number of trials expected
        max_trials: Maximum number of trials expected (None = no limit)
        expected_stop_reason: Expected stop reason string
        best_score_range: Expected (min, max) range for best score
        required_metrics: List of metric names that must be present
        error_type: Expected exception type for failure scenarios
        error_message_contains: Substring expected in error message
    """

    outcome: ExpectedOutcome = ExpectedOutcome.SUCCESS
    min_trials: int = 1
    max_trials: int | None = None
    expected_stop_reason: str | None = None
    best_score_range: tuple[float, float] | None = None
    required_metrics: list[str] | None = None
    error_type: type[Exception] | None = None
    error_message_contains: str | None = None


@dataclass
class TestScenario:
    """Complete specification for an optimizer test scenario.

    This dataclass declaratively specifies all inputs and expected outputs
    for a single test case, enabling easy review and modification.

    Attributes:
        name: Unique identifier for the scenario
        description: Human-readable description
        markers: pytest markers to apply (e.g., ["unit", "slow"])

        injection_mode: Config injection mode
        execution_mode: Execution mode

        config_space: Configuration space dictionary
        default_config: Optional default configuration

        objectives: List of objective names or ObjectiveSpec instances
        constraints: List of ConstraintSpec instances

        evaluator: Evaluator specification
        dataset_path: Path to dataset file (or None to generate)
        dataset_size: Number of examples to generate if no path

        max_trials: Maximum trials to run
        timeout: Timeout in seconds
        parallel_config: Parallel execution configuration
        mock_mode_config: Mock mode configuration

        tvl_spec_path: Path to TVL specification file
        tvl_environment: TVL environment name

        expected: Expected result specification

        function_should_raise: Exception type the function should raise
        function_raise_on_call: Call number on which to raise (1-based)
        function_return_value: Custom return value for the function
    """

    # Identity
    name: str
    description: str
    markers: list[str] = field(default_factory=list)

    # Core configuration
    injection_mode: Literal["context", "parameter", "attribute", "seamless"] = "context"
    execution_mode: Literal[
        "edge_analytics", "privacy", "hybrid", "standard", "cloud"
    ] = "edge_analytics"

    # Configuration space
    config_space: dict[str, Any] = field(default_factory=dict)
    default_config: dict[str, Any] | None = None

    # Objectives
    objectives: list[str | ObjectiveSpec] = field(default_factory=lambda: ["accuracy"])

    # Constraints
    constraints: list[ConstraintSpec] = field(default_factory=list)

    # Evaluation
    evaluator: EvaluatorSpec = field(default_factory=EvaluatorSpec)
    dataset_path: str | None = None
    dataset_size: int = 3

    # Execution parameters
    max_trials: int = 5
    timeout: float = 30.0
    parallel_config: dict[str, Any] | None = None
    mock_mode_config: dict[str, Any] | None = None

    # TVL integration
    tvl_spec_path: str | None = None
    tvl_environment: str | None = None

    # Expected results
    expected: ExpectedResult = field(default_factory=ExpectedResult)

    # Function behavior injection
    function_should_raise: type[Exception] | None = None
    function_raise_on_call: int | None = None  # 1-based call number
    function_return_value: Any = None
    custom_function: Callable[..., Any] | None = None

    # Trace expectations (for tracing validation)
    trace_expectations: TraceExpectations | None = None

    # Gist template for tooltip display in viewer
    # Format: Plain text with dynamic placeholders like {status()}, {error_type()}, etc.
    # Example: "empty-dataset -> {error_type()} | {status()}"
    # Available functions: status(), outcome(), error_type(), trial_count(), best_score(),
    #                      stop_reason(), duration(), config_space_size(), injection_mode(), algorithm()
    gist_template: str | None = None

    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        if not self.name:
            raise ValueError("Scenario name cannot be empty")

        if not self.config_space and not self.tvl_spec_path:
            # Provide a default config space for testing
            self.config_space = {
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            }

    def get_objective_names(self) -> list[str]:
        """Get list of objective names."""
        return [
            obj.name if isinstance(obj, ObjectiveSpec) else obj
            for obj in self.objectives
        ]

    def has_weighted_objectives(self) -> bool:
        """Check if any objectives have non-default weights."""
        for obj in self.objectives:
            if isinstance(obj, ObjectiveSpec) and obj.weight != 1.0:
                return True
        return False

    def has_constraints(self) -> bool:
        """Check if scenario has any constraints."""
        return len(self.constraints) > 0

    def expects_failure(self) -> bool:
        """Check if scenario expects a failure outcome."""
        return self.expected.outcome == ExpectedOutcome.FAILURE
