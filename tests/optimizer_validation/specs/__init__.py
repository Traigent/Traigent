"""Test scenario specifications for optimizer validation."""

from .builders import (
    basic_scenario,
    config_space_scenario,
    constrained_scenario,
    evaluator_scenario,
    failure_scenario,
    multi_objective_scenario,
)
from .scenario import (
    ConstraintSpec,
    EvaluatorSpec,
    ExpectedOutcome,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)
from .trace_expectations import (
    TraceExpectations,
    basic_expectations,
    failure_expectations,
    multi_objective_expectations,
)
from .validators import ResultValidator

__all__ = [
    # Scenario classes
    "TestScenario",
    "ObjectiveSpec",
    "ConstraintSpec",
    "EvaluatorSpec",
    "ExpectedResult",
    "ExpectedOutcome",
    # Trace expectations
    "TraceExpectations",
    "basic_expectations",
    "multi_objective_expectations",
    "failure_expectations",
    # Builders
    "basic_scenario",
    "multi_objective_scenario",
    "constrained_scenario",
    "failure_scenario",
    "evaluator_scenario",
    "config_space_scenario",
    # Validators
    "ResultValidator",
]
