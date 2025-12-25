"""Shared test fixtures and configuration for optimizer validation tests.

This module provides:
- Scenario runner for executing TestScenario specifications
- Dataset factory for creating test datasets
- Result validator fixture
- Test function factories
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample

from .specs.scenario import EvaluatorSpec, ObjectiveSpec, TestScenario
from .specs.validators import ResultValidator, ValidationResult

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult

    from .tracing.analyzer import TraceAnalyzer, TraceValidationResult
    from .tracing.capture import CapturedTrace, TraceCapture


# Ensure mock mode is enabled for all tests in this module
@pytest.fixture(autouse=True)
def ensure_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock mode is enabled for all optimizer validation tests."""
    monkeypatch.setenv("TRAIGENT_MOCK_MODE", "true")
    monkeypatch.setenv("MOCK_MODE", "true")


@pytest.fixture(autouse=True)
def set_trace_test_context(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Automatically set test context for tracing before each test.

    This adds informative metadata to traces so you can identify
    which test generated which trace in Jaeger.
    """
    try:
        from traigent.core.tracing import clear_test_context, set_test_context
    except ImportError:
        yield
        return

    # Extract test information from pytest request
    test_name = request.node.name  # e.g., "test_injection_mode_basic[context]"
    test_module = request.node.module.__name__ if request.node.module else None

    # Try to get docstring as description
    test_description = None
    if hasattr(request.node, "function") and request.node.function:
        test_description = request.node.function.__doc__
        if test_description:
            # Take first line only
            test_description = test_description.strip().split("\n")[0]

    # Extract class name if it's a method
    test_class = None
    if request.node.cls:
        test_class = request.node.cls.__name__

    # Build extra attributes
    extra: dict[str, Any] = {}
    if test_class:
        extra["class"] = test_class

    # Extract parametrize values if present
    if hasattr(request.node, "callspec"):
        params = request.node.callspec.params
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                extra[f"param.{key}"] = value

    set_test_context(
        test_name=test_name,
        test_description=test_description,
        test_module=test_module,
        **extra,
    )

    yield

    # Clear context after test
    clear_test_context()


@pytest.fixture
def temp_dataset_dir() -> Path:
    """Create temporary directory for test datasets within the project.

    Uses project data directory to satisfy path validation requirements.
    """
    # Use the data directory within the test package
    datasets_dir = Path(__file__).parent / "data" / "_generated"
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir


@pytest.fixture
def dataset_factory(temp_dataset_dir: Path) -> Callable[..., tuple[str, Dataset]]:
    """Factory for creating test datasets.

    Returns a function that creates a dataset and writes it to a JSONL file.

    Usage:
        path, dataset = dataset_factory("my_dataset", size=5)
    """

    def create_dataset(
        name: str,
        size: int = 3,
        input_key: str = "text",
        include_malformed: bool = False,
    ) -> tuple[str, Dataset]:
        """Create a test dataset.

        Args:
            name: Dataset name
            size: Number of examples
            input_key: Key for input data
            include_malformed: Whether to include a malformed example

        Returns:
            Tuple of (jsonl_path, Dataset)
        """
        examples = []
        for i in range(size):
            example = EvaluationExample(
                input_data={input_key: f"test_input_{i}"},
                expected_output=f"expected_output_{i}",
            )
            examples.append(example)

        if include_malformed:
            # Add malformed example with missing input key
            examples.append(
                EvaluationExample(
                    input_data={},  # Missing input key
                    expected_output=None,
                )
            )

        dataset = Dataset(examples=examples, name=name)

        # Write to JSONL file
        jsonl_path = temp_dataset_dir / f"{name}.jsonl"
        with open(jsonl_path, "w") as f:
            for ex in examples:
                json.dump(
                    {
                        "input": ex.input_data,
                        "output": ex.expected_output,
                    },
                    f,
                )
                f.write("\n")

        return str(jsonl_path), dataset

    return create_dataset


def create_test_function(
    scenario: TestScenario,
) -> Callable[..., str]:
    """Create a test function based on scenario specification.

    Handles function behavior injection (raising, returning None, etc.)
    Creates appropriate function signature for injection mode.
    """
    call_count = [0]

    if scenario.injection_mode == "parameter":
        # Function with traigent_config parameter for parameter injection
        def test_func_with_config(
            text: str = "", traigent_config: Any = None, **kwargs: Any
        ) -> str:
            call_count[0] += 1

            # Check if function should raise
            if scenario.function_should_raise:
                if (
                    scenario.function_raise_on_call is None
                    or call_count[0] == scenario.function_raise_on_call
                ):
                    raise scenario.function_should_raise("Simulated error")

            # Check for custom return value
            if scenario.function_return_value is not None:
                return scenario.function_return_value

            # Default behavior
            return f"output: {text}"

        return test_func_with_config
    else:
        # Standard function for other injection modes
        def test_func(text: str = "", **kwargs: Any) -> str:
            call_count[0] += 1

            # Check if function should raise
            if scenario.function_should_raise:
                if (
                    scenario.function_raise_on_call is None
                    or call_count[0] == scenario.function_raise_on_call
                ):
                    raise scenario.function_should_raise("Simulated error")

            # Check for custom return value
            if scenario.function_return_value is not None:
                return scenario.function_return_value

            # Default behavior
            return f"output: {text}"

        return test_func


def build_objectives(
    objectives_spec: list[str | ObjectiveSpec],
) -> list[str]:
    """Convert objective specifications to simple list for decorator.

    For now, return simple string list. Multi-objective weighting
    is handled internally by the optimizer.
    """
    return [
        obj.name if isinstance(obj, ObjectiveSpec) else obj for obj in objectives_spec
    ]


def build_evaluator_kwargs(spec: EvaluatorSpec) -> dict[str, Any]:
    """Build evaluator keyword arguments from specification."""
    if spec.type == "custom" and spec.evaluator_fn:
        return {"custom_evaluator": spec.evaluator_fn}
    elif spec.type == "scoring_function" and spec.scoring_fn:
        return {"scoring_function": spec.scoring_fn}
    elif spec.type == "metric_functions" and spec.metric_fns:
        return {"metric_functions": spec.metric_fns}
    return {}


@pytest.fixture
def scenario_runner(
    dataset_factory: Callable[..., tuple[str, Dataset]],
) -> Callable[..., tuple[Any, OptimizationResult | Exception]]:
    """Runner for executing test scenarios.

    Returns a function that takes a TestScenario and returns
    (decorated_function, result_or_exception).

    Usage:
        func, result = await scenario_runner(scenario)
        if isinstance(result, Exception):
            # Handle expected failure
        else:
            # Validate result
    """

    async def run_scenario(
        scenario: TestScenario,
    ) -> tuple[Any, OptimizationResult | Exception]:
        """Execute a test scenario and return the result.

        Args:
            scenario: The test scenario specification

        Returns:
            Tuple of (decorated_function, result_or_exception)
        """
        # Import here to avoid circular imports and ensure fresh state
        from traigent.api.decorators import optimize

        # Create dataset if needed
        if scenario.dataset_path:
            dataset_path = scenario.dataset_path
        else:
            dataset_path, _ = dataset_factory(
                f"scenario_{scenario.name}",
                size=scenario.dataset_size,
            )

        # Build objectives
        objectives = build_objectives(scenario.objectives)

        # Build constraints
        constraints = [c.constraint_fn for c in scenario.constraints]

        # Build evaluator kwargs
        evaluator_kwargs = build_evaluator_kwargs(scenario.evaluator)

        # Create the test function
        test_func = create_test_function(scenario)

        # Build injection kwargs
        injection_kwargs: dict[str, Any] = {
            "injection_mode": scenario.injection_mode,
        }
        if scenario.injection_mode == "parameter":
            injection_kwargs["config_param"] = "traigent_config"

        # Apply decorator
        try:
            decorated = optimize(
                configuration_space=scenario.config_space,
                default_config=scenario.default_config,
                objectives=objectives,
                constraints=constraints if constraints else None,
                injection=injection_kwargs,
                execution={"execution_mode": scenario.execution_mode},
                mock=scenario.mock_mode_config,
                evaluation={"eval_dataset": dataset_path, **evaluator_kwargs},
                tvl_spec=scenario.tvl_spec_path,
                tvl_environment=scenario.tvl_environment,
            )(test_func)
        except Exception as e:
            return None, e

        # Run optimization
        try:
            result = await decorated.optimize(
                max_trials=scenario.max_trials,
                timeout=scenario.timeout,
            )
            return decorated, result
        except Exception as e:
            return decorated, e

    return run_scenario


@pytest.fixture
def result_validator() -> Callable[[TestScenario, Any], ValidationResult]:
    """Factory for creating result validators.

    Returns a function that validates an optimization result
    against a scenario specification.

    Usage:
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
    """

    def validate(
        scenario: TestScenario,
        result: OptimizationResult | Exception,
    ) -> ValidationResult:
        validator = ResultValidator(scenario, result)
        return validator.validate()

    return validate


# Commonly used config spaces for reuse


@pytest.fixture
def simple_config_space() -> dict[str, Any]:
    """Simple categorical-only config space."""
    return {
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.3, 0.7],
    }


@pytest.fixture
def continuous_config_space() -> dict[str, Any]:
    """Config space with continuous parameters (ranges)."""
    return {
        "temperature": (0.0, 1.0),
        "top_p": (0.5, 1.0),
    }


@pytest.fixture
def mixed_config_space() -> dict[str, Any]:
    """Config space with both categorical and continuous parameters."""
    return {
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": (0.0, 1.0),
        "max_tokens": [100, 500, 1000],
    }


# Data directory path


@pytest.fixture
def test_data_dir() -> Path:
    """Get path to the test data directory."""
    return Path(__file__).parent / "data"


# Helper for loading YAML scenario overrides


def load_scenario_overrides(path: Path) -> dict[str, Any]:
    """Load scenario overrides from a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of scenario overrides
    """
    if not path.exists():
        return {}

    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # YAML not installed, return empty
        return {}


# ============================================================================
# Tracing Fixtures
# ============================================================================


@pytest.fixture
def trace_capture(request: pytest.FixtureRequest) -> Generator[Any, None, None]:
    """Fixture for capturing and analyzing traces during test execution.

    Usage:
        def test_example(trace_capture):
            trace_capture.start()
            # Run optimization...
            trace_capture.start_scenario(scenario)
            # After test...
            captured = trace_capture.finish()
            # Analyze trace...
    """
    try:
        from .tracing.capture import TraceCapture
    except ImportError:
        pytest.skip(
            "OpenTelemetry not installed. Install with: pip install traigent[tracing]"
        )
        return

    test_name = request.node.name
    capture = TraceCapture(test_name)
    yield capture
    capture.shutdown()


@pytest.fixture
def traced_scenario_runner(
    scenario_runner: Callable[..., tuple[Any, OptimizationResult | Exception]],
    trace_capture: Any,
) -> Callable[..., tuple[Any, OptimizationResult | Exception, Any]]:
    """Scenario runner that also captures traces.

    Returns a function that takes a TestScenario and returns
    (decorated_function, result_or_exception, captured_trace).

    Usage:
        func, result, trace = await traced_scenario_runner(scenario)
        if isinstance(result, Exception):
            # Handle expected failure
        else:
            # Validate result and trace
    """

    async def run_with_trace(
        scenario: TestScenario,
    ) -> tuple[Any, OptimizationResult | Exception, Any]:
        """Execute a test scenario with trace capture.

        Args:
            scenario: The test scenario specification

        Returns:
            Tuple of (decorated_function, result_or_exception, captured_trace)
        """
        trace_capture.start()
        trace_capture.start_scenario(scenario)

        func, result = await scenario_runner(scenario)

        captured = trace_capture.finish()
        return func, result, captured

    return run_with_trace


@pytest.fixture
def trace_validator() -> Callable[..., Any]:
    """Validator for trace data.

    Returns a function that validates a trace against scenario expectations.

    Usage:
        validation = trace_validator(scenario, trace, result)
        assert validation.passed, validation.errors
    """
    try:
        from .tracing.analyzer import TraceAnalyzer
    except ImportError:
        pytest.skip(
            "OpenTelemetry not installed. Install with: pip install traigent[tracing]"
        )
        return

    def validate(
        scenario: TestScenario,
        trace: CapturedTrace,
        result: OptimizationResult | None = None,
    ) -> TraceValidationResult:
        """Validate trace against scenario expectations.

        Args:
            scenario: Test scenario with trace expectations
            trace: Captured trace data
            result: Optimization result for consistency checks

        Returns:
            TraceValidationResult with errors and warnings
        """
        analyzer = TraceAnalyzer(trace, result=result)
        return analyzer.validate(
            expectations=scenario.trace_expectations,
            skip_invariants=(
                scenario.trace_expectations.skip_invariants
                if scenario.trace_expectations
                else None
            ),
        )

    return validate
