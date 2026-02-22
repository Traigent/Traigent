"""Shared test fixtures and configuration for optimizer validation tests.

This module provides:
- Scenario runner for executing TestScenario specifications
- Dataset factory for creating test datasets
- Result validator fixture
- Test function factories
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample

from .specs.scenario import EvaluatorSpec, ExpectedResult, ObjectiveSpec, TestScenario
from .specs.validators import ResultValidator, ValidationResult

if TYPE_CHECKING:
    from traigent.api.types import OptimizationResult

    from .tracing.analyzer import TraceValidationResult
    from .tracing.capture import CapturedTrace


def _serialize_config_space(config_space: dict[str, Any]) -> dict[str, Any]:
    """Serialize config space with cardinality analysis for evidence JSON."""
    result = {}
    categorical_product = 1
    has_continuous = False

    for key, value in config_space.items():
        if isinstance(value, (list, tuple)):
            # Categorical or continuous range
            if isinstance(value, tuple) and len(value) == 2:
                # Continuous range
                result[key] = {
                    "type": "continuous",
                    "range": list(value),
                    "cardinality": "∞",
                }
                has_continuous = True
            else:
                # Categorical
                cardinality = len(value)
                result[key] = {
                    "type": "categorical",
                    "values": list(value),
                    "cardinality": cardinality,
                }
                categorical_product *= cardinality
        else:
            result[key] = {"type": "fixed", "value": str(value), "cardinality": 1}

    # Add summary for search space analysis
    result["_summary"] = {
        "categorical_combinations": categorical_product,
        "has_continuous": has_continuous,
        "total_space": "infinite" if has_continuous else categorical_product,
    }
    return result


def _serialize_expected(expected: ExpectedResult) -> dict[str, Any]:
    """Serialize ExpectedResult for evidence JSON."""
    return {
        "outcome": expected.outcome.name,
        "min_trials": expected.min_trials,
        "max_trials": expected.max_trials,
        "expected_stop_reason": expected.expected_stop_reason,
        "best_score_range": (
            list(expected.best_score_range) if expected.best_score_range else None
        ),
        "required_metrics": expected.required_metrics,
        "error_type": (expected.error_type.__name__ if expected.error_type else None),
        "error_message_contains": expected.error_message_contains,
    }


def _serialize_actual(result: Any) -> dict[str, Any]:
    """Serialize OptimizationResult or Exception for evidence JSON."""
    if isinstance(result, Exception):
        return {
            "type": "exception",
            "exception_type": type(result).__name__,
            "message": str(result)[:500],
        }
    # OptimizationResult
    try:
        return {
            "type": "success",
            "trial_count": len(result.trials) if hasattr(result, "trials") else 0,
            "stop_reason": getattr(result, "stop_reason", None),
            "best_score": getattr(result, "best_score", None),
            "best_config": getattr(result, "best_config", None),
            "duration": getattr(result, "duration", None),
        }
    except Exception as exc:
        return {
            "type": "exception",
            "exception_type": type(exc).__name__,
            "message": f"Serialization error: {exc}",
        }


def _serialize_checks(validation: ValidationResult) -> list[dict[str, Any]]:
    """Serialize validation checks for evidence JSON."""
    # Build a map of errors by category
    error_map = {e.category: e for e in validation.errors}

    checks = []
    # Check each standard validation category
    for category in [
        "outcome",
        "trial_count",
        "stop_reason",
        "best_score",
        "required_metrics",
        "error_type",
        "error_message",
    ]:
        if category in error_map:
            e = error_map[category]
            checks.append(
                {
                    "name": category,
                    "check": category,
                    "passed": False,
                    "expected": e.expected,
                    "actual": e.actual,
                    "message": e.message,
                }
            )
        else:
            # Category passed (no error)
            checks.append({"name": category, "check": category, "passed": True})

    return checks


def _get_trial_score(trial: Any, objectives: list[str] | None = None) -> float | None:
    """Extract the primary score from a trial's metrics.

    Args:
        trial: A TrialResult object
        objectives: List of objective names to look for

    Returns:
        The primary score value, or None if not found
    """
    if not hasattr(trial, "metrics") or not trial.metrics:
        return None

    metrics = trial.metrics

    # Try objective names first
    if objectives:
        for obj_name in objectives:
            if obj_name in metrics:
                return metrics[obj_name]

    # Try common score field names
    for key in ["score", "accuracy", "objective", "loss", "error"]:
        if key in metrics:
            return metrics[key]

    # Return first metric if only one
    if len(metrics) == 1:
        return list(metrics.values())[0]

    return None


def _serialize_trials(
    result: Any, objectives: list[str] | None = None
) -> list[dict[str, Any]]:
    """Serialize all trials with full details for evidence JSON.

    Args:
        result: An OptimizationResult or Exception
        objectives: List of objective names for score extraction

    Returns:
        List of serialized trial data with configs, scores, and status
    """
    if isinstance(result, Exception) or not hasattr(result, "trials"):
        return []

    trials = []
    best_score = getattr(result, "best_score", None)

    for i, trial in enumerate(result.trials):
        score = _get_trial_score(trial, objectives)

        trial_data: dict[str, Any] = {
            "index": i + 1,
            "trial_id": getattr(trial, "trial_id", None),
            "config": getattr(trial, "config", {}),
            "score": score,
            "status": (
                trial.status.value
                if hasattr(trial, "status") and hasattr(trial.status, "value")
                else str(getattr(trial, "status", "unknown"))
            ),
            "duration": getattr(trial, "duration", None),
            "is_best": score is not None
            and best_score is not None
            and score == best_score,
            "metrics": (
                dict(getattr(trial, "metrics", {})) if hasattr(trial, "metrics") else {}
            ),
        }

        # Include error message if present
        error_msg = getattr(trial, "error_message", None)
        if error_msg:
            trial_data["error"] = str(error_msg)[:200]

        trials.append(trial_data)

    return trials


def _load_dataset_info(
    path: str | None, max_samples: int = 3
) -> tuple[list[dict[str, Any]], int | None]:
    """Load samples and count lines from a dataset file for evidence display.

    Args:
        path: Path to JSONL dataset file
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (samples list, total line count or None if file not found)
    """
    if not path:
        return [], None

    try:
        samples: list[dict[str, Any]] = []
        total_lines = 0
        with open(path) as f:
            for i, line in enumerate(f):
                total_lines += 1
                if i < max_samples:
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        return samples, total_lines
    except OSError:
        return [], None


def _get_generated_dataset_path(scenario_name: str) -> str:
    """Get the expected path for an auto-generated dataset.

    Args:
        scenario_name: The scenario name used in dataset generation

    Returns:
        Expected path to the generated JSONL file
    """
    # Sanitize scenario name to prevent path injection
    # Only allow alphanumeric, underscore, and hyphen characters
    safe_name = "".join(c for c in scenario_name if c.isalnum() or c in "_-")
    if not safe_name:
        safe_name = "unnamed"
    base_dir = Path(__file__).parent / "data" / "_generated"
    return str(base_dir / f"scenario_{safe_name}.jsonl")


def _resolve_gist_template(
    template: str | None,
    scenario: TestScenario,
    result: Any,
    validation: ValidationResult,
) -> dict[str, Any]:
    """Resolve gist template to both raw template and resolved values.

    Gist templates use a function syntax with placeholders like {status()}, {error_type()},
    etc. This function extracts all available values and resolves the template to a
    final human-readable string.

    Args:
        template: Gist template string with placeholders like {status()}
        scenario: The test scenario specification
        result: The optimization result or exception
        validation: The validation result

    Returns:
        Dict with 'template' (raw), 'resolved' (final string), and 'values' (dict of resolved values)
    """
    if not template:
        return {"template": None, "resolved": None, "values": {}}

    # Extract values for all available functions
    values: dict[str, str] = {}

    # status() - overall pass/fail
    values["status"] = "PASS" if validation.passed else "FAIL"

    # outcome() - success/exception
    if isinstance(result, Exception):
        values["outcome"] = "EXCEPTION"
    else:
        values["outcome"] = "SUCCESS"

    # error_type() - exception class name
    if isinstance(result, Exception):
        values["error_type"] = type(result).__name__
    else:
        values["error_type"] = "--"

    # trial_count() - number of trials run
    if hasattr(result, "trials") and result.trials:
        values["trial_count"] = str(len(result.trials))
    else:
        values["trial_count"] = "0"

    # best_score() - best score achieved
    if hasattr(result, "best_score") and result.best_score is not None:
        values["best_score"] = f"{result.best_score:.4f}"
    else:
        values["best_score"] = "--"

    # stop_reason() - why optimization stopped
    values["stop_reason"] = getattr(result, "stop_reason", "--") or "--"

    # duration() - total duration
    duration = getattr(result, "duration", None)
    values["duration"] = f"{duration:.2f}s" if duration else "--"

    # config_space_size() - from scenario
    config_space = scenario.config_space
    if config_space:
        categorical_product = 1
        has_continuous = False
        for value in config_space.values():
            if isinstance(value, tuple) and len(value) == 2:
                has_continuous = True
            elif isinstance(value, list):
                categorical_product *= len(value)
        if has_continuous:
            values["config_space_size"] = "infinite"
        else:
            values["config_space_size"] = f"{categorical_product} combos"
    else:
        values["config_space_size"] = "?"

    # injection_mode() - from scenario
    values["injection_mode"] = str(scenario.injection_mode)

    # algorithm() - from mock_mode_config
    if scenario.mock_mode_config:
        algo = scenario.mock_mode_config.get("optimizer", "")
        if not algo and scenario.mock_mode_config.get("sampler"):
            algo = f"optuna_{scenario.mock_mode_config.get('sampler')}"
        values["algorithm"] = algo or "--"
    else:
        values["algorithm"] = "--"

    # expected_outcome() - what the test expects
    values["expected_outcome"] = scenario.expected.outcome.name

    # expected_trials() - expected trial count range
    min_t = scenario.expected.min_trials
    max_t = scenario.expected.max_trials
    if max_t:
        values["expected_trials"] = f"{min_t}-{max_t}"
    else:
        values["expected_trials"] = f">={min_t}"

    # Resolve template by replacing all {func()} placeholders
    resolved = template
    for func_name, value in values.items():
        resolved = resolved.replace(f"{{{func_name}()}}", value)

    return {
        "template": template,
        "resolved": resolved,
        "values": values,
    }


def emit_test_evidence(
    scenario: TestScenario,
    result: Any,
    validation: ValidationResult,
    dataset_path: str | None = None,
) -> None:
    """Emit structured evidence JSON for the test viewer.

    This evidence is captured by pytest-json-report and extracted
    by the viewer server to show full validation details.

    The evidence includes:
    - Complete scenario configuration (inputs)
    - Dataset information with path and sample content
    - Expected vs actual comparison
    - Per-trial results with configs and scores
    - Validation check details

    Args:
        scenario: The test scenario specification
        result: The optimization result or exception
        validation: The validation result
        dataset_path: Optional path to the dataset used (for display)
    """
    # Extract objective names for score lookup
    objective_names = [
        obj.name if hasattr(obj, "name") else obj for obj in scenario.objectives
    ]

    # Determine actual dataset path
    # Priority: explicit parameter > scenario specification > auto-generated path
    if dataset_path:
        actual_path = dataset_path
    elif scenario.dataset_path:
        actual_path = scenario.dataset_path
    else:
        # Auto-generated dataset - try to find it
        actual_path = _get_generated_dataset_path(scenario.name)

    # Load dataset samples and get actual file size
    samples, actual_size = _load_dataset_info(actual_path)

    # Extract algorithm from mock_mode_config if present
    algorithm = None
    if scenario.mock_mode_config:
        algorithm = scenario.mock_mode_config.get("optimizer")
        if not algorithm and scenario.mock_mode_config.get("sampler"):
            algorithm = f"optuna_{scenario.mock_mode_config.get('sampler')}"

    # Extract parallel mode
    parallel_mode = None
    if scenario.parallel_config:
        parallel_mode = scenario.parallel_config.get("mode")
        if not parallel_mode:
            trial_concurrency = scenario.parallel_config.get("trial_concurrency", 1)
            parallel_mode = "parallel" if trial_concurrency > 1 else "sequential"

    # Serialize constraints
    constraint_names = (
        [c.name for c in scenario.constraints] if scenario.constraints else []
    )

    # Resolve gist template
    gist_template = getattr(scenario, "gist_template", None)
    gist_data = _resolve_gist_template(gist_template, scenario, result, validation)

    evidence = {
        "type": "TEST_EVIDENCE",
        "gist": gist_data,
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "config_space": _serialize_config_space(scenario.config_space),
            "injection_mode": scenario.injection_mode,
            "execution_mode": scenario.execution_mode,
            "algorithm": algorithm,
            "parallel_mode": parallel_mode,
            "max_trials": scenario.max_trials,
            "timeout": scenario.timeout,
            "dataset_size": scenario.dataset_size,
            "objectives": objective_names,
            "constraints": constraint_names,
            "gist_template": gist_template,
        },
        "dataset": {
            "path": actual_path,
            "size": actual_size if actual_size is not None else scenario.dataset_size,
            "configured_size": scenario.dataset_size,
            "samples": samples,
            "is_auto_generated": scenario.dataset_path is None and dataset_path is None,
        },
        "expected": _serialize_expected(scenario.expected),
        "actual": _serialize_actual(result),
        "trials": _serialize_trials(result, objective_names),
        "validation_checks": _serialize_checks(validation),
        "passed": validation.passed,
        "warnings": validation.warnings,
    }
    # Use logging to emit evidence - this is captured by pytest-json-report
    # in the call.log field, which serve.py extracts for the viewer
    import logging

    evidence_logger = logging.getLogger("traigent.test_evidence")
    evidence_logger.info(json.dumps(evidence))


# Ensure mock mode is enabled for all tests in this module
@pytest.fixture(autouse=True)
def ensure_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock LLM and offline mode are enabled for all optimizer validation tests."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
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
    if scenario.custom_function:
        return scenario.custom_function

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
        # Opt-in for attribute mode + parallel execution to avoid safety guard
        parallel_config_candidate = scenario.parallel_config
        if parallel_config_candidate is None and scenario.mock_mode_config:
            parallel_trials = scenario.mock_mode_config.get("parallel_trials")
            if parallel_trials is not None:
                parallel_config_candidate = {"trial_concurrency": parallel_trials}
                if parallel_trials > 1:
                    parallel_config_candidate["mode"] = "parallel"
        # Note: attribute mode + parallel trials is now unconditionally blocked.
        # Test scenarios should not use this combination - it will raise ValueError.

        # Generate per-test seed for mock mode reproducibility
        # Use scenario name to create deterministic seed unique to each test
        mock_config = (
            dict(scenario.mock_mode_config) if scenario.mock_mode_config else {}
        )
        if "random_seed" not in mock_config:
            # Use a stable hash to avoid process-randomized Python hash() output.
            seed_bytes = hashlib.sha256(scenario.name.encode("utf-8")).digest()
            mock_config["random_seed"] = int.from_bytes(seed_bytes[:8], "big") % (
                2**31
            )

        # Apply decorator
        try:
            decorated = optimize(
                configuration_space=scenario.config_space,
                default_config=scenario.default_config,
                objectives=objectives,
                constraints=constraints if constraints else None,
                injection=injection_kwargs,
                execution={"execution_mode": scenario.execution_mode},
                mock=mock_config,
                evaluation={"eval_dataset": dataset_path, **evaluator_kwargs},
                tvl_spec=scenario.tvl_spec_path,
                tvl_environment=scenario.tvl_environment,
            )(test_func)
        except Exception as e:
            return None, e

        # Run optimization
        # Extract parallel configuration from scenario
        # parallel_trials from mock_mode_config needs to be converted to parallel_config
        parallel_config = None
        if scenario.parallel_config:
            parallel_config = scenario.parallel_config
        elif scenario.mock_mode_config:
            parallel_trials = scenario.mock_mode_config.get("parallel_trials")
            if parallel_trials is not None:
                # Convert legacy parallel_trials to parallel_config format
                parallel_config = {"trial_concurrency": parallel_trials}
                if parallel_trials > 1:
                    parallel_config["mode"] = "parallel"

        try:
            optimize_kwargs: dict[str, Any] = {
                "max_trials": scenario.max_trials,
                "timeout": scenario.timeout,
            }
            if parallel_config is not None:
                optimize_kwargs["parallel_config"] = parallel_config

            result = await decorated.optimize(**optimize_kwargs)
            return decorated, result
        except Exception as e:
            return decorated, e

    return run_scenario


@pytest.fixture
def result_validator() -> Callable[..., ValidationResult]:
    """Factory for creating result validators with behavioral validation.

    Returns a function that validates an optimization result
    against a scenario specification, including automatic behavioral
    validation based on scenario dimensions.

    Automatically emits structured evidence JSON for the test viewer,
    enabling full visibility into expected vs actual values and
    per-check validation results.

    Usage:
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # With dataset path for evidence display:
        validation = result_validator(scenario, result, dataset_path="/path/to/data.jsonl")

        # Skip specific behavioral validators:
        validation = result_validator(scenario, result, skip_behavioral=["grid_search"])
    """

    def validate(
        scenario: TestScenario,
        result: OptimizationResult | Exception,
        dataset_path: str | None = None,
        skip_behavioral: list[str] | None = None,
    ) -> ValidationResult:
        validator = ResultValidator(scenario, result)
        validation_result = validator.validate()

        # Apply behavioral validators (only for non-exception results)
        if not isinstance(result, Exception):
            from .specs.behavioral_validators import apply_behavioral_validators

            behavioral_results = apply_behavioral_validators(
                scenario, result, skip_validators=skip_behavioral
            )

            # Merge behavioral errors into main validation result
            for bv_result in behavioral_results:
                for error in bv_result.errors:
                    validation_result.add_error(
                        category=f"behavioral:{bv_result.validator_name}:{error.category}",
                        message=error.message,
                        expected=error.expected,
                        actual=error.actual,
                    )
                for warning in bv_result.warnings:
                    validation_result.add_warning(
                        f"[{bv_result.validator_name}] {warning}"
                    )

        # Emit evidence for the test viewer
        emit_test_evidence(
            scenario, result, validation_result, dataset_path=dataset_path
        )

        return validation_result

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
