"""Tests for evaluator configurations.

Tests default evaluator, custom evaluators, scoring functions, and metric functions.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    EvaluatorSpec,
    ExpectedResult,
    TestScenario,
    basic_scenario,
    evaluator_scenario,
)


class TestDefaultEvaluator:
    """Tests for the default evaluator."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_evaluator_works(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test default evaluator with basic scenario."""
        scenario = basic_scenario(
            name="default_evaluator",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_evaluator_produces_metrics(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test default evaluator produces accuracy metrics."""
        scenario = basic_scenario(
            name="default_with_metrics",
            max_trials=2,
            expected=ExpectedResult(required_metrics=["accuracy"]),
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestCustomEvaluator:
    """Tests for custom evaluator functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_basic(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator is used correctly."""
        from traigent.api.types import ExampleResult

        def custom_eval(func, config, example) -> ExampleResult:
            # Call the function
            actual_output = func(example.input_data.get("text", ""))

            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": 0.8, "custom_metric": 1.0},
                execution_time=0.01,
                success=True,
            )

        scenario = evaluator_scenario(
            name="custom_evaluator",
            evaluator_type="custom",
            evaluator_fn=custom_eval,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestScoringFunction:
    """Tests for scoring function evaluators."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_basic(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function evaluator."""

        def simple_scorer(expected: Any, actual: Any) -> float:
            # Simple string comparison
            if str(expected) == str(actual):
                return 1.0
            elif str(expected).lower() in str(actual).lower():
                return 0.5
            return 0.0

        scenario = evaluator_scenario(
            name="scoring_function",
            evaluator_type="scoring_function",
            scoring_fn=simple_scorer,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_with_partial_match(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function with partial matching."""

        def partial_scorer(expected: Any, actual: Any) -> float:
            exp_str = str(expected).lower()
            act_str = str(actual).lower()

            if exp_str == act_str:
                return 1.0

            # Count overlapping words
            exp_words = set(exp_str.split())
            act_words = set(act_str.split())

            if not exp_words:
                return 0.0

            overlap = len(exp_words & act_words)
            return overlap / len(exp_words)

        scenario = evaluator_scenario(
            name="partial_scorer",
            evaluator_type="scoring_function",
            scoring_fn=partial_scorer,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMetricFunctions:
    """Tests for metric function evaluators."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_metric_function(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator with single metric function."""

        def accuracy_metric(expected: Any, actual: Any) -> float:
            return 1.0 if str(expected) == str(actual) else 0.0

        scenario = evaluator_scenario(
            name="single_metric_fn",
            evaluator_type="metric_functions",
            metric_fns={"accuracy": accuracy_metric},
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_metric_functions(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator with multiple metric functions."""

        def accuracy_metric(expected: Any, actual: Any) -> float:
            return 1.0 if str(expected) == str(actual) else 0.0

        def length_ratio_metric(expected: Any, actual: Any) -> float:
            exp_len = len(str(expected))
            act_len = len(str(actual))
            if exp_len == 0:
                return 0.0
            return min(act_len / exp_len, 1.0)

        def contains_expected_metric(expected: Any, actual: Any) -> float:
            return 1.0 if str(expected) in str(actual) else 0.0

        scenario = evaluator_scenario(
            name="multi_metric_fn",
            evaluator_type="metric_functions",
            metric_fns={
                "accuracy": accuracy_metric,
                "length_ratio": length_ratio_metric,
                "containment": contains_expected_metric,
            },
            max_trials=2,
            expected=ExpectedResult(
                required_metrics=["accuracy", "length_ratio", "containment"]
            ),
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEvaluatorWithDataset:
    """Tests for evaluators with specific dataset configurations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_with_small_dataset(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator with minimal dataset."""
        scenario = basic_scenario(
            name="small_dataset_eval",
            dataset_size=2,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_with_larger_dataset(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator with larger dataset."""
        scenario = basic_scenario(
            name="larger_dataset_eval",
            dataset_size=10,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
