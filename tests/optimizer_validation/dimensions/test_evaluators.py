"""Tests for evaluator configurations.

Tests default evaluator, custom evaluators, scoring functions, and metric
functions.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
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
            gist_template="default-eval -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="default-metrics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="custom-eval -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="scoring-fn -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="partial-scorer -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="single-metric -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="multi-metric -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="small-dataset -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="large-dataset -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEvaluatorEdgeCases:
    """Tests for edge cases in evaluator behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_returns_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that returns None instead of ExampleResult."""
        from traigent.api.types import ExampleResult

        def none_evaluator(func, config, example) -> ExampleResult:
            return None  # type: ignore[return-value]

        scenario = evaluator_scenario(
            name="evaluator_returns_none",
            evaluator_type="custom",
            evaluator_fn=none_evaluator,
            max_trials=2,
            gist_template="eval-none -> {error_type()} | {status()}",
        )

        _, _ = await scenario_runner(scenario)

        # Should handle None return gracefully

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_raises_exception(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that raises an exception."""
        from traigent.api.types import ExampleResult

        def raising_evaluator(func, config, example) -> ExampleResult:
            raise ValueError("Evaluator failed")

        scenario = evaluator_scenario(
            name="evaluator_raises",
            evaluator_type="custom",
            evaluator_fn=raising_evaluator,
            max_trials=2,
            gist_template="eval-raises -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle exception gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_returns_nan(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function that returns float('nan')."""

        def nan_scorer(expected: Any, actual: Any) -> float:
            return float("nan")

        scenario = evaluator_scenario(
            name="scoring_nan",
            evaluator_type="scoring_function",
            scoring_fn=nan_scorer,
            max_trials=2,
            gist_template="scoring-nan -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle NaN scores
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_returns_inf(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function that returns float('inf')."""

        def inf_scorer(expected: Any, actual: Any) -> float:
            return float("inf")

        scenario = evaluator_scenario(
            name="scoring_inf",
            evaluator_type="scoring_function",
            scoring_fn=inf_scorer,
            max_trials=2,
            gist_template="scoring-inf -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle infinity scores
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_returns_negative(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function that returns negative values."""

        def negative_scorer(expected: Any, actual: Any) -> float:
            return -0.5

        scenario = evaluator_scenario(
            name="scoring_negative",
            evaluator_type="scoring_function",
            scoring_fn=negative_scorer,
            max_trials=2,
            gist_template="scoring-neg -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle negative scores
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_returns_above_one(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function that returns values > 1.0."""

        def above_one_scorer(expected: Any, actual: Any) -> float:
            return 1.5  # Above 1.0

        scenario = evaluator_scenario(
            name="scoring_above_one",
            evaluator_type="scoring_function",
            scoring_fn=above_one_scorer,
            max_trials=2,
            gist_template="scoring-gt1 -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle scores > 1.0
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_function_returns_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metric function that returns string instead of float."""

        def string_metric(expected: Any, actual: Any) -> Any:
            return "not a number"

        scenario = evaluator_scenario(
            name="metric_returns_string",
            evaluator_type="metric_functions",
            metric_fns={"accuracy": string_metric},
            max_trials=2,
            gist_template="metric-str -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle wrong return type
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_raises_exception(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function that raises an exception."""

        def raising_scorer(expected: Any, actual: Any) -> float:
            raise RuntimeError("Scorer failed")

        scenario = evaluator_scenario(
            name="scoring_raises",
            evaluator_type="scoring_function",
            scoring_fn=raising_scorer,
            max_trials=2,
            gist_template="scoring-raises -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle exception gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_function_returns_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metric function that returns None."""

        def none_metric(expected: Any, actual: Any) -> Any:
            return None

        scenario = evaluator_scenario(
            name="metric_returns_none",
            evaluator_type="metric_functions",
            metric_fns={"accuracy": none_metric},
            max_trials=2,
            gist_template="metric-none -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle None return
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_intermittent_failure(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that fails intermittently."""
        from traigent.api.types import ExampleResult

        call_count = [0]

        def intermittent_evaluator(func, config, example) -> ExampleResult:
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("Intermittent failure")

            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": 0.8},
                execution_time=0.01,
                success=True,
            )

        scenario = evaluator_scenario(
            name="intermittent_evaluator",
            evaluator_type="custom",
            evaluator_fn=intermittent_evaluator,
            max_trials=3,
            gist_template="intermittent -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle intermittent failures
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_metric_functions_dict(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with empty metric functions dictionary."""
        scenario = evaluator_scenario(
            name="empty_metrics",
            evaluator_type="metric_functions",
            metric_fns={},  # Empty dict
            max_trials=2,
            gist_template="empty-metrics -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should fail or use defaults
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
