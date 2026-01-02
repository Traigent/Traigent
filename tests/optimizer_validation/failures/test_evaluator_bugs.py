"""Tests for handling bugs in evaluation functions.

Tests scenarios where custom evaluators raise exceptions,
return invalid types, or produce empty/malformed metrics.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    evaluator_scenario,
)


class TestEvaluatorRaises:
    """Tests for evaluators that raise exceptions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_always_raises(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator that always raises."""

        def failing_evaluator(func, config, example):
            raise ValueError("Evaluator failed intentionally")

        scenario = evaluator_scenario(
            name="evaluator_always_raises",
            evaluator_type="custom",
            evaluator_fn=failing_evaluator,
            should_fail=True,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
            gist_template="eval-raises -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - trials may fail but shouldn't crash
        if not isinstance(result, Exception):
            # Expect failed trials
            assert len(result.trials) >= 1
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_raises_type_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that raises TypeError."""

        def type_error_evaluator(func, config, example):
            raise TypeError("Invalid type in evaluator")

        scenario = evaluator_scenario(
            name="evaluator_type_error",
            evaluator_type="custom",
            evaluator_fn=type_error_evaluator,
            should_fail=True,
            max_trials=2,
            gist_template="eval-type-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        if not isinstance(result, Exception):
            assert len(result.trials) >= 1
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_raises_runtime_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that raises RuntimeError."""

        def runtime_error_evaluator(func, config, example):
            raise RuntimeError("Runtime error in evaluator")

        scenario = evaluator_scenario(
            name="evaluator_runtime_error",
            evaluator_type="custom",
            evaluator_fn=runtime_error_evaluator,
            should_fail=True,
            max_trials=2,
            gist_template="eval-runtime-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        if not isinstance(result, Exception):
            assert len(result.trials) >= 1
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEvaluatorReturnsInvalidType:
    """Tests for evaluators that return wrong types."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_returns_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning string instead of ExampleResult."""

        def string_evaluator(func, config, example):
            return "not an ExampleResult"

        scenario = evaluator_scenario(
            name="evaluator_returns_string",
            evaluator_type="custom",
            evaluator_fn=string_evaluator,
            max_trials=2,
            gist_template="eval-ret-str -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully (may raise or produce failed trials)
        # The key is no unexpected crash

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_returns_dict(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning dict instead of ExampleResult."""

        def dict_evaluator(func, config, example):
            return {"accuracy": 0.8}  # Wrong type

        scenario = evaluator_scenario(
            name="evaluator_returns_dict",
            evaluator_type="custom",
            evaluator_fn=dict_evaluator,
            max_trials=2,
            gist_template="eval-ret-dict -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_returns_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning None."""

        def none_evaluator(func, config, example):
            return None

        scenario = evaluator_scenario(
            name="evaluator_returns_none",
            evaluator_type="custom",
            evaluator_fn=none_evaluator,
            max_trials=2,
            gist_template="eval-ret-none -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEvaluatorMalformedOutput:
    """Tests for evaluators that return malformed results."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_empty_metrics(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning empty metrics dict."""
        from traigent.api.types import ExampleResult

        def empty_metrics_evaluator(func, config, example):
            return ExampleResult(
                example_id="test",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output="output",
                metrics={},  # Empty metrics
                execution_time=0.01,
                success=True,
            )

        scenario = evaluator_scenario(
            name="evaluator_empty_metrics",
            evaluator_type="custom",
            evaluator_fn=empty_metrics_evaluator,
            max_trials=2,
            gist_template="eval-empty-metrics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - empty metrics might be allowed
        if not isinstance(result, Exception):
            assert len(result.trials) >= 1
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_negative_metrics(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning negative metric values."""
        from traigent.api.types import ExampleResult

        def negative_metrics_evaluator(func, config, example):
            return ExampleResult(
                example_id="test",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output="output",
                metrics={"accuracy": -0.5, "cost": -1.0},  # Negative values
                execution_time=0.01,
                success=True,
            )

        scenario = evaluator_scenario(
            name="evaluator_negative_metrics",
            evaluator_type="custom",
            evaluator_fn=negative_metrics_evaluator,
            max_trials=2,
            gist_template="eval-neg-metrics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        if not isinstance(result, Exception):
            assert len(result.trials) >= 1
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_nan_metrics(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning NaN metric values."""
        from traigent.api.types import ExampleResult

        def nan_metrics_evaluator(func, config, example):
            return ExampleResult(
                example_id="test",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output="output",
                metrics={"accuracy": float("nan")},  # NaN value
                execution_time=0.01,
                success=True,
            )

        scenario = evaluator_scenario(
            name="evaluator_nan_metrics",
            evaluator_type="custom",
            evaluator_fn=nan_metrics_evaluator,
            max_trials=2,
            gist_template="eval-nan-metrics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestScoringFunctionBugs:
    """Tests for bugs in scoring functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_raises(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function that raises exception."""

        def failing_scorer(expected: Any, actual: Any) -> float:
            raise ValueError("Scoring failed")

        scenario = evaluator_scenario(
            name="scorer_raises",
            evaluator_type="scoring_function",
            scoring_fn=failing_scorer,
            max_trials=2,
            gist_template="scorer-raises -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_returns_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function returning string instead of float."""

        def string_scorer(expected: Any, actual: Any) -> Any:
            return "0.8"  # String instead of float

        scenario = evaluator_scenario(
            name="scorer_returns_string",
            evaluator_type="scoring_function",
            scoring_fn=string_scorer,
            max_trials=2,
            gist_template="scorer-ret-str -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully (may convert or fail)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMetricFunctionBugs:
    """Tests for bugs in metric functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_function_raises(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metric function that raises exception."""

        def failing_metric(expected: Any, actual: Any) -> float:
            raise ValueError("Metric calculation failed")

        scenario = evaluator_scenario(
            name="metric_fn_raises",
            evaluator_type="metric_functions",
            metric_fns={"accuracy": failing_metric},
            max_trials=2,
            gist_template="metric-raises -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_one_metric_function_fails(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test when one of multiple metric functions fails."""

        def good_metric(expected: Any, actual: Any) -> float:
            return 0.8

        def bad_metric(expected: Any, actual: Any) -> float:
            raise ValueError("This metric failed")

        scenario = evaluator_scenario(
            name="partial_metric_failure",
            evaluator_type="metric_functions",
            metric_fns={
                "good": good_metric,
                "bad": bad_metric,
            },
            max_trials=2,
            gist_template="partial-metric-fail -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
