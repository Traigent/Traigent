"""Tests for Haystack early stopping on constraint violation.

Coverage: Epic 4, Story 4.4 (Early Stopping on Constraint Violation)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from traigent.integrations.haystack import (
    EvaluationDataset,
    HaystackEvaluator,
    latency_constraint,
)
from traigent.integrations.haystack.execution import (
    ExampleResult,
    RunResult,
    execute_with_config,
)


class TestEarlyStopCallback:
    """Tests for execute_with_config early stop callback."""

    def test_callback_not_called_when_none(self):
        """Test execution works without callback."""
        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        result = execute_with_config(
            pipeline=pipeline,
            config={},
            dataset=dataset,
            early_stop_callback=None,
        )

        assert result.success
        assert len(result.example_results) == 1
        assert not result.stopped_early

    def test_callback_triggers_early_stop(self):
        """Test callback can trigger early stopping."""
        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "test1"}, "expected": "result1"},
                {"input": {"query": "test2"}, "expected": "result2"},
                {"input": {"query": "test3"}, "expected": "result3"},
                {"input": {"query": "test4"}, "expected": "result4"},
                {"input": {"query": "test5"}, "expected": "result5"},
            ]
        )

        # Callback that stops after 2 examples
        def stop_after_two(results: list[ExampleResult]) -> bool:
            return len(results) >= 2

        result = execute_with_config(
            pipeline=pipeline,
            config={},
            dataset=dataset,
            early_stop_callback=stop_after_two,
        )

        assert result.stopped_early
        assert len(result.example_results) == 2  # Stopped after 2

    def test_callback_not_triggered_when_returns_false(self):
        """Test callback returning False continues execution."""
        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "test1"}, "expected": "result1"},
                {"input": {"query": "test2"}, "expected": "result2"},
                {"input": {"query": "test3"}, "expected": "result3"},
            ]
        )

        # Callback that never stops
        def never_stop(results: list[ExampleResult]) -> bool:
            return False

        result = execute_with_config(
            pipeline=pipeline,
            config={},
            dataset=dataset,
            early_stop_callback=never_stop,
        )

        assert not result.stopped_early
        assert len(result.example_results) == 3  # All executed


class TestRunResultStoppedEarly:
    """Tests for RunResult.stopped_early field."""

    def test_default_is_false(self):
        """Test stopped_early defaults to False."""
        result = RunResult(config={})
        assert result.stopped_early is False

    def test_can_be_set_true(self):
        """Test stopped_early can be set to True."""
        result = RunResult(config={}, stopped_early=True)
        assert result.stopped_early is True


class TestEvaluatorEarlyStopParams:
    """Tests for HaystackEvaluator early stopping parameters."""

    def test_default_early_stop_disabled(self):
        """Test early stopping is disabled by default."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        assert evaluator.early_stop_on_violation is False

    def test_early_stop_can_be_enabled(self):
        """Test early stopping can be enabled."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            early_stop_on_violation=True,
        )

        assert evaluator.early_stop_on_violation is True

    def test_violation_threshold_default(self):
        """Test default violation threshold is 0.5."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        assert evaluator.violation_threshold == 0.5

    def test_min_examples_before_stop_default(self):
        """Test default min examples before stop is 3."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        assert evaluator.min_examples_before_stop == 3


class TestEarlyStopCallbackCreation:
    """Tests for _create_early_stop_callback method."""

    def test_returns_none_when_disabled(self):
        """Test callback is None when early stopping disabled."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            early_stop_on_violation=False,
            constraints=latency_constraint(p95_ms=100),
        )

        callback = evaluator._create_early_stop_callback()
        assert callback is None

    def test_returns_none_without_constraints(self):
        """Test callback is None without constraints."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            early_stop_on_violation=True,
            constraints=[],
        )

        callback = evaluator._create_early_stop_callback()
        assert callback is None

    def test_returns_none_without_latency_constraints(self):
        """Test callback is None without latency constraints."""
        from traigent.integrations.haystack import cost_constraint

        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            early_stop_on_violation=True,
            constraints=[cost_constraint(max_cost=0.05)],  # No latency
        )

        callback = evaluator._create_early_stop_callback()
        assert callback is None

    def test_returns_callback_with_latency_constraints(self):
        """Test callback is returned with latency constraints."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            early_stop_on_violation=True,
            constraints=latency_constraint(p95_ms=100),
        )

        callback = evaluator._create_early_stop_callback()
        assert callback is not None
        assert callable(callback)


class TestExampleViolatesLatency:
    """Tests for _example_violates_latency method."""

    def test_failed_example_is_violation(self):
        """Test failed examples count as violations."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        result = ExampleResult(
            example_index=0,
            input={"query": "test"},
            output=None,
            success=False,
            error="Failed",
        )

        constraints = latency_constraint(p95_ms=100)
        assert evaluator._example_violates_latency(result, constraints) is True

    def test_fast_example_not_violation(self):
        """Test fast examples don't violate."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        result = ExampleResult(
            example_index=0,
            input={"query": "test"},
            output={"response": "ok"},
            success=True,
            execution_time=0.05,  # 50ms
        )

        constraints = latency_constraint(p95_ms=100)  # 100ms limit
        assert evaluator._example_violates_latency(result, constraints) is False

    def test_slow_example_is_violation(self):
        """Test slow examples violate."""
        pipeline = MagicMock()
        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
        )

        result = ExampleResult(
            example_index=0,
            input={"query": "test"},
            output={"response": "ok"},
            success=True,
            execution_time=0.2,  # 200ms
        )

        constraints = latency_constraint(p95_ms=100)  # 100ms limit
        assert evaluator._example_violates_latency(result, constraints) is True


class TestEarlyStoppingIntegration:
    """Integration tests for early stopping."""

    @pytest.mark.asyncio
    async def test_early_stopping_triggers_on_slow_examples(self):
        """Test early stopping triggers when examples are slow."""
        pipeline = MagicMock()

        # Simulate slow responses
        def slow_run(**kwargs):
            time.sleep(0.02)  # 20ms - will exceed 10ms constraint
            return {"llm": {"replies": ["Response"]}}

        pipeline.run.side_effect = slow_run

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": f"test{i}"}, "expected": f"result{i}"}
                for i in range(10)  # 10 examples
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            early_stop_on_violation=True,
            violation_threshold=0.5,  # Stop when >50% violate
            min_examples_before_stop=3,
            constraints=latency_constraint(p95_ms=10),  # 10ms limit
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Should have stopped early
        assert result.aggregated_metrics.get("stopped_early") is True
        # Should have less than 10 examples
        assert result.total_examples < 10

    @pytest.mark.asyncio
    async def test_no_early_stopping_when_disabled(self):
        """Test no early stopping when disabled."""
        pipeline = MagicMock()

        def slow_run(**kwargs):
            time.sleep(0.02)  # 20ms
            return {"llm": {"replies": ["Response"]}}

        pipeline.run.side_effect = slow_run

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": f"test{i}"}, "expected": f"result{i}"}
                for i in range(5)
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            early_stop_on_violation=False,  # Disabled
            constraints=latency_constraint(p95_ms=10),
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Should not have stopped early
        assert result.aggregated_metrics.get("stopped_early") is not True
        # All examples executed
        assert result.total_examples == 5

    @pytest.mark.asyncio
    async def test_no_early_stopping_for_fast_examples(self):
        """Test no early stopping when examples are fast enough."""
        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": f"test{i}"}, "expected": f"result{i}"}
                for i in range(5)
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            early_stop_on_violation=True,
            violation_threshold=0.5,
            min_examples_before_stop=2,
            constraints=latency_constraint(p95_ms=10000),  # 10s limit - very lenient
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Should not have stopped early
        assert result.aggregated_metrics.get("stopped_early") is not True
        # All examples executed
        assert result.total_examples == 5

    @pytest.mark.asyncio
    async def test_min_examples_respected(self):
        """Test early stopping waits for min examples."""
        pipeline = MagicMock()

        def slow_run(**kwargs):
            time.sleep(0.02)  # 20ms
            return {"llm": {"replies": ["Response"]}}

        pipeline.run.side_effect = slow_run

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": f"test{i}"}, "expected": f"result{i}"}
                for i in range(10)
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            early_stop_on_violation=True,
            violation_threshold=0.1,  # Very low - would stop immediately
            min_examples_before_stop=5,  # But must wait for 5
            constraints=latency_constraint(p95_ms=10),
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Should have at least min_examples_before_stop
        assert result.total_examples >= 5
