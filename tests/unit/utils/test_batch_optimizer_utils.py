"""Tests for batch optimizer utilities."""

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.metrics import MetricsEvaluationResult as EvaluationResult
from traigent.invokers.base import InvocationResult
from traigent.utils.batch_optimizer_utils import (
    BatchOptimizationHelper,
    BatchOptimizationStats,
    create_batch_progress_callback,
    parallel_config_evaluation,
)


class MockInvoker:
    """Mock invoker for testing."""

    async def invoke(self, func, config, input_data):
        """Mock single invocation."""
        return InvocationResult(
            result=f"mock_output_{input_data.get('value', 0)}",
            is_successful=True,
            execution_time=0.1,
        )

    async def invoke_batch(self, func, config, input_batch):
        """Mock batch invocation."""
        return [
            InvocationResult(
                result=f"mock_output_{input_data.get('value', 0)}",
                is_successful=True,
                execution_time=0.1,
            )
            for input_data in input_batch
        ]


class MockEvaluator:
    """Mock evaluator for testing."""

    async def evaluate(self, invocation_results, expected_outputs, dataset):
        """Mock evaluation."""
        successful_count = sum(1 for r in invocation_results if r.is_successful)
        total_count = len(invocation_results)

        return EvaluationResult(
            metrics={
                "accuracy": successful_count / max(1, total_count),
                "success_rate": successful_count / max(1, total_count),
            },
            total_invocations=total_count,
            successful_invocations=successful_count,
            duration=0.5,
        )


class TestBatchOptimizationStats:
    """Test suite for BatchOptimizationStats."""

    def test_init_default_values(self):
        """Test BatchOptimizationStats initialization."""
        stats = BatchOptimizationStats()

        assert stats.total_configurations == 0
        assert stats.processed_configurations == 0
        assert stats.successful_evaluations == 0
        assert stats.failed_evaluations == 0
        assert stats.total_duration == 0.0
        assert stats.avg_batch_size == 0.0
        assert stats.throughput == 0.0

    def test_init_custom_values(self):
        """Test BatchOptimizationStats initialization with custom values."""
        stats = BatchOptimizationStats(
            total_configurations=10,
            processed_configurations=8,
            successful_evaluations=6,
            failed_evaluations=2,
            total_duration=5.0,
            throughput=1.6,
        )

        assert stats.total_configurations == 10
        assert stats.processed_configurations == 8
        assert stats.successful_evaluations == 6
        assert stats.failed_evaluations == 2
        assert stats.total_duration == 5.0
        assert stats.throughput == 1.6


class TestBatchOptimizationHelper:
    """Test suite for BatchOptimizationHelper."""

    def setup_method(self):
        """Set up test data."""
        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
                EvaluationExample(input_data={"value": 3}, expected_output="expected3"),
            ],
            name="test_dataset",
        )

        self.configurations = [
            {"param1": 1, "param2": 0.1},
            {"param1": 2, "param2": 0.2},
        ]

        self.invoker = MockInvoker()
        self.evaluator = MockEvaluator()

    def test_init_default_values(self):
        """Test BatchOptimizationHelper initialization with defaults."""
        helper = BatchOptimizationHelper()

        assert helper.adaptive_batching is True
        assert helper.adaptive_sizer is not None
        assert helper.batch_size == 10
        assert isinstance(helper.stats, BatchOptimizationStats)

    def test_init_non_adaptive(self):
        """Test BatchOptimizationHelper initialization without adaptive batching."""
        helper = BatchOptimizationHelper(adaptive_batching=False, initial_batch_size=5)

        assert helper.adaptive_batching is False
        assert helper.adaptive_sizer is None
        assert helper.batch_size == 5

    def test_evaluate_configurations_batch_success(self):
        """Test successful batch evaluation of configurations."""
        import asyncio

        async def run_test():
            helper = BatchOptimizationHelper(
                adaptive_batching=False, initial_batch_size=2
            )

            def mock_func(value, config):
                return f"processed_{value}_{config.get('param1', 0)}"

            progress_updates = []

            def progress_callback(processed, total):
                progress_updates.append((processed, total))

            results = await helper.evaluate_configurations_batch(
                configurations=self.configurations,
                func=mock_func,
                dataset=self.dataset,
                invoker=self.invoker,
                evaluator=self.evaluator,
                progress_callback=progress_callback,
            )

            # Check results
            assert len(results) == 2
            assert all(r is not None for r in results)
            assert all(isinstance(r, EvaluationResult) for r in results)

            # Check progress updates
            assert len(progress_updates) == 2
            assert progress_updates[-1] == (2, 2)  # Final progress

            # Check statistics
            stats = helper.get_optimization_stats()
            assert stats.total_configurations == 2
            assert stats.processed_configurations == 2
            assert stats.successful_evaluations == 2
            assert stats.failed_evaluations == 0
            assert stats.total_duration > 0

        asyncio.run(run_test())

    def test_evaluate_configurations_batch_with_failure(self):
        """Test batch evaluation with some failures."""
        import asyncio

        async def run_test():
            helper = BatchOptimizationHelper(adaptive_batching=False)

            # Mock function that fails for certain configs
            def failing_func(value, config):
                if config.get("param1") == 2:
                    raise ValueError("Mock failure")
                return f"processed_{value}"

            # Mock evaluator that returns None for failed configs
            class FailingEvaluator:
                async def evaluate(self, invocation_results, expected_outputs, dataset):
                    # Simulate evaluation failure
                    raise ValueError("Evaluation failed")

            failing_evaluator = FailingEvaluator()

            results = await helper.evaluate_configurations_batch(
                configurations=self.configurations,
                func=failing_func,
                dataset=self.dataset,
                invoker=self.invoker,
                evaluator=failing_evaluator,
            )

            # All evaluations should fail
            assert len(results) == 2
            assert all(r is None for r in results)

            # Check statistics
            stats = helper.get_optimization_stats()
            assert stats.total_configurations == 2
            assert stats.processed_configurations == 2
            assert stats.successful_evaluations == 0
            assert stats.failed_evaluations == 2

        asyncio.run(run_test())

    def test_get_current_batch_size_non_adaptive(self):
        """Test getting current batch size without adaptive batching."""
        helper = BatchOptimizationHelper(adaptive_batching=False, initial_batch_size=5)

        assert helper._get_current_batch_size(10) == 5
        assert helper._get_current_batch_size(3) == 3  # Limited by total items

    def test_get_current_batch_size_adaptive(self):
        """Test getting current batch size with adaptive batching."""
        helper = BatchOptimizationHelper(adaptive_batching=True, initial_batch_size=5)

        # Should use adaptive sizer
        batch_size = helper._get_current_batch_size(10)
        assert batch_size <= 10
        assert batch_size >= 1

    def test_reset_stats(self):
        """Test resetting statistics."""
        helper = BatchOptimizationHelper()

        # Modify stats
        helper.stats.total_configurations = 5
        helper.stats.successful_evaluations = 3

        # Reset
        helper.reset_stats()

        # Should be back to defaults
        assert helper.stats.total_configurations == 0
        assert helper.stats.successful_evaluations == 0

    def test_evaluate_configurations_input_validation(self):
        """Invalid inputs should raise informative errors."""
        import asyncio

        async def run_invalid_tests():
            helper = BatchOptimizationHelper()

            with pytest.raises(TypeError):
                await helper.evaluate_configurations_batch(
                    configurations="not-a-list",  # type: ignore[arg-type]
                    func=lambda *_args, **_kwargs: None,
                    dataset=self.dataset,
                    invoker=self.invoker,
                    evaluator=self.evaluator,
                )

            with pytest.raises(TypeError):
                await helper.evaluate_configurations_batch(
                    configurations=[{"param": 1}],
                    func="not-callable",  # type: ignore[arg-type]
                    dataset=self.dataset,
                    invoker=self.invoker,
                    evaluator=self.evaluator,
                )

            empty_dataset = Dataset(examples=[], name="empty")
            with pytest.raises(ValueError):
                await helper.evaluate_configurations_batch(
                    configurations=[{"param": 1}],
                    func=lambda *_args, **_kwargs: None,
                    dataset=empty_dataset,
                    invoker=self.invoker,
                    evaluator=self.evaluator,
                )

            class BadInvoker:
                pass

            with pytest.raises(TypeError):
                await helper.evaluate_configurations_batch(
                    configurations=[{"param": 1}],
                    func=lambda *_args, **_kwargs: None,
                    dataset=self.dataset,
                    invoker=BadInvoker(),
                    evaluator=self.evaluator,
                )

            with pytest.raises(TypeError):
                await helper.evaluate_configurations_batch(
                    configurations=[{"param": 1}],
                    func=lambda *_args, **_kwargs: None,
                    dataset=self.dataset,
                    invoker=self.invoker,
                    evaluator=object(),  # type: ignore[arg-type]
                )

            with pytest.raises(TypeError):
                await helper.evaluate_configurations_batch(
                    configurations=[{"param": 1}],
                    func=lambda *_args, **_kwargs: None,
                    dataset=self.dataset,
                    invoker=self.invoker,
                    evaluator=self.evaluator,
                    progress_callback="not-callable",  # type: ignore[arg-type]
                )

        asyncio.run(run_invalid_tests())


class TestParallelConfigEvaluation:
    """Test suite for parallel configuration evaluation."""

    def setup_method(self):
        """Set up test data."""
        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
            ],
            name="test_dataset",
        )

        self.configurations = [
            {"param1": 1},
            {"param1": 2},
            {"param1": 3},
        ]

        self.invoker = MockInvoker()
        self.evaluator = MockEvaluator()

    def test_parallel_config_evaluation_success(self):
        """Test successful parallel configuration evaluation."""
        import asyncio

        async def run_test():
            def mock_func(value, config):
                return f"processed_{value}_{config.get('param1', 0)}"

            results = await parallel_config_evaluation(
                configurations=self.configurations,
                func=mock_func,
                dataset=self.dataset,
                invoker=self.invoker,
                evaluator=self.evaluator,
                max_parallel=2,
                batch_size=1,
            )

            # Check results
            assert len(results) == 3
            assert all(r is not None for r in results)
            assert all(isinstance(r, EvaluationResult) for r in results)

            # All should be successful
            for result in results:
                assert result.successful_invocations == 2  # Dataset has 2 examples

        asyncio.run(run_test())

    def test_parallel_config_evaluation_empty_list(self):
        """Test parallel evaluation with empty configuration list."""
        import asyncio

        async def run_test():
            def mock_func(value, config):
                return f"processed_{value}"

            results = await parallel_config_evaluation(
                configurations=[],
                func=mock_func,
                dataset=self.dataset,
                invoker=self.invoker,
                evaluator=self.evaluator,
            )

            assert len(results) == 0

        asyncio.run(run_test())


class TestProgressCallback:
    """Test suite for progress callback utilities."""

    def test_create_batch_progress_callback(self):
        """Test creating batch progress callback."""
        callback = create_batch_progress_callback(log_interval=5)

        # Should be a callable
        assert callable(callback)

        # Should not raise errors when called
        callback(5, 10)
        callback(10, 10)

    def test_progress_callback_edge_cases(self):
        """Test progress callback with edge cases."""
        callback = create_batch_progress_callback()

        # Test with zero total
        callback(0, 0)

        # Test with partial progress
        callback(3, 10)

        # Test completion
        callback(10, 10)
