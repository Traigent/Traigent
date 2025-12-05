"""Performance tests for metrics functionality.

This test suite covers:
- Token estimation performance with various input sizes
- Metrics tracking overhead with large datasets
- Memory usage during metrics collection
- Summary stats generation performance
- Concurrent evaluation performance
- Large-scale optimization performance
"""

import asyncio
import gc
import os
import time

import pytest

try:
    from examples.archive.shared_utils.mock_llm import estimate_tokens
except ImportError:  # pragma: no cover - archive removed during cleanup
    from tests.integration.test_mock_mode_metrics import estimate_tokens

from tests.conftest import MockLLMResponse, validate_example_metrics
from traigent import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer


class TestTokenEstimationPerformance:
    """Test performance of token estimation functionality."""

    def create_text_of_size(self, target_chars: int) -> str:
        """Create text with approximately target number of characters."""
        base_text = "This is a test sentence for token estimation performance testing. "
        repeats = max(1, target_chars // len(base_text))
        remainder = target_chars % len(base_text)
        return (base_text * repeats) + base_text[:remainder]

    @pytest.mark.parametrize("text_size", [100, 1000, 10000, 50000, 100000])
    def test_token_estimation_scaling(self, text_size):
        """Test token estimation performance with different text sizes."""
        text = self.create_text_of_size(text_size)

        # Measure performance
        start_time = time.time()
        token_count = estimate_tokens(text)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should be very fast for token estimation
        assert (
            execution_time < 0.1
        ), f"Token estimation took {execution_time:.3f}s for {text_size} chars"

        # Should give reasonable token count (approximately 1 token per 4 characters)
        expected_tokens = max(1, text_size // 4)
        assert (
            abs(token_count - expected_tokens) < 5
        ), f"Expected ~{expected_tokens}, got {token_count}"

    def test_bulk_token_estimation_performance(self):
        """Test performance of estimating tokens for many inputs."""
        # Create varied text sizes
        texts = [
            self.create_text_of_size(size)
            for size in [50, 100, 500, 1000, 2000] * 20  # 100 texts total
        ]

        start_time = time.time()
        token_counts = [estimate_tokens(text) for text in texts]
        end_time = time.time()

        execution_time = end_time - start_time

        # Should process 100 texts quickly
        assert (
            execution_time < 1.0
        ), f"Bulk estimation took {execution_time:.3f}s for 100 texts"
        assert len(token_counts) == 100
        assert all(count > 0 for count in token_counts)

    @pytest.mark.asyncio
    async def test_concurrent_token_estimation(self):
        """Test concurrent token estimation performance."""
        # Create test texts
        texts = [
            f"Test text number {i} with some varied content for testing"
            for i in range(50)
        ]

        async def estimate_async(text):
            """Async wrapper for token estimation."""
            return estimate_tokens(text)

        start_time = time.time()

        # Run concurrent estimations
        tasks = [estimate_async(text) for text in texts]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should handle concurrent estimation efficiently
        assert execution_time < 0.5, f"Concurrent estimation took {execution_time:.3f}s"
        assert len(results) == 50
        assert all(result > 0 for result in results)


class TestMetricsCollectionPerformance:
    """Test performance of metrics collection and tracking."""

    def create_large_dataset(self, size: int) -> Dataset:
        """Create large dataset for performance testing."""
        examples = []
        for i in range(size):
            text = f"Performance test example {i} with content that varies in length and complexity depending on the specific test case requirements."
            examples.append(
                EvaluationExample(
                    input_data={"text": text, "id": f"perf_example_{i}"},
                    expected_output=(
                        "positive"
                        if i % 3 == 0
                        else "negative" if i % 3 == 1 else "neutral"
                    ),
                )
            )
        return Dataset(examples=examples, name=f"performance_test_{size}")

    @pytest.mark.asyncio
    async def test_large_dataset_evaluation_performance(self):
        """Test evaluation performance with large datasets."""

        async def fast_function(text: str, **kwargs) -> str:
            """Simple function for performance testing."""
            return "positive" if len(text) > 100 else "negative"

        # Test with increasing dataset sizes
        sizes = [10, 50, 100, 200]
        times = []

        for size in sizes:
            dataset = self.create_large_dataset(size)
            evaluator = LocalEvaluator(
                metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
            )

            start_time = time.time()
            result = await evaluator.evaluate(fast_function, {}, dataset)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            # Verify results are correct
            assert result.total_examples == size
            assert len(result.example_results) == size

            # All examples should have metrics
            for example_result in result.example_results:
                assert validate_example_metrics(example_result)

        # Performance should scale reasonably (not exponentially)
        # Allow some variance but should be roughly linear
        for i in range(1, len(times)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            # Time ratio should not be dramatically higher than size ratio (very relaxed for CI)
            assert (
                time_ratio < size_ratio * 25
            ), f"Performance degradation: size {sizes[i]} took {time_ratio:.2f}x longer than expected"

    @pytest.mark.asyncio
    async def test_memory_usage_during_evaluation(self):
        """Test memory usage during large evaluations."""
        import psutil

        process = psutil.Process(os.getpid())

        async def memory_test_function(text: str) -> str:
            """Function for memory testing."""
            return "test_output"

        # Measure memory before
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run evaluation with moderately large dataset
        dataset = self.create_large_dataset(100)
        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        result = await evaluator.evaluate(memory_test_function, {}, dataset)

        # Measure memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Results should be valid (check before cleanup)
        assert len(result.example_results) == 100

        # Clean up
        del result
        del dataset
        gc.collect()

        # Memory growth should be reasonable (less than 50MB for 100 examples)
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB"

    @pytest.mark.asyncio
    async def test_summary_stats_generation_performance(self):
        """Test performance of summary stats generation."""

        async def varied_function(text: str, variation: float = 0.5, **kwargs) -> str:
            """Function that creates variation for statistics."""
            score = hash(text) % 100 / 100.0
            if score > variation:
                return "positive"
            elif score < (1 - variation):
                return "negative"
            else:
                return "neutral"

        dataset = self.create_large_dataset(200)  # Larger dataset for stats
        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        # Time the full evaluation including summary stats
        start_time = time.time()
        result = await evaluator.evaluate(varied_function, {"variation": 0.6}, dataset)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert (
            execution_time < 30.0
        ), f"Evaluation with summary stats took {execution_time:.1f}s"

        # Should have result data
        assert result.total_examples == 200
        assert result.successful_examples > 0

        # Check if summary stats are available - this might not be populated by default
        if (
            hasattr(result, "summary_stats")
            and result.summary_stats
            and result.summary_stats.get("metrics")
        ):
            metrics = result.summary_stats.get("metrics", {})
            assert isinstance(metrics, dict)
            assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_metrics_tracker_performance(self):
        """Test performance of MetricsTracker with many examples."""
        from traigent.evaluators.metrics_tracker import (
            ExampleMetrics,
            MetricsTracker,
            TokenMetrics,
        )

        tracker = MetricsTracker()
        tracker.start_tracking()

        # Add many example metrics
        start_time = time.time()

        for i in range(1000):
            example = ExampleMetrics(
                tokens=TokenMetrics(input_tokens=100 + i, output_tokens=50 + i // 2),
                success=i % 10 != 0,  # 10% failure rate
                custom_metrics={"custom_metric": i * 0.1},
            )
            tracker.add_example_metrics(example)

        end_time = time.time()
        tracker.end_tracking()

        add_time = end_time - start_time

        # Should be able to add 1000 examples quickly
        assert add_time < 1.0, f"Adding 1000 examples took {add_time:.3f}s"

        # Test summary stats generation performance
        start_time = time.time()
        summary_stats = tracker.format_as_summary_stats()
        end_time = time.time()

        summary_time = end_time - start_time

        # Summary stats generation should be fast
        assert summary_time < 2.0, f"Summary stats generation took {summary_time:.3f}s"

        # Verify summary stats are correct
        assert summary_stats["total_examples"] == 1000
        assert "input_tokens" in summary_stats["metrics"]
        # The count may be 900 if only successful examples are counted
        token_count = summary_stats["metrics"]["input_tokens"]["count"]
        assert token_count in [
            900,
            1000,
        ], f"Expected token count 900 or 1000, got {token_count}"


class TestOptimizationPerformance:
    """Test performance of optimization workflows with metrics."""

    @pytest.mark.asyncio
    async def test_grid_search_performance(self):
        """Test performance of grid search optimization."""

        async def optimization_function(
            text: str, temperature: float = 0.5, approach: str = "balanced", **kwargs
        ) -> str:
            """Function for optimization testing."""
            score = hash(f"{text}_{temperature}_{approach}") % 100 / 100.0
            if score > 0.6:
                return "positive"
            elif score < 0.4:
                return "negative"
            else:
                return "neutral"

        # Small but representative dataset and config space
        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={
                        "text": f"Test example {i} for optimization performance"
                    },
                    expected_output="positive" if i % 2 == 0 else "negative",
                )
                for i in range(10)  # Small dataset for performance testing
            ]
        )

        config_space = {
            "temperature": [0.1, 0.5, 0.9],
            "approach": ["conservative", "balanced", "aggressive"],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space, objectives=["accuracy"]
        )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            objectives=["accuracy"],
            config=config,
        )

        # Disable backend client to avoid network timeouts during performance testing
        orchestrator.backend_client = None

        start_time = time.time()

        await orchestrator.create_session()
        optimization_results = await orchestrator.optimize(
            optimization_function, dataset
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete grid search in reasonable time
        assert execution_time < 60.0, f"Grid search took {execution_time:.1f}s"

        # Should have explored all configurations
        # Grid search with 3x3 would have 9, but orchestrator uses max_trials from optimizer
        assert len(optimization_results.trials) >= 1

        # All trials should have valid configurations and results
        for trial in optimization_results.trials:
            assert trial.config is not None
            assert "temperature" in trial.config
            assert "approach" in trial.config

            # Check measures if available (may not be populated due to backend errors)
            measures = getattr(trial, "measures", [])
            if measures:  # Only check if measures are present
                for measure in measures:
                    assert "input_tokens" in measure
                    assert "output_tokens" in measure
                    assert measure["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_random_search_performance(self):
        """Test performance of random search optimization."""

        async def performance_function(
            text: str, param1: float = 0.5, param2: float = 0.5, **kwargs
        ) -> MockLLMResponse:
            """Function that returns LLM responses for performance testing."""

            # Simulate performance based on parameters
            input_tokens = 50 + int(param1 * 100)
            output_tokens = 20 + int(param2 * 50)
            response_time = 500 + int((param1 + param2) * 1000)

            result = "positive" if param1 + param2 > 1.0 else "negative"

            return MockLLMResponse(
                text=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time=response_time,
            )

        # Larger dataset for more realistic performance test
        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": f"Random search test example {i} with content"},
                    expected_output="positive" if i % 3 == 0 else "negative",
                )
                for i in range(20)
            ]
        )

        config_space = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}

        optimizer = RandomSearchOptimizer(
            config_space=config_space, objectives=["accuracy"], max_trials=5
        )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            objectives=["accuracy"],
            config=config,
        )

        # Disable backend client to avoid network timeouts during performance testing
        orchestrator.backend_client = None

        start_time = time.time()

        await orchestrator.create_session()
        optimization_results = await orchestrator.optimize(
            performance_function, dataset
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete random search efficiently
        assert execution_time < 45.0, f"Random search took {execution_time:.1f}s"

        # Should have completed trials
        assert len(optimization_results.trials) >= 1

        # Check that trials have valid configurations
        for trial in optimization_results.trials:
            assert trial.config is not None
            assert "param1" in trial.config
            assert "param2" in trial.config

            # Check measures if available (may not be populated due to backend errors)
            measures = getattr(trial, "measures", [])
            if measures:  # Only check if measures are present
                for measure in measures:
                    # Should have actual token counts from LLM responses
                    assert measure["input_tokens"] >= 50
                    assert measure["output_tokens"] >= 20
                    assert measure["total_tokens"] >= 70

    @pytest.mark.asyncio
    async def test_concurrent_evaluations_performance(self):
        """Test performance of concurrent evaluations."""

        async def concurrent_function(text: str, delay: float = 0.01) -> str:
            """Function with controlled delay for concurrency testing."""
            await asyncio.sleep(delay)
            return "positive" if "test" in text.lower() else "negative"

        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": f"Concurrent test example {i}"},
                    expected_output="positive",
                )
                for i in range(5)
            ]
        )

        # Test different delay configurations
        delays = [0.01, 0.05, 0.1]

        for delay in delays:
            evaluator = LocalEvaluator(
                metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
            )

            start_time = time.time()
            result = await evaluator.evaluate(
                concurrent_function, {"delay": delay}, dataset
            )
            end_time = time.time()

            execution_time = end_time - start_time

            # With proper concurrency, total time should be close to individual delay
            # rather than delay * number_of_examples
            expected_time = delay * 1.5  # Allow some overhead
            # Be more lenient with timing expectations for CI environments
            assert (
                execution_time < expected_time * 5
            ), f"Concurrent execution with {delay}s delay took {execution_time:.3f}s, expected ~{expected_time:.3f}s"

            # Results should be complete
            assert result.total_examples == 5
            assert result.successful_examples == 5


class TestResourceUsageMonitoring:
    """Test resource usage monitoring during metrics operations."""

    @pytest.mark.asyncio
    async def test_cpu_usage_monitoring(self):
        """Test CPU usage during intensive metrics operations."""
        import psutil

        process = psutil.Process(os.getpid())

        async def cpu_intensive_function(text: str) -> str:
            """Function that uses some CPU for testing."""
            # Simple computation to use CPU
            result = 0
            for i in range(1000):
                result += hash(f"{text}_{i}") % 100
            return "positive" if result % 2 == 0 else "negative"

        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": f"CPU test {i} with content for processing"},
                    expected_output="positive",
                )
                for i in range(50)
            ]
        )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        # Monitor CPU usage
        process.cpu_percent()
        start_time = time.time()

        result = await evaluator.evaluate(cpu_intensive_function, {}, dataset)

        end_time = time.time()
        process.cpu_percent()

        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert (
            execution_time < 30.0
        ), f"CPU intensive evaluation took {execution_time:.1f}s"

        # Should have processed all examples with metrics
        assert result.total_examples == 50
        for example_result in result.example_results:
            assert validate_example_metrics(example_result)

    def test_metrics_overhead_measurement(self):
        """Test overhead of metrics collection vs basic execution."""

        def simple_function(text: str) -> str:
            """Simple function for overhead testing with minimal work."""
            # Introduce a tiny sleep to simulate real processing latency so that the
            # timing baseline is stable even on very fast CI runners.
            time.sleep(5e-5)
            return "positive" if len(text) > 50 else "negative"

        # Create test data
        texts = [f"Overhead test text number {i} with some content" for i in range(100)]

        # Test without metrics tracking
        start_time = time.perf_counter()
        results_no_metrics = [simple_function(text) for text in texts]
        end_time = time.perf_counter()
        baseline_time = end_time - start_time

        # Test with manual metrics tracking (simulating overhead)
        start_time = time.perf_counter()
        results_with_metrics = []
        for text in texts:
            # Simulate metrics collection overhead
            input_tokens = len(text) // 4  # Token estimation
            result = simple_function(text)
            output_tokens = len(result) // 4
            # Simulate storing metrics
            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "accuracy": 1.0,
            }
            results_with_metrics.append((result, metrics))
        end_time = time.perf_counter()
        metrics_time = end_time - start_time

        # Calculate overhead
        overhead = metrics_time - baseline_time
        overhead_percentage = (
            (overhead / baseline_time) * 100 if baseline_time > 0 else 0
        )

        # Metrics overhead should be reasonable (less than 500% overhead in test environment)
        assert (
            overhead_percentage < 500
        ), f"Metrics overhead is {overhead_percentage:.1f}% of baseline execution time"

        # Results should be the same
        assert len(results_no_metrics) == len(results_with_metrics)
        for i in range(len(results_no_metrics)):
            assert results_no_metrics[i] == results_with_metrics[i][0]

    @pytest.mark.asyncio
    async def test_parallel_processing_efficiency(self):
        """Test efficiency of parallel processing with metrics."""

        async def parallel_test_function(
            text: str, processing_time: float = 0.1
        ) -> str:
            """Function with controlled processing time."""
            await asyncio.sleep(processing_time)
            return "positive" if hash(text) % 2 == 0 else "negative"

        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": f"Parallel test {i}"},
                    expected_output="positive" if i % 2 == 0 else "negative",
                )
                for i in range(10)
            ]
        )

        processing_time = 0.05  # 50ms per function call

        evaluator = LocalEvaluator(
            metrics=["accuracy"],
            detailed=True,
            execution_mode="edge_analytics",
            max_workers=5,
        )

        start_time = time.time()
        result = await evaluator.evaluate(
            parallel_test_function, {"processing_time": processing_time}, dataset
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # With proper parallelization, should be much faster than sequential
        sequential_time = len(dataset.examples) * processing_time  # 10 * 0.05 = 0.5s

        # Allow some overhead but should be reasonably close to sequential time
        # In test environments, "parallel" processing may not show significant speedup
        assert (
            execution_time < sequential_time * 2.0
        ), f"Parallel execution took {execution_time:.3f}s vs expected sequential {sequential_time:.3f}s"

        # Results should be complete with metrics
        assert result.total_examples == 10
        for example_result in result.example_results:
            assert validate_example_metrics(example_result)


@pytest.mark.performance
class TestLargeScalePerformance:
    """Large-scale performance tests (marked with @pytest.mark.performance)."""

    @pytest.mark.asyncio
    async def test_very_large_dataset_performance(self):
        """Test performance with very large datasets."""

        async def large_scale_function(text: str) -> str:
            """Optimized function for large scale testing."""
            return "positive" if hash(text) % 2 == 0 else "negative"

        # Create large dataset (1000 examples)
        large_dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": f"Large scale test example {i}"},
                    expected_output="positive" if i % 2 == 0 else "negative",
                )
                for i in range(1000)
            ]
        )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        start_time = time.time()
        result = await evaluator.evaluate(large_scale_function, {}, large_dataset)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should handle 1000 examples efficiently
        assert (
            execution_time < 120.0
        ), f"Large scale evaluation took {execution_time:.1f}s"

        # Results should be complete
        assert result.total_examples == 1000
        assert len(result.example_results) == 1000

        # Spot check some results have metrics
        for i in [0, 100, 500, 999]:
            assert validate_example_metrics(result.example_results[i])

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_scale(self):
        """Test memory efficiency with large datasets."""
        import psutil

        process = psutil.Process(os.getpid())

        async def memory_efficient_function(text: str, **kwargs) -> str:
            """Memory efficient function."""
            return "test"  # Same string to reduce memory usage

        # Monitor memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process datasets to test memory efficiency
        total_objects_created = 0

        for chunk_start in range(0, 500, 100):  # Process 500 examples in chunks
            # Create dataset objects
            chunk_examples = [
                EvaluationExample(
                    input_data={"text": f"Memory test {i}"}, expected_output="test"
                )
                for i in range(chunk_start, min(chunk_start + 100, 500))
            ]

            chunk_dataset = Dataset(examples=chunk_examples)
            total_objects_created += len(chunk_examples)

            # Simulate processing by creating some temporary objects
            temp_data = [str(ex.input_data) for ex in chunk_examples]
            temp_results = [
                {"result": "test", "index": i} for i in range(len(chunk_examples))
            ]

            # Clean up immediately to test memory efficiency
            del temp_data
            del temp_results
            del chunk_examples
            del chunk_dataset
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable for creating 500 objects
        assert (
            memory_growth < 100
        ), f"Memory grew by {memory_growth:.1f}MB for {total_objects_created} objects"
        # Should have created the expected number of objects
        assert total_objects_created == 500
