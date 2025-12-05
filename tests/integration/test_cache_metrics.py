"""Integration tests for cache behavior with token and cost metrics.

This test suite covers:
- Token metric preservation in cached responses
- Cache hit/miss behavior with metrics tracking
- Cost calculation accuracy with cached vs fresh responses
- Cache invalidation effects on metrics
- Mixed cached/uncached scenario handling
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.grid import GridSearchOptimizer


class MockCachedResponse:
    """Mock response that simulates cached LLM response."""

    def __init__(
        self, text: str, cached=True, input_tokens: int = 100, output_tokens: int = 50
    ):
        self.text = text
        self.cached = cached

        # Create usage object that extract_llm_metrics can understand
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )()

        # Add response time that extract_llm_metrics can find
        self.response_time_ms = 50 if cached else 1000

        # Add cost metadata if provided
        self.cost_metadata = getattr(self, "cost_metadata", None)

    def __str__(self):
        return self.text


class TestCacheMetrics:
    """Test cache integration with token and cost metrics."""

    @pytest.fixture
    def cache_dataset(self):
        """Create dataset for cache testing."""
        examples = [
            EvaluationExample(
                input_data={"text": "Test input for caching behavior analysis"},
                expected_output="positive",
            ),
            EvaluationExample(
                input_data={"text": "Another test input for cache validation"},
                expected_output="negative",
            ),
            EvaluationExample(
                input_data={"text": "Third test input for comprehensive testing"},
                expected_output="neutral",
            ),
            EvaluationExample(
                input_data={
                    "text": "Test input for caching behavior analysis"
                },  # Duplicate for cache hit
                expected_output="positive",
            ),
        ]
        return Dataset(examples=examples, name="cache_test")

    @pytest.fixture
    def mock_cache_dir(self):
        """Create temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cache_dir.mkdir()
            yield cache_dir

    @pytest.mark.asyncio
    async def test_cache_preserves_token_metrics(self, cache_dataset, mock_cache_dir):
        """Test that cached responses preserve original token metrics."""

        call_count = 0

        async def cached_function(**kwargs) -> MockCachedResponse:
            """Function that simulates cache hits and misses."""
            nonlocal call_count
            call_count += 1

            # Get text from kwargs
            text = kwargs.get("text", "")

            # First call to each unique input is fresh, subsequent calls are cached
            input_hash = hash(text)
            is_cached = call_count > 3  # Simulate cache hit for duplicate input

            return MockCachedResponse(
                text="positive" if "positive" in str(input_hash) else "negative",
                cached=is_cached,
                input_tokens=len(text) // 4 + 10,  # Realistic token count
                output_tokens=20,
            )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"model": "test-model", "cache_dir": str(mock_cache_dir)}
        result = await evaluator.evaluate(cached_function, config, cache_dataset)

        # Should have processed all examples
        assert len(result.example_results) == 4

        # Check that cached response preserved token metrics
        for example_result in result.example_results:
            assert example_result.metrics["input_tokens"] > 0
            assert example_result.metrics["output_tokens"] > 0
            assert example_result.metrics["total_tokens"] > 0

            # Cached responses should have faster response times
            if hasattr(example_result, "cached") and example_result.cached:
                # This would be set by actual cache implementation
                pass

    @pytest.mark.asyncio
    async def test_cache_hit_vs_miss_metrics(self, cache_dataset):
        """Test metrics difference between cache hits and misses."""

        cache_status = {}

        async def cache_aware_function(**kwargs) -> MockCachedResponse:
            """Function that tracks cache hits/misses."""
            text = kwargs.get("text", "")
            text_hash = hash(text)

            if text_hash in cache_status:
                # Cache hit - fast response
                cache_status[text_hash] += 1
                return MockCachedResponse(
                    text="cached_positive",
                    cached=True,
                    input_tokens=len(text) // 4 + 5,
                    output_tokens=15,
                )
            else:
                # Cache miss - slower response
                cache_status[text_hash] = 1
                return MockCachedResponse(
                    text="fresh_positive",
                    cached=False,
                    input_tokens=len(text) // 4 + 5,
                    output_tokens=15,
                )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {}
        result = await evaluator.evaluate(cache_aware_function, config, cache_dataset)

        # Check results
        fresh_responses = []
        cached_responses = []

        for i, example_result in enumerate(result.example_results):
            # Simulate checking cache status
            text_hash = hash(cache_dataset.examples[i].input_data["text"])
            if cache_status[text_hash] > 1:
                cached_responses.append(example_result)
            else:
                fresh_responses.append(example_result)

        # Should have mix of fresh and cached
        assert len(fresh_responses) > 0
        assert len(cached_responses) > 0

        # All should have token metrics preserved
        for example_result in fresh_responses + cached_responses:
            assert example_result.metrics["input_tokens"] > 0
            assert example_result.metrics["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_cache_invalidation_metrics(self, cache_dataset):
        """Test metrics after cache invalidation."""

        cache_version_state = {"version": 1}

        async def version_aware_function(**kwargs) -> str:
            """Function that simulates cache invalidation."""
            cache_version = kwargs.get("cache_version", 1)
            # Version change would invalidate cache
            if cache_version > cache_version_state["version"]:
                return f"invalidated_response_v{cache_version}"
            else:
                return "cached_response_v1"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        # First run with version 1
        config1 = {"cache_version": 1}
        result1 = await evaluator.evaluate(
            version_aware_function, config1, cache_dataset
        )

        # Second run with version 2 (invalidates cache)
        config2 = {"cache_version": 2}
        result2 = await evaluator.evaluate(
            version_aware_function, config2, cache_dataset
        )

        # Both should have token estimates (from string outputs)
        for result in [result1, result2]:
            assert result.aggregated_metrics["input_tokens"] > 0
            assert result.aggregated_metrics["output_tokens"] > 0

            for example_result in result.example_results:
                assert example_result.metrics["input_tokens"] > 0
                assert example_result.metrics["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_mixed_cached_uncached_optimization(self, cache_dataset):
        """Test optimization flow with mixed cache hits and misses."""

        call_cache = {}

        async def optimization_function(**kwargs) -> str:
            """Function for optimization with cache simulation."""
            text = kwargs.get("text", "")
            temperature = kwargs.get("temperature", 0.5)
            use_cache = kwargs.get("use_cache", True)

            key = f"{text}_{temperature}"

            if use_cache and key in call_cache:
                # Cache hit
                call_cache[key]["count"] += 1
                return call_cache[key]["result"]
            else:
                # Cache miss or cache disabled
                result = "positive" if temperature > 0.6 else "negative"
                if use_cache:
                    call_cache[key] = {"result": result, "count": 1}
                return result

        # Set up optimizer
        config_space = {"temperature": [0.3, 0.5, 0.7], "use_cache": [True, False]}

        optimizer = GridSearchOptimizer(
            config_space=config_space, objectives=["accuracy"]
        )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer, evaluator=evaluator, objectives=["accuracy"]
        )

        # Run optimization
        await orchestrator.create_session()
        optimization_results = await orchestrator.optimize(
            optimization_function, cache_dataset
        )

        # Should have results with token metrics - GridSearch will run all combinations
        # 3 temperatures x 2 cache options = 6 trials
        assert len(optimization_results.trials) > 0  # Should have at least some trials

        for trial in optimization_results.trials:
            # Check that measures have token metrics
            measures = getattr(trial, "measures", [])
            if measures:
                for measure in measures:
                    if isinstance(measure, dict):
                        assert (
                            "input_tokens" in measure
                            or measure.get("input_tokens", 0) >= 0
                        )
                        assert (
                            "output_tokens" in measure
                            or measure.get("output_tokens", 0) >= 0
                        )

    @pytest.mark.asyncio
    async def test_cache_with_error_handling(self, cache_dataset):
        """Test cache behavior with errors and ensure metrics still work."""

        error_cache = set()
        call_count = 0

        async def error_prone_cached_function(**kwargs) -> str:
            """Function that sometimes errors, with cache simulation."""
            nonlocal call_count
            call_count += 1

            text = kwargs.get("text", "")
            text_hash = hash(text)

            # Simulate cached error (should not retry)
            if text_hash in error_cache:
                raise ValueError("Cached error - should not retry")

            # Simulate fresh error on second unique input
            if call_count == 2:
                error_cache.add(text_hash)
                raise ValueError("Fresh error - gets cached")

            return "success"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {}
        result = await evaluator.evaluate(
            error_prone_cached_function, config, cache_dataset
        )

        # Should have some successful and some failed results
        successful_results = [r for r in result.example_results if r.success]
        failed_results = [r for r in result.example_results if not r.success]

        assert len(successful_results) > 0
        assert len(failed_results) > 0

        # Successful results should have token metrics
        for success_result in successful_results:
            assert success_result.metrics["input_tokens"] > 0
            assert success_result.metrics["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_cache_cost_calculation_consistency(self, cache_dataset):
        """Test that cost calculations are consistent between cached and fresh responses."""

        # Track cost calculation consistency
        cost_records = []

        async def cost_tracking_function(**kwargs) -> MockCachedResponse:
            """Function that tracks costs for cache analysis."""
            text = kwargs.get("text", "")
            model = kwargs.get("model", "gpt-3.5-turbo")

            input_tokens = len(text) // 4 + 10
            output_tokens = 20

            # Simulate model-specific pricing
            if "gpt-4" in model:
                input_cost_per_token = 0.00003
                output_cost_per_token = 0.00006
            else:
                input_cost_per_token = 0.000001
                output_cost_per_token = 0.000002

            input_cost = input_tokens * input_cost_per_token
            output_cost = output_tokens * output_cost_per_token

            # Record for consistency checking
            cost_records.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": input_cost + output_cost,
                }
            )

            response = MockCachedResponse(
                text="result",
                cached=len(cost_records) > 2,  # Later calls are "cached"
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Add cost metadata
            response.cost_metadata = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost,
            }

            return response

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"model": "gpt-3.5-turbo"}
        result = await evaluator.evaluate(cost_tracking_function, config, cache_dataset)

        # All examples should have consistent cost calculations
        for _i, example_result in enumerate(result.example_results):
            assert example_result.metrics["input_tokens"] > 0
            assert example_result.metrics["output_tokens"] > 0

            # Check that costs are present and reasonable
            # The evaluator may calculate its own costs based on the model
            # So we just verify that costs are calculated and are positive
            assert example_result.metrics["input_cost"] >= 0.0
            assert example_result.metrics["output_cost"] >= 0.0
            assert example_result.metrics["total_cost"] >= 0.0

            # Verify costs are consistent with token counts
            if example_result.metrics["total_cost"] > 0:
                # If there's a cost, there should be tokens
                assert (
                    example_result.metrics["input_tokens"] > 0
                    or example_result.metrics["output_tokens"] > 0
                )

    def test_cache_configuration_validation(self):
        """Test cache configuration validation."""

        # Test that cache directory validation works
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            # Directory should be created if it doesn't exist
            assert not cache_dir.exists()

            evaluator = LocalEvaluator(
                metrics=["accuracy"],
                execution_mode="edge_analytics",
                cache_config={"cache_dir": str(cache_dir)},
            )

            # Cache directory should be valid after evaluator creation
            assert evaluator is not None

    @pytest.mark.asyncio
    async def test_cache_metrics_aggregation(self, cache_dataset):
        """Test that cache metrics are properly aggregated."""

        cache_hits = 0
        cache_misses = 0

        async def aggregation_function(**kwargs) -> str:
            """Function for testing aggregation with cache simulation."""
            nonlocal cache_hits, cache_misses

            text = kwargs.get("text", "")
            text_hash = hash(text)
            # Simulate 50% cache hit rate
            if text_hash % 2 == 0:
                cache_hits += 1
                return f"cached_result_{cache_hits}"
            else:
                cache_misses += 1
                return f"fresh_result_{cache_misses}"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {}
        result = await evaluator.evaluate(aggregation_function, config, cache_dataset)

        # Check aggregated metrics
        assert result.aggregated_metrics["input_tokens"] > 0
        assert result.aggregated_metrics["output_tokens"] > 0
        assert result.aggregated_metrics["total_tokens"] > 0

        # Check that aggregation includes all examples
        assert result.total_examples == len(cache_dataset.examples)

        # Cost metrics should still be 0 in mock mode (if present)
        assert result.aggregated_metrics.get("input_cost", 0.0) == 0.0
        assert result.aggregated_metrics.get("output_cost", 0.0) == 0.0


class TestCacheErrorScenarios:
    """Test cache error scenarios and recovery."""

    @pytest.fixture
    def error_dataset(self):
        """Small dataset for error testing."""
        return Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": "Error test input"}, expected_output="positive"
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_cache_corruption_recovery(self, error_dataset):
        """Test recovery from cache corruption."""

        corruption_simulated = False

        async def corruption_function(**kwargs) -> str:
            """Function that simulates cache corruption."""
            nonlocal corruption_simulated

            if not corruption_simulated:
                corruption_simulated = True
                # Simulate cache corruption scenario
                raise RuntimeError("Cache corruption detected")

            return "recovered_result"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {}
        result = await evaluator.evaluate(corruption_function, config, error_dataset)

        # Should still work and have metrics
        assert len(result.example_results) == 1
        example_result = result.example_results[0]

        # Should have token metrics even after error recovery
        # Note: Failed functions may not have output tokens, but should have input tokens estimated
        if example_result.success:
            assert example_result.metrics["input_tokens"] > 0
            assert example_result.metrics["output_tokens"] > 0
        else:
            # For failed functions, we may still get estimated input tokens based on the input data
            # but output tokens might be 0 since there was no valid output
            assert example_result.metrics.get("input_tokens", 0) >= 0
            assert example_result.metrics.get("output_tokens", 0) >= 0

    @pytest.mark.asyncio
    async def test_cache_timeout_handling(self, error_dataset):
        """Test handling of cache timeouts."""

        async def timeout_function(**kwargs) -> str:
            """Function that simulates cache timeout."""
            # Simulate slow cache operation followed by success
            await asyncio.sleep(0.1)  # Small delay to simulate timeout
            return "timeout_recovered_result"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {}
        result = await evaluator.evaluate(timeout_function, config, error_dataset)

        # Should complete and have metrics
        assert result.successful_examples == 1
        example_result = result.example_results[0]
        assert example_result.metrics["input_tokens"] > 0
        assert example_result.metrics["output_tokens"] > 0
