"""End-to-end integration tests for complete metrics flow.

This test suite covers:
- Complete flow from function execution to measures array
- Integration between LocalEvaluator, MetricsTracker, and Orchestrator
- Metrics flow across different execution modes
- Summary stats generation and validation
- Backend submission format validation
- Real-world optimization scenarios with comprehensive metrics tracking
"""

import os

import pytest

from traigent import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer


class MockLLMResponse:
    """Mock LLM response with comprehensive metadata."""

    def __init__(
        self,
        text: str,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cost: float = 0.003,
        response_time: float = 1500,
    ):
        self.text = text
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )()
        self.response_time_ms = response_time
        self.cost_metadata = {
            "input_cost": cost * 0.4,
            "output_cost": cost * 0.6,
            "total_cost": cost,
        }

    def __str__(self):
        return self.text


class TestEndToEndMetricsFlow:
    """Test complete end-to-end metrics flow."""

    @pytest.fixture
    def comprehensive_dataset(self):
        """Dataset with varied inputs for comprehensive testing."""
        examples = []
        sentiments = ["positive", "negative", "neutral"]
        texts = [
            "This product is absolutely fantastic and exceeded all expectations!",
            "Terrible quality, completely broken and unusable product.",
            "It's an okay product, nothing special but works as expected.",
            "Amazing quality and excellent customer service experience!",
            "Poor design and very disappointing overall experience.",
            "Average product with standard features and decent quality.",
            "Outstanding performance and remarkable build quality!",
            "Awful experience, would not recommend to anyone.",
        ]

        for i, text in enumerate(texts):
            examples.append(
                EvaluationExample(
                    input_data={"text": text, "id": f"example_{i}"},
                    expected_output=sentiments[i % len(sentiments)],
                )
            )

        return Dataset(examples=examples, name="comprehensive_test")

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        original_mock = os.environ.get("MOCK_MODE", "")
        os.environ["MOCK_MODE"] = "true"
        yield
        os.environ["MOCK_MODE"] = original_mock

    @pytest.mark.asyncio
    async def test_complete_metrics_flow_string_function(
        self, comprehensive_dataset, mock_environment, monkeypatch
    ):
        """Test complete flow with function returning strings."""
        # Override mock mode for exact accuracy testing
        monkeypatch.setenv("TRAIGENT_MOCK_MODE", "false")

        async def sentiment_analysis_function(**kwargs) -> str:
            """Realistic sentiment analysis function for testing."""
            text = kwargs.get("text", "")
            temperature = kwargs.get("temperature", 0.5)
            kwargs.get("model", "test")
            text_lower = text.lower()

            # Simulate model behavior based on temperature
            if temperature < 0.4:
                # Conservative model
                if any(
                    word in text_lower
                    for word in ["fantastic", "amazing", "outstanding", "excellent"]
                ):
                    return "positive"
                elif any(
                    word in text_lower
                    for word in ["terrible", "awful", "broken", "disappointing"]
                ):
                    return "negative"
                else:
                    return "neutral"
            elif temperature > 0.7:
                # More sensitive model
                if any(
                    word in text_lower
                    for word in [
                        "good",
                        "great",
                        "excellent",
                        "fantastic",
                        "amazing",
                        "outstanding",
                    ]
                ):
                    return "positive"
                elif any(
                    word in text_lower
                    for word in [
                        "bad",
                        "poor",
                        "terrible",
                        "awful",
                        "broken",
                        "disappointing",
                    ]
                ):
                    return "negative"
                else:
                    return "neutral"
            else:
                # Balanced model
                if any(
                    word in text_lower
                    for word in ["excellent", "fantastic", "amazing", "outstanding"]
                ):
                    return "positive"
                elif any(
                    word in text_lower
                    for word in ["terrible", "awful", "broken", "disappointing"]
                ):
                    return "negative"
                else:
                    return "neutral"

        # Test with different execution modes
        for execution_mode in ["edge_analytics", "privacy"]:
            evaluator = LocalEvaluator(
                metrics=["accuracy"], detailed=True, execution_mode=execution_mode
            )

            config = {"temperature": 0.5, "model": "test-model"}
            result = await evaluator.evaluate(
                sentiment_analysis_function, config, comprehensive_dataset
            )

            # Check basic evaluation results
            assert result.total_examples == len(comprehensive_dataset.examples)
            assert result.successful_examples > 0
            assert len(result.example_results) == len(comprehensive_dataset.examples)

            # Check that all example results have token metrics
            for _i, example_result in enumerate(result.example_results):
                assert "input_tokens" in example_result.metrics
                assert "output_tokens" in example_result.metrics
                assert "total_tokens" in example_result.metrics
                assert "input_cost" in example_result.metrics
                assert "output_cost" in example_result.metrics
                assert "total_cost" in example_result.metrics

                # Token values should be positive
                assert example_result.metrics["input_tokens"] > 0
                assert example_result.metrics["output_tokens"] > 0
                assert example_result.metrics["total_tokens"] > 0

                # Cost should be 0 in mock mode
                assert example_result.metrics["input_cost"] == 0.0
                assert example_result.metrics["output_cost"] == 0.0
                assert example_result.metrics["total_cost"] == 0.0

                # Should have accuracy metric
                assert "accuracy" in example_result.metrics
                assert example_result.metrics["accuracy"] in [0.0, 1.0]

            # Check aggregated metrics
            assert "input_tokens" in result.aggregated_metrics
            assert "output_tokens" in result.aggregated_metrics
            assert "total_tokens" in result.aggregated_metrics
            assert (
                "cost" in result.aggregated_metrics
            )  # Cost is aggregated into single field
            assert "accuracy" in result.aggregated_metrics

            # Aggregated values should be reasonable
            assert result.aggregated_metrics["input_tokens"] > 0
            assert result.aggregated_metrics["output_tokens"] > 0
            assert result.aggregated_metrics["total_tokens"] > 0
            assert 0.0 <= result.aggregated_metrics["accuracy"] <= 1.0

            # Check summary stats if present
            if hasattr(result, "summary_stats") and result.summary_stats:
                metrics_list = result.summary_stats.get("metrics", {})
                expected_metrics = [
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_cost",
                    "output_cost",
                    "total_cost",
                    "accuracy",
                    "response_time_ms",
                ]
                for metric in expected_metrics:
                    assert metric in metrics_list

    @pytest.mark.asyncio
    async def test_complete_metrics_flow_llm_response_function(
        self, comprehensive_dataset, mock_environment
    ):
        """Test complete flow with function returning LLM response objects."""

        async def llm_response_function(**kwargs) -> MockLLMResponse:
            """Function that returns LLM response objects."""
            text = kwargs.get("text", "")
            temperature = kwargs.get("temperature", 0.5)
            model = kwargs.get("model", "gpt-3.5-turbo")

            # Simulate different costs based on model
            base_cost = 0.002 if "gpt-3.5" in model else 0.006
            input_tokens = max(10, len(text) // 4)
            output_tokens = max(5, 10 + int(temperature * 20))
            response_time = 1000 + int(temperature * 1000)

            # Simple sentiment logic
            text_lower = text.lower()
            if any(
                word in text_lower for word in ["fantastic", "amazing", "excellent"]
            ):
                sentiment = "positive"
            elif any(word in text_lower for word in ["terrible", "awful", "broken"]):
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return MockLLMResponse(
                text=sentiment,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=base_cost * (input_tokens + output_tokens) / 150,
                response_time=response_time,
            )

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"temperature": 0.7, "model": "gpt-3.5-turbo"}
        result = await evaluator.evaluate(
            llm_response_function, config, comprehensive_dataset
        )

        # Should extract real LLM metrics (not estimates)
        for example_result in result.example_results:
            # Should have used actual token counts from LLM response
            assert example_result.metrics["input_tokens"] >= 10
            assert example_result.metrics["output_tokens"] >= 5

            # Token total should match input + output
            assert (
                example_result.metrics["total_tokens"]
                == example_result.metrics["input_tokens"]
                + example_result.metrics["output_tokens"]
            )

    @pytest.mark.asyncio
    async def test_orchestrator_measures_array_creation(
        self, comprehensive_dataset, mock_environment
    ):
        """Test that orchestrator correctly creates measures array from evaluation results."""

        async def configurable_function(**kwargs) -> str:
            """Function with multiple configurable parameters."""
            text = kwargs.get("text", "")
            approach = kwargs.get("approach", "balanced")
            sensitivity = kwargs.get("sensitivity", 0.5)
            text_lower = text.lower()

            # Different approaches to sentiment analysis
            if approach == "conservative":
                threshold_words = ["fantastic", "terrible"]
            elif approach == "aggressive":
                threshold_words = ["good", "great", "excellent", "bad", "poor", "awful"]
            else:  # balanced
                threshold_words = ["excellent", "amazing", "terrible", "awful"]

            # Apply sensitivity
            positive_score = sum(
                1
                for word in threshold_words[: len(threshold_words) // 2]
                if word in text_lower
            )
            negative_score = sum(
                1
                for word in threshold_words[len(threshold_words) // 2 :]
                if word in text_lower
            )

            adjusted_positive = positive_score * sensitivity
            adjusted_negative = negative_score * sensitivity

            if adjusted_positive > adjusted_negative:
                return "positive"
            elif adjusted_negative > adjusted_positive:
                return "negative"
            else:
                return "neutral"

        # Set up optimization
        config_space = {
            "approach": ["conservative", "balanced", "aggressive"],
            "sensitivity": [0.3, 0.7, 1.0],
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

        # Run optimization
        await orchestrator.create_session()
        optimization_results = await orchestrator.optimize(
            configurable_function, comprehensive_dataset
        )

        # Check that trials have measures arrays
        assert (
            len(optimization_results.trials) == 9
        )  # 3 approaches × 3 sensitivity values

        for trial in optimization_results.trials:
            # Each trial should have some form of measurements/metrics
            # Check for different possible attribute names
            measures = getattr(trial, "measures", None)
            if measures is None:
                # Try other possible attribute names
                measures = getattr(trial, "metrics", [])
                if not measures:
                    measures = getattr(trial, "results", [])
                if not measures:
                    measures = getattr(trial, "example_results", [])

            # If no measures found directly, check if the trial has evaluation results
            if not measures and hasattr(trial, "result"):
                eval_result = trial.result
                if hasattr(eval_result, "example_results"):
                    measures = eval_result.example_results

            # Should have some form of results
            assert (
                len(measures) > 0
            ), f"No measures found in trial. Trial attributes: {dir(trial)}"

            for measure in measures:
                # Handle both dict format and object format
                if hasattr(measure, "metrics"):
                    # ExampleResult object format
                    metrics = measure.metrics
                    assert "input_tokens" in metrics
                    assert "output_tokens" in metrics
                    assert "total_tokens" in metrics
                    assert "accuracy" in metrics

                    # Token values should be positive
                    assert metrics["input_tokens"] > 0
                    assert metrics["output_tokens"] > 0
                    assert metrics["total_tokens"] > 0
                    assert metrics["accuracy"] in [0.0, 1.0]

                    # Cost should be 0 in mock mode
                    assert metrics["input_cost"] == 0.0
                    assert metrics["output_cost"] == 0.0
                    assert metrics["total_cost"] == 0.0
                elif isinstance(measure, dict):
                    # Dict format
                    assert "input_tokens" in measure
                    assert "output_tokens" in measure
                    assert "total_tokens" in measure

                    # Token values should be positive
                    assert measure["input_tokens"] > 0
                    assert measure["output_tokens"] > 0
                    assert measure["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_metrics_flow_with_failures(
        self, comprehensive_dataset, mock_environment, monkeypatch
    ):
        """Test metrics flow when some function calls fail."""
        # Override mock mode for exact accuracy testing
        monkeypatch.setenv("TRAIGENT_MOCK_MODE", "false")

        call_count = 0

        async def error_prone_function(**kwargs) -> str:
            """Function that fails sometimes."""
            nonlocal call_count
            call_count += 1

            text = kwargs.get("text", "")
            error_rate = kwargs.get("error_rate", 0.2)

            # Simulate errors based on error_rate
            import random

            random.seed(call_count)  # Deterministic for testing
            if random.random() < error_rate:
                raise ValueError(f"Simulated error on call {call_count}")

            # Simple sentiment analysis for successful calls
            if "excellent" in text.lower() or "amazing" in text.lower():
                return "positive"
            elif "terrible" in text.lower() or "awful" in text.lower():
                return "negative"
            else:
                return "neutral"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"error_rate": 0.25}  # 25% error rate
        result = await evaluator.evaluate(
            error_prone_function, config, comprehensive_dataset
        )

        # Should have mix of successful and failed results
        successful_results = [r for r in result.example_results if r.success]
        failed_results = [r for r in result.example_results if not r.success]

        assert len(successful_results) > 0
        assert len(failed_results) > 0
        assert len(successful_results) + len(failed_results) == len(
            comprehensive_dataset.examples
        )

        # Successful results should have full metrics
        for success_result in successful_results:
            assert success_result.metrics["input_tokens"] > 0
            assert success_result.metrics["output_tokens"] > 0
            assert success_result.metrics["accuracy"] in [0.0, 1.0]

        # Failed results should have minimal metrics
        for failed_result in failed_results:
            # May or may not have token metrics depending on when the error occurred
            assert not failed_result.success
            # Check for error information - could be error_message or exception_message
            assert (
                hasattr(failed_result, "error_message")
                and failed_result.error_message is not None
            ) or (
                hasattr(failed_result, "exception_message")
                and failed_result.exception_message is not None
            )

        # Aggregated metrics should only include successful examples
        assert result.successful_examples == len(successful_results)
        assert result.total_examples == len(comprehensive_dataset.examples)

    @pytest.mark.asyncio
    async def test_random_search_metrics_flow(
        self, comprehensive_dataset, mock_environment
    ):
        """Test metrics flow with RandomSearchOptimizer."""

        async def random_optimizable_function(**kwargs) -> str:
            """Function with continuous parameters for random search."""
            text = kwargs.get("text", "")
            threshold = kwargs.get("threshold", 0.5)
            boost = kwargs.get("boost", 1.0)
            text_lower = text.lower()

            # Score based on positive/negative words
            positive_words = [
                "excellent",
                "fantastic",
                "amazing",
                "great",
                "outstanding",
            ]
            negative_words = ["terrible", "awful", "horrible", "bad", "disappointing"]

            positive_score = (
                sum(1 for word in positive_words if word in text_lower) * boost
            )
            negative_score = (
                sum(1 for word in negative_words if word in text_lower) * boost
            )

            if positive_score > threshold:
                return "positive"
            elif negative_score > threshold:
                return "negative"
            else:
                return "neutral"

        # Set up random search
        config_space = {
            "threshold": (0.1, 1.5),  # Continuous range
            "boost": (0.5, 2.0),  # Continuous range
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space, objectives=["accuracy"], max_trials=6
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

        # Run random search optimization
        await orchestrator.create_session()
        optimization_results = await orchestrator.optimize(
            random_optimizable_function, comprehensive_dataset
        )

        # Should have generated random configurations
        assert (
            len(optimization_results.trials) == 6
        )  # max_trials from RandomSearchOptimizer

        # Each trial should have different configurations
        configs = [trial.config for trial in optimization_results.trials]
        assert len({tuple(sorted(config.items())) for config in configs}) > 1

        # All trials should have complete metrics
        for trial in optimization_results.trials:
            # Use the same flexible approach to find measures
            measures = getattr(trial, "measures", None)
            if measures is None:
                measures = getattr(trial, "metrics", [])
                if not measures:
                    measures = getattr(trial, "results", [])
                if not measures:
                    measures = getattr(trial, "example_results", [])

            # If no measures found directly, check if the trial has evaluation results
            if not measures and hasattr(trial, "result"):
                eval_result = trial.result
                if hasattr(eval_result, "example_results"):
                    measures = eval_result.example_results

            assert len(measures) > 0, "No measures found in trial"

            # Check measures have all required fields
            for measure in measures:
                if hasattr(measure, "metrics"):
                    # ExampleResult object format
                    metrics = measure.metrics
                    assert "input_tokens" in metrics and metrics["input_tokens"] > 0
                    assert "output_tokens" in metrics and metrics["output_tokens"] > 0
                    assert "accuracy" in metrics and metrics["accuracy"] in [0.0, 1.0]
                elif isinstance(measure, dict):
                    # Dict format
                    assert "input_tokens" in measure and measure["input_tokens"] > 0
                    assert "output_tokens" in measure and measure["output_tokens"] > 0
                    assert "accuracy" in measure and measure["accuracy"] in [0.0, 1.0]

    @pytest.mark.asyncio
    async def test_backend_submission_format(
        self, comprehensive_dataset, mock_environment
    ):
        """Test that metrics are formatted correctly for backend submission."""

        async def backend_test_function(text: str, model: str = "test") -> str:
            """Simple function for testing backend format."""
            return "positive" if len(text) > 50 else "negative"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"model": "test-model"}
        result = await evaluator.evaluate(
            backend_test_function, config, comprehensive_dataset
        )

        # Test MetricsTracker's format_for_backend method
        if hasattr(result, "_metrics_tracker") and result._metrics_tracker:
            backend_format = result._metrics_tracker.format_for_backend()

            # Check required fields for backend submission
            required_fields = [
                "score",
                "accuracy",
                "duration",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "response_time_ms",
                "cost",
                "total_examples",
                "successful_examples",
            ]

            for field in required_fields:
                assert field in backend_format, f"Missing {field} in backend format"

            # Check data types
            assert isinstance(backend_format["score"], (int, float))
            assert isinstance(backend_format["accuracy"], (int, float))
            assert isinstance(backend_format["duration"], (int, float))
            assert isinstance(backend_format["input_tokens"], (int, float))
            assert isinstance(backend_format["output_tokens"], (int, float))
            assert isinstance(backend_format["total_tokens"], (int, float))
            assert isinstance(backend_format["response_time_ms"], (int, float))
            assert isinstance(backend_format["cost"], (int, float))
            assert isinstance(backend_format["total_examples"], int)
            assert isinstance(backend_format["successful_examples"], int)

            # Check value ranges
            assert 0.0 <= backend_format["accuracy"] <= 1.0
            assert backend_format["total_examples"] > 0
            assert backend_format["successful_examples"] >= 0
            assert (
                backend_format["successful_examples"]
                <= backend_format["total_examples"]
            )

    @pytest.mark.asyncio
    async def test_summary_stats_pandas_format(
        self, comprehensive_dataset, mock_environment
    ):
        """Test that summary_stats follow pandas.describe() format."""

        async def stats_test_function(
            text: str, variation: float = 0.5, **kwargs
        ) -> str:
            """Function that creates variation for statistics."""
            # Create some variation based on input
            text_hash = hash(text) % 100
            adjusted_variation = variation + (
                text_hash / 1000
            )  # Small variation per input

            if adjusted_variation > 0.6:
                return "positive"
            elif adjusted_variation < 0.4:
                return "negative"
            else:
                return "neutral"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {"variation": 0.5}
        result = await evaluator.evaluate(
            stats_test_function, config, comprehensive_dataset
        )

        # Check summary stats format
        if hasattr(result, "summary_stats") and result.summary_stats:
            summary_stats = result.summary_stats

            # Should have pandas.describe() structure
            assert "metrics" in summary_stats
            assert "execution_time" in summary_stats
            assert "total_examples" in summary_stats
            assert "metadata" in summary_stats

            # Metadata should indicate pandas format
            assert summary_stats["metadata"]["aggregation_method"] == "pandas.describe"
            assert "timestamp" in summary_stats["metadata"]
            assert "sdk_version" in summary_stats["metadata"]

            # Each metric should have pandas.describe() fields
            for metric_name, stats in summary_stats["metrics"].items():
                required_stats = [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
                for stat_field in required_stats:
                    assert stat_field in stats, f"Missing {stat_field} in {metric_name}"

                # Check reasonable values
                assert stats["count"] > 0
                assert (
                    stats["min"]
                    <= stats["25%"]
                    <= stats["50%"]
                    <= stats["75%"]
                    <= stats["max"]
                )
                assert stats["mean"] >= stats["min"]
                assert stats["mean"] <= stats["max"]


class TestMetricsFlowPerformance:
    """Test performance aspects of metrics flow."""

    @pytest.fixture
    def large_dataset(self):
        """Larger dataset for performance testing."""
        examples = []
        for i in range(50):  # Reasonably sized for testing
            examples.append(
                EvaluationExample(
                    input_data={
                        "text": f"Performance test input number {i} with some additional content to vary length"
                    },
                    expected_output="positive" if i % 2 == 0 else "negative",
                )
            )
        return Dataset(examples=examples, name="performance_test")

    @pytest.mark.asyncio
    async def test_metrics_flow_performance(self, large_dataset):
        """Test that metrics flow performs reasonably with larger datasets."""

        async def performance_function(text: str) -> str:
            """Simple function for performance testing."""
            return "positive" if "test" in text.lower() else "negative"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        import time

        start_time = time.time()

        config = {}
        result = await evaluator.evaluate(performance_function, config, large_dataset)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 30.0  # Should complete within 30 seconds

        # Should have processed all examples
        assert result.total_examples == len(large_dataset.examples)
        assert len(result.example_results) == len(large_dataset.examples)

        # All examples should have metrics
        for example_result in result.example_results:
            assert "input_tokens" in example_result.metrics
            assert "output_tokens" in example_result.metrics
            assert example_result.metrics["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, large_dataset):
        """Test that metrics flow doesn't consume excessive memory."""

        async def memory_test_function(text: str) -> str:
            """Function for memory testing."""
            # Create consistent output to avoid memory bloat from varied strings
            return "test_output"

        evaluator = LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
        )

        config = {}
        result = await evaluator.evaluate(memory_test_function, config, large_dataset)

        # Check that results are reasonable in size
        import sys

        result_size = sys.getsizeof(result)

        # Should not be excessively large (adjust threshold based on needs)
        assert result_size < 10 * 1024 * 1024  # Less than 10MB for test data

        # Verify data integrity
        assert len(result.example_results) == len(large_dataset.examples)
        for example_result in result.example_results:
            assert example_result.metrics["input_tokens"] > 0
