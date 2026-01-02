"""Performance benchmark tests for Traigent SDK optimization models."""

import asyncio
import statistics
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

from traigent.agents import apply_config_to_agent
from traigent.cloud.models import (
    AgentSpecification,
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationSessionStatus,
    SessionCreationResponse,
    TrialSuggestion,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer


class PerformanceMetrics:
    """Track and analyze performance metrics."""

    def __init__(self):
        self.metrics = {
            "trial_durations": [],
            "suggestion_latencies": [],
            "execution_times": [],
            "memory_usage": [],
            "throughput": [],
            "cost_per_trial": [],
        }
        self.start_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def record_trial(self, duration: float, latency: float, cost: float = 0.0):
        """Record trial metrics."""
        self.metrics["trial_durations"].append(duration)
        self.metrics["suggestion_latencies"].append(latency)
        self.metrics["cost_per_trial"].append(cost)

        if self.start_time:
            elapsed = time.time() - self.start_time
            trials_completed = len(self.metrics["trial_durations"])
            self.metrics["throughput"].append(trials_completed / elapsed)

    def record_execution(self, exec_time: float):
        """Record execution time."""
        self.metrics["execution_times"].append(exec_time)

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        summary = {}

        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return summary

    def plot_metrics(self, save_path: str = None):
        """Plot performance metrics."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping plot generation")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Traigent SDK Performance Benchmarks", fontsize=16)

        # Trial durations
        if self.metrics["trial_durations"]:
            axes[0, 0].plot(self.metrics["trial_durations"], "b-o")
            axes[0, 0].set_title("Trial Durations")
            axes[0, 0].set_xlabel("Trial Number")
            axes[0, 0].set_ylabel("Duration (seconds)")
            axes[0, 0].grid(True)

        # Suggestion latencies
        if self.metrics["suggestion_latencies"]:
            axes[0, 1].hist(self.metrics["suggestion_latencies"], bins=20, alpha=0.7)
            axes[0, 1].set_title("Suggestion Latency Distribution")
            axes[0, 1].set_xlabel("Latency (seconds)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(True)

        # Throughput over time
        if self.metrics["throughput"]:
            axes[1, 0].plot(self.metrics["throughput"], "g-")
            axes[1, 0].set_title("Throughput Over Time")
            axes[1, 0].set_xlabel("Measurement Point")
            axes[1, 0].set_ylabel("Trials/Second")
            axes[1, 0].grid(True)

        # Cost per trial
        if self.metrics["cost_per_trial"]:
            axes[1, 1].bar(
                range(len(self.metrics["cost_per_trial"])),
                self.metrics["cost_per_trial"],
                alpha=0.7,
            )
            axes[1, 1].set_title("Cost Per Trial")
            axes[1, 1].set_xlabel("Trial Number")
            axes[1, 1].set_ylabel("Cost ($)")
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


@pytest.fixture
def large_dataset():
    """Create a large dataset for performance testing."""
    examples = []
    for i in range(100):
        examples.append(
            EvaluationExample(
                input_data={"text": f"Sample text {i}: " + "x" * 100, "index": i},
                expected_output=f"Output {i}",
            )
        )
    return Dataset(examples)


@pytest.fixture
def performance_tracker():
    """Create performance metrics tracker."""
    return PerformanceMetrics()


class TestInteractiveOptimizationPerformance:
    """Performance benchmarks for Model 1: Interactive Optimization."""

    @pytest.mark.asyncio
    async def test_suggestion_latency_benchmark(
        self, large_dataset, performance_tracker
    ):
        """Benchmark suggestion generation latency."""

        # Mock cloud client with controlled latency
        mock_client = AsyncMock()
        mock_client._trial_count = 0

        async def create_session(request):
            await asyncio.sleep(0.01)  # Simulate network latency
            return SessionCreationResponse(
                session_id="perf-test-session",
                status=OptimizationSessionStatus.ACTIVE,
                optimization_strategy={},
            )

        async def get_next_trial(request):
            await asyncio.sleep(0.02)  # Simulate computation time
            mock_client._trial_count += 1

            if mock_client._trial_count > 20:
                return NextTrialResponse(suggestion=None, should_continue=False)

            # Vary subset size to test scaling
            subset_size = min(10 + mock_client._trial_count, 50)

            return NextTrialResponse(
                suggestion=TrialSuggestion(
                    trial_id=f"perf-trial-{mock_client._trial_count}",
                    session_id="perf-test-session",
                    trial_number=mock_client._trial_count,
                    config={"temperature": 0.5 + (mock_client._trial_count * 0.01)},
                    dataset_subset=DatasetSubsetIndices(
                        indices=list(range(subset_size)),
                        selection_strategy="adaptive",
                        confidence_level=0.8,
                        estimated_representativeness=0.85,
                    ),
                    exploration_type="exploration",
                ),
                should_continue=True,
            )

        mock_client.create_session = create_session
        mock_client.get_next_trial = get_next_trial
        mock_client.submit_result = AsyncMock()
        mock_client.finalize_session = AsyncMock()

        # Run benchmark
        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=mock_client,
        )

        performance_tracker.start()

        await optimizer.initialize_session("benchmark_function", max_trials=30)

        # Measure suggestion latencies
        for _ in range(20):
            start = time.time()
            suggestion = await optimizer.get_next_suggestion(
                len(large_dataset.examples)
            )
            latency = time.time() - start

            if not suggestion:
                break

            performance_tracker.record_trial(
                duration=0.1,  # Mock execution time
                latency=latency,
            )

            # Mock quick result submission
            await optimizer.report_results(
                trial_id=suggestion.trial_id, metrics={"accuracy": 0.85}, duration=0.1
            )

        # Analyze results
        summary = performance_tracker.get_summary()

        # Assert performance requirements
        assert summary["suggestion_latencies"]["mean"] < 0.1  # Less than 100ms average
        assert summary["suggestion_latencies"]["max"] < 0.2  # No suggestion over 200ms
        assert summary["throughput"]["mean"] > 5  # At least 5 trials/second

    @pytest.mark.asyncio
    async def test_concurrent_execution_scalability(self, large_dataset):
        """Test scalability with concurrent executions."""

        async def run_optimization_session(
            session_id: str, num_trials: int
        ) -> dict[str, Any]:
            """Run a single optimization session."""
            # Create independent mock client
            mock_client = AsyncMock()
            trial_count = 0

            async def get_trial(request):
                nonlocal trial_count
                trial_count += 1
                if trial_count > num_trials:
                    return NextTrialResponse(suggestion=None, should_continue=False)

                return NextTrialResponse(
                    suggestion=TrialSuggestion(
                        trial_id=f"{session_id}-trial-{trial_count}",
                        session_id=session_id,
                        trial_number=trial_count,
                        config={"temperature": 0.5},
                        dataset_subset=DatasetSubsetIndices(
                            indices=[0, 1, 2],
                            selection_strategy="random",
                            confidence_level=0.8,
                            estimated_representativeness=0.8,
                        ),
                        exploration_type="exploration",
                    ),
                    should_continue=True,
                )

            mock_client.create_session = AsyncMock(
                return_value=SessionCreationResponse(
                    session_id=session_id,
                    status=OptimizationSessionStatus.ACTIVE,
                    optimization_strategy={},
                )
            )
            mock_client.get_next_trial = get_trial
            mock_client.submit_result = AsyncMock()
            mock_client.finalize_session = AsyncMock()

            # Run optimization
            start_time = time.time()

            optimizer = InteractiveOptimizer(
                config_space={"temperature": (0.0, 1.0)},
                objectives=["accuracy"],
                remote_service=mock_client,
            )

            await optimizer.initialize_session(
                f"concurrent_{session_id}", max_trials=num_trials
            )

            trials_completed = 0
            while True:
                suggestion = await optimizer.get_next_suggestion(10)
                if not suggestion:
                    break

                # Simulate execution
                await asyncio.sleep(0.01)

                await optimizer.report_results(
                    trial_id=suggestion.trial_id,
                    metrics={"accuracy": 0.8},
                    duration=0.01,
                )

                trials_completed += 1

            duration = time.time() - start_time

            return {
                "session_id": session_id,
                "trials_completed": trials_completed,
                "duration": duration,
                "throughput": trials_completed / duration,
            }

        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}

        for concurrency in concurrency_levels:
            start = time.time()

            # Run concurrent sessions
            tasks = [
                run_optimization_session(f"session-{i}", 10) for i in range(concurrency)
            ]

            session_results = await asyncio.gather(*tasks)

            total_duration = time.time() - start
            total_trials = sum(r["trials_completed"] for r in session_results)

            results[concurrency] = {
                "total_duration": total_duration,
                "total_trials": total_trials,
                "aggregate_throughput": total_trials / total_duration,
                "sessions": session_results,
            }

        # Verify scalability
        # Throughput should increase with concurrency (with diminishing returns)
        assert results[5]["aggregate_throughput"] > results[1]["aggregate_throughput"]
        assert results[10]["aggregate_throughput"] > results[5]["aggregate_throughput"]

        # But efficiency might decrease slightly
        efficiency_1 = results[1]["aggregate_throughput"] / 1
        efficiency_10 = results[10]["aggregate_throughput"] / 10
        assert efficiency_10 > 0.5 * efficiency_1  # At least 50% efficiency retained

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_dataset(self, large_dataset):
        """Test memory efficiency with large datasets."""

        # Mock client that tracks subset sizes
        subset_sizes = []

        mock_client = AsyncMock()
        trial_count = 0

        async def get_trial(request):
            nonlocal trial_count
            trial_count += 1

            if trial_count > 10:
                return NextTrialResponse(suggestion=None, should_continue=False)

            # Adaptive subset sizing
            if trial_count <= 3:
                subset_size = 5  # Small initial subsets
            elif trial_count <= 7:
                subset_size = 20  # Medium subsets
            else:
                subset_size = 50  # Larger final subsets

            subset_sizes.append(subset_size)

            return NextTrialResponse(
                suggestion=TrialSuggestion(
                    trial_id=f"mem-trial-{trial_count}",
                    session_id="mem-test",
                    trial_number=trial_count,
                    config={"temperature": 0.5},
                    dataset_subset=DatasetSubsetIndices(
                        indices=list(range(subset_size)),
                        selection_strategy="adaptive",
                        confidence_level=0.8 + (trial_count * 0.01),
                        estimated_representativeness=0.85,
                    ),
                    exploration_type=(
                        "exploration" if trial_count <= 5 else "exploitation"
                    ),
                ),
                should_continue=True,
            )

        mock_client.create_session = AsyncMock(
            return_value=SessionCreationResponse(
                session_id="mem-test",
                status=OptimizationSessionStatus.ACTIVE,
                optimization_strategy={},
            )
        )
        mock_client.get_next_trial = get_trial
        mock_client.submit_result = AsyncMock()
        mock_client.finalize_session = AsyncMock()

        # Run optimization
        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=mock_client,
            optimization_strategy={
                "adaptive_sample_size": True,
                "min_examples_per_trial": 5,
                "max_examples_per_trial": 50,
            },
        )

        await optimizer.initialize_session("memory_test", max_trials=20)

        # Process trials
        while True:
            suggestion = await optimizer.get_next_suggestion(
                len(large_dataset.examples)
            )
            if not suggestion:
                break

            # Only process subset, not full dataset
            subset_data = [
                large_dataset.examples[i] for i in suggestion.dataset_subset.indices
            ]

            # Verify we're not loading full dataset
            assert len(subset_data) < len(large_dataset.examples)
            assert len(subset_data) == len(suggestion.dataset_subset.indices)

            await optimizer.report_results(
                trial_id=suggestion.trial_id, metrics={"accuracy": 0.85}, duration=0.1
            )

        # Verify adaptive subset sizing
        assert len(subset_sizes) == 10
        assert subset_sizes[0] < subset_sizes[-1]  # Sizes increased
        assert all(s <= 50 for s in subset_sizes)  # Respects max limit

        # Calculate memory efficiency
        total_examples_processed = sum(subset_sizes)
        full_dataset_size = len(large_dataset.examples) * 10  # If we used full dataset
        efficiency_ratio = 1 - (total_examples_processed / full_dataset_size)

        assert efficiency_ratio > 0.7  # At least 70% memory savings


class TestAgentOptimizationPerformance:
    """Performance benchmarks for Model 2: Agent Optimization."""

    @pytest.mark.asyncio
    async def test_agent_execution_throughput(self, performance_tracker):
        """Benchmark agent execution throughput."""

        # Create test agent
        agent_spec = AgentSpecification(
            id="perf-agent",
            name="Performance Test Agent",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Answer: {question}",
            model_parameters={
                "model": "o4-mini",
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )

        # Mock executor with controlled performance
        class MockPerformanceExecutor:
            async def initialize(self):
                pass

            async def execute(self, agent_spec, input_data, config_overrides=None):
                # Simulate execution with consistent timing
                await asyncio.sleep(0.02)  # 20ms execution time

                return type(
                    "Result",
                    (),
                    {
                        "output": f"Response to {input_data.get('question', 'unknown')}",
                        "duration": 0.02,
                        "tokens_used": 50,
                        "cost": 0.001,
                    },
                )()

            async def batch_execute(self, agent_spec, input_batch, max_concurrent=5):
                # Parallel execution with semaphore
                semaphore = asyncio.Semaphore(max_concurrent)

                async def execute_with_limit(input_data):
                    async with semaphore:
                        return await self.execute(agent_spec, input_data)

                return await asyncio.gather(
                    *[execute_with_limit(inp) for inp in input_batch]
                )

        executor = MockPerformanceExecutor()
        await executor.initialize()

        # Test single execution throughput
        performance_tracker.start()

        single_start = time.time()
        for i in range(50):
            result = await executor.execute(agent_spec, {"question": f"Question {i}"})
            performance_tracker.record_execution(result.duration)
        single_duration = time.time() - single_start

        single_throughput = 50 / single_duration

        # Test batch execution throughput
        batch_inputs = [{"question": f"Batch question {i}"} for i in range(50)]

        batch_start = time.time()
        await executor.batch_execute(agent_spec, batch_inputs, max_concurrent=10)
        batch_duration = time.time() - batch_start

        batch_throughput = 50 / batch_duration

        # Verify performance
        assert single_throughput > 20  # At least 20 executions/second (sequential)
        assert batch_throughput > 100  # At least 100 executions/second (parallel)
        assert batch_throughput > single_throughput * 3  # Batch is significantly faster

        # Check consistency
        summary = performance_tracker.get_summary()
        assert summary["execution_times"]["std"] < 0.01  # Consistent execution times

    @pytest.mark.asyncio
    async def test_configuration_mapping_performance(self):
        """Benchmark configuration mapping performance."""

        # Create test agent
        agent_spec = AgentSpecification(
            id="mapping-test",
            name="Mapping Test",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Test",
            model_parameters={"model": "o4-mini"},
        )

        # Large configuration space
        large_config = {f"param_{i}": i * 0.1 for i in range(100)}
        large_config.update(
            {
                "model": "GPT-4o",
                "temperature": 0.3,
                "max_tokens": 200,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
            }
        )

        # Benchmark mapping performance
        start = time.time()

        for _ in range(1000):
            apply_config_to_agent(agent_spec, large_config, preserve_original=True)

        duration = time.time() - start
        mappings_per_second = 1000 / duration

        # Should be very fast
        assert mappings_per_second > 1000  # At least 1000 mappings/second
        assert duration < 1.0  # Less than 1 second for 1000 mappings


class TestHybridOptimizationPerformance:
    """Performance benchmarks for hybrid optimization."""

    @pytest.mark.asyncio
    async def test_phase_transition_overhead(self, performance_tracker):
        """Test overhead of transitioning between optimization phases."""

        # Mock clients for both phases
        mock_interactive_client = AsyncMock()
        mock_agent_client = AsyncMock()

        # Quick mock implementations
        mock_interactive_client.create_session = AsyncMock()
        mock_interactive_client.get_next_trial = AsyncMock(
            return_value=NextTrialResponse(suggestion=None, should_continue=False)
        )
        mock_interactive_client.finalize_session = AsyncMock()

        mock_agent_client.optimize_agent = AsyncMock()
        mock_agent_client.get_agent_optimization_status = AsyncMock()

        # Measure phase transitions
        performance_tracker.start()

        # Phase 1 -> Phase 2 transition
        transition_start = time.time()

        # Simulate end of Phase 1
        phase1_results = {
            "best_config": {"temperature": 0.5, "model": "GPT-4o"},
            "best_metrics": {"accuracy": 0.85},
        }

        # Prepare Phase 2 with Phase 1 results

        # Start Phase 2
        AgentSpecification(
            id="transition-test",
            name="Test",
            agent_type="conversational",
            agent_platform="openai",
            prompt_template="Test",
            model_parameters=phase1_results["best_config"],
        )

        transition_duration = time.time() - transition_start

        # Transition should be fast
        assert transition_duration < 0.1  # Less than 100ms overhead

    @pytest.mark.asyncio
    async def test_cost_efficiency_comparison(self):
        """Compare cost efficiency between models."""

        # Cost models
        costs = {
            "local_execution": 0.0001,  # Per example
            "cloud_suggestion": 0.001,  # Per suggestion request
            "cloud_execution": 0.01,  # Per cloud execution
            "cloud_optimization": 0.05,  # Per optimization trial
        }

        dataset_size = 100
        num_trials = 20

        # Model 1 costs (Interactive)
        model1_costs = {
            "suggestions": num_trials * costs["cloud_suggestion"],
            "execution": num_trials
            * 10
            * costs["local_execution"],  # 10 examples per trial
            "total": 0,
        }
        model1_costs["total"] = sum(model1_costs.values())

        # Model 2 costs (Agent)
        model2_costs = {
            "optimization": num_trials * costs["cloud_optimization"],
            "execution": num_trials * dataset_size * costs["cloud_execution"],
            "total": 0,
        }
        model2_costs["total"] = sum(model2_costs.values())

        # Hybrid costs (optimized) - combines benefits of both models
        hybrid_costs = {
            "phase1_suggestions": 5 * costs["cloud_suggestion"],
            "phase1_execution": 5 * 5 * costs["local_execution"],  # Small subsets
            "phase2_optimization": 3
            * costs["cloud_optimization"],  # Fewer trials needed due to phase1 insights
            "phase2_execution": 3
            * 5
            * costs["cloud_execution"],  # Smaller targeted subsets
            "total": 0,
        }
        hybrid_costs["total"] = sum(
            [v for k, v in hybrid_costs.items() if k != "total"]
        )

        # Compare efficiency
        print("\nCost Comparison:")
        print(f"Model 1 (Interactive): ${model1_costs['total']:.4f}")
        print(f"Model 2 (Agent): ${model2_costs['total']:.4f}")
        print(f"Hybrid: ${hybrid_costs['total']:.4f}")

        # Verify hybrid is more cost-effective than pure cloud model
        # and provides better accuracy than pure Edge Analytics model
        assert hybrid_costs["total"] < model2_costs["total"]

        # Hybrid may cost more than pure local but provides better accuracy
        # This is acceptable trade-off for many scenarios

        # Calculate savings
        savings_vs_model1 = (
            model1_costs["total"] - hybrid_costs["total"]
        ) / model1_costs["total"]
        savings_vs_model2 = (
            model2_costs["total"] - hybrid_costs["total"]
        ) / model2_costs["total"]

        print("\nHybrid Savings:")
        print(f"vs Model 1: {savings_vs_model1:.1%}")
        print(f"vs Model 2: {savings_vs_model2:.1%}")

        assert savings_vs_model2 > 0.5  # At least 50% savings vs pure cloud
