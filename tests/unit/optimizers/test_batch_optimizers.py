"""Tests for batch optimization strategies."""

import asyncio

from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample, EvaluationResult
from traigent.invokers.base import InvocationResult
from traigent.optimizers.batch_optimizers import (
    AdaptiveBatchOptimizer,
    BatchOptimizationConfig,
    MultiObjectiveBatchOptimizer,
    ParallelBatchOptimizer,
)
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer


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

        accuracy = successful_count / max(1, total_count)
        success_rate = successful_count / max(1, total_count)

        return EvaluationResult(
            config={},
            total_examples=total_count,
            successful_examples=successful_count,
            duration=1.0,
            aggregated_metrics={
                "accuracy": accuracy,
                "success_rate": success_rate,
                "avg_execution_time": 0.1,
            },
        )


class TestBatchOptimizationConfig:
    """Test suite for BatchOptimizationConfig."""

    def test_init_default_values(self):
        """Test BatchOptimizationConfig initialization with defaults."""
        config = BatchOptimizationConfig()

        assert config.max_parallel_trials == 4
        assert config.batch_size == 10
        assert config.adaptive_batching is True
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 0.001
        assert config.distributed_workers == 1
        assert config.enable_checkpointing is True
        assert config.memory_limit_mb == 1000.0

    def test_init_custom_values(self):
        """Test BatchOptimizationConfig initialization with custom values."""
        config = BatchOptimizationConfig(
            max_parallel_trials=8,
            batch_size=20,
            adaptive_batching=False,
            early_stopping_patience=5,
            memory_limit_mb=2000.0,
        )

        assert config.max_parallel_trials == 8
        assert config.batch_size == 20
        assert config.adaptive_batching is False
        assert config.early_stopping_patience == 5
        assert config.memory_limit_mb == 2000.0


class TestParallelBatchOptimizer:
    """Test suite for ParallelBatchOptimizer."""

    def setup_method(self):
        """Set up test data."""
        self.config_space = {"param1": [1, 2, 3], "param2": [0.1, 0.2, 0.3]}
        self.objectives = ["accuracy"]

        self.base_optimizer = GridSearchOptimizer(self.config_space, self.objectives)
        self.batch_config = BatchOptimizationConfig(max_parallel_trials=2, batch_size=2)

        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
                EvaluationExample(input_data={"value": 3}, expected_output="expected3"),
            ],
            name="test_dataset",
        )

    def test_init(self):
        """Test ParallelBatchOptimizer initialization."""
        optimizer = ParallelBatchOptimizer(
            base_optimizer=self.base_optimizer,
            batch_config=self.batch_config,
            objectives=self.objectives,
        )

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert optimizer.base_optimizer == self.base_optimizer
        assert optimizer.batch_config == self.batch_config
        assert optimizer.adaptive_sizer is not None

    def test_optimize_basic(self):
        """Test basic parallel batch optimization."""

        async def run_test():
            optimizer = ParallelBatchOptimizer(
                base_optimizer=self.base_optimizer, batch_config=self.batch_config
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}_{config.custom_params.get('param1', 0)}"

            invoker = MockInvoker()
            evaluator = MockEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=4,
            )

            assert result.best_config is not None
            assert result.best_score >= 0
            assert len(result.trials) <= 4
            assert result.duration > 0
            assert "parallel_workers" in result.convergence_info
            assert result.convergence_info["parallel_workers"] == 2

        asyncio.run(run_test())

    def test_optimize_early_stopping(self):
        """Test early stopping in parallel optimization."""

        async def run_test():
            # Configure for early stopping
            early_config = BatchOptimizationConfig(
                max_parallel_trials=2, early_stopping_patience=2
            )

            optimizer = ParallelBatchOptimizer(
                base_optimizer=self.base_optimizer, batch_config=early_config
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}"

            invoker = MockInvoker()
            evaluator = MockEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=10,  # More than needed for early stopping
            )

            # Should stop early
            assert len(result.trials) < 10
            assert "early_stopped" in result.convergence_info

        asyncio.run(run_test())


class TestMultiObjectiveBatchOptimizer:
    """Test suite for MultiObjectiveBatchOptimizer."""

    def setup_method(self):
        """Set up test data."""
        self.config_space = {"param1": [1, 2], "param2": [0.1, 0.2]}
        self.objectives = ["accuracy", "success_rate"]
        self.batch_config = BatchOptimizationConfig(batch_size=2)

        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
            ],
            name="test_dataset",
        )

    def test_init(self):
        """Test MultiObjectiveBatchOptimizer initialization."""
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=self.objectives,
            batch_config=self.batch_config,
            pareto_frontier_size=20,
        )

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert optimizer.batch_config == self.batch_config
        assert optimizer.pareto_frontier_size == 20
        assert len(optimizer.pareto_frontier) == 0

    def test_optimize_multi_objective(self):
        """Test multi-objective batch optimization."""

        async def run_test():
            optimizer = MultiObjectiveBatchOptimizer(
                configuration_space=self.config_space,
                objectives=self.objectives,
                batch_config=self.batch_config,
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}_{config.custom_params.get('param1', 0)}"

            # Mock evaluator with multiple objectives
            class MultiObjectiveEvaluator:
                async def evaluate(self, invocation_results, expected_outputs, dataset):
                    successful_count = sum(
                        1 for r in invocation_results if r.is_successful
                    )
                    total_count = len(invocation_results)

                    return EvaluationResult(
                        config={},
                        total_examples=total_count,
                        successful_examples=successful_count,
                        duration=1.0,
                        aggregated_metrics={
                            "accuracy": successful_count / max(1, total_count),
                            "success_rate": successful_count / max(1, total_count),
                        },
                    )

            invoker = MockInvoker()
            evaluator = MultiObjectiveEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=5,
            )

            assert result.best_config is not None
            assert result.best_score >= 0
            assert len(result.trials) == 5
            assert "pareto_frontier" in result.convergence_info
            assert "objectives" in result.convergence_info
            assert result.convergence_info["objectives"] == self.objectives

        asyncio.run(run_test())

    def test_dominates(self):
        """Test dominance relationship checking."""
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=["accuracy", "success_rate"],
            batch_config=self.batch_config,
        )

        # Test clear dominance
        scores1 = {"accuracy": 0.9, "success_rate": 0.8}
        scores2 = {"accuracy": 0.7, "success_rate": 0.6}
        assert optimizer._dominates(scores1, scores2) is True
        assert optimizer._dominates(scores2, scores1) is False

        # Test partial dominance (not dominated)
        scores3 = {"accuracy": 0.9, "success_rate": 0.5}
        scores4 = {"accuracy": 0.6, "success_rate": 0.8}
        assert optimizer._dominates(scores3, scores4) is False
        assert optimizer._dominates(scores4, scores3) is False

        # Test equal scores
        scores5 = {"accuracy": 0.8, "success_rate": 0.7}
        scores6 = {"accuracy": 0.8, "success_rate": 0.7}
        assert optimizer._dominates(scores5, scores6) is False

    def test_select_best_from_pareto_empty(self):
        """Test selecting best from empty Pareto frontier."""
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=self.objectives,
            batch_config=self.batch_config,
        )

        assert optimizer._select_best_from_pareto() is None


class TestAdaptiveBatchOptimizer:
    """Test suite for AdaptiveBatchOptimizer."""

    def setup_method(self):
        """Set up test data."""
        self.config_space = {"param1": [1, 2], "param2": [0.1, 0.2]}
        self.objectives = ["accuracy"]

        self.base_optimizer = RandomSearchOptimizer(self.config_space, self.objectives)
        self.batch_config = BatchOptimizationConfig(
            batch_size=2, adaptive_batching=True
        )

        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
                EvaluationExample(input_data={"value": 3}, expected_output="expected3"),
            ],
            name="test_dataset",
        )

    def test_init(self):
        """Test AdaptiveBatchOptimizer initialization."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert optimizer.base_optimizer == self.base_optimizer
        assert optimizer.batch_config == self.batch_config
        assert optimizer.adaptive_sizer is not None
        assert len(optimizer.performance_history) == 0

    def test_optimize_adaptive(self):
        """Test adaptive batch optimization."""

        async def run_test():
            optimizer = AdaptiveBatchOptimizer(
                base_optimizer=self.base_optimizer, batch_config=self.batch_config
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}_{config.custom_params.get('param1', 0)}"

            invoker = MockInvoker()
            evaluator = MockEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=3,
            )

            assert result.best_config is not None
            assert result.best_score >= 0
            assert len(result.trials) == 3
            assert result.duration > 0
            assert "adaptive_batching" in result.convergence_info
            assert result.convergence_info["adaptive_batching"] is True
            assert "final_batch_size" in result.convergence_info
            assert "performance_history" in result.convergence_info

        asyncio.run(run_test())

    def test_update_performance_history(self):
        """Test performance history updates."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        # Create mock trial
        from traigent.optimizers.results import Trial

        trial = Trial(
            configuration={"param1": 1},
            score=0.8,
            duration=1.0,
            metadata={
                "trial_index": 0,
                "batch_size": 2,
                "throughput": 3.0,
                "error_rate": 0.1,
            },
        )

        optimizer._update_performance_history(trial)

        assert len(optimizer.performance_history) == 1
        history_entry = optimizer.performance_history[0]
        assert history_entry["trial_index"] == 0
        assert history_entry["score"] == 0.8
        assert history_entry["batch_size"] == 2
        assert history_entry["throughput"] == 3.0
        assert history_entry["error_rate"] == 0.1

    def test_update_performance_history_failed_trial(self):
        """Test performance history with failed trial."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        # Create failed trial
        from traigent.optimizers.results import Trial

        failed_trial = Trial(
            configuration={"param1": 1},
            score=float("-inf"),
            duration=1.0,
            metadata={"error": "Failed", "failed": True},
        )

        optimizer._update_performance_history(failed_trial)

        # Should not add failed trials to history
        assert len(optimizer.performance_history) == 0

    def test_performance_history_size_limit(self):
        """Test performance history size limit."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        # Add many trials to exceed limit
        from traigent.optimizers.results import Trial

        for i in range(150):  # Exceed the 100 limit
            trial = Trial(
                configuration={"param1": 1},
                score=0.8,
                duration=1.0,
                metadata={
                    "trial_index": i,
                    "batch_size": 2,
                    "throughput": 3.0,
                    "error_rate": 0.1,
                },
            )
            optimizer._update_performance_history(trial)

        # Should maintain only last 100 entries
        assert len(optimizer.performance_history) == 100
        # Should have the most recent entries
        assert optimizer.performance_history[0]["trial_index"] == 50
        assert optimizer.performance_history[-1]["trial_index"] == 149
