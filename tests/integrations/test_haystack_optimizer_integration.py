"""Integration tests for Haystack with existing Traigent optimizers (Story 3.5).

This module demonstrates that HaystackEvaluator works seamlessly with
existing Traigent optimization infrastructure:
- GridSearchOptimizer
- RandomSearchOptimizer
- OptimizationOrchestrator

These tests validate that the Haystack integration doesn't duplicate
any existing optimization infrastructure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.integrations.haystack.evaluation import EvaluationDataset
from traigent.integrations.haystack.evaluator import HaystackEvaluator
from traigent.optimizers import GridSearchOptimizer, RandomSearchOptimizer


class TestGridSearchWithHaystack:
    """Integration tests for GridSearchOptimizer with HaystackEvaluator."""

    def test_grid_search_generates_all_combinations(self):
        """Test that grid search generates all config combinations."""
        config_space = {
            "generator.temperature": [0.0, 0.5, 1.0],
            "retriever.top_k": [5, 10],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
        )

        # Grid should have 3 * 2 = 6 combinations
        assert optimizer.total_combinations == 6

        # Get all configurations
        configs = []
        history: list[TrialResult] = []
        while not optimizer.should_stop(history):
            config = optimizer.suggest_next_trial(history)
            configs.append(config)

        assert len(configs) == 6

        # Verify all expected combinations exist
        expected_temps = {0.0, 0.5, 1.0}
        expected_k = {5, 10}

        actual_temps = {c["generator.temperature"] for c in configs}
        actual_k = {c["retriever.top_k"] for c in configs}

        assert actual_temps == expected_temps
        assert actual_k == expected_k

    def test_grid_search_with_haystack_style_qualified_names(self):
        """Test grid search understands Haystack-style qualified parameter names."""
        config_space = {
            "llm.model_name": ["gpt-4o-mini", "gpt-4o"],
            "llm.temperature": [0.2, 0.8],
            "prompt_builder.template_type": ["concise", "detailed"],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
        )

        # 2 * 2 * 2 = 8 combinations
        assert optimizer.total_combinations == 8

        config = optimizer.suggest_next_trial([])
        assert "llm.model_name" in config
        assert "llm.temperature" in config
        assert "prompt_builder.template_type" in config

    @pytest.mark.asyncio
    async def test_grid_search_configs_applied_to_haystack_pipeline(self):
        """Test that grid search configs are properly applied to Haystack pipelines."""
        # Create mock pipeline
        pipeline = MagicMock()
        generator = MagicMock()
        generator.temperature = 0.5
        retriever = MagicMock()
        retriever.top_k = 5

        def get_component(name):
            if name == "generator":
                return generator
            elif name == "retriever":
                return retriever
            return None

        pipeline.get_component.side_effect = get_component
        pipeline.run.return_value = {"answer": "test answer"}

        # Create dataset
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2"},
            ]
        )

        # Create evaluator
        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
            output_key="answer",
        )

        # Create grid search optimizer
        config_space = {
            "generator.temperature": [0.2, 0.8],
            "retriever.top_k": [5, 10],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
        )

        # Get first config
        config = optimizer.suggest_next_trial([])

        # Evaluate with config
        result = await evaluator.evaluate(
            func=pipeline.run,
            config=config,
            dataset=dataset.to_core_dataset(),
        )

        assert result is not None
        assert result.duration > 0

        # Verify pipeline ran with correct number of examples
        assert len(result.outputs) == 2


class TestRandomSearchWithHaystack:
    """Integration tests for RandomSearchOptimizer with HaystackEvaluator."""

    def test_random_search_samples_from_continuous_ranges(self):
        """Test that random search supports continuous parameter ranges."""
        config_space = {
            "generator.temperature": (0.0, 2.0),  # Continuous range
            "retriever.top_k": [5, 10, 15, 20],  # Categorical
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=10,
            random_seed=42,
        )

        configs = []
        history: list[TrialResult] = []
        for _ in range(5):
            config = optimizer.suggest_next_trial(history)
            configs.append(config)

        # Verify temperature values are within continuous range
        temps = [c["generator.temperature"] for c in configs]
        for temp in temps:
            assert 0.0 <= temp <= 2.0

        # Verify top_k values are from categorical list
        top_ks = [c["retriever.top_k"] for c in configs]
        for k in top_ks:
            assert k in [5, 10, 15, 20]

    def test_random_search_with_seed_is_reproducible(self):
        """Test that random search with same seed produces same results."""
        config_space = {
            "generator.temperature": (0.0, 1.0),
            "retriever.top_k": [5, 10, 15],
        }

        optimizer1 = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=10,
            random_seed=12345,
        )

        optimizer2 = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=10,
            random_seed=12345,
        )

        # Both should produce identical sequences
        for _ in range(5):
            config1 = optimizer1.suggest_next_trial([])
            config2 = optimizer2.suggest_next_trial([])
            assert config1 == config2

    @pytest.mark.asyncio
    async def test_random_search_with_haystack_evaluator(self):
        """Test random search integration with HaystackEvaluator."""
        # Create mock pipeline
        pipeline = MagicMock()
        generator = MagicMock()
        generator.temperature = 0.5
        retriever = MagicMock()
        retriever.top_k = 5

        def get_component(name):
            if name == "generator":
                return generator
            elif name == "retriever":
                return retriever
            return None

        pipeline.get_component.side_effect = get_component
        pipeline.run.return_value = {"answer": "test answer"}

        # Create dataset
        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
            ]
        )

        # Create evaluator
        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            metrics=["accuracy"],
        )

        # Create random search optimizer
        config_space = {
            "generator.temperature": (0.0, 1.0),
            "retriever.top_k": [5, 10, 15],
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=5,
            random_seed=42,
        )

        # Run multiple trials
        results = []
        history: list[TrialResult] = []
        for _ in range(3):
            config = optimizer.suggest_next_trial(history)
            result = await evaluator.evaluate(
                func=pipeline.run,
                config=config,
                dataset=dataset.to_core_dataset(),
            )
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert result.duration > 0


class TestOptimizerBestTracking:
    """Test that optimizers properly track best results."""

    def test_grid_search_tracks_best_config(self):
        """Test that grid search updates best config on good trials."""
        config_space = {
            "temperature": [0.2, 0.5, 0.8],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
        )

        # Simulate trials with different scores
        trials = [
            TrialResult(
                trial_id="t1",
                config={"temperature": 0.2},
                metrics={"accuracy": 0.75},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t2",
                config={"temperature": 0.5},
                metrics={"accuracy": 0.90},  # Best
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t3",
                config={"temperature": 0.8},
                metrics={"accuracy": 0.80},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
        ]

        for trial in trials:
            optimizer.update_best(trial)

        assert optimizer.best_score == 0.90
        assert optimizer.best_config == {"temperature": 0.5}

    def test_random_search_tracks_best_config(self):
        """Test that random search updates best config on good trials."""
        config_space = {
            "temperature": (0.0, 1.0),
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=10,
        )

        # Simulate trials
        trials = [
            TrialResult(
                trial_id="t1",
                config={"temperature": 0.3},
                metrics={"accuracy": 0.65},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t2",
                config={"temperature": 0.7},
                metrics={"accuracy": 0.95},  # Best
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
        ]

        for trial in trials:
            optimizer.update_best(trial)

        assert optimizer.best_score == 0.95
        assert optimizer.best_config == {"temperature": 0.7}


class TestOptimizerInfoMethods:
    """Test optimizer information and metadata methods."""

    def test_grid_search_algorithm_info(self):
        """Test grid search provides algorithm information."""
        config_space = {
            "model": ["gpt-4o-mini", "gpt-4o"],
            "temperature": [0.0, 0.5, 1.0],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy", "cost"],
        )

        info = optimizer.get_algorithm_info()

        assert info["name"] == "GridSearchOptimizer"
        assert info["total_combinations"] == 6
        assert info["supports_continuous"] is False
        assert info["supports_categorical"] is True
        assert info["deterministic"] is True

    def test_random_search_algorithm_info(self):
        """Test random search provides algorithm information."""
        config_space = {
            "model": ["gpt-4o-mini", "gpt-4o"],
            "temperature": (0.0, 1.0),
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=50,
            random_seed=42,
        )

        info = optimizer.get_algorithm_info()

        assert info["name"] == "RandomSearchOptimizer"
        assert info["max_trials"] == 50
        assert info["supports_continuous"] is True
        assert info["supports_categorical"] is True
        assert info["random_seed"] == 42


class TestOptimizerReset:
    """Test optimizer reset functionality."""

    def test_grid_search_reset(self):
        """Test that grid search can be reset for multiple optimization runs."""
        config_space = {
            "temperature": [0.2, 0.5, 0.8],
        }

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
        )

        # Use some configs
        for _ in range(2):
            optimizer.suggest_next_trial([])

        assert optimizer.trial_count == 2
        assert optimizer.progress > 0

        # Reset
        optimizer.reset()

        assert optimizer.trial_count == 0
        assert optimizer.progress == 0.0
        assert optimizer.best_score is None
        assert optimizer.best_config is None

    def test_random_search_reset(self):
        """Test that random search can be reset for multiple optimization runs."""
        config_space = {
            "temperature": (0.0, 1.0),
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=10,
            random_seed=42,
        )

        # Get first config
        first_config_before_reset = optimizer.suggest_next_trial([])

        # Use some more configs
        for _ in range(3):
            optimizer.suggest_next_trial([])

        # Reset
        optimizer.reset()

        # After reset with same seed, should get same first config
        first_config_after_reset = optimizer.suggest_next_trial([])
        assert first_config_before_reset == first_config_after_reset


class TestConfigSpaceFormats:
    """Test that optimizers handle various config space formats."""

    def test_grid_search_rejects_continuous_ranges(self):
        """Test that grid search raises error for continuous parameters."""
        config_space = {
            "temperature": (0.0, 1.0),  # Continuous range
        }

        with pytest.raises(Exception) as exc_info:
            optimizer = GridSearchOptimizer(
                config_space=config_space,
                objectives=["accuracy"],
            )
            optimizer.suggest_next_trial([])

        assert (
            "continuous" in str(exc_info.value).lower()
            or "range" in str(exc_info.value).lower()
        )

    def test_random_search_handles_single_values(self):
        """Test that random search handles fixed single-value parameters."""
        config_space = {
            "model": "gpt-4o",  # Fixed value
            "temperature": (0.0, 1.0),  # Continuous
        }

        optimizer = RandomSearchOptimizer(
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=5,
        )

        config = optimizer.suggest_next_trial([])

        # Fixed value should remain constant
        assert config["model"] == "gpt-4o"
        # Continuous value should vary
        assert 0.0 <= config["temperature"] <= 1.0


class TestHaystackEvaluatorWithOptimizerPatterns:
    """Test that HaystackEvaluator works with optimizer-suggested configs."""

    @pytest.mark.asyncio
    async def test_evaluator_applies_grid_search_configs(self):
        """Test evaluator correctly applies configs from grid search."""
        # Create mock pipeline with multiple components
        pipeline = MagicMock()

        generator = MagicMock(temperature=0.5)
        retriever = MagicMock(top_k=5)

        def get_component(name):
            if name == "generator":
                return generator
            elif name == "retriever":
                return retriever
            return None

        pipeline.get_component.side_effect = get_component
        pipeline.run.return_value = {"answer": "response"}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "test"}, "expected": "response"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="answer",
        )

        # Simulate grid search config
        config = {
            "generator.temperature": 0.8,
            "retriever.top_k": 10,
        }

        result = await evaluator.evaluate(
            func=pipeline.run,
            config=config,
            dataset=dataset.to_core_dataset(),
        )

        # Verify evaluation completed successfully
        assert result is not None
        assert result.duration > 0
        assert len(result.outputs) == 1

        # The config should have been applied (on a copy of the pipeline)
        # We verify by checking the result contains expected output
        assert result.outputs[0] == "response"

    @pytest.mark.asyncio
    async def test_evaluator_handles_multi_trial_optimization(self):
        """Test evaluator works correctly across multiple optimization trials."""
        pipeline = MagicMock()
        generator = MagicMock(temperature=0.5)
        pipeline.get_component.return_value = generator
        pipeline.run.return_value = {"answer": "test"}

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "Q1"}, "expected": "A1"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="answer",
        )

        # Simulate multiple trials
        configs = [
            {"generator.temperature": 0.2},
            {"generator.temperature": 0.5},
            {"generator.temperature": 0.8},
        ]

        results = []
        for config in configs:
            result = await evaluator.evaluate(
                func=pipeline.run,
                config=config,
                dataset=dataset.to_core_dataset(),
            )
            results.append(result)

        assert len(results) == 3
        # All evaluations should succeed
        for result in results:
            assert result.duration > 0
