"""Integration test for full optimization workflow."""

import json
from pathlib import Path

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample


class TestFullOptimizationWorkflow:
    """Test the complete optimization workflow end-to-end."""

    @pytest.fixture(autouse=True)
    def dataset_root(self, monkeypatch, tmp_path_factory):
        """Ensure dataset files live under the trusted root."""
        root = tmp_path_factory.mktemp("full_optimization_datasets")
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(root))
        return Path(root)

    @pytest.fixture
    def evaluation_dataset(self):
        """Create evaluation dataset for testing."""
        examples = [
            EvaluationExample({"x": 1, "multiplier": 2}, 2),
            EvaluationExample({"x": 2, "multiplier": 2}, 4),
            EvaluationExample({"x": 3, "multiplier": 2}, 6),
            EvaluationExample({"x": 4, "multiplier": 2}, 8),
            EvaluationExample({"x": 5, "multiplier": 2}, 10),
        ]
        return Dataset(examples, name="math_test")

    @pytest.fixture
    def jsonl_dataset_file(self, dataset_root: Path):
        """Create JSONL dataset file."""
        data = [
            {"input": {"x": 1, "multiplier": 2}, "output": 2},
            {"input": {"x": 2, "multiplier": 2}, "output": 4},
            {"input": {"x": 3, "multiplier": 2}, "output": 6},
            {"input": {"x": 4, "multiplier": 2}, "output": 8},
            {"input": {"x": 5, "multiplier": 2}, "output": 10},
        ]

        dataset_path = dataset_root / "math_dataset.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(item) for item in data), encoding="utf-8"
        )

        try:
            yield str(dataset_path)
        finally:
            dataset_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_optimization_with_dataset_object(self, evaluation_dataset):
        """Test optimization using Dataset object."""
        from traigent.config.types import TraigentConfig

        @traigent.optimize(
            eval_dataset=evaluation_dataset,
            objectives=["accuracy"],
            configuration_space={"strategy": ["multiply", "add"], "factor": [1, 2, 3]},
            default_config={"strategy": "multiply", "factor": 2},
            injection_mode="parameter",
        )
        def math_function(x: int, config: TraigentConfig) -> int:
            """Simple math function for testing."""
            strategy = config.custom_params.get("strategy", "multiply")
            factor = config.custom_params.get("factor", 1)

            if strategy == "multiply":
                return x * factor
            elif strategy == "add":
                return x + factor
            else:
                return x

        # Run optimization
        result = await math_function.optimize(
            algorithm="grid",
            max_trials=6,  # 2 * 3 = 6 combinations
        )

        # Verify results - should have some trials
        assert len(result.trials) > 0
        assert result.best_config is not None

    @pytest.mark.asyncio
    async def test_optimization_with_jsonl_file(self, jsonl_dataset_file):
        """Test optimization using JSONL file."""
        from traigent.config.types import TraigentConfig

        @traigent.optimize(
            eval_dataset=jsonl_dataset_file,
            objectives=["accuracy"],
            configuration_space={"operation": ["double", "triple", "half"]},
            injection_mode="parameter",
        )
        def number_processor(x: int, config: TraigentConfig) -> int:
            """Process numbers according to configuration."""
            operation = config.custom_params.get("operation", "double")

            if operation == "double":
                return x * 2
            elif operation == "triple":
                return x * 3
            elif operation == "half":
                return x // 2
            else:
                return x

        # Run optimization
        result = await number_processor.optimize(algorithm="grid", max_trials=3)

        # Verify results - should have some trials
        assert len(result.trials) > 0

    @pytest.mark.asyncio
    async def test_random_search_optimization(self, evaluation_dataset):
        """Test optimization using random search algorithm."""
        current_config = {"base": 0, "multiplier": 1}

        @traigent.optimize(
            eval_dataset=evaluation_dataset,
            objectives=["accuracy"],
            configuration_space={"base": [0, 1, 2], "multiplier": [1, 2, 3]},
        )
        def configurable_function(x: int, multiplier: int) -> int:
            """Function with configurable base and multiplier."""
            base = current_config.get("base", 0)
            mult = current_config.get("multiplier", 1)
            return (x + base) * mult

        # Patch evaluator
        async def patched_evaluate_single(
            evaluator_self, func, config, example, example_index
        ):
            nonlocal current_config
            old_config = current_config.copy()
            current_config.update(config)

            try:
                result = await original_method(
                    evaluator_self, func, config, example, example_index
                )
                return result
            finally:
                current_config = old_config

        from traigent.evaluators.local import LocalEvaluator

        original_method = LocalEvaluator._evaluate_single_detailed
        LocalEvaluator._evaluate_single_detailed = patched_evaluate_single

        try:
            # Run optimization with random search
            result = await configurable_function.optimize(
                algorithm="random",
                max_trials=5,
                random_seed=42,  # For reproducibility
            )

            # Verify results
            assert len(result.trials) == 5
            assert result.best_score >= 0
            assert result.algorithm == "RandomSearchOptimizer"

            # Should have tried different configurations
            configs = [trial.config for trial in result.trials]
            assert len({tuple(sorted(c.items())) for c in configs}) > 1

        finally:
            LocalEvaluator._evaluate_single_detailed = original_method

    def test_optimization_state_management(self, evaluation_dataset):
        """Test optimization state management methods."""

        @traigent.optimize(
            eval_dataset=evaluation_dataset, configuration_space={"param": [1, 2, 3]}
        )
        def test_function(x: int, multiplier: int) -> int:
            return x * 2

        # Initial state
        assert not test_function.is_optimization_complete()
        assert test_function.get_best_config() is None
        assert test_function.get_optimization_results() is None
        assert len(test_function.get_optimization_history()) == 0

        # Test reset (should not fail even when no optimization run)
        test_function.reset_optimization()
        assert not test_function.is_optimization_complete()

    def test_configuration_override(self, evaluation_dataset):
        """Test configuration override functionality."""

        @traigent.optimize(
            eval_dataset=evaluation_dataset,
            objectives=["accuracy"],
            configuration_space={"param": [1, 2, 3, 4, 5]},
        )
        def test_function(x: int, multiplier: int) -> int:
            return x * 2

        # Test override config creation
        override = traigent.override_config(
            objectives=["accuracy", "success_rate"], max_trials=3
        )

        assert override["objectives"] == ["accuracy", "success_rate"]
        assert override["max_trials"] == 3

    def test_strategy_configuration(self):
        """Test strategy configuration."""
        # Test valid strategy
        strategy = traigent.set_strategy(algorithm="grid", parallel_workers=2)

        assert strategy.algorithm == "grid"
        assert strategy.parallel_workers == 2

        # Test invalid algorithm
        with pytest.raises(ValueError, match="Unknown algorithm"):
            traigent.set_strategy(algorithm="nonexistent")

    def test_global_configuration(self):
        """Test global configuration management."""
        # Test configuration
        result = traigent.configure(logging_level="DEBUG", parallel_workers=4)

        assert result is True

        config = traigent.api.functions.get_global_config()
        assert config["logging_level"] == "DEBUG"
        assert config["parallel_workers"] == 4

        # Test invalid values
        with pytest.raises(ValueError):
            traigent.configure(parallel_workers=0)

        with pytest.raises(ValueError):
            traigent.configure(logging_level="INVALID")

    def test_version_info(self):
        """Test version information retrieval."""
        info = traigent.get_version_info()

        assert "version" in info
        assert "algorithms" in info
        assert "features" in info
        assert "integrations" in info
        assert info["version"] == "0.8.0"
        assert "grid" in info["algorithms"]
        assert "random" in info["algorithms"]

    def test_available_strategies(self):
        """Test available strategies information."""
        strategies = traigent.get_available_strategies()

        assert "grid" in strategies
        assert "random" in strategies

        grid_info = strategies["grid"]
        assert "name" in grid_info
        assert "description" in grid_info
        assert "supports_continuous" in grid_info
        assert "supports_categorical" in grid_info

        random_info = strategies["random"]
        assert random_info["supports_continuous"] is True
        assert random_info["supports_categorical"] is True
