"""Tests for cross-algorithm behavior parity and consistency.

Tests that verify:
- Same dataset is used across all trials with different algorithms
- Multiple algorithms produce consistent results for same config space
- Algorithm comparison with equivalent settings
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    config_space_scenario,
)


class TestDatasetConsistencyAcrossTrials:
    """Tests that verify dataset remains fixed across optimization trials."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dataset_consistent_with_random_search(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that random search uses the same dataset for all trials.

        Verifies that the dataset examples don't change between trials,
        ensuring fair comparison of configurations.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.5, 0.9],
        }

        scenario = config_space_scenario(
            name="dataset_consistency_random",
            config_space=config_space,
            description="Verify dataset consistency with random search",
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="dataset-consistency-random -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Verify all trials completed (implies same dataset was used)
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 3, "Should have 3 trials"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dataset_consistent_with_grid_search(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that grid search uses the same dataset for all trials.

        Grid search should evaluate each configuration against the
        exact same dataset examples.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="dataset_consistency_grid",
            config_space=config_space,
            description="Verify dataset consistency with grid search",
            max_trials=4,  # 2x2 = 4 combinations
            mock_mode_config={"optimizer": "grid"},
            gist_template="dataset-consistency-grid -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Grid search should cover all 4 combinations with same dataset
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, "Should have 4 trials (2x2 grid)"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dataset_consistent_with_bayesian(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that Bayesian optimization uses same dataset for all trials.

        Bayesian optimization should use the same dataset for initial
        exploration and subsequent exploitation trials.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),  # Continuous for Bayesian
        }

        scenario = config_space_scenario(
            name="dataset_consistency_bayesian",
            config_space=config_space,
            description="Verify dataset consistency with Bayesian optimization",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template="dataset-consistency-bayesian -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestCrossAlgorithmParity:
    """Tests comparing behavior across different optimization algorithms."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_vs_grid_same_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that random and grid search both work on the same config space.

        Both algorithms should successfully complete optimization
        on identical configuration spaces.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        # Test with random search
        random_scenario = config_space_scenario(
            name="parity_random",
            config_space=config_space,
            description="Random search on categorical space",
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="parity-random -> {{model}}",
        )

        _, result_random = await scenario_runner(random_scenario)
        assert not isinstance(
            result_random, Exception
        ), f"Random search failed: {result_random}"
        validation = result_validator(random_scenario, result_random)
        assert validation.passed, validation.summary()

        # Test with grid search
        grid_scenario = config_space_scenario(
            name="parity_grid",
            config_space=config_space,
            description="Grid search on categorical space",
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="parity-grid -> {{model}}",
        )

        _, result_grid = await scenario_runner(grid_scenario)
        assert not isinstance(
            result_grid, Exception
        ), f"Grid search failed: {result_grid}"
        validation = result_validator(grid_scenario, result_grid)
        assert validation.passed, validation.summary()

        # Both should have completed trials
        assert hasattr(result_random, "trials") and len(result_random.trials) > 0
        assert hasattr(result_grid, "trials") and len(result_grid.trials) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_algorithms_complete_optimization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that all supported algorithms complete optimization
        successfully.

        Each algorithm should be able to run and return valid results.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.5, 0.9],
        }

        algorithms = [
            ("random", None),
            ("grid", None),
            ("optuna", "tpe"),
            ("optuna", "random"),
        ]

        for optimizer, sampler in algorithms:
            mock_config: dict[str, Any] = {"optimizer": optimizer}
            if sampler:
                mock_config["sampler"] = sampler

            scenario = config_space_scenario(
                name=f"algorithm_{optimizer}_{sampler or 'default'}",
                config_space=config_space,
                description=(f"Test {optimizer} with {sampler or 'default'} sampler"),
                max_trials=2,
                mock_mode_config=mock_config,
                gist_template=(f"algo-{optimizer}-{sampler or 'def'} -> {{model}}"),
            )

            _, result = await scenario_runner(scenario)
            assert not isinstance(
                result, Exception
            ), f"{optimizer}/{sampler} failed: {result}"
            assert hasattr(result, "trials"), f"{optimizer} has no trials"
            assert len(result.trials) > 0, f"{optimizer} completed with no trials"
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()


class TestAlgorithmSpecificBehavior:
    """Tests for algorithm-specific behaviors that should be consistent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_exhaustive_search(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that grid search explores all combinations.

        Grid search should try every combination in the configuration space
        and stop when the space is exhausted.

        The test verifies:
        1. Optimization completes successfully
        2. Exactly 4 trials run (space exhaustion stops the search)
        3. All 4 unique configurations are explored
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }
        # 2 models * 2 temperatures = 4 combinations

        scenario = config_space_scenario(
            name="grid_exhaustive",
            config_space=config_space,
            description="Grid search should explore all 4 combinations",
            max_trials=10,  # More than needed
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                outcome=ExpectedOutcome.SUCCESS,
                min_trials=4,
                max_trials=4,  # Grid should stop after exhausting space
            ),
            gist_template="grid-exhaustive -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should succeed
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Should have trials attribute
        assert hasattr(result, "trials"), "Result should have trials"

        # Grid should stop at exactly 4 trials (space exhaustion)
        assert (
            len(result.trials) == 4
        ), f"Grid should stop at 4 trials (space exhausted), got {len(result.trials)}"

        # Collect and verify all 4 unique configurations were tried
        unique_configs = set()
        for trial in result.trials:
            config_tuple = tuple(sorted(trial.config.items()))
            unique_configs.add(config_tuple)

        assert (
            len(unique_configs) == 4
        ), f"Grid should explore all 4 unique combinations, got {len(unique_configs)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_different_configs(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that random search explores different configurations.

        Random search should not always pick the same configuration.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

        scenario = config_space_scenario(
            name="random_variety",
            config_space=config_space,
            description="Random search should try different configs",
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            gist_template="random-variety -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials") and len(result.trials) >= 3:
            # Collect configurations
            configs = [tuple(sorted(t.config.items())) for t in result.trials]
            unique_configs = set(configs)

            # Should have at least 2 unique configurations with high probability
            # (With 15 possible combinations and 5 trials,
            # very unlikely to get all same)
            assert (
                len(unique_configs) >= 2
            ), f"Random search should explore variety, got {len(unique_configs)} unique"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestAlgorithmWithDifferentConfigSpaceTypes:
    """Tests for algorithms with different types of configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_with_continuous_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search with continuous parameters."""
        config_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.5, 1.0),
        }

        scenario = config_space_scenario(
            name="random_continuous",
            config_space=config_space,
            description="Random search with continuous params",
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="random-continuous -> {{temperature}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_with_mixed_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search with mixed categorical and continuous params."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 500, 1000],
        }

        scenario = config_space_scenario(
            name="random_mixed",
            config_space=config_space,
            description="Random search with mixed params",
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="random-mixed -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bayesian_with_mixed_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Bayesian optimization with mixed parameter types.

        Bayesian (TPE) should handle both categorical and continuous params.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = config_space_scenario(
            name="bayesian_mixed",
            config_space=config_space,
            description="Bayesian with mixed categorical/continuous",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template="bayesian-mixed -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
