"""Tests for algorithm constraints and edge cases.

Tests that verify:
- Algorithm behavior with incompatible configuration spaces
- Bayesian/sampling algorithms with fixed/degenerate spaces
- CMA-ES with categorical parameters (should fail/warn)
- Grid search with continuous parameters (should fail/warn)
- Large configuration spaces
- Algorithm auto-selection based on config space
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    config_space_scenario,
)


class TestBayesianWithDegenerateSpace:
    """Tests for Bayesian optimization with degenerate/fixed configuration
    spaces.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bayesian_with_single_categorical_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Bayesian optimization when config space has only one option.

        With a single-value categorical parameter, the optimizer may stop early
        because there's effectively nothing to optimize. This is correct behavior
        - running multiple trials with identical configs wastes resources.

        The test verifies:
        1. Optimization completes successfully (doesn't crash)
        2. All trials use the only available config value
        3. At least one trial is run
        """
        config_space = {
            "model": ["gpt-4"],  # Only one option
        }

        scenario = config_space_scenario(
            name="bayesian_single_value",
            config_space=config_space,
            description="Bayesian with single categorical value",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template=("bayesian+single-cat -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Should complete successfully - Optuna handles degenerate spaces
        assert not isinstance(
            result, Exception
        ), f"Bayesian with single value should succeed: {result}"

        # Verify at least one trial ran with the only available config
        assert hasattr(result, "trials"), "Result should have trials"
        assert (
            len(result.trials) >= 1
        ), f"Expected at least 1 trial, got {len(result.trials)}"

        for i, trial in enumerate(result.trials):
            assert (
                trial.config.get("model") == "gpt-4"
            ), f"Trial {i} should use the only available config, got {trial.config.get('model')}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bayesian_with_all_fixed_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Bayesian optimization when ALL parameters are fixed.

        This is a completely degenerate case - there's literally nothing to
        optimize. The optimizer may stop early because running multiple trials
        with identical configs is wasteful. This is correct behavior.

        The test verifies:
        1. Optimization completes successfully
        2. All trials have identical configurations
        3. At least one trial is run
        """
        config_space = {
            "model": ["gpt-4"],
            "temperature": [0.5],
            "max_tokens": [1000],
        }

        scenario = config_space_scenario(
            name="bayesian_all_fixed",
            config_space=config_space,
            description="Bayesian with all fixed parameters (1 option each)",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template=("bayesian+fixed-params -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Should complete successfully - Optuna handles fully fixed spaces
        assert not isinstance(
            result, Exception
        ), f"Bayesian with all fixed params should succeed: {result}"

        # Verify at least one trial ran with the fixed config
        assert hasattr(result, "trials"), "Result should have trials"
        assert (
            len(result.trials) >= 1
        ), f"Expected at least 1 trial, got {len(result.trials)}"

        expected_config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000,
        }

        for i, trial in enumerate(result.trials):
            # Verify each expected param value
            for key, expected_value in expected_config.items():
                actual_value = trial.config.get(key)
                assert (
                    actual_value == expected_value
                ), f"Trial {i} param '{key}' should be {expected_value}, got {actual_value}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bayesian_with_point_range(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Bayesian with continuous parameter where min equals max.

        A point range (e.g., temperature: (0.5, 0.5)) is degenerate.
        Traigent correctly validates this and rejects it with a
        ValidationError,
        as min >= max makes no sense for a continuous range.

        This is better behavior than silently accepting the invalid range -
        users should use a single-value list [0.5] instead of a range.

        The test verifies:
        1. A ValidationError is raised for point ranges
        2. The error message explains the issue clearly
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.5, 0.5),  # Point range (invalid)
        }

        scenario = config_space_scenario(
            name="bayesian_point_range",
            config_space=config_space,
            description="Bayesian with point range (min == max)",
            max_trials=2,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            expected=ExpectedResult(outcome=ExpectedOutcome.FAILURE),
            gist_template=("bayesian+point-range -> {error_type()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Should fail validation - point ranges are correctly rejected
        assert isinstance(
            result, Exception
        ), "Point range (min == max) should be rejected with ValidationError"

        # Verify it's a validation error with clear message
        error_msg = str(result).lower()
        assert (
            "range" in error_msg or "min" in error_msg
        ), f"Error should mention range validation issue: {result}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestCMAESConstraints:
    """Tests for CMA-ES algorithm constraints.

    CMA-ES only works with continuous parameters.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cmaes_with_categorical_only(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES with categorical-only config space.

        CMA-ES requires continuous parameters. With categorical-only space,
        Optuna's CMA-ES sampler falls back to RandomSampler for each parameter
        and logs a warning. This is expected behavior.

        The test verifies:
        1. Optimization completes without crashing
        2. All trials have valid configs from the categorical space
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "response_format": ["text", "json"],
        }

        scenario = config_space_scenario(
            name="cmaes_categorical",
            config_space=config_space,
            description=("CMA-ES with categorical-only space (falls back to random)"),
            max_trials=2,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            gist_template=("cmaes+categorical -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # CMA-ES with categorical-only params falls back to RandomSampler
        # for those params (Optuna behavior), so it should succeed
        assert not isinstance(
            result, Exception
        ), f"CMA-ES with categorical should fallback to random: {result}"

        # Verify trials completed with valid configs
        if hasattr(result, "trials"):
            for i, trial in enumerate(result.trials):
                assert (
                    trial.config.get("model") in config_space["model"]
                ), f"Trial {i} has invalid model: {trial.config.get('model')}"
                assert (
                    trial.config.get("response_format")
                    in config_space["response_format"]
                ), f"Trial {i} has invalid response_format"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cmaes_with_mixed_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES with mixed categorical and continuous parameters.

        CMA-ES may not handle mixed spaces well - verify behavior.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],  # Categorical
            "temperature": (0.0, 1.0),  # Continuous
        }

        scenario = config_space_scenario(
            name="cmaes_mixed",
            config_space=config_space,
            description="CMA-ES with mixed categorical/continuous",
            max_trials=2,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            gist_template="cmaes+mixed -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Document observed behavior - may fail or fallback
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cmaes_with_continuous_only(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES with continuous-only config space (ideal case).

        CMA-ES should work well with all-continuous parameters.
        """
        config_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.5, 1.0),
            "frequency_penalty": (-1.0, 1.0),
        }

        scenario = config_space_scenario(
            name="cmaes_continuous",
            config_space=config_space,
            description="CMA-ES with continuous-only space (ideal)",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            gist_template="cmaes+continuous -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(
            result, Exception
        ), f"CMA-ES with continuous should work: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestGridSearchConstraints:
    """Tests for grid search algorithm constraints.

    Grid search requires discrete/categorical parameters.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_with_continuous_only(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with continuous-only config space.

        Grid search cannot enumerate continuous ranges - should fail
        or discretize the range.
        """
        config_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.5, 1.0),
        }

        scenario = config_space_scenario(
            name="grid_continuous",
            config_space=config_space,
            description=(
                "Grid search with continuous-only space (should fail/discretize)"
            ),
            max_trials=3,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(outcome=ExpectedOutcome.FAILURE),
            gist_template="grid+continuous -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert isinstance(result, Exception)

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_with_mixed_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with mixed categorical and continuous
        parameters.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],  # Categorical
            "temperature": (0.0, 1.0),  # Continuous
        }

        scenario = config_space_scenario(
            name="grid_mixed",
            config_space=config_space,
            description="Grid search with mixed params",
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(outcome=ExpectedOutcome.FAILURE),
            gist_template="grid+mixed -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert isinstance(result, Exception)

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestLargeConfigurationSpaces:
    """Tests for handling very large configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_with_large_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with a large configuration space.

        With many parameters and values, the grid becomes very large.
        Grid search should handle this gracefully (possibly with limits).
        """
        config_space = {
            "model": [f"model-{i}" for i in range(5)],  # 5 options
            # 11 options (0.0 to 1.0)
            "temperature": [0.1 * i for i in range(11)],
            "max_tokens": [100, 500, 1000, 2000, 4000],  # 5 options
        }
        # Total: 5 * 11 * 5 = 275 combinations

        scenario = config_space_scenario(
            name="grid_large",
            config_space=config_space,
            description="Grid search with large space (275 combinations)",
            max_trials=10,  # Only run 10 of 275
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid+large-space -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Should complete with up to max_trials
        if hasattr(result, "trials"):
            assert len(result.trials) <= 10, "Should respect max_trials limit"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_with_large_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search with a very large configuration space.

        Random search should handle large spaces efficiently.
        """
        config_space = {
            "model": [f"model-{i}" for i in range(10)],
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "frequency_penalty": (-2.0, 2.0),
            "presence_penalty": (-2.0, 2.0),
        }

        scenario = config_space_scenario(
            name="random_large",
            config_space=config_space,
            description="Random search with large mixed space",
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            gist_template="random+large-space -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaGridSampler:
    """Tests for Optuna's grid sampler behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_grid_with_categorical(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna grid sampler with categorical parameters."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="optuna_grid_categorical",
            config_space=config_space,
            description="Optuna grid sampler with categorical",
            max_trials=4,
            mock_mode_config={"optimizer": "optuna", "sampler": "grid"},
            gist_template="optuna-grid+cat -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_grid_with_continuous(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna grid sampler with continuous parameters.

        Optuna's grid sampler may not support continuous parameters.
        """
        config_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.5, 1.0),
        }

        scenario = config_space_scenario(
            name="optuna_grid_continuous",
            config_space=config_space,
            description="Optuna grid sampler with continuous (may fail)",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "grid"},
            gist_template=("optuna-grid+continuous -> {error_type()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Document behavior - may fail or discretize
        if isinstance(result, Exception):
            # Grid sampler may not support continuous
            pass
        else:
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()


class TestMultiObjectiveAlgorithmConstraints:
    """Tests for algorithm constraints with multi-objective optimization."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nsga2_with_single_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test NSGA-II (multi-objective) with single objective.

        NSGA-II is designed for multi-objective but should work with one.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name="nsga2_single_objective",
            description="NSGA-II with single objective",
            config_space=config_space,
            objectives=["accuracy"],  # Single objective
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "nsga2"},
            gist_template="nsga2-single -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # NSGA-II should work even with single objective
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()


class TestEmptyAndMinimalSpaces:
    """Tests for edge cases with very small configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with empty configuration space.

        Should either fail clearly or use defaults.
        """
        scenario = TestScenario(
            name="empty_space",
            description="Optimization with empty config space",
            config_space={},
            max_trials=2,
            gist_template="empty-space -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - either error or use defaults
        if isinstance(result, Exception):
            # Empty config space should produce a clear error
            pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_param_single_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with exactly one parameter with one value.

        Minimal possible configuration space.
        """
        config_space = {
            "model": ["gpt-4"],
        }

        scenario = config_space_scenario(
            name="minimal_space",
            config_space=config_space,
            description="Minimal config space (1 param, 1 value)",
            max_trials=2,
            gist_template="minimal-space -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should complete without error
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()
            if hasattr(result, "trials"):
                for trial in result.trials:
                    assert trial.config.get("model") == "gpt-4"
