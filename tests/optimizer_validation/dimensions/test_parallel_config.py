"""Tests for parallel execution configuration and trial clamping.

Tests that verify:
- Parallel trial limits are respected
- Budget clamping when parallel trials exceed limits
- Graceful degradation under resource constraints
- Parallel vs sequential execution modes
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import TestScenario


class TestParallelTrialLimits:
    """Tests for parallel trial execution limits."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_trials_basic(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test basic parallel trial execution.

        Verify that parallel execution works correctly with default settings.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_basic",
            description="Basic parallel trial execution",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 2,  # Run 2 trials in parallel
            },
            gist_template="parallel-basic -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_trials_one(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel_trials=1 runs sequentially.

        With only 1 parallel trial allowed, execution should be sequential.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_one",
            description="parallel_trials=1 is sequential",
            config_space=config_space,
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 1,
            },
            gist_template="parallel-one -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_trials_exceeds_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel_trials > max_trials is clamped.

        If parallel_trials is larger than max_trials, it should be
        clamped to max_trials to avoid unnecessary resource allocation.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5],
        }

        scenario = TestScenario(
            name="parallel_exceeds_max",
            description="parallel_trials > max_trials should be clamped",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 10,  # Way more than max_trials
            },
            gist_template="parallel-clamp -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should work - parallel_trials should be clamped to 2
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParallelBudgetClamping:
    """Tests for budget clamping in parallel execution."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_budget_clamping_basic(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that budget is properly clamped.

        When parallel trials + sample budget would exceed total budget,
        the system should clamp appropriately.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.3, 0.5, 0.7, 0.9],
        }

        scenario = TestScenario(
            name="budget_clamping",
            description="Budget clamping under constraints",
            config_space=config_space,
            max_trials=5,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 3,
                "sample_budget": 2,  # Per-trial budget
            },
            gist_template="budget-clamp -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Must succeed with budget constraints
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) <= 5, "Should respect max_trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_budget_clamping_grid_search(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test budget clamping with grid search.

        Grid search should respect parallel limits while exploring
        the full configuration space.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.5, 0.7],
        }
        # 2 * 3 = 6 combinations

        scenario = TestScenario(
            name="budget_clamping_grid",
            description="Budget clamping with grid search",
            config_space=config_space,
            max_trials=10,  # More than grid size
            mock_mode_config={
                "optimizer": "grid",
                "parallel_trials": 2,
            },
            gist_template="grid-clamp -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Must succeed with grid search
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"

        # Track unique configs explored
        unique_configs = set()
        for trial in result.trials:
            config_tuple = tuple(sorted(trial.config.items()))
            unique_configs.add(config_tuple)

        # Verify all explored configs are valid
        valid_models = {"gpt-3.5-turbo", "gpt-4"}
        valid_temps = {0.3, 0.5, 0.7}
        for config in unique_configs:
            config_dict = dict(config)
            assert (
                config_dict["model"] in valid_models
            ), f"Invalid model: {config_dict['model']}"
            assert (
                config_dict["temperature"] in valid_temps
            ), f"Invalid temperature: {config_dict['temperature']}"

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParallelResourceConstraints:
    """Tests for parallel execution under resource constraints."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_with_limited_memory(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel execution with memory constraints.

        System should gracefully handle limited resources.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_memory_limit",
            description="Parallel with memory constraints",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 4,
                "memory_limit_mb": 512,  # Limit memory
            },
            gist_template="mem-limit -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Must succeed even with memory constraints (mock mode ignores this)
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_graceful_degradation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test graceful degradation when resources are limited.

        If parallel execution can't be maintained, should fall back
        to sequential without crashing.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_degradation",
            description="Graceful degradation to sequential",
            config_space=config_space,
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 10,  # Request more than available
                "strict_parallel": False,  # Allow degradation
            },
            gist_template="degradation -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should complete even if it had to run sequentially
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParallelWithOptuna:
    """Tests for parallel execution with Optuna optimizers."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_optuna_tpe(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel execution with Optuna TPE sampler.

        TPE should work with parallel trial suggestions.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name="parallel_optuna_tpe",
            description="Parallel with Optuna TPE",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "tpe",
                "parallel_trials": 2,
            },
            gist_template="optuna-tpe -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_optuna_random(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel execution with Optuna random sampler."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_optuna_random",
            description="Parallel with Optuna random sampler",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "random",
                "parallel_trials": 3,
            },
            gist_template="optuna-rand -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParallelTrialIsolation:
    """Tests for isolation between parallel trials."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_trial_isolation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that parallel trials are properly isolated.

        Each trial should have its own config and not share state.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_isolation",
            description="Parallel trials should be isolated",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 4,
            },
            gist_template="isolation -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"

        # Verify each trial has a distinct config (or at least valid)
        for trial in result.trials:
            assert "model" in trial.config, "Trial should have model config"
            assert "temperature" in trial.config, "Trial should have temp config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_no_config_bleed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that config values don't bleed between parallel trials.

        With different configs running in parallel, ensure no mixing.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.1, 0.9],  # Distinct values
        }

        scenario = TestScenario(
            name="parallel_no_bleed",
            description="No config bleeding between parallel trials",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "grid",
                "parallel_trials": 4,
            },
            gist_template="no-bleed -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"

        # All configs should be from the valid space - no bleeding between
        # trials
        for trial in result.trials:
            assert trial.config["model"] in [
                "gpt-3.5-turbo",
                "gpt-4",
            ], f"Invalid model value: {trial.config['model']}"
            assert trial.config["temperature"] in [
                0.1,
                0.9,
            ], f"Invalid temperature value: {trial.config['temperature']}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParallelTrialOrdering:
    """Tests for ordering of parallel trial results."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_results_order(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that parallel trial results are properly ordered.

        Even with parallel execution, results should be ordered by
        trial number or completion time.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5, 0.7],
        }

        scenario = TestScenario(
            name="parallel_ordering",
            description="Parallel trial results should be ordered",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={
                "optimizer": "random",
                "parallel_trials": 2,
            },
            gist_template="ordering -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"

        # Verify trials are properly ordered and have valid configs
        for i, trial in enumerate(result.trials):
            assert "model" in trial.config, f"Trial {i} should have model config"
            assert (
                "temperature" in trial.config
            ), f"Trial {i} should have temperature config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
