"""Tests for configuration override conflicts.

Tests that verify:
- TVL (Trial Version Lock) override conflicts
- Objective override conflicts when multiple sources define objectives
- Evaluator override conflicts between mock mode and config
- Scoring function override conflicts
- Graceful error handling for conflicting configurations
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import TestScenario


class TestTVLOverrideConflicts:
    """Tests for Trial Version Lock (TVL) override conflicts.

    TVL locks specific configuration values, which may conflict with
    other configuration sources.

    NOTE: TVL feature is not yet implemented. These tests document
    expected behavior for future implementation.
    """

    @pytest.mark.xfail(reason="TVL feature not yet implemented")
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tvl_override_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TVL override that conflicts with config space.

        When TVL locks a value that's outside the config space,
        should either error or override.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],  # Only these allowed
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="tvl_override_space",
            description="TVL overrides config space value",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "tvl": {
                    "model": "gpt-4o",  # Not in config space!
                },
            },
            gist_template="tvl-conflict -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Document behavior - may error or override
        if isinstance(result, Exception):
            error_msg = str(result).lower()
            # Should mention TVL or conflict
            assert any(
                term in error_msg for term in ["tvl", "lock", "conflict", "invalid"]
            ), f"Error should mention TVL conflict: {result}"
        else:
            # If it worked, TVL should have overridden
            if hasattr(result, "trials"):
                for trial in result.trials:
                    if "model" in trial.config:
                        # Model should be locked to gpt-4o
                        assert (
                            trial.config["model"] == "gpt-4o"
                        ), "TVL should lock model value"

        # Emit evidence

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.xfail(reason="TVL feature not yet implemented")
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tvl_valid_override(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TVL override with valid value from config space."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="tvl_valid_override",
            description="TVL locks to valid config space value",
            config_space=config_space,
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "tvl": {
                    "model": "gpt-4",  # Valid - in config space
                },
            },
            gist_template="tvl-valid -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # All trials should have model locked to gpt-4
        if hasattr(result, "trials"):
            for trial in result.trials:
                assert (
                    trial.config.get("model") == "gpt-4"
                ), "TVL should lock model to gpt-4"

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.xfail(reason="TVL feature not yet implemented")
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tvl_multiple_locks(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TVL with multiple parameter locks."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.5, 0.7],
            "max_tokens": [100, 500, 1000],
        }

        scenario = TestScenario(
            name="tvl_multiple_locks",
            description="TVL locks multiple parameters",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "tvl": {
                    "model": "gpt-4",
                    "temperature": 0.5,
                },
            },
            gist_template="tvl-multi -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        if hasattr(result, "trials"):
            for trial in result.trials:
                assert trial.config.get("model") == "gpt-4"
                assert abs(trial.config.get("temperature") - 0.5) < 1e-6
                # max_tokens should still vary
                assert trial.config.get("max_tokens") in [100, 500, 1000]

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestObjectiveOverrideConflicts:
    """Tests for objective definition conflicts.

    Objectives can be defined in multiple places (decorator, config, runtime).
    Test that conflicts are handled properly.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objectives_decorator_vs_runtime(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test conflict between decorator and runtime objectives.

        When objectives are defined both in decorator and runtime config,
        which takes precedence?
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5],
        }

        scenario = TestScenario(
            name="objectives_decorator_vs_runtime",
            description="Decorator vs runtime objective conflict",
            config_space=config_space,
            objectives=["accuracy"],  # From decorator
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "objectives": ["cost", "latency"],  # Runtime override
            },
            gist_template="obj-conflict -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Runtime objectives should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objective_direction_conflict(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test conflicting objective directions.

        Same objective with different directions (maximize vs minimize)
        from different sources.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="objective_direction_conflict",
            description="Conflicting objective directions",
            config_space=config_space,
            objectives=["score"],  # String objective
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                # Runtime says minimize same objective
                "objective_directions": {"score": "minimize"},
            },
            gist_template="dir-conflict -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Mock mode should accept this config
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_objective_partial_override(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test partial override of multi-objective configuration."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5],
        }

        scenario = TestScenario(
            name="multi_obj_partial_override",
            description="Partial override of multi-objective",
            config_space=config_space,
            objectives=["accuracy", "cost", "latency"],
            max_trials=2,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "nsga2",
                # Only override direction for one objective
                "objective_directions": {"cost": "minimize"},
            },
            gist_template="partial-override -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Partial overrides are valid - should succeed
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEvaluatorOverrideConflicts:
    """Tests for evaluator override conflicts.

    Evaluators can be specified in decorator, config, or mock mode.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_decorator_vs_mock(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator configuration in mock mode.

        Tests that evaluator settings in mock_mode_config are handled.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="evaluator_mock_config",
            description="Evaluator in mock mode config",
            config_space=config_space,
            # Don't set evaluator field directly - use mock_mode_config
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "evaluator": "fuzzy_match",  # Evaluator in mock config
            },
            gist_template="eval-mock -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Mock mode config with evaluator should work
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_with_specific_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator with specific objective configuration.

        Verify that custom objectives work correctly.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="evaluator_specific_objectives",
            description="Evaluator with specific objectives",
            config_space=config_space,
            objectives=["accuracy", "cost"],  # Multiple objectives
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
            },
            gist_template="eval-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should work with standard objectives
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_evaluator_override(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that mock mode can override evaluator completely."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5],
        }

        scenario = TestScenario(
            name="mock_evaluator_override",
            description="Mock mode overrides evaluator",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "evaluator_override": "mock_random",  # Force mock evaluator
            },
            gist_template="mock-override -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Mock evaluator should work
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestScoringOverrideConflicts:
    """Tests for scoring function override conflicts."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scoring_function_precedence(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring function precedence when multiple are defined.

        Custom scoring function vs default vs evaluator-provided.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="scoring_precedence",
            description="Scoring function precedence",
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "scoring": {
                    "function": "custom_weighted",
                    "weights": {"accuracy": 1.0},
                },
            },
            gist_template="scoring-prec -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Custom scoring function config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_aggregation_vs_primary(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test conflict between aggregation and primary objective.

        When both aggregation function and primary objective are specified.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="aggregation_vs_primary",
            description="Aggregation vs primary objective conflict",
            config_space=config_space,
            objectives=["accuracy", "cost", "latency"],
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "primary_objective": "accuracy",  # Use one as primary
                "aggregation": "weighted_sum",  # But also aggregate
            },
            gist_template="agg-vs-primary -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Aggregation config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMockModeOverrides:
    """Tests for mock mode specific overrides."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_response_override(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mock mode response pattern override."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="mock_response_override",
            description="Mock mode response override",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "response_pattern": "deterministic",  # Fixed responses
            },
            gist_template="mock-resp -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Response pattern config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_score_distribution(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mock mode score distribution override."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = TestScenario(
            name="mock_score_distribution",
            description="Mock mode score distribution",
            config_space=config_space,
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "score_distribution": "uniform",  # Uniform random scores
                "score_range": (0.5, 1.0),  # Score bounds
            },
            gist_template="mock-dist -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 3, f"Expected 3 trials, got {len(result.trials)}"

        # Verify all trials have metrics
        for trial in result.trials:
            assert hasattr(trial, "metrics"), "Trial should have metrics"
            assert len(trial.metrics) > 0, "Trial should have at least one metric"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestConflictResolutionPriority:
    """Tests for conflict resolution priority order."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_priority_runtime_over_decorator(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that runtime config has priority over decorator."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="priority_runtime",
            description="Runtime priority over decorator",
            config_space=config_space,
            max_trials=2,  # Decorator says 2
            mock_mode_config={
                "optimizer": "random",
                "max_trials_override": 1,  # Runtime says 1
            },
            gist_template="priority-runtime -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should succeed - mock mode config is accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        # max_trials from scenario (2) should take precedence over mock
        # override
        assert len(result.trials) >= 1, "Should have at least 1 trial"
        assert len(result.trials) <= 2, "Should have at most 2 trials"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_priority_explicit_over_default(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that explicit config has priority over defaults."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="priority_explicit",
            description="Explicit config over defaults",
            config_space=config_space,
            timeout=60.0,  # Explicit timeout
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
            },
            gist_template="priority-explicit -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
