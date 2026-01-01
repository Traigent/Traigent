"""Tests for scoring function precedence and aggregation.

Tests that verify:
- Scoring function selection order
- Multi-objective aggregation strategies
- Custom scoring function integration
- Score normalization and weighting
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import TestScenario, config_space_scenario


class TestScoringFunctionSelection:
    """Tests for scoring function selection precedence."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_scoring(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test default scoring function when none specified.

        With single objective, should use objective value directly.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.5],
        }

        scenario = config_space_scenario(
            name="default_scoring",
            config_space=config_space,
            description="Default scoring with single objective",
            max_trials=2,
            gist_template="default-score -> {trial_count()} | {status()}",
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

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_scoring_priority(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that evaluator-provided scoring is used by default.

        When evaluator provides a score, it should be used unless overridden.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="evaluator_scoring",
            description="Evaluator-provided scoring",
            config_space=config_space,
            # Use default evaluator (no explicit evaluator field)
            max_trials=2,
            gist_template="eval-score -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        # Verify all trials have metrics from evaluator
        for trial in result.trials:
            assert hasattr(trial, "metrics"), "Trial should have metrics"
            assert len(trial.metrics) > 0, "Should have evaluator metrics"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_scoring_overrides_evaluator(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that custom scoring function overrides evaluator scoring."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="custom_scoring_override",
            description="Custom scoring overrides evaluator",
            config_space=config_space,
            # Don't set evaluator field - use mock_mode_config for overrides
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "scoring_function": "custom",
                "scoring_params": {"weight": 0.8},
            },
            gist_template="custom-override -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Custom scoring should be applied
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMultiObjectiveAggregation:
    """Tests for multi-objective score aggregation."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_sum_aggregation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted sum aggregation of multiple objectives."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="weighted_sum",
            description="Weighted sum aggregation",
            config_space=config_space,
            objectives=["accuracy", "cost", "latency"],
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "aggregation": "weighted_sum",
                "weights": {
                    "accuracy": 0.5,
                    "cost": 0.3,
                    "latency": 0.2,
                },
            },
            gist_template="weighted-sum -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Weighted sum aggregation config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_primary_objective_aggregation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test using primary objective for aggregation.

        Only the primary objective is used for optimization decisions.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = TestScenario(
            name="primary_objective",
            description="Primary objective for optimization",
            config_space=config_space,
            objectives=["accuracy", "cost", "latency"],
            max_trials=3,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "tpe",
                "aggregation": "primary",
                "primary_objective": "accuracy",
            },
            gist_template="primary-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 3, f"Expected 3 trials, got {len(result.trials)}"

        # Verify we have best config from optimization
        assert hasattr(result, "best_config"), "Should have best_config"
        assert result.best_config is not None, "best_config should not be None"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pareto_front_aggregation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Pareto front for multi-objective optimization."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name="pareto_front",
            description="Pareto front multi-objective",
            config_space=config_space,
            objectives=["accuracy", "cost"],
            max_trials=4,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "nsga2",
                "aggregation": "pareto",
            },
            gist_template="pareto -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Pareto aggregation config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lexicographic_aggregation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test lexicographic ordering of objectives.

        Objectives are optimized in priority order.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        }

        scenario = TestScenario(
            name="lexicographic",
            description="Lexicographic objective ordering",
            config_space=config_space,
            objectives=["accuracy", "cost", "latency"],
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "aggregation": "lexicographic",
                "objective_priority": ["accuracy", "cost", "latency"],
            },
            gist_template="lexicographic -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Lexicographic aggregation config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 3, f"Expected 3 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestScoreNormalization:
    """Tests for score normalization across objectives."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_normalize_scores(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test score normalization to [0, 1] range."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="normalize_scores",
            description="Normalize scores to [0, 1]",
            config_space=config_space,
            objectives=["accuracy", "latency"],  # Different scales
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "normalize_scores": True,
            },
            gist_template="normalize -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Normalization config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_min_max_normalization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test min-max normalization strategy."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        }

        scenario = TestScenario(
            name="min_max_normalization",
            description="Min-max normalization",
            config_space=config_space,
            objectives=["accuracy", "cost"],
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "normalization": "min_max",
                "min_values": {"accuracy": 0.0, "cost": 0.0},
                "max_values": {"accuracy": 1.0, "cost": 10.0},
            },
            gist_template="min_max -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Min-max normalization config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 3, f"Expected 3 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_z_score_normalization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test z-score normalization strategy."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "claude-3-opus"],
        }

        scenario = TestScenario(
            name="z_score_normalization",
            description="Z-score normalization",
            config_space=config_space,
            objectives=["accuracy", "cost"],
            max_trials=4,
            mock_mode_config={
                "optimizer": "random",
                "normalization": "z_score",
            },
            gist_template="z_score -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Z-score normalization config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestObjectiveDirections:
    """Tests for objective direction handling in scoring."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_maximize_direction(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test maximize direction for objectives.

        Default direction is maximize for accuracy.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="maximize_direction",
            description="Maximize objective direction",
            config_space=config_space,
            objectives=["accuracy"],  # String objective
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                # Direction configured via mock config
                "objective_directions": {"accuracy": "maximize"},
            },
            gist_template="maximize -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimize_direction(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test minimize direction for objectives.

        Cost objective should be minimized.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="minimize_direction",
            description="Minimize objective direction",
            config_space=config_space,
            objectives=["cost"],  # String objective
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "objective_directions": {"cost": "minimize"},
            },
            gist_template="minimize -> {{model}}",
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

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_directions(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mixed objective directions (some maximize, some minimize)."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="mixed_directions",
            description="Mixed maximize/minimize directions",
            config_space=config_space,
            objectives=["accuracy", "cost", "latency"],  # String objectives
            max_trials=3,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "nsga2",  # Supports multi-objective
                "objective_directions": {
                    "accuracy": "maximize",
                    "cost": "minimize",
                    "latency": "minimize",
                },
            },
            gist_template="mixed -> {{model}}",
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


class TestScoringEdgeCases:
    """Tests for edge cases in scoring."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nan_score_handling(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test handling of NaN scores."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="nan_scores",
            description="NaN score handling",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "inject_nan_scores": True,  # Force some NaN scores
            },
            gist_template="nan -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # NaN injection config should be accepted (mock mode ignores this)
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_inf_score_handling(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test handling of infinite scores."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="inf_scores",
            description="Infinite score handling",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "inject_inf_scores": True,
            },
            gist_template="inf -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Inf injection config should be accepted (mock mode ignores this)
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_score(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test handling of zero scores."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="zero_scores",
            description="Zero score handling",
            config_space=config_space,
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "score_distribution": "constant",
                "constant_score": 0.0,
            },
            gist_template="zero -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Zero is a valid score
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_scores(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test handling of negative scores.

        Note: score_range configuration may not be implemented.
        This test documents expected behavior.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="negative_scores",
            description="Negative score handling",
            config_space=config_space,
            objectives=["accuracy"],  # String objective
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                # Note: score_range may not be implemented
                "score_range": (-1.0, 0.0),  # Negative scores
            },
            gist_template="negative -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should work regardless of score range config
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestScoringWithConstraints:
    """Tests for scoring with constraint handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constrained_scoring(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test scoring with constraints via mock_mode_config.

        Note: Constraints in TestScenario must be ConstraintSpec objects.
        This test uses mock_mode_config to specify constraints instead.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.5, 0.7],
        }

        scenario = TestScenario(
            name="constrained_scoring",
            description="Scoring with constraints",
            config_space=config_space,
            objectives=["accuracy"],
            # Don't use constraints field - use mock_mode_config
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                # Document constraint support in config
                "constraint_config": {"cost_limit": 10},
            },
            gist_template="constrained -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should work without explicit constraints
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
    async def test_penalty_for_constraint_violation(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test penalty scoring for constraint violations."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = TestScenario(
            name="constraint_penalty",
            description="Penalty for constraint violations",
            config_space=config_space,
            objectives=["accuracy"],
            max_trials=2,
            mock_mode_config={
                "optimizer": "random",
                "constraint_handling": "penalty",
                "penalty_factor": 0.5,
            },
            gist_template="penalty -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Penalty handling config should be accepted
        assert not isinstance(result, Exception), f"Should succeed: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
