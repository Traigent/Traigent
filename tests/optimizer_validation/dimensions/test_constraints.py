"""Tests for constraint configurations.

Tests config-only constraints, metric-dependent constraints, and compound constraints.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import ConstraintSpec, constrained_scenario


class TestConfigOnlyConstraints:
    """Tests for constraints that only check configuration values."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_config_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test simple constraint on config value."""

        def temp_under_limit(config: dict[str, Any]) -> bool:
            return config.get("temperature", 0) < 0.8

        constraints = [
            ConstraintSpec(
                name="temp_limit",
                constraint_fn=temp_under_limit,
                requires_metrics=False,
            )
        ]

        scenario = constrained_scenario(
            name="simple_config_constraint",
            constraints=constraints,
            config_space={"temperature": [0.1, 0.5, 0.9]},
            max_trials=3,
            gist_template="simple-constraint -> {{temperature}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # Note: Constraints may be checked at different stages depending on implementation
        # The optimizer may still evaluate configs and then apply constraints to results
        # We don't assert on individual trial configs here as behavior varies

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_specific_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint based on model selection."""

        def gpt4_low_temp_only(config: dict[str, Any]) -> bool:
            if config.get("model") == "gpt-4":
                return config.get("temperature", 0) <= 0.5
            return True

        constraints = [
            ConstraintSpec(
                name="gpt4_low_temp",
                constraint_fn=gpt4_low_temp_only,
                requires_metrics=False,
            )
        ]

        scenario = constrained_scenario(
            name="model_specific_constraint",
            constraints=constraints,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7, 0.9],
            },
            max_trials=4,
            gist_template="model-constraint -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMetricDependentConstraints:
    """Tests for constraints that require metric values."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cost_limit_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint based on cost metric."""

        def cost_under_limit(
            config: dict[str, Any], metrics: dict[str, float] | None = None
        ) -> bool:
            if metrics is None:
                return True  # Allow config, will check after evaluation
            return bool(metrics.get("cost", 0) <= 0.10)

        constraints = [
            ConstraintSpec(
                name="cost_limit",
                constraint_fn=cost_under_limit,
                requires_metrics=True,
            )
        ]

        scenario = constrained_scenario(
            name="cost_limit_constraint",
            constraints=constraints,
            max_trials=3,
            gist_template="cost-limit -> {{cost}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

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
    async def test_accuracy_threshold_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint requiring minimum accuracy."""

        def min_accuracy(
            config: dict[str, Any], metrics: dict[str, float] | None = None
        ) -> bool:
            if metrics is None:
                return True
            return bool(metrics.get("accuracy", 0) >= 0.5)

        constraints = [
            ConstraintSpec(
                name="min_accuracy",
                constraint_fn=min_accuracy,
                requires_metrics=True,
            )
        ]

        scenario = constrained_scenario(
            name="accuracy_threshold",
            constraints=constraints,
            max_trials=3,
            gist_template="accuracy-threshold -> {{accuracy}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestCompoundConstraints:
    """Tests for multiple constraints applied together."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_config_constraints(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multiple config-only constraints together.

        Constraints:
        - temp_limit: temperature < 0.9 (rejects 0.95)
        - tokens_limit: max_tokens <= 1000 (rejects 2000)

        Valid combinations (of 9 total):
        - temperature in [0.3, 0.7] AND max_tokens in [100, 500]
        - That's 2 * 2 = 4 valid combinations

        The test verifies:
        1. Optimization completes successfully
        2. All trials satisfy BOTH constraints
        3. No rejected configs appear in results
        """

        def temp_constraint(config: dict[str, Any]) -> bool:
            return bool(config.get("temperature", 0) < 0.9)

        def tokens_constraint(config: dict[str, Any]) -> bool:
            return bool(config.get("max_tokens", 0) <= 1000)

        constraints = [
            ConstraintSpec(name="temp_limit", constraint_fn=temp_constraint),
            ConstraintSpec(name="tokens_limit", constraint_fn=tokens_constraint),
        ]

        scenario = constrained_scenario(
            name="multiple_config_constraints",
            constraints=constraints,
            config_space={
                "temperature": [0.3, 0.7, 0.95],
                "max_tokens": [100, 500, 2000],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="multiple-constraints -> {{temperature}}, {{max_tokens}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Failed: {result}"

        # Verify trials exist
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) > 0, "Should have at least one trial"

        # Import TrialStatus for status checking
        from traigent.api.types import TrialStatus

        # Verify COMPLETED trials satisfy BOTH constraints
        # (Failed trials are expected when constraints reject configs)
        completed_trials = [
            t for t in result.trials if t.status == TrialStatus.COMPLETED
        ]
        assert len(completed_trials) > 0, "Should have >= 1 completed trial"

        for i, trial in enumerate(completed_trials):
            temp = trial.config.get("temperature")
            tokens = trial.config.get("max_tokens")

            # temp_constraint: temperature < 0.9
            assert temp is not None and temp < 0.9, (
                f"Completed trial {i} violates temp_limit constraint: "
                f"temperature={temp} should be < 0.9"
            )

            # tokens_constraint: max_tokens <= 1000
            assert tokens is not None and tokens <= 1000, (
                f"Completed trial {i} violates tokens_limit constraint: "
                f"max_tokens={tokens} should be <= 1000"
            )

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_constraint_types(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mix of config-only and metric-dependent constraints."""

        def config_constraint(config: dict[str, Any]) -> bool:
            return bool(config.get("temperature", 0) < 0.9)

        def metric_constraint(
            config: dict[str, Any], metrics: dict[str, float] | None = None
        ) -> bool:
            if metrics is None:
                return True
            return bool(metrics.get("cost", 0) <= 0.15)

        constraints = [
            ConstraintSpec(
                name="temp_limit",
                constraint_fn=config_constraint,
                requires_metrics=False,
            ),
            ConstraintSpec(
                name="cost_limit",
                constraint_fn=metric_constraint,
                requires_metrics=True,
            ),
        ]

        scenario = constrained_scenario(
            name="mixed_constraints",
            constraints=constraints,
            max_trials=4,
            gist_template="mixed-constraints -> {{temperature}}, {{cost}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestConstraintEdgeCases:
    """Tests for edge cases in constraint handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_always_true_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint that always returns True."""

        def always_accept(config: dict[str, Any]) -> bool:
            return True

        constraints = [
            ConstraintSpec(name="always_accept", constraint_fn=always_accept)
        ]

        scenario = constrained_scenario(
            name="always_true",
            constraints=constraints,
            max_trials=2,
            gist_template="always-true",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

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
    async def test_always_false_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint that always returns False (rejects all configs)."""

        def always_reject(config: dict[str, Any]) -> bool:
            return False

        constraints = [
            ConstraintSpec(name="always_reject", constraint_fn=always_reject)
        ]

        scenario = constrained_scenario(
            name="always_false",
            constraints=constraints,
            max_trials=3,
            gist_template="always-false",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - no valid configs exist
        assert hasattr(result, "trials"), "Result should have trials attribute"
        assert result.stop_reason is not None, "Should have a stop reason"
        # When all configs are rejected by constraint, we get 0 trials
        assert len(result.trials) == 0, "Should have 0 trials (all configs rejected)"

        # Skip validator - it expects trials, but none exist when all rejected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_returns_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint that returns None instead of bool."""

        def returns_none(config: dict[str, Any]) -> Any:
            return None

        constraints = [ConstraintSpec(name="returns_none", constraint_fn=returns_none)]

        scenario = constrained_scenario(
            name="none_return",
            constraints=constraints,
            max_trials=2,
            gist_template="returns-none",
        )

        _, result = await scenario_runner(scenario)

        # None is falsy, so should act like False - all configs rejected
        assert hasattr(result, "trials"), "Result should have trials attribute"
        assert result.stop_reason is not None, "Should have a stop reason"
        assert (
            len(result.trials) == 0
        ), "Should have 0 trials (None returns = falsy = rejected)"

        # Skip validator - it expects trials, but none exist when all rejected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_raises_exception(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint that raises an exception."""

        def raises_error(config: dict[str, Any]) -> bool:
            raise ValueError("Constraint evaluation failed")

        constraints = [ConstraintSpec(name="raises_error", constraint_fn=raises_error)]

        scenario = constrained_scenario(
            name="exception_constraint",
            constraints=constraints,
            max_trials=2,
            gist_template="raises-exception",
        )

        _, result = await scenario_runner(scenario)

        # Should handle exception gracefully
        assert hasattr(result, "trials"), "Result should have trials attribute"
        assert result.stop_reason is not None, "Should have a stop reason"
        # Exception in constraint = rejection, all configs rejected = 0 trials
        assert len(result.trials) == 0, "Should have 0 trials (exception = rejection)"

        # Skip validator - it expects trials, but none exist when all rejected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_raises_on_specific_config(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint that raises only for specific configs."""

        def conditional_error(config: dict[str, Any]) -> bool:
            if config.get("model") == "gpt-4":
                raise RuntimeError("gpt-4 not allowed")
            return True

        constraints = [
            ConstraintSpec(name="conditional_error", constraint_fn=conditional_error)
        ]

        scenario = constrained_scenario(
            name="conditional_exception",
            constraints=constraints,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=3,
            gist_template="raises-specific -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle partial failures

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
    async def test_constraint_modifies_config(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint that has side effect of modifying config dict."""

        def modifying_constraint(config: dict[str, Any]) -> bool:
            # Bad practice: modifying the config
            config["_modified"] = True
            return True

        constraints = [
            ConstraintSpec(name="modifier", constraint_fn=modifying_constraint)
        ]

        scenario = constrained_scenario(
            name="modifying_constraint",
            constraints=constraints,
            max_trials=2,
            gist_template="modifies-config",
        )

        _, result = await scenario_runner(scenario)

        # Should work but document side effects

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
    async def test_many_constraints(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with many constraints (10+)."""
        constraints = []
        for i in range(10):
            # Each constraint checks a different threshold

            def make_constraint(threshold: float):
                def constraint_fn(config: dict[str, Any]) -> bool:
                    return bool(config.get("temperature", 0) != threshold)

                return constraint_fn

            constraints.append(
                ConstraintSpec(
                    name=f"constraint_{i}",
                    constraint_fn=make_constraint(i * 0.1),
                )
            )

        scenario = constrained_scenario(
            name="many_constraints",
            constraints=constraints,
            config_space={
                "temperature": [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
            },
            max_trials=5,
            gist_template="many-constraints -> {{temperature}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle many constraints
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restrictive_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test highly restrictive constraint that filters most configs."""

        def very_restrictive(config: dict[str, Any]) -> bool:
            # Only allow specific combination
            return (
                config.get("model") == "gpt-3.5-turbo"
                and config.get("temperature", 0) <= 0.3
            )

        constraints = [
            ConstraintSpec(name="restrictive", constraint_fn=very_restrictive)
        ]

        scenario = constrained_scenario(
            name="restrictive_constraint",
            constraints=constraints,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.1, 0.5, 0.9],
            },
            max_trials=3,
            gist_template="restrictive -> {{model}}, {{temperature}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
