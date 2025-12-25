"""Tests for constraint configurations.

Tests config-only constraints, metric-dependent constraints, and compound constraints.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ConstraintSpec,
    ExpectedResult,
    TestScenario,
    constrained_scenario,
)


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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify high-temperature configs were filtered
        if hasattr(result, "successful_trials"):
            for trial in result.successful_trials:
                assert trial.config.get("temperature", 0) < 0.8

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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
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
            return metrics.get("cost", 0) <= 0.10

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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
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
            return metrics.get("accuracy", 0) >= 0.5

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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
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
        """Test multiple config-only constraints together."""

        def temp_constraint(config: dict[str, Any]) -> bool:
            return config.get("temperature", 0) < 0.9

        def tokens_constraint(config: dict[str, Any]) -> bool:
            return config.get("max_tokens", 0) <= 1000

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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
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
            return config.get("temperature", 0) < 0.9

        def metric_constraint(
            config: dict[str, Any], metrics: dict[str, float] | None = None
        ) -> bool:
            if metrics is None:
                return True
            return metrics.get("cost", 0) <= 0.15

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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
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
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
