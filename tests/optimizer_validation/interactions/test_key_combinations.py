"""Tests for key injection mode + execution mode combinations.

Tests strategic combinations (4-5) rather than full matrix (20 combos).
Focus on common usage patterns and known edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ConstraintSpec,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
    basic_scenario,
    multi_objective_scenario,
)


class TestKeyInjectionExecutionCombinations:
    """Test strategic injection mode + execution mode combinations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_edge_analytics_default(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test most common default: context injection + edge_analytics.

        This is the default configuration for most users.
        """
        scenario = basic_scenario(
            name="context_edge_analytics",
            injection_mode="context",
            execution_mode="edge_analytics",
            max_trials=3,
            gist_template="context+edge -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_seamless_hybrid_zero_code_change(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test seamless injection with hybrid execution.

        Zero-code-change optimization with cloud fallback capability.
        """
        scenario = basic_scenario(
            name="seamless_hybrid",
            injection_mode="seamless",
            execution_mode="hybrid",
            max_trials=3,
            gist_template="seamless+hybrid -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parameter_cloud_explicit_config(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parameter injection with cloud execution.

        Explicit config parameter for cloud-based optimization.
        """
        scenario = basic_scenario(
            name="parameter_cloud",
            injection_mode="parameter",
            execution_mode="cloud",
            max_trials=3,
            gist_template="parameter+cloud -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attribute_edge_analytics_simple(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test attribute injection with edge_analytics.

        Simple attribute storage for local optimization.
        """
        scenario = basic_scenario(
            name="attribute_edge_analytics",
            injection_mode="attribute",
            execution_mode="edge_analytics",
            max_trials=3,
            gist_template="attribute+edge -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMultiObjectiveWithConstraints:
    """Test multi-objective optimization combined with constraints."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_objectives_with_config_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted multi-objective with config-only constraint."""

        def temp_limit(config: dict[str, Any]) -> bool:
            return bool(config.get("temperature", 0) < 0.9)

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ]

        constraints = [
            ConstraintSpec(name="temp_limit", constraint_fn=temp_limit),
        ]

        scenario = TestScenario(
            name="weighted_with_config_constraint",
            description="Weighted multi-objective with config constraint",
            objectives=objectives,
            constraints=constraints,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7, 0.95],
            },
            max_trials=4,
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template=("weighted+constraint -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_objective_with_metric_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multi-objective with metric-dependent constraint."""

        def cost_limit(
            config: dict[str, Any], metrics: dict[str, float] | None = None
        ) -> bool:
            if metrics is None:
                return True
            return bool(metrics.get("cost", 0) <= 0.15)

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="latency", orientation="minimize", weight=1.0),
        ]

        constraints = [
            ConstraintSpec(
                name="cost_limit",
                constraint_fn=cost_limit,
                requires_metrics=True,
            ),
        ]

        scenario = TestScenario(
            name="multi_obj_metric_constraint",
            description="Multi-objective with metric constraint",
            objectives=objectives,
            constraints=constraints,
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=4,
            expected=ExpectedResult(required_metrics=["accuracy", "latency"]),
            gist_template=(
                "multi-obj+metric-constraint -> {trial_count()} | {status()}"
            ),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestComplexWorkflows:
    """Test complete optimization workflows with multiple features."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_featured_workflow(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test realistic workflow with multiple features combined.

        Features:
        - Multi-objective optimization (accuracy + cost)
        - Mixed config space (categorical + numeric)
        - Config constraint
        - Edge analytics execution
        """

        def model_temp_constraint(config: dict[str, Any]) -> bool:
            # GPT-4 should use lower temperature
            if config.get("model") == "gpt-4":
                return bool(config.get("temperature", 0) <= 0.7)
            return True

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        constraints = [
            ConstraintSpec(name="model_temp", constraint_fn=model_temp_constraint),
        ]

        scenario = TestScenario(
            name="full_featured_workflow",
            description="Complete workflow with all features",
            injection_mode="context",
            execution_mode="edge_analytics",
            objectives=objectives,
            constraints=constraints,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 500, 1000],
            },
            max_trials=5,
            expected=ExpectedResult(
                min_trials=3,
                required_metrics=["accuracy", "cost"],
            ),
            gist_template="full-workflow -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Check we got multiple successful trials
        assert len(result.trials) >= 1

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_apply_best_config_after_optimization(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization followed by applying best config."""
        scenario = basic_scenario(
            name="apply_best_config",
            injection_mode="context",
            execution_mode="edge_analytics",
            max_trials=3,
            gist_template="apply-best -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Validate the optimization completed
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Apply best config and verify
        if hasattr(func, "apply_best_config"):
            func.apply_best_config()
            # Best config should now be active
            if hasattr(func, "current_config"):
                assert func.current_config is not None


class TestEdgeCaseCombinations:
    """Test edge case combinations that might have issues."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_config_value_with_multi_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test single-value config with multi-objective."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="single_value_multi_obj",
            objectives=objectives,
            config_space={
                "model": ["gpt-4"],  # Single value
                "temperature": [0.5, 0.7],  # Multiple values
            },
            max_trials=2,
            gist_template=("single-val+multi-obj -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_many_objectives_small_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test many objectives with small config space."""
        # Use only built-in metrics supported by the evaluator
        objectives = [
            ObjectiveSpec(name="accuracy", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
            ObjectiveSpec(name="latency", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="many_objectives_small_space",
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
            },
            max_trials=3,
            gist_template="3-obj+small-space -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
