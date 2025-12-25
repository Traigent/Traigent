"""Tests for objective configurations.

Tests single/multiple objectives, weighted objectives, and orientation settings.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
    basic_scenario,
    multi_objective_scenario,
)


class TestSingleObjective:
    """Tests for single-objective optimization."""

    @pytest.mark.parametrize(
        "objective",
        ["accuracy", "cost", "latency"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_objective_types(
        self,
        objective: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with different single objectives."""
        scenario = basic_scenario(
            name=f"single_{objective}",
            objectives=[objective],
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_accuracy_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test default objective is accuracy when not specified."""
        scenario = TestScenario(
            name="default_objective",
            description="Default objective (accuracy)",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=2,
            # objectives not specified - should default to ["accuracy"]
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMultiObjective:
    """Tests for multi-objective optimization."""

    @pytest.mark.parametrize("objective_count", [2, 3])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_objective_counts(
        self,
        objective_count: int,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with varying numbers of objectives."""
        # Use only built-in metrics supported by the evaluator
        objective_names = ["accuracy", "cost", "latency"]
        objectives = [
            ObjectiveSpec(name=objective_names[i], weight=1.0)
            for i in range(objective_count)
        ]

        scenario = multi_objective_scenario(
            name=f"multi_{objective_count}_objectives",
            objectives=objectives,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_accuracy_and_cost_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test common accuracy + cost multi-objective optimization."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="accuracy_cost_combo",
            objectives=objectives,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestWeightedObjectives:
    """Tests for weighted multi-objective optimization."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unequal_weights(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multi-objective with unequal weights."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ]

        scenario = multi_objective_scenario(
            name="weighted_unequal",
            objectives=objectives,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert scenario.has_weighted_objectives()
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extreme_weight_ratio(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multi-objective with extreme weight ratio."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.95),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.05),
        ]

        scenario = multi_objective_scenario(
            name="weighted_extreme",
            objectives=objectives,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestObjectiveOrientations:
    """Tests for objective orientation settings."""

    @pytest.mark.parametrize("orientation", ["maximize", "minimize"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orientation_types(
        self,
        orientation: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test maximize vs minimize orientations."""
        # Use "accuracy" for maximize, "cost" for minimize (built-in metrics)
        metric_name = "accuracy" if orientation == "maximize" else "cost"
        objectives = [
            ObjectiveSpec(name=metric_name, orientation=orientation, weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name=f"orientation_{orientation}",
            objectives=objectives,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_orientations(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objectives with mixed orientations."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
            ObjectiveSpec(name="latency", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="mixed_orientations",
            objectives=objectives,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestObjectiveWithBounds:
    """Tests for objectives with value bounds."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bounded_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with explicit bounds."""
        objectives = [
            ObjectiveSpec(
                name="accuracy",
                orientation="maximize",
                weight=1.0,
                bounds=(0.0, 1.0),
            ),
        ]

        scenario = multi_objective_scenario(
            name="bounded_accuracy",
            objectives=objectives,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
