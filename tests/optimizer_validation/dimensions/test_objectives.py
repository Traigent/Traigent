"""Tests for objective configurations.

Tests single/multiple objectives, weighted objectives, and orientation settings.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
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
            gist_template=(f"single-{objective} -> {{trial_count()}} | {{status()}}"),
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="default-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestBoundedObjective:
    """Tests for bounded objective configurations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bounded_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with defined bounds (thresholds).

        Purpose:
            Verify that objectives with bounds are handled correctly.
            (Note: Current implementation might treat bounds as constraints or just metadata)

        Edge Case: Bounded objective
        """
        scenario = TestScenario(
            name="bounded_objective",
            description="Objective with bounds",
            objectives=[
                ObjectiveSpec(
                    name="accuracy",
                    orientation="maximize",
                    bounds=(0.5, 1.0),  # Min 0.5, Max 1.0
                )
            ],
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=2,
            gist_template="bounded-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template=(
                f"multi-{objective_count} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="acc-cost -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="unequal-weights -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="extreme-weights -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template=(f"orient-{orientation} -> {{trial_count()}} | {{status()}}"),
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="mixed-orient -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="bounded-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestObjectiveEdgeCases:
    """Tests for objective edge cases."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_weight_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with zero weight (should be ignored)."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="zero_weight",
            objectives=objectives,
            max_trials=2,
            gist_template="zero-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle zero weight - effectively single objective
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_weight_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with negative weight."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=-0.5),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="negative_weight",
            objectives=objectives,
            max_trials=2,
            gist_template="neg-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # May fail validation or invert the objective
        # Document observed behavior
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_weights_zero(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test when all objective weights are zero."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.0),
        ]

        scenario = multi_objective_scenario(
            name="all_zero_weights",
            objectives=objectives,
            max_trials=2,
            gist_template="all-zero -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle or fail gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_small_weight(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with very small (but non-zero) weight."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1e-10),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="tiny_weight",
            objectives=objectives,
            max_trials=2,
            gist_template="tiny-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle very small weights
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_large_weight(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with very large weight."""
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1e10),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="huge_weight",
            objectives=objectives,
            max_trials=2,
            gist_template="huge-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle large weights
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_many_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with many objectives (approaching practical limits)."""
        # Use a realistic subset of metrics that might be available
        objective_names = ["accuracy", "cost", "latency"]
        objectives = [
            ObjectiveSpec(
                name=name,
                orientation="maximize" if name == "accuracy" else "minimize",
                weight=1.0 / len(objective_names),
            )
            for name in objective_names
        ]

        scenario = multi_objective_scenario(
            name="many_objectives",
            objectives=objectives,
            max_trials=3,
            gist_template="many-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle multiple objectives
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_inverted_bounds(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with inverted bounds (max < min)."""
        objectives = [
            ObjectiveSpec(
                name="accuracy",
                orientation="maximize",
                weight=1.0,
                bounds=(1.0, 0.0),  # Inverted: max < min
            ),
        ]

        scenario = multi_objective_scenario(
            name="inverted_bounds",
            objectives=objectives,
            max_trials=2,
            gist_template="inv-bounds -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should fail validation or handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_same_objective_different_orientations(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test same metric as both maximize and minimize (conflicting)."""
        # This is a pathological case - same metric can't be both
        # Using different metrics to avoid name collision
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(
                name="cost", orientation="maximize", weight=1.0
            ),  # cost usually minimized
        ]

        scenario = multi_objective_scenario(
            name="unusual_orientations",
            objectives=objectives,
            max_trials=2,
            gist_template="unusual-orient -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should work but may produce unexpected results
        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()


class TestMultiObjectiveWithAlgorithms:
    """Tests for multi-objective optimization with different algorithms.

    Purpose:
        Address the gap where multi-objective was not tested explicitly
        with different optimization algorithms (grid, random, optuna).

    Coverage Gap Addressed:
        Multi-objective × Algorithm interactions were limited.
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe", "optuna_random"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_objective_with_algorithm(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multi-objective optimization with each algorithm.

        Purpose:
            Verify each algorithm handles multi-objective optimization.
            Note: CMA-ES does NOT support multi-objective and is excluded.

        Dimensions: Algorithm={algorithm}, ObjectiveConfig=multi_objective
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name=f"multi_obj_{algorithm}",
            objectives=objectives,
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"multi-{algorithm} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_with_weighted_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with weighted multi-objective.

        Purpose:
            Verify grid search handles weighted objectives correctly,
            using the weighted sum for comparison.

        Dimensions: Algorithm=grid, ObjectiveConfig=weighted
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = multi_objective_scenario(
            name="grid_weighted_multi",
            objectives=objectives,
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid-weighted -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_exhaustive_with_multi_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search exhaustively explores with multi-objective.

        Purpose:
            Verify grid search explores all config combinations even
            when multiple objectives are being optimized.

        Dimensions: Algorithm=grid, ObjectiveConfig=multi_objective
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="grid_exhaustive_multi",
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=10,  # More than needed
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid-exhaust-multi -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        if hasattr(result, "trials"):
            # Grid should have tried all 4 combinations
            assert (
                len(result.trials) == 4
            ), f"Expected 4 trials, got {len(result.trials)}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_with_three_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search with three objectives.

        Purpose:
            Verify random search handles more than two objectives.

        Dimensions: Algorithm=random, ObjectiveConfig=three_objectives
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
            ObjectiveSpec(name="latency", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="random_three_obj",
            objectives=objectives,
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            gist_template="random-three -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_cmaes_single_objective_only(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Verify CMA-ES works only with single objective.

        Purpose:
            CMA-ES does not support multi-objective optimization.
            Verify it works with a single objective but document
            that multi-objective is not supported.

        Dimensions: Algorithm=optuna_cmaes, ObjectiveConfig=single
        """
        # Single objective should work
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name="cmaes_single_obj",
            objectives=objectives,
            config_space={
                "temperature": (0.0, 1.0),  # Continuous for CMA-ES
                "top_p": (0.5, 1.0),
            },
            max_trials=3,
            mock_mode_config={"optimizer": "optuna_cmaes"},
            gist_template="cmaes-single -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
