"""Comprehensive edge case tests for objective system."""

import json
import warnings

import pytest

from traigent.api.decorators import optimize
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.objectives import (
    ObjectiveDefinition,
    ObjectiveSchema,
    create_default_objectives,
)
from traigent.utils.multi_objective import (
    ParetoFrontCalculator,
    ParetoPoint,
    scalarize_objectives,
)


class TestSingleObjectiveMinimize:
    """Tests for single objective minimize scenarios."""

    def test_single_objective_minimize_normalization(self):
        """Test normalization for single minimize objective."""
        schema = create_default_objectives(["cost"], orientations={"cost": "minimize"})

        # Test normalization with custom ranges
        normalized = schema.normalize_value("cost", 0.05, min_val=0.01, max_val=0.10)
        # For minimize: lower is better, so 0.05 is in the middle-upper range
        expected = (0.10 - 0.05) / (0.10 - 0.01)  # (max - value) / (max - min)
        assert abs(normalized - expected) < 1e-10

        # Test boundary values
        assert schema.normalize_value("cost", 0.01, 0.01, 0.10) == 1.0  # Best (minimum)
        assert (
            schema.normalize_value("cost", 0.10, 0.01, 0.10) == 0.0
        )  # Worst (maximum)

    def test_single_objective_minimize_in_optimization_result(self):
        """Test OptimizationResult with single minimize objective."""
        from datetime import datetime

        from traigent.api.types import OptimizationStatus

        # Create trials with varying costs
        trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"model": f"model_{i}"},
                metrics={"cost": cost},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
            for i, cost in enumerate([0.10, 0.05, 0.15, 0.03, 0.08])
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"model": "model_3"},  # Lowest cost
            best_score=0.03,
            optimization_id="test_opt",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["cost"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        # Calculate weighted scores with minimize objective
        weighted_results = result.calculate_weighted_scores(
            minimize_objectives=["cost"]
        )

        # Find the best score (should be the trial with lowest cost)
        best_weighted_score = max(
            score for _, score in weighted_results["weighted_scores"]
        )

        # Verify trial with cost 0.03 has the best score
        for trial, score in weighted_results["weighted_scores"]:
            if trial.metrics["cost"] == 0.03:
                assert (
                    abs(score - best_weighted_score) < 1e-10
                )  # This should be the best
                break
        else:
            pytest.fail("Trial with cost 0.03 not found as best")

    def test_single_objective_pareto_front(self):
        """Test Pareto front with single objective."""
        from datetime import datetime

        trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"param": i},
                metrics={"cost": cost},
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=datetime.now(),
            )
            for i, cost in enumerate([0.10, 0.05, 0.15])
        ]

        schema = create_default_objectives(["cost"], orientations={"cost": "minimize"})
        calculator = ParetoFrontCalculator(objective_schema=schema)

        # With single objective, only the best point is on Pareto front
        front = calculator.calculate_pareto_front(trials, ["cost"])
        assert len(front) == 1
        assert front[0].objectives["cost"] == 0.05  # Best (minimum) value


class TestProvidedBoundsVsObservedRanges:
    """Tests for provided bounds vs observed ranges."""

    def test_normalization_with_provided_bounds(self):
        """Test that provided bounds are used for normalization."""
        obj = ObjectiveDefinition(
            name="accuracy", orientation="maximize", weight=1.0, bounds=(0.0, 1.0)
        )
        schema = ObjectiveSchema.from_objectives([obj])

        # Normalize using provided bounds
        normalized = schema.normalize_value("accuracy", 0.75)
        assert normalized == 0.75  # Direct mapping with 0-1 bounds

        # Value outside bounds should be clipped
        assert schema.normalize_value("accuracy", 1.5) == 1.0  # Clipped to max
        assert schema.normalize_value("accuracy", -0.5) == 0.0  # Clipped to min

    def test_normalization_with_observed_ranges(self):
        """Test normalization with observed ranges when bounds not provided."""
        from datetime import datetime

        trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={},
                metrics={"accuracy": acc},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
            for i, acc in enumerate([0.70, 0.85, 0.92, 0.78])
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={},
            best_score=0.92,
            optimization_id="test",
            duration=4.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy"],
            algorithm="test",
            timestamp=datetime.now(),
        )

        # Calculate weighted scores uses observed ranges
        weighted_results = result.calculate_weighted_scores()

        # Check normalization ranges were calculated from observed data
        ranges = weighted_results["normalization_ranges"]
        assert ranges["accuracy"] == (0.70, 0.92)

    def test_mixed_bounds_and_observed(self):
        """Test with some objectives having bounds and others using observed ranges."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.5, bounds=(0.0, 1.0)),
            ObjectiveDefinition("latency", "minimize", 0.5),  # No bounds
        ]
        schema = ObjectiveSchema.from_objectives(objectives)

        # Accuracy uses provided bounds
        metrics = {"accuracy": 0.8, "latency": 50}
        normalized = schema.normalize_metrics(
            metrics, ranges={"latency": (10, 100)}  # Observed range for latency
        )

        assert normalized["accuracy"] == 0.8  # Direct with 0-1 bounds
        expected_latency = (100 - 50) / (100 - 10)  # minimize normalization
        assert abs(normalized["latency"] - expected_latency) < 1e-10


class TestZeroRangeHandling:
    """Tests for zero-range (constant value) handling."""

    def test_zero_range_normalization(self):
        """Test normalization when all values are the same."""
        schema = create_default_objectives(["accuracy"])

        # When min == max, should return 0.5 (middle value)
        normalized = schema.normalize_value(
            "accuracy", 0.85, min_val=0.85, max_val=0.85
        )
        assert normalized == 0.5

    def test_zero_range_in_optimization_result(self):
        """Test optimization result when all trials have same metric value."""
        from datetime import datetime

        # All trials have same accuracy
        trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"seed": i},
                metrics={"accuracy": 0.90},  # Same for all
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
            for i in range(5)
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"seed": 0},
            best_score=0.90,
            optimization_id="test",
            duration=5.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy"],
            algorithm="test",
            timestamp=datetime.now(),
        )

        weighted_results = result.calculate_weighted_scores()

        # All should have same score when values are identical
        scores = [score for _, score in weighted_results["weighted_scores"]]
        assert all(abs(s - scores[0]) < 1e-10 for s in scores)

    def test_epsilon_tolerance_in_normalization(self):
        """Test that epsilon tolerance handles near-zero ranges."""
        schema = create_default_objectives(["accuracy"])

        # Very small range (below epsilon)
        normalized = schema.normalize_value(
            "accuracy", 0.8500001, min_val=0.85, max_val=0.8500002, epsilon=1e-6
        )
        assert normalized == 0.5  # Should return middle value for near-zero range

    def test_boundary_epsilon_collapses_to_half(self):
        """Spans in the boundary band [1e-10, 1e-9) collapse under the new
        normative epsilon (TraigentSchema multi_objective_semantics_schema.json
        v1.0.0). Pre-rollout this band normalized linearly.
        """
        schema = create_default_objectives(["accuracy"])

        # Span = 5e-10: between the old 1e-10 and the new 1e-9 epsilon.
        normalized = schema.normalize_value(
            "accuracy", 0.85, min_val=0.85, max_val=0.85 + 5e-10
        )
        assert normalized == 0.5

        # normalize_metrics() must agree (it forwards epsilon to normalize_value).
        bulk = schema.normalize_metrics(
            {"accuracy": 0.85}, ranges={"accuracy": (0.85, 0.85 + 5e-10)}
        )
        assert bulk["accuracy"] == 0.5


class TestMixedOrientationParetoFront:
    """Tests for Pareto fronts with mixed maximize/minimize objectives."""

    def test_mixed_orientation_dominance(self):
        """Test Pareto dominance with mixed orientations."""
        create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
        )

        # Create points
        point1 = ParetoPoint(
            config={}, objectives={"accuracy": 0.9, "cost": 0.05}, trial=None
        )
        point2 = ParetoPoint(
            config={}, objectives={"accuracy": 0.8, "cost": 0.10}, trial=None
        )

        # point1 dominates point2 (higher accuracy, lower cost)
        maximize_map = {"accuracy": True, "cost": False}
        assert point1.dominates(point2, maximize_map)
        assert not point2.dominates(point1, maximize_map)

    def test_pareto_front_calculation_mixed(self):
        """Test Pareto front calculation with mixed orientations."""
        from datetime import datetime

        trials = [
            TrialResult(
                trial_id="t1",
                config={},
                metrics={"accuracy": 0.9, "cost": 0.10},
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t2",
                config={},
                metrics={"accuracy": 0.85, "cost": 0.05},
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t3",
                config={},
                metrics={"accuracy": 0.8, "cost": 0.15},  # Dominated
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=datetime.now(),
            ),
        ]

        schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
        )
        calculator = ParetoFrontCalculator(objective_schema=schema)

        front = calculator.calculate_pareto_front(trials, ["accuracy", "cost"])

        # t1 and t2 should be on front, t3 is dominated
        assert len(front) == 2
        front_ids = {p.trial.trial_id for p in front}
        assert "t1" in front_ids
        assert "t2" in front_ids
        assert "t3" not in front_ids

    def test_epsilon_tolerance_in_dominance(self):
        """Test epsilon tolerance in Pareto dominance."""
        # Points that are nearly equal
        point1 = ParetoPoint(
            config={}, objectives={"accuracy": 0.9000001, "cost": 0.05}, trial=None
        )
        point2 = ParetoPoint(
            config={}, objectives={"accuracy": 0.9, "cost": 0.0500001}, trial=None
        )

        maximize_map = {"accuracy": True, "cost": False}

        # With epsilon tolerance, neither dominates (values too close)
        assert not point1.dominates(point2, maximize_map, epsilon=1e-5)
        assert not point2.dominates(point1, maximize_map, epsilon=1e-5)


class TestDecoratorPrecedenceWarnings:
    """Tests for decorator precedence and warnings."""

    def test_objective_kwargs_rejected_with_schema(self):
        """Legacy objective kwargs should raise when ObjectiveSchema is used."""
        schema = create_default_objectives(["accuracy", "cost"])

        with pytest.raises(
            TypeError,
            match="Unknown keyword arguments.*objective_orientations",
        ):

            @optimize(
                objectives=schema,
                objective_orientations={"cost": "minimize"},
                configuration_space={"temperature": [0.1, 0.5, 0.9]},
            )
            def dummy_function():
                pass

    def test_legacy_objective_kwargs_rejected_for_lists(self):
        """Legacy orientation/weight kwargs are no longer supported for lists."""
        with pytest.raises(
            TypeError,
            match="Unknown keyword arguments.*objective_weights",
        ):

            @optimize(
                objectives=["accuracy", "cost"],
                objective_weights={"accuracy": 0.7, "cost": 0.3},
                configuration_space={"temperature": [0.1, 0.5, 0.9]},
            )
            def dummy_function():
                pass

    @pytest.mark.asyncio
    async def test_runtime_objective_kwargs_rejected(self):
        """Passing legacy kwargs to optimize() should raise an error."""

        @optimize(
            eval_dataset="examples/datasets/multi-objective-tradeoff/evaluation_set.jsonl",
            objectives=["accuracy"],
            configuration_space={"temperature": [0.0]},
        )
        def decorated(question: str) -> str:
            return "42"

        with pytest.raises(
            ValueError,
            match="objective_orientations/objective_weights are no longer supported",
        ):
            await decorated.optimize(objective_weights={"accuracy": 1.0})

    def test_no_warning_with_objectiveschema_only(self):
        """Test no warnings when using ObjectiveSchema correctly."""
        schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"cost": "minimize"},
            weights={"accuracy": 0.7, "cost": 0.3},
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @optimize(
                objectives=schema,
                configuration_space={"temperature": [0.1, 0.5, 0.9]},  # Required
            )
            def dummy_function():
                pass

            # Should not have any deprecation or precedence warnings
            # (might have other warnings from the decorator internals)
            precedence_warnings = [
                warning
                for warning in w
                if "ObjectiveSchema takes precedence" in str(warning.message)
            ]
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(precedence_warnings) == 0
            assert len(deprecation_warnings) == 0


class TestSerializationRoundTrips:
    """Tests for serialization and deserialization round trips."""

    def test_objective_definition_round_trip(self):
        """Test ObjectiveDefinition to_dict and from_dict round trip."""
        original = ObjectiveDefinition(
            name="f1_score",
            orientation="maximize",
            weight=0.7,
            normalization="z_score",
            bounds=(0.0, 1.0),
            unit="ratio",
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = ObjectiveDefinition.from_dict(data)

        # Verify all fields match
        assert restored.name == original.name
        assert restored.orientation == original.orientation
        assert restored.weight == original.weight
        assert restored.normalization == original.normalization
        assert restored.bounds == original.bounds
        assert restored.unit == original.unit

    def test_objective_schema_json_round_trip(self):
        """Test ObjectiveSchema JSON serialization round trip."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.5, bounds=(0, 1)),
            ObjectiveDefinition("cost", "minimize", 0.3, unit="USD"),
            ObjectiveDefinition("latency", "minimize", 0.2, normalization="robust"),
        ]
        original = ObjectiveSchema.from_objectives(objectives, schema_version="1.2.3")

        # JSON round trip
        json_str = original.to_json()
        restored = ObjectiveSchema.from_json(json_str)

        # Verify schema matches
        assert len(restored.objectives) == 3
        assert restored.schema_version == "1.2.3"
        assert restored.weights_sum == 1.0

        # Verify each objective
        for orig_obj, rest_obj in zip(
            original.objectives, restored.objectives, strict=False
        ):
            assert rest_obj.name == orig_obj.name
            assert rest_obj.orientation == orig_obj.orientation
            assert rest_obj.weight == orig_obj.weight
            assert rest_obj.normalization == orig_obj.normalization
            assert rest_obj.bounds == orig_obj.bounds
            assert rest_obj.unit == orig_obj.unit

    def test_complex_schema_persistence(self):
        """Test that complex schemas can be saved and loaded correctly."""
        # Create complex schema with all features
        objectives = [
            ObjectiveDefinition(
                name=f"metric_{i}",
                orientation="maximize" if i % 2 == 0 else "minimize",
                weight=1.0 / 10,  # Equal weights
                normalization=["min_max", "z_score", "robust"][i % 3],
                bounds=(0, 100 * (i + 1)) if i % 2 == 0 else None,
                unit=["percentage", "ms", "MB", None][i % 4],
            )
            for i in range(10)
        ]

        original = ObjectiveSchema.from_objectives(objectives)

        # Simulate file save/load
        data = original.to_dict()
        json_str = json.dumps(data, indent=2)
        loaded_data = json.loads(json_str)
        restored = ObjectiveSchema.from_dict(loaded_data)

        # Verify all objectives preserved
        assert len(restored.objectives) == 10
        for orig, rest in zip(original.objectives, restored.objectives, strict=False):
            assert rest.to_dict() == orig.to_dict()

        # Verify computed fields
        assert abs(restored.weights_sum - original.weights_sum) < 1e-10
        assert restored.weights_normalized == original.weights_normalized


class TestMultiObjectiveScalarization:
    """Tests for multi-objective scalarization with orientations."""

    def test_scalarize_with_mixed_orientations(self):
        """Test scalarization with mixed maximize/minimize objectives."""
        objectives_dict = {"accuracy": 0.9, "cost": 0.05, "latency": 50}
        weights = {"accuracy": 0.5, "cost": 0.3, "latency": 0.2}
        minimize_objectives = ["cost", "latency"]

        # Scalarize with mixed orientations
        score = scalarize_objectives(
            objectives_dict, weights, minimize_objectives=minimize_objectives
        )

        # Manually compute expected score
        expected = (
            0.5 * 0.9  # accuracy (maximize)
            + 0.3 * (-0.05)  # cost (minimize, negated)
            + 0.2 * (-50)  # latency (minimize, negated)
        )
        assert abs(score - expected) < 1e-10

    def test_scalarize_with_objective_schema(self):
        """Test scalarization using ObjectiveSchema."""
        schema = create_default_objectives(
            ["accuracy", "cost", "latency"],
            orientations={"cost": "minimize", "latency": "minimize"},
            weights={"accuracy": 0.5, "cost": 0.3, "latency": 0.2},
        )

        objectives_dict = {"accuracy": 0.9, "cost": 0.05, "latency": 50}

        # Scalarize using schema
        score = scalarize_objectives(
            objectives_dict,
            weights={},  # Will be overridden by schema
            objective_schema=schema,
        )

        # Same expected result as above
        expected = (
            0.5 * 0.9  # accuracy (maximize)
            + 0.3 * (-0.05)  # cost (minimize)
            + 0.2 * (-50)  # latency (minimize)
        )
        assert abs(score - expected) < 1e-10
