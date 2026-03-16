"""Tests for objective definitions and validation."""

import json

import pytest

from traigent.core.objectives import (
    ObjectiveDefinition,
    ObjectiveSchema,
    create_default_objectives,
)


class TestObjectiveDefinition:
    """Test ObjectiveDefinition class."""

    def test_valid_objective_creation(self):
        """Test creating a valid objective definition."""
        obj = ObjectiveDefinition(
            name="accuracy",
            orientation="maximize",
            weight=0.7,
            normalization="min_max",
            bounds=(0.0, 1.0),
            unit="percentage",
        )

        assert obj.name == "accuracy"
        assert obj.orientation == "maximize"
        assert obj.weight == 0.7
        assert obj.normalization == "min_max"
        assert obj.bounds == (0.0, 1.0)
        assert obj.unit == "percentage"

    def test_negative_weight_validation(self):
        """Test that negative weights are rejected."""
        with pytest.raises(ValueError, match="finite positive"):
            ObjectiveDefinition(name="cost", orientation="minimize", weight=-0.5)

    def test_zero_weight_validation(self):
        """Test that zero weights are rejected."""
        with pytest.raises(ValueError, match="finite positive"):
            ObjectiveDefinition(name="cost", orientation="minimize", weight=0.0)

    def test_invalid_orientation_validation(self):
        """Test that invalid orientations are rejected."""
        with pytest.raises(
            ValueError, match="Orientation must be 'maximize', 'minimize', or 'band'"
        ):
            ObjectiveDefinition(
                name="accuracy",
                orientation="optimize",
                weight=1.0,  # Invalid
            )

    def test_invalid_bounds_validation(self):
        """Test bounds validation."""
        # Min >= Max
        with pytest.raises(ValueError, match="min .* must be less than max"):
            ObjectiveDefinition(
                name="accuracy",
                orientation="maximize",
                weight=1.0,
                bounds=(1.0, 0.0),  # Invalid: min > max
            )

        # Wrong tuple size
        with pytest.raises(ValueError, match="Bounds must be a tuple"):
            ObjectiveDefinition(
                name="accuracy",
                orientation="maximize",
                weight=1.0,
                bounds=(0.0,),  # Invalid: single value
            )

    def test_invalid_normalization(self):
        """Test that invalid normalization strategies are rejected."""
        with pytest.raises(ValueError, match="Normalization must be one of"):
            ObjectiveDefinition(
                name="accuracy",
                orientation="maximize",
                weight=1.0,
                normalization="invalid",
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        obj = ObjectiveDefinition(
            name="cost",
            orientation="minimize",
            weight=0.3,
            bounds=(0.0, 100.0),
            unit="USD",
        )

        data = obj.to_dict()
        assert data == {
            "name": "cost",
            "orientation": "minimize",
            "weight": 0.3,
            "normalization": "min_max",
            "bounds": [0.0, 100.0],
            "unit": "USD",
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "latency",
            "orientation": "minimize",
            "weight": 0.2,
            "normalization": "z_score",
            "bounds": [0.0, 5.0],
            "unit": "seconds",
        }

        obj = ObjectiveDefinition.from_dict(data)
        assert obj.name == "latency"
        assert obj.orientation == "minimize"
        assert obj.weight == 0.2
        assert obj.normalization == "z_score"
        assert obj.bounds == (0.0, 5.0)
        assert obj.unit == "seconds"


class TestObjectiveSchema:
    """Test ObjectiveSchema class."""

    def test_schema_creation_from_objectives(self):
        """Test creating schema from objectives with auto-normalization."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        assert len(schema.objectives) == 2
        assert schema.weights_sum == 1.0
        assert schema.weights_normalized["accuracy"] == 0.7
        assert schema.weights_normalized["cost"] == 0.3
        assert schema.schema_version == "1.0.0"

    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 2.0),
            ObjectiveDefinition("cost", "minimize", 1.0),
            ObjectiveDefinition("latency", "minimize", 1.0),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        assert schema.weights_sum == 4.0
        assert schema.weights_normalized["accuracy"] == 0.5  # 2.0/4.0
        assert schema.weights_normalized["cost"] == 0.25  # 1.0/4.0
        assert schema.weights_normalized["latency"] == 0.25  # 1.0/4.0

        # Verify normalized weights sum to 1.0
        total = sum(schema.weights_normalized.values())
        assert abs(total - 1.0) < 1e-10

    def test_empty_objectives_validation(self):
        """Test that empty objectives list is rejected."""
        with pytest.raises(ValueError, match="At least one objective must be"):
            ObjectiveSchema.from_objectives([])

    def test_duplicate_objective_names(self):
        """Test that duplicate objective names are rejected."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.5),
            ObjectiveDefinition("accuracy", "maximize", 0.5),  # Duplicate
        ]

        with pytest.raises(ValueError, match="Duplicate objective names found"):
            ObjectiveSchema.from_objectives(objectives)

    def test_weights_sum_validation(self):
        """Test validation of weights_sum consistency."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]

        # Create with incorrect weights_sum
        with pytest.raises(ValueError, match="weights_sum .* doesn't match"):
            ObjectiveSchema(
                objectives=objectives,
                weights_sum=2.0,  # Incorrect sum
                weights_normalized={"accuracy": 0.7, "cost": 0.3},
            )

    def test_normalized_weights_validation(self):
        """Test validation of normalized weights consistency."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]

        # Create with incorrect normalized weights
        with pytest.raises(ValueError, match="Normalized weight .* is incorrect"):
            ObjectiveSchema(
                objectives=objectives,
                weights_sum=1.0,
                weights_normalized={"accuracy": 0.5, "cost": 0.5},  # Incorrect
            )

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        objectives = [
            ObjectiveDefinition(
                name="accuracy",
                orientation="maximize",
                weight=0.6,
                bounds=(0.0, 1.0),
                unit="percentage",
            ),
            ObjectiveDefinition(
                name="cost", orientation="minimize", weight=0.4, unit="USD"
            ),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        # Serialize to JSON
        json_str = schema.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert "objectives" in data
        assert "weights_sum" in data
        assert "weights_normalized" in data
        assert "schema_version" in data

        # Deserialize from JSON
        schema2 = ObjectiveSchema.from_json(json_str)

        # Verify round-trip
        assert len(schema2.objectives) == 2
        assert schema2.weights_sum == 1.0
        assert schema2.weights_normalized["accuracy"] == 0.6
        assert schema2.weights_normalized["cost"] == 0.4
        assert schema2.objectives[0].bounds == (0.0, 1.0)
        assert schema2.objectives[0].unit == "percentage"

    def test_get_objective(self):
        """Test getting objective by name."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        # Get existing objective
        obj = schema.get_objective("accuracy")
        assert obj is not None
        assert obj.name == "accuracy"
        assert obj.orientation == "maximize"

        # Get non-existent objective
        obj = schema.get_objective("latency")
        assert obj is None

    def test_get_orientation(self):
        """Test getting orientation by objective name."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        assert schema.get_orientation("accuracy") == "maximize"
        assert schema.get_orientation("cost") == "minimize"
        assert schema.get_orientation("latency") is None

    def test_get_normalized_weight(self):
        """Test getting normalized weight by objective name."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 2.0),
            ObjectiveDefinition("cost", "minimize", 1.0),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        assert abs(schema.get_normalized_weight("accuracy") - 2.0 / 3.0) < 1e-10
        assert abs(schema.get_normalized_weight("cost") - 1.0 / 3.0) < 1e-10
        assert schema.get_normalized_weight("latency") == 0.0

    def test_validate_metrics(self):
        """Test metrics validation against schema."""
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7, bounds=(0.0, 1.0)),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        # Valid metrics
        errors = schema.validate_metrics({"accuracy": 0.9, "cost": 0.01})
        assert len(errors) == 0

        # Missing metric
        errors = schema.validate_metrics({"accuracy": 0.9})
        assert len(errors) == 1
        assert "Missing metric for objective 'cost'" in errors[0]

        # Extra metric
        errors = schema.validate_metrics(
            {"accuracy": 0.9, "cost": 0.01, "latency": 1.0}
        )
        assert len(errors) == 1
        assert "Metric 'latency' is not a defined objective" in errors[0]

        # Out of bounds
        errors = schema.validate_metrics({"accuracy": 1.5, "cost": 0.01})
        assert len(errors) == 1
        assert "outside bounds" in errors[0]


class TestCreateDefaultObjectives:
    """Test create_default_objectives helper function."""

    def test_default_objectives_creation(self):
        """Test creating objectives with defaults."""
        schema = create_default_objectives(["accuracy", "cost", "latency"])

        assert len(schema.objectives) == 3

        # Check default orientations
        assert schema.get_orientation("accuracy") == "maximize"
        assert schema.get_orientation("cost") == "minimize"
        assert schema.get_orientation("latency") == "minimize"

        # Check equal default weights
        assert schema.get_normalized_weight("accuracy") == 1.0 / 3.0
        assert schema.get_normalized_weight("cost") == 1.0 / 3.0
        assert schema.get_normalized_weight("latency") == 1.0 / 3.0

    def test_custom_orientations_and_weights(self):
        """Test creating objectives with custom orientations and weights."""
        schema = create_default_objectives(
            ["accuracy", "cost", "custom_metric"],
            orientations={"custom_metric": "minimize"},
            weights={"accuracy": 0.5, "cost": 0.3, "custom_metric": 0.2},
        )

        assert schema.get_orientation("accuracy") == "maximize"
        assert schema.get_orientation("cost") == "minimize"
        assert schema.get_orientation("custom_metric") == "minimize"

        assert schema.get_normalized_weight("accuracy") == 0.5
        assert schema.get_normalized_weight("cost") == 0.3
        assert schema.get_normalized_weight("custom_metric") == 0.2

    def test_percentage_style_weights_are_normalized(self):
        """Test that percentage-style weights are normalized automatically."""
        schema = create_default_objectives(
            ["accuracy", "cost"],
            weights={"accuracy": 70.0, "cost": 30.0},
        )

        assert schema.weights_sum == pytest.approx(100.0)
        assert schema.get_normalized_weight("accuracy") == pytest.approx(0.7)
        assert schema.get_normalized_weight("cost") == pytest.approx(0.3)

    def test_unknown_metric_defaults(self):
        """Test that unknown metrics default to maximize."""
        schema = create_default_objectives(["unknown_metric"])

        assert schema.get_orientation("unknown_metric") == "maximize"
        assert schema.get_normalized_weight("unknown_metric") == 1.0

    def test_empty_names_validation(self):
        """Test that empty names list is rejected."""
        with pytest.raises(ValueError, match="At least one objective name"):
            create_default_objectives([])


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_multi_objective_cost_accuracy_scenario(self):
        """Test a typical cost vs accuracy optimization scenario."""
        # Define objectives with realistic settings
        objectives = [
            ObjectiveDefinition(
                name="accuracy",
                orientation="maximize",
                weight=0.7,
                bounds=(0.0, 1.0),
                unit="percentage",
            ),
            ObjectiveDefinition(
                name="cost",
                orientation="minimize",
                weight=0.3,
                bounds=(0.0, 0.1),  # $0 to $0.10 per request
                unit="USD",
            ),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        # Simulate trial metrics
        trial_metrics = [
            {"accuracy": 0.95, "cost": 0.08},  # High accuracy, high cost
            {"accuracy": 0.85, "cost": 0.02},  # Medium accuracy, low cost
            {"accuracy": 0.75, "cost": 0.01},  # Low accuracy, very low cost
        ]

        # Validate all metrics
        for metrics in trial_metrics:
            errors = schema.validate_metrics(metrics)
            assert len(errors) == 0

        # Check that orientations are correct for decision making
        assert schema.get_orientation("accuracy") == "maximize"  # Want high accuracy
        assert schema.get_orientation("cost") == "minimize"  # Want low cost

    def test_performance_optimization_scenario(self):
        """Test a performance optimization scenario with multiple metrics."""
        objectives = [
            ObjectiveDefinition("throughput", "maximize", 0.4, unit="requests/sec"),
            ObjectiveDefinition(
                "latency", "minimize", 0.3, bounds=(0, 1000), unit="ms"
            ),
            ObjectiveDefinition(
                "error_rate", "minimize", 0.2, bounds=(0, 0.1), unit="percentage"
            ),
            ObjectiveDefinition("memory", "minimize", 0.1, unit="MB"),
        ]

        schema = ObjectiveSchema.from_objectives(objectives)

        # Verify all orientations are correct
        assert schema.get_orientation("throughput") == "maximize"
        assert schema.get_orientation("latency") == "minimize"
        assert schema.get_orientation("error_rate") == "minimize"
        assert schema.get_orientation("memory") == "minimize"

        # Verify weights sum to 1.0
        total_weight = sum(schema.weights_normalized.values())
        assert abs(total_weight - 1.0) < 1e-10

        # Test persistence and loading
        json_str = schema.to_json()
        loaded_schema = ObjectiveSchema.from_json(json_str)

        # Verify loaded schema matches original
        assert len(loaded_schema.objectives) == 4
        assert loaded_schema.get_normalized_weight("throughput") == 0.4
        assert loaded_schema.objectives[1].bounds == (0, 1000)


class TestWeightValidationHardening:
    """Tests for hardened weight validation."""

    def test_nan_weight_rejected(self):
        with pytest.raises(ValueError, match="finite positive"):
            ObjectiveDefinition("x", "maximize", float("nan"))

    def test_inf_weight_rejected(self):
        with pytest.raises(ValueError, match="finite positive"):
            ObjectiveDefinition("x", "maximize", float("inf"))

    def test_neg_inf_weight_rejected(self):
        with pytest.raises(ValueError, match="finite positive"):
            ObjectiveDefinition("x", "maximize", float("-inf"))

    def test_multi_objective_no_single_dominance(self):
        """No single objective can have 100% weight in multi-objective."""
        with pytest.raises(ValueError, match="100% of the weight"):
            ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition("a", "maximize", 1000000),
                    ObjectiveDefinition("b", "maximize", 0.0001),
                ]
            )

    def test_single_objective_weight_1_allowed(self):
        """Single objective with weight=1.0 is valid."""
        schema = ObjectiveSchema.from_objectives(
            [ObjectiveDefinition("a", "maximize", 1.0)]
        )
        assert abs(schema.get_normalized_weight("a") - 1.0) < 1e-10

    def test_from_dict_null_weight_defaults(self):
        """from_dict with null weight uses default 1.0."""
        obj = ObjectiveDefinition.from_dict(
            {"name": "x", "orientation": "maximize", "weight": None}
        )
        assert abs(obj.weight - 1.0) < 1e-10

    def test_from_dict_zero_weight_rejected(self):
        """from_dict with weight=0 is rejected."""
        with pytest.raises(ValueError, match="finite positive"):
            ObjectiveDefinition.from_dict(
                {"name": "x", "orientation": "maximize", "weight": 0}
            )

    def test_arbitrary_positive_weights_normalized(self):
        """Large weights are accepted and normalized."""
        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition("a", "maximize", 100),
                ObjectiveDefinition("b", "maximize", 50),
            ]
        )
        assert abs(schema.get_normalized_weight("a") - 2 / 3) < 1e-10
        assert abs(schema.get_normalized_weight("b") - 1 / 3) < 1e-10

    def test_already_normalized_weights_accepted(self):
        """Weights summing to 1.0 pass through without issue."""
        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition("a", "maximize", 0.3),
                ObjectiveDefinition("b", "maximize", 0.5),
                ObjectiveDefinition("c", "maximize", 0.2),
            ]
        )
        assert abs(schema.weights_sum - 1.0) < 1e-10

    def test_balanced_multi_objective_accepted(self):
        """Reasonably balanced multi-objective weights pass."""
        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition("a", "maximize", 0.9),
                ObjectiveDefinition("b", "maximize", 0.1),
            ]
        )
        assert abs(schema.get_normalized_weight("a") - 0.9) < 1e-10
        assert abs(schema.get_normalized_weight("b") - 0.1) < 1e-10
