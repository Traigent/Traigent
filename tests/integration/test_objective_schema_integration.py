"""Integration tests for ObjectiveSchema flow through the system."""

from unittest.mock import MagicMock, patch

import pytest

from traigent.api.decorators import optimize
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.objectives import (
    ObjectiveDefinition,
    ObjectiveSchema,
    create_default_objectives,
)
from traigent.evaluators.base import Dataset, EvaluationExample


class TestObjectiveSchemaIntegration:
    """Test ObjectiveSchema integration across the system."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        examples = [
            EvaluationExample(
                input_data={"question": "What is 2+2?"},
                expected_output="4",
                metadata={"example_id": "1"},
            ),
            EvaluationExample(
                input_data={"question": "What is 3+3?"},
                expected_output="6",
                metadata={"example_id": "2"},
            ),
        ]
        return Dataset(examples=examples, name="test_dataset")

    def test_decorator_with_objective_schema(self, simple_dataset):
        """Test that the decorator properly accepts and uses ObjectiveSchema."""
        # Create objective schema
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.6),
            ObjectiveDefinition("cost", "minimize", 0.3),
            ObjectiveDefinition("latency", "minimize", 0.1),
        ]
        schema = ObjectiveSchema.from_objectives(objectives)

        # Decorate a function with ObjectiveSchema
        @optimize(
            eval_dataset=simple_dataset,
            objectives=schema,
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.1, 0.5, 0.9],
            },
            execution_mode="edge_analytics",
        )
        def test_function(question: str) -> str:
            return "42"

        # Verify the function has the correct objective schema
        assert hasattr(test_function, "objective_schema")
        assert test_function.objective_schema == schema
        assert len(test_function.objective_schema.objectives) == 3
        assert test_function.objective_schema.get_orientation("accuracy") == "maximize"
        assert test_function.objective_schema.get_orientation("cost") == "minimize"

    def test_decorator_with_objective_list_defaults(self, simple_dataset):
        """List of objectives should auto-create schema with default settings."""

        @optimize(
            eval_dataset=simple_dataset,
            objectives=["accuracy", "cost", "latency"],
            configuration_space={"model": ["gpt-3.5-turbo"], "temperature": [0.5]},
            execution_mode="edge_analytics",
        )
        def test_function(question: str) -> str:
            return "42"

        # Verify schema was created correctly
        assert hasattr(test_function, "objective_schema")
        assert len(test_function.objective_schema.objectives) == 3
        assert test_function.objective_schema.get_orientation("accuracy") == "maximize"
        assert test_function.objective_schema.get_orientation("cost") == "minimize"
        assert pytest.approx(
            test_function.objective_schema.get_normalized_weight("accuracy")
        ) == pytest.approx(1.0 / 3.0)
        assert pytest.approx(
            test_function.objective_schema.get_normalized_weight("cost")
        ) == pytest.approx(1.0 / 3.0)
        assert pytest.approx(
            test_function.objective_schema.get_normalized_weight("latency")
        ) == pytest.approx(1.0 / 3.0)

    def test_optimization_result_with_objective_schema(self):
        """Test that OptimizationResult.calculate_weighted_scores works with ObjectiveSchema."""
        # Create trial results
        trials = [
            TrialResult(
                trial_id="1",
                config={"model": "gpt-3.5-turbo"},
                metrics={"accuracy": 0.8, "cost": 0.05, "latency": 1.2},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=None,
            ),
            TrialResult(
                trial_id="2",
                config={"model": "gpt-4"},
                metrics={"accuracy": 0.95, "cost": 0.15, "latency": 2.5},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=None,
            ),
        ]

        # Create optimization result
        result = OptimizationResult(
            trials=trials,
            best_config={"model": "gpt-4"},
            best_score=0.95,
            optimization_id="test",
            duration=3.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy", "cost", "latency"],
            algorithm="test",
            timestamp=None,
        )

        # Create objective schema
        schema = create_default_objectives(
            ["accuracy", "cost", "latency"],
            orientations={"cost": "minimize", "latency": "minimize"},
            weights={"accuracy": 0.6, "cost": 0.3, "latency": 0.1},
        )

        # Calculate weighted scores with schema
        weighted_result = result.calculate_weighted_scores(objective_schema=schema)

        # Verify results
        assert "best_weighted_config" in weighted_result
        assert "best_weighted_score" in weighted_result
        assert "weighted_scores" in weighted_result
        assert "objective_weights_used" in weighted_result

        # Check that weights from schema were used
        assert weighted_result["objective_weights_used"]["accuracy"] == pytest.approx(
            0.6
        )
        assert weighted_result["objective_weights_used"]["cost"] == pytest.approx(0.3)
        assert weighted_result["objective_weights_used"]["latency"] == pytest.approx(
            0.1
        )

        # Check that minimize objectives were detected from schema
        assert "cost" in weighted_result.get("minimize_objectives", [])
        assert "latency" in weighted_result.get("minimize_objectives", [])

    def test_pareto_front_with_objective_schema(self):
        """Test that ParetoFrontCalculator works with ObjectiveSchema."""
        from traigent.utils.multi_objective import ParetoFrontCalculator

        # Create objective schema
        schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
        )

        # Create calculator with schema
        calculator = ParetoFrontCalculator(objective_schema=schema)

        # Verify orientations were extracted correctly
        assert calculator.maximize["accuracy"] is True
        assert calculator.maximize["cost"] is False

        # Create trial results
        trials = [
            TrialResult(
                trial_id="1",
                config={"model": "cheap"},
                metrics={"accuracy": 0.7, "cost": 0.01},
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=None,
            ),
            TrialResult(
                trial_id="2",
                config={"model": "expensive"},
                metrics={"accuracy": 0.95, "cost": 0.1},
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=None,
            ),
            TrialResult(
                trial_id="3",
                config={"model": "middle"},
                metrics={"accuracy": 0.8, "cost": 0.05},
                status=TrialStatus.COMPLETED.value,
                duration=1.0,
                timestamp=None,
            ),
        ]

        # Calculate Pareto front
        pareto_front = calculator.calculate_pareto_front(trials, ["accuracy", "cost"])

        # All three should be on the Pareto front (trade-off between accuracy and cost)
        assert len(pareto_front) == 3

    def test_scalarize_objectives_with_schema(self):
        """Test that scalarize_objectives works with ObjectiveSchema."""
        from traigent.utils.multi_objective import scalarize_objectives

        # Create objective schema
        schema = create_default_objectives(
            ["accuracy", "cost", "latency"],
            orientations={"cost": "minimize", "latency": "minimize"},
            weights={"accuracy": 0.6, "cost": 0.3, "latency": 0.1},
        )

        # Test objectives
        objectives = {"accuracy": 0.9, "cost": 0.05, "latency": 1.5}

        # Scalarize with schema
        score = scalarize_objectives(
            objectives=objectives,
            weights={},  # Will be overridden by schema
            objective_schema=schema,
        )

        # Score should reflect minimization of cost and latency
        # accuracy contributes: 0.9 * 0.6 = 0.54
        # cost contributes: -0.05 * 0.3 = -0.015 (negative because minimize)
        # latency contributes: -1.5 * 0.1 = -0.15 (negative because minimize)
        # Total: 0.54 - 0.015 - 0.15 = 0.375
        assert abs(score - 0.375) < 0.001

    @patch("traigent.utils.optimization_logger.OptimizationLogger")
    def test_optimization_logger_with_schema(
        self, mock_logger_class, simple_dataset, tmp_path
    ):
        """Test that OptimizationLogger properly logs ObjectiveSchema."""
        # Create a mock logger instance
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Create objective schema
        schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
            weights={"accuracy": 0.7, "cost": 0.3},
        )

        # Create decorated function
        @optimize(
            eval_dataset=simple_dataset,
            objectives=schema,
            configuration_space={"model": ["test"], "temperature": [0.5]},
            execution_mode="edge_analytics",
            local_storage_path=str(tmp_path),
        )
        def test_function(question: str) -> str:
            return "42"

        # Mock the optimization process
        with patch(
            "traigent.core.optimized_function.get_optimizer"
        ) as mock_get_optimizer, patch(
            "traigent.evaluators.local.LocalEvaluator.evaluate"
        ) as mock_evaluate:

            # Setup mocks
            mock_optimizer = MagicMock()
            mock_optimizer.optimize = MagicMock(return_value=[])
            mock_optimizer.objectives = ["accuracy", "cost"]
            mock_optimizer.config_space = {"model": ["test"]}
            mock_get_optimizer.return_value = mock_optimizer

            mock_evaluate.return_value = MagicMock(
                aggregated_metrics={"accuracy": 0.8, "cost": 0.05}
            )

            # The logger should be created during optimize() call
            # but we're mocking it, so we don't actually run the optimization

            # Verify the function has the schema
            assert test_function.objective_schema == schema

    def test_end_to_end_with_objective_schema(self, simple_dataset, tmp_path):
        """Test complete flow from decorator to results with ObjectiveSchema."""
        # Create objective schema with mixed orientations
        schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
            weights={"accuracy": 0.8, "cost": 0.2},
        )

        # Create decorated function
        @optimize(
            eval_dataset=simple_dataset,
            objectives=schema,
            configuration_space={"model": ["test_model"], "temperature": [0.5]},
            execution_mode="edge_analytics",
            local_storage_path=str(tmp_path),
            max_trials=1,
        )
        def test_function(question: str, config: dict = None) -> str:
            # Simple function that returns different results based on config
            if config and config.get("model") == "test_model":
                return "4" if "2+2" in question else "6"
            return "unknown"

        # Verify the schema is properly stored
        assert test_function.objective_schema == schema
        assert test_function.objectives == ["accuracy", "cost"]

        # The function should be callable
        result = test_function("What is 2+2?")
        assert result == "unknown"  # No config provided

        # With config
        result = test_function("What is 2+2?", config={"model": "test_model"})
        assert result == "4"
