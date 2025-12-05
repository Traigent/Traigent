"""Tests for optimization logger with objective schema integration."""

import json
import tempfile
from pathlib import Path

import pytest

from traigent.core.objectives import (
    ObjectiveDefinition,
    ObjectiveSchema,
    create_default_objectives,
)
from traigent.utils.optimization_logger import OptimizationLogger


class TestOptimizationLoggerObjectives:
    """Test OptimizationLogger with ObjectiveSchema integration."""

    def test_log_session_with_objective_schema(self, tmp_path):
        """Test logging session with ObjectiveSchema."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="test_experiment",
            session_id="test_session_123",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Create objective schema
        objectives = [
            ObjectiveDefinition("accuracy", "maximize", 0.7),
            ObjectiveDefinition("cost", "minimize", 0.3),
        ]
        schema = ObjectiveSchema.from_objectives(objectives)

        # Log session start with schema
        logger.log_session_start(
            config={"model": "gpt-4o", "temperature": 0.7},
            objectives=schema,
            algorithm="bayesian",
        )

        # Verify objectives file was created
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        assert objectives_file.exists()

        # Load and verify objectives data
        with open(objectives_file) as f:
            data = json.load(f)

        assert "objectives" in data
        assert len(data["objectives"]) == 2
        objectives_by_name = {obj["name"]: obj for obj in data["objectives"]}
        assert objectives_by_name["accuracy"]["weight"] == 0.7
        assert objectives_by_name["cost"]["weight"] == 0.3
        assert data["weights_sum"] == pytest.approx(1.0)
        assert data["weights_normalized"]["accuracy"] == pytest.approx(0.7)
        assert data["weights_normalized"]["cost"] == pytest.approx(0.3)

    def test_load_checkpoint_uses_default_base_path(self, monkeypatch, tmp_path):
        """load_checkpoint should resolve default base path when none provided."""
        # Create temporary base path with required structure
        base_path = Path(tempfile.mkdtemp())
        experiments = base_path / "experiments" / "exp_default"
        run_path = experiments / "runs" / "run_001"
        checkpoints = run_path / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)

        latest_filename = "latest_checkpoint_v2.json"
        checkpoint_filename = "checkpoint_00001_v2.json"
        trial_history_filename = "trial_history_v2.json"

        (checkpoints / latest_filename).write_text(
            json.dumps({"checkpoint_file": checkpoint_filename})
        )
        (checkpoints / checkpoint_filename).write_text(
            json.dumps({"trial_count": 5, "state": {"foo": "bar"}})
        )
        (checkpoints / trial_history_filename).write_text(
            json.dumps([{"trial": 5, "status": "completed"}])
        )

        base_path_env = base_path
        monkeypatch.setenv(
            OptimizationLogger.ENV_BASE_PATH,
            str(base_path_env),
        )

        loaded = OptimizationLogger.load_checkpoint("exp_default", "run_001")

        assert loaded["trial_count"] == 5
        assert loaded["state"]["foo"] == "bar"
        assert loaded["trial_history"] == [{"trial": 5, "status": "completed"}]
        assert loaded["run_path"] == str(run_path)

    def test_log_session_with_objective_list(self, tmp_path):
        """Test logging session with list of objectives and auto-creation of schema."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="test_experiment",
            session_id="test_session_456",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Log session start with list of objectives
        logger.log_session_start(
            config={"model": "claude-3-5-sonnet"},
            objectives=["accuracy", "cost", "latency"],
            algorithm="grid_search",
        )

        # Verify objectives file was created
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        assert objectives_file.exists()

        # Load and verify objectives data
        with open(objectives_file) as f:
            data = json.load(f)

        assert len(data["objectives"]) == 3

        # Check orientations
        objectives_by_name = {obj["name"]: obj for obj in data["objectives"]}
        assert objectives_by_name["accuracy"]["orientation"] == "maximize"
        assert objectives_by_name["cost"]["orientation"] == "minimize"
        assert objectives_by_name["latency"]["orientation"] == "minimize"

        # Check weights
        assert objectives_by_name["accuracy"]["weight"] == 1.0
        assert objectives_by_name["cost"]["weight"] == 1.0
        assert objectives_by_name["latency"]["weight"] == 1.0

        # Check normalized weights
        assert data["weights_sum"] == 3.0
        assert (
            pytest.approx(data["weights_normalized"]["accuracy"], rel=1e-6) == 1.0 / 3.0
        )
        assert pytest.approx(data["weights_normalized"]["cost"], rel=1e-6) == 1.0 / 3.0
        assert (
            pytest.approx(data["weights_normalized"]["latency"], rel=1e-6) == 1.0 / 3.0
        )

    def test_log_session_with_default_orientations(self, tmp_path):
        """Test that default orientations are applied correctly."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="test_experiment",
            session_id="test_session_789",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Log session start without explicit orientations
        logger.log_session_start(
            config={"model": "gpt-4o"},
            objectives=["accuracy", "cost", "custom_metric"],
            algorithm="random_search",
        )

        # Load objectives data
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        with open(objectives_file) as f:
            data = json.load(f)

        # Check default orientations
        objectives_by_name = {obj["name"]: obj for obj in data["objectives"]}
        assert (
            objectives_by_name["accuracy"]["orientation"] == "maximize"
        )  # Default for accuracy
        assert (
            objectives_by_name["cost"]["orientation"] == "minimize"
        )  # Default for cost
        assert (
            objectives_by_name["custom_metric"]["orientation"] == "maximize"
        )  # Default for unknown

        # Check equal weights were assigned
        assert abs(data["weights_normalized"]["accuracy"] - 1 / 3) < 1e-10
        assert abs(data["weights_normalized"]["cost"] - 1 / 3) < 1e-10
        assert abs(data["weights_normalized"]["custom_metric"] - 1 / 3) < 1e-10

    def test_backward_compatibility(self, tmp_path):
        """Test that the logger still works with legacy code."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="legacy_test",
            session_id="legacy_123",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Log session start with just a list of objectives (legacy style)
        logger.log_session_start(
            config={"model": "gpt-3.5-turbo"},
            objectives=["accuracy", "speed"],
            algorithm="bayesian",
        )

        # Should still create a valid objectives file
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        assert objectives_file.exists()

        # Load and verify it has the new schema format
        with open(objectives_file) as f:
            data = json.load(f)

        assert "objectives" in data
        assert "weights_sum" in data
        assert "weights_normalized" in data
        assert "schema_version" in data

        # Check that objectives were created with defaults
        assert len(data["objectives"]) == 2
        objectives_by_name = {obj["name"]: obj for obj in data["objectives"]}
        assert objectives_by_name["accuracy"]["orientation"] == "maximize"
        assert (
            objectives_by_name["speed"]["orientation"] == "maximize"
        )  # Unknown defaults to maximize

    def test_complex_objective_schema(self, tmp_path):
        """Test logging with complex objective definitions including bounds and units."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="complex_test",
            session_id="complex_123",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Create complex objective schema
        objectives = [
            ObjectiveDefinition(
                name="accuracy",
                orientation="maximize",
                weight=0.5,
                bounds=(0.0, 1.0),
                unit="percentage",
            ),
            ObjectiveDefinition(
                name="cost",
                orientation="minimize",
                weight=0.3,
                bounds=(0.0, 0.1),
                unit="USD",
            ),
            ObjectiveDefinition(
                name="latency",
                orientation="minimize",
                weight=0.2,
                normalization="z_score",
                bounds=(0, 1000),
                unit="ms",
            ),
        ]
        schema = ObjectiveSchema.from_objectives(objectives)

        # Log session
        logger.log_session_start(
            config={"model": "claude-3-opus"},
            objectives=schema,
            algorithm="multi_objective_bayesian",
        )

        # Load and verify
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        with open(objectives_file) as f:
            data = json.load(f)

        # Verify complex fields are preserved
        objectives_by_name = {obj["name"]: obj for obj in data["objectives"]}

        assert objectives_by_name["accuracy"]["bounds"] == [0.0, 1.0]
        assert objectives_by_name["accuracy"]["unit"] == "percentage"

        assert objectives_by_name["cost"]["bounds"] == [0.0, 0.1]
        assert objectives_by_name["cost"]["unit"] == "USD"

        assert objectives_by_name["latency"]["normalization"] == "z_score"
        assert objectives_by_name["latency"]["bounds"] == [0, 1000]
        assert objectives_by_name["latency"]["unit"] == "ms"

    def test_objectives_file_can_be_reloaded(self, tmp_path):
        """Test that saved objectives file can be reloaded as ObjectiveSchema."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="reload_test",
            session_id="reload_123",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Create and save schema
        original_schema = create_default_objectives(
            ["accuracy", "cost", "latency"],
            orientations={"latency": "minimize", "cost": "minimize"},
            weights={"accuracy": 0.6, "cost": 0.3, "latency": 0.1},
        )

        logger.log_session_start(
            config={"model": "test"}, objectives=original_schema, algorithm="test"
        )

        # Load the saved file
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        with open(objectives_file) as f:
            data = json.load(f)

        # Remove extra fields added by logger
        data.pop("algorithm", None)
        data.pop("timestamp", None)

        # Reload as ObjectiveSchema
        loaded_schema = ObjectiveSchema.from_dict(data)

        # Verify it matches original
        assert len(loaded_schema.objectives) == 3
        assert loaded_schema.get_orientation("accuracy") == "maximize"
        assert loaded_schema.get_orientation("cost") == "minimize"
        assert loaded_schema.get_orientation("latency") == "minimize"
        assert loaded_schema.get_normalized_weight("accuracy") == pytest.approx(0.6)
        assert loaded_schema.get_normalized_weight("cost") == pytest.approx(0.3)
        assert loaded_schema.get_normalized_weight("latency") == pytest.approx(0.1)

    def test_enhanced_metadata_persistence(self, tmp_path):
        """Test that enhanced metadata fields are persisted correctly."""
        # Create logger
        logger = OptimizationLogger(
            experiment_name="metadata_test",
            session_id="metadata_123",
            execution_mode="edge_analytics",
            base_path=tmp_path,
        )

        # Create objective schema with all features
        objectives = [
            ObjectiveDefinition(
                name="f1_score",
                orientation="maximize",
                weight=0.4,
                bounds=(0.0, 1.0),
                unit="ratio",
            ),
            ObjectiveDefinition(
                name="inference_time",
                orientation="minimize",
                weight=0.3,
                bounds=(0, 100),
                unit="ms",
            ),
            ObjectiveDefinition(
                name="memory_usage",
                orientation="minimize",
                weight=0.3,
                bounds=(0, 1024),
                unit="MB",
            ),
        ]
        schema = ObjectiveSchema.from_objectives(objectives)

        # Log session
        logger.log_session_start(
            config={"model": "test"}, objectives=schema, algorithm="pareto_optimization"
        )

        # Load and verify enhanced metadata
        objectives_file = (
            logger.run_path / "meta" / logger.file_manager.get_filename("objectives")
        )
        with open(objectives_file) as f:
            data = json.load(f)

        # Check metadata section
        assert "metadata" in data
        metadata = data["metadata"]
        assert metadata["algorithm"] == "pareto_optimization"
        assert metadata["normalization_strategy"] == "min_max"
        assert metadata["weights_normalized"] is True
        assert metadata["schema_version"] == "1.0.0"
        assert "timestamp" in metadata

        # Check summary section
        assert "summary" in data
        summary = data["summary"]

        # Verify names list
        assert summary["names"] == ["f1_score", "inference_time", "memory_usage"]

        # Verify orientations dict
        assert summary["orientations"] == {
            "f1_score": "maximize",
            "inference_time": "minimize",
            "memory_usage": "minimize",
        }

        # Verify weights dict
        assert summary["weights"] == {
            "f1_score": 0.4,
            "inference_time": 0.3,
            "memory_usage": 0.3,
        }

        # Verify normalized weights
        assert summary["normalized_weights"] == {
            "f1_score": 0.4,
            "inference_time": 0.3,
            "memory_usage": 0.3,
        }
        assert summary["weights_sum"] == 1.0

        # Verify bounds
        assert summary["bounds"] == {
            "f1_score": [0.0, 1.0],
            "inference_time": [0, 100],
            "memory_usage": [0, 1024],
        }

        # Verify top-level backward compatibility fields
        assert data["algorithm"] == "pareto_optimization"
        assert "timestamp" in data

        # Verify core schema fields are still present
        assert "objectives" in data
        assert "weights_sum" in data
        assert "weights_normalized" in data
        assert "schema_version" in data
