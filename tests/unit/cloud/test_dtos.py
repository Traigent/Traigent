"""Comprehensive tests for traigent.cloud.dtos module.

Tests cover all Data Transfer Objects (DTOs) for OptiGen Backend Integration
with focus on schema compliance, serialization, and edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.cloud.dtos import (
    ConfigurationRunDTO,
    ConfigurationsDTO,
    ExperimentDTO,
    ExperimentRunDTO,
    InfrastructureDTO,
)


class TestInfrastructureDTO:
    """Test InfrastructureDTO dataclass."""

    def test_create_with_defaults(self):
        """Test creating InfrastructureDTO with default values."""
        infra = InfrastructureDTO()

        assert infra.infrastructure_id == "local-infra"
        assert infra.compute == "cpu"
        assert infra.memory == "8GB"
        assert infra.timeout == 3600
        assert infra.created_at is None
        assert infra.updated_at is None

    def test_create_with_custom_values(self):
        """Test creating InfrastructureDTO with custom values."""
        now = datetime.now(UTC).isoformat()

        infra = InfrastructureDTO(
            infrastructure_id="gpu-cluster-1",
            compute="gpu",
            memory="32GB",
            timeout=7200,
            created_at=now,
            updated_at=now,
        )

        assert infra.infrastructure_id == "gpu-cluster-1"
        assert infra.compute == "gpu"
        assert infra.memory == "32GB"
        assert infra.timeout == 7200
        assert infra.created_at == now
        assert infra.updated_at == now

    def test_to_dict_with_defaults(self):
        """Test to_dict() with default values."""
        infra = InfrastructureDTO()
        result = infra.to_dict()

        assert result["infrastructure_id"] == "local-infra"
        assert result["compute"] == "cpu"
        assert result["memory"] == "8GB"
        assert result["timeout"] == 3600
        assert "created_at" in result
        assert "updated_at" in result
        # Timestamps should be generated
        assert result["created_at"] is not None
        assert result["updated_at"] is not None

    def test_to_dict_preserves_custom_timestamps(self):
        """Test that to_dict() preserves custom timestamps."""
        custom_time = "2024-01-01T00:00:00Z"

        infra = InfrastructureDTO(
            created_at=custom_time,
            updated_at=custom_time,
        )

        result = infra.to_dict()

        assert result["created_at"] == custom_time
        assert result["updated_at"] == custom_time

    def test_gpu_infrastructure(self):
        """Test GPU infrastructure configuration."""
        infra = InfrastructureDTO(
            infrastructure_id="gpu-node",
            compute="gpu",
            memory="64GB",
            timeout=10800,
        )

        assert infra.compute == "gpu"
        assert infra.memory == "64GB"

    def test_minimal_timeout(self):
        """Test infrastructure with minimal timeout."""
        infra = InfrastructureDTO(timeout=60)
        assert infra.timeout == 60

    def test_large_memory_config(self):
        """Test infrastructure with large memory."""
        infra = InfrastructureDTO(memory="128GB")
        assert infra.memory == "128GB"


class TestConfigurationsDTO:
    """Test ConfigurationsDTO dataclass."""

    def test_create_with_defaults(self):
        """Test creating ConfigurationsDTO with defaults."""
        config = ConfigurationsDTO()

        assert isinstance(config.infrastructure, InfrastructureDTO)
        assert config.parameters == {}

    def test_create_with_custom_infrastructure(self):
        """Test creating ConfigurationsDTO with custom infrastructure."""
        infra = InfrastructureDTO(
            infrastructure_id="custom-infra",
            compute="tpu",
            memory="16GB",
        )

        config = ConfigurationsDTO(infrastructure=infra)

        assert config.infrastructure.infrastructure_id == "custom-infra"
        assert config.infrastructure.compute == "tpu"

    def test_create_with_parameters(self):
        """Test creating ConfigurationsDTO with parameters."""
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        }

        config = ConfigurationsDTO(parameters=params)

        assert config.parameters == params
        assert config.parameters["learning_rate"] == 0.001

    def test_to_dict(self):
        """Test to_dict() serialization."""
        params = {"model": "gpt-4", "temperature": 0.7}
        config = ConfigurationsDTO(parameters=params)

        result = config.to_dict()

        assert "infrastructure" in result
        assert "parameters" in result
        assert result["parameters"] == params
        assert isinstance(result["infrastructure"], dict)

    def test_complex_nested_parameters(self):
        """Test with complex nested parameter structure."""
        params = {
            "model_config": {
                "architecture": "transformer",
                "layers": 12,
                "hidden_size": 768,
            },
            "training": {
                "optimizer": "adam",
                "scheduler": "cosine",
            },
        }

        config = ConfigurationsDTO(parameters=params)

        assert config.parameters["model_config"]["layers"] == 12
        assert config.parameters["training"]["optimizer"] == "adam"


class TestExperimentDTO:
    """Test ExperimentDTO dataclass."""

    def test_create_minimal_experiment(self):
        """Test creating experiment with minimal required fields."""
        exp = ExperimentDTO(
            id="exp-001",
            name="Test Experiment",
            description="A test experiment",
        )

        assert exp.id == "exp-001"
        assert exp.name == "Test Experiment"
        assert exp.description == "A test experiment"

    def test_default_values(self):
        """Test default values for optional fields."""
        exp = ExperimentDTO(
            id="exp-002",
            name="Default Test",
            description="Test defaults",
        )

        # Default configurations
        assert isinstance(exp.configurations, ConfigurationsDTO)

        # Default measures
        assert exp.measures == ["score", "cost", "latency"]

        # Default placeholder IDs for Edge Analytics mode
        assert exp.agent_id == "local-agent-001"
        assert exp.evaluation_set_id == "local-evalset-001"
        assert exp.model_parameters_id == "local-model-params-001"

        # Optional fields default to None/empty
        assert exp.benchmark_id is None
        assert exp.experiment_parameters == {}
        assert exp.metadata == {}
        assert exp.status is None

    def test_create_with_custom_configurations(self):
        """Test creating experiment with custom configurations."""
        config = ConfigurationsDTO(parameters={"model": "gpt-4", "temperature": 0.9})

        exp = ExperimentDTO(
            id="exp-003",
            name="Custom Config Experiment",
            description="Test custom config",
            configurations=config,
        )

        assert exp.configurations.parameters["model"] == "gpt-4"
        assert exp.configurations.parameters["temperature"] == 0.9

    def test_create_with_custom_measures(self):
        """Test experiment with custom measures."""
        custom_measures = ["accuracy", "precision", "recall", "f1_score"]

        exp = ExperimentDTO(
            id="exp-004",
            name="ML Metrics Experiment",
            description="Test ML metrics",
            measures=custom_measures,
        )

        assert exp.measures == custom_measures
        assert len(exp.measures) == 4

    def test_create_with_backend_ids(self):
        """Test experiment with backend IDs (not Edge Analytics mode)."""
        exp = ExperimentDTO(
            id="exp-005",
            name="Backend Experiment",
            description="Backend integration test",
            agent_id="agent-backend-123",
            evaluation_set_id="evalset-456",
            model_parameters_id="model-params-789",
            benchmark_id="benchmark-001",
        )

        assert exp.agent_id == "agent-backend-123"
        assert exp.evaluation_set_id == "evalset-456"
        assert exp.model_parameters_id == "model-params-789"
        assert exp.benchmark_id == "benchmark-001"

    def test_experiment_with_metadata(self):
        """Test experiment with metadata."""
        metadata = {
            "owner": "test_user",
            "project": "ml_optimization",
            "version": "1.0.0",
            "tags": ["production", "critical"],
        }

        exp = ExperimentDTO(
            id="exp-006",
            name="Metadata Experiment",
            description="Test metadata",
            metadata=metadata,
        )

        assert exp.metadata["owner"] == "test_user"
        assert exp.metadata["version"] == "1.0.0"
        assert "production" in exp.metadata["tags"]

    def test_experiment_with_parameters(self):
        """Test experiment with experiment_parameters."""
        exp_params = {
            "max_trials": 100,
            "timeout": 3600,
            "optimization_direction": "maximize",
        }

        exp = ExperimentDTO(
            id="exp-007",
            name="Parameterized Experiment",
            description="Test experiment parameters",
            experiment_parameters=exp_params,
        )

        assert exp.experiment_parameters["max_trials"] == 100
        assert exp.experiment_parameters["optimization_direction"] == "maximize"

    def test_to_dict_minimal(self):
        """Test to_dict() with minimal experiment."""
        exp = ExperimentDTO(
            id="exp-008",
            name="Minimal",
            description="Minimal experiment",
        )

        result = exp.to_dict()

        assert result["id"] == "exp-008"
        assert result["name"] == "Minimal"
        assert result["description"] == "Minimal experiment"
        assert "configurations" in result
        assert "measures" in result

    def test_to_dict_complete(self):
        """Test to_dict() with complete experiment data."""
        exp = ExperimentDTO(
            id="exp-009",
            name="Complete",
            description="Complete experiment",
            configurations=ConfigurationsDTO(parameters={"model": "gpt-4"}),
            measures=["accuracy", "cost"],
            agent_id="agent-001",
            evaluation_set_id="eval-001",
            model_parameters_id="model-001",
            benchmark_id="bench-001",
            experiment_parameters={"max_trials": 50},
            metadata={"version": "1.0"},
            status="running",
        )

        result = exp.to_dict()

        assert result["id"] == "exp-009"
        assert result["status"] == "running"
        assert result["benchmark_id"] == "bench-001"
        assert result["experiment_parameters"]["max_trials"] == 50
        assert result["metadata"]["version"] == "1.0"

    def test_experiment_status_values(self):
        """Test various experiment status values."""
        statuses = ["pending", "running", "completed", "failed", "cancelled"]

        for status in statuses:
            exp = ExperimentDTO(
                id=f"exp-{status}",
                name=f"{status} Experiment",
                description=f"Test {status} status",
                status=status,
            )
            assert exp.status == status


class TestExperimentRunDTO:
    """Test ExperimentRunDTO dataclass."""

    def test_create_minimal_run(self):
        """Test creating experiment run with minimal fields."""
        run = ExperimentRunDTO(
            id="run-001",
            experiment_id="exp-001",
        )

        assert run.id == "run-001"
        assert run.experiment_id == "exp-001"

    def test_default_values(self):
        """Test default values for optional fields."""
        run = ExperimentRunDTO(
            id="run-002",
            experiment_id="exp-002",
        )

        assert run.status == "pending"
        assert run.start_time is None
        assert run.end_time is None
        assert run.summary_stats == {}
        assert run.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating experiment run with all fields."""
        start = datetime.now(UTC).isoformat()
        end = datetime.now(UTC).isoformat()

        summary = {
            "total_trials": 100,
            "best_score": 0.95,
            "avg_score": 0.87,
        }

        metadata = {
            "optimizer": "bayesian",
            "duration_seconds": 3600,
        }

        run = ExperimentRunDTO(
            id="run-003",
            experiment_id="exp-003",
            status="completed",
            start_time=start,
            end_time=end,
            summary_stats=summary,
            metadata=metadata,
        )

        assert run.status == "completed"
        assert run.start_time == start
        assert run.end_time == end
        assert run.summary_stats["best_score"] == 0.95
        assert run.metadata["optimizer"] == "bayesian"

    def test_to_dict(self):
        """Test to_dict() serialization."""
        run = ExperimentRunDTO(
            id="run-004",
            experiment_id="exp-004",
            status="running",
            summary_stats={"trials_completed": 50},
        )

        result = run.to_dict()

        assert result["id"] == "run-004"
        assert result["experiment_id"] == "exp-004"
        assert result["status"] == "running"
        assert result["summary_stats"]["trials_completed"] == 50

    def test_run_lifecycle_statuses(self):
        """Test run through different lifecycle statuses."""
        statuses = ["pending", "running", "completed", "failed", "cancelled"]

        for status in statuses:
            run = ExperimentRunDTO(
                id=f"run-{status}",
                experiment_id="exp-001",
                status=status,
            )
            assert run.status == status


class TestConfigurationRunDTO:
    """Test ConfigurationRunDTO dataclass."""

    def test_create_minimal_configuration_run(self):
        """Test creating configuration run with minimal fields."""
        config_run = ConfigurationRunDTO(
            id="config-run-001",
            experiment_run_id="run-001",
            trial_number=1,
        )

        assert config_run.id == "config-run-001"
        assert config_run.experiment_run_id == "run-001"
        assert config_run.trial_number == 1

    def test_default_values(self):
        """Test default values for optional fields."""
        config_run = ConfigurationRunDTO(
            id="config-run-002",
            experiment_run_id="run-002",
            trial_number=2,
        )

        assert config_run.configuration == {}
        assert config_run.measures == {}
        assert config_run.status == "pending"
        assert config_run.error_message is None
        assert config_run.start_time is None
        assert config_run.end_time is None
        assert config_run.metadata == {}

    def test_create_with_configuration(self):
        """Test trial with configuration parameters."""
        config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        config_run = ConfigurationRunDTO(
            id="config-run-003",
            experiment_run_id="run-003",
            trial_number=3,
            configuration=config,
        )

        assert config_run.configuration["model"] == "gpt-4"
        assert config_run.configuration["temperature"] == 0.7
        assert config_run.configuration["max_tokens"] == 1000

    def test_create_with_measures(self):
        """Test trial with measure results."""
        measures = {
            "accuracy": 0.92,
            "cost": 0.05,
            "latency": 1.2,
        }

        config_run = ConfigurationRunDTO(
            id="config-run-004",
            experiment_run_id="run-004",
            trial_number=4,
            measures=measures,
        )

        assert config_run.measures["accuracy"] == 0.92
        assert config_run.measures["cost"] == 0.05
        assert config_run.measures["latency"] == 1.2

    def test_trial_with_error(self):
        """Test trial with error information."""
        config_run = ConfigurationRunDTO(
            id="config-run-005",
            experiment_run_id="run-005",
            trial_number=5,
            status="failed",
            error_message="Timeout exceeded",
        )

        assert config_run.status == "failed"
        assert config_run.error_message == "Timeout exceeded"

    def test_trial_with_timestamps(self):
        """Test trial with timing information."""
        start = "2024-01-01T10:00:00Z"
        end = "2024-01-01T10:05:00Z"

        config_run = ConfigurationRunDTO(
            id="config-run-006",
            experiment_run_id="run-006",
            trial_number=6,
            start_time=start,
            end_time=end,
        )

        assert config_run.start_time == start
        assert config_run.end_time == end

    def test_trial_with_metadata(self):
        """Test trial with metadata."""
        metadata = {
            "optimizer_suggested": True,
            "acquisition_function": "ei",
            "exploration_factor": 0.1,
        }

        config_run = ConfigurationRunDTO(
            id="config-run-007",
            experiment_run_id="run-007",
            trial_number=7,
            metadata=metadata,
        )

        assert config_run.metadata["optimizer_suggested"] is True
        assert config_run.metadata["acquisition_function"] == "ei"

    def test_to_dict(self):
        """Test to_dict() serialization."""
        config_run = ConfigurationRunDTO(
            id="config-run-008",
            experiment_run_id="run-008",
            trial_number=8,
            configuration={"model": "gpt-3.5-turbo"},
            measures={"score": 0.88},
            status="completed",
        )

        result = config_run.to_dict()

        assert result["id"] == "config-run-008"
        assert result["trial_number"] == 8
        assert result["configuration"]["model"] == "gpt-3.5-turbo"
        assert result["measures"]["score"] == 0.88
        assert result["status"] == "completed"

    def test_trial_statuses(self):
        """Test various trial status values."""
        statuses = ["pending", "running", "completed", "failed", "skipped"]

        for idx, status in enumerate(statuses, 1):
            config_run = ConfigurationRunDTO(
                id=f"config-run-{idx}",
                experiment_run_id="run-001",
                trial_number=idx,
                status=status,
            )
            assert config_run.status == status

    def test_trial_number_sequence(self):
        """Test trial numbers in sequence."""
        for i in range(1, 101):
            config_run = ConfigurationRunDTO(
                id=f"config-run-{i:03d}",
                experiment_run_id="run-001",
                trial_number=i,
            )
            assert config_run.trial_number == i


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_configurations(self):
        """Test DTOs with empty configurations."""
        exp = ExperimentDTO(
            id="exp-empty",
            name="Empty",
            description="Empty configs",
            configurations=ConfigurationsDTO(parameters={}),
        )

        assert exp.configurations.parameters == {}

    def test_large_parameter_dict(self):
        """Test with large parameter dictionary."""
        large_params = {f"param_{i}": i for i in range(1000)}

        config = ConfigurationsDTO(parameters=large_params)

        assert len(config.parameters) == 1000
        assert config.parameters["param_500"] == 500

    def test_unicode_in_descriptions(self):
        """Test DTOs with Unicode characters."""
        exp = ExperimentDTO(
            id="exp-unicode",
            name="实验 Experiment",
            description="Test with émojis 🚀 and spëcial chars",
        )

        assert "实验" in exp.name
        assert "🚀" in exp.description

    def test_null_optional_fields(self):
        """Test explicitly setting optional fields to None."""
        exp = ExperimentDTO(
            id="exp-null",
            name="Null Test",
            description="Test nulls",
            benchmark_id=None,
            status=None,
            created_at=None,
        )

        assert exp.benchmark_id is None
        assert exp.status is None
        assert exp.created_at is None

    def test_very_long_strings(self):
        """Test DTOs with very long string values."""
        long_desc = "A" * 10000

        exp = ExperimentDTO(
            id="exp-long",
            name="Long Description",
            description=long_desc,
        )

        assert len(exp.description) == 10000

    def test_nested_dict_in_metadata(self):
        """Test deeply nested dictionary in metadata."""
        nested_meta = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "deep_value",
                    }
                }
            }
        }

        exp = ExperimentDTO(
            id="exp-nested",
            name="Nested",
            description="Nested metadata",
            metadata=nested_meta,
        )

        assert exp.metadata["level1"]["level2"]["level3"]["level4"] == "deep_value"

    def test_numeric_string_ids(self):
        """Test DTOs with numeric string IDs."""
        config_run = ConfigurationRunDTO(
            id="12345",
            experiment_run_id="67890",
            trial_number=999,
        )

        assert config_run.id == "12345"
        assert config_run.experiment_run_id == "67890"

    def test_zero_timeout(self):
        """Test infrastructure with zero timeout."""
        infra = InfrastructureDTO(timeout=0)
        assert infra.timeout == 0

    def test_negative_trial_number(self):
        """Test trial with negative number (edge case)."""
        config_run = ConfigurationRunDTO(
            id="config-run-neg",
            experiment_run_id="run-001",
            trial_number=-1,
        )

        assert config_run.trial_number == -1

    def test_special_characters_in_ids(self):
        """Test IDs with special characters."""
        exp = ExperimentDTO(
            id="exp-test_123-v2.0",
            name="Special Chars",
            description="Test special char IDs",
        )

        assert exp.id == "exp-test_123-v2.0"
