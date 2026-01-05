"""Data Transfer Objects for Traigent Backend Integration.

This module provides DTOs that conform to Traigent schemas while supporting
privacy-preserving defaults for Edge Analytics mode execution.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import optigen_schemas for validation (optional)
try:
    from optigen_schemas.validator import SchemaValidator

    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    logger.debug("optigen_schemas not available, validation disabled")


@dataclass
class InfrastructureDTO:
    """Infrastructure configuration DTO."""

    infrastructure_id: str = "local-infra"
    compute: str = "cpu"
    memory: str = "8GB"
    timeout: int = 3600
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "infrastructure_id": self.infrastructure_id,
            "compute": self.compute,
            "memory": self.memory,
            "timeout": self.timeout,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "updated_at": self.updated_at or datetime.now(UTC).isoformat(),
        }


@dataclass
class ConfigurationsDTO:
    """Experiment configurations DTO."""

    infrastructure: InfrastructureDTO = field(default_factory=InfrastructureDTO)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "infrastructure": self.infrastructure.to_dict(),
            "parameters": self.parameters,
        }


@dataclass
class ExperimentDTO:
    """Experiment DTO based on experiment_schema.json.

    For Edge Analytics mode, sensitive fields are set to placeholder values to preserve privacy.
    """

    id: str
    name: str
    description: str
    configurations: ConfigurationsDTO = field(default_factory=ConfigurationsDTO)
    measures: list[str] = field(default_factory=lambda: ["score", "cost", "latency"])

    # Required fields - use placeholder IDs for Edge Analytics mode
    agent_id: str = "local-agent-001"
    evaluation_set_id: str = "local-evalset-001"
    model_parameters_id: str = "local-model-params-001"

    # Optional fields
    benchmark_id: str | None = None

    # Optional fields
    experiment_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "configurations": self.configurations.to_dict(),
            "measures": self.measures,
            "experiment_parameters": self.experiment_parameters,
            "metadata": self.metadata,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "updated_at": self.updated_at or datetime.now(UTC).isoformat(),
        }

        # Required fields always included
        result["agent_id"] = self.agent_id
        result["evaluation_set_id"] = self.evaluation_set_id
        result["model_parameters_id"] = self.model_parameters_id

        # Include optional fields only if they have values
        if self.benchmark_id is not None:
            result["benchmark_id"] = self.benchmark_id
        if self.status is not None:
            result["status"] = self.status

        return result

    def validate(self) -> bool:
        """Validate the DTO against optigen_schemas (optional)."""
        if not VALIDATOR_AVAILABLE:
            return True

        try:
            validator = SchemaValidator()
            validator.validate_json_by_schema("experiment", self.to_dict())
            return True
        except Exception as e:
            logger.warning(f"Validation failed (non-blocking): {e}")
            return False


@dataclass
class ExperimentRunDTO:
    """Experiment Run DTO based on experiment_run_schema.json."""

    id: str
    experiment_id: str

    # Lifecycle and status metadata
    status: str = "pending"
    start_time: str | None = None
    end_time: str | None = None
    summary_stats: dict[str, Any] = field(default_factory=dict)

    # Optional payloads
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_data: dict[str, Any] | None = None
    results: dict[str, Any | None] | None = None
    error_message: str | None = None

    # Audit fields
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "summary_stats": self.summary_stats,
            "metadata": self.metadata,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "updated_at": self.updated_at or datetime.now(UTC).isoformat(),
        }

        if self.start_time:
            result["start_time"] = self.start_time
        if self.end_time:
            result["end_time"] = self.end_time
        if self.experiment_data is not None:
            result["experiment_data"] = self.experiment_data
        if self.results is not None:
            result["results"] = self.results
        if self.error_message:
            result["error_message"] = self.error_message

        return result


@dataclass
class ConfigurationRunDTO:
    """Configuration Run DTO based on configuration_run_schema.json."""

    id: str
    experiment_run_id: str
    trial_number: int

    # Configuration payload
    configuration: dict[str, Any] = field(default_factory=dict)
    measures: dict[str, Any] = field(default_factory=dict)

    # Optional fields
    status: str = "pending"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: str | None = None
    end_time: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {
            "id": self.id,
            "experiment_run_id": self.experiment_run_id,
            "trial_number": self.trial_number,
            "configuration": self.configuration,
            "measures": self.measures,
            "metadata": self.metadata,
            "status": self.status,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "updated_at": self.updated_at or datetime.now(UTC).isoformat(),
        }

        if self.error_message:
            result["error_message"] = self.error_message
        if self.start_time:
            result["start_time"] = self.start_time
        if self.end_time:
            result["end_time"] = self.end_time

        return result

    @property
    def experiment_parameters(self) -> dict[str, Any]:
        """Backward-compatible view of configuration metadata."""
        dataset_subset = self.metadata.get("dataset_subset", {})
        return {
            "trial_number": self.trial_number,
            "config": self.metadata.get("config", self.configuration.get("parameters")),
            "dataset_subset": dataset_subset,
        }


# Helper functions for creating privacy-preserving DTOs


def create_local_experiment(
    experiment_id: str,
    name: str,
    description: str,
    configuration_space: dict[str, Any],
    max_trials: int = 10,
    dataset_size: int = 100,
) -> ExperimentDTO:
    """Create an experiment DTO for Edge Analytics mode with privacy-preserving defaults."""
    return ExperimentDTO(
        id=experiment_id,
        name=name,
        description=description,
        configurations=ConfigurationsDTO(
            infrastructure=InfrastructureDTO(), parameters=configuration_space
        ),
        experiment_parameters={
            "max_trials": max_trials,
            "configuration_space": configuration_space,
        },
        metadata={
            "execution_mode": "edge_analytics",
            "privacy_mode": True,
            "dataset_size": dataset_size,
            "created_with": "traigent-local",
        },
    )


def create_local_experiment_run(
    run_id: str,
    experiment_id: str,
    function_name: str,
    configuration_space: dict[str, Any],
    objectives: list[str],
    max_trials: int = 10,
    dataset_size: int = 100,
) -> ExperimentRunDTO:
    """Create an experiment run DTO for Edge Analytics mode."""
    return ExperimentRunDTO(
        id=run_id,
        experiment_id=experiment_id,
        experiment_data={
            "configurations": {
                "infrastructure": {
                    "compute": ["cpu"],  # Array format for experiment run
                    "memory": [8192],  # Array of MB values
                    "timeout": [3600],  # Array of seconds
                },
                "parameters": {
                    "function_name": function_name,
                    "configuration_space": configuration_space,
                    "objectives": objectives,
                    "max_trials": max_trials,
                },
            },
            "metadata": {"dataset_size": dataset_size, "privacy_mode": True},
        },
        metadata={"execution_mode": "edge_analytics", "privacy_mode": True},
        status="not_started",  # Set correct status for backend compatibility
    )


def create_local_configuration_run(
    config_id: str,
    experiment_run_id: str,
    trial_number: int,
    config: dict[str, Any],
    dataset_subset_info: dict[str, Any | None] | None = None,
) -> ConfigurationRunDTO:
    """Create a configuration run DTO for Edge Analytics mode."""
    # Privacy-preserving dataset subset info
    if dataset_subset_info is None:
        dataset_subset_info = {
            "indices": [],  # Empty for privacy
            "selection_strategy": "privacy_mode",
            "confidence_level": 0.0,
            "estimated_representativeness": 0.0,
        }

    # Only include allowed fields in experiment_parameters
    return ConfigurationRunDTO(
        id=config_id,
        experiment_run_id=experiment_run_id,
        trial_number=trial_number,
        configuration={
            "generator_config_id": "local-generator-001",  # Privacy placeholder
            "evaluator_config_id": "local-evaluator-001",  # Privacy placeholder
            "model_parameters_id": "local-model-params-001",  # Privacy placeholder
            "infrastructure": InfrastructureDTO().to_dict(),
            "parameters": config,
        },
        metadata={
            "config": config,
            "dataset_subset": dataset_subset_info,
            "privacy_mode": True,
        },
        status="not_started",  # Set correct status for backend compatibility
    )
