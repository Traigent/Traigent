"""Data Transfer Objects for Traigent Backend Integration.

This module provides DTOs that conform to Traigent schemas while supporting
privacy-preserving defaults for Edge Analytics mode execution.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import re
from collections import UserDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical persisted status strings used by experiment/configuration run DTOs.
# Keep these aligned with TraigentSchema, not internal optimizer/runtime states.
EXPERIMENT_RUN_STATUS_VALUES = frozenset(
    {
        "not_started",
        "running",
        "completed",
        "failed",
        "cancelled",
        "paused",
        "partially_deleted",
    }
)
CONFIGURATION_RUN_STATUS_VALUES = frozenset(
    {
        "not_started",
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "paused",
        "partially_deleted",
    }
)


def _get_schema_validator_class() -> Any:
    """Load the optional TraigentSchema validator lazily."""
    from traigent_schema.validator import SchemaValidator

    return SchemaValidator


def _warn_if_unknown_status(
    *,
    status: str,
    allowed: frozenset[str],
    dto_name: str,
) -> None:
    """Log unknown status values while preserving backward compatibility."""
    if status not in allowed:
        logger.warning(
            "%s received non-canonical status '%s'. Allowed statuses: %s",
            dto_name,
            status,
            sorted(allowed),
        )


class ExampleMeasure:
    """Type-safe per-example measure with nested format.

    Expected structure:
        {
            "example_id": "ex_a3f4b2c8_0",
            "metrics": {"score": 0.85, "cost": 0.05, ...}
        }

    Validates:
    - example_id must be a string
    - metrics must be a dict with Python identifier keys
    - metrics values must be numeric (int, float) or None
    - Maximum 50 keys in metrics

    Use this class for per-example results. For trial-level measures,
    use MeasuresDict instead.
    """

    MAX_METRICS = 50
    KEY_PATTERN = re.compile(r"^[a-zA-Z_]\w*$")

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize from a nested measure dict.

        Args:
            data: Dict with 'example_id' (str) and 'metrics' (dict) keys

        Raises:
            ValueError: If validation fails
        """
        self.example_id: str | None = data.get("example_id")
        self.metrics: dict[str, float | int | None] = data.get("metrics", {})
        self._validate()

    def _validate(self) -> None:
        """Validate the nested measure structure."""
        # example_id must be string if present
        if self.example_id is not None and not isinstance(self.example_id, str):
            raise ValueError(
                f"example_id must be a string, got {type(self.example_id).__name__}"
            )

        # metrics must be a dict
        if not isinstance(self.metrics, dict):
            raise ValueError(
                f"metrics must be a dict, got {type(self.metrics).__name__}"
            )

        # Check max keys
        if len(self.metrics) > self.MAX_METRICS:
            raise ValueError(
                f"metrics cannot exceed {self.MAX_METRICS} keys, got {len(self.metrics)}"
            )

        # Validate each metric
        for key, value in self.metrics.items():
            # Keys must be Python identifiers
            if not isinstance(key, str):
                raise ValueError(f"metric key must be string, got {type(key).__name__}")
            if not self.KEY_PATTERN.match(key):
                raise ValueError(
                    f"metric key '{key}' must match pattern ^[a-zA-Z_]\\w*$"
                )
            # Values must be numeric or None (bool is explicitly rejected)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"metric '{key}' must be numeric, got {type(value).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "example_id": self.example_id,
            "metrics": self.metrics,
        }

    def __repr__(self) -> str:
        return (
            f"ExampleMeasure(example_id={self.example_id!r}, metrics={self.metrics!r})"
        )


class MeasuresDict(UserDict):
    """Type-safe measures dict with validation for trial-level metrics.

    Inherits from UserDict (not dict) to properly intercept all mutation
    operations including update(), |=, and other bulk operations.

    Note: This class is for TRIAL-LEVEL measures (aggregated metrics).
    For per-example measures with nested format, use ExampleMeasure instead.

    Enforces:
    - Maximum 50 keys to prevent unbounded memory usage
    - String keys matching Python identifier pattern (^[a-zA-Z_][a-zA-Z0-9_]*$)
    - Numeric value types only (int, float, None) for optimization metrics
    - Non-numeric values log warnings in Phase 0 (will be rejected in v2.0)

    Raises:
        ValueError: If key limit exceeded or key pattern invalid
        TypeError: If key is not string or value is not primitive type
    """

    MAX_KEYS = 50
    KEY_PATTERN = re.compile(r"^[a-zA-Z_]\w*$")

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize with optional data dictionary.

        Args:
            data: Optional dict to initialize with (will be validated)

        Raises:
            ValueError: If data exceeds MAX_KEYS
            TypeError: If data contains invalid key or value types
        """
        super().__init__()
        if data:
            # Validate first, then use parent's update to populate self.data
            self._validate_dict(data)
            self.data.update(data)

    def _validate_dict(self, data: dict[str, Any]) -> None:
        """Validate measures dictionary.

        Args:
            data: Dictionary to validate

        Raises:
            ValueError: If data exceeds MAX_KEYS or key pattern invalid
            TypeError: If data contains invalid key or value types
        """
        if len(data) > self.MAX_KEYS:
            raise ValueError(
                f"Measures cannot exceed {self.MAX_KEYS} keys, got {len(data)}"
            )

        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"Measure key must be string, got {type(key).__name__}")

            # NEW: Validate key pattern (Python identifier syntax)
            if not self.KEY_PATTERN.match(key):
                raise ValueError(
                    f"Measure key '{key}' must match pattern ^[a-zA-Z_]\\w*$ "
                    f"(Python identifier syntax). "
                    f"Use underscores instead of hyphens or spaces. "
                    f"Invalid: 'my-metric', '123abc'. Valid: 'my_metric', 'metric_123'."
                )

            # bool is not treated as numeric for contract parity with JSON Schema
            if isinstance(value, bool):
                raise TypeError(
                    f"Measure '{key}' must be numeric type (int, float, None), got bool"
                )

            # NEW: Phase 0 - Warn on non-numeric values (enforce in Phase 2/v2.0)
            if not isinstance(value, (int, float, type(None))):
                logger.warning(
                    f"Measure '{key}' has non-numeric value type {type(value).__name__}. "
                    f"Non-numeric metrics will be rejected in Traigent v2.0. "
                    f"Store non-numeric data in configuration run metadata instead.",
                    extra={
                        "key": key,
                        "value_type": type(value).__name__,
                        "hint": "Use run_metadata or workflow_metadata for non-numeric data",
                    },
                )
                # Phase 0: Allow but warn (backward compatible)
                # Phase 2: Uncomment to enforce
                # raise TypeError(
                #     f"Measure '{key}' must be numeric type (int, float, None), "
                #     f"got {type(value).__name__}. "
                #     f"Non-numeric data should be stored in configuration run metadata."
                # )

    def __setitem__(self, key: str, value: Any) -> None:
        """Validate on assignment.

        Args:
            key: Measure key (must be string matching Python identifier pattern)
            value: Measure value (must be numeric type)

        Raises:
            ValueError: If adding would exceed MAX_KEYS or key pattern invalid
            TypeError: If key is not string or value is not primitive type
        """
        if len(self.data) >= self.MAX_KEYS and key not in self.data:
            raise ValueError(f"Measures cannot exceed {self.MAX_KEYS} keys")

        if not isinstance(key, str):
            raise TypeError(f"Key must be string, got {type(key).__name__}")

        # NEW: Validate key pattern (Python identifier syntax)
        if not self.KEY_PATTERN.match(key):
            raise ValueError(
                f"Measure key '{key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ "
                f"(Python identifier syntax). "
                f"Use underscores instead of hyphens or spaces."
            )

        # bool is not treated as numeric for contract parity with JSON Schema
        if isinstance(value, bool):
            raise TypeError(
                f"Value for measure '{key}' must be numeric type (int, float, None), got bool"
            )

        # NEW: Phase 0 - Warn on non-numeric values (enforce in Phase 2/v2.0)
        if not isinstance(value, (int, float, type(None))):
            logger.warning(
                f"Setting non-numeric measure '{key}': {type(value).__name__}. "
                f"This will be rejected in Traigent v2.0. "
                f"Use run_metadata or workflow_metadata for non-numeric data."
            )
            # Phase 0: Allow but warn (backward compatible)
            # Phase 2: Uncomment to enforce
            # raise TypeError(
            #     f"Value must be numeric type (int, float, None), "
            #     f"got {type(value).__name__}"
            # )

        self.data[key] = value

    def __ior__(self, other: dict[str, Any] | Mapping[str, Any]) -> "MeasuresDict":  # type: ignore[override,misc]
        """Support |= operator with validation.

        Args:
            other: Mapping to merge with this MeasuresDict

        Returns:
            Self for chaining

        Raises:
            ValueError: If validation fails
            TypeError: If key or value types invalid
        """
        # Validate all items from other mapping before merging
        if isinstance(other, Mapping):
            for key, value in other.items():
                self[key] = value  # Use __setitem__ for validation
        else:
            raise TypeError(
                f"unsupported operand type(s) for |=: 'MeasuresDict' and '{type(other).__name__}'"
            )
        return self


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
        """Validate the DTO against TraigentSchema.

        By default, validation is strict - failures raise exceptions.
        Set TRAIGENT_STRICT_VALIDATION=false to make validation non-blocking.
        Internal environments that need schema validation should install
        the optional ``internal_schema`` dependency bundle.

        Returns:
            True if validation passed

        Raises:
            DTOSerializationError: If strict mode enabled and validation fails
        """
        import os

        from traigent.utils.exceptions import DTOSerializationError

        # Check if strict validation is enabled (default: true)
        strict_mode = os.getenv("TRAIGENT_STRICT_VALIDATION", "true").lower() == "true"

        try:
            validator = _get_schema_validator_class()()
            errors = validator.validate_json(self.to_dict(), "experiment")
            if errors:
                raise ValueError("; ".join(errors))
            return True
        except ImportError as e:
            message = (
                "ExperimentDTO validation requires the internal "
                "'traigent-schema' package. Install Traigent with the "
                "'internal_schema' extra in internal environments."
            )
            logger.warning(
                "Optional TraigentSchema validator unavailable",
                extra={
                    "dto_class": "ExperimentDTO",
                    "dto_id": self.id,
                    "error": str(e),
                    "strict_mode": strict_mode,
                    "install_hint": "pip install -e '.[internal_schema]'",
                },
            )

            if strict_mode:
                raise DTOSerializationError(
                    message,
                    dto_class="ExperimentDTO",
                    dto_id=self.id,
                ) from e

            return False
        except Exception as e:
            logger.error(
                "DTO validation failed",
                extra={
                    "dto_class": "ExperimentDTO",
                    "dto_id": self.id,
                    "error": str(e),
                    "strict_mode": strict_mode,
                },
            )

            if strict_mode:
                raise DTOSerializationError(
                    f"ExperimentDTO validation failed: {e}",
                    dto_class="ExperimentDTO",
                    dto_id=self.id,
                ) from e

            return False


@dataclass
class ExperimentRunDTO:
    """Experiment Run DTO based on experiment_run_schema.json."""

    id: str
    experiment_id: str

    # Lifecycle and status metadata
    status: str = "not_started"
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
        _warn_if_unknown_status(
            status=self.status,
            allowed=EXPERIMENT_RUN_STATUS_VALUES,
            dto_name="ExperimentRunDTO",
        )
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
    measures: MeasuresDict = field(default_factory=MeasuresDict)

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
        _warn_if_unknown_status(
            status=self.status,
            allowed=CONFIGURATION_RUN_STATUS_VALUES,
            dto_name="ConfigurationRunDTO",
        )
        result: dict[str, Any] = {
            "id": self.id,
            "experiment_run_id": self.experiment_run_id,
            "trial_number": self.trial_number,
            "configuration": self.configuration,
            "measures": dict(self.measures),  # Convert MeasuresDict to plain dict
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
