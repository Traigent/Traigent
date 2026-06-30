"""Data Transfer Objects for Traigent Backend Integration.

This module provides DTOs that conform to Traigent schemas while supporting
privacy-preserving defaults for Edge Analytics mode execution.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import re
from collections import UserDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

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


@dataclass(frozen=True)
class QuotaExceededErrorDTO:
    """Canonical backend quota-exceeded error payload."""

    resource_type: str
    current_usage: float
    limit: float
    message: str
    reset_at: str | None = None
    upgrade_url: str | None = "/billing"
    error_code: str = "quota_exceeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "resource_type": self.resource_type,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "reset_at": self.reset_at,
            "upgrade_url": self.upgrade_url,
            "message": self.message,
        }


@dataclass(frozen=True)
class WalletInsufficientBalanceErrorDTO:
    """Canonical backend wallet insufficient-balance error payload."""

    available_usd: str
    required_usd: str
    message: str
    operation_id: str | None = None
    operation_group_id: str | None = None
    error_code: str = "wallet_insufficient_balance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "available_usd": self.available_usd,
            "required_usd": self.required_usd,
            "operation_id": self.operation_id,
            "operation_group_id": self.operation_group_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class WalletTopUpPackDTO:
    """Public wallet credit pack metadata."""

    pack_id: str
    credit_usd: str

    def to_dict(self) -> dict[str, Any]:
        return {"pack_id": self.pack_id, "credit_usd": self.credit_usd}


@dataclass(frozen=True)
class WalletTopUpPacksResponseDTO:
    """Standard backend response wrapper for public wallet credit packs."""

    packs: list[WalletTopUpPackDTO]
    message: str = "Success"
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": {"packs": [pack.to_dict() for pack in self.packs]},
        }


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
    # ``re.ASCII`` pins ``\w`` to ``[A-Za-z0-9_]`` so the regex matches the
    # documented ``^[a-zA-Z_][a-zA-Z0-9_]*$`` contract exactly. Without it,
    # ``\w`` would also accept Unicode word characters (e.g. ``"a_πmetric"``),
    # which the SDK evaluators' identifier gate rejects — that divergence is the
    # Unicode-identifier bypass closed in the composite-knobs metrics channel.
    # ``traigent.evaluators.metrics_tracker.USER_METRIC_KEY_PATTERN`` mirrors this
    # exactly; the consistency test pins the two together.
    KEY_PATTERN = re.compile(r"^[a-zA-Z_]\w*$", re.ASCII)

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
class ExperimentListRunSummaryDTO:
    """Compact experiment-run summary embedded in experiment list responses."""

    id: str
    experiment_id: str
    status: str
    run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    configuration_runs_count: int = 0
    summary_stats: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the compact backend list-response shape."""
        _warn_if_unknown_status(
            status=self.status,
            allowed=EXPERIMENT_RUN_STATUS_VALUES,
            dto_name="ExperimentListRunSummaryDTO",
        )
        result: dict[str, Any] = {
            "id": self.id,
            "run_id": self.run_id if self.run_id is not None else self.id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "configuration_runs_count": int(self.configuration_runs_count),
        }
        for field_name in ("created_at", "updated_at", "started_at", "completed_at"):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        if self.summary_stats is not None:
            result["summary_stats"] = self.summary_stats
        if self.metrics is not None:
            result["metrics"] = self.metrics
        return result


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
    dataset_id: str | None = None
    benchmark_id: str | None = None

    # Optional fields
    experiment_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    configuration_runs_count: int | None = None
    total_examples: int | None = None
    optimization_runs_count: int | None = None
    experiment_run: ExperimentListRunSummaryDTO | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        explicit_dataset_aliases = [
            value for value in (self.dataset_id, self.benchmark_id) if value is not None
        ]
        if self.evaluation_set_id != "local-evalset-001":
            explicit_dataset_aliases.append(self.evaluation_set_id)
        if len(set(explicit_dataset_aliases)) > 1:
            raise ValueError(
                "Conflicting dataset aliases supplied for ExperimentDTO; "
                "dataset_id, evaluation_set_id, and benchmark_id must match"
            )
        resolved_dataset_id = (
            self.dataset_id
            or self.benchmark_id
            or (
                self.evaluation_set_id
                if self.evaluation_set_id != "local-evalset-001"
                else None
            )
            or self.evaluation_set_id
        )
        has_explicit_legacy_dataset_reference = (
            self.dataset_id is not None
            or self.benchmark_id is not None
            or self.evaluation_set_id != "local-evalset-001"
        )
        result: dict[str, Any] = {
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
        result["dataset_id"] = resolved_dataset_id
        result["evaluation_set_id"] = resolved_dataset_id
        result["eval_dataset_id"] = resolved_dataset_id
        result["model_parameters_id"] = self.model_parameters_id

        # Include optional fields only if they have values
        if has_explicit_legacy_dataset_reference and resolved_dataset_id is not None:
            result["benchmark_id"] = resolved_dataset_id
        if self.status is not None:
            result["status"] = self.status
        if self.configuration_runs_count is not None:
            result["configuration_runs_count"] = int(self.configuration_runs_count)
        if self.total_examples is not None:
            result["total_examples"] = int(self.total_examples)
        if self.optimization_runs_count is not None:
            result["optimization_runs_count"] = int(self.optimization_runs_count)
        if self.experiment_run is not None:
            if isinstance(self.experiment_run, ExperimentListRunSummaryDTO):
                result["experiment_run"] = self.experiment_run.to_dict()
            else:
                result["experiment_run"] = self.experiment_run

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


def _copy_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def _extra_fields(source: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in source.items() if key not in known}


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _required_str(payload: Mapping[str, Any], key: str, *, dto_name: str) -> str:
    value = payload.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"{dto_name} requires non-empty {key}.")
    return str(value)


@dataclass(frozen=True)
class ExperimentGroupSourceExperimentDTO:
    """Source experiment/run member of an experiment group."""

    experiment_id: str
    experiment_run_id: str | None = None
    name: str | None = None
    status: str | None = None
    dataset_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ExperimentGroupSourceExperimentDTO":
        known = {
            "experiment_id",
            "id",
            "experiment_run_id",
            "run_id",
            "name",
            "status",
            "dataset_id",
            "created_at",
            "updated_at",
            "metadata",
        }
        experiment_id = str(payload.get("experiment_id") or payload.get("id") or "")
        return cls(
            experiment_id=experiment_id,
            experiment_run_id=(
                str(payload["experiment_run_id"])
                if payload.get("experiment_run_id") is not None
                else (
                    str(payload["run_id"])
                    if payload.get("run_id") is not None
                    else None
                )
            ),
            name=str(payload["name"]) if payload.get("name") is not None else None,
            status=(
                str(payload["status"]) if payload.get("status") is not None else None
            ),
            dataset_id=(
                str(payload["dataset_id"])
                if payload.get("dataset_id") is not None
                else None
            ),
            created_at=(
                str(payload["created_at"])
                if payload.get("created_at") is not None
                else None
            ),
            updated_at=(
                str(payload["updated_at"])
                if payload.get("updated_at") is not None
                else None
            ),
            metadata=_copy_dict(payload.get("metadata")),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.extra_fields)
        result.update(
            {
                "experiment_id": self.experiment_id,
                "experiment_run_id": self.experiment_run_id,
                "name": self.name,
                "status": self.status,
                "dataset_id": self.dataset_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "metadata": deepcopy(self.metadata),
            }
        )
        return result


@dataclass(frozen=True)
class GroupedConfigurationRunRowDTO:
    """Configuration-run row returned from a grouped/cohort read.

    The three source IDs are intentionally first-class fields. Consumers must
    join on these IDs, not on tuned variables, objectives, or configuration hash.
    """

    configuration_run_id: str
    experiment_run_id: str
    experiment_id: str
    configuration: dict[str, Any] = field(default_factory=dict)
    measures: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    trial_number: int | None = None
    dataset_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroupedConfigurationRunRowDTO":
        known = {
            "configuration_run_id",
            "id",
            "experiment_run_id",
            "experiment_id",
            "configuration",
            "measures",
            "metrics",
            "status",
            "trial_number",
            "dataset_id",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
            "metadata",
        }
        config_run_id = payload.get("configuration_run_id", payload.get("id"))
        measures = payload.get("measures", payload.get("metrics"))
        trial_number = payload.get("trial_number")
        return cls(
            configuration_run_id=_required_str(
                {"configuration_run_id": config_run_id},
                "configuration_run_id",
                dto_name="GroupedConfigurationRunRowDTO",
            ),
            experiment_run_id=_required_str(
                payload,
                "experiment_run_id",
                dto_name="GroupedConfigurationRunRowDTO",
            ),
            experiment_id=_required_str(
                payload,
                "experiment_id",
                dto_name="GroupedConfigurationRunRowDTO",
            ),
            configuration=_copy_dict(payload.get("configuration")),
            measures=_copy_dict(measures),
            status=(
                str(payload["status"]) if payload.get("status") is not None else None
            ),
            trial_number=(
                _int_or_default(trial_number, 0) if trial_number is not None else None
            ),
            dataset_id=(
                str(payload["dataset_id"])
                if payload.get("dataset_id") is not None
                else None
            ),
            started_at=(
                str(payload["started_at"])
                if payload.get("started_at") is not None
                else None
            ),
            completed_at=(
                str(payload["completed_at"])
                if payload.get("completed_at") is not None
                else None
            ),
            created_at=(
                str(payload["created_at"])
                if payload.get("created_at") is not None
                else None
            ),
            updated_at=(
                str(payload["updated_at"])
                if payload.get("updated_at") is not None
                else None
            ),
            metadata=_copy_dict(payload.get("metadata")),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.extra_fields)
        result.update(
            {
                "configuration_run_id": self.configuration_run_id,
                "experiment_run_id": self.experiment_run_id,
                "experiment_id": self.experiment_id,
                "configuration": deepcopy(self.configuration),
                "measures": deepcopy(self.measures),
                "status": self.status,
                "trial_number": self.trial_number,
                "dataset_id": self.dataset_id,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "metadata": deepcopy(self.metadata),
            }
        )
        return result


@dataclass(frozen=True)
class ExperimentGroupStatusSummaryDTO:
    """Canonical experiment-group status summary."""

    experiment_run_status_counts: dict[str, int] = field(default_factory=dict)
    configuration_run_status_counts: dict[str, int] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | None
    ) -> "ExperimentGroupStatusSummaryDTO":
        if payload is None:
            payload = {}
        known = {
            "experiment_run_status_counts",
            "configuration_run_status_counts",
        }
        return cls(
            experiment_run_status_counts={
                str(key): _int_or_default(value, 0)
                for key, value in _copy_dict(
                    payload.get("experiment_run_status_counts")
                ).items()
            },
            configuration_run_status_counts={
                str(key): _int_or_default(value, 0)
                for key, value in _copy_dict(
                    payload.get("configuration_run_status_counts")
                ).items()
            },
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.extra_fields)
        result.update(
            {
                "experiment_run_status_counts": dict(self.experiment_run_status_counts),
                "configuration_run_status_counts": dict(
                    self.configuration_run_status_counts
                ),
            }
        )
        return result


@dataclass(frozen=True)
class ExperimentGroupOverviewDTO:
    """Experiment-group overview row."""

    group_id: str = ""
    name: str = ""
    agent_id: str | None = None
    description: str | None = None
    project_id: str | None = None
    dataset_id: str | None = None
    source_experiments: list[ExperimentGroupSourceExperimentDTO] = field(
        default_factory=list
    )
    experiment_count: int = 0
    experiment_run_count: int = 0
    configuration_run_count: int = 0
    first_experiment_created_at: str | None = None
    last_experiment_updated_at: str | None = None
    first_experiment_run_created_at: str | None = None
    last_experiment_run_updated_at: str | None = None
    status_summary: ExperimentGroupStatusSummaryDTO = field(
        default_factory=ExperimentGroupStatusSummaryDTO
    )
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def configuration_runs_count(self) -> int:
        """Backward-compatible alias for the canonical singular field."""
        return self.configuration_run_count

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentGroupOverviewDTO":
        known = {
            "group_id",
            "id",
            "name",
            "agent_id",
            "description",
            "project_id",
            "dataset_id",
            "source_experiments",
            "experiments",
            "experiment_count",
            "experiments_count",
            "experiment_run_count",
            "experiment_runs_count",
            "configuration_run_count",
            "configuration_runs_count",
            "first_experiment_created_at",
            "last_experiment_updated_at",
            "first_experiment_run_created_at",
            "last_experiment_run_updated_at",
            "status_summary",
            "created_at",
            "updated_at",
            "metadata",
        }
        source_rows = payload.get("source_experiments", payload.get("experiments", []))
        return cls(
            group_id=str(payload.get("group_id") or payload.get("id") or ""),
            name=str(payload.get("name") or ""),
            agent_id=(
                str(payload["agent_id"])
                if payload.get("agent_id") is not None
                else None
            ),
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else None
            ),
            project_id=(
                str(payload["project_id"])
                if payload.get("project_id") is not None
                else None
            ),
            dataset_id=(
                str(payload["dataset_id"])
                if payload.get("dataset_id") is not None
                else None
            ),
            source_experiments=[
                ExperimentGroupSourceExperimentDTO.from_dict(item)
                for item in _copy_list(source_rows)
                if isinstance(item, Mapping)
            ],
            experiment_count=_int_or_default(
                payload.get("experiment_count", payload.get("experiments_count")),
                len(_copy_list(source_rows)),
            ),
            experiment_run_count=_int_or_default(
                payload.get(
                    "experiment_run_count", payload.get("experiment_runs_count")
                ),
                0,
            ),
            configuration_run_count=_int_or_default(
                payload.get(
                    "configuration_run_count",
                    payload.get("configuration_runs_count"),
                ),
                0,
            ),
            first_experiment_created_at=(
                str(payload["first_experiment_created_at"])
                if payload.get("first_experiment_created_at") is not None
                else None
            ),
            last_experiment_updated_at=(
                str(payload["last_experiment_updated_at"])
                if payload.get("last_experiment_updated_at") is not None
                else None
            ),
            first_experiment_run_created_at=(
                str(payload["first_experiment_run_created_at"])
                if payload.get("first_experiment_run_created_at") is not None
                else None
            ),
            last_experiment_run_updated_at=(
                str(payload["last_experiment_run_updated_at"])
                if payload.get("last_experiment_run_updated_at") is not None
                else None
            ),
            status_summary=ExperimentGroupStatusSummaryDTO.from_dict(
                payload.get("status_summary")
                if isinstance(payload.get("status_summary"), Mapping)
                else None
            ),
            created_at=(
                str(payload["created_at"])
                if payload.get("created_at") is not None
                else None
            ),
            updated_at=(
                str(payload["updated_at"])
                if payload.get("updated_at") is not None
                else None
            ),
            metadata=_copy_dict(payload.get("metadata")),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.extra_fields)
        result.update(
            {
                "group_id": self.group_id,
                "name": self.name,
                "agent_id": self.agent_id,
                "description": self.description,
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "source_experiments": [
                    experiment.to_dict() for experiment in self.source_experiments
                ],
                "experiment_count": self.experiment_count,
                "experiment_run_count": self.experiment_run_count,
                "configuration_run_count": self.configuration_run_count,
                "first_experiment_created_at": self.first_experiment_created_at,
                "last_experiment_updated_at": self.last_experiment_updated_at,
                "first_experiment_run_created_at": self.first_experiment_run_created_at,
                "last_experiment_run_updated_at": self.last_experiment_run_updated_at,
                "status_summary": self.status_summary.to_dict(),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "metadata": deepcopy(self.metadata),
            }
        )
        return result


@dataclass(frozen=True)
class ExperimentGroupDetailDTO(ExperimentGroupOverviewDTO):
    """Detailed experiment group, including grouped configuration rows."""

    grouped_configurations: list[GroupedConfigurationRunRowDTO] = field(
        default_factory=list
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentGroupDetailDTO":
        if isinstance(payload.get("group"), Mapping):
            overview_payload = dict(cast(Mapping[str, Any], payload["group"]))
            if "source_experiments" not in overview_payload:
                overview_payload["source_experiments"] = payload.get(
                    "source_experiments",
                    [],
                )
        else:
            overview_payload = dict(payload)

        overview = ExperimentGroupOverviewDTO.from_dict(overview_payload)
        rows = payload.get(
            "grouped_configurations",
            payload.get("configuration_runs", payload.get("configurations", [])),
        )
        extra = dict(overview.extra_fields)
        for key in (
            "group",
            "source_experiments",
            "grouped_configurations",
            "configuration_runs",
            "configurations",
        ):
            extra.pop(key, None)
        return cls(
            group_id=overview.group_id,
            name=overview.name,
            agent_id=overview.agent_id,
            description=overview.description,
            project_id=overview.project_id,
            dataset_id=overview.dataset_id,
            source_experiments=overview.source_experiments,
            experiment_count=overview.experiment_count,
            experiment_run_count=overview.experiment_run_count,
            configuration_run_count=overview.configuration_run_count,
            first_experiment_created_at=overview.first_experiment_created_at,
            last_experiment_updated_at=overview.last_experiment_updated_at,
            first_experiment_run_created_at=overview.first_experiment_run_created_at,
            last_experiment_run_updated_at=overview.last_experiment_run_updated_at,
            status_summary=overview.status_summary,
            created_at=overview.created_at,
            updated_at=overview.updated_at,
            metadata=overview.metadata,
            extra_fields=extra,
            grouped_configurations=[
                GroupedConfigurationRunRowDTO.from_dict(item)
                for item in _copy_list(rows)
                if isinstance(item, Mapping)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["grouped_configurations"] = [
            row.to_dict() for row in self.grouped_configurations
        ]
        return result


@dataclass(frozen=True)
class ExperimentGroupsPageDTO:
    """Paginated experiment-group overview response."""

    items: list[ExperimentGroupOverviewDTO]
    page: int = 1
    page_size: int = 50
    total: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_previous: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentGroupsPageDTO":
        known = {
            "items",
            "experiment_groups",
            "groups",
            "page",
            "page_size",
            "per_page",
            "total",
            "total_items",
            "total_pages",
            "has_next",
            "has_previous",
            "has_prev",
        }
        rows = payload.get(
            "items", payload.get("experiment_groups", payload.get("groups", []))
        )
        page_size = _int_or_default(
            payload.get("page_size", payload.get("per_page")), 50
        )
        return cls(
            items=[
                ExperimentGroupOverviewDTO.from_dict(item)
                for item in _copy_list(rows)
                if isinstance(item, Mapping)
            ],
            page=_int_or_default(payload.get("page"), 1),
            page_size=page_size,
            total=_int_or_default(payload.get("total", payload.get("total_items")), 0),
            total_pages=_int_or_default(payload.get("total_pages"), 0),
            has_next=_bool_or_default(payload.get("has_next"), False),
            has_previous=_bool_or_default(
                payload.get("has_previous", payload.get("has_prev")),
                False,
            ),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.extra_fields)
        result.update(
            {
                "items": [item.to_dict() for item in self.items],
                "page": self.page,
                "page_size": self.page_size,
                "total": self.total,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_previous": self.has_previous,
            }
        )
        return result


@dataclass(frozen=True)
class GroupedConfigurationRunsPageDTO:
    """Paginated grouped configuration-run response."""

    items: list[GroupedConfigurationRunRowDTO]
    page: int = 1
    page_size: int = 50
    total: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_previous: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroupedConfigurationRunsPageDTO":
        known = {
            "items",
            "configuration_runs",
            "grouped_configurations",
            "rows",
            "page",
            "page_size",
            "per_page",
            "total",
            "total_items",
            "total_pages",
            "has_next",
            "has_previous",
            "has_prev",
        }
        rows = payload.get(
            "items",
            payload.get(
                "configuration_runs",
                payload.get("grouped_configurations", payload.get("rows", [])),
            ),
        )
        page_size = _int_or_default(
            payload.get("page_size", payload.get("per_page")), 50
        )
        return cls(
            items=[
                GroupedConfigurationRunRowDTO.from_dict(item)
                for item in _copy_list(rows)
                if isinstance(item, Mapping)
            ],
            page=_int_or_default(payload.get("page"), 1),
            page_size=page_size,
            total=_int_or_default(payload.get("total", payload.get("total_items")), 0),
            total_pages=_int_or_default(payload.get("total_pages"), 0),
            has_next=_bool_or_default(payload.get("has_next"), False),
            has_previous=_bool_or_default(
                payload.get("has_previous", payload.get("has_prev")),
                False,
            ),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.extra_fields)
        result.update(
            {
                "items": [item.to_dict() for item in self.items],
                "page": self.page,
                "page_size": self.page_size,
                "total": self.total,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_previous": self.has_previous,
            }
        )
        return result


@dataclass
class EvaluatorDTO:
    """Evaluator definition DTO based on observability/evaluator_definition_schema.json."""

    name: str
    measure_id: str
    target_type: str
    judge_config: dict[str, Any]
    id: str | None = None
    description: str | None = None
    sampling_rate: float = 1.0
    target_filters: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    primary_measure_id: str | None = None
    measure: dict[str, Any] | None = None
    created_by: str | None = None
    updated_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the canonical evaluator definition shape."""
        result: dict[str, Any] = {
            "name": self.name,
            "measure_id": self.measure_id,
            "target_type": self.target_type,
            "judge_config": deepcopy(self.judge_config),
            "sampling_rate": self.sampling_rate,
            "target_filters": deepcopy(self.target_filters),
            "is_active": self.is_active,
        }
        for field_name in (
            "id",
            "description",
            "primary_measure_id",
            "measure",
            "created_by",
            "updated_by",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = deepcopy(value)
        return result


@dataclass
class MeasureDTO:
    """Measure DTO based on measures/measure_schema.json."""

    id: str
    label: str
    description: str
    category: str
    measure_type: str = "quality"
    evaluation_method: str = "llm_based"
    target_aspect: str = "response"
    metric_type: str = "single_turn"
    output_type: str = "continuous"
    agent_types: list[str] = field(default_factory=lambda: ["chat"])
    domain_min: float | None = 0.0
    domain_max: float | None = 1.0
    inverse: bool = False
    is_custom: bool = True
    version: str = "1.0.0"
    target_types: list[str] = field(default_factory=list)
    allowed_score_sources: list[str] = field(default_factory=list)
    criteria: list[str] | None = None
    python_packages: list[dict[str, Any]] | None = None
    measure_parameters: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the canonical measure schema shape."""
        result: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "measure_type": self.measure_type,
            "evaluation_method": self.evaluation_method,
            "target_aspect": self.target_aspect,
            "metric_type": self.metric_type,
            "output_type": self.output_type,
            "agent_types": list(self.agent_types),
            "inverse": self.inverse,
            "is_custom": self.is_custom,
            "target_types": list(self.target_types),
            "allowed_score_sources": list(self.allowed_score_sources),
        }
        if self.domain_min is not None:
            result["domain_min"] = self.domain_min
        if self.domain_max is not None:
            result["domain_max"] = self.domain_max
        if self.criteria is not None:
            result["criteria"] = list(self.criteria)
        if self.python_packages is not None:
            result["python_packages"] = deepcopy(self.python_packages)
        if self.measure_parameters is not None:
            result["measure_parameters"] = deepcopy(self.measure_parameters)
        return result


@dataclass
class PlannerDraftDTO:
    """Planner draft DTO for the planner spine."""

    description: str
    agent: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    measures: list[MeasureDTO | dict[str, Any]] = field(default_factory=list)
    draft_id: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the canonical planner draft shape."""
        result: dict[str, Any] = {
            "description": self.description,
            "measures": [
                (
                    measure.to_dict()
                    if isinstance(measure, MeasureDTO)
                    else deepcopy(measure)
                )
                for measure in self.measures
            ],
            "metadata": deepcopy(self.metadata),
        }
        for field_name in (
            "agent",
            "benchmark",
            "draft_id",
            "status",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = deepcopy(value)
        return result


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
