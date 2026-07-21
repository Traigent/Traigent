"""Type definitions for Traigent SDK public API."""

# Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Compatibility FUNC-API-ENTRY FUNC-ORCH-LIFECYCLE REQ-API-001 REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import hashlib
import json
import re
import traceback as traceback_module
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, cast

from traigent.security.redaction import redact_sensitive_data, redact_sensitive_text
from traigent.utils.exceptions import PlatformCapabilityError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

SelectionGrade = Literal["advisory"]
ADVISORY_SELECTION_GRADE: SelectionGrade = "advisory"

# Whole-token markers used ONLY by the last-resort, schema-less orientation
# heuristic (see ``_name_suggests_minimize``). Declared
# ``ObjectiveSchema.orientation`` is always authoritative when a schema is
# available; this list is never consulted in that case.
_MINIMIZE_NAME_TOKENS: frozenset[str] = frozenset(
    {"cost", "latency", "error", "loss", "time", "duration"}
)


def _tokenize_metric_name(name: str) -> set[str]:
    """Split a metric name into lowercased word tokens.

    Handles ``snake_case``, ``kebab-case``, spaces, and ``camelCase`` so that,
    e.g., ``"response_time"`` / ``"responseTime"`` both yield ``{"response",
    "time"}`` while ``"uptime"`` yields ``{"uptime"}`` (crucially NOT
    ``{"time"}``). Prevents arbitrary-substring false positives such as
    ``"uptime" ⊃ "time"``.
    """

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)}


def _name_suggests_minimize(name: str) -> bool:
    """Heuristic: does this metric name look like a minimize objective?

    Whole-token match against :data:`_MINIMIZE_NAME_TOKENS` (not substring),
    so ``"uptime"`` is NOT flagged as minimize while ``"latency_ms"`` and
    ``"total_cost"`` are. Best-effort only — used solely as a last resort when
    no declared :class:`~traigent.core.objectives.ObjectiveSchema` orientation
    is available.
    """

    return bool(_tokenize_metric_name(name) & _MINIMIZE_NAME_TOKENS)


# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from traigent.core.objectives import ObjectiveSchema


__all__ = [
    "SelectionGrade",
    "OptimizationStatus",
    "TrialStatus",
    "MetricCoverage",
    "ComparabilityInfo",
    "StopReason",
    "TrialDatetimeFormat",
    "Trial",
    "TrialError",
    "TrialResult",
    "PresetSelection",
    "serialize_trials",
    "ExampleResult",
    "ExperimentStats",
    "OptimizationResult",
    "SensitivityAnalysis",
    "ConfigurationComparison",
    "ParetoFront",
    "StrategyConfig",
    "OptimizationJob",
    "AgentType",
    "AgentMeta",
    "AgentDefinition",
    "GlobalConfiguration",
    "AgentConfiguration",
    "ConfigSpace",
    "Metrics",
    "Objectives",
]


class OptimizationStatus(StrEnum):
    """Status of an optimization run."""

    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Neutral fallback for statuses the SDK does not recognize (e.g. a newer
    # backend member). Never assert success/failure on an unknown value
    # (issue #1302, AC3).
    UNKNOWN = "unknown"


class TrialStatus(StrEnum):
    """Status of a single trial."""

    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PRUNED = "pruned"
    # Neutral fallback for statuses the SDK does not recognize. Used by inbound
    # deserialization so an unrecognized status is never silently treated as
    # COMPLETED (issue #1302, AC3).
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MetricCoverage:
    """Coverage summary for a single metric across examples."""

    present: int
    total: int
    ratio: float


@dataclass(slots=True)
class ComparabilityInfo:
    """Comparability metadata for ranking-safety decisions."""

    schema_version: str = "1.0"
    primary_objective: str = "accuracy"
    evaluation_mode: str = "unknown"
    total_examples: int = 0
    examples_with_primary_metric: int = 0
    coverage_ratio: float = 0.0
    derivation_path: str = "none"
    ranking_eligible: bool = False
    warning_codes: list[str] = field(default_factory=list)
    per_metric_coverage: dict[str, MetricCoverage] = field(default_factory=dict)
    missing_example_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "primary_objective": self.primary_objective,
            "evaluation_mode": self.evaluation_mode,
            "total_examples": int(self.total_examples),
            "examples_with_primary_metric": int(self.examples_with_primary_metric),
            "coverage_ratio": float(self.coverage_ratio),
            "derivation_path": self.derivation_path,
            "ranking_eligible": bool(self.ranking_eligible),
            "warning_codes": list(self.warning_codes),
            "per_metric_coverage": {
                metric: {
                    "present": int(coverage.present),
                    "total": int(coverage.total),
                    "ratio": float(coverage.ratio),
                }
                for metric, coverage in self.per_metric_coverage.items()
            },
            "missing_example_ids": list(self.missing_example_ids),
        }


# Type alias for optimization stop reasons
# Using Literal provides type safety and IDE autocompletion
StopReason = Literal[
    "max_trials_reached",
    "max_samples_reached",
    "timeout",
    "cost_limit",
    "metric_limit",
    "optimizer",
    "plateau",
    "convergence",
    "semantic_saturation",
    "user_cancelled",
    "condition",  # Generic stop condition triggered
    "error",  # Optimization failed due to an exception
    "vendor_error",  # Provider error (rate limit/quota/service)
    "network_error",  # Connectivity failure
]

TrialDatetimeFormat = Literal["iso", "epoch"]


def _validate_trial_datetime_format(datetime_format: TrialDatetimeFormat) -> None:
    """Validate supported datetime formats for public trial serialization."""
    if datetime_format not in ("iso", "epoch"):
        raise ValueError(
            f"datetime_format must be 'iso' or 'epoch', got {datetime_format!r}"
        )


def _serialize_datetime(
    value: datetime,
    *,
    datetime_format: TrialDatetimeFormat,
) -> str | float:
    """Serialize datetime using the requested wire format."""
    _validate_trial_datetime_format(datetime_format)

    if datetime_format == "iso":
        return value.isoformat()

    return value.timestamp()


def _try_to_dict(
    value: Any,
    *,
    datetime_format: TrialDatetimeFormat,
) -> tuple[bool, Any]:
    """Attempt to call ``value.to_dict()``, returning *(ok, result)*.

    Tries passing *datetime_format* first, then falls back to a bare call.
    Returns ``(False, None)`` when the object has no usable ``to_dict``.
    """
    if not (hasattr(value, "to_dict") and callable(value.to_dict)):
        return False, None

    to_dict = value.to_dict
    try:
        return True, to_dict(datetime_format=datetime_format)
    except TypeError:
        pass
    try:
        return True, to_dict()
    except TypeError:
        return False, None


def _json_safe_trial_value(
    value: Any,
    *,
    datetime_format: TrialDatetimeFormat,
) -> Any:
    """Convert a value into a JSON-safe representation for trial export."""
    if value is None:
        return None

    ok, converted = _try_to_dict(value, datetime_format=datetime_format)
    if ok:
        return _json_safe_trial_value(converted, datetime_format=datetime_format)

    if isinstance(value, datetime):
        return _serialize_datetime(value, datetime_format=datetime_format)

    if isinstance(value, dict):
        return {
            key: _json_safe_trial_value(item, datetime_format=datetime_format)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe_trial_value(item, datetime_format=datetime_format)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


@dataclass
class Trial:
    """Configuration and metadata for a single optimization trial."""

    trial_id: str
    config: dict[str, Any]
    timestamp: datetime
    status: TrialStatus = TrialStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialError:
    """Structured diagnostic context for a failed trial."""

    message: str
    error_type: str
    traceback: str
    timestamp: datetime
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        config: Mapping[str, Any],
        timestamp: datetime | None = None,
    ) -> TrialError:
        """Build a structured trial error from an exception."""
        formatted_traceback = "".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        ).strip()
        if not formatted_traceback:
            formatted_traceback = "".join(
                traceback_module.format_exception_only(type(exc), exc)
            ).strip()

        return cls(
            message=str(exc),
            error_type=type(exc).__name__,
            traceback=formatted_traceback,
            timestamp=timestamp or datetime.now(UTC),
            config=dict(config),
        )

    def to_dict(
        self,
        *,
        datetime_format: TrialDatetimeFormat = "iso",
    ) -> dict[str, Any]:
        """Convert structured error details to a JSON-ready dictionary."""
        return {
            "message": redact_sensitive_text(self.message),
            "error_type": self.error_type,
            "traceback": redact_sensitive_text(self.traceback),
            "timestamp": _serialize_datetime(
                self.timestamp, datetime_format=datetime_format
            ),
            "config": redact_sensitive_data(
                _json_safe_trial_value(self.config, datetime_format=datetime_format)
            ),
        }

    def __repr__(self) -> str:
        return (
            "TrialError("
            "message='<redacted>', "
            f"error_type={self.error_type!r}, "
            f"timestamp={self.timestamp!r}, "
            "config='<redacted>'"
            ")"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialError:
        """Reconstruct a structured trial error from a dictionary."""
        raw_timestamp = data.get("timestamp")
        timestamp = datetime.now(UTC)
        if isinstance(raw_timestamp, str):
            try:
                timestamp = datetime.fromisoformat(raw_timestamp)
            except ValueError:
                pass
        elif isinstance(raw_timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(raw_timestamp, UTC)

        raw_config = data.get("config")
        config = raw_config if isinstance(raw_config, dict) else {}

        return cls(
            message=str(data.get("message", "")),
            error_type=str(data.get("error_type", "")),
            traceback=str(data.get("traceback", "")),
            timestamp=timestamp,
            config=config,
        )


@dataclass
class TrialResult:
    """Result of a single optimization trial.

    Trial-metrics contract (issue #1845) — the meaning of each score-like value:

    * ``score`` — the trial's **optimization signal**: the exact value
      ``best_config`` argmaxes. For a single-objective run it is the objective
      metric's value (e.g. ``metrics["accuracy"]`` when the objective is
      ``"accuracy"``); for a weighted multi-objective run it is the weighted
      selection basis (issue #1682). This is the attribute to read when you want
      "the trial's score"; it mirrors ``metrics["score"]``.
    * ``metrics["score"]`` — same optimization signal, kept in the metrics map
      for transport/serialization parity.
    * ``metrics["accuracy"]`` and any other objective/diagnostic keys — the
      individual metric values. When a custom ``scoring_function`` defines the
      objective, ``metrics["accuracy"]`` is the custom scorer's mean.
    * ``metrics["exact_match_default"]`` — present ONLY when a custom
      ``scoring_function`` owns the ``"accuracy"`` objective. It holds the SDK's
      built-in case-insensitive exact-match scorer as a DIAGNOSTIC. It is never
      the optimization signal and is not what ``best_config`` uses.
    """

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    status: TrialStatus
    duration: float
    timestamp: datetime
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: TrialError | None = None
    # Optimization signal — the value best_config argmaxes (objective metric for
    # single-objective runs, weighted basis for weighted runs). Populated at the
    # trial-construction site; None until the objective value is known or when
    # the objective metric is absent (e.g. a failed trial). See class docstring.
    score: float | None = None

    @property
    def is_successful(self) -> bool:
        """Check if trial completed and produced at least one successful example."""
        if self.status != TrialStatus.COMPLETED:
            return False
        metadata = self.metadata or {}
        successful_examples = metadata.get("successful_examples")
        if isinstance(successful_examples, bool):
            return True
        if isinstance(successful_examples, (int, float)):
            return successful_examples > 0
        if successful_examples is not None:
            try:
                return float(successful_examples) > 0
            except (TypeError, ValueError):
                return True
        return True

    def get_metric(self, name: str, default: float | None = None) -> float | None:
        """Get a specific metric value."""
        return self.metrics.get(name, default)

    @property
    def error_type(self) -> str | None:
        """Get the error type for a failed trial, when available."""
        return self.error.error_type if self.error else None

    @property
    def error_traceback(self) -> str | None:
        """Get the formatted traceback for a failed trial, when available."""
        return self.error.traceback if self.error else None

    def __repr__(self) -> str:
        return (
            "TrialResult("
            f"trial_id={self.trial_id!r}, "
            "config='<redacted>', "
            f"metrics={self.metrics!r}, "
            f"score={self.score!r}, "
            f"status={self.status!r}, "
            f"duration={self.duration!r}, "
            f"timestamp={self.timestamp!r}, "
            "error_message='<redacted>', "
            "metadata='<redacted>', "
            f"error_type={self.error_type!r}"
            ")"
        )

    def to_dict(
        self,
        *,
        include_config: bool = True,
        include_metrics: bool = True,
        include_metadata: bool = True,
        datetime_format: TrialDatetimeFormat = "iso",
    ) -> dict[str, Any]:
        """Convert the trial result to a JSON-ready dictionary.

        Args:
            include_config: Include the trial configuration payload.
            include_metrics: Include the trial metrics payload.
            include_metadata: Include metadata captured during the trial.
            datetime_format: Timestamp encoding, either ``"iso"`` or ``"epoch"``.
        """
        result: dict[str, Any] = {
            "trial_id": self.trial_id,
            "score": self.score,
            "status": self.status.value,
            "duration": float(self.duration),
            "timestamp": _serialize_datetime(
                self.timestamp, datetime_format=datetime_format
            ),
            "error_message": redact_sensitive_text(self.error_message),
            "error": _json_safe_trial_value(
                self.error, datetime_format=datetime_format
            ),
        }

        if include_config:
            result["config"] = redact_sensitive_data(
                _json_safe_trial_value(self.config, datetime_format=datetime_format)
            )
        if include_metrics:
            result["metrics"] = _json_safe_trial_value(
                self.metrics, datetime_format=datetime_format
            )
        if include_metadata:
            result["metadata"] = redact_sensitive_data(
                _json_safe_trial_value(self.metadata, datetime_format=datetime_format)
            )

        return result


@dataclass
class PresetSelection:
    """Advisory strategy-preset selection result.

    Preset selection is deliberately separate from ``best_config``. It is a
    task-local recommendation and never a statistical certificate.
    """

    preset_name: str
    params: dict[str, Any]
    selection_grade: SelectionGrade
    selection_rationale: str
    status: str
    selected_config: dict[str, Any] | None = None
    selected_configs: list[dict[str, Any]] = field(default_factory=list)
    selected_trial_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure advisory preset records cannot carry certificate claims."""
        if self.selection_grade != ADVISORY_SELECTION_GRADE:
            raise ValueError("preset selection_grade must be 'advisory'.")

    def to_metadata(self) -> dict[str, Any]:
        """Return the schema-shaped strategy_preset metadata payload."""
        metadata: dict[str, Any] = {
            "preset_name": self.preset_name,
            "params": dict(self.params),
            "selection_grade": self.selection_grade,
        }
        if self.selection_rationale:
            metadata["selection_rationale"] = self.selection_rationale[:280]
        return metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-ready dictionary."""
        return {
            **self.to_metadata(),
            "status": self.status,
            "selected_config": (
                dict(self.selected_config) if self.selected_config is not None else None
            ),
            "selected_configs": [dict(config) for config in self.selected_configs],
            "selected_trial_indices": list(self.selected_trial_indices),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PresetSelection | None:
        """Reconstruct a PresetSelection from persisted result metadata."""
        if not isinstance(data, Mapping):
            return None
        selected_config_raw = data.get("selected_config")
        selected_configs_raw = data.get("selected_configs")
        selected_indices_raw = data.get("selected_trial_indices")
        if data.get("selection_grade", ADVISORY_SELECTION_GRADE) != (
            ADVISORY_SELECTION_GRADE
        ):
            return None
        return cls(
            preset_name=str(data.get("preset_name", "")),
            params=dict(data.get("params") or {}),
            selection_grade=ADVISORY_SELECTION_GRADE,
            selection_rationale=str(data.get("selection_rationale", "")),
            status=str(data.get("status", "selected")),
            selected_config=(
                dict(selected_config_raw)
                if isinstance(selected_config_raw, Mapping)
                else None
            ),
            selected_configs=[
                dict(config)
                for config in (selected_configs_raw or [])
                if isinstance(config, Mapping)
            ],
            selected_trial_indices=[
                int(index)
                for index in (selected_indices_raw or [])
                if isinstance(index, (int, float)) and not isinstance(index, bool)
            ],
        )


def serialize_trials(
    trials: Sequence[TrialResult],
    *,
    include_config: bool = True,
    include_metrics: bool = True,
    include_metadata: bool = True,
    datetime_format: TrialDatetimeFormat = "iso",
) -> list[dict[str, Any]]:
    """Serialize trial results into JSON-ready dictionaries.

    Args:
        trials: Trial results to serialize.
        include_config: Include each trial's configuration payload.
        include_metrics: Include each trial's metrics payload.
        include_metadata: Include each trial's metadata payload.
        datetime_format: Timestamp encoding, either ``"iso"`` or ``"epoch"``.
    """
    _validate_trial_datetime_format(datetime_format)

    return [
        trial.to_dict(
            include_config=include_config,
            include_metrics=include_metrics,
            include_metadata=include_metadata,
            datetime_format=datetime_format,
        )
        for trial in trials
    ]


@dataclass
class ExampleResult:
    """Result of evaluating a function on a single dataset example."""

    example_id: str
    input_data: dict[str, Any]
    expected_output: Any
    actual_output: Any
    metrics: dict[str, float]
    execution_time: float
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Per-example numeric metrics unpacked from a decorated function that
    #: returned ``(output, metrics_dict)``. Populated by the evaluator's strict
    #: 2-tuple unpack rule (``BaseEvaluator._unpack_user_metrics``); ``None`` for
    #: every other return shape. NOT persisted via ``to_dict`` — these merge
    #: into the per-example custom metrics (and then mean-aggregate onto the
    #: trial) through the normal metrics path, not the example record.
    user_metrics: dict[str, float] | None = None

    @property
    def is_successful(self) -> bool:
        """Check if example evaluation was successful."""
        return self.success and self.error_message is None

    def get_metric(self, name: str, default: float | None = None) -> float | None:
        """Get a specific metric value."""
        return self.metrics.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        from traigent.utils.persistence import _safe_json_value

        return {
            "example_id": self.example_id,
            "input_data": _safe_json_value(self.input_data),
            "expected_output": _safe_json_value(self.expected_output),
            "actual_output": _safe_json_value(self.actual_output),
            "metrics": _safe_json_value(self.metrics),
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": _safe_json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleResult:
        """Reconstruct ExampleResult from a dictionary."""
        return cls(
            example_id=data.get("example_id", ""),
            input_data=data.get("input_data", {}),
            expected_output=data.get("expected_output"),
            actual_output=data.get("actual_output"),
            metrics=data.get("metrics", {}),
            execution_time=data.get("execution_time", 0.0),
            success=data.get("success", False),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExperimentStats:
    """Aggregated statistics for a complete optimization experiment."""

    total_duration: float
    total_cost: float
    unique_configurations: int
    trial_counts: dict[str, int]
    average_trial_duration: float | None = None
    cost_per_configuration: float | None = None
    success_rate: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return stats as a plain dictionary for logging/reporting."""

        return {
            "total_duration": self.total_duration,
            "total_cost": self.total_cost,
            "unique_configurations": self.unique_configurations,
            "trial_counts": dict(self.trial_counts),
            "average_trial_duration": self.average_trial_duration,
            "cost_per_configuration": self.cost_per_configuration,
            "success_rate": self.success_rate,
            "error_message": self.error_message,
        }


@dataclass
class EvalAuditFlag:
    """One eval-dataset example flagged as a likely defect (issue #1880).

    Attributes:
        example_id: The flagged example's id (matrix row key).
        detectors: Names of the detectors that flagged it, e.g.
            ``["never-correct"]``, ``["token-leak"]``,
            ``["cross-family-consensus-on-wrong"]`` (or several).
        suggested_answer: For a cross-family consensus-on-wrong flag, the answer
            the unrelated families agreed on — the "recovers the fix" correction
            when the recorded gold is wrong. ``None`` for the other detectors.
    """

    example_id: str
    detectors: list[str] = field(default_factory=list)
    suggested_answer: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view; omits ``suggested_answer`` when absent."""
        out: dict[str, Any] = {
            "example_id": self.example_id,
            "detectors": list(self.detectors),
        }
        if self.suggested_answer is not None:
            out["suggested_answer"] = self.suggested_answer
        return out


@dataclass
class DetectorStat:
    """Per-detector summary line in an eval audit (issue #1880).

    Attributes:
        name: Detector name.
        active: Whether the detector ran (D7 is inactive with <2 model families;
            D3 is inactive with too few always-correct items for a robust IQR).
        flagged_count: Number of examples this detector flagged.
        eligible_count: Size of the population this detector could flag from.
        flag_rate: ``flagged_count / eligible_count`` (0 when no eligibles).
        lift: Base-rate-adjusted enrichment (definition is per-detector; see
            ``traigent.utils.eval_audit``). ``None`` when undefined.
        note: Why the detector was skipped/inactive, when applicable.
    """

    name: str
    active: bool
    flagged_count: int
    eligible_count: int
    flag_rate: float
    lift: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "name": self.name,
            "active": self.active,
            "flagged_count": self.flagged_count,
            "eligible_count": self.eligible_count,
            "flag_rate": self.flag_rate,
            "lift": self.lift,
            "note": self.note,
        }


@dataclass
class EvalAuditSummary:
    """Roll-up counts and lift for an eval audit (issue #1880).

    Attributes:
        example_count: Number of examples (matrix rows) audited.
        config_count: Number of configurations/replicates (matrix columns).
        family_count: Number of distinct known model families in the run.
        total_flagged: Distinct examples flagged by at least one detector.
        detectors: Per-detector :class:`DetectorStat` keyed by detector name.
    """

    example_count: int
    config_count: int
    family_count: int
    total_flagged: int
    detectors: dict[str, DetectorStat] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "example_count": self.example_count,
            "config_count": self.config_count,
            "family_count": self.family_count,
            "total_flagged": self.total_flagged,
            "detectors": {
                name: stat.to_dict() for name, stat in self.detectors.items()
            },
        }


@dataclass
class DefectSignal:
    """One feature's contribution to an item's continuous defect score (#1881).

    The score is a logistic function ``sigmoid(b0 + Σ wi·xi)`` over per-item
    telemetry features. This records one term of that sum for explainability, so a
    reviewer can see *which* signal moved an item's suspicion, and in which
    direction. Signals are ranked by ``|contribution|`` (strongest driver first).

    Attributes:
        feature: Feature name (``mean_wrong`` / ``never_correct`` / ``instability``).
        value: The feature's value for this item (``xi``).
        weight: The model coefficient applied to the feature (``wi``).
        contribution: ``weight * value`` — the feature's additive push on the
            score's logit. The SIGN is the direction: a positive contribution
            RAISED suspicion, a negative one (possible only under a negative refit
            weight) LOWERED it. Magnitude is how strongly it moved the score.
    """

    feature: str
    value: float
    weight: float
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "feature": self.feature,
            "value": self.value,
            "weight": self.weight,
            "contribution": self.contribution,
        }


@dataclass
class ItemDefectScore:
    """Continuous per-item defect score + dataset-relative rank (issue #1881).

    The continuous layer on top of #1880's binary flags: instead of a yes/no,
    each scored item carries a suspicion score plus its rank within this run, so a
    reviewer gets a sorted worklist and can start at the worst items and stop when
    the signal thins out. Computed as a pure, deterministic function of the
    outcome matrix (#1838) — no LLM calls.

    Attributes:
        example_id: The scored example's id (matrix row key).
        defect_score: Heuristic suspicion score in [0, 1] —
            ``sigmoid(b0 + Σ wi·xi)`` over the telemetry features. With the
            DEFAULT (illustrative, hand-chosen) coefficients this is NOT a
            calibrated probability; it is calibrated only if you refit the
            coefficients on your own labeled subset. Use ``defect_percentile`` to
            RANK items by default.
        defect_percentile: The item's rank within THIS run's scored items —
            the fraction of scored items with a score <= this item's (so the most
            suspicious item is 1.0). This is the trustworthy default output: "top
            5% most suspicious" (percentile >= 0.95) works regardless of absolute
            calibration.
        features: The raw telemetry features that fed the score
            (``never_correct``, ``mean_wrong``, ``instability``).
        contributing_signals: Per-feature ``weight * value`` contributions above
            a small threshold, sorted by contribution descending — the
            explainability view of which signal drove the score.
    """

    example_id: str
    defect_score: float
    defect_percentile: float
    features: dict[str, float] = field(default_factory=dict)
    contributing_signals: list[DefectSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "example_id": self.example_id,
            "defect_score": self.defect_score,
            "defect_percentile": self.defect_percentile,
            "features": dict(self.features),
            "contributing_signals": [s.to_dict() for s in self.contributing_signals],
        }


@dataclass
class EvalAudit:
    """Opt-in eval-dataset defect report for a run (issue #1880 + #1881).

    Computed on demand from the already-persisted per-example x per-config
    outcome matrix (#1838) — no LLM calls, no network. Obtain it via
    :attr:`OptimizationResult.eval_audit`.

    Attributes:
        flagged: Examples flagged by one or more binary detectors (#1880),
            sorted by id.
        summary: Per-detector counts + lift so the user knows how many flags to
            expect and how concentrated they are versus chance.
        scored: The continuous layer (#1881): every scored item (ran in >=2
            configs) with its ``defect_score`` + ``defect_percentile`` +
            ``contributing_signals``, sorted by score descending — a ranked
            worklist over ALL scored items, not just the flagged ones.
    """

    flagged: list[EvalAuditFlag] = field(default_factory=list)
    summary: EvalAuditSummary | None = None
    scored: list[ItemDefectScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "flagged": [flag.to_dict() for flag in self.flagged],
            "summary": self.summary.to_dict() if self.summary else None,
            "scored": [s.to_dict() for s in self.scored],
        }


@dataclass
class OptimizationResult:
    """Complete results from an optimization run.

    Attributes:
        trials: List of all trial results from the optimization.
        best_config: The configuration that achieved the best score, or None
            when no eligible trial produced a winner.
        best_score: The best objective score achieved (None when no eligible trial).
        optimization_id: Unique identifier for this optimization run.
        duration: Total wall-clock time in seconds.
        convergence_info: Dictionary with convergence statistics.
        status: Final status of the optimization (completed, failed, etc.).
        objectives: List of objective names being optimized.
        algorithm: Name of the optimization algorithm used.
        timestamp: When the optimization completed.
        metadata: Additional metadata from the optimization run.
        preset_selection: Advisory strategy-preset selection, when requested.
        total_cost: Total API cost incurred (if tracked).
        total_tokens: Total tokens consumed (if tracked).
        metrics: Aggregated metrics across all trials.
        stop_reason: Why the optimization stopped. One of:
            - "max_trials_reached": Hit the configured max_trials limit
            - "max_samples_reached": Hit the max samples/examples limit
            - "timeout": Exceeded the timeout duration
            - "cost_limit": Hit the cost budget limit mid-run; pre-run cost
              approval declines raise CostLimitExceeded instead.
            - "metric_limit": Hit a soft cumulative metric limit
            - "optimizer": Optimizer decided to stop (exhausted search space)
            - "plateau": Detected optimization plateau (no improvement)
            - "convergence": Built-in hypervolume convergence condition triggered
            - "semantic_saturation": Per-example quality and continuous
              objectives saturated
            - "user_cancelled": User cancelled or declined cost approval
            - "condition": Generic stop condition was triggered
            - "error": Optimization failed due to an exception
            - None: Unknown or not set
        experiment_id: Backend experiment identifier (None if offline/not synced).
        experiment_run_id: Backend experiment-run identifier for analytics reads
            (None if offline/not synced).
        cloud_url: Direct link to the experiment on the cloud portal (None if offline).
        run_label: Human-readable run identifier (e.g. answer_question_20260315_143022_a3f1b2).
        best_config_margin: Winner-vs-runner-up paired margin significance
            qualifying ``best_config`` (issue #1866). Additive — never changes
            the winner. ``None`` when there is no runner-up to compare against;
            otherwise ``{runner_up, delta, ci95, p_value, verdict,
            effective_alpha, n_configs, ...}`` with ``verdict`` in
            ``"clear" | "statistical_tie" | "na"``. The significance threshold is
            Bonferroni-corrected for the best-of-``n_configs`` selection
            (``effective_alpha = alpha / max(1, n_configs - 1)``), applied to both
            the p-value and the CI level. A margin whose CI includes 0 at typical
            n is a tie, not a decision.
    """

    trials: list[TrialResult]
    best_config: dict[str, Any] | None
    best_score: float | None
    optimization_id: str
    duration: float
    convergence_info: dict[str, Any]
    status: OptimizationStatus
    objectives: list[str]
    algorithm: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    preset_selection: PresetSelection | None = None

    # Cost and token tracking
    total_cost: float | None = None
    total_tokens: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # Stop reason - why the optimization stopped (see class docstring for values)
    stop_reason: StopReason | None = None

    # Selection reason for no-winner or fail-closed terminal results. This is
    # distinct from stop_reason: stop_reason explains why execution stopped,
    # while reason_code explains why result selection did not produce a winner.
    reason_code: str | None = None

    # Run identification and cloud link
    experiment_id: str | None = None
    cloud_url: str | None = None
    run_label: str | None = None
    experiment_run_id: str | None = None

    # User-facing, non-fatal warnings about this run (issue #1407). Populated
    # when the run hit a money-correctness gap that the user should see — e.g.
    # one or more models priced to $0 at runtime (a model hard-coded in the
    # optimized function body that no pricing table covers), which silently
    # under-reports spend. Empty by default. Each entry is a human-readable
    # remediation message; structured codes live in ``warning_codes``.
    warnings: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)

    # Provenance of this result (issue #1265): "backend" when the run was
    # tracked end-to-end on the cloud backend, "local" when it was computed
    # locally — either an intentionally local mode or a hybrid/cloud run that
    # degraded to local-only because the backend was unreachable mid-run. Also
    # mirrored in ``metadata["source"]`` for callers that inspect metadata.
    source: str = "backend"

    # Winner-vs-runner-up paired margin significance (issue #1866). Additive
    # qualification of ``best_config`` — it never changes which config won. None
    # when there is no runner-up to compare against (< 2 distinct eligible
    # configs, or no primary objective / per-example data unavailable). When set,
    # a dict ``{runner_up, runner_up_trial_id, delta, ci95, p_value, verdict,
    # test, n_shared_examples, effective_alpha, n_configs, ...}`` where
    # ``verdict`` is ``"clear"`` (the winner significantly beats the runner-up on
    # the primary objective at the multiplicity-corrected ``effective_alpha``),
    # ``"statistical_tie"`` (the margin's CI includes 0 / p above effective_alpha
    # — the winner is interchangeable with the runner-up), or ``"na"`` (two
    # configs but no shared per-example data for a paired test).
    best_config_margin: dict[str, Any] | None = None

    _experiment_stats: ExperimentStats | None = field(
        default=None, init=False, repr=False
    )

    @property
    def experiment_stats(self) -> ExperimentStats:
        """Lazy-loaded, memoized statistics about the full experiment."""

        if self._experiment_stats is None:
            self._experiment_stats = self._calculate_experiment_stats()
        return self._experiment_stats

    @property
    def successful_trials(self) -> list[TrialResult]:
        """Get only successful trials."""
        return [trial for trial in self.trials if trial.is_successful]

    @property
    def failed_trials(self) -> list[TrialResult]:
        """Get only failed trials."""
        return [trial for trial in self.trials if trial.status == TrialStatus.FAILED]

    @property
    def example_matrix(self) -> dict[str, Any]:
        """The per-example x per-trial outcome matrix for this run (issue #1838).

        Built in-memory from the per-example outcomes already carried on each
        trial (``trial.metadata["example_results"]``) — no recomputation and no
        re-collection. ``examples`` is empty when the run captured no
        per-example detail. See ``traigent.utils.outcome_matrix`` for the schema
        and ``load_outcome_matrix`` for reading the persisted artifact.
        """
        from traigent.utils.outcome_matrix import build_outcome_matrix

        return build_outcome_matrix(self)

    @property
    def eval_audit(self) -> EvalAudit | None:
        """Opt-in eval-dataset defect audit for this run (issues #1880 + #1881).

        Runs the deterministic defect detectors (never-correct, token-leak,
        cross-family consensus-on-wrong) over this run's per-example x per-config
        outcome matrix (#1838), and computes the continuous per-item defect score
        + dataset-relative percentile (#1881) exposed as ``EvalAudit.scored`` — a
        ranked worklist over all scored items. It is **opt-in and zero-cost by
        default**: nothing is computed until you read this property, and it makes
        no LLM calls and no network requests — it is a pure function of data the
        run already produced.

        Returns ``None`` when an audit is not applicable — the run captured no
        per-example detail, or spanned fewer than two configurations (columns),
        for which never-correct/consensus are meaningless. When present, the
        cross-family consensus detector activates only if the config space spans
        >=2 model families; single-family runs still get never-correct + token-leak.

        See ``traigent.utils.eval_audit`` for the exact detector rules, thresholds,
        the model-family mapping, and the lift definitions.
        """
        from traigent.utils.eval_audit import compute_eval_audit

        return compute_eval_audit(self.example_matrix)

    @property
    def persistence_failed(self) -> bool:
        """Whether backend persistence/finalization failed after the run completed."""
        return bool(
            isinstance(self.metadata, dict)
            and self.metadata.get("persistence_status") == "failed"
        )

    @property
    def success_rate(self) -> float:
        """Calculate trial success rate."""
        if "OBJECTIVE_UNMATCHED" in self.warning_codes:
            return 0.0
        if not self.trials:
            return 0.0
        return len(self.successful_trials) / len(self.trials)

    def _is_aggregated_run(self) -> bool:
        """Whether the winner was selected over config-aggregated MEAN metrics.

        In aggregated/hybrid mode (``aggregate_configs`` True) result selection
        ranks configs by the MEAN of their replicate metrics and sets
        ``best_score`` to the winning config's mean of the primary objective
        (see ``result_selection._select_best_aggregated`` /
        ``_select_best_weighted_aggregated``). Both stamp
        ``session_summary["selection_mode"] == "aggregated_mean"`` and a
        ``samples_per_config`` map into the result metadata, either of which
        authoritatively marks the run as aggregated.
        """
        metadata = self.metadata
        if not isinstance(metadata, dict):
            return False
        summary = metadata.get("session_summary")
        if not isinstance(summary, dict):
            return False
        return summary.get("selection_mode") == "aggregated_mean" or bool(
            summary.get("samples_per_config")
        )

    def _aggregated_mean_metrics(self) -> dict[str, float] | None:
        """Reconstruct the winning config's MEAN metrics for aggregated runs.

        The naive "first trial whose config == best_config" match returns a
        SINGLE replicate's raw metrics, which contradicts ``best_score`` (the
        replicate MEAN) whenever the winning config has 2+ replicates — making
        ``best_config.json`` self-contradictory and baking a single-replicate
        value into ``export_config``. This averages each numeric metric over the
        winning config's successful replicate set, on the SAME basis as
        ``best_score``: it mirrors ``result_selection._compute_mean_metrics``
        (numeric metrics only, ``examples_attempted`` excluded). The primary
        objective is pinned to ``best_score`` so
        ``best_metrics[primary] == best_score`` holds exactly, immune to
        float summation-order drift.

        Returns ``None`` when the run was not aggregated or no winning-config
        replicates carry metrics, so the caller keeps the single-trial path
        (local runs are unchanged).
        """
        if not self._is_aggregated_run() or not self.best_config:
            return None
        # Prefer the authoritative winner replicate ids stamped by result
        # selection (#1854): grouping is keyed by the config PROJECTED onto
        # config_space_keys while best_config keeps the first trial's FULL
        # config, so full-dict equality silently drops replicates whose aux
        # keys differ. Config equality remains only as a fallback for older
        # persisted results that predate the stamp.
        winning_ids: set[str] | None = None
        metadata = self.metadata
        if isinstance(metadata, dict):
            summary = metadata.get("session_summary")
            if isinstance(summary, dict):
                stamped = summary.get("winning_trial_ids")
                if isinstance(stamped, list) and stamped:
                    winning_ids = {str(trial_id) for trial_id in stamped}
        if winning_ids is not None:
            replicates = [
                trial
                for trial in self.trials
                if trial.is_successful
                and trial.metrics
                and str(trial.trial_id) in winning_ids
            ]
        else:
            replicates = [
                trial
                for trial in self.trials
                if trial.is_successful
                and trial.metrics
                and trial.config == self.best_config
            ]
        if not replicates:
            return None
        numeric_sums: dict[str, float] = {}
        numeric_counts: dict[str, int] = {}
        for trial in replicates:
            for name, value in trial.metrics.items():
                if name == "examples_attempted":
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric_sums[name] = numeric_sums.get(name, 0.0) + float(value)
                    numeric_counts[name] = numeric_counts.get(name, 0) + 1
        if not numeric_sums:
            return None
        mean_metrics: dict[str, float] = {
            name: total / numeric_counts[name] for name, total in numeric_sums.items()
        }
        primary = self.objectives[0] if self.objectives else None
        if (
            primary is not None
            and self.best_score is not None
            and primary in mean_metrics
        ):
            mean_metrics[primary] = self.best_score
        return mean_metrics

    @property
    def best_metrics(self) -> dict[str, float]:
        """Get best metrics from the best trial.

        A result with NO WINNER — ``best_config`` missing/empty AND
        ``best_score`` of ``None`` — makes no winner-metric claims. A valid
        winner whose config happens to be empty still carries a score and
        keeps its metrics.

        In aggregated/hybrid runs the winning config's metrics are the MEAN
        across its replicate set (consistent with ``best_score``), not a single
        replicate — see ``_aggregated_mean_metrics``.
        """
        if not self.best_config and self.best_score is None:
            return {}
        if not self.trials:
            return {}
        aggregated_metrics = self._aggregated_mean_metrics()
        if aggregated_metrics is not None:
            return aggregated_metrics
        if not self.objectives:
            for trial in self.trials:
                if (
                    trial.metrics
                    and trial.is_successful
                    and self.best_config
                    and trial.config == self.best_config
                ):
                    return dict(trial.metrics)
            for trial in self.trials:
                if trial.metrics:
                    return dict(trial.metrics)
            return {}
        # F-C fix: match the trial whose config equals best_config so that
        # best_metrics is always consistent with the actual winning trial, not
        # the highest-scoring trial (which differs for minimize-primary or
        # multi-objective runs).
        if self.best_config:
            for trial in self.trials:
                if trial.metrics and trial.config == self.best_config:
                    return dict(trial.metrics)
        # Fall back to best_trial_id stored in metadata (set by result_selection)
        # when no config match is found (aggregated configs, degenerate configs).
        best_trial_id = self.metadata.get("best_trial_id") if self.metadata else None
        if best_trial_id:
            for trial in self.trials:
                if trial.trial_id == best_trial_id and trial.metrics:
                    return dict(trial.metrics)
        # Last resort: return any successful trial's metrics.
        for trial in self.trials:
            if trial.metrics and trial.is_successful:
                return dict(trial.metrics)
        return {}

    def _calculate_objective_ranges(self) -> dict[str, tuple[float, float]]:
        """Calculate min/max ranges for each objective across all successful trials.

        Returns:
            Dictionary mapping objective names to (min, max) tuples
        """
        ranges = {}

        for obj in self.objectives:
            values = []
            for trial in self.successful_trials:
                if trial.metrics and obj in trial.metrics:
                    value = trial.metrics[obj]
                    if value is not None:
                        values.append(value)

            if values:
                ranges[obj] = (min(values), max(values))
            else:
                # Default range if no valid values
                ranges[obj] = (0.0, 1.0)

        return ranges

    @staticmethod
    def _compute_average_response_time(metadata: dict[str, Any] | None) -> float | None:
        """Compute the mean response time (in seconds) from trial metadata."""

        if not isinstance(metadata, dict):
            return None

        times: list[float] = []

        entries = metadata.get("example_results")
        if entries is None:
            entries = metadata.get("measures")

        if entries:
            if not isinstance(entries, (list, tuple)):
                entries = [entries]
            for entry in entries:
                value = OptimizationResult._extract_response_time(entry)
                if value is not None:
                    times.append(value)

        if not times:
            return None

        return float(sum(times) / len(times))

    @staticmethod
    def _compute_average_response_time_ms(
        metadata: dict[str, Any] | None,
    ) -> float | None:
        """Compute the mean response time in milliseconds from trial metadata."""
        average_seconds = OptimizationResult._compute_average_response_time(metadata)
        if average_seconds is None:
            return None
        return float(average_seconds * 1000.0)

    @staticmethod
    def _coerce_seconds(value: Any) -> float | None:
        """Convert a value to seconds if possible."""
        return OptimizationResult._coerce_float(value)

    @staticmethod
    def _coerce_millis(value: Any) -> float | None:
        """Convert a millisecond value to seconds if possible."""
        seconds = OptimizationResult._coerce_float(value)
        return None if seconds is None else seconds / 1000.0

    @staticmethod
    def _extract_response_time_from_metrics(
        metrics: Any,
        *,
        parent: Mapping[str, Any] | None = None,
    ) -> float | None:
        if isinstance(metrics, Mapping):
            if parent is not None and metrics is parent:
                return None
            return OptimizationResult._extract_response_time_from_mapping(metrics)

        if isinstance(metrics, (list, tuple)):
            for item in metrics:
                if isinstance(item, Mapping):
                    nested = OptimizationResult._extract_response_time_from_mapping(
                        item
                    )
                    if nested is not None:
                        return nested

        return None

    @staticmethod
    def _extract_response_time_from_mapping(
        mapping: Mapping[str, Any],
    ) -> float | None:
        direct = OptimizationResult._coerce_seconds(mapping.get("response_time"))
        if direct is not None:
            return direct

        millis = OptimizationResult._coerce_millis(mapping.get("response_time_ms"))
        if millis is not None:
            return millis

        nested = OptimizationResult._extract_response_time_from_metrics(
            mapping.get("metrics"),
            parent=mapping,
        )
        if nested is not None:
            return nested

        return OptimizationResult._coerce_seconds(mapping.get("execution_time"))

    @staticmethod
    def _extract_response_time(entry: Any) -> float | None:
        """Extract response time (seconds) from dict-like metadata or trial/example objects."""

        if entry is None:
            return None

        if isinstance(entry, Mapping):
            return OptimizationResult._extract_response_time_from_mapping(entry)

        candidate = OptimizationResult._extract_response_time_from_metrics(
            getattr(entry, "metrics", None)
        )
        if candidate is not None:
            return candidate

        return OptimizationResult._coerce_seconds(
            getattr(entry, "execution_time", None)
        )

    @staticmethod
    def _generate_config_hash(config: dict[str, Any]) -> str:
        """Generate a deterministic hash for a configuration dictionary."""

        try:
            from traigent.utils.hashing import (
                generate_config_hash as _generate_config_hash,
            )

            generate_hash = cast(Callable[[dict[str, Any]], str], _generate_config_hash)
            return generate_hash(config)
        except Exception:
            logger.debug("generate_config_hash unavailable, using fallback SHA256 hash")
            sorted_config = json.dumps(config or {}, sort_keys=True)
            return hashlib.sha256(sorted_config.encode()).hexdigest()[:16]

    @dataclass
    class _TrialAggregation:
        counts: dict[str, int]
        unique_configs: set[str]
        sum_duration: float
        counted_durations: int
        other_status: int
        exceptions: int
        computed_total_cost: float

    def _calculate_experiment_stats(self) -> ExperimentStats:
        """Compute aggregate statistics for the optimization experiment."""
        base_counts = {
            "total": len(self.trials),
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "running": 0,
            "pending": 0,
            "not_started": 0,
            "exceptions": 0,
        }

        total_duration = self._resolve_initial_duration()

        try:
            aggregation = self._aggregate_trial_data(
                base_counts.copy(), include_cost=self.total_cost is None
            )
            trial_counts = aggregation.counts
            if aggregation.other_status:
                trial_counts["other"] = aggregation.other_status
            trial_counts["exceptions"] = aggregation.exceptions

            if total_duration <= 0.0 and aggregation.counted_durations:
                total_duration = aggregation.sum_duration

            average_trial_duration = (
                aggregation.sum_duration / aggregation.counted_durations
                if aggregation.counted_durations
                else None
            )

            unique_configuration_count = len(aggregation.unique_configs)
            total_cost = self._resolve_total_cost(aggregation.computed_total_cost)
            cost_per_configuration = (
                total_cost / unique_configuration_count
                if unique_configuration_count
                else None
            )

            success_rate = self.success_rate if self.trials else None

            return ExperimentStats(
                total_duration=float(total_duration),
                total_cost=float(total_cost),
                unique_configurations=unique_configuration_count,
                trial_counts=trial_counts,
                average_trial_duration=average_trial_duration,
                cost_per_configuration=cost_per_configuration,
                success_rate=success_rate,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            fallback_counts = base_counts.copy()
            return ExperimentStats(
                total_duration=0.0,
                total_cost=0.0,
                unique_configurations=0,
                trial_counts=fallback_counts,
                average_trial_duration=None,
                cost_per_configuration=None,
                success_rate=None,
                error_message=str(exc),
            )

    def _resolve_initial_duration(self) -> float:
        """Safely coerce the experiment duration to float."""

        if self.duration is None:
            return 0.0

        try:
            return float(self.duration)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_known_status(status: TrialStatus) -> bool:
        return status in (
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.CANCELLED,
            TrialStatus.RUNNING,
            TrialStatus.PENDING,
            TrialStatus.NOT_STARTED,
        )

    def _record_trial_config(
        self, trial: TrialResult, unique_configs: set[str]
    ) -> None:
        if trial.config is None:
            return
        unique_configs.add(self._generate_config_hash(trial.config))

    def _record_trial_duration(
        self, trial: TrialResult, aggregation: _TrialAggregation
    ) -> None:
        trial_duration = self._coerce_float(trial.duration)
        if trial_duration is None:
            return
        aggregation.sum_duration += trial_duration
        aggregation.counted_durations += 1

    @staticmethod
    def _has_trial_exception(trial: TrialResult) -> bool:
        if trial.status != TrialStatus.FAILED:
            return False
        if trial.error_message:
            return True
        return isinstance(trial.metadata, dict) and bool(trial.metadata.get("failed"))

    def _resolve_trial_cost(self, trial: TrialResult) -> float:
        trial_cost = self._extract_trial_cost(trial)
        return trial_cost if trial_cost is not None else 0.0

    def _aggregate_trial_data(
        self, initial_counts: dict[str, int], include_cost: bool
    ) -> _TrialAggregation:
        """Aggregate trial metrics used for experiment statistics."""

        aggregation = self._TrialAggregation(
            counts=initial_counts,
            unique_configs=set(),
            sum_duration=0.0,
            counted_durations=0,
            other_status=0,
            exceptions=0,
            computed_total_cost=0.0,
        )

        for trial in self.trials:
            self._record_trial_config(trial, aggregation.unique_configs)
            self._record_trial_duration(trial, aggregation)
            self._increment_status_count(trial, aggregation.counts)
            if not self._is_known_status(trial.status):
                aggregation.other_status += 1
            if self._has_trial_exception(trial):
                aggregation.exceptions += 1
            if include_cost:
                aggregation.computed_total_cost += self._resolve_trial_cost(trial)

        return aggregation

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        """Attempt to convert a value to float, returning None on failure."""

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _increment_status_count(
        self, trial: TrialResult, counts: dict[str, int]
    ) -> None:
        """Increment status counters for a given trial."""

        if trial.status == TrialStatus.COMPLETED:
            counts["completed"] += 1
        elif trial.status == TrialStatus.FAILED:
            counts["failed"] += 1
        elif trial.status == TrialStatus.CANCELLED:
            counts["cancelled"] += 1
        elif trial.status == TrialStatus.RUNNING:
            counts["running"] += 1
        elif trial.status == TrialStatus.PENDING:
            counts["pending"] += 1
        elif trial.status == TrialStatus.NOT_STARTED:
            counts["not_started"] += 1

    def _extract_trial_cost(self, trial: TrialResult) -> float | None:
        """Extract total cost from a trial result if available."""

        cost_component = None
        if trial.metrics and "total_cost" in trial.metrics:
            cost_component = trial.metrics.get("total_cost")
        elif isinstance(trial.metadata, dict) and "cost" in trial.metadata:
            cost_meta = trial.metadata.get("cost")
            if isinstance(cost_meta, dict):
                cost_component = cost_meta.get("total_cost")
            else:
                cost_component = cost_meta

        if cost_component is None:
            return None

        return self._coerce_float(cost_component)

    def _resolve_total_cost(self, computed_total_cost: float) -> float:
        """Determine the total cost for the experiment."""

        if self.total_cost is not None:
            try:
                return float(self.total_cost)
            except (TypeError, ValueError):
                return 0.0

        return computed_total_cost

    def _auto_detect_minimize_objectives(self) -> list[str]:
        """Infer minimize objectives from names — last-resort fallback only.

        This runs ONLY when no declared :class:`ObjectiveSchema` orientation and
        no explicit ``minimize_objectives`` list are available. The declared
        orientation is always authoritative when a schema is threaded through
        (see :meth:`_prepare_objective_preferences`). Because a name-based guess
        can be wrong in both directions (a custom minimize metric such as
        ``spend``/``perplexity`` is silently treated as maximize; a maximize
        metric can no longer be mis-flagged, e.g. ``uptime`` is not read as
        ``time``), a :class:`UserWarning` names each guessed orientation and how
        to override it.
        """

        detected = [obj for obj in self.objectives if _name_suggests_minimize(obj)]

        if self.objectives:
            minimize_set = set(detected)
            guessed = ", ".join(
                f"'{obj}'={'minimize' if obj in minimize_set else 'maximize'}"
                for obj in self.objectives
            )
            warnings.warn(
                "Objective orientation was guessed from metric names because no "
                "ObjectiveSchema orientation (or explicit minimize_objectives list) "
                f"was provided: {guessed}. This guess can be wrong for custom metric "
                "names and is NOT authoritative. Declare orientation explicitly by "
                "building an ObjectiveSchema with ObjectiveDefinition(name=..., "
                "orientation='minimize'|'maximize') and passing it as "
                "objective_schema.",
                UserWarning,
                stacklevel=3,
            )

        return detected

    def _resolve_schema_preferences(
        self, objective_schema: ObjectiveSchema | None
    ) -> tuple[dict[str, float] | None, list[str] | None, Any | None]:
        if objective_schema is None:
            return None, None, None

        # Import here to avoid circular dependency during module load
        from traigent.core.objectives import ObjectiveSchema as _ObjectiveSchema

        if isinstance(objective_schema, _ObjectiveSchema):
            weights = {
                obj_def.name: obj_def.weight for obj_def in objective_schema.objectives
            }
            minimize = [
                obj_def.name
                for obj_def in objective_schema.objectives
                if obj_def.orientation == "minimize"
            ]
            return weights, minimize, objective_schema

        return None, None, objective_schema

    def _ensure_objective_weights(
        self, weights: dict[str, float] | None
    ) -> dict[str, float]:
        if weights is None:
            return dict.fromkeys(self.objectives, 1.0)

        normalized = dict(weights)
        for obj in self.objectives:
            normalized.setdefault(obj, 1.0)
        return normalized

    def _prepare_objective_preferences(
        self,
        objective_weights: dict[str, float] | None,
        minimize_objectives: list[str] | None,
        objective_schema: ObjectiveSchema | None,
    ) -> tuple[dict[str, float], list[str], Any | None]:
        """Resolve effective weights, minimize list, and schema for scoring."""

        weights = dict(objective_weights) if objective_weights else None
        minimize = (
            list(minimize_objectives) if minimize_objectives is not None else None
        )

        schema_weights, schema_minimize, schema = self._resolve_schema_preferences(
            objective_schema
        )
        if schema_weights is not None or schema_minimize is not None:
            weights = schema_weights
            minimize = schema_minimize

        if minimize is None:
            minimize = self._auto_detect_minimize_objectives()

        weights = self._ensure_objective_weights(weights)

        return weights, minimize, schema

    @staticmethod
    def _normalize_weight_map(weights: dict[str, float]) -> dict[str, float]:
        """Normalize weight mapping to sum to 1 when total weight is positive."""

        total_weight = sum(weights.values())
        if total_weight > 0:
            return {key: value / total_weight for key, value in weights.items()}
        return weights

    @staticmethod
    def _legacy_normalize_value(
        value: Any, min_val: float, max_val: float, minimize: bool
    ) -> float | None:
        """Normalize a single metric value using legacy orientation rules.

        Zero-span tolerance and fallback follow the TraigentSchema
        multi_objective_semantics meta-contract (epsilon=1e-9, fallback=0.5).
        """

        if value is None:
            return None

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if abs(max_val - min_val) < 1e-9:
            return 0.5

        span = max_val - min_val
        if minimize:
            normalized = (max_val - numeric) / span
        else:
            normalized = (numeric - min_val) / span

        return float(normalized)

    def _normalize_trial_metrics(
        self,
        trial: TrialResult,
        ranges: dict[str, tuple[float, float]],
        minimize_objectives: list[str],
        objective_schema: Any | None,
    ) -> dict[str, float]:
        """Produce normalized metric map for a trial."""

        if not trial.metrics:
            return {}

        if objective_schema is not None:
            schema_normalized = objective_schema.normalize_metrics(
                trial.metrics, ranges
            )
            return cast(dict[str, float], schema_normalized or {})

        normalized: dict[str, float] = {}
        minimize_lookup = set(minimize_objectives)

        for obj in self.objectives:
            if obj not in trial.metrics:
                warnings.warn(
                    f"Objective '{obj}' was not found in the evaluator's returned metrics "
                    f"(got: {sorted(trial.metrics)}). Its weighted score defaults to 0. "
                    "Return it explicitly from your custom_evaluator, e.g. "
                    f"return {{'accuracy': ..., '{obj}': ...}}.",
                    UserWarning,
                    stacklevel=4,
                )
                continue
            if obj not in ranges:
                continue

            min_val, max_val = ranges[obj]
            normalized_value = self._legacy_normalize_value(
                trial.metrics[obj], min_val, max_val, obj in minimize_lookup
            )
            if normalized_value is not None:
                normalized[obj] = normalized_value

        return normalized

    def _score_trial(
        self, normalized_metrics: dict[str, float], objective_weights: dict[str, float]
    ) -> float:
        """Calculate weighted score for a normalized metric map."""

        return sum(
            objective_weights.get(obj, 0.0) * normalized_metrics.get(obj, 0.0)
            for obj in self.objectives
        )

    def _weighted_score_for_trial(
        self,
        trial: TrialResult,
        normalized: dict[str, float],
        normalized_weights: dict[str, float],
        ranges: dict[str, tuple[float, float]],
        objective_schema: Any | None,
    ) -> float:
        """Weighted score for one trial, single-sourced with live selection.

        With a real ObjectiveSchema this delegates to
        ``ObjectiveSchema.compute_weighted_score(metrics, ranges=...)`` — the
        same scoring function live selection uses (issue #1682) — so post-hoc
        analysis and live selection share one implementation. Without a
        schema, the documented legacy cross-SDK path (auto-detected minimize
        list + observed-range normalization) is retained.
        """
        # Import here to avoid circular dependency during module load
        from traigent.core.objectives import ObjectiveSchema as _ObjectiveSchema

        if isinstance(objective_schema, _ObjectiveSchema):
            weighted = objective_schema.compute_weighted_score(
                trial.metrics or {}, ranges=ranges
            )
            return float(weighted) if weighted is not None else 0.0
        return self._score_trial(normalized, normalized_weights) if normalized else 0.0

    def _compute_weighted_scores(
        self,
        ranges: dict[str, tuple[float, float]],
        objective_weights: dict[str, float],
        minimize_objectives: list[str],
        objective_schema: Any | None,
    ) -> list[tuple[TrialResult, float]]:
        """Compute weighted scores for each successful trial."""

        weighted_scores: list[tuple[TrialResult, float]] = []

        for trial in self.successful_trials:
            if not trial.metrics:
                continue

            normalized = self._normalize_trial_metrics(
                trial, ranges, minimize_objectives, objective_schema
            )
            weighted_score = self._weighted_score_for_trial(
                trial, normalized, objective_weights, ranges, objective_schema
            )
            weighted_scores.append((trial, weighted_score))

        return weighted_scores

    def _select_best_weighted_result(
        self,
        weighted_scores: list[tuple[TrialResult, float]],
        objective_orientations: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, float]:
        """Select the best configuration from computed weighted scores.

        Weighted ties are broken with the SAME policy as the authoritative
        terminal selector (issue #1846). All trials whose weighted score is
        within the conservative primary-score tolerance of the maximum
        (``_primary_scores_tied``) are gathered, then resolved with
        ``apply_tie_breaker`` — secondary metrics in the run's declared
        objective order, then ``trial_id``. A plain ``max`` would instead
        crown the first tied trial by iteration order and could name the
        OPPOSITE winner from ``results.best_config`` on the common
        Pareto-endpoint tie (two configs sharing the top weighted aggregate),
        breaking the "the two artifacts agree" guarantee. Single-sourcing the
        tie-break here keeps ``best_weighted_config`` identical to the terminal
        ``best_config`` on ties as well as on unique maxima.
        """
        # Lazy import: result_selection imports TrialResult from this module,
        # so a module-level import would create a cycle. Runtime-only call, so
        # the deferred import is free (same pattern as the ObjectiveSchema
        # imports elsewhere in this class).
        from traigent.core.result_selection import (  # noqa: PLC0415
            _primary_scores_tied,
            apply_tie_breaker,
        )

        if not weighted_scores:
            fallback_trial = (
                self.successful_trials[0] if self.successful_trials else None
            )
            best_config = fallback_trial.config if fallback_trial else self.best_config
            return best_config, 0.0

        best_score = max(score for _, score in weighted_scores)
        tied_trials = [
            trial
            for trial, score in weighted_scores
            if _primary_scores_tied(score, best_score)
        ]
        if len(tied_trials) > 1:
            # Empty tie_breakers ⇒ the default ``min_abs_deviation`` strategy,
            # exactly what the terminal selector applies for the common run
            # (no custom promotion_policy.tie_breaker configured).
            best_trial = apply_tie_breaker(
                tied_trials,
                {},
                self.objectives[0] if self.objectives else "",
                band_target=None,
                objective_order=self.objectives,
                objective_orientations=objective_orientations,
            )
        else:
            best_trial = tied_trials[0]
        return best_trial.config, best_score

    def score_trials(
        self,
        objective_weights: dict[str, float] | None = None,
        minimize_objectives: list[str] | None = None,
        objective_schema: ObjectiveSchema | None = None,
    ) -> list[dict[str, Any]]:
        """Per-trial normalized + weighted scores for cross-SDK parity.

        Returns the per-objective normalized values and weighted score for
        every successful trial. This is the public surface used by the
        ``traigent-cross-sdk-benchmarks`` normalization harness; it shares
        all preference-resolution and ranging logic with
        :meth:`calculate_weighted_scores`.

        Args:
            objective_weights: Same as :meth:`calculate_weighted_scores`.
            minimize_objectives: Same as :meth:`calculate_weighted_scores`.
            objective_schema: Same as :meth:`calculate_weighted_scores`.

        Returns:
            List of dicts, one per successful trial, each containing:
                - ``trial_id``: Trial identifier.
                - ``normalized``: Per-objective normalized values (0..1).
                - ``weighted``: Weighted-sum score using sum-to-one weights.
        """
        if not self.successful_trials:
            return []

        (
            resolved_weights,
            resolved_minimize,
            resolved_schema,
        ) = self._prepare_objective_preferences(
            objective_weights, minimize_objectives, objective_schema
        )
        normalized_weights = self._normalize_weight_map(resolved_weights)
        ranges = self._calculate_objective_ranges()

        per_trial: list[dict[str, Any]] = []
        for trial in self.successful_trials:
            if not trial.metrics:
                per_trial.append(
                    {"trial_id": trial.trial_id, "normalized": {}, "weighted": 0.0}
                )
                continue
            normalized = self._normalize_trial_metrics(
                trial, ranges, resolved_minimize, resolved_schema
            )
            weighted = self._weighted_score_for_trial(
                trial, normalized, normalized_weights, ranges, resolved_schema
            )
            per_trial.append(
                {
                    "trial_id": trial.trial_id,
                    "normalized": dict(normalized),
                    "weighted": float(weighted),
                }
            )
        return per_trial

    def calculate_weighted_scores(
        self,
        objective_weights: dict[str, float] | None = None,
        minimize_objectives: list[str] | None = None,
        objective_schema: ObjectiveSchema | None = None,
    ) -> dict[str, Any]:
        """Calculate weighted scores for all trials using post-experiment normalization.

        This method allows proper multi-objective scoring after the optimization has
        completed, using the full range of observed values for normalization.

        Winner-claim note (FR-SDK-FAIL-CLOSED-PROMOTION-V1): the per-trial
        weighted scores are DESCRIPTIVE statistics and always computed; the
        ``best_weighted_config`` key is a winner claim and is omitted when the
        result carries no certified winner (missing/empty ``best_config``).

        Args:
            objective_weights: Dictionary of objective weights (defaults to equal weights)
            minimize_objectives: List of objectives to minimize (e.g., ["cost", "latency"])
                                If None, auto-detects based on common names
            objective_schema: ObjectiveSchema with orientations and weights (overrides other params)

        Returns:
            Dictionary containing:
                - best_weighted_config: Configuration with best weighted score
                - best_weighted_score: Best weighted score achieved
                - weighted_scores: List of (trial, weighted_score) tuples
                - normalization_ranges: Min/max ranges used for normalization
                - objective_weights_used: Actual weights used (for transparency)
        """
        if not self.successful_trials:
            return {
                **(
                    {"best_weighted_config": self.best_config}
                    if (self.best_config or self.best_score is not None)
                    else {}
                ),
                "best_weighted_score": 0.0,
                "weighted_scores": [],
                "normalization_ranges": {},
                "objective_weights_used": {},
            }

        (
            resolved_weights,
            resolved_minimize,
            resolved_schema,
        ) = self._prepare_objective_preferences(
            objective_weights, minimize_objectives, objective_schema
        )

        normalized_weights = self._normalize_weight_map(resolved_weights)

        ranges = self._calculate_objective_ranges()

        weighted_scores = self._compute_weighted_scores(
            ranges, normalized_weights, resolved_minimize, resolved_schema
        )

        # Break weighted ties in the DECLARED direction (issue #1955): reuse the
        # already-resolved minimize list so the tie-break honors exactly the same
        # orientation as the weighted-score normalization above — a
        # minimize-oriented secondary can no longer crown the WORSE config on a
        # weighted tie (terminal/post-hoc parity).
        _resolved_minimize_set = set(resolved_minimize)
        tie_break_orientations = {
            objective: (
                "minimize" if objective in _resolved_minimize_set else "maximize"
            )
            for objective in self.objectives
        }
        best_weighted_config, best_weighted_score = self._select_best_weighted_result(
            weighted_scores, tie_break_orientations
        )

        return {
            **(
                {"best_weighted_config": best_weighted_config}
                if (self.best_config or self.best_score is not None)
                else {}
            ),
            "best_weighted_score": best_weighted_score,
            "weighted_scores": weighted_scores,
            "normalization_ranges": ranges,
            "objective_weights_used": normalized_weights,
            "minimize_objectives": resolved_minimize,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        import pandas as pd  # noqa: PLC0415

        data = []
        for trial in self.trials:
            row = {
                "trial_id": trial.trial_id,
                "status": trial.status.value,
                "duration": trial.duration,
                "timestamp": trial.timestamp,
                **trial.config,
                **trial.metrics,
            }
            avg_response_time = self._compute_average_response_time(trial.metadata)
            avg_response_time_ms = self._compute_average_response_time_ms(
                trial.metadata
            )
            row["avg_response_time"] = (
                float(avg_response_time) if avg_response_time is not None else None
            )
            row["avg_response_time_ms"] = (
                float(avg_response_time_ms)
                if avg_response_time_ms is not None
                else None
            )
            metadata = trial.metadata or {}
            if "successful_examples" in metadata:
                row["successful_examples"] = metadata.get("successful_examples")
            if "examples_attempted" in metadata:
                row["examples_attempted"] = metadata.get("examples_attempted")
            if "successful_examples" in metadata and "examples_attempted" in metadata:
                row["example_success_summary"] = (
                    f"{metadata.get('successful_examples')}/"
                    f"{metadata.get('examples_attempted')} examples succeeded"
                )
            if "total_example_cost" in metadata:
                row["total_cost"] = metadata.get("total_example_cost")
            elif "total_cost" not in row:
                # Fall back to metrics if already present
                row["total_cost"] = metadata.get("total_cost")
            data.append(row)
        return pd.DataFrame(data)

    @staticmethod
    def _initialize_aggregation_group(config: dict[str, Any]) -> dict[str, Any]:
        return {
            "config": config,
            "samples_count": 0,
            "metrics_sum": {},
            "metrics_count": {},
            "duration_sum": 0.0,
        }

    def _accumulate_trial_metrics(
        self, trial: TrialResult, group: dict[str, Any]
    ) -> None:
        for key, value in (trial.metrics or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                logger.debug(f"Skipping non-numeric metric '{key}': {exc}")
                continue
            group["metrics_sum"][key] = group["metrics_sum"].get(key, 0.0) + numeric
            group["metrics_count"][key] = group["metrics_count"].get(key, 0) + 1

    def _group_trials_for_aggregation(self) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}

        for trial in self.trials:
            cfg = trial.config or {}
            cfg_hash = self._generate_config_hash(cfg)

            group = groups.setdefault(cfg_hash, self._initialize_aggregation_group(cfg))
            group["samples_count"] += 1
            group["duration_sum"] += float(trial.duration or 0.0)

            avg_response_time = self._compute_average_response_time(trial.metadata)
            avg_response_time_ms = self._compute_average_response_time_ms(
                trial.metadata
            )
            if avg_response_time is not None:
                group["response_time_sum"] = (
                    group.get("response_time_sum", 0.0) + avg_response_time
                )
                group["response_time_count"] = group.get("response_time_count", 0) + 1
            if avg_response_time_ms is not None:
                group["response_time_ms_sum"] = (
                    group.get("response_time_ms_sum", 0.0) + avg_response_time_ms
                )
                group["response_time_ms_count"] = (
                    group.get("response_time_ms_count", 0) + 1
                )

            self._accumulate_trial_metrics(trial, group)

        return groups

    @staticmethod
    def _build_aggregated_rows(
        groups: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cfg_hash, group in groups.items():
            row: dict[str, Any] = {
                "config_hash": cfg_hash,
                "samples_count": group["samples_count"],
            }
            for ck, cv in (group["config"] or {}).items():
                row[ck] = cv
            for mk, total in group["metrics_sum"].items():
                count = max(1, group["metrics_count"].get(mk, 1))
                row[f"{mk}"] = total / count
            row["duration"] = (
                (group["duration_sum"] / group["samples_count"])
                if group["samples_count"] > 0
                else 0.0
            )
            if group.get("response_time_count"):
                row["avg_response_time"] = (
                    group["response_time_sum"] / group["response_time_count"]
                )
            if group.get("response_time_ms_count"):
                row["avg_response_time_ms"] = (
                    group["response_time_ms_sum"] / group["response_time_ms_count"]
                )
            rows.append(row)
        return rows

    @staticmethod
    def _sort_aggregated_dataframe(
        df: pd.DataFrame, primary_objective: str | None
    ) -> pd.DataFrame:
        if not primary_objective or primary_objective not in df.columns:
            return df
        ascending = _name_suggests_minimize(primary_objective)
        return df.sort_values(
            by=[primary_objective], ascending=ascending, na_position="last"
        )

    def to_aggregated_dataframe(
        self, primary_objective: str | None = None
    ) -> pd.DataFrame:
        """Aggregate trials by configuration and compute mean metrics.

        Groups repeated samples of the same configuration and computes
        aggregated statistics (currently mean for metrics and duration).

        Args:
            primary_objective: Optional primary objective name to include for clarity.

        Returns:
            DataFrame with one row per unique configuration containing:
                - config parameters as columns
                - samples_count: number of samples for this config
                - <metric>_mean for each metric present
                - duration_mean
        """
        import pandas as pd  # noqa: PLC0415

        if not self.trials:
            return pd.DataFrame()

        groups = self._group_trials_for_aggregation()
        rows = self._build_aggregated_rows(groups)
        df = pd.DataFrame(rows)
        return self._sort_aggregated_dataframe(df, primary_objective)

    def _extract_examples_attempted(self, trial: TrialResult) -> int | None:
        metadata = trial.metadata or {}
        examples_attempted = metadata.get("examples_attempted")
        if examples_attempted is None and hasattr(trial, "example_results"):
            example_results = trial.example_results or []
            examples_attempted = len(example_results)
        if examples_attempted is None:
            return None
        try:
            return int(examples_attempted)
        except (TypeError, ValueError):
            return None

    def _extract_summary_cost(self, trial: TrialResult) -> float | None:
        cost_value = trial.metrics.get("total_cost") if trial.metrics else None
        if cost_value is None:
            cost_value = (trial.metadata or {}).get("total_example_cost")
        return self._coerce_float(cost_value)

    @staticmethod
    def _extract_model_name(trial: TrialResult) -> str | None:
        if not trial.config:
            return None
        return trial.config.get("model")

    def _aggregate_summary_data(
        self,
    ) -> tuple[float, float, int, Counter[str]]:
        total_duration = sum(float(trial.duration or 0.0) for trial in self.trials)
        total_cost = 0.0
        total_examples = 0
        trials_per_model: Counter[str] = Counter()

        for trial in self.trials:
            examples_attempted = self._extract_examples_attempted(trial)
            if examples_attempted is not None:
                total_examples += examples_attempted

            cost_value = self._extract_summary_cost(trial)
            if cost_value is not None:
                total_cost += cost_value

            model_name = self._extract_model_name(trial)
            if model_name:
                trials_per_model[model_name] += 1

        return total_duration, total_cost, total_examples, trials_per_model

    def analyze(
        self,
        objective: str | None = None,
        *,
        importance_method: Literal[
            "variance", "correlation", "permutation"
        ] = "variance",
        elimination_threshold: float = 0.05,
        min_trials_per_value: int = 3,
        directions: dict[str, Literal["maximize", "minimize"]] | None = None,
        configuration_space: dict[str, Any] | None = None,
    ) -> Any:
        """Analyze this optimization result to get variable insights and elimination suggestions.

        This method provides post-optimization analysis to help understand which
        parameters matter, which values are dominated, and suggests refinements
        to the configuration space for future optimization runs.

        Requires the traigent-tuned-variables plugin to be installed for full
        functionality. Install with: pip install traigent-tuned-variables

        Args:
            objective: Primary objective to analyze. Defaults to first objective.
            importance_method: Method for importance calculation:
                - "variance": Variance-based (default, fast)
                - "correlation": Correlation-based
                - "permutation": Permutation-based (more accurate, slower)
            elimination_threshold: Threshold below which variables are considered
                unimportant (default: 0.05)
            min_trials_per_value: Minimum trials per value for reliable statistics
                (default: 3)
            directions: Dictionary mapping objective names to "maximize" or "minimize".
                If not specified, auto-detects based on naming patterns.
            configuration_space: Explicit configuration space. If not provided,
                attempts to infer from trial configs.

        Returns:
            OptimizationAnalysis object containing:
                - variables: Dict of VariableAnalysis for each parameter
                - elimination_suggestions: List of suggested eliminations
                - refined_space: Auto-pruned configuration space for next run

        Raises:
            ImportError: If traigent-tuned-variables plugin is not installed

        Example::

            result = my_agent.optimize()
            analysis = result.analyze("accuracy")
            for var_name, var_analysis in analysis.variables.items():
                print(f"{var_name}: importance={var_analysis.importance:.3f}")
            # Get refined space for next optimization
            refined = analysis.get_refined_space(["accuracy"])
        """
        try:
            from traigent_tuned_variables import VariableAnalyzer
        except ImportError:
            raise ImportError(
                "The analyze() method requires the traigent-tuned-variables plugin. "
                "Install with: pip install traigent-tuned-variables"
            ) from None

        # Use first objective if not specified
        objective_name = objective or (self.objectives[0] if self.objectives else None)
        if objective_name is None:
            raise ValueError(
                "No objective specified and no objectives found in result. "
                "Please specify an objective to analyze."
            )

        # Auto-detect directions if not provided
        if directions is None:
            directions = {}
            for obj in self.objectives:
                directions[obj] = (
                    "minimize" if _name_suggests_minimize(obj) else "maximize"
                )

        # Create analyzer
        analyzer = VariableAnalyzer(
            self,
            importance_method=importance_method,
            elimination_threshold=elimination_threshold,
            min_trials_per_value=min_trials_per_value,
            directions=directions,
            configuration_space=configuration_space,
        )

        return analyzer.analyze(objective_name)

    def get_summary(self) -> dict[str, Any]:
        """Compute high-level summary statistics about the optimization run."""

        total_trials = len(self.trials)
        status_counter = Counter(trial.status for trial in self.trials)

        (
            total_duration,
            total_cost,
            total_examples,
            trials_per_model,
        ) = self._aggregate_summary_data()

        summary = {
            "total_trials": total_trials,
            "completed_trials": status_counter.get(TrialStatus.COMPLETED, 0),
            "pruned_trials": status_counter.get(TrialStatus.PRUNED, 0),
            "failed_trials": status_counter.get(TrialStatus.FAILED, 0),
            "cancelled_trials": status_counter.get(TrialStatus.CANCELLED, 0),
            "total_duration": total_duration,
            "total_cost": total_cost,
            "total_examples_attempted": total_examples,
            "non_failed_trials": total_trials
            - status_counter.get(TrialStatus.FAILED, 0),
            "trials_per_model": dict(trials_per_model),
        }

        # Include best configuration information if available
        summary["best_config"] = self.best_config
        summary["best_score"] = self.best_score
        if self.preset_selection is not None:
            summary["preset_selection"] = self.preset_selection.to_dict()

        return summary


@dataclass
class SensitivityAnalysis:
    """Results of parameter sensitivity analysis."""

    parameter_importance: dict[str, float]
    parameter_interactions: dict[tuple[str, str], float]
    most_important_parameter: str
    statistical_significance: dict[str, float]
    method: str
    confidence_level: float

    def get_top_parameters(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N most important parameters."""
        sorted_params = sorted(
            self.parameter_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return sorted_params[:n]


@dataclass
class ConfigurationComparison:
    """Results of comparing multiple configurations."""

    configurations: list[dict[str, Any]]
    comparison_metrics: dict[str, list[float]]
    statistical_tests: dict[str, dict[str, float]]
    significant_differences: list[tuple[int, int, str]]
    confidence_level: float

    def get_best_configuration(self, metric: str) -> tuple[int, dict[str, Any]]:
        """Get the best configuration for a specific metric."""
        import numpy as np  # noqa: PLC0415

        if metric not in self.comparison_metrics:
            raise ValueError(f"Metric '{metric}' not found in comparison") from None

        values = self.comparison_metrics[metric]
        best_idx = int(np.argmax(values))
        return best_idx, self.configurations[best_idx]


@dataclass
class ParetoFront:
    """Pareto-optimal configurations for multi-objective optimization."""

    configurations: list[dict[str, Any]]
    objective_values: np.ndarray[Any, Any]
    objectives: list[str]
    is_maximized: list[bool]

    def get_best_balanced_config(self) -> dict[str, Any]:
        """Get configuration with best balance across objectives."""
        import numpy as np  # noqa: PLC0415

        # Simple implementation: closest to ideal point
        if len(self.configurations) == 0 or self.objective_values.size == 0:
            raise ValueError("No configurations in Pareto front")

        # Handle single configuration case
        if len(self.configurations) == 1:
            return self.configurations[0]

        # Normalize objectives to [0, 1]
        normalized = self.objective_values.copy()

        # Handle 1D array (single objective)
        if normalized.ndim == 1:
            normalized = normalized.reshape(-1, 1)

        for i, maximize in enumerate(self.is_maximized):
            col = normalized[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                if maximize:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    normalized[:, i] = (max_val - col) / (max_val - min_val)
            else:
                # All values are the same, set to 1.0 for ideal
                normalized[:, i] = 1.0

        # Find closest to ideal point (1, 1, ..., 1)
        if normalized.shape[1] == 1:
            # Single objective - just find the best (1.0 is ideal)
            best_idx = np.argmax(normalized[:, 0])
        else:
            distances = np.linalg.norm(normalized - 1.0, axis=1)
            best_idx = np.argmin(distances)

        return self.configurations[best_idx]  # type: ignore[no-any-return]

    def plot_trade_offs(self, x_objective: str, y_objective: str) -> None:
        """Plot trade-offs between two objectives.

        .. note::
            Multi-objective plotting is an experimental scaffold that has
            not shipped yet. The data backing the plot — ``configurations``,
            ``objective_values``, ``objectives``, ``is_maximized`` — is real
            and can be passed to any plotting library directly. To render a
            Pareto trade-off chart today, use that data with matplotlib,
            plotly, or your visualization library of choice. ``plot_trade_offs``
            itself raises ``NotImplementedError`` until first-party plotting
            ships.
        """
        raise NotImplementedError(
            "ParetoFront.plot_trade_offs() is an experimental visualization "
            "scaffold and has not shipped yet. To render a Pareto trade-off "
            "chart today, read the public fields (configurations, "
            "objective_values, objectives, is_maximized) and pass them to "
            "matplotlib/plotly directly. "
            "Tracking: internal tracking"
        )


@dataclass
class StrategyConfig:
    """Configuration container for optimization strategy parameters.

    .. note::
        This is a **construction utility** — the fields below are
        validated and stored, but they are not yet consumed by the
        SDK's optimization runtime. To configure runtime execution
        today, use the equivalent supported entry points:

        - ``algorithm``: pass via ``@traigent.optimize(algorithm=...)``
          or ``OptimizedFunction.optimize(algorithm=...)``.
        - ``parallel_workers``: pass via
          ``traigent.configure(parallel_workers=...)`` (which also accepts
          a ``parallel_config`` for mode / trial_concurrency /
          example_concurrency / thread_workers).
        - ``resource_limits``: there is **no supported runtime equivalent
          yet** for structured resource limits. For coarse cost/time
          bounds today, use ``cost_limit=...`` /
          ``TRAIGENT_RUN_COST_LIMIT`` on the decorator and
          ``OptimizedFunction.optimize(timeout=...)`` at call time.
          Structured ``resource_limits`` are not wired into the
          runtime yet.

        Future first-party wiring would need to connect
        ``StrategyConfig`` to the optimization pipeline. Holding this
        object does not change optimization behavior.
    """

    algorithm: str
    algorithm_config: dict[str, Any] = field(default_factory=dict)
    parallel_workers: int = 1
    resource_limits: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate strategy configuration."""
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")


@dataclass
class OptimizationJob:
    """Handle for background optimization job."""

    job_id: str
    status: OptimizationStatus
    progress: float
    estimated_completion: datetime | None

    def is_complete(self) -> bool:
        """Check if optimization job is complete."""
        return self.status in {
            OptimizationStatus.COMPLETED,
            OptimizationStatus.FAILED,
            OptimizationStatus.CANCELLED,
        }

    def wait(self, timeout: float | None = None) -> OptimizationResult:
        """Wait for job completion and return results.

        .. note::
            Background-job waiting is part of an experimental scaffold that
            has not shipped yet. The surrounding ``OptimizationJob`` dataclass
            is still useful for inspecting job state via ``.status``,
            ``.progress``, and ``.is_complete()``, but ``wait()`` itself
            cannot block on a real backend until the feature lands.

            For synchronous optimization today, use
            ``OptimizedFunction.optimize()`` (returned by ``@traigent.optimize``)
            which returns an ``OptimizationResult`` directly.
        """
        timeout_note = f" Requested timeout: {timeout}." if timeout is not None else ""
        raise PlatformCapabilityError(
            "OptimizationJob.wait() is part of an experimental background-job "
            "API that has not shipped yet. Use OptimizedFunction.optimize() "
            "for synchronous optimization, or query .status / .is_complete() "
            f"on this job handle for non-blocking state.{timeout_note} "
            "Tracking: internal tracking"
        )


# =============================================================================
# Multi-Agent Configuration Types
# =============================================================================

# Agent type for semantic grouping in multi-agent experiments
AgentType = Literal["llm", "retriever", "router", "tool", "custom"]


@dataclass
class AgentMeta:
    """Optional UI metadata for agent visualization.

    Attributes:
        color: Hex color code for charts and separators (e.g., "#4299E1")
        icon: Icon identifier for UI display (e.g., "robot", "search")
        description: Tooltip description for the agent
    """

    color: str | None = None
    icon: str | None = None
    description: str | None = None


@dataclass
class AgentDefinition:
    """Definition for a single agent within a multi-agent experiment.

    Attributes:
        display_name: Human-readable name shown in UI (e.g., "Financial Agent")
        parameter_keys: List of parameter names belonging to this agent
        measure_ids: List of measure/metric IDs belonging to this agent
        primary_model: Key of the primary model parameter (for Trade-off Analysis)
        order: Display order (lower values appear first)
        agent_type: Semantic type of the agent
        meta: Optional UI metadata
    """

    display_name: str
    parameter_keys: list[str] = field(default_factory=list)
    measure_ids: list[str] = field(default_factory=list)
    primary_model: str | None = None
    order: int | None = None
    agent_type: AgentType | None = None
    meta: AgentMeta | None = None


@dataclass
class GlobalConfiguration:
    """Global (non-agent-specific) parameters and measures.

    Attributes:
        parameter_keys: Parameters not tied to any specific agent
        measure_ids: Measures not tied to any specific agent (e.g., total_cost)
        order: Display order (default: last)
    """

    parameter_keys: list[str] = field(default_factory=list)
    measure_ids: list[str] = field(default_factory=list)
    order: int | None = None


@dataclass
class AgentConfiguration:
    """Complete agent configuration for a multi-agent experiment.

    This configuration is provided by the SDK in experiment_parameters to
    explicitly map parameters and measures to agents, replacing fragile
    label-parsing in the frontend.

    Attributes:
        version: Schema version for compatibility (currently "1.0")
        agents: Map of agent_id to AgentDefinition
        global_config: Configuration for global (non-agent) params/measures
        auto_inferred: True if SDK auto-generated from naming patterns

    Example:
        >>> config = AgentConfiguration(
        ...     agents={
        ...         "financial": AgentDefinition(
        ...             display_name="Financial Agent",
        ...             parameter_keys=["financial_model", "financial_temperature"],
        ...             measure_ids=["financial_accuracy"],
        ...         ),
        ...         "legal": AgentDefinition(
        ...             display_name="Legal Agent",
        ...             parameter_keys=["legal_model"],
        ...             measure_ids=["legal_accuracy"],
        ...         ),
        ...     },
        ...     global_config=GlobalConfiguration(
        ...         measure_ids=["total_cost", "total_latency"],
        ...     ),
        ... )
        >>> payload = config.to_dict()
    """

    version: str = "1.0"
    agents: dict[str, AgentDefinition] = field(default_factory=dict)
    global_config: GlobalConfiguration | None = None
    auto_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary suitable for inclusion in experiment_parameters.
        """
        from dataclasses import asdict

        result: dict[str, Any] = {
            "version": self.version,
            "auto_inferred": self.auto_inferred,
            "agents": {},
        }

        for agent_id, agent in self.agents.items():
            agent_dict: dict[str, Any] = {
                "display_name": agent.display_name,
                "parameter_keys": agent.parameter_keys,
                "measure_ids": agent.measure_ids,
            }
            if agent.primary_model is not None:
                agent_dict["primary_model"] = agent.primary_model
            if agent.order is not None:
                agent_dict["order"] = agent.order
            if agent.agent_type is not None:
                agent_dict["agent_type"] = agent.agent_type
            if agent.meta is not None:
                agent_dict["meta"] = asdict(agent.meta)
            result["agents"][agent_id] = agent_dict

        if self.global_config is not None:
            global_dict: dict[str, Any] = {
                "parameter_keys": self.global_config.parameter_keys,
                "measure_ids": self.global_config.measure_ids,
            }
            if self.global_config.order is not None:
                global_dict["order"] = self.global_config.order
            result["global"] = global_dict

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfiguration:
        """Reconstruct AgentConfiguration from a dictionary.

        Args:
            data: Dictionary from to_dict() or backend response.

        Returns:
            AgentConfiguration instance.
        """
        agents: dict[str, AgentDefinition] = {}
        for agent_id, agent_data in data.get("agents", {}).items():
            meta = None
            if "meta" in agent_data:
                meta = AgentMeta(
                    color=agent_data["meta"].get("color"),
                    icon=agent_data["meta"].get("icon"),
                    description=agent_data["meta"].get("description"),
                )
            agents[agent_id] = AgentDefinition(
                display_name=agent_data.get("display_name", agent_id),
                parameter_keys=agent_data.get("parameter_keys", []),
                measure_ids=agent_data.get("measure_ids", []),
                primary_model=agent_data.get("primary_model"),
                order=agent_data.get("order"),
                agent_type=agent_data.get("agent_type"),
                meta=meta,
            )

        global_config = None
        global_data = data.get("global")
        if global_data:
            global_config = GlobalConfiguration(
                parameter_keys=global_data.get("parameter_keys", []),
                measure_ids=global_data.get("measure_ids", []),
                order=global_data.get("order"),
            )

        return cls(
            version=data.get("version", "1.0"),
            agents=agents,
            global_config=global_config,
            auto_inferred=data.get("auto_inferred", False),
        )


# Type aliases for convenience
ConfigSpace = dict[str, list[Any] | tuple[Any, Any] | Any]
Metrics = dict[str, float]
Objectives = list[str]
