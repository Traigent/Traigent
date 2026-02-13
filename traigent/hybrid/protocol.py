"""Protocol definitions for Hybrid API mode.

Request/response dataclasses for external agentic service communication,
aligned with TVL 0.9 (Tuned Variables Language) specification.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION TVL-0.9-COMPLIANCE TRANSPORT-ABSTRACTION

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class BatchOptions:
    """Options for batch execution control.

    Attributes:
        parallelism: Maximum concurrent executions within batch
        fail_fast: Stop batch on first failure
        timeout_per_item_ms: Per-item timeout (0 = use request timeout)
    """

    parallelism: int = 1
    fail_fast: bool = False
    timeout_per_item_ms: int = 0


@dataclass(slots=True)
class HybridExecuteRequest:
    """Request to execute agent with configuration on inputs.

    Attributes:
        request_id: Idempotency key for retry safety (UUID)
        capability_id: Identifier for the agent capability to invoke
        config: Configuration parameters (TVAR values)
        inputs: List of input examples to process
        session_id: Session ID for stateful agents (echoed from previous response)
        batch_options: Optional batch control settings
        timeout_ms: Request timeout in milliseconds
    """

    capability_id: str
    config: dict[str, Any]
    inputs: list[dict[str, Any]]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    batch_options: BatchOptions | None = None
    timeout_ms: int = 30000

    def __post_init__(self) -> None:
        """Log malformed inputs that violate the OpenAPI contract."""
        missing_indices: list[int] = []
        for idx, item in enumerate(self.inputs):
            if not isinstance(item, dict) or "input_id" not in item:
                missing_indices.append(idx)

        if missing_indices:
            logger.warning(
                "HybridExecuteRequest inputs missing required input_id at indices %s",
                missing_indices,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "config": self.config,
            "inputs": self.inputs,
            "timeout_ms": self.timeout_ms,
        }
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.batch_options is not None:
            result["batch_options"] = {
                "parallelism": self.batch_options.parallelism,
                "fail_fast": self.batch_options.fail_fast,
                "timeout_per_item_ms": self.batch_options.timeout_per_item_ms,
            }
        return result


@dataclass(slots=True)
class HybridExecuteResponse:
    """Response from agent execution.

    Attributes:
        request_id: Echoed request ID for correlation
        execution_id: Unique ID for this execution (for evaluate reference)
        status: Execution status
        outputs: Per-input results with output data
        operational_metrics: cost_usd, latency_ms, tokens, etc.
        quality_metrics: Optional quality metrics if combined mode
        session_id: Session ID for stateful agents
        error: Error details if status is failed/partial
    """

    request_id: str
    execution_id: str
    status: Literal["completed", "partial", "failed"]
    outputs: list[dict[str, Any]]
    operational_metrics: dict[str, float]
    quality_metrics: dict[str, float] | None = None
    session_id: str | None = None
    error: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridExecuteResponse:
        """Create from dictionary (API response)."""
        return cls(
            request_id=data["request_id"],
            execution_id=data.get("execution_id", str(uuid.uuid4())),
            status=data["status"],
            outputs=data.get("outputs", []),
            operational_metrics=data.get("operational_metrics", {}),
            quality_metrics=data.get("quality_metrics"),
            session_id=data.get("session_id"),
            error=data.get("error"),
        )

    def get_total_cost(self) -> float:
        """Extract total cost from operational metrics."""
        return self.operational_metrics.get(
            "total_cost_usd", self.operational_metrics.get("cost_usd", 0.0)
        )


@dataclass(slots=True)
class HybridEvaluateRequest:
    """Request to evaluate outputs against targets.

    Attributes:
        request_id: Idempotency key for retry safety
        capability_id: Identifier for the evaluation capability
        execution_id: Reference to previous execute (avoids resending outputs)
        evaluations: List of output+target pairs to evaluate
        config: Optional config for evaluation-time parameters
        session_id: Session ID for stateful agents
        timeout_ms: Optional server-side timeout budget in milliseconds
    """

    capability_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str | None = None
    evaluations: list[dict[str, Any]] | None = None
    config: dict[str, Any] | None = None
    session_id: str | None = None
    timeout_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "capability_id": self.capability_id,
        }
        if self.execution_id is not None:
            result["execution_id"] = self.execution_id
        if self.evaluations is not None:
            result["evaluations"] = self.evaluations
        if self.config is not None:
            result["config"] = self.config
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.timeout_ms is not None:
            result["timeout_ms"] = self.timeout_ms
        return result


@dataclass(slots=True)
class HybridEvaluateResponse:
    """Response from evaluation.

    Attributes:
        request_id: Echoed request ID for correlation
        status: Evaluation status
        results: Per-example evaluation results with metrics
        aggregate_metrics: Aggregated metrics (mean, std, n per metric)
    """

    request_id: str
    status: Literal["completed", "partial", "failed"]
    results: list[dict[str, Any]]
    aggregate_metrics: dict[str, dict[str, float | int]]
    error: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridEvaluateResponse:
        """Create from dictionary (API response)."""
        return cls(
            request_id=data["request_id"],
            status=data["status"],
            results=data.get("results", []),
            aggregate_metrics=data.get("aggregate_metrics", {}),
            error=data.get("error"),
        )


@dataclass(slots=True)
class ServiceCapabilities:
    """Service capabilities discovered via handshake.

    Returned by the capabilities() endpoint to indicate what
    features the external service supports.

    Attributes:
        version: API version (e.g., "1.0")
        supports_evaluate: Whether separate evaluate endpoint is available
        supports_keep_alive: Whether keep-alive heartbeat is supported
        supports_streaming: Whether streaming responses are supported
        max_batch_size: Maximum inputs per execute request
        max_payload_bytes: Maximum request payload size (None = unlimited)
    """

    version: str
    supports_evaluate: bool = True
    supports_keep_alive: bool = False
    supports_streaming: bool = False
    max_batch_size: int = 100
    max_payload_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServiceCapabilities:
        """Create from dictionary (API response)."""
        return cls(
            version=data.get("version", "1.0"),
            supports_evaluate=data.get("supports_evaluate", True),
            supports_keep_alive=data.get("supports_keep_alive", False),
            supports_streaming=data.get("supports_streaming", False),
            max_batch_size=data.get("max_batch_size", 100),
            max_payload_bytes=data.get("max_payload_bytes"),
        )


@dataclass(slots=True)
class TVARDefinition:
    """TVAR definition following TVL 0.9 specification.

    Defines a tunable variable with its type, domain, and constraints.

    Attributes:
        name: Variable name (must be valid Python identifier)
        type: Variable type (bool, int, float, str, enum)
        domain: Domain specification (values for enum, range for numeric)
        default: Default value (None = first in domain)
        agent: Agent name for multi-agent grouping
        is_tool: True if this TVAR selects/configures an MCP tool
        constraints: Conditional constraints (e.g., "requires model == gpt-4")
    """

    name: str
    type: Literal["bool", "int", "float", "str", "enum"]
    domain: dict[str, Any]
    default: Any = None
    agent: str | None = None
    is_tool: bool = False
    constraints: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TVARDefinition:
        """Create from dictionary (API response)."""
        domain = data.get("domain", {})
        if not isinstance(domain, dict):
            domain = {}

        # Accept wrapper-friendly top-level values/range/resolution keys.
        if "values" in data and "values" not in domain:
            domain["values"] = data["values"]
        if "range" in data and "range" not in domain:
            domain["range"] = data["range"]
        if "resolution" in data and "resolution" not in domain:
            domain["resolution"] = data["resolution"]

        return cls(
            name=data["name"],
            type=data["type"],
            domain=domain,
            default=data.get("default"),
            agent=data.get("agent"),
            is_tool=data.get("is_tool", False),
            constraints=data.get("constraints"),
        )

    def to_traigent_config_space(self) -> Any:
        """Convert TVAR to Traigent configuration space format.

        Returns:
            Configuration space entry in Traigent format:
            - enum: list of values
            - bool: [True, False]
            - int: {"low": min, "high": max, "type": "int"}
            - float: {"low": min, "high": max, "step": resolution}
            - str: list of allowed values
        """
        if self.type == "enum":
            return self.domain.get("values", [])
        elif self.type == "bool":
            return [True, False]
        elif self.type == "int":
            range_spec = self.domain.get("range", [0, 100])
            return {
                "low": range_spec[0],
                "high": range_spec[1],
                "type": "int",
            }
        elif self.type == "float":
            range_spec = self.domain.get("range", [0.0, 1.0])
            result: dict[str, Any] = {
                "low": range_spec[0],
                "high": range_spec[1],
            }
            if "resolution" in self.domain:
                result["step"] = self.domain["resolution"]
            return result
        elif self.type == "str":
            return self.domain.get("values", [])
        else:
            # Fallback for unknown types
            return self.domain.get("values", [])


@dataclass(slots=True)
class ConfigSpaceResponse:
    """Response from config-space discovery endpoint.

    Attributes:
        schema_version: TVL schema version (e.g., "0.9")
        capability_id: Identifier for the capability
        tvars: List of TVAR definitions (also accessible as 'tunables')
        constraints: Structural and behavioral constraints (legacy or typed TVL 0.9)
        objectives: Optional objective definitions (TVL 0.9 compatible JSON)
        exploration: Optional exploration config (strategy, budgets, convergence)
        promotion_policy: Optional promotion policy definition
        defaults: Optional default configuration values
        measures: Optional metric names produced by the service
    """

    schema_version: str
    capability_id: str
    tvars: list[TVARDefinition]
    constraints: dict[str, Any] | list[Any] | None = None
    objectives: list[dict[str, Any]] | None = None
    exploration: dict[str, Any] | None = None
    promotion_policy: dict[str, Any] | None = None
    defaults: dict[str, Any] | None = None
    measures: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigSpaceResponse:
        """Create from dictionary (API response).

        Accepts both 'tvars' and 'tunables' keys for compatibility.
        """
        # Accept both 'tunables' (client-friendly) and 'tvars' (internal)
        tvar_data = data.get("tunables") or data.get("tvars", [])
        tvars = [
            TVARDefinition.from_dict(t) if isinstance(t, dict) else t for t in tvar_data
        ]
        return cls(
            schema_version=data.get("schema_version", "0.9"),
            capability_id=data.get("capability_id", ""),
            tvars=tvars,
            constraints=data.get("constraints"),
            objectives=data.get("objectives"),
            exploration=data.get("exploration"),
            promotion_policy=data.get("promotion_policy"),
            defaults=data.get("defaults"),
            measures=data.get("measures"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Includes both 'tunables' (client-friendly) and 'tvars' (backward compat).
        """
        tvar_dicts = [
            {
                "name": tvar.name,
                "type": tvar.type,
                "domain": tvar.domain,
                "default": tvar.default,
                **({"agent": tvar.agent} if tvar.agent else {}),
                **({"is_tool": tvar.is_tool} if tvar.is_tool else {}),
                **({"constraints": tvar.constraints} if tvar.constraints else {}),
            }
            for tvar in self.tvars
        ]
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "capability_id": self.capability_id,
            "tunables": tvar_dicts,  # Client-facing name
            "tvars": tvar_dicts,  # Backward compatibility
            "constraints": self.constraints or {},
        }
        if self.objectives is not None:
            result["objectives"] = self.objectives
        if self.exploration is not None:
            result["exploration"] = self.exploration
        if self.promotion_policy is not None:
            result["promotion_policy"] = self.promotion_policy
        if self.defaults is not None:
            result["defaults"] = self.defaults
        if self.measures is not None:
            result["measures"] = self.measures
        return result

    @property
    def tunables(self) -> list[TVARDefinition]:
        """Client-friendly alias for tvars."""
        return self.tvars

    def to_traigent_config_space(self) -> dict[str, Any]:
        """Convert all TVARs to Traigent configuration space format."""
        return {tvar.name: tvar.to_traigent_config_space() for tvar in self.tvars}


@dataclass(slots=True)
class HealthCheckResponse:
    """Response from health check endpoint.

    Attributes:
        status: Health status (healthy, degraded, unhealthy)
        version: Service version string
        uptime_seconds: Service uptime
        details: Additional health details
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str | None = None
    uptime_seconds: float | None = None
    details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthCheckResponse:
        """Create from dictionary (API response)."""
        return cls(
            status=data.get("status", "unhealthy"),
            version=data.get("version"),
            uptime_seconds=data.get("uptime_seconds"),
            details=data.get("details"),
        )
