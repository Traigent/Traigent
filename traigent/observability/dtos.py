"""DTOs for the Traigent observability client."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

MAX_IDENTIFIER_LENGTH = 128
MAX_NAME_LENGTH = 255
MAX_ENVIRONMENT_LENGTH = 64
MAX_LABEL_LENGTH = 64
MAX_STATUS_LENGTH = 64
OBSERVABILITY_STATUSES = frozenset({"running", "completed", "failed", "rejected"})
_EXECUTION_CONTEXT_UNSET: Any = object()


def utc_now() -> datetime:
    return datetime.now(UTC)


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return to_iso(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return repr(value)


def _validate_required_string(name: str, value: str, *, max_length: int) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > max_length:
        raise ValueError(
            f"{name} must be less than or equal to {max_length} characters"
        )


def _validate_optional_string(name: str, value: str | None, *, max_length: int) -> None:
    if value is not None and len(value) > max_length:
        raise ValueError(
            f"{name} must be less than or equal to {max_length} characters"
        )


def _validate_non_negative(name: str, value: int | float | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _optional_int(payload: dict[str, Any], name: str) -> int | None:
    value = payload.get(name)
    return None if value is None else int(value)


def _optional_float(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    return None if value is None else float(value)


def _validate_observability_status(name: str, value: str) -> None:
    _validate_required_string(name, value, max_length=MAX_STATUS_LENGTH)
    if value not in OBSERVABILITY_STATUSES:
        allowed = ", ".join(sorted(OBSERVABILITY_STATUSES))
        raise ValueError(f"{name} must be one of: {allowed}")


class ObservationType(StrEnum):
    """Observation kinds accepted by the observability ingest contract."""

    SPAN = "span"
    GENERATION = "generation"
    EVENT = "event"
    TOOL_CALL = "tool_call"
    AGENT = "agent"
    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EVALUATOR = "evaluator"
    EMBEDDING = "embedding"
    GUARDRAIL = "guardrail"


class ThumbRating(StrEnum):
    """Wave 2 thumbs-feedback values."""

    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class PromptReferenceDTO:
    """Prompt registry reference attached to a trace or observation."""

    name: str
    version: int | None = None
    label: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_required_string("name", self.name, max_length=MAX_NAME_LENGTH)
        _validate_optional_string("label", self.label, max_length=MAX_LABEL_LENGTH)
        if self.version is not None and self.version < 1:
            raise ValueError("version must be greater than or equal to 1")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
            "label": self.label,
            "variables": to_jsonable(self.variables),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, init=False)
class ExecutionContextDTO:
    """Versioned, content-free trace lineage accepted by the backend.

    Omitted identifier fields inherit client defaults. An explicitly supplied
    ``None`` is retained by :meth:`to_dict` so a per-trace context can clear a
    default, for example ``ExecutionContextDTO(toolset_id=None)``.
    """

    schema_version: str = "1.0"
    agent_id: str | None = None
    agent_version: str | None = None
    release_id: str | None = None
    deployment_id: str | None = None
    code_revision: str | None = None
    configuration_id: str | None = None
    configuration_version: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
    toolset_id: str | None = None
    toolset_version: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    experiment_run_id: str | None = None
    configuration_run_id: str | None = None
    optimization_run_id: str | None = None
    intervention_id: str | None = None
    _provided_fields: frozenset[str] = field(
        default_factory=frozenset, init=False, repr=False, compare=False
    )

    def __init__(
        self,
        schema_version: str = "1.0",
        agent_id: str | None = _EXECUTION_CONTEXT_UNSET,
        agent_version: str | None = _EXECUTION_CONTEXT_UNSET,
        release_id: str | None = _EXECUTION_CONTEXT_UNSET,
        deployment_id: str | None = _EXECUTION_CONTEXT_UNSET,
        code_revision: str | None = _EXECUTION_CONTEXT_UNSET,
        configuration_id: str | None = _EXECUTION_CONTEXT_UNSET,
        configuration_version: str | None = _EXECUTION_CONTEXT_UNSET,
        prompt_id: str | None = _EXECUTION_CONTEXT_UNSET,
        prompt_version: str | None = _EXECUTION_CONTEXT_UNSET,
        toolset_id: str | None = _EXECUTION_CONTEXT_UNSET,
        toolset_version: str | None = _EXECUTION_CONTEXT_UNSET,
        evaluator_id: str | None = _EXECUTION_CONTEXT_UNSET,
        evaluator_version: str | None = _EXECUTION_CONTEXT_UNSET,
        dataset_id: str | None = _EXECUTION_CONTEXT_UNSET,
        dataset_version: str | None = _EXECUTION_CONTEXT_UNSET,
        experiment_run_id: str | None = _EXECUTION_CONTEXT_UNSET,
        configuration_run_id: str | None = _EXECUTION_CONTEXT_UNSET,
        optimization_run_id: str | None = _EXECUTION_CONTEXT_UNSET,
        intervention_id: str | None = _EXECUTION_CONTEXT_UNSET,
    ) -> None:
        values = {
            "agent_id": agent_id,
            "agent_version": agent_version,
            "release_id": release_id,
            "deployment_id": deployment_id,
            "code_revision": code_revision,
            "configuration_id": configuration_id,
            "configuration_version": configuration_version,
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "toolset_id": toolset_id,
            "toolset_version": toolset_version,
            "evaluator_id": evaluator_id,
            "evaluator_version": evaluator_version,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "experiment_run_id": experiment_run_id,
            "configuration_run_id": configuration_run_id,
            "optimization_run_id": optimization_run_id,
            "intervention_id": intervention_id,
        }
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "_provided_fields",
            frozenset(
                name
                for name, value in values.items()
                if value is not _EXECUTION_CONTEXT_UNSET
            ),
        )
        for name, value in values.items():
            object.__setattr__(
                self,
                name,
                None if value is _EXECUTION_CONTEXT_UNSET else value,
            )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be '1.0'")
        for name, value in self.__dict__.items():
            if name in {"schema_version", "_provided_fields"}:
                continue
            _validate_optional_string(name, value, max_length=MAX_IDENTIFIER_LENGTH)
            if value == "":
                raise ValueError(f"{name} must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionContextDTO:
        allowed = {
            name for name in cls.__dataclass_fields__ if not name.startswith("_")
        }
        unsupported = sorted(set(payload).difference(allowed))
        if unsupported:
            raise ValueError(
                "execution_context contains unsupported field(s): "
                + ", ".join(unsupported)
            )
        context = cls(**payload)
        object.__setattr__(context, "_provided_fields", frozenset(payload))
        return context

    def to_dict(self) -> dict[str, str | None]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
            and (value is not None or key in self._provided_fields)
        }


@dataclass
class CorrelationIds:
    """Optional OTEL/distributed-tracing correlation identifiers."""

    otel_trace_id: str | None = None
    otel_span_id: str | None = None
    otel_parent_span_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.otel_trace_id:
            result["otel_trace_id"] = self.otel_trace_id
        if self.otel_span_id:
            result["otel_span_id"] = self.otel_span_id
        if self.otel_parent_span_id:
            result["otel_parent_span_id"] = self.otel_parent_span_id
        return result


@dataclass
class SessionDTO:
    """Observability session payload."""

    id: str
    user_id: str | None = None
    environment: str | None = None
    release: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_required_string("id", self.id, max_length=MAX_IDENTIFIER_LENGTH)
        _validate_optional_string("user_id", self.user_id, max_length=255)
        _validate_optional_string(
            "environment", self.environment, max_length=MAX_ENVIRONMENT_LENGTH
        )
        _validate_optional_string(
            "release", self.release, max_length=MAX_IDENTIFIER_LENGTH
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "user_id": self.user_id,
            "environment": self.environment,
            "release": self.release,
            "tags": list(self.tags),
            "metadata": to_jsonable(self.metadata),
            "started_at": to_iso(self.started_at),
            "ended_at": to_iso(self.ended_at),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class ObservationDTO:
    """Observability observation payload."""

    id: str
    type: ObservationType
    name: str
    status: str = "running"
    parent_observation_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    model_name: str | None = None
    tool_name: str | None = None
    input_data: Any = None
    output_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    correlation_ids: CorrelationIds | None = None
    prompt_reference: PromptReferenceDTO | None = None
    children: list[ObservationDTO] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_required_string("id", self.id, max_length=MAX_IDENTIFIER_LENGTH)
        _validate_required_string("name", self.name, max_length=MAX_NAME_LENGTH)
        _validate_observability_status("status", self.status)
        _validate_optional_string(
            "parent_observation_id",
            self.parent_observation_id,
            max_length=MAX_IDENTIFIER_LENGTH,
        )
        _validate_optional_string(
            "model_name", self.model_name, max_length=MAX_NAME_LENGTH
        )
        _validate_optional_string(
            "tool_name", self.tool_name, max_length=MAX_NAME_LENGTH
        )
        _validate_non_negative("latency_ms", self.latency_ms)
        _validate_non_negative("input_tokens", self.input_tokens)
        _validate_non_negative("output_tokens", self.output_tokens)
        _validate_non_negative("total_tokens", self.total_tokens)
        _validate_non_negative("cost_usd", self.cost_usd)
        if self.type is ObservationType.EVENT and self.children:
            raise ValueError("event observations cannot have children")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "status": self.status,
            "parent_observation_id": self.parent_observation_id,
            "started_at": to_iso(self.started_at),
            "ended_at": to_iso(self.ended_at),
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": (
                self.total_tokens
                if self.total_tokens is not None
                else (
                    self.input_tokens + self.output_tokens
                    if self.input_tokens is not None and self.output_tokens is not None
                    else None
                )
            ),
            "cost_usd": self.cost_usd,
            "model_name": self.model_name,
            "tool_name": self.tool_name,
            "input_data": to_jsonable(self.input_data),
            "output_data": to_jsonable(self.output_data),
            "metadata": to_jsonable(self.metadata),
            "children": [child.to_dict() for child in self.children],
        }
        if self.correlation_ids:
            payload["correlation_ids"] = self.correlation_ids.to_dict()
        if self.prompt_reference:
            payload["prompt_reference"] = self.prompt_reference.to_dict()
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class TraceDTO:
    """Observability trace payload."""

    id: str
    name: str
    status: str = "running"
    session_id: str | None = None
    user_id: str | None = None
    environment: str | None = None
    release: str | None = None
    custom_trace_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    output_data: Any = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    correlation_ids: CorrelationIds | None = None
    prompt_reference: PromptReferenceDTO | None = None
    execution_context: ExecutionContextDTO | None = None
    session: SessionDTO | None = None
    observations: list[ObservationDTO] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_required_string("id", self.id, max_length=MAX_IDENTIFIER_LENGTH)
        _validate_required_string("name", self.name, max_length=MAX_NAME_LENGTH)
        _validate_observability_status("status", self.status)
        _validate_optional_string(
            "session_id", self.session_id, max_length=MAX_IDENTIFIER_LENGTH
        )
        _validate_optional_string("user_id", self.user_id, max_length=255)
        _validate_optional_string(
            "environment", self.environment, max_length=MAX_ENVIRONMENT_LENGTH
        )
        _validate_optional_string(
            "release", self.release, max_length=MAX_IDENTIFIER_LENGTH
        )
        _validate_optional_string(
            "custom_trace_id",
            self.custom_trace_id,
            max_length=MAX_IDENTIFIER_LENGTH,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "environment": self.environment,
            "release": self.release,
            "custom_trace_id": self.custom_trace_id,
            "tags": list(self.tags),
            "metadata": to_jsonable(self.metadata),
            "input_data": to_jsonable(self.input_data),
            "output_data": to_jsonable(self.output_data),
            "started_at": to_iso(self.started_at),
            "ended_at": to_iso(self.ended_at),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }
        if self.session is not None:
            payload["session"] = self.session.to_dict()
        if self.correlation_ids is not None:
            payload["correlation_ids"] = self.correlation_ids.to_dict()
        if self.prompt_reference is not None:
            payload["prompt_reference"] = self.prompt_reference.to_dict()
        if self.execution_context is not None:
            payload["execution_context"] = self.execution_context.to_dict()
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class FlushResult:
    """Client-visible flush result."""

    success: bool
    items_sent: int
    items_pending: int
    items_dropped: int
    successful_batches: int
    failed_batches: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceCommentRecord:
    """Stored trace comment."""

    id: str
    trace_id: str
    author_user_id: str
    content: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceCommentRecord:
        return cls(
            id=payload["id"],
            trace_id=payload["trace_id"],
            author_user_id=payload["author_user_id"],
            content=payload["content"],
            created_at=from_iso(payload.get("created_at")),
            updated_at=from_iso(payload.get("updated_at")),
        )


@dataclass
class TraceCommentsResponse:
    """Trace comment list response."""

    trace_id: str
    count: int
    items: list[TraceCommentRecord]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceCommentsResponse:
        return cls(
            trace_id=payload["trace_id"],
            count=int(payload.get("count", 0)),
            items=[
                TraceCommentRecord.from_dict(item) for item in payload.get("items", [])
            ],
        )


@dataclass
class TraceFeedbackSummary:
    """Aggregated thumbs counts for a trace."""

    up_count: int = 0
    down_count: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceFeedbackSummary:
        return cls(
            up_count=int(payload.get("up_count", 0)),
            down_count=int(payload.get("down_count", 0)),
        )


@dataclass
class TraceFeedbackRecord:
    """Stored per-user feedback on a trace."""

    id: str
    trace_id: str
    author_user_id: str
    rating: ThumbRating
    comment: str | None = None
    correction_output: Any = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceFeedbackRecord:
        return cls(
            id=payload["id"],
            trace_id=payload["trace_id"],
            author_user_id=payload["author_user_id"],
            rating=ThumbRating(payload["rating"]),
            comment=payload.get("comment"),
            correction_output=payload.get("correction_output"),
            created_at=from_iso(payload.get("created_at")),
            updated_at=from_iso(payload.get("updated_at")),
        )


@dataclass
class TraceFeedbackResponse:
    """Feedback upsert response."""

    feedback: TraceFeedbackRecord
    summary: TraceFeedbackSummary

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceFeedbackResponse:
        return cls(
            feedback=TraceFeedbackRecord.from_dict(payload["feedback"]),
            summary=TraceFeedbackSummary.from_dict(payload.get("summary", {})),
        )


@dataclass
class TraceCollaborationState:
    """Bookmark/publish state and feedback summary for a trace."""

    is_bookmarked: bool = False
    bookmarked_at: datetime | None = None
    bookmarked_by: str | None = None
    is_published: bool = False
    published_at: datetime | None = None
    published_by: str | None = None
    comment_count: int = 0
    feedback_summary: TraceFeedbackSummary = field(default_factory=TraceFeedbackSummary)
    current_user_feedback: TraceFeedbackRecord | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceCollaborationState:
        current_feedback = payload.get("current_user_feedback")
        return cls(
            is_bookmarked=bool(payload.get("is_bookmarked", False)),
            bookmarked_at=from_iso(payload.get("bookmarked_at")),
            bookmarked_by=payload.get("bookmarked_by"),
            is_published=bool(payload.get("is_published", False)),
            published_at=from_iso(payload.get("published_at")),
            published_by=payload.get("published_by"),
            comment_count=int(payload.get("comment_count", 0)),
            feedback_summary=TraceFeedbackSummary.from_dict(
                payload.get("feedback_summary", {})
            ),
            current_user_feedback=(
                TraceFeedbackRecord.from_dict(current_feedback)
                if current_feedback
                else None
            ),
        )


@dataclass
class PromptLinkRecord:
    """Resolved prompt linkage returned by observability queries."""

    id: str
    trace_id: str
    observation_id: str | None
    prompt_id: str
    prompt_version_id: str
    prompt_name: str
    prompt_type: str
    prompt_version: int
    prompt_label: str | None
    variables: dict[str, Any]
    trace_name: str
    trace_status: str
    observation_name: str | None
    session_id: str | None
    user_id: str | None
    environment: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    latency_ms: int | None
    linked_at: datetime | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptLinkRecord:
        return cls(
            id=payload["id"],
            trace_id=payload["trace_id"],
            observation_id=payload.get("observation_id"),
            prompt_id=payload.get("prompt_id", ""),
            prompt_version_id=payload.get("prompt_version_id", ""),
            prompt_name=payload.get("prompt_name", ""),
            prompt_type=payload.get("prompt_type", ""),
            prompt_version=int(payload.get("prompt_version", 0)),
            prompt_label=payload.get("prompt_label"),
            variables=dict(payload.get("variables") or {}),
            trace_name=payload.get("trace_name", ""),
            trace_status=payload.get("trace_status", ""),
            observation_name=payload.get("observation_name"),
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id"),
            environment=payload.get("environment"),
            input_tokens=_optional_int(payload, "input_tokens"),
            output_tokens=_optional_int(payload, "output_tokens"),
            total_tokens=_optional_int(payload, "total_tokens"),
            cost_usd=_optional_float(payload, "cost_usd"),
            latency_ms=_optional_int(payload, "latency_ms"),
            linked_at=from_iso(payload.get("linked_at")),
        )


@dataclass
class PaginationInfo:
    """Pagination envelope returned by observability list endpoints."""

    page: int
    per_page: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaginationInfo:
        return cls(
            page=int(payload.get("page", 1)),
            per_page=int(payload.get("per_page", 20)),
            total=int(payload.get("total", 0)),
            total_pages=int(payload.get("total_pages", 1)),
            has_next=bool(payload.get("has_next", False)),
            has_prev=bool(payload.get("has_prev", False)),
        )


@dataclass
class TraceRecord:
    """Trace summary/detail returned by the query API."""

    id: str
    name: str
    status: str
    session_id: str | None = None
    user_id: str | None = None
    environment: str | None = None
    release: str | None = None
    custom_trace_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    output_data: Any = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    observation_count: int = 0
    root_observation_count: int = 0
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    total_latency_ms: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_bookmarked: bool = False
    bookmarked_at: datetime | None = None
    bookmarked_by: str | None = None
    is_published: bool = False
    published_at: datetime | None = None
    published_by: str | None = None
    prompt_links: list[PromptLinkRecord] = field(default_factory=list)
    execution_context: dict[str, Any] | None = None
    session: SessionRecord | None = None
    collaboration: TraceCollaborationState | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        include_session: bool = True,
        include_collaboration: bool = True,
    ) -> TraceRecord:
        session_payload = payload.get("session")
        collaboration_payload = payload.get("collaboration")
        return cls(
            id=payload["id"],
            name=payload["name"],
            status=payload.get("status", "running"),
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id"),
            environment=payload.get("environment"),
            release=payload.get("release"),
            custom_trace_id=payload.get("custom_trace_id"),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            input_data=payload.get("input_data"),
            output_data=payload.get("output_data"),
            started_at=from_iso(payload.get("started_at")),
            ended_at=from_iso(payload.get("ended_at")),
            observation_count=int(payload.get("observation_count", 0)),
            root_observation_count=int(payload.get("root_observation_count", 0)),
            total_input_tokens=_optional_int(payload, "total_input_tokens"),
            total_output_tokens=_optional_int(payload, "total_output_tokens"),
            total_tokens=_optional_int(payload, "total_tokens"),
            total_cost_usd=_optional_float(payload, "total_cost_usd"),
            total_latency_ms=_optional_int(payload, "total_latency_ms"),
            created_at=from_iso(payload.get("created_at")),
            updated_at=from_iso(payload.get("updated_at")),
            is_bookmarked=bool(payload.get("is_bookmarked", False)),
            bookmarked_at=from_iso(payload.get("bookmarked_at")),
            bookmarked_by=payload.get("bookmarked_by"),
            is_published=bool(payload.get("is_published", False)),
            published_at=from_iso(payload.get("published_at")),
            published_by=payload.get("published_by"),
            prompt_links=[
                PromptLinkRecord.from_dict(item)
                for item in payload.get("prompt_links") or []
            ],
            execution_context=(
                dict(payload["execution_context"])
                if isinstance(payload.get("execution_context"), dict)
                else None
            ),
            session=(
                SessionRecord.from_dict(session_payload, include_traces=False)
                if include_session and isinstance(session_payload, dict)
                else None
            ),
            collaboration=(
                TraceCollaborationState.from_dict(collaboration_payload)
                if include_collaboration and isinstance(collaboration_payload, dict)
                else None
            ),
        )


@dataclass
class SessionRecord:
    """Session summary/detail returned by the query API."""

    id: str
    user_id: str | None = None
    environment: str | None = None
    release: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    trace_count: int = 0
    observation_count: int = 0
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    total_latency_ms: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    traces: list[TraceRecord] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        include_traces: bool = True,
    ) -> SessionRecord:
        traces_payload = payload.get("traces") or []
        return cls(
            id=payload["id"],
            user_id=payload.get("user_id"),
            environment=payload.get("environment"),
            release=payload.get("release"),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            started_at=from_iso(payload.get("started_at")),
            ended_at=from_iso(payload.get("ended_at")),
            trace_count=int(payload.get("trace_count", 0)),
            observation_count=int(payload.get("observation_count", 0)),
            total_input_tokens=_optional_int(payload, "total_input_tokens"),
            total_output_tokens=_optional_int(payload, "total_output_tokens"),
            total_tokens=_optional_int(payload, "total_tokens"),
            total_cost_usd=_optional_float(payload, "total_cost_usd"),
            total_latency_ms=_optional_int(payload, "total_latency_ms"),
            created_at=from_iso(payload.get("created_at")),
            updated_at=from_iso(payload.get("updated_at")),
            traces=(
                [
                    TraceRecord.from_dict(
                        trace_payload,
                        include_session=False,
                        include_collaboration=False,
                    )
                    for trace_payload in traces_payload
                ]
                if include_traces
                else []
            ),
        )


@dataclass
class ObservationRecord:
    """Observation tree node returned by the query API."""

    id: str
    trace_id: str
    type: ObservationType
    name: str
    status: str
    depth: int = 0
    sequence_number: int = 0
    parent_observation_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    model_name: str | None = None
    tool_name: str | None = None
    input_data: Any = None
    output_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_links: list[PromptLinkRecord] = field(default_factory=list)
    children: list[ObservationRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservationRecord:
        return cls(
            id=payload["id"],
            trace_id=payload["trace_id"],
            type=ObservationType(payload["type"]),
            name=payload["name"],
            status=payload.get("status", "running"),
            depth=int(payload.get("depth", 0)),
            sequence_number=int(payload.get("sequence_number", 0)),
            parent_observation_id=payload.get("parent_observation_id"),
            started_at=from_iso(payload.get("started_at")),
            ended_at=from_iso(payload.get("ended_at")),
            latency_ms=_optional_int(payload, "latency_ms"),
            input_tokens=_optional_int(payload, "input_tokens"),
            output_tokens=_optional_int(payload, "output_tokens"),
            total_tokens=_optional_int(payload, "total_tokens"),
            cost_usd=_optional_float(payload, "cost_usd"),
            model_name=payload.get("model_name"),
            tool_name=payload.get("tool_name"),
            input_data=payload.get("input_data"),
            output_data=payload.get("output_data"),
            metadata=dict(payload.get("metadata") or {}),
            prompt_links=[
                PromptLinkRecord.from_dict(item)
                for item in payload.get("prompt_links") or []
            ],
            children=[cls.from_dict(child) for child in payload.get("children", [])],
        )


@dataclass
class TraceListResponse:
    """Trace list response envelope."""

    items: list[TraceRecord]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceListResponse:
        return cls(
            items=[TraceRecord.from_dict(item) for item in payload.get("items", [])],
            pagination=PaginationInfo.from_dict(payload.get("pagination", {})),
        )


@dataclass
class SessionListResponse:
    """Session list response envelope."""

    items: list[SessionRecord]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionListResponse:
        return cls(
            items=[
                SessionRecord.from_dict(item, include_traces=False)
                for item in payload.get("items", [])
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination", {})),
        )


@dataclass
class TraceObservationsResponse:
    """Nested trace observation response envelope."""

    trace_id: str
    observation_count: int
    items: list[ObservationRecord]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceObservationsResponse:
        return cls(
            trace_id=payload["trace_id"],
            observation_count=int(payload.get("observation_count", 0)),
            items=[
                ObservationRecord.from_dict(item) for item in payload.get("items", [])
            ],
        )


ObservationDTO.__annotations__["children"] = list[ObservationDTO]
SessionRecord.__annotations__["traces"] = list[TraceRecord]
ObservationRecord.__annotations__["children"] = list[ObservationRecord]
