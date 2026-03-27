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


class ObservationType(StrEnum):
    """Frozen Wave 1 observation types."""

    SPAN = "span"
    GENERATION = "generation"
    EVENT = "event"
    TOOL_CALL = "tool_call"


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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    cost_usd: float = 0.0
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
                else self.input_tokens + self.output_tokens
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
    session: SessionDTO | None = None
    observations: list[ObservationDTO] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_required_string("id", self.id, max_length=MAX_IDENTIFIER_LENGTH)
        _validate_required_string("name", self.name, max_length=MAX_NAME_LENGTH)
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
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
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
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            latency_ms=int(payload.get("latency_ms", 0)),
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
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_bookmarked: bool = False
    bookmarked_at: datetime | None = None
    bookmarked_by: str | None = None
    is_published: bool = False
    published_at: datetime | None = None
    published_by: str | None = None
    prompt_links: list[PromptLinkRecord] = field(default_factory=list)
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
            total_input_tokens=int(payload.get("total_input_tokens", 0)),
            total_output_tokens=int(payload.get("total_output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            total_cost_usd=float(payload.get("total_cost_usd", 0.0)),
            total_latency_ms=int(payload.get("total_latency_ms", 0)),
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
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
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
            total_input_tokens=int(payload.get("total_input_tokens", 0)),
            total_output_tokens=int(payload.get("total_output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            total_cost_usd=float(payload.get("total_cost_usd", 0.0)),
            total_latency_ms=int(payload.get("total_latency_ms", 0)),
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
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
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
            latency_ms=int(payload.get("latency_ms", 0)),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
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
