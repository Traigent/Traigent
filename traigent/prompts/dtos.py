"""DTOs for the Traigent prompt management client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PromptType(StrEnum):
    TEXT = "text"
    CHAT = "chat"


@dataclass(frozen=True)
class PaginationInfo:
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


@dataclass(frozen=True)
class ChatPromptMessage:
    role: str
    content: str
    name: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChatPromptMessage:
        return cls(
            role=str(payload.get("role", "")),
            content=str(payload.get("content", "")),
            name=payload.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class PromptVersionRecord:
    id: str
    version: int
    prompt_type: PromptType
    prompt_text: str | None
    chat_messages: list[ChatPromptMessage]
    config: dict[str, Any]
    commit_message: str | None
    variable_names: list[str]
    labels: list[str]
    created_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptVersionRecord:
        return cls(
            id=str(payload.get("id", "")),
            version=int(payload.get("version", 0)),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            prompt_text=payload.get("prompt_text"),
            chat_messages=[
                ChatPromptMessage.from_dict(message)
                for message in payload.get("chat_messages") or []
            ],
            config=dict(payload.get("config") or {}),
            commit_message=payload.get("commit_message"),
            variable_names=[str(item) for item in payload.get("variable_names") or []],
            labels=[str(item) for item in payload.get("labels") or []],
            created_by=payload.get("created_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class PromptSummary:
    id: str
    name: str
    prompt_type: PromptType
    description: str | None
    tags: list[str]
    latest_version: int
    version_count: int
    labels: dict[str, int]
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptSummary:
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            description=payload.get("description"),
            tags=[str(item) for item in payload.get("tags") or []],
            latest_version=int(payload.get("latest_version", 0)),
            version_count=int(payload.get("version_count", 0)),
            labels={
                str(key): int(value)
                for key, value in (payload.get("labels") or {}).items()
            },
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class PromptDetail(PromptSummary):
    versions: list[PromptVersionRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptDetail:
        summary = PromptSummary.from_dict(payload)
        return cls(
            id=summary.id,
            name=summary.name,
            prompt_type=summary.prompt_type,
            description=summary.description,
            tags=summary.tags,
            latest_version=summary.latest_version,
            version_count=summary.version_count,
            labels=summary.labels,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            versions=[
                PromptVersionRecord.from_dict(version)
                for version in payload.get("versions") or []
            ],
        )


@dataclass(frozen=True)
class PromptListResponse:
    items: list[PromptSummary]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptListResponse:
        return cls(
            items=[
                PromptSummary.from_dict(item) for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    description: str | None
    version: int
    prompt_type: PromptType
    prompt_text: str | None
    chat_messages: list[ChatPromptMessage]
    config: dict[str, Any]
    commit_message: str | None
    variable_names: list[str]
    labels: list[str]
    resolved_label: str | None
    created_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResolvedPrompt:
        return cls(
            name=str(payload.get("name", "")),
            description=payload.get("description"),
            version=int(payload.get("version", 0)),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            prompt_text=payload.get("prompt_text"),
            chat_messages=[
                ChatPromptMessage.from_dict(message)
                for message in payload.get("chat_messages") or []
            ],
            config=dict(payload.get("config") or {}),
            commit_message=payload.get("commit_message"),
            variable_names=[str(item) for item in payload.get("variable_names") or []],
            labels=[str(item) for item in payload.get("labels") or []],
            resolved_label=payload.get("resolved_label"),
            created_by=payload.get("created_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )

    def to_prompt_reference(
        self,
        *,
        label: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
            "variables": dict(variables or {}),
        }
        resolved_label = label if label is not None else self.resolved_label
        if resolved_label:
            payload["label"] = resolved_label
        return payload


@dataclass(frozen=True)
class PromptAnalyticsTotals:
    link_count: int
    trace_count: int
    observation_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: int
    last_used_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptAnalyticsTotals:
        return cls(
            link_count=int(payload.get("link_count", 0)),
            trace_count=int(payload.get("trace_count", 0)),
            observation_count=int(payload.get("observation_count", 0)),
            total_input_tokens=int(payload.get("total_input_tokens", 0)),
            total_output_tokens=int(payload.get("total_output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            total_cost_usd=float(payload.get("total_cost_usd", 0.0)),
            total_latency_ms=int(payload.get("total_latency_ms", 0)),
            last_used_at=payload.get("last_used_at"),
        )


@dataclass(frozen=True)
class PromptVersionAnalytics:
    version: int
    prompt_type: PromptType
    labels: list[str]
    link_count: int
    trace_count: int
    observation_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: int
    last_used_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptVersionAnalytics:
        return cls(
            version=int(payload.get("version", 0)),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            labels=[str(item) for item in payload.get("labels") or []],
            link_count=int(payload.get("link_count", 0)),
            trace_count=int(payload.get("trace_count", 0)),
            observation_count=int(payload.get("observation_count", 0)),
            total_input_tokens=int(payload.get("total_input_tokens", 0)),
            total_output_tokens=int(payload.get("total_output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            total_cost_usd=float(payload.get("total_cost_usd", 0.0)),
            total_latency_ms=int(payload.get("total_latency_ms", 0)),
            last_used_at=payload.get("last_used_at"),
        )


@dataclass(frozen=True)
class PromptUsageLinkRecord:
    id: str
    trace_id: str
    observation_id: str | None
    prompt_id: str
    prompt_version_id: str
    prompt_name: str
    prompt_type: PromptType
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
    linked_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptUsageLinkRecord:
        return cls(
            id=str(payload.get("id", "")),
            trace_id=str(payload.get("trace_id", "")),
            observation_id=payload.get("observation_id"),
            prompt_id=str(payload.get("prompt_id", "")),
            prompt_version_id=str(payload.get("prompt_version_id", "")),
            prompt_name=str(payload.get("prompt_name", "")),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            prompt_version=int(payload.get("prompt_version", 0)),
            prompt_label=payload.get("prompt_label"),
            variables=dict(payload.get("variables") or {}),
            trace_name=str(payload.get("trace_name", "")),
            trace_status=str(payload.get("trace_status", "")),
            observation_name=payload.get("observation_name"),
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id"),
            environment=payload.get("environment"),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            latency_ms=int(payload.get("latency_ms", 0)),
            linked_at=payload.get("linked_at"),
        )


@dataclass(frozen=True)
class PromptAnalytics:
    prompt_id: str
    prompt_name: str
    prompt_type: PromptType
    totals: PromptAnalyticsTotals
    versions: list[PromptVersionAnalytics]
    recent_links: list[PromptUsageLinkRecord]
    recent_links_pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptAnalytics:
        return cls(
            prompt_id=str(payload.get("prompt_id", "")),
            prompt_name=str(payload.get("prompt_name", "")),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            totals=PromptAnalyticsTotals.from_dict(payload.get("totals") or {}),
            versions=[
                PromptVersionAnalytics.from_dict(item)
                for item in payload.get("versions") or []
            ],
            recent_links=[
                PromptUsageLinkRecord.from_dict(item)
                for item in payload.get("recent_links") or []
            ],
            recent_links_pagination=PaginationInfo.from_dict(
                payload.get("recent_links_pagination") or {}
            ),
        )


@dataclass(frozen=True)
class PromptPlaygroundConfig:
    provider: str | None
    model: str | None
    temperature: float | None
    max_tokens: int | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptPlaygroundConfig:
        return cls(
            provider=payload.get("provider"),
            model=payload.get("model"),
            temperature=(
                float(payload["temperature"])
                if payload.get("temperature") is not None
                else None
            ),
            max_tokens=(
                int(payload["max_tokens"])
                if payload.get("max_tokens") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PromptPlaygroundTokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptPlaygroundTokenUsage:
        return cls(
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
        )


@dataclass(frozen=True)
class PromptPlaygroundResult:
    prompt_name: str
    prompt_type: PromptType
    resolved_version: int
    resolved_label: str | None
    variables: dict[str, Any]
    config: PromptPlaygroundConfig
    rendered_prompt_text: str | None
    rendered_chat_messages: list[ChatPromptMessage]
    executed: bool
    output: Any
    token_usage: PromptPlaygroundTokenUsage | None
    cost_usd: float
    latency_ms: int
    trace_id: str | None
    trace_status: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptPlaygroundResult:
        token_usage_payload = payload.get("token_usage")
        return cls(
            prompt_name=str(payload.get("prompt_name", "")),
            prompt_type=PromptType(payload.get("prompt_type", PromptType.TEXT.value)),
            resolved_version=int(payload.get("resolved_version", 0)),
            resolved_label=payload.get("resolved_label"),
            variables=dict(payload.get("variables") or {}),
            config=PromptPlaygroundConfig.from_dict(payload.get("config") or {}),
            rendered_prompt_text=payload.get("rendered_prompt_text"),
            rendered_chat_messages=[
                ChatPromptMessage.from_dict(message)
                for message in payload.get("rendered_chat_messages") or []
            ],
            executed=bool(payload.get("executed", False)),
            output=payload.get("output"),
            token_usage=(
                PromptPlaygroundTokenUsage.from_dict(token_usage_payload)
                if isinstance(token_usage_payload, dict)
                else None
            ),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            latency_ms=int(payload.get("latency_ms", 0)),
            trace_id=payload.get("trace_id"),
            trace_status=payload.get("trace_status"),
        )
