"""DTOs for Agent Arena SDK responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArenaExecutionRef:
    """Explicit Arena linkage for typed session creation."""

    provider_source_id: str
    run_id: str | None = None
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider_source_id": self.provider_source_id,
            "run_id": self.run_id,
        }


@dataclass
class ArenaProviderSource:
    """Configured provider source available to Arena."""

    id: str
    label: str
    provider: str
    kind: str
    status: str
    allowed_models: list[str] = field(default_factory=list)
    attribution_payload: dict[str, Any] = field(default_factory=dict)
    has_current_actor_consent: bool = False
    current_consent: dict[str, Any] | None = None
    disclosure: dict[str, Any] | None = None
    optimization_token_budget: int | None = None
    optimization_tokens_used: int = 0
    optimization_tokens_remaining: int | None = None
    is_session_scoped: bool = False
    session_scope_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArenaProviderSource:
        return cls(
            id=payload["id"],
            label=payload.get("label", payload["id"]),
            provider=payload.get("provider", ""),
            kind=payload.get("kind", ""),
            status=payload.get("status", ""),
            allowed_models=list(payload.get("allowed_models") or []),
            attribution_payload=dict(payload.get("attribution_payload") or {}),
            has_current_actor_consent=bool(payload.get("has_current_actor_consent")),
            current_consent=payload.get("current_consent"),
            disclosure=payload.get("disclosure"),
            optimization_token_budget=payload.get("optimization_token_budget"),
            optimization_tokens_used=int(payload.get("optimization_tokens_used") or 0),
            optimization_tokens_remaining=payload.get("optimization_tokens_remaining"),
            is_session_scoped=bool(payload.get("is_session_scoped")),
            session_scope_id=payload.get("session_scope_id"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass
class ArenaRunProviderSession:
    """Child provider session attached to an Arena run."""

    provider_source_id: str
    provider: str
    label: str
    kind: str
    status: str
    session_id: str | None = None
    experiment_id: str | None = None
    experiment_run_id: str | None = None
    allowed_models: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArenaRunProviderSession:
        return cls(
            provider_source_id=payload["provider_source_id"],
            provider=payload.get("provider", ""),
            label=payload.get("label", payload["provider_source_id"]),
            kind=payload.get("kind", ""),
            status=payload.get("status", ""),
            session_id=payload.get("session_id"),
            experiment_id=payload.get("experiment_id"),
            experiment_run_id=payload.get("experiment_run_id"),
            allowed_models=list(payload.get("allowed_models") or []),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
        )


@dataclass
class ArenaLeaderboardRow:
    """Leaderboard row for one provider in an Arena run."""

    provider_source_id: str
    provider: str
    label: str
    kind: str
    status: str
    success_count: int = 0
    failure_count: int = 0
    usage_count: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    average_latency_ms: float | None = None
    average_accuracy: float | None = None
    weighted_score: float | None = None
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArenaLeaderboardRow:
        return cls(
            provider_source_id=payload["provider_source_id"],
            provider=payload.get("provider", ""),
            label=payload.get("label", payload["provider_source_id"]),
            kind=payload.get("kind", ""),
            status=payload.get("status", ""),
            success_count=int(payload.get("success_count") or 0),
            failure_count=int(payload.get("failure_count") or 0),
            usage_count=int(payload.get("usage_count") or 0),
            total_cost_usd=float(payload.get("total_cost_usd") or 0.0),
            total_tokens=int(payload.get("total_tokens") or 0),
            average_latency_ms=payload.get("average_latency_ms"),
            average_accuracy=payload.get("average_accuracy"),
            weighted_score=payload.get("weighted_score"),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
        )


@dataclass
class ArenaLeaderboard:
    """Leaderboard payload for an Arena run."""

    run_id: str
    status: str
    ranking_weights: dict[str, float] = field(default_factory=dict)
    rows: list[ArenaLeaderboardRow] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArenaLeaderboard:
        return cls(
            run_id=payload["run_id"],
            status=payload.get("status", ""),
            ranking_weights=dict(payload.get("ranking_weights") or {}),
            rows=[
                ArenaLeaderboardRow.from_dict(item)
                for item in (payload.get("rows") or [])
                if isinstance(item, dict)
            ],
        )


@dataclass
class ArenaRun:
    """Arena run aggregate."""

    id: str
    function_name: str
    status: str
    name: str | None = None
    provider_source_ids: list[str] = field(default_factory=list)
    provider_sessions: list[ArenaRunProviderSession] = field(default_factory=list)
    ranking_weights: dict[str, float] = field(default_factory=dict)
    leaderboard: ArenaLeaderboard | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArenaRun:
        leaderboard_payload = payload.get("leaderboard")
        return cls(
            id=payload["id"],
            name=payload.get("name"),
            function_name=payload.get("function_name", ""),
            status=payload.get("status", ""),
            provider_source_ids=list(payload.get("provider_source_ids") or []),
            provider_sessions=[
                ArenaRunProviderSession.from_dict(item)
                for item in (payload.get("provider_sessions") or [])
                if isinstance(item, dict)
            ],
            ranking_weights=dict(payload.get("ranking_weights") or {}),
            leaderboard=(
                ArenaLeaderboard.from_dict(leaderboard_payload)
                if isinstance(leaderboard_payload, dict)
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            completed_at=payload.get("completed_at"),
        )


@dataclass
class ArenaInvokeResult:
    """Result of brokered Arena invoke."""

    usage_id: str
    run_id: str
    provider_source_id: str
    provider: str
    model: str
    content: Any
    token_usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: int | None = None
    source_kind: str | None = None
    sponsored: bool = False
    replayed: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArenaInvokeResult:
        return cls(
            usage_id=payload["usage_id"],
            run_id=payload.get("run_id", ""),
            provider_source_id=payload.get("provider_source_id", ""),
            provider=payload.get("provider", ""),
            model=payload.get("model", ""),
            content=payload.get("content"),
            token_usage=dict(payload.get("token_usage") or {}),
            cost_usd=float(payload.get("cost_usd") or 0.0),
            latency_ms=payload.get("latency_ms"),
            source_kind=payload.get("source_kind"),
            sponsored=bool(payload.get("sponsored")),
            replayed=bool(payload.get("replayed")),
        )
