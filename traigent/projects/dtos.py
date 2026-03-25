"""DTOs for tenant-scoped project management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectDTO:
    id: str
    tenant_id: str
    name: str
    slug: str
    description: str | None
    is_default: bool
    status: str
    created_by: str | None
    updated_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectDTO:
        return cls(
            id=str(payload.get("id", "")),
            tenant_id=str(payload.get("tenant_id", "")),
            name=str(payload.get("name", "")),
            slug=str(payload.get("slug", "")),
            description=payload.get("description"),
            is_default=bool(payload.get("is_default", False)),
            status=str(payload.get("status", "active")),
            created_by=payload.get("created_by"),
            updated_by=payload.get("updated_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


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
            per_page=int(payload.get("per_page", 0)),
            total=int(payload.get("total", 0)),
            total_pages=int(payload.get("total_pages", 1)),
            has_next=bool(payload.get("has_next", False)),
            has_prev=bool(payload.get("has_prev", False)),
        )


@dataclass(frozen=True)
class ProjectListResponse:
    items: list[ProjectDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectListResponse:
        return cls(
            items=[
                ProjectDTO.from_dict(item or {}) for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class ProjectRateLimitPolicySettingsDTO:
    enabled: bool
    api_calls_per_minute: int | None
    evaluator_runs_per_hour: int | None
    export_jobs_per_day: int | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectRateLimitPolicySettingsDTO:
        return cls(
            enabled=bool(payload.get("enabled", True)),
            api_calls_per_minute=(
                int(payload["api_calls_per_minute"])
                if payload.get("api_calls_per_minute") is not None
                else None
            ),
            evaluator_runs_per_hour=(
                int(payload["evaluator_runs_per_hour"])
                if payload.get("evaluator_runs_per_hour") is not None
                else None
            ),
            export_jobs_per_day=(
                int(payload["export_jobs_per_day"])
                if payload.get("export_jobs_per_day") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ProjectRateLimitPolicyDTO:
    tenant_id: str
    project_id: str
    updated_at: str | None
    updated_by: str | None
    policy: ProjectRateLimitPolicySettingsDTO

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectRateLimitPolicyDTO:
        return cls(
            tenant_id=str(payload.get("tenant_id", "")),
            project_id=str(payload.get("project_id", "")),
            updated_at=payload.get("updated_at"),
            updated_by=payload.get("updated_by"),
            policy=ProjectRateLimitPolicySettingsDTO.from_dict(
                payload.get("policy") or {}
            ),
        )


@dataclass(frozen=True)
class ProjectRetentionPolicySettingsDTO:
    export_artifact_retention_days: int
    materialized_export_retention_days: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectRetentionPolicySettingsDTO:
        return cls(
            export_artifact_retention_days=int(
                payload["export_artifact_retention_days"]
            ),
            materialized_export_retention_days=int(
                payload["materialized_export_retention_days"]
            ),
        )


@dataclass(frozen=True)
class ProjectRetentionPolicyDTO:
    tenant_id: str
    project_id: str
    updated_at: str | None
    updated_by: str | None
    policy: ProjectRetentionPolicySettingsDTO

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectRetentionPolicyDTO:
        return cls(
            tenant_id=str(payload.get("tenant_id", "")),
            project_id=str(payload.get("project_id", "")),
            updated_at=payload.get("updated_at"),
            updated_by=payload.get("updated_by"),
            policy=ProjectRetentionPolicySettingsDTO.from_dict(
                payload.get("policy") or {}
            ),
        )


@dataclass(frozen=True)
class ProjectExportPolicySettingsDTO:
    allow_materialized_exports: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectExportPolicySettingsDTO:
        return cls(
            allow_materialized_exports=bool(
                payload.get("allow_materialized_exports", False)
            ),
        )


@dataclass(frozen=True)
class ProjectExportPolicyDTO:
    tenant_id: str
    project_id: str
    updated_at: str | None
    updated_by: str | None
    policy: ProjectExportPolicySettingsDTO

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectExportPolicyDTO:
        return cls(
            tenant_id=str(payload.get("tenant_id", "")),
            project_id=str(payload.get("project_id", "")),
            updated_at=payload.get("updated_at"),
            updated_by=payload.get("updated_by"),
            policy=ProjectExportPolicySettingsDTO.from_dict(
                payload.get("policy") or {}
            ),
        )
