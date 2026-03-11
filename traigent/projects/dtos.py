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
