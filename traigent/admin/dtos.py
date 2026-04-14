"""DTOs for enterprise tenant, membership, and SSO admin flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from traigent.observability.dtos import PaginationInfo
from traigent.security.tenant import TenantStatus, TenantTier


class TenantMembershipRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class TenantMembershipStatus(StrEnum):
    ACTIVE = "active"
    INVITED = "invited"
    SUSPENDED = "suspended"


class SSOProviderType(StrEnum):
    OIDC = "oidc"
    SAML = "saml"


@dataclass(frozen=True)
class TenantDTO:
    id: str
    name: str
    slug: str
    status: TenantStatus
    tier: TenantTier
    quotas: dict[str, Any]
    metadata: dict[str, Any]
    membership_count: int
    has_sso_config: bool
    created_by: str | None
    updated_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TenantDTO:
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            slug=str(payload.get("slug", "")),
            status=TenantStatus(payload.get("status", TenantStatus.ACTIVE.value)),
            tier=TenantTier(payload.get("tier", TenantTier.FREE.value)),
            quotas=dict(payload.get("quotas") or {}),
            metadata=dict(payload.get("metadata") or {}),
            membership_count=int(payload.get("membership_count", 0)),
            has_sso_config=bool(payload.get("has_sso_config", False)),
            created_by=payload.get("created_by"),
            updated_by=payload.get("updated_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class TenantMembershipDTO:
    id: str
    tenant_id: str
    user_id: str
    role: TenantMembershipRole
    status: TenantMembershipStatus
    is_default: bool
    user: dict[str, Any] | None
    created_by: str | None
    updated_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TenantMembershipDTO:
        return cls(
            id=str(payload.get("id", "")),
            tenant_id=str(payload.get("tenant_id", "")),
            user_id=str(payload.get("user_id", "")),
            role=TenantMembershipRole(
                payload.get("role", TenantMembershipRole.MEMBER.value)
            ),
            status=TenantMembershipStatus(
                payload.get("status", TenantMembershipStatus.ACTIVE.value)
            ),
            is_default=bool(payload.get("is_default", False)),
            user=(
                dict(payload.get("user") or {})
                if payload.get("user") is not None
                else None
            ),
            created_by=payload.get("created_by"),
            updated_by=payload.get("updated_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class TenantSSOConfigDTO:
    id: str
    tenant_id: str
    provider_type: SSOProviderType
    enabled: bool
    enforce_sso: bool
    settings: dict[str, Any]
    allowed_domains: list[str]
    created_by: str | None
    updated_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TenantSSOConfigDTO:
        return cls(
            id=str(payload.get("id", "")),
            tenant_id=str(payload.get("tenant_id", "")),
            provider_type=SSOProviderType(
                payload.get("provider_type", SSOProviderType.OIDC.value)
            ),
            enabled=bool(payload.get("enabled", False)),
            enforce_sso=bool(payload.get("enforce_sso", False)),
            settings=dict(payload.get("settings") or {}),
            allowed_domains=[
                str(item) for item in payload.get("allowed_domains") or []
            ],
            created_by=payload.get("created_by"),
            updated_by=payload.get("updated_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class TenantListResponse:
    items: list[TenantDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TenantListResponse:
        return cls(
            items=[TenantDTO.from_dict(item) for item in payload.get("items") or []],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class TenantMembershipListResponse:
    items: list[TenantMembershipDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TenantMembershipListResponse:
        return cls(
            items=[
                TenantMembershipDTO.from_dict(item)
                for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )
