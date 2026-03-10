"""Sync-friendly enterprise admin client."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib import error, request
from urllib.parse import urlencode

from traigent.admin.config import EnterpriseAdminConfig
from traigent.admin.dtos import (
    SSOProviderType,
    TenantDTO,
    TenantListResponse,
    TenantMembershipDTO,
    TenantMembershipListResponse,
    TenantMembershipRole,
    TenantMembershipStatus,
    TenantSSOConfigDTO,
)
from traigent.security.tenant import TenantStatus, TenantTier
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


class EnterpriseAdminClient:
    """Client for tenant, membership, and SSO admin APIs."""

    def __init__(
        self,
        config: EnterpriseAdminConfig | None = None,
        *,
        request_sender=None,
    ) -> None:
        self.config = config or EnterpriseAdminConfig()
        self._request_sender_override = request_sender

    def list_tenants(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        status: TenantStatus | str | None = None,
        tier: TenantTier | str | None = None,
    ) -> TenantListResponse:
        if isinstance(status, str):
            status = TenantStatus(status)
        if isinstance(tier, str):
            tier = TenantTier(tier)
        path = self._build_query_path(
            "/tenants",
            page=page,
            per_page=per_page,
            search=search,
            status=status.value if status else None,
            tier=tier.value if tier else None,
        )
        payload = self._request_json("GET", path)
        return TenantListResponse.from_dict(self._unwrap_data(payload, "tenant list"))

    def get_tenant(self, tenant_id: str) -> TenantDTO:
        payload = self._request_json("GET", f"/tenants/{tenant_id}")
        return TenantDTO.from_dict(self._unwrap_data(payload, "tenant detail"))

    def create_tenant(
        self,
        *,
        name: str,
        slug: str,
        status: TenantStatus | str = TenantStatus.ACTIVE,
        tier: TenantTier | str = TenantTier.FREE,
        quotas: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TenantDTO:
        if isinstance(status, str):
            status = TenantStatus(status)
        if isinstance(tier, str):
            tier = TenantTier(tier)
        payload = self._request_json(
            "POST",
            "/tenants",
            {
                "name": name,
                "slug": slug,
                "status": status.value,
                "tier": tier.value,
                "quotas": dict(quotas or {}),
                "metadata": dict(metadata or {}),
            },
        )
        return TenantDTO.from_dict(self._unwrap_data(payload, "tenant create"))

    def update_tenant(self, tenant_id: str, **fields: Any) -> TenantDTO:
        payload = dict(fields)
        if "status" in payload and isinstance(payload["status"], TenantStatus):
            payload["status"] = payload["status"].value
        if "tier" in payload and isinstance(payload["tier"], TenantTier):
            payload["tier"] = payload["tier"].value
        response = self._request_json("PATCH", f"/tenants/{tenant_id}", payload)
        return TenantDTO.from_dict(self._unwrap_data(response, "tenant update"))

    def list_tenant_memberships(
        self,
        tenant_id: str,
        *,
        page: int = 1,
        per_page: int = 20,
        status: TenantMembershipStatus | str | None = None,
        role: TenantMembershipRole | str | None = None,
    ) -> TenantMembershipListResponse:
        if isinstance(status, str):
            status = TenantMembershipStatus(status)
        if isinstance(role, str):
            role = TenantMembershipRole(role)
        path = self._build_query_path(
            f"/tenants/{tenant_id}/memberships",
            page=page,
            per_page=per_page,
            status=status.value if status else None,
            role=role.value if role else None,
        )
        payload = self._request_json("GET", path)
        return TenantMembershipListResponse.from_dict(
            self._unwrap_data(payload, "tenant membership list")
        )

    def create_tenant_membership(
        self,
        tenant_id: str,
        *,
        user_id: str,
        role: TenantMembershipRole | str = TenantMembershipRole.MEMBER,
        status: TenantMembershipStatus | str = TenantMembershipStatus.ACTIVE,
        is_default: bool = False,
    ) -> TenantMembershipDTO:
        if isinstance(role, str):
            role = TenantMembershipRole(role)
        if isinstance(status, str):
            status = TenantMembershipStatus(status)
        payload = self._request_json(
            "POST",
            f"/tenants/{tenant_id}/memberships",
            {
                "user_id": user_id,
                "role": role.value,
                "status": status.value,
                "is_default": is_default,
            },
        )
        return TenantMembershipDTO.from_dict(
            self._unwrap_data(payload, "tenant membership create")
        )

    def update_tenant_membership(
        self,
        tenant_id: str,
        membership_id: str,
        **fields: Any,
    ) -> TenantMembershipDTO:
        payload = dict(fields)
        if "role" in payload and isinstance(payload["role"], TenantMembershipRole):
            payload["role"] = payload["role"].value
        if "status" in payload and isinstance(
            payload["status"], TenantMembershipStatus
        ):
            payload["status"] = payload["status"].value
        response = self._request_json(
            "PATCH",
            f"/tenants/{tenant_id}/memberships/{membership_id}",
            payload,
        )
        return TenantMembershipDTO.from_dict(
            self._unwrap_data(response, "tenant membership update")
        )

    def get_tenant_sso_config(self, tenant_id: str) -> TenantSSOConfigDTO | None:
        payload = self._request_json("GET", f"/tenants/{tenant_id}/sso")
        data = self._unwrap_data(payload, "tenant SSO config")
        if data is None:
            return None
        return TenantSSOConfigDTO.from_dict(data)

    def upsert_tenant_sso_config(
        self,
        tenant_id: str,
        *,
        provider_type: SSOProviderType | str,
        enabled: bool = True,
        enforce_sso: bool = False,
        settings: dict[str, Any] | None = None,
        allowed_domains: list[str] | None = None,
    ) -> TenantSSOConfigDTO:
        if isinstance(provider_type, str):
            provider_type = SSOProviderType(provider_type)
        response = self._request_json(
            "PUT",
            f"/tenants/{tenant_id}/sso",
            {
                "provider_type": provider_type.value,
                "enabled": enabled,
                "enforce_sso": enforce_sso,
                "settings": dict(settings or {}),
                "allowed_domains": list(allowed_domains or []),
            },
        )
        return TenantSSOConfigDTO.from_dict(
            self._unwrap_data(response, "tenant SSO upsert")
        )

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        full_path = f"{self.config.api_path}{path}"
        if self._request_sender_override is not None:
            return cast(
                dict[str, Any],
                self._request_sender_override(method, full_path, payload),
            )
        return self._request_json_sync(method, full_path, payload)

    def _request_json_sync(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.config.backend_origin}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with request.urlopen(  # nosec B310 - backend_origin is caller-configured API endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code in {401, 403}:
                    raise AuthenticationError("Authentication failed")
                if status_code >= 400:
                    raise ClientError(
                        parsed.get("message") or "Enterprise admin request failed"
                    )
                return parsed
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8") if exc.fp else ""
                parsed = json.loads(body) if body else {}
            finally:
                exc.close()
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    parsed.get("message") or "Authentication failed"
                ) from exc
            raise ClientError(
                parsed.get("message")
                or f"Enterprise admin request failed with status {exc.code}"
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(str(exc.reason)) from exc

    def _unwrap_data(self, payload: dict[str, Any], label: str) -> Any:
        if "data" not in payload:
            raise ClientError(f"Unexpected response structure for {label}")
        return payload["data"]

    def _build_query_path(self, path: str, **query_params: Any) -> str:
        filtered = {
            key: value for key, value in query_params.items() if value is not None
        }
        if not filtered:
            return path
        return f"{path}?{urlencode(filtered)}"
