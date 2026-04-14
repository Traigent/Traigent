from __future__ import annotations

from urllib import error

import pytest

from traigent import EnterpriseAdminClient
from traigent.admin import SSOProviderType, TenantMembershipRole, TenantMembershipStatus
from traigent.security.tenant import TenantStatus, TenantTier
from traigent.utils.exceptions import AuthenticationError, TraigentConnectionError


def test_enterprise_admin_client_manages_tenants_memberships_and_sso():
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        if method == "GET" and path.startswith("/api/v1beta/admin/tenants?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "tenant_1",
                            "name": "Acme Corp",
                            "slug": "acme",
                            "status": "active",
                            "tier": "professional",
                            "quotas": {"max_users": 25},
                            "metadata": {"region": "us"},
                            "membership_count": 1,
                            "has_sso_config": True,
                            "created_at": "2026-03-10T12:00:00+00:00",
                            "updated_at": "2026-03-10T12:00:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        if method == "GET" and path.startswith("/api/v1beta/admin/tenants/tenant_1/memberships"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "membership_1",
                            "tenant_id": "tenant_1",
                            "user_id": "user_1",
                            "role": "admin",
                            "status": "active",
                            "is_default": True,
                            "user": {"id": "user_1", "email": "user@example.com"},
                            "created_at": "2026-03-10T12:05:00+00:00",
                            "updated_at": "2026-03-10T12:05:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        if path.endswith("/sso") and method == "GET":
            return {
                "data": {
                    "id": "sso_1",
                    "tenant_id": "tenant_1",
                    "provider_type": "oidc",
                    "enabled": True,
                    "enforce_sso": True,
                    "settings": {"issuer": "https://issuer.example"},
                    "allowed_domains": ["acme.example"],
                    "created_at": "2026-03-10T12:10:00+00:00",
                    "updated_at": "2026-03-10T12:10:00+00:00",
                }
            }
        if method == "POST" and path == "/api/v1beta/admin/tenants":
            return {
                "data": {
                    "id": "tenant_1",
                    "name": "Acme Corp",
                    "slug": "acme",
                    "status": "active",
                    "tier": "professional",
                    "quotas": {"max_users": 25},
                    "metadata": {"region": "us"},
                    "membership_count": 0,
                    "has_sso_config": False,
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "updated_at": "2026-03-10T12:00:00+00:00",
                }
            }
        if method == "POST" and path.endswith("/memberships"):
            return {
                "data": {
                    "id": "membership_1",
                    "tenant_id": "tenant_1",
                    "user_id": "user_1",
                    "role": "admin",
                    "status": "active",
                    "is_default": True,
                    "user": {"id": "user_1"},
                    "created_at": "2026-03-10T12:05:00+00:00",
                    "updated_at": "2026-03-10T12:05:00+00:00",
                }
            }
        if method == "PUT" and path.endswith("/sso"):
            return {
                "data": {
                    "id": "sso_1",
                    "tenant_id": "tenant_1",
                    "provider_type": "saml",
                    "enabled": True,
                    "enforce_sso": False,
                    "settings": {"entry_point": "https://idp.example/sso"},
                    "allowed_domains": ["acme.example"],
                    "created_at": "2026-03-10T12:10:00+00:00",
                    "updated_at": "2026-03-10T12:11:00+00:00",
                }
            }
        return {
            "data": {
                "id": "tenant_1",
                "name": "Acme Corp",
                "slug": "acme",
                "status": "suspended",
                "tier": "enterprise",
                "quotas": {"max_users": 100},
                "metadata": {"region": "eu"},
                "membership_count": 1,
                "has_sso_config": True,
                "created_at": "2026-03-10T12:00:00+00:00",
                "updated_at": "2026-03-10T12:12:00+00:00",
            }
        }

    client = EnterpriseAdminClient(request_sender=request_sender)
    tenants = client.list_tenants(search="acme", status=TenantStatus.ACTIVE, tier=TenantTier.PROFESSIONAL)
    created_tenant = client.create_tenant(
        name="Acme Corp",
        slug="acme",
        status=TenantStatus.ACTIVE,
        tier=TenantTier.PROFESSIONAL,
        quotas={"max_users": 25},
        metadata={"region": "us"},
    )
    updated_tenant = client.update_tenant("tenant_1", status=TenantStatus.SUSPENDED, tier=TenantTier.ENTERPRISE)
    memberships = client.list_tenant_memberships(
        "tenant_1",
        status=TenantMembershipStatus.ACTIVE,
        role=TenantMembershipRole.ADMIN,
    )
    created_membership = client.create_tenant_membership(
        "tenant_1",
        user_id="user_1",
        role=TenantMembershipRole.ADMIN,
        is_default=True,
    )
    sso_config = client.get_tenant_sso_config("tenant_1")
    updated_sso = client.upsert_tenant_sso_config(
        "tenant_1",
        provider_type=SSOProviderType.SAML,
        settings={"entry_point": "https://idp.example/sso"},
        allowed_domains=["acme.example"],
    )

    assert tenants.items[0].tier == TenantTier.PROFESSIONAL
    assert created_tenant.slug == "acme"
    assert updated_tenant.status == TenantStatus.SUSPENDED
    assert memberships.items[0].role == TenantMembershipRole.ADMIN
    assert created_membership.is_default is True
    assert sso_config is not None
    assert sso_config.provider_type == SSOProviderType.OIDC
    assert updated_sso.provider_type == SSOProviderType.SAML

    assert calls[0] == (
        "GET",
        "/api/v1beta/admin/tenants?page=1&per_page=20&search=acme&status=active&tier=professional",
        None,
    )
    assert calls[1][1] == "/api/v1beta/admin/tenants"


def test_enterprise_admin_client_maps_auth_and_network_errors(monkeypatch):
    auth_client = EnterpriseAdminClient()
    monkeypatch.setattr(
        "traigent.admin.client.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            error.HTTPError(
                url="https://example.com",
                code=401,
                msg="unauthorized",
                hdrs=None,
                fp=None,
            )
        ),
    )
    with pytest.raises(AuthenticationError):
        auth_client.get_tenant("blocked")

    forbidden_client = EnterpriseAdminClient()
    monkeypatch.setattr(
        "traigent.admin.client.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            error.HTTPError(
                url="https://example.com",
                code=403,
                msg="forbidden",
                hdrs=None,
                fp=None,
            )
        ),
    )
    with pytest.raises(AuthenticationError):
        forbidden_client.get_tenant("forbidden")

    network_client = EnterpriseAdminClient()
    monkeypatch.setattr(
        "traigent.admin.client.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error.URLError("offline")),
    )
    with pytest.raises(TraigentConnectionError):
        network_client.get_tenant("offline")
