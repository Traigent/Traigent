from __future__ import annotations

from urllib import error

import pytest

from traigent.projects import ProjectManagementClient
from traigent.utils.exceptions import AuthenticationError, ClientError


def test_project_management_client_crud_and_list() -> None:
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None = None):
        calls.append((method, path, payload))
        if method == "GET" and path.startswith("?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "project_alpha",
                            "tenant_id": "tenant_acme",
                            "name": "Alpha",
                            "slug": "alpha",
                            "description": None,
                            "is_default": True,
                            "status": "active",
                            "created_by": "user_1",
                            "updated_by": "user_1",
                            "created_at": "2026-03-11T00:00:00Z",
                            "updated_at": "2026-03-11T00:00:00Z",
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
        if method == "GET" and path.endswith("/policies/rate-limits"):
            return {
                "data": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "updated_at": "2026-03-11T00:00:00Z",
                    "updated_by": "user_1",
                    "policy": {
                        "enabled": True,
                        "api_calls_per_minute": 180,
                        "evaluator_runs_per_hour": 24,
                        "export_jobs_per_day": 8,
                    },
                }
            }
        if method == "PATCH" and path.endswith("/policies/rate-limits"):
            return {
                "data": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "updated_at": "2026-03-12T00:00:00Z",
                    "updated_by": "user_1",
                    "policy": {
                        "enabled": True,
                        "api_calls_per_minute": 240,
                        "evaluator_runs_per_hour": 24,
                        "export_jobs_per_day": 8,
                    },
                }
            }
        if method == "GET" and path.endswith("/policies/retention"):
            return {
                "data": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "updated_at": "2026-03-11T00:00:00Z",
                    "updated_by": "user_1",
                    "policy": {
                        "export_artifact_retention_days": 30,
                        "materialized_export_retention_days": 7,
                    },
                }
            }
        if method == "PATCH" and path.endswith("/policies/retention"):
            return {
                "data": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "updated_at": "2026-03-12T00:00:00Z",
                    "updated_by": "user_1",
                    "policy": {
                        "export_artifact_retention_days": 45,
                        "materialized_export_retention_days": 14,
                    },
                }
            }
        return {
            "data": {
                "id": "project_alpha",
                "tenant_id": "tenant_acme",
                "name": "Alpha",
                "slug": "alpha",
                "description": "Primary project",
                "is_default": True,
                "status": "active",
                "created_by": "user_1",
                "updated_by": "user_1",
                "created_at": "2026-03-11T00:00:00Z",
                "updated_at": "2026-03-11T00:00:00Z",
            }
        }

    client = ProjectManagementClient(request_sender=request_sender)

    projects = client.list_projects(search="alp")
    created = client.create_project(name="Alpha", slug="alpha")
    fetched = client.get_project("project_alpha")
    updated = client.update_project("project_alpha", description="Primary project")
    policy = client.get_rate_limit_policy("project_alpha")
    updated_policy = client.update_rate_limit_policy(
        "project_alpha", api_calls_per_minute=240
    )
    retention_policy = client.get_retention_policy("project_alpha")
    updated_retention_policy = client.update_retention_policy(
        "project_alpha",
        export_artifact_retention_days=45,
        materialized_export_retention_days=14,
    )

    assert calls[0] == ("GET", "?page=1&per_page=20&search=alp", None)
    assert calls[1] == ("POST", "", {"name": "Alpha", "slug": "alpha", "description": None})
    assert calls[2] == ("GET", "/project_alpha", None)
    assert calls[3] == ("PATCH", "/project_alpha", {"description": "Primary project"})
    assert calls[4] == ("GET", "/project_alpha/policies/rate-limits", None)
    assert calls[5] == (
        "PATCH",
        "/project_alpha/policies/rate-limits",
        {"api_calls_per_minute": 240},
    )
    assert calls[6] == ("GET", "/project_alpha/policies/retention", None)
    assert calls[7] == (
        "PATCH",
        "/project_alpha/policies/retention",
        {
            "export_artifact_retention_days": 45,
            "materialized_export_retention_days": 14,
        },
    )
    assert projects.items[0].slug == "alpha"
    assert created.id == "project_alpha"
    assert fetched.tenant_id == "tenant_acme"
    assert updated.description == "Primary project"
    assert policy.policy.api_calls_per_minute == 180
    assert updated_policy.policy.api_calls_per_minute == 240
    assert retention_policy.policy.export_artifact_retention_days == 30
    assert updated_retention_policy.policy.materialized_export_retention_days == 14


def test_project_management_client_validates_override_response_shape() -> None:
    client = ProjectManagementClient(request_sender=lambda *_args, **_kwargs: None)

    with pytest.raises(ClientError, match="override must return a dictionary payload"):
        client.list_projects()


def test_project_management_client_rejects_missing_data_payload() -> None:
    client = ProjectManagementClient(request_sender=lambda *_args, **_kwargs: {"data": []})

    with pytest.raises(ClientError, match="Unexpected response structure for project list"):
        client.list_projects()


def test_project_management_client_maps_http_errors(monkeypatch) -> None:
    client = ProjectManagementClient()

    def raise_unauthorized(*_args, **_kwargs):
        raise error.HTTPError(
            url="https://backend.example/api/v1beta/projects",
            code=401,
            msg="unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_unauthorized)

    with pytest.raises(AuthenticationError, match="status 401"):
        client.list_projects()


def test_project_management_client_maps_generic_http_failures(monkeypatch) -> None:
    client = ProjectManagementClient()

    def raise_conflict(*_args, **_kwargs):
        raise error.HTTPError(
            url="https://backend.example/api/v1beta/projects",
            code=409,
            msg="conflict",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_conflict)

    with pytest.raises(ClientError, match="status 409"):
        client.create_project(name="Alpha")
