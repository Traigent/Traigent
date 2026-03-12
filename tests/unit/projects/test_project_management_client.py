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

    assert calls[0] == ("GET", "?page=1&per_page=20&search=alp", None)
    assert calls[1] == ("POST", "", {"name": "Alpha", "slug": "alpha", "description": None})
    assert calls[2] == ("GET", "/project_alpha", None)
    assert calls[3] == ("PATCH", "/project_alpha", {"description": "Primary project"})
    assert projects.items[0].slug == "alpha"
    assert created.id == "project_alpha"
    assert fetched.tenant_id == "tenant_acme"
    assert updated.description == "Primary project"


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
