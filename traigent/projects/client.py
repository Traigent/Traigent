"""Sync-friendly project management client."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request
from urllib.parse import quote, urlencode

from traigent.projects.config import ProjectManagementConfig
from traigent.projects.dtos import (
    ProjectDTO,
    ProjectListResponse,
    ProjectRateLimitPolicyDTO,
    ProjectRetentionPolicyDTO,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


class ProjectManagementClient:
    def __init__(
        self, config: ProjectManagementConfig | None = None, *, request_sender=None
    ) -> None:
        self.config = config or ProjectManagementConfig()
        self._request_sender_override = request_sender

    def list_projects(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        status: str | None = None,
    ) -> ProjectListResponse:
        query = urlencode(
            {
                k: str(v)
                for k, v in {
                    "page": page,
                    "per_page": per_page,
                    "search": search,
                    "status": status,
                }.items()
                if v is not None and v != ""
            }
        )
        path = f"?{query}" if query else ""
        payload = self._request_json("GET", path)
        return ProjectListResponse.from_dict(self._unwrap_data(payload, "project list"))

    def get_project(self, project_id: str) -> ProjectDTO:
        payload = self._request_json("GET", f"/{quote(project_id, safe='')}")
        return ProjectDTO.from_dict(self._unwrap_data(payload, "project detail"))

    def create_project(
        self,
        *,
        name: str,
        slug: str | None = None,
        description: str | None = None,
    ) -> ProjectDTO:
        payload = self._request_json(
            "POST",
            "",
            {"name": name, "slug": slug, "description": description},
        )
        return ProjectDTO.from_dict(self._unwrap_data(payload, "project create"))

    def update_project(self, project_id: str, **fields: Any) -> ProjectDTO:
        payload = self._request_json("PATCH", f"/{quote(project_id, safe='')}", fields)
        return ProjectDTO.from_dict(self._unwrap_data(payload, "project update"))

    def get_rate_limit_policy(self, project_id: str) -> ProjectRateLimitPolicyDTO:
        payload = self._request_json(
            "GET",
            f"/{quote(project_id, safe='')}/policies/rate-limits",
        )
        return ProjectRateLimitPolicyDTO.from_dict(
            self._unwrap_data(payload, "project rate limit policy")
        )

    def update_rate_limit_policy(
        self,
        project_id: str,
        **fields: Any,
    ) -> ProjectRateLimitPolicyDTO:
        payload = self._request_json(
            "PATCH",
            f"/{quote(project_id, safe='')}/policies/rate-limits",
            fields,
        )
        return ProjectRateLimitPolicyDTO.from_dict(
            self._unwrap_data(payload, "project rate limit policy update")
        )

    def get_retention_policy(self, project_id: str) -> ProjectRetentionPolicyDTO:
        payload = self._request_json(
            "GET",
            f"/{quote(project_id, safe='')}/policies/retention",
        )
        return ProjectRetentionPolicyDTO.from_dict(
            self._unwrap_data(payload, "project retention policy")
        )

    def update_retention_policy(
        self,
        project_id: str,
        **fields: Any,
    ) -> ProjectRetentionPolicyDTO:
        payload = self._request_json(
            "PATCH",
            f"/{quote(project_id, safe='')}/policies/retention",
            fields,
        )
        return ProjectRetentionPolicyDTO.from_dict(
            self._unwrap_data(payload, "project retention policy update")
        )

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._request_sender_override is not None:
            result = self._request_sender_override(method, path, payload)
            if not isinstance(result, dict):
                raise ClientError(
                    "Project request sender override must return a dictionary payload"
                )
            return result
        return self._request_json_sync(method, path, payload)

    def _request_json_sync(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        encoded_payload = (
            json.dumps(payload).encode("utf-8") if payload is not None else None
        )
        http_request = request.Request(
            f"{self.config.backend_origin}{self.config.api_path}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with request.urlopen(
                http_request, timeout=self.config.request_timeout
            ) as response:  # nosec B310
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise ClientError(
                        f"Project request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                    )
                return parsed
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8") if exc.fp else ""
            finally:
                exc.close()
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Project request rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Project request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to project backend at {self.config.backend_origin}"
            ) from exc

    @staticmethod
    def _unwrap_data(payload: dict[str, Any], label: str) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ClientError(f"Unexpected response structure for {label}")
        return data
