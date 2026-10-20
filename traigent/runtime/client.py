"""Thin synchronous client for the production-requirement routes.

Auth is the SDK's usual ``X-API-Key``; redirects are refused (``requests``
would forward the key to the new host).  Every failure is mapped to a closed
code: response bodies are never included in a message.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from traigent.config.backend_config import BackendConfig
from traigent.runtime.model import (
    E_BACKEND,
    E_EGRESS,
    E_NO_CREDENTIALS,
    RuntimeRequirementError,
)

API_PREFIX = "/api/v1beta/projects"
_TIMEOUT = (5.0, 30.0)
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: Codes the backend is known to emit on these routes; anything else is ``backend_error``.
KNOWN_CODES = frozenset(
    {
        "production_requirements_disabled",
        "production_requirements_unavailable_in_shared_tenant",
        "authentication_required",
        "authorization_denied",
        "not_found",
        "invalid_requirement",
        "agent_not_found",
        "requirement_exists",
        "requirement_retired",
        "requirement_not_found",
        "requirement_weaker_than_floor",
        "closure_unreachable",
        "evaluation_schedule_mismatch",
        "tenant_not_registered",
        "revision_conflict",
        "tvl_import_too_large",
        "tvl_contract_unsupported_version",
        "tvl_contract_invalid",
        "tvl_contract_inconsistent",
        "tvl_environment_binding_invalid",
        "runtime_unsupported_capability",
        "response_action_unsupported",
    }
)


def _closed_code(status: int, body: Any) -> str:
    if isinstance(body, dict):
        code = body.get("error_code")
        if isinstance(code, str) and _CODE_RE.fullmatch(code) and code in KNOWN_CODES:
            return code
    if status == 401:
        return "authentication_required"
    if status == 403:
        return "authorization_denied"
    return E_BACKEND


class RuntimeClient:
    """Project-scoped calls used by ``apply``, ``watch`` and ``respond``."""

    def __init__(
        self,
        project_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.project_id = project_id
        self._origin = (base_url or BackendConfig.get_backend_url()).rstrip("/")
        self._api_key = api_key or BackendConfig.get_api_key()

    def _url(self, suffix: str) -> str:
        return f"{self._origin}{API_PREFIX}/{quote(self.project_id, safe='')}{suffix}"

    def request(
        self, method: str, suffix: str, body: dict[str, Any] | None = None
    ) -> tuple[int, Any]:
        """Return ``(status, parsed JSON or None)``; transport failures are closed errors."""
        from traigent.cloud.client import cloud_backend_egress_disabled

        if cloud_backend_egress_disabled():
            raise RuntimeRequirementError(E_EGRESS)
        if not self._api_key:
            raise RuntimeRequirementError(E_NO_CREDENTIALS)
        import requests

        from traigent.cloud.user_agent import get_sdk_user_agent
        from traigent.labels.transport_gate import preflight_requests

        url = self._url(suffix)
        headers = {
            "X-API-Key": self._api_key,
            "Accept": "application/json",
            "User-Agent": get_sdk_user_agent(),
        }
        kwargs: dict[str, Any] = {}
        if body is not None:
            kwargs["json"] = body
        try:
            preflight_requests(method, url, **kwargs)
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=_TIMEOUT,
                allow_redirects=False,
                **kwargs,
            )
        except RuntimeRequirementError:
            raise
        except Exception:  # noqa: BLE001 - never surface transport text
            raise RuntimeRequirementError(E_BACKEND, "transport") from None
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        return response.status_code, parsed

    def _expect(self, status: int, body: Any, ok: tuple[int, ...]) -> Any:
        if status in ok:
            if not isinstance(body, dict):
                raise RuntimeRequirementError(E_BACKEND, "response")
            return body
        raise RuntimeRequirementError(_closed_code(status, body))

    # -- routes -----------------------------------------------------------------
    def create_requirement(
        self, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any] | None]:
        """``POST /production-requirements``; 409 is returned, not raised."""
        status, parsed = self.request("POST", "/production-requirements", body)
        if status == 409 and _closed_code(status, parsed) == "requirement_exists":
            return status, None
        return status, self._expect(status, parsed, (200, 201))

    def import_tvl(self, body: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        """``POST /production-requirements/tvl-import``: 201 created, 200 identical, 409."""
        status, parsed = self.request(
            "POST", "/production-requirements/tvl-import", body
        )
        if status == 409 and _closed_code(status, parsed) == "requirement_exists":
            return status, None
        return status, self._expect(status, parsed, (200, 201))

    def list_requirements(self) -> list[dict[str, Any]]:
        status, parsed = self.request("GET", "/production-requirements")
        items = self._expect(status, parsed, (200,)).get("items")
        if not isinstance(items, list):
            raise RuntimeRequirementError(E_BACKEND, "response")
        return [i for i in items if isinstance(i, dict)]

    def list_breaches(self, agent_id: str) -> tuple[list[dict[str, Any]], bool | None]:
        """Breach rows plus ``has_more`` (``None`` when the backend does not report it)."""
        status, parsed = self.request(
            "GET", f"/agents/{quote(agent_id, safe='/')}/production-breaches"
        )
        body = self._expect(status, parsed, (200,))
        items = body.get("items")
        if not isinstance(items, list):
            raise RuntimeRequirementError(E_BACKEND, "response")
        has_more = body.get("has_more")
        return [i for i in items if isinstance(i, dict)], (
            has_more if isinstance(has_more, bool) else None
        )

    def get_next_step(self, agent_id: str) -> dict[str, Any]:
        status, parsed = self.request(
            "GET", f"/agent-readiness/{quote(agent_id, safe='/')}/next-step"
        )
        step: dict[str, Any] = self._expect(status, parsed, (200,))
        return step
