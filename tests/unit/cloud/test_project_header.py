"""Writes carry X-Project-Id when TRAIGENT_PROJECT_ID is configured."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.auth import AuthManager
from traigent.cloud.client import CloudServiceError

API_KEY = "tg_" + "a" * 61  # pragma: allowlist secret
BODY_422 = (
    '{"success":false,"error_code":"PROJECT_CONTEXT_REQUIRED",'
    '"message":"customer text must not appear"}'
)


@pytest.fixture
def no_project(monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)


@pytest.fixture
def with_project(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "proj_123")


@pytest.mark.parametrize("target", ["cloud", "backend", "both"])
def test_common_headers_carry_project_when_configured(with_project, target):
    headers: dict[str, str] = {}
    AuthManager(api_key=API_KEY)._add_common_headers(headers, target=target)
    assert headers["X-Project-Id"] == "proj_123"


def _auth_headers() -> dict[str, str]:
    manager = AuthManager(api_key=API_KEY)

    async def _noop() -> None:
        return None

    async def _mode_headers(mode, target):
        return {"X-API-Key": "k"}  # pragma: allowlist secret

    manager._ensure_authenticated_for_headers = _noop  # type: ignore[method-assign]
    manager._headers_for_mode = _mode_headers  # type: ignore[method-assign]
    manager._latch_emitted_credential = lambda headers: None  # type: ignore[method-assign]
    manager._credentials = Mock()
    return asyncio.run(manager.get_auth_headers())


def test_auth_headers_carry_project_for_write_calls(with_project):
    # session create, next-trial, results and finalize all build headers here.
    headers = _auth_headers()
    assert headers["X-Project-Id"] == "proj_123"


def test_no_project_header_when_unconfigured(no_project):
    headers = _auth_headers()
    assert "X-Project-Id" not in headers


def test_caller_set_header_not_overwritten(with_project):
    from traigent.config.project import apply_project_header

    headers = {"x-project-id": "caller_proj"}
    apply_project_header(headers)
    assert headers == {"x-project-id": "caller_proj"}


def test_common_headers_keep_preset_project(with_project):
    headers = {"X-Project-Id": "caller_proj"}
    AuthManager(api_key=API_KEY)._add_common_headers(headers, target="backend")
    assert headers["X-Project-Id"] == "caller_proj"


def test_client_get_headers_carry_project(with_project):
    from traigent.cloud.client import TraigentCloudClient

    client = TraigentCloudClient.__new__(TraigentCloudClient)
    client.auth = Mock()

    async def _h():
        return {"X-API-Key": "k"}  # pragma: allowlist secret

    client.auth.get_headers = _h
    headers = asyncio.run(client._get_headers())
    assert headers["X-Project-Id"] == "proj_123"


def test_422_project_context_required_maps_to_clear_error():
    ops = ApiOperations(Mock())
    with pytest.raises(CloudServiceError) as exc:
        ops._handle_session_error(422, BODY_422)
    assert "TRAIGENT_PROJECT_ID" in str(exc.value)
    assert "customer text" not in str(exc.value)


def test_other_422_keeps_generic_message():
    ops = ApiOperations(Mock())
    with pytest.raises(CloudServiceError) as exc:
        ops._handle_session_error(422, '{"error_code":"validation_error"}')
    assert "TRAIGENT_PROJECT_ID" not in str(exc.value)
