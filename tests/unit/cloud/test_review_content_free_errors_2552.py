"""Canary tests: backend text must not reach repr/str of trial results or admin errors."""

from __future__ import annotations

import io
import json
from unittest.mock import patch
from urllib import error

import pytest

from traigent.admin.client import (
    AuthenticationError,
    ClientError,
    EnterpriseAdminClient,
)
from traigent.admin.config import EnterpriseAdminConfig
from traigent.cloud.trial_operations import TrialSubmissionResult

CANARY = "customer text here"


def test_trial_submission_rejection_repr_and_str_hide_reason() -> None:
    result = TrialSubmissionResult.rejected(CANARY)

    assert CANARY not in repr(result)
    assert CANARY not in str(result)
    assert result.reason == CANARY
    assert result.permanent_rejection is True


def _http_error(code: int, body: str) -> error.HTTPError:
    return error.HTTPError(
        url="https://backend.invalid/x",
        code=code,
        msg="err",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _admin_client() -> EnterpriseAdminClient:
    return EnterpriseAdminClient(
        config=EnterpriseAdminConfig(backend_origin="https://backend.invalid")
    )


@pytest.mark.parametrize("code", [401, 403])
def test_admin_auth_error_carries_no_backend_text(code: int) -> None:
    client = _admin_client()
    body = json.dumps({"message": CANARY, "error_code": "unauthorized"})
    with (
        patch("traigent.admin.client.raise_if_backend_offline"),
        patch(
            "traigent.admin.client.request.urlopen",
            side_effect=_http_error(code, body),
        ),
        pytest.raises(AuthenticationError) as excinfo,
    ):
        client._request_json_sync("GET", "/api/v1/admin/x")
    assert CANARY not in str(excinfo.value)
    assert str(excinfo.value) == f"Authentication failed (HTTP {code} | unauthorized)"


def test_admin_server_error_carries_no_backend_text() -> None:
    client = _admin_client()
    body = json.dumps({"message": CANARY})
    with (
        patch("traigent.admin.client.raise_if_backend_offline"),
        patch(
            "traigent.admin.client.request.urlopen",
            side_effect=_http_error(500, body),
        ),
        pytest.raises(ClientError) as excinfo,
    ):
        client._request_json_sync("GET", "/api/v1/admin/x")
    assert CANARY not in str(excinfo.value)
    assert str(excinfo.value) == "Admin request failed (HTTP 500 | unknown)"
