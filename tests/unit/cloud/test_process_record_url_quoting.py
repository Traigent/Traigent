"""Process-record routes keep the session id a single, quoted path segment."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient

BASE = "https://api.example.com"


def _client() -> BackendIntegratedClient:
    return BackendIntegratedClient(
        api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
        base_url=BASE,
        enable_fallback=False,
    )


def _call(method: str, sid: Any) -> str:
    client = _client()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"success": True}
    with (
        patch.object(client, "_raise_if_backend_egress_disabled"),
        patch.object(client, "_get_sync_auth_headers", return_value={}),
        patch("traigent.cloud.backend_client.preflight_requests"),
        patch("requests.request", return_value=resp) as req,
    ):
        if method == "POST":
            client.finalize_process_record_sync(sid, {})
        else:
            client.get_process_record_sync(sid)
    return str(req.call_args.args[1])


@pytest.mark.parametrize("sid", ["a/b", "../x", "a?b#c", "a b"])
@pytest.mark.parametrize("method", ["POST", "GET"])
def test_id_is_one_quoted_segment(method: str, sid: str) -> None:
    url = _call(method, sid)
    tail = url.split("/certification/process-records/", 1)[1]
    seg = tail.split("/finalize")[0] if method == "POST" else tail
    assert seg not in {"", ".", ".."}
    assert "/" not in seg and "?" not in url and "#" not in url


def test_valid_uuid_unchanged() -> None:
    sid = "123e4567-e89b-12d3-a456-426614174000"
    assert _call("POST", sid).endswith(f"/certification/process-records/{sid}/finalize")
    assert _call("GET", 42).endswith("/certification/process-records/42")
