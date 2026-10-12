"""#2552 review round 7: remaining backend-text leak sites (canary)."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, Mock
from urllib import error as urlerror

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.benchmark_client import BenchmarkClient, BenchmarkClientConfig
from traigent.cloud.client import CloudServiceError
from traigent.utils.exceptions import AuthenticationError
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.trial_operations import TrialOperations, TrialSubmissionResult
from traigent.core.backend_session_manager import _format_untracked_warning_block
from traigent.core.session_types import (
    SessionCreationFailureDetail,
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.utils.exceptions import ClientError

pytestmark = pytest.mark.backend_online

CANARY = "customer text here"
BODY = json.dumps({"error_code": "validation_error", "message": CANARY})


def _blob(caplog, *objs) -> str:
    parts = [r.getMessage() for r in caplog.records] + [str(o) for o in objs]
    return " ".join(parts).lower()


class _Resp:
    def __init__(self, status: int, body: str = BODY) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def json(self):
        return json.loads(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _client(status: int):
    sess = Mock()
    sess.get = sess.post = Mock(return_value=_Resp(status))
    client = Mock()
    client._ensure_session = AsyncMock(return_value=sess)
    client._reset_http_session = Mock()
    client.backend_config.api_base_url = "https://api.example.com"
    client.auth_manager.auth.get_headers = AsyncMock(return_value={})
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["get_hybrid_session_status", "finalize_hybrid_session"]
)
async def test_hybrid_status_and_finalize_exclude_body(method, caplog):
    ops = SessionOperations(_client(500))
    ops._raise_if_backend_egress_disabled = Mock()
    with caplog.at_level(logging.DEBUG), pytest.raises(CloudServiceError) as info:
        await getattr(ops, method)("s1")
    assert CANARY not in _blob(caplog, info.value)
    assert "500" in str(info.value)


@pytest.mark.parametrize("body", [BODY, CANARY, json.dumps({"error": CANARY})])
def test_trial_rejection_reason_never_falls_back_to_body(body, caplog):
    ops = TrialOperations(Mock())
    with caplog.at_level(logging.DEBUG):
        result = ops._handle_trial_error_response(422, "t1", "s1", "http://x", body)
    assert isinstance(result, TrialSubmissionResult)
    assert result.reason is None
    assert CANARY not in _blob(caplog)


def test_trial_rejection_keeps_details_reason_for_caller(caplog):
    body = json.dumps({"message": CANARY, "details": {"reason": "domain mismatch"}})
    ops = TrialOperations(Mock())
    with caplog.at_level(logging.DEBUG):
        result = ops._handle_trial_error_response(422, "t1", "s1", "http://x", body)
    assert result.reason == "domain mismatch"
    assert CANARY not in _blob(caplog)


@pytest.mark.parametrize("message", [CANARY, None])
def test_stale_head_error_uses_fixed_message(message):
    payload = {
        "code": "STALE_HEAD_GENERATION",
        "current_generation": 5,
        "expected_generation": 3,
        "message": message,
    }
    response = Mock()
    response.json = Mock(return_value=payload)
    exc = BackendIntegratedClient._stale_head_error_from_response(
        response, expected_head_generation=3, agent_id="a", environment=None
    )
    assert exc is not None
    assert CANARY not in str(exc).lower()
    assert exc.current_generation == 5 and exc.expected_generation == 3
    assert "5" in str(exc)


def _failure(missing):
    detail = SessionCreationFailureDetail.from_http_response(
        403,
        json.dumps(
            {"error_code": "forbidden", "details": {"missing_permissions": missing}}
        ),
    )
    return SessionCreationResult(
        session_id="local",
        backend_connected=False,
        failure_reason=SessionCreationFailureReason.AUTH,
        failure_detail="HTTP 403 | forbidden",
        failure_response=detail,
    ), detail


def test_missing_permissions_allowlist_omits_free_text():
    from traigent.core.backend_session_manager import _classify_session_creation_failure

    result, detail = _failure(["sessions:create", CANARY])
    block = _format_untracked_warning_block(
        result,
        _classify_session_creation_failure(
            result.failure_reason, detail, failure_detail=result.failure_detail
        ),
        aborting=False,
    )
    assert "`sessions:create`" in block
    assert CANARY not in block.lower()
    assert "1 entry omitted" in block


def test_detail_and_result_repr_str_exclude_text():
    result, detail = _failure([CANARY])
    detail2 = SessionCreationFailureDetail.from_http_response(
        400,
        json.dumps({"error_code": CANARY, "message": CANARY}),
    )
    for obj in (detail, detail2, result):
        assert CANARY not in repr(obj).lower()
        assert CANARY not in f"{obj!r}".lower()
    assert CANARY not in str(detail).lower()
    assert CANARY not in str(detail2).lower()
    assert (
        "customer"
        not in repr(
            SessionCreationFailureDetail(
                status_code=400, error_code=CANARY, message=CANARY, raw_body=CANARY
            )
        ).lower()
    )


@pytest.mark.parametrize(
    "code,exc_type", [(401, AuthenticationError), (500, ClientError)]
)
@pytest.mark.parametrize("body", [BODY, CANARY])
def test_benchmark_client_errors_exclude_body(code, exc_type, body, monkeypatch):
    import io

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    cfg = BenchmarkClientConfig(
        backend_origin="https://api.example.test",
        api_key="k",
        tenant_id=None,
        project_id=None,
    )
    client = BenchmarkClient(cfg)
    http_err = urlerror.HTTPError(
        "https://api.example.test", code, "x", {}, io.BytesIO(body.encode())
    )
    monkeypatch.setattr(
        "traigent.cloud.benchmark_client.request.urlopen",
        Mock(side_effect=http_err),
    )
    with pytest.raises(exc_type) as info:
        client._post("/api/v1/datasets/generate", {})
    assert CANARY not in str(info.value).lower()
    assert str(code) in str(info.value)
