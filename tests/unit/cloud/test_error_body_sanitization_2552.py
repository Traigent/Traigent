"""#2552: backend error bodies never reach SDK exceptions or logs (canary)."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.cloud.dataset_converter import DatasetConverter
from traigent.cloud.optimizer_client import OptimizerDirectClient
from traigent.core.session_types import safe_http_error_summary

pytestmark = pytest.mark.backend_online

CANARY = "customer text here"
BODY = json.dumps({"error_code": "validation_error", "message": CANARY})
PLAIN = CANARY


class _Resp:
    def __init__(self, status: int, body: str = BODY) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self._body = body

    async def text(self) -> str:
        return self._body

    async def json(self):
        return json.loads(self._body)

    def raise_for_status(self) -> None:
        raise AssertionError("should not be reached")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _blob(caplog, exc: BaseException | None = None) -> str:
    parts = [r.getMessage() for r in caplog.records]
    if exc is not None:
        parts.append(str(exc))
    return " ".join(parts).lower()


class TestHelper:
    def test_status_and_known_code_only(self):
        assert safe_http_error_summary(400, BODY) == "HTTP 400 | validation_error"

    @pytest.mark.parametrize(
        "body",
        [PLAIN, None, "", json.dumps({"error_code": CANARY}), json.dumps([CANARY])],
    )
    def test_unknown_or_unparseable_is_unknown(self, body):
        out = safe_http_error_summary(500, body)
        assert out == "HTTP 500 | unknown"
        assert CANARY not in out

    def test_message_never_included(self):
        out = safe_http_error_summary(
            429, json.dumps({"error_code": "quota_exceeded", "message": CANARY})
        )
        assert out == "HTTP 429 | quota_exceeded"


@pytest.fixture
def client():
    c = TraigentCloudClient(
        api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
        base_url="http://localhost:8000",
        enable_fallback=False,
    )
    c._ensure_session = AsyncMock()
    c._get_headers = AsyncMock(return_value={})
    c._raise_if_backend_egress_disabled = Mock()
    return c


def _wire(client, status: int, body: str = BODY):
    sess = Mock()
    resp = _Resp(status, body)
    for verb in ("get", "post", "put", "delete"):
        setattr(sess, verb, Mock(return_value=resp))
    client._aio_session = sess
    return sess


def _agent_calls(client):
    return {
        "start_agent": lambda: client.start_agent_optimization(Mock()),
        "execute_agent": lambda: client.execute_agent(Mock(agent_spec="x")),
        "agent_status": lambda: client.get_agent_optimization_status("opt"),
        "cancel_agent": lambda: client.cancel_agent_optimization("opt"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 500])
@pytest.mark.parametrize(
    "name",
    [
        "next_trial",
        "submit_result",
        "finalize",
        "start_agent",
        "execute_agent",
        "agent_status",
        "cancel_agent",
        "list_sessions",
    ],
)
@pytest.mark.parametrize("body", [BODY, PLAIN])
async def test_client_paths_exclude_body(client, caplog, name, status, body):
    _wire(client, status, body)
    calls = {
        "next_trial": lambda: client.get_next_trial("s1"),
        "submit_result": lambda: client.submit_trial_result(
            "s1", "t1", {"a": 1.0}, 1.0
        ),
        "finalize": lambda: client.finalize_session("s1"),
        "list_sessions": lambda: client.list_sessions(),
        **_agent_calls(client),
    }
    ser = {
        "_serialize_agent_optimization_request": lambda *_: {},
        "_serialize_agent_execution_request": lambda *_: {},
    }
    with (
        patch.multiple(client, **ser),
        caplog.at_level(logging.DEBUG),
        pytest.raises(Exception) as info,  # noqa: B017 - wrapper types vary
    ):
        await calls[name]()
    assert CANARY not in _blob(caplog, info.value)
    assert str(status) in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [BODY, PLAIN])
async def test_ownership_403_excludes_body(client, caplog, body):
    _wire(client, 403, body)
    with caplog.at_level(logging.DEBUG), pytest.raises(CloudServiceError) as info:
        await client.get_next_trial("s1")
    assert CANARY not in _blob(caplog, info.value)


@pytest.mark.asyncio
async def test_delete_session_log_excludes_body(client, caplog):
    _wire(client, 500)
    with caplog.at_level(logging.DEBUG):
        assert await client.delete_session("s1") is False
    assert CANARY not in _blob(caplog)


@pytest.mark.asyncio
async def test_status_check_excludes_body(client, caplog):
    _wire(client, 500)
    with caplog.at_level(logging.DEBUG), pytest.raises(CloudServiceError) as info:
        await client._check_service_status_attempt("http://x/health", 1, 1)
    assert CANARY not in _blob(caplog, info.value)


class TestDatasetConverter:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c._create_backend_example_set(
                Mock(name="m", type="t", description="d")
            ),
            lambda c: c._upload_examples_to_backend(
                "12345678-1234-1234-1234-123456789abc", [{"a": 1}]
            ),
            lambda c: c._fetch_backend_example_set(
                "12345678-1234-1234-1234-123456789abc"
            ),
            lambda c: c._fetch_backend_examples("12345678-1234-1234-1234-123456789abc"),
        ],
    )
    async def test_example_set_paths(self, call, caplog):
        conv = DatasetConverter("https://api.example.com")
        sess = Mock()
        resp = _Resp(500)
        sess.post = sess.get = Mock(return_value=resp)
        conv._session = sess
        with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as info:
            await call(conv)
        assert CANARY not in _blob(caplog, info.value)
        assert "500" in str(info.value)


class TestOptimizerClient:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.get_next_configuration("s"),
            lambda c: c.get_session_status("s"),
            lambda c: c._submit_single("s", {"a": 1}),
            lambda c: c._submit_batch("s", [{"a": 1}]),
        ],
    )
    async def test_403_paths(self, call, caplog):
        oc = OptimizerDirectClient("https://api.example.com/opt", "tok")
        sess = Mock()
        sess.get = sess.post = Mock(return_value=_Resp(403))
        oc.session = sess
        with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as info:
            await call(oc)
        assert CANARY not in _blob(caplog, info.value)
        assert "403" in str(info.value)


class TestApiOperationsAndSync:
    @pytest.mark.asyncio
    async def test_experiment_run_status_log_excludes_body(self, caplog):
        from traigent.cloud.api_operations import ApiOperations

        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        ops = ApiOperations(mock_client)
        ops._raise_if_backend_egress_disabled = Mock()
        sess = Mock()
        sess.put = Mock(return_value=_Resp(500))
        holder = Mock()
        holder.__aenter__ = AsyncMock(return_value=sess)
        holder.__aexit__ = AsyncMock(return_value=False)
        with (
            patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp,
            caplog.at_level(logging.DEBUG),
        ):
            mock_aiohttp.ClientSession = Mock(return_value=holder)
            mock_aiohttp.ClientTimeout = Mock()
            await ops.update_experiment_run_status_on_completion("run_1", "completed")
        assert CANARY not in _blob(caplog)
        assert any("500" in r.getMessage() for r in caplog.records)

    def test_transient_backend_response_excludes_body(self):
        from traigent.cloud.backend_client import (
            _raise_for_transient_backend_response,
            _SyncBackendTransientError,
        )

        resp = Mock(status_code=503, text=BODY, headers={})
        with pytest.raises(_SyncBackendTransientError) as info:
            _raise_for_transient_backend_response(resp, "Agent-head fetch")
        assert CANARY not in str(info.value).lower()
        assert "503" in str(info.value)
