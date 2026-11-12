"""Real-call-path coverage for Retry-After handling (#3011, #3010).

Each test drives the production request/finalize path with a faked HTTP layer
and a patched sleep, so nothing really waits.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud import resilient_client as rc
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError
from traigent.utils.exceptions import RateLimitError, RetryableError
from traigent.utils.retry_after import MAX_RETRY_AFTER_SECONDS

# Opt into the connected code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


class _Resp:
    def __init__(self, status: int, headers: dict[str, str]) -> None:
        self.status = status
        self.headers = headers

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *_a) -> bool:
        return False

    async def json(self) -> dict:
        return {"ok": True}

    async def text(self) -> str:
        return ""


class _Session:
    def __init__(self, replies: list[_Resp], counter: list[int]) -> None:
        self._replies = replies
        self._counter = counter

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_a) -> bool:
        return False

    def request(self, *_a, **_k) -> _Resp:
        index = min(self._counter[0], len(self._replies) - 1)
        self._counter[0] += 1
        return self._replies[index]


def _drive(replies: list[_Resp], *, max_delay: float = 60.0):
    counter = [0]
    sleeps: list[float] = []
    client = rc.ResilientClient(
        max_retries=2, base_delay=0.1, max_delay=max_delay, jitter_factor=0
    )

    async def _sleep(delay, *_a, **_k) -> None:
        sleeps.append(delay)

    with (
        mock.patch(
            "traigent.cloud.client.raise_if_cloud_egress_disabled",
            lambda *a, **k: None,
        ),
        mock.patch("aiohttp.ClientSession", lambda *a, **k: _Session(replies, counter)),
        mock.patch.object(rc.asyncio, "sleep", _sleep),
    ):
        outcome: object
        try:
            outcome = asyncio.run(
                rc.resilient_backend_request(client, "GET", "https://example.test/x")
            )
        except Exception as exc:  # noqa: BLE001
            outcome = exc
    return outcome, counter[0], sleeps


def _http_date(seconds: int) -> str:
    return format_datetime(datetime.now(UTC) + timedelta(seconds=seconds), usegmt=True)


@pytest.mark.parametrize("status", [429, 503, 500])
def test_request_path_honours_http_date_retry_after(status):
    ok = _Resp(200, {})
    outcome, attempts, sleeps = _drive(
        [_Resp(status, {"Retry-After": _http_date(20)}), ok]
    )

    assert outcome == {"ok": True}
    assert attempts == 2
    assert len(sleeps) == 1 and 17.0 <= sleeps[0] <= 21.0


@pytest.mark.parametrize("status", [429, 503])
def test_request_path_honours_delta_seconds(status):
    outcome, attempts, sleeps = _drive(
        [_Resp(status, {"Retry-After": "12"}), _Resp(200, {})]
    )

    assert outcome == {"ok": True}
    assert sleeps == [12.0]


def test_request_path_stops_when_ask_exceeds_ceiling():
    outcome, attempts, sleeps = _drive([_Resp(429, {"Retry-After": "600"})] * 3)

    assert isinstance(outcome, Exception)
    assert attempts == 1
    assert sleeps == []


def test_calculate_delay_uses_response_header_on_structured_429():
    class Boom(Exception):
        status_code = 429

        class response:  # noqa: N801
            headers = {"Retry-After": "30"}

    client = rc.ResilientClient(base_delay=1.0, max_delay=60.0, jitter_factor=0)
    assert client.calculate_delay(0, Boom("x")) == 30.0


# --- finalize through the manager's real path -------------------------------


def _finalize_client(status: int, headers: dict[str, str]):
    with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
        client = BackendIntegratedClient(
            api_key="placeholder-key",  # pragma: allowlist secret
            base_url="http://localhost:5000",
        )
    client.auth_manager = Mock()
    client.auth_manager.auth = None  # no owner fingerprint for the 403 message
    client.auth_manager.augment_headers = AsyncMock(
        return_value={"Authorization": "Bearer test-key"}
    )
    client.session_bridge.create_session_mapping(
        session_id="sess-1",
        experiment_id="exp-1",
        experiment_run_id="run-1",
        function_name="f",
        configuration_space={},
        objectives=["accuracy"],
    )
    response = AsyncMock()
    response.status = status
    response.headers = headers
    response.text = AsyncMock(return_value="")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.post = Mock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return client, session


def _manager(client):
    from tests.unit.core.test_backend_session_manager import (  # noqa: PLC0415
        BackendSessionManager,
        OptimizationStatus,
    )

    manager = BackendSessionManager(
        backend_client=client,
        traigent_config=Mock(no_egress=False),
        objectives=["accuracy"],
        objective_schema=Mock(),
        optimizer=Mock(),
        optimization_id="opt",
        optimization_status=OptimizationStatus.RUNNING,
    )
    manager._backend_tracking_enabled = True
    manager.finalize_egress_allowed = lambda: True  # type: ignore[method-assign]
    manager._remote_session_completed = lambda _sid: False  # type: ignore[method-assign]
    return manager


def _finalize(status, headers):
    from traigent.api.types import OptimizationStatus

    client, session = _finalize_client(status, headers)
    manager = _manager(client)
    sleeps: list[float] = []
    with (
        patch("aiohttp.ClientSession", return_value=session),
        patch("traigent.core.backend_session_manager.time.sleep", sleeps.append),
    ):
        try:
            manager.finalize_session("sess-1", OptimizationStatus.COMPLETED)
            outcome: object = None
        except Exception as exc:  # noqa: BLE001
            outcome = exc
    return outcome, session.post.call_count, sleeps


def test_finalize_429_retry_after_is_honoured_through_manager():
    outcome, posts, sleeps = _finalize(429, {"Retry-After": "7"})

    assert posts == 3  # all attempts spent, each wait = the server's ask
    assert sleeps == [7.0, 7.0]
    assert isinstance(outcome, RateLimitError)


def test_finalize_429_above_ceiling_stops_without_retry():
    outcome, posts, sleeps = _finalize(
        429, {"Retry-After": str(int(MAX_RETRY_AFTER_SECONDS) + 5)}
    )

    assert posts == 1
    assert sleeps == []
    assert isinstance(outcome, RateLimitError)


def test_finalize_403_stops_the_loop_through_manager():
    outcome, posts, sleeps = _finalize(403, {})

    assert posts == 1
    assert sleeps == []
    assert isinstance(outcome, CloudServiceError)


def test_finalize_503_retry_after_is_honoured_through_manager():
    outcome, posts, sleeps = _finalize(503, {"Retry-After": "9"})

    assert posts == 3
    assert sleeps == [9.0, 9.0]
    assert isinstance(outcome, RetryableError)


def test_finalize_503_above_ceiling_stops_without_retry():
    outcome, posts, sleeps = _finalize(
        503, {"Retry-After": str(int(MAX_RETRY_AFTER_SECONDS) + 5)}
    )

    assert posts == 1
    assert sleeps == []
    assert isinstance(outcome, RetryableError)
