"""Measure real async trial HTTP requests and parent session lifecycle locally."""

import asyncio
import json
import statistics
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudEgressBlockedError
from traigent.cloud.trial_operations import TrialOperations

pytestmark = pytest.mark.backend_online


class _LocalHeaders:
    """Synthetic per-request context; the local server needs no credentials."""

    def __init__(self):
        self.auth = self
        self.request_number = 0

    async def get_headers(self):
        return {"traceparent": "stale-default-context"}

    async def augment_headers(self, headers):
        self.request_number += 1
        return {
            **headers,
            "X-Test-Request": str(self.request_number),
            "traceparent": f"request-context-{self.request_number}",
        }


class _LocalTrialClient(BackendIntegratedClient):
    """Keep real transport/lifecycle methods without credential discovery."""

    def __init__(self, origin):
        self.backend_config = SimpleNamespace(
            backend_base_url=origin, api_base_url=origin
        )
        self.no_egress = False
        self._url_invalid = False
        self.timeout = 30
        self.auth_manager = _LocalHeaders()
        self._session = None
        self._session_lock = asyncio.Lock()
        self._session_finalizer = None
        self._active_sessions = {}
        self._cost_budget_armed_sessions = set()
        self._api_ops = ApiOperations(self)
        self._trial_ops = TrialOperations(self)
        # Keep the measurement scoped to slot/result HTTP requests. Config-run
        # status/measures backfills are a separate existing backend operation.
        self._update_config_run_status = AsyncMock(return_value=False)


@pytest.fixture
async def trial_server():
    requests = []
    connections = set()

    async def handle(request):
        connections.add(request.transport)
        requests.append(
            {
                "action": request.match_info["action"],
                "body": await request.json(),
                "context": request.headers.get("traceparent"),
                "request_number": request.headers.get("X-Test-Request"),
            }
        )
        if request.match_info["action"] == "next-trial":
            return web.json_response(
                {"suggestion": {"trial_id": f"trial-{len(requests)}"}},
                headers=(
                    {"Connection": "close"}
                    if request.match_info["session_id"] == "reconnect-session"
                    else None
                ),
            )
        return web.json_response({"continue_optimization": True})

    application = web.Application()
    application.router.add_post("/sessions/{session_id}/{action}", handle)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests, connections
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_async_slot_and_result_reuse_connection_and_keep_fresh_headers(
    trial_server,
):
    origin, requests, connections = trial_server
    client = _LocalTrialClient(origin)
    durations = []
    try:
        for index in range(6):
            started = time.perf_counter()
            slot = await client.request_trial_slot("local-session")
            assert slot.trial_id == f"trial-{index * 2 + 1}"
            assert (
                await client._trial_ops.submit_trial_result_via_session(
                    "local-session",
                    slot.trial_id,
                    {"temperature": 0.1},
                    {"accuracy": 1.0},
                    "completed",
                )
                is True
            )
            durations.append(time.perf_counter() - started)
        print(
            json.dumps(
                {
                    "requests": len(requests),
                    "connections": len(connections),
                    "trial_pair_median_seconds": statistics.median(durations),
                    "trial_pairs_total_seconds": sum(durations),
                }
            ),
            flush=True,
        )
        assert len(requests) == 12
        assert len(connections) == 1
        assert [request["request_number"] for request in requests] == [
            str(index) for index in range(1, 13)
        ]
        assert [request["context"] for request in requests] == [
            f"request-context-{index}" for index in range(1, 13)
        ]
        for request in requests[::2]:
            assert request["body"] == {
                "session_id": "local-session",
                "previous_results": [],
            }
        for index, request in enumerate(requests[1::2]):
            assert request["body"]["trial_id"] == f"trial-{index * 2 + 1}"
            assert request["body"]["config"] == {"temperature": 0.1}
            assert request["body"]["metrics"] == {"accuracy": 1.0}
        session = client._session
        assert session is not None and not session.closed
        assert "traceparent" not in session.headers
        finalizer = client._session_finalizer
        assert finalizer is not None and finalizer.alive
        await client.close()
        assert session.closed
        assert not finalizer.alive
        assert client._session is None
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["offline", "no_egress"])
async def test_blocked_trial_calls_do_not_allocate_session(
    trial_server, monkeypatch, policy
):
    origin, requests, connections = trial_server
    client = _LocalTrialClient(origin)
    try:
        if policy == "offline":
            monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
            assert (await client.request_trial_slot("local-session")).trial_id is None
            assert (
                await client._trial_ops.submit_trial_result_via_session(
                    "local-session", "trial", {}, {}, "completed"
                )
                is None
            )
        else:
            client.no_egress = True
            with pytest.raises(CloudEgressBlockedError):
                await client.request_trial_slot("local-session")
            with pytest.raises(CloudEgressBlockedError):
                await client._trial_ops.submit_trial_result_via_session(
                    "local-session", "trial", {}, {}, "completed"
                )
        assert client._session is None
        assert not requests
        assert not connections
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_closed_connections_and_sessions_recover_without_losing_ownership(
    trial_server,
):
    origin, requests, connections = trial_server
    client = _LocalTrialClient(origin)
    try:
        first = await client.request_trial_slot("reconnect-session")
        pooled = client._session
        assert first.trial_id == "trial-1"
        assert pooled is not None and not pooled.closed
        second = await client.request_trial_slot("local-session")
        assert second.trial_id == "trial-2"
        assert len(connections) == 2
        assert client._session is pooled
        await pooled.close()
        third = await client.request_trial_slot("local-session")
        assert third.trial_id == "trial-3"
        replacement = client._session
        assert replacement is not None and replacement is not pooled
        assert not replacement.closed
        assert len(connections) == 3
        assert [request["context"] for request in requests] == [
            "request-context-1",
            "request-context-2",
            "request-context-3",
        ]
        await client.close()
        assert replacement.closed
    finally:
        await client.close()
