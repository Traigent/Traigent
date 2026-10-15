"""Measure real async trial HTTP requests and parent session lifecycle locally."""

import asyncio
import json
import socket
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudEgressBlockedError
from traigent.cloud.session_operations import SessionOperations
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
        self._active_sessions_lock = threading.Lock()
        self._session_credentials = {}
        self._request_nonces = set()
        self._max_nonces = 1000
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
                "action": request.match_info.get("action", "summary-stats"),
                "body": await request.json(),
                "context": request.headers.get("traceparent"),
                "request_number": request.headers.get("X-Test-Request"),
            }
        )
        if request.match_info.get("action") == "next-trial":
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
    application.router.add_put("/configuration-runs/{trial_id}/summary-stats", handle)
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


@pytest.mark.asyncio
async def test_remaining_async_trial_producers_share_pool_and_fresh_headers(
    trial_server,
):
    origin, requests, connections = trial_server
    client = _LocalTrialClient(origin)
    try:
        for index in range(4):
            trial_id = f"trial-{index}"
            assert (
                await client._trial_ops.register_trial_start(
                    "local-session", trial_id, {"temperature": 0.1}
                )
                is True
            )
            assert (
                await client._trial_ops.submit_summary_stats(
                    "local-session",
                    trial_id,
                    {"temperature": 0.1},
                    {"metrics": {"accuracy": {"mean": 1.0}}, "total_examples": 1},
                )
                is True
            )
            assert (
                await client._trial_ops.update_trial_weighted_scores(
                    trial_id, 0.75, {"accuracy": [0.0, 1.0]}, {"accuracy": 1.0}
                )
                is True
            )
        assert len(requests) == 12
        assert len(connections) == 1
        assert client._session is not None and not client._session.closed
        assert [r["context"] for r in requests] == [
            f"request-context-{i}" for i in range(1, 13)
        ]
        for index in range(4):
            registration, summary, weighted = requests[index * 3 : index * 3 + 3]
            assert registration["body"]["status"] == "RUNNING"
            assert summary["body"]["metrics"] == {"accuracy": 1.0}
            assert weighted["body"]["summary_stats"]["weighted_score"] == 0.75
    finally:
        session = client._session
        await client.close()
        if session is not None:
            assert session.closed


@pytest.fixture
def threaded_trial_server():
    connections = set()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            connections.add(self.connection)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            result = (
                {"suggestion": {"trial_id": f"trial-{len(requests)}"}}
                if self.path.endswith("next-trial")
                else {"continue_optimization": True}
            )
            encoded = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_DELETE(self):
            if self.path.startswith("/sessions/failing-session"):
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)  # -> aiohttp.ClientError
                return
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, connections
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_sync_registration_does_not_replace_or_close_parent_loop_pool(
    threaded_trial_server, monkeypatch
):
    import aiohttp

    origin, requests, connections = threaded_trial_server
    created = []
    original_session = aiohttp.ClientSession

    def track_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", track_session)
    client = _LocalTrialClient(origin)
    try:
        assert (await client.request_trial_slot("local-session")).trial_id == "trial-1"
        pooled = client._session
        for index in range(2):
            assert (
                client._trial_ops.register_trial_start_sync(
                    "local-session", f"sync-{index}", {}
                )
                is True
            )
            assert client._session is pooled and not pooled.closed
            assert created[-1].closed
        assert (await client.request_trial_slot("local-session")).trial_id == "trial-4"
        assert client._session is pooled
        assert len(requests) == 4
        assert len(connections) == 3
        assert len(created) == 3
    finally:
        await client.close()
    assert all(session.closed for session in created)


def test_sync_registration_without_running_loop_closes_each_transient_session(
    threaded_trial_server, monkeypatch
):
    import aiohttp

    origin, requests, connections = threaded_trial_server
    created = []
    original_session = aiohttp.ClientSession

    def track_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", track_session)
    client = _LocalTrialClient(origin)
    for index in range(2):
        assert (
            client._trial_ops.register_trial_start_sync(
                "local-session", f"sync-{index}", {}
            )
            is True
        )
        assert client._session is None
        assert created[-1].closed
    assert len(requests) == len(connections) == len(created) == 2


def test_pooled_session_bound_to_another_loop_is_never_reused(
    threaded_trial_server, monkeypatch
):
    """A pool bound to another, still-open loop must not serve a different loop."""
    import aiohttp

    origin, requests, _connections = threaded_trial_server
    client = _LocalTrialClient(origin)

    # Another loop (still open, e.g. owned by a different thread) owns the shared pool.
    loop_a = asyncio.new_event_loop()
    pooled = loop_a.run_until_complete(client._ensure_session())
    assert pooled._loop is loop_a
    created = []
    original_session = aiohttp.ClientSession

    def track_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        created.append(session)
        return session

    try:
        monkeypatch.setattr(aiohttp, "ClientSession", track_session)

        async def run_on_new_loop():
            slot = await client.request_trial_slot("local-session")
            assert slot.trial_id == "trial-1"
            assert (
                await client._trial_ops.submit_trial_result_via_session(
                    "local-session", slot.trial_id, {}, {"accuracy": 0.5}, "completed"
                )
                is True
            )

        asyncio.run(run_on_new_loop())
        # Loop B got its own single pool (reused for both calls); loop A's is untouched.
        assert len(created) == 1 and created[0] is not pooled
        assert client._session is pooled and not pooled.closed
        assert len(requests) == 2
    finally:
        loop_a.run_until_complete(client.close())
        loop_a.close()


def test_pool_created_by_finished_loop_is_replaced_for_next_sync_consumer(
    threaded_trial_server,
):
    """asyncio.run(trial call) then a *_sync wrapper must not hit the dead pool."""
    origin, _requests, _connections = threaded_trial_server
    client = _LocalTrialClient(origin)
    asyncio.run(client.request_trial_slot("local-session"))
    stale = client._session
    assert stale is not None and stale._loop.is_closed()

    async def fresh():
        return await client._ensure_session()

    replacement = asyncio.run(fresh())
    assert replacement is not stale and replacement._loop.is_closed()
    assert client._session is replacement


@pytest.mark.asyncio
async def test_sync_wrappers_in_running_loop_never_use_foreign_loop_pool(
    threaded_trial_server, monkeypatch
):
    """delete_session_sync inside a running loop gets a loop-local session."""
    import aiohttp

    origin, _requests, _connections = threaded_trial_server
    client = _LocalTrialClient(origin)
    ops = SessionOperations(client)
    pool = await client._ensure_session()  # owned by this (still running) loop
    created = []
    original_session = aiohttp.ClientSession

    def track_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", track_session)
    try:
        assert ops.delete_session_sync("local-session") is True
        assert len(created) == 1 and created[0] is not pool
        assert created[0].closed and not pool.closed
        assert client._session is pool
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_sync_wrapper_failure_resets_only_loop_local_session(
    threaded_trial_server, monkeypatch
):
    import aiohttp

    origin, _requests, _connections = threaded_trial_server
    client = _LocalTrialClient(origin)
    ops = SessionOperations(client)
    pool = await client._ensure_session()
    created = []
    original_session = aiohttp.ClientSession

    def track_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", track_session)
    try:
        assert ops.delete_session_sync("failing-session") is False
        assert created and all(s.closed for s in created)
        assert not pool.closed and client._session is pool
    finally:
        await client.close()


def test_async_consumer_on_second_open_loop_gets_its_own_pool(threaded_trial_server):
    """Round-4 scenario: pool made on open loop A is never used from loop B."""
    origin, _requests, _connections = threaded_trial_server
    client = _LocalTrialClient(origin)
    ops = SessionOperations(client)
    loop_a = asyncio.new_event_loop()
    try:
        assert (
            loop_a.run_until_complete(client.request_trial_slot("local-session"))
        ).trial_id
        pool_a = client._session
        assert pool_a._loop is loop_a

        async def delete_on_b():
            ok = await ops.delete_session("local-session")
            return ok, await client._ensure_session()

        ok, pool_b = asyncio.run(delete_on_b())
        assert ok is True
        assert pool_b is not pool_a and pool_b._loop is not loop_a
        # Loop A's pool is untouched and still serves loop A.
        assert not pool_a.closed
        assert (
            loop_a.run_until_complete(client.request_trial_slot("local-session"))
        ).trial_id
        assert client._session is pool_a
    finally:
        loop_a.run_until_complete(client.close())
        loop_a.close()


def test_closed_loop_pools_are_released_and_context_entry_keeps_no_orphan(
    threaded_trial_server,
):
    origin, _requests, _connections = threaded_trial_server
    client = _LocalTrialClient(origin)
    loop_a = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(client.request_trial_slot("local-session"))
        pool_a = client._session

        asyncio.run(client.request_trial_slot("local-session"))  # pool on loop B
        ((_ref, pool_b),) = client._loop_pools().values()
        # Next request on A purges B's closed-loop pool even though A's is cached.
        loop_a.run_until_complete(client.request_trial_slot("local-session"))
        assert pool_b.closed and not client._loop_pools()

        async def enter_after_trial():
            await client.request_trial_slot("local-session")
            before = client._session
            async with client:
                pass
            return before

        loop_c = asyncio.new_event_loop()
        try:
            client._session = None
            before = loop_c.run_until_complete(enter_after_trial())
            assert before.closed  # not orphaned by __aenter__
        finally:
            loop_c.close()
        assert not pool_a.closed
    finally:
        loop_a.run_until_complete(client.close())
        loop_a.close()


def test_dead_primary_pool_is_released_while_secondary_pool_serves():
    """Primary on closed loop A is disposed even when B's cached pool is returned."""
    import http.server

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
    )
    client = _LocalTrialClient(f"http://127.0.0.1:{server.server_port}")
    server.server_close()
    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        pool_a = loop_a.run_until_complete(client._ensure_session())
        pool_b = loop_b.run_until_complete(client._ensure_session())
        assert client._session is pool_a and pool_b is not pool_a
        loop_a.close()
        assert loop_b.run_until_complete(client._ensure_session()) is pool_b
        assert pool_a.closed and client._session is None
    finally:
        loop_b.run_until_complete(client.close())
        loop_b.close()
