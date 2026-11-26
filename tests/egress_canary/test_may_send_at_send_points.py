"""Exercise SDK handoffs with the policy changing inside an in-progress send."""

import ast
import importlib.util
import json
import socket
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import httpx
import pytest
import requests
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.egress_canary import http_egress_scan
from traigent.cloud import (
    auth,
    backend_client,
    privacy_operations,
    session_operations,
    trial_operations,
)
from traigent.core import trace_export_gate
from traigent.integrations.observability import workflow_traces
from traigent.observability import ObservabilityClient, ObservabilityConfig
from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.transport import TransportResponse
from traigent.utils import optimization_only_egress

pytestmark = pytest.mark.backend_online
OPT_OUT = "TRAIGENT_DISABLE_TELEMETRY"
OFFLINE = "TRAIGENT_OFFLINE_MODE"
FLAGS = (None, OPT_OUT, OFFLINE)
ORIGIN = "https://8.8.8.8"
HEADERS = {"X-API-Key": "test-key"}  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def no_unexpected_egress(monkeypatch):
    escaped = []

    def unexpected(*args, **kwargs):
        escaped.append("unexpected network handoff")
        raise AssertionError(escaped[-1])

    for name in (
        "create_connection",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
    ):
        monkeypatch.setattr(socket, name, unexpected)
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, unexpected)
    monkeypatch.setattr(requests.Session, "send", unexpected)
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected)
    monkeypatch.setattr(httpx.Client, "send", unexpected)
    monkeypatch.setattr(aiohttp.ClientSession, "_request", unexpected)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", unexpected)
    for flag in (*FLAGS[1:], "TRAIGENT_OFFLINE"):
        monkeypatch.delenv(flag, raising=False)
    yield
    assert not escaped


def _flip(monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv(flag, "true")


def _spans():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    with provider.get_tracer(__name__).start_as_current_span("canary"):
        pass
    spans = memory.get_finished_spans()
    provider.shutdown()
    return spans


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("point", ("entry", "validation", "retry"))
def test_dedicated_export(monkeypatch, flag, point):
    def validate(*args, **kwargs):
        if point == "validation":
            _flip(monkeypatch, flag)
        return True

    check = Mock(side_effect=validate)
    monkeypatch.setattr(optimization_only_egress, "ensure_trace_egress_allowed", check)
    statuses = [TransportResponse(503), TransportResponse(200)]
    post = Mock(side_effect=statuses if point == "retry" else [TransportResponse(200)])
    exporter = TraigentOTLPExporter(
        SimpleNamespace(post=post), sleep=lambda _: _flip(monkeypatch, flag)
    )
    exporter.egress_credential = ("test-key", ORIGIN)
    spans = _spans()
    if point == "entry":
        _flip(monkeypatch, flag)
    result = exporter.export_batch(spans)

    if flag is None:
        assert result.exported == 1
        assert post.call_count == check.call_count == (2 if point == "retry" else 1)
        assert post.call_args.args[0]
    else:
        assert result.dropped_offline == 1
        assert result.exported == 0
        assert post.call_count == (point == "retry")
        assert check.call_count == (point != "entry")


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("point", ("entry", "validation", "next"))
def test_legacy_batches(monkeypatch, flag, point):
    validations = []
    calls = []

    def validate(*args, **kwargs):
        validations.append(args)
        if point == "validation":
            _flip(monkeypatch, flag)
        return True

    def send(request, **kwargs):
        calls.append((request.full_url, json.loads(request.data)))
        if point == "next":
            _flip(monkeypatch, flag)
        return BytesIO(b"{}")

    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validate
    )
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin=ORIGIN,
            api_key="test-key",  # pragma: allowlist secret
            batch_size=20,
            max_buffer_age=60,
            enable_atexit_flush=False,
        )
    )
    monkeypatch.setattr(client, "_open_http_request", send)
    monkeypatch.setattr(client._transport, "max_traces_per_request", 1)
    try:
        for trace_id in ("one", "two"):
            client.start_trace(trace_id, trace_id=trace_id)
            client.end_trace(trace_id)
        if point == "entry":
            _flip(monkeypatch, flag)
        result = client.flush()
        sent = 2 if flag is None else int(point == "next")
        assert len(calls) == sent
        assert len(validations) == (2 if flag is None else int(point != "entry"))
        assert result.success and result.items_pending == result.items_dropped == 0
        assert result.items_sent == result.successful_batches == sent
        assert result.items_suppressed == 2 - sent
        if calls:
            assert calls[0][0].endswith("/api/v1beta/observability/ingest")
            assert calls[0][1]["traces"][0]["id"] == "one"
        if flag is not None:
            monkeypatch.delenv(flag)
            assert client.flush().items_sent == sent
            assert len(calls) == sent
        assert client.close().items_suppressed == 2 - sent
    finally:
        client.close()


class _Response:
    def __init__(self, status, body=None):
        self.status = self.status_code = status
        self.body = body or {}
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def json(self, **kwargs):
        return self.body

    def raise_for_status(self):
        assert self.status < 400


class _Session(_Response):
    def __init__(self, post):
        super().__init__(200)
        self.post = self.get = self.put = self.delete = post


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize(
    ("point", "async_send"),
    [
        ("entry", False),
        ("validation", False),
        ("entry", True),
        ("validation", True),
        ("worker", True),
    ],
)
async def test_workflow_ingest(monkeypatch, flag, point, async_send):
    client = workflow_traces.WorkflowTracesClient(
        backend_url=ORIGIN, auth_token="test-key"
    )
    validate = Mock(return_value=True)

    def validation(*args, **kwargs):
        validate(*args, **kwargs)
        if point == "validation":
            _flip(monkeypatch, flag)
        return True

    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validation
    )
    graph = SimpleNamespace(to_dict=lambda: {"name": "canary"})
    if async_send:

        async def run_validation(callback):
            if point == "worker":
                _flip(monkeypatch, flag)
            return callback()

        monkeypatch.setattr(workflow_traces.asyncio, "to_thread", run_validation)
        post = Mock(return_value=_Response(200, {"success": True}))
        monkeypatch.setattr(
            workflow_traces.aiohttp, "ClientSession", lambda **_: _Session(post)
        )
        monkeypatch.setattr(workflow_traces, "gate_session", lambda session: session)
        send = client.ingest_traces_async
    else:
        post = Mock(
            return_value=SimpleNamespace(
                status_code=200,
                json=lambda: {"success": True},
                raise_for_status=lambda: None,
            )
        )
        monkeypatch.setattr(workflow_traces.requests, "post", post)
        send = client.ingest_traces

    if point == "entry":
        _flip(monkeypatch, flag)
    result = await send(graph=graph) if async_send else send(graph=graph)
    assert result.success == (flag is None)
    assert validate.call_count == int(flag is None or point == "validation")
    assert post.call_count == int(flag is None)
    if flag is None:
        assert post.call_args.args == (f"{ORIGIN}/api/v1/traces/ingest",)
        assert post.call_args.kwargs["json"] == {"graph": {"name": "canary"}}
        assert post.call_args.kwargs["headers"]["X-API-Key"] == HEADERS["X-API-Key"]


@pytest.mark.parametrize("flag", (None, OFFLINE))
@pytest.mark.parametrize("point", ("validation", "session"))
async def test_key_validator(monkeypatch, flag, point):
    async def validate(url, **kwargs):
        if point == "validation":
            _flip(monkeypatch, flag)
        return url

    class Session(_Session):
        async def __aenter__(self):
            if point == "session":
                _flip(monkeypatch, flag)
            return self

    manager = auth.AuthManager(auth.UnifiedAuthConfig(backend_base_url=ORIGIN))
    monkeypatch.setattr(auth, "validate_cloud_base_url_async", validate)
    post = Mock(return_value=_Response(200, {"valid": True}))
    monkeypatch.setattr(auth.aiohttp, "ClientSession", lambda **_: Session(post))
    monkeypatch.setattr(auth, "gate_session", lambda session: session)
    result = await manager._validate_api_key_with_backend(HEADERS["X-API-Key"])
    assert result == (None if flag is None else "backend offline mode enabled")
    assert post.call_count == int(flag is None)
    if flag is None:
        assert post.call_args.args[0].endswith("/keys/validate")


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("point", ("auth", "retry"))
async def test_ordinary_trial(monkeypatch, flag, point):
    async def headers(*args):
        if point == "auth":
            _flip(monkeypatch, flag)
        return HEADERS

    def post_response(*args, **kwargs):
        if point == "retry" and post.call_count == 1:
            _flip(monkeypatch, flag)
            return _Response(429)
        return _Response(201)

    post = Mock(side_effect=post_response)
    augment = AsyncMock(side_effect=headers)
    client = SimpleNamespace(
        no_egress=False,
        backend_config=SimpleNamespace(api_base_url=ORIGIN),
        auth_manager=SimpleNamespace(augment_headers=augment),
        _map_to_backend_status=lambda *args, **kwargs: "COMPLETED",
    )
    ops = trial_operations.TrialOperations(client)
    monkeypatch.setattr(
        ops,
        "_prepare_trial_result_payload",
        lambda *args: ({"metrics": {"score": 1}}, {}),
    )
    monkeypatch.setattr(ops, "_dispatch_results_response", AsyncMock(return_value=True))
    monkeypatch.setattr(trial_operations, "_submit_retry_delay", lambda *_: 0)

    @asynccontextmanager
    async def session():
        yield _Session(post)

    monkeypatch.setattr(ops, "_trial_http_session", session)
    result = await ops.submit_trial_result_via_session(
        "session", "trial", {}, {}, "completed"
    )
    assert result is (None if flag == OFFLINE else True)
    assert post.call_count == int(flag != OFFLINE) + int(point == "retry")
    augment.assert_awaited_once()
    if post.called:
        assert post.call_args.args == (f"{ORIGIN}/sessions/session/results",)


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("point", ("entry", "auth", "preflight", "next", "retry"))
def test_dataset_features(monkeypatch, flag, point):
    def headers(**kwargs):
        if point == "auth":
            _flip(monkeypatch, flag)
        return HEADERS

    def send(*args, **kwargs):
        if point in ("next", "retry") and post.call_count == 1:
            _flip(monkeypatch, flag)
            if point == "retry":
                raise requests.exceptions.ConnectionError
        return SimpleNamespace(status_code=200)

    def preflight(*args, **kwargs):
        if point == "preflight":
            _flip(monkeypatch, flag)

    monkeypatch.setattr(backend_client, "preflight_requests", preflight)
    monkeypatch.setattr(
        backend_client,
        "_SYNC_BACKEND_RETRY_CONFIG",
        replace(backend_client._SYNC_BACKEND_RETRY_CONFIG, initial_delay=0),
    )
    build_headers = Mock(side_effect=headers)
    client = SimpleNamespace(
        timeout=5,
        max_retries=1,
        _raise_if_backend_egress_disabled=Mock(),
        _get_sync_auth_headers=build_headers,
        _build_feature_upload_url=lambda run: f"{ORIGIN}/runs/{run}/features",
    )
    client._post_example_feature_chunk = (
        backend_client.BackendIntegratedClient._post_example_feature_chunk.__get__(
            client
        )
    )
    monkeypatch.setattr(
        backend_client,
        "_example_feature_chunks",
        lambda features: ([feature] for feature in features),
    )
    post = Mock(side_effect=send)
    monkeypatch.setattr(requests, "post", post)
    if point == "entry":
        _flip(monkeypatch, flag)
    result = backend_client.BackendIntegratedClient.upload_example_features(
        client, "run", "simhash_v1", [{"x": 1}, {"x": 2}]
    )
    assert result is (flag is None)
    assert post.call_count == (
        2 + int(point == "retry") if flag is None else int(point in ("next", "retry"))
    )
    assert build_headers.call_count == int(flag is None or point != "entry")
    if post.called:
        assert post.call_args_list[0].kwargs["json"]["features"] == [{"x": 1}]


@pytest.mark.parametrize("flag", (None, OFFLINE))
@pytest.mark.parametrize("finalize", (False, True))
async def test_hybrid_session(monkeypatch, flag, finalize):
    async def headers():
        _flip(monkeypatch, flag)
        return HEADERS

    response = (
        _Response(200, {"done": True})
        if finalize
        else _Response(
            201,
            {"session_id": "session", "token": "token", "optimizer_endpoint": ORIGIN},
        )
    )
    post = Mock(return_value=response)
    get_headers = AsyncMock(side_effect=headers)
    client = SimpleNamespace(
        no_egress=False,
        backend_config=SimpleNamespace(api_base_url=ORIGIN, project_id=None),
        auth_manager=SimpleNamespace(auth=SimpleNamespace(get_headers=get_headers)),
        _active_sessions_lock=RLock(),
        _active_sessions={},
        _revoke_security_session=Mock(),
        _ensure_session=AsyncMock(return_value=_Session(post)),
    )
    ops = session_operations.SessionOperations(client)
    monkeypatch.setattr(ops, "_with_project", lambda value: value)

    async def send():
        if finalize:
            return await ops.finalize_hybrid_session("session")
        return await ops.create_hybrid_session("canary", {}, {})

    if flag is None:
        assert await send() == (
            {"done": True} if finalize else ("session", "token", ORIGIN)
        )
        post.assert_called_once()
        assert post.call_args.args[0].endswith(
            "/hybrid/sessions/session/finalize" if finalize else "/hybrid/sessions"
        )
    else:
        from traigent.cloud.client import CloudEgressBlockedError

        with pytest.raises(CloudEgressBlockedError, match="offline"):
            await send()
        post.assert_not_called()
    get_headers.assert_awaited_once()


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("point", ("open", "entry", "validation", "retry"))
@pytest.mark.parametrize("plugin", (False, True), ids=("core", "plugin"))
def test_cached_otlp_export(monkeypatch, flag, point, plugin):
    from opentelemetry.sdk.trace.export import SpanExportResult

    from traigent.core import tracing

    otlp = pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    if plugin:
        path = (
            Path(__file__).resolve().parents[2]
            / "plugins/traigent-tracing/traigent_tracing/tracing.py"
        )
        spec = importlib.util.spec_from_file_location("canary_tracing_plugin", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = tracing
        monkeypatch.setattr(
            tracing, "OTLPHttpSpanExporter", otlp.OTLPSpanExporter, raising=False
        )
        monkeypatch.setattr(tracing, "_provider", None, raising=False)
        monkeypatch.setattr(tracing, "_frozen_api_key", None, raising=False)

    providers, exporters, sends = [], [], []
    monkeypatch.setattr(module, "_initialized", False)
    monkeypatch.setattr(module, "_tracer", None)
    monkeypatch.setattr(module, "TRACING_ENABLED", True)
    monkeypatch.setattr(
        module, "trace", SimpleNamespace(set_tracer_provider=providers.append)
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", ORIGIN)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=test-key"
    )  # pragma: allowlist secret
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", raising=False)

    def processor(exporter):
        exporters.append(exporter)
        return SimpleSpanProcessor(exporter)

    monkeypatch.setattr(module, "BatchSpanProcessor", processor)
    validate_setup = Mock(return_value=True)
    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validate_setup
    )

    validate_export = Mock(return_value=False)
    monkeypatch.setattr(tracing, "_optimization_only_blocks_export", validate_export)
    prepared = Mock(return_value=True)

    def validate_request(session, request):
        prepared(request)
        if point == "validation":
            _flip(monkeypatch, flag)
        return True

    def send(session, request, **kwargs):
        sends.append(request)
        response = requests.Response()
        response.status_code = 200
        if point == "retry" and len(sends) == 1:
            _flip(monkeypatch, flag)
            response.status_code = 503
        response._content = b""
        return response

    monkeypatch.setattr(
        trace_export_gate._GatedSession, "_prepared_request_allowed", validate_request
    )
    monkeypatch.setattr(requests.Session, "send", send)
    if point == "open":
        _flip(monkeypatch, flag)
    if point == "open" and flag is not None:
        assert module.get_tracer() is None
        validate_setup.assert_not_called()
        return

    assert module.get_tracer() is not None
    validate_setup.assert_called_once()
    try:
        if point == "entry":
            _flip(monkeypatch, flag)
        result = exporters[0].export(_spans())
        assert result is SpanExportResult.SUCCESS
        assert len(sends) == (
            1 + int(point == "retry") if flag is None else int(point == "retry")
        )
        expected_prepared = int(flag is None or point != "entry") + int(
            flag is None and point == "retry"
        )
        assert prepared.call_count == expected_prepared
        assert validate_export.call_count == (
            expected_prepared + 1 if expected_prepared else 0
        )
        if sends:
            assert sends[0].url == f"{ORIGIN}/v1/traces"
            assert sends[0].body
            assert sends[0].headers["X-API-Key"] == HEADERS["X-API-Key"]
    finally:
        providers[0].shutdown()


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize(
    "point", ("entry", "factory", "client", "create", "trial", "retry")
)
async def test_usage_analytics(monkeypatch, flag, point):
    from traigent.config.backend_config import BackendConfig
    from traigent.config.types import TraigentConfig
    from traigent.utils import local_analytics

    auth_calls = []
    requests_seen = []

    def auth_headers(op):
        auth_calls.append(op)
        if point == op:
            _flip(monkeypatch, flag)
        return HEADERS

    def post(url, **kwargs):
        requests_seen.append((url, kwargs["json"]))
        if url.endswith("/hybrid/sessions"):
            return _Response(
                201,
                {
                    "session_id": "session",
                    "token": "token",
                    "optimizer_endpoint": ORIGIN,
                },
            )
        trial_count = sum(url.endswith("/results") for url, _ in requests_seen)
        if point == "retry" and trial_count == 1:
            _flip(monkeypatch, flag)
            return _Response(429)
        return _Response(201, {"success": True})

    client = backend_client.BackendIntegratedClient.__new__(
        backend_client.BackendIntegratedClient
    )
    client.no_egress = False
    client.backend_config = SimpleNamespace(
        api_base_url=ORIGIN, backend_base_url=ORIGIN, project_id=None
    )
    client._active_sessions_lock = RLock()
    client._active_sessions = {}
    client._session_ops = session_operations.SessionOperations(client)
    client._trial_ops = trial_operations.TrialOperations(client)
    client._privacy_ops = privacy_operations.PrivacyOperations(client)
    client._check_rate_limit = AsyncMock()
    client._ensure_session = AsyncMock(return_value=_Session(post))
    client.close = AsyncMock()
    client._pool_for_loop = Mock(return_value=_Session(post))
    client._generate_request_nonce = Mock(return_value="canary")
    client._map_to_backend_status = Mock(return_value="COMPLETED")
    client.auth_manager = SimpleNamespace(
        auth=SimpleNamespace(get_headers=AsyncMock(side_effect=[HEADERS])),
        augment_headers=AsyncMock(),
    )

    async def session_headers():
        op = "client" if not auth_calls else "create"
        return auth_headers(op)

    async def trial_headers(*args):
        return auth_headers("trial")

    client.auth_manager.auth.get_headers.side_effect = session_headers
    client.auth_manager.augment_headers.side_effect = trial_headers
    monkeypatch.setattr(client._session_ops, "_with_project", lambda headers: headers)
    monkeypatch.setattr(
        client._trial_ops,
        "_prepare_trial_result_payload",
        lambda *args: ({"metrics": args[3]}, {}),
    )
    monkeypatch.setattr(
        client._trial_ops, "_dispatch_results_response", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(trial_operations, "_submit_retry_delay", lambda *_: 0)

    @asynccontextmanager
    async def session():
        yield _Session(post)

    monkeypatch.setattr(client._trial_ops, "_trial_http_session", session)
    monkeypatch.setattr(BackendConfig, "get_api_key", lambda: "a" * 30)
    monkeypatch.setattr(BackendConfig, "get_backend_url", lambda: ORIGIN)

    def make_client(**kwargs):
        if point == "factory":
            _flip(monkeypatch, flag)
        return client

    get_client = Mock(side_effect=make_client)
    monkeypatch.setattr(backend_client, "get_backend_client", get_client)
    analytics = local_analytics.LocalAnalytics.__new__(local_analytics.LocalAnalytics)
    analytics.config = TraigentConfig(enable_usage_analytics=True)
    analytics.enabled = True
    analytics.user_id = "test-user"
    analytics.collect_usage_stats = Mock(return_value={"total_sessions": 1})
    analytics._update_last_submission = Mock()
    if point == "entry":
        _flip(monkeypatch, flag)

    result = await analytics.submit_usage_stats(force=True)
    if flag is None:
        assert result["success"] is True and result["session_id"] == "session", result
        analytics._update_last_submission.assert_called_once()
        expected_auth = ["client", "create", "trial"]
        expected_requests = 2 + int(point == "retry")
    else:
        assert result == {"success": False, "reason": "Analytics disabled"}
        analytics._update_last_submission.assert_not_called()
        auth_count = {
            "entry": 0,
            "factory": 0,
            "client": 1,
            "create": 2,
            "trial": 3,
            "retry": 3,
        }[point]
        expected_auth = ["client", "create", "trial"][:auth_count]
        expected_requests = {
            "entry": 0,
            "factory": 0,
            "client": 0,
            "create": 0,
            "trial": 1,
            "retry": 2,
        }[point]
    assert len(requests_seen) == expected_requests
    assert auth_calls == expected_auth
    assert get_client.call_count == int(flag is None or point != "entry")
    if requests_seen:
        assert requests_seen[0][1]["search_space"]["type"] == "analytics"
    if len(requests_seen) > 1:
        assert requests_seen[1][1]["metrics"]["total_sessions"] == 1


@pytest.mark.parametrize("prepared_call", (False, True), ids=("submit", "prepared"))
@pytest.mark.parametrize("point", ("online", "opt-out", "entry", "client", "retry"))
async def test_explicit_economics(monkeypatch, prepared_call, point):
    from tests.unit.economics.test_client import _event, _ingest_body, _resp
    from traigent.economics import EconomicsTelemetryClient
    from traigent.utils.error_handler import OfflineModeError

    client = EconomicsTelemetryClient(backend_url=ORIGIN, api_key=HEADERS["X-API-Key"])
    event = _event()
    prepared = client.prepare([event], project_id="proj-1")
    status = 503 if point == "retry" else 201
    post = AsyncMock(
        side_effect=[
            _resp(status, _ingest_body(prepared)),
            _resp(201, _ingest_body(prepared)),
        ]
    )
    client._client = SimpleNamespace(post=post)
    if point == "client":
        get_client = client._get_client

        def flip_after_client():
            result = get_client()
            _flip(monkeypatch, OFFLINE)
            return result

        monkeypatch.setattr(client, "_get_client", flip_after_client)
    monkeypatch.setattr(
        client,
        "_backoff",
        AsyncMock(side_effect=lambda *_: _flip(monkeypatch, OFFLINE)),
    )
    # Use the same batch when testing submission through either public entry point.
    monkeypatch.setattr(client, "prepare", Mock(return_value=prepared))

    async def submit():
        if prepared_call:
            return await client.submit_prepared(prepared)
        return await client.submit([event], project_id="proj-1")

    if point in ("entry", "client", "retry"):
        if point == "entry":
            _flip(monkeypatch, OFFLINE)
        with pytest.raises(OfflineModeError):
            await submit()
        assert post.await_count == int(point == "retry")
    else:
        if point == "opt-out":
            _flip(monkeypatch, OPT_OUT)
        result = await submit()
        assert result.accepted == 1
        post.assert_awaited_once()
    if post.called:
        assert post.call_args.args[0].endswith("/economics/telemetry")
        assert post.call_args.kwargs["content"] == prepared.content


@pytest.mark.parametrize("flag", FLAGS)
def test_workflow_setup(monkeypatch, flag):
    validation = Mock(return_value=True)
    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validation
    )
    provider = Mock()
    monkeypatch.setattr(workflow_traces.trace, "set_tracer_provider", provider)
    _flip(monkeypatch, flag)
    exporter = workflow_traces.setup_workflow_tracing(
        backend_url=ORIGIN, auth_token=HEADERS["X-API-Key"]
    )
    try:
        assert (exporter is not None) == (flag is None)
        assert validation.call_count == provider.call_count == int(flag is None)
    finally:
        if provider.called:
            provider.call_args.args[0].shutdown()


@pytest.mark.parametrize("flag", FLAGS)
def test_dedicated_preflight_before_validation(monkeypatch, flag):
    from traigent.observability.otel import api

    validate = Mock(return_value=True)
    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validate
    )
    # The initializer took an online snapshot; flags may change before preflight.
    _flip(monkeypatch, flag)
    assert api._preflight_allows_export(HEADERS["X-API-Key"], ORIGIN, False) == (
        flag is None
    )
    assert validate.call_count == int(flag is None)


@pytest.mark.parametrize("flag", (None, OFFLINE))
async def test_cloud_client_health_attempt(monkeypatch, flag):
    from traigent.cloud.client import CloudEgressBlockedError, TraigentCloudClient

    async def headers():
        _flip(monkeypatch, flag)
        return HEADERS

    get = Mock(return_value=_Response(200, {"status": "ok"}))
    client = TraigentCloudClient.__new__(TraigentCloudClient)
    client.no_egress = False
    client._aio_session = _Session(get)
    client._get_headers = AsyncMock(side_effect=headers)
    url = f"{ORIGIN}/health"
    if flag is None:
        assert await client._check_service_status_attempt(url, 1, 1) == {"status": "ok"}
        get.assert_called_once_with(url, headers=HEADERS)
    else:
        with pytest.raises(CloudEgressBlockedError):
            await client._check_service_status_attempt(url, 1, 1)
        get.assert_not_called()
    client._get_headers.assert_awaited_once()


@pytest.mark.parametrize("flag", (None, OFFLINE))
@pytest.mark.parametrize("operation", ("register", "slot", "summary", "weights"))
async def test_trial_operation_handoffs(monkeypatch, flag, operation):
    async def headers(*args):
        _flip(monkeypatch, flag)
        return HEADERS

    post = Mock(return_value=_Response(200, {"suggestion": {"trial_id": "trial"}}))
    auth_headers = AsyncMock(side_effect=headers)
    client = SimpleNamespace(
        no_egress=False,
        backend_config=SimpleNamespace(api_base_url=ORIGIN),
        auth_manager=SimpleNamespace(augment_headers=auth_headers),
        _map_to_backend_status=lambda *args, **kwargs: "RUNNING",
        _privacy_mode=False,
    )
    ops = trial_operations.TrialOperations(client)

    @asynccontextmanager
    async def session():
        yield _Session(post)

    monkeypatch.setattr(ops, "_trial_http_session", session)
    if operation == "register":
        result = await ops.register_trial_start("session", "trial", {})
    elif operation == "slot":
        result = await ops.request_trial_slot("session")
    elif operation == "summary":
        result = await ops.submit_summary_stats(
            "session", "trial", {}, {"total_examples": 1}
        )
    else:
        result = await ops.update_trial_weighted_scores("trial", 1)
    assert post.call_count == int(flag is None)
    auth_headers.assert_awaited_once()
    if operation == "slot":
        assert result == (
            trial_operations.TrialSlotResult.acquired("trial")
            if flag is None
            else trial_operations.TrialSlotResult.unavailable()
        )
    else:
        assert result is (True if flag is None else None)


@pytest.mark.parametrize("flag", (None, OFFLINE))
async def test_key_validator_requests_fallback(monkeypatch, flag):
    async def validate(url, **kwargs):
        return url

    async def inline_executor(executor, operation, *args):
        return operation(*args)

    def before_post(*args, **kwargs):
        _flip(monkeypatch, flag)

    import asyncio

    manager = auth.AuthManager(config=auth.UnifiedAuthConfig(backend_base_url=ORIGIN))
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", inline_executor)
    monkeypatch.setattr(auth, "AIOHTTP_AVAILABLE", False)
    monkeypatch.setattr(auth, "validate_cloud_base_url_async", validate)
    monkeypatch.setattr(auth, "preflight_requests", before_post)
    post = Mock(
        return_value=SimpleNamespace(status_code=200, json=lambda: {"valid": True})
    )
    monkeypatch.setattr(requests, "post", post)
    result = await manager._validate_api_key_with_backend(HEADERS["X-API-Key"])
    assert result == (None if flag is None else "backend offline mode enabled")
    assert post.call_count == int(flag is None)


@pytest.mark.parametrize("flag", FLAGS)
def test_workflow_span_export(monkeypatch, flag):
    from opentelemetry.sdk.trace.export import SpanExportResult

    exporter = workflow_traces.TraigentSpanExporter(
        backend_url=ORIGIN, auth_token=HEADERS["X-API-Key"]
    )
    exporter.set_configuration_run_id("trial")

    def validate(*args, **kwargs):
        _flip(monkeypatch, flag)
        return True

    validation = Mock(side_effect=validate)
    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validation
    )
    post = Mock(
        return_value=SimpleNamespace(
            status_code=200,
            json=lambda: {"success": True},
            raise_for_status=lambda: None,
        )
    )
    monkeypatch.setattr(requests, "post", post)
    result = exporter.export(_spans())
    assert result is (
        SpanExportResult.SUCCESS if flag is None else SpanExportResult.FAILURE
    )
    assert post.call_count == int(flag is None)
    validation.assert_called_once()
    if post.called:
        assert post.call_args.kwargs["json"]["spans"]["configuration_run_id"] == "trial"


@pytest.mark.parametrize("flag", FLAGS)
def test_legacy_retry(monkeypatch, flag):
    from urllib.error import HTTPError

    calls = []
    validations = Mock(return_value=True)
    monkeypatch.setattr(
        optimization_only_egress, "ensure_trace_egress_allowed", validations
    )
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin=ORIGIN,
            api_key=HEADERS["X-API-Key"],
            batch_size=20,
            max_buffer_age=60,
            enable_atexit_flush=False,
        )
    )
    monkeypatch.setattr(client._transport._retry_handler.config, "initial_delay", 0)

    def send(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            _flip(monkeypatch, flag)
            raise HTTPError(request.full_url, 503, "test", {}, BytesIO(b"{}"))
        return BytesIO(b"{}")

    monkeypatch.setattr(client, "_open_http_request", send)
    try:
        trace_id = client.start_trace("canary")
        client.end_trace(trace_id)
        result = client.close()
        assert result.success and result.items_pending == result.failed_batches == 0
        assert (result.items_sent, result.items_suppressed) == (
            (1, 0) if flag is None else (0, 1)
        )
        assert len(calls) == validations.call_count == (2 if flag is None else 1)
    finally:
        client.close()


def test_telemetry_policy_reads_flags_and_run_scope_live(monkeypatch):
    from traigent.utils.env_config import offline_run_scope
    from traigent.utils.send_policy import may_send_telemetry

    assert may_send_telemetry()
    monkeypatch.setenv(OPT_OUT, "unrecognized")
    assert may_send_telemetry()
    monkeypatch.setenv(OPT_OUT, "  YeS  ")
    assert not may_send_telemetry()
    monkeypatch.delenv(OPT_OUT)
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    assert not may_send_telemetry()
    monkeypatch.delenv("TRAIGENT_OFFLINE")
    with offline_run_scope(True):
        assert not may_send_telemetry()
    assert may_send_telemetry()


@pytest.mark.parametrize("flag", (None, OFFLINE))
@pytest.mark.parametrize(
    "path", ("create", "get", "finalize", "delete", "finalize_api")
)
async def test_session_operation_handoffs(monkeypatch, flag, path):
    from traigent.cloud import api_operations
    from traigent.cloud.client import CloudEgressBlockedError

    async def headers(*args):
        _flip(monkeypatch, flag)
        return HEADERS

    status = 201 if path == "create" else 200
    post = Mock(return_value=_Response(status, {"session_id": "session"}))
    auth_headers = AsyncMock(side_effect=headers)
    client = SimpleNamespace(
        no_egress=False,
        backend_config=SimpleNamespace(api_base_url=ORIGIN, project_id=None),
        auth_manager=SimpleNamespace(
            auth=SimpleNamespace(get_headers=auth_headers), augment_headers=auth_headers
        ),
        _active_sessions_lock=RLock(),
        _active_sessions={"session": object()},
        _ensure_session=AsyncMock(return_value=_Session(post)),
        _revoke_security_session=Mock(),
        session_bridge=SimpleNamespace(
            get_session_mapping=lambda _: True, _session_mappings={"session": "run"}
        ),
    )
    ops = session_operations.SessionOperations(client)
    monkeypatch.setattr(ops, "_with_project", lambda headers: headers)
    for module in (api_operations, session_operations):
        monkeypatch.setattr(module, "gate_session", lambda session, **kwargs: session)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_: _Session(post))

    if path == "create":
        api_ops = api_operations.ApiOperations(client)

        class EnterSession(_Session):
            async def __aenter__(self):
                _flip(monkeypatch, flag)
                return self

        monkeypatch.setattr(aiohttp, "ClientSession", lambda **_: EnterSession(post))
        monkeypatch.setattr(api_ops, "_gate_origins", lambda: ())
        parsed = api_operations.TraigentSessionApiResult("session", "experiment", "run")
        monkeypatch.setattr(
            api_ops, "_parse_session_response", AsyncMock(return_value=parsed)
        )

        async def send():
            return await api_ops._post_session_creation(
                {"function_name": "canary"}, HEADERS, None
            )

        if flag is None:
            assert await send() is parsed
        else:
            with pytest.raises(CloudEgressBlockedError):
                await send()
    elif path == "get":
        if flag is None:
            assert await ops.get_hybrid_session_status("session") == {
                "session_id": "session"
            }
        else:
            with pytest.raises(CloudEgressBlockedError):
                await ops.get_hybrid_session_status("session")
    elif path == "finalize":
        if flag is None:
            assert await ops.finalize_hybrid_session("session") == {
                "session_id": "session"
            }
        else:
            with pytest.raises(CloudEgressBlockedError):
                await ops.finalize_hybrid_session("session")
    elif path == "delete":
        assert await ops.delete_session("session")
        assert not client.session_bridge._session_mappings
    else:
        result = await ops._finalize_session_via_api("session", None)
        assert result == ({"session_id": "session"} if flag is None else None)

    assert auth_headers.await_count == int(path != "create")
    assert post.call_count == int(flag is None)


@pytest.mark.parametrize("flag", FLAGS)
def test_manager_rechecks_offline_even_without_config(monkeypatch, flag):
    from traigent.core.backend_session_manager import BackendSessionManager

    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._no_egress = False
    manager._traigent_config = None
    _flip(monkeypatch, flag)
    assert manager._egress_disabled() == (flag == OFFLINE)
    assert manager.finalize_egress_allowed() == (flag != OFFLINE)


@pytest.mark.parametrize("flag", (None, OFFLINE))
@pytest.mark.parametrize(
    ("operation", "args", "endpoint", "status"),
    [
        ("_submit_optimization", ({},), "/optimize", 200),
        ("fetch_tvar_recommendations", (), "/optimization/recommendations", 200),
        ("delete_session", ("session",), "/sessions/session", 204),
        ("list_sessions", (), "/sessions", 200),
        ("create_optimization_session", ("canary", {}, ["score"]), "/sessions", 201),
        ("get_next_trial", ("session",), "/sessions/session/next-trial", 200),
        (
            "submit_trial_result",
            ("session", "trial", {}, 1.0),
            "/sessions/session/results",
            204,
        ),
        ("finalize_optimization", ("session",), "/sessions/session/finalize", 200),
        ("optimize_agent", (None, None, {}), "/agent/optimize", 200),
        ("execute_agent", (SimpleNamespace(agent_spec=None),), "/agent/execute", 200),
        (
            "get_agent_optimization_status",
            ("operation",),
            "/agent/optimize/operation/status",
            200,
        ),
        (
            "cancel_agent_optimization",
            ("operation",),
            "/agent/optimize/operation/cancel",
            200,
        ),
    ],
    ids=lambda value: (
        value if isinstance(value, str) and not value.startswith("/") else None
    ),
)
async def test_cloud_client_handoffs(
    monkeypatch, flag, operation, args, endpoint, status
):
    from traigent.cloud.client import CloudEgressBlockedError, TraigentCloudClient

    async def headers():
        _flip(monkeypatch, flag)
        return HEADERS

    post = Mock(return_value=_Response(status, {"session_id": "session"}))
    client = TraigentCloudClient.__new__(TraigentCloudClient)
    client.no_egress = False
    client.api_base_url = ORIGIN
    client.config = {}
    client._aio_session = _Session(post)
    client._ensure_session = AsyncMock(return_value=client._aio_session)
    client._get_headers = AsyncMock(side_effect=headers)
    client._session_owners = {}
    client._remember_session_owner = Mock()
    client._remember_cost_budget_armed_session = Mock()
    client._should_skip_tvar_recommendations_fetch = Mock(return_value=False)
    for name in (
        "session",
        "next_trial",
        "trial_result",
        "finalization",
        "agent_optimization",
        "agent_execution",
    ):
        setattr(client, f"_serialize_{name}_request", Mock(return_value={}))
        setattr(
            client,
            f"_deserialize_{name}_response",
            Mock(return_value=SimpleNamespace(session_id="session")),
        )
    client._serialize_trial_result = Mock(return_value={})
    client._deserialize_agent_optimization_status = Mock(return_value={"status": "ok"})

    send = getattr(client, operation)
    if flag is not None and operation != "fetch_tvar_recommendations":
        with pytest.raises(CloudEgressBlockedError):
            await send(*args)
    else:
        result = await send(*args)
        if operation == "delete_session":
            assert result is True
        if operation == "list_sessions":
            assert result == []
        if operation == "create_optimization_session":
            assert result.session_id == "session"
    assert post.call_count == int(flag is None)
    client._get_headers.assert_awaited_once()
    if post.called:
        assert post.call_args.args == (f"{ORIGIN}{endpoint}",)
        assert post.call_args.kwargs["headers"]["X-API-Key"] == HEADERS["X-API-Key"]


@pytest.mark.parametrize("flag", (None, OFFLINE))
async def test_cloud_client_retry_checks_before_auth(monkeypatch, flag):
    from traigent.cloud.client import CloudEgressBlockedError, TraigentCloudClient
    from traigent.utils.retry import RetryConfig

    post = Mock(side_effect=[_Response(429), _Response(200, {"success": True})])
    client = TraigentCloudClient.__new__(TraigentCloudClient)
    client.no_egress = False
    client.api_base_url = ORIGIN
    client._aio_session = _Session(post)
    client._ensure_session = AsyncMock()
    client._get_headers = AsyncMock(return_value=dict(HEADERS))

    def delay(*args):
        _flip(monkeypatch, flag)
        return 0

    monkeypatch.setattr(RetryConfig, "calculate_delay", delay)
    if flag is None:
        assert await client._submit_optimization({}) == {"success": True}
    else:
        with pytest.raises(CloudEgressBlockedError):
            await client._submit_optimization({})
    assert (
        post.call_count == client._get_headers.await_count == (2 if flag is None else 1)
    )
    if flag is None:
        assert (
            post.call_args_list[0].kwargs["headers"]
            == post.call_args_list[1].kwargs["headers"]
        )


@pytest.mark.parametrize("flag", FLAGS)
def test_legacy_control_uses_cached_setting(monkeypatch, flag):
    send = Mock(return_value={"ok": True})
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin=ORIGIN,
            api_key=HEADERS["X-API-Key"],
            enable_atexit_flush=False,
        ),
        request_sender=send,
    )
    try:
        _flip(monkeypatch, flag)
        assert client._request_json("POST", "/feedback", {"rating": "up"}) == {
            "ok": True
        }
        send.assert_called_once()
    finally:
        client.close()


class _HttpHandoffs(ast.NodeVisitor):
    """Inventory direct HTTP calls and calls on session/client-like receivers."""

    _methods = {
        "delete",
        "get",
        "head",
        "options",
        "patch",
        "post",
        "put",
        "request",
        "send",
    }
    _receivers = (
        "session",
        "http",
        "client",
        "opener",
        "pool",
        "transport",
        "connector",
    )

    def __init__(self, tree):
        self.calls = Counter()
        self.functions = []
        self.aliases = http_egress_scan._aliases(tree)

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        dotted = http_egress_scan._dotted(node.func)
        head, _, tail = dotted.partition(".")
        target = self.aliases.get(head, head)
        resolved = f"{target}.{tail}" if tail else target
        receiver, _, method = resolved.rpartition(".")
        transport = receiver.split(".")[0] in {"aiohttp", "httpx", "requests", "urllib"}
        session = any(part in receiver.lower() for part in self._receivers)
        http_call = method in self._methods and (
            transport or session or method in {"post", "request", "patch"}
        )
        urllib_call = method == "open" and session or resolved.endswith(".urlopen")
        sdk_handoff = method in {"_open_http_request", "_request"} or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
            and node.func.attr == "send"
        )
        if http_call or urllib_call or sdk_handoff:
            function = self.functions[-1] if self.functions else "<module>"
            self.calls[f"{function}/{method or node.func.attr}"] += 1
        self.generic_visit(node)


def _http_handoffs(root):
    for file in sorted(root.rglob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        visitor = _HttpHandoffs(tree)
        visitor.visit(tree)
        if visitor.calls:
            yield file.relative_to(root).as_posix(), visitor.calls


def test_new_http_handoffs_require_a_canary():
    root = Path(__file__).resolve().parents[2] / "traigent"
    expected = {
        path: Counter(calls.split()) for path, calls in _EXISTING_HANDOFFS.items()
    }
    for (path, call), canary in _NEW_HANDOFF_CANARIES.items():
        assert callable(globals().get(canary)), f"Missing canary: {canary}"
        expected.setdefault(path, Counter())[call] += 1
    actual = dict(_http_handoffs(root))
    unexpected = {
        path: dict(calls - expected.get(path, Counter()))
        for path, calls in actual.items()
        if calls - expected.get(path, Counter())
    }
    assert not unexpected, f"Add a capture test and a canary entry for: {unexpected}"


# New HTTP sites go here with the name of the test that captures their handoff.
_NEW_HANDOFF_CANARIES = {}

# Baseline by module, enclosing function and method. Repeat entries for multiple
# handoffs with the same identifier so another call in the function is detected.
_EXISTING_HANDOFFS = {
    "analytics/example_insights.py": (
        "_poll_endpoint/get compute_scores/post get_job_status/get"
    ),
    "analytics/optimization_plan.py": "get_optimization_plan/post",
    "certification/bind.py": "_post/request check_g1_readiness/request",
    "certification/client_keys.py": "enroll_client_key/request",
    "certification/transport.py": "request/request",
    "cli/auth_commands.py": (
        "_authenticate_with_backend/post _authenticate_with_backend/post "
        "_check_api_key/post _check_api_key_state/post _poll_device_token_once/post "
        "_start_device_authorization/post"
    ),
    "cli/local_commands.py": "_sync_all_sessions/get",
    "cloud/analytics_client.py": "_get_json/get _post_json/post compare_runs/post",
    "cloud/api_operations.py": (
        "_post_session_creation/post _send_config_run_measures/put "
        "create_traigent_session_via_api/get report_intermediate_progress/post "
        "update_config_run_status/put update_experiment_run_status_on_completion/put"
    ),
    "cloud/auth.py": (
        "_oauth2_client_credentials_flow/post _validate_api_key_with_backend/post "
        "_validate_api_key_with_backend_sync/post"
    ),
    "cloud/backend_bridges.py": (
        "get_all_trial_mappings/get get_session_mapping/get "
        "get_session_mapping_summary/get get_trial_count/get get_trial_mapping/get "
        "remove_trial_mapping/get"
    ),
    "cloud/backend_client.py": (
        "_get_best_config/get _get_head/get _pool_for_loop/get _post_best_config/post "
        "_post_example_features/post _process_records_request_sync/request "
        "_protected_request/request fetch_content_identity_grant_sync/post "
        "get_interaction_policy/get get_session_token/get submit_result/get"
    ),
    "cloud/backend_synchronizer.py": (
        "_select_sync_queue_sessions/get _start_queued_session_syncs/get"
    ),
    "cloud/billing.py": "get_session_costs/get",
    "cloud/client.py": (
        "_check_service_status_attempt/get _raise_ownership_error/get "
        "cancel_agent_optimization/post create_optimization_session/post "
        "delete_session/delete execute_agent/post fetch_tvar_recommendations/get "
        "finalize_optimization/post get_agent_optimization_status/get "
        "get_next_trial/post list_sessions/get optimize_agent/post submit_request/post "
        "submit_trial_result/post"
    ),
    "cloud/content_less.py": "replay_refused/get",
    "cloud/dataset_converter.py": (
        "_create_backend_example_set/post _fetch_backend_example_set/get "
        "_fetch_backend_examples/get _upload_examples_to_backend/post"
    ),
    "cloud/optimizer_client.py": (
        "_submit_batch/post _submit_single/post get_next_configuration/get "
        "get_session_status/get"
    ),
    "cloud/password_auth_handler.py": "perform_login/post",
    "cloud/privacy_operations.py": (
        "get_next_privacy_trial/get submit_privacy_trial_results/get"
    ),
    "cloud/resilient_client.py": "_send_once/request",
    "cloud/session_operations.py": (
        "_delete_session_via_api/delete _finalization_warm_start/get "
        "_finalize_session_via_api/post _recover_finalization_mapping/get "
        "create_hybrid_session/post finalize_hybrid_session/post "
        "get_hybrid_session_status/get"
    ),
    "cloud/sessions.py": (
        "_is_better_result/get cleanup_completed_sessions/get create/get delete/get "
        "get/get get_session_state/get get_session_summary/get list_active/get "
        "list_active/get"
    ),
    "cloud/sync_manager.py": (
        "_session_identity/get _session_identity/get _session_identity/get "
        "_session_labels/get _session_labels/get _sync_create_session/get "
        "_sync_create_session/post _sync_finalize_session/post _sync_next_trial/post "
        "_sync_session_results/post convert_session_to_traigent_format/get "
        "convert_session_to_traigent_format/get"
    ),
    "cloud/token_manager.py": "perform_refresh/post perform_refresh/post",
    "cloud/trial_operations.py": (
        "register_trial_start/post request_trial_slot/post resolve_privacy_enabled/get "
        "resolve_privacy_enabled/get submit_summary_stats/post "
        "submit_trial_result_via_session/post update_trial_weighted_scores/put"
    ),
    "cloud/trial_tracker.py": (
        "cleanup_old_trials/get get_active_trials/get get_completed_trials/get "
        "get_failed_trials/get get_session_trial_summary/get get_session_trials/get "
        "get_trial_state/get get_trial_state/get remove_session_trials/get"
    ),
    "cloud/url_security.py": "open_no_redirect/open",
    "core/backend_session_manager.py": (
        "_add_warm_start_transfer/get attach_session_metadata/get "
        "build_session_aggregation_payload/get build_session_aggregation_payload/get "
        "build_session_aggregation_payload/get build_session_aggregation_payload/get"
    ),
    "core/metadata_helpers.py": "merge_run_metrics_into_session_summary/get",
    "core/resume.py": "lookup/get lookup/get",
    "core/trace_export_gate.py": "send/send",
    "economics/client.py": "_post_with_retry/post",
    "hybrid/http_transport.py": (
        "_request/request benchmarks/_request benchmarks/_request capabilities/_request "
        "discover_config_space/_request evaluate/_request execute/_request "
        "execute/_request health_check/_request keep_alive/_request"
    ),
    "hybrid/lifecycle.py": "get_session_info/get is_session_alive/get",
    "integrations/langfuse/client.py": (
        "_fetch_observations_v2/get _fetch_observations_v2_async/get "
        "_get_observations_async_legacy/get _get_observations_http_legacy/get "
        "_get_trace_async_legacy/get _get_trace_http_legacy/get"
    ),
    "integrations/llms/nous_auth.py": "_request_new_token/post",
    "integrations/model_discovery/azure_discovery.py": "_fetch_models_from_sdk/get",
    "integrations/observability/workflow_traces.py": (
        "ingest_traces/post ingest_traces_async/post"
    ),
    "labels/transport.py": (
        "_call/_request _request/request _write_label_negotiated/_request"
    ),
    "labels/transport_gate.py": (
        "_request/_request _request/_request send/send send/send send/send send/send "
        "send/send send/send"
    ),
    "observability/client.py": (
        "_open_http_request/open _post_batch_sync/_open_http_request "
        "_request_json_sync/_open_http_request"
    ),
    "observability/decorators.py": "__enter__/get _finish/get",
    "observability/otel/exporter.py": "_send_with_retry/post",
    "observability/otel/transport.py": "post/open",
    "optimizers/interactive_optimizer.py": (
        "_is_better_result/get _metadata_with_tvar_observation/get "
        "_metadata_with_tvar_observation/get _metadata_with_tvar_observation/get"
    ),
    "projects/client.py": "_request_json_sync/open",
    "providers/validation.py": "_check_openrouter/get _openrouter_balance/get",
    "reporting/example_map.py": "resolve_map_target/get wire_rows_from_session/get",
    "runtime/client.py": (
        "create_requirement/request import_tvl/request list_requirements/request "
        "send/request"
    ),
    "runtime/hosted.py": "_default/send",
    "security/session_manager.py": (
        "_get_session/get _get_session/get _get_session_created_time/get "
        "_get_user_sessions/get _revoke_if_redis_session_expired/get "
        "_revoke_if_redis_session_expired/get revoke_session/delete validate_session/get "
        "validate_session/get validate_session/get"
    ),
    "snapshot/backends/github.py": (
        "_ensure_ready/_request _read/_request _require_branch/_request _send_once/send "
        "create_if_absent/_request"
    ),
    "storage/local_storage.py": (
        "_cached_result_in_session/get _cached_result_in_session/get "
        "_session_dataset_name/get _session_dataset_name/get"
    ),
    "utils/local_analytics.py": "_average_config_space_size/get",
    "utils/optimization_analyzer.py": (
        "_append_index_session/get _append_index_session/get _append_index_session/get "
        "_append_index_session/get get_run_summary/get get_run_summary/get "
        "get_run_summary/get"
    ),
    "utils/optimization_only_egress.py": "declared_for/get",
    "wrapper/service.py": "_touch_or_create_session/get",
}
