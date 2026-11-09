# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Crew declarations at the real session-create boundary, per mode.

The crew and role keys are customer content. They may appear ONLY in the final HTTP
body of a standard (full) typed session create, as ``metadata.crew_declaration``.
Content-less, optimization_only and the legacy contract never send them. Each test
captures the final body at the transport and sweeps log records, spans, baggage, local
files and exception text for the canary.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud import content_less as cl
from traigent.cloud._aiohttp_compat import aiohttp
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.auth import AuthManager
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.models import SessionCreationRequest
from traigent.utils.crew_declaration import NOT_SENT_MESSAGE, CrewDeclaration

pytestmark = pytest.mark.backend_online

CANARY_CREW = "canary-crew-7f3a91"
CANARY_ROLE = "canary role 7f3a91 (triage)"
URL = "https://api.test"
KEY = "tg_" + "a" * 61
_PUBLIC_ADDRINFO = [(2, 1, 6, "", ("93.184.216.34", 0))]
CREW_ID = "crew_" + "ab" * 16


def _body() -> dict[str, Any]:
    return {
        "session_id": "s1",
        "status": "active",
        "metadata": {"experiment_id": "e", "experiment_run_id": "r"},
        "tenant_id": "tenant_" + "1" * 32,
        "project_id": "project_" + "2" * 32,
        "capabilities": ["content_less_v1"],
        "agent_ref": "agentref_" + "ab" * 16,
        "dataset_ref": "datasetref_" + "cd" * 16,
        "evaluator_ref": "evaluatorref_" + "ef" * 16,
        "crew_binding": {"status": "pending_portal", "crew_id": CREW_ID},
    }


class _Response:
    status = 201

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return _body()

    async def text(self) -> str:
        return ""


class _Wire:
    sent: list[Any] = []

    def __init__(self, *args: Any, headers: Any = None, **kwargs: Any) -> None:
        self.headers = dict(headers or {})

    async def __aenter__(self) -> _Wire:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def close(self) -> None:
        return None

    def post(self, url: Any, *args: Any, **kwargs: Any) -> _Response:
        _Wire.sent.append(kwargs.get("json"))
        return _Response()


async def _accept_key(self, api_key):  # noqa: ARG001 - backend validation boundary
    return None


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in os.environ if n.startswith("TRAIGENT_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    _Wire.sent = []
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setattr(aiohttp, "ClientSession", _Wire)
    monkeypatch.setattr(
        "traigent.labels.transport.send_session_labels", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", _accept_key)
    return _Wire.sent


def _client(**attrs: Any) -> BackendIntegratedClient:
    with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
        client = BackendIntegratedClient(api_key=KEY, base_url=URL)
    for name, value in attrs.items():
        setattr(client, name, value)
    return client


def _request(**overrides: Any) -> SessionCreationRequest:
    base: dict[str, Any] = {
        "function_name": "pkg.mod.fn (abc123)",
        "agent_key": "agent-x",
        "configuration_space": {"temperature": [0.1, 0.9]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 2, "name": "evalset-x"},
        "metadata": {"evaluation_set": "evalset-x"},
        "max_trials": 3,
        "crew_declaration": CrewDeclaration(CANARY_CREW, CANARY_ROLE),
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _has_canary(value: Any) -> bool:
    text = json.dumps(value, default=repr) if not isinstance(value, str) else value
    return CANARY_CREW in text or "7f3a91" in text


def _sweep_files(folder: Path) -> None:
    files = [path for path in folder.rglob("*") if path.is_file()]
    assert files, "no local result file was written: the sweep would prove nothing"
    for path in files:
        assert "7f3a91" not in path.read_text(encoding="utf-8", errors="ignore")


def _spans_and_baggage_clean() -> None:
    pytest.importorskip("opentelemetry.baggage")
    from opentelemetry import baggage

    assert not _has_canary({k: str(v) for k, v in baggage.get_all().items()})
    exporter = _SPANS["exporter"]
    spans = exporter.get_finished_spans()
    assert spans, "no span was exported: the sweep would prove nothing"
    assert not _has_canary([dict(s.attributes or {}) for s in spans])
    assert not _has_canary(
        [[dict(e.attributes or {}) for e in s.events] for s in spans]
    )


def _logs_clean(caplog: pytest.LogCaptureFixture) -> None:
    assert caplog.records, "no log record was captured: the sweep would prove nothing"
    assert not _has_canary(caplog.text)
    assert not _has_canary([r.getMessage() for r in caplog.records])


_SPANS: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _in_memory_spans(monkeypatch: pytest.MonkeyPatch):
    """Route the SDK tracer to an in-memory exporter when OpenTelemetry is installed."""
    _SPANS.clear()
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import traigent.core.tracing as core_tracing

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    if hasattr(core_tracing, "get_tracer"):
        monkeypatch.setattr(
            core_tracing, "get_tracer", lambda: provider.get_tracer("crew-wire")
        )
    _SPANS["exporter"] = exporter
    yield
    _SPANS.clear()


async def _create(client: Any, request: SessionCreationRequest):
    """Real session create inside a real SDK session span, then the local record.

    Guarantees every sink the sweeps read (span, log record, local file) is produced.
    """
    from traigent.core.tracing import optimization_session_span
    from traigent.storage.local_storage import LocalStorageManager

    with optimization_session_span(
        request.function_name,
        max_trials=request.max_trials,
        objectives=["accuracy"],
        config_space=request.configuration_space,
    ):
        result = await ApiOperations(client).create_traigent_session_via_api(request)
    logging.getLogger("traigent.test.crew_wire").info("session created")
    LocalStorageManager().create_session(
        request.function_name,
        optimization_config={"configuration_space": request.configuration_space},
        metadata={"evaluation_set": "evalset-x"},
        session_id=result[0],
    )
    return result


@pytest.mark.asyncio
async def test_full_mode_sends_only_in_metadata_and_exposes_binding(
    wire, tmp_path, caplog
):
    caplog.set_level(logging.DEBUG)
    result = await _create(_client(), _request())
    body = wire[-1]
    assert body["metadata"]["crew_declaration"] == {
        "crew_key": CANARY_CREW,
        "role_key": CANARY_ROLE,
    }
    elsewhere = {k: v for k, v in body.items() if k != "metadata"}
    elsewhere["metadata"] = {
        k: v for k, v in body["metadata"].items() if k != "crew_declaration"
    }
    assert not _has_canary(elsewhere)
    assert result.crew_binding.status == "pending_portal"
    _logs_clean(caplog)
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_content_less_final_body_has_no_declaration(wire, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    result = await _create(
        _client(), _request(content_less=cl.ContentLessOptions(enabled=True))
    )
    assert result.crew_binding.status == "unknown"  # backend said pending_portal
    body = wire[-1]
    assert "capabilities" in body  # premise: this is the content-less shape
    assert "crew_declaration" not in body["metadata"]
    assert not _has_canary(body)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    _logs_clean(caplog)
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_optimization_only_final_body_has_no_declaration(wire, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    result = await _create(_client(optimization_only=True), _request())
    assert result.crew_binding.status == "unknown"
    body = wire[-1]
    assert not _has_canary(body)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    _logs_clean(caplog)
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_legacy_contract_final_body_has_no_declaration(
    wire, tmp_path, caplog, monkeypatch
):
    monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
    caplog.set_level(logging.DEBUG)
    result = await _create(_client(), _request())
    assert result.crew_binding.status == "unknown"
    body = wire[-1]
    assert "problem_statement" in body  # premise: the legacy shape was sent
    assert not _has_canary(body)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    _logs_clean(caplog)
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_error_text_never_repeats_the_keys(wire, monkeypatch):
    class _Failing(_Response):
        status = 500

        async def text(self) -> str:
            return "server error"

    def post(self, url, *args, **kwargs):
        _Wire.sent.append(kwargs.get("json"))
        return _Failing()

    monkeypatch.setattr(_Wire, "post", post)
    with pytest.raises(Exception) as caught:  # noqa: PT011 - any failure type
        await _create(_client(), _request())
    assert not _has_canary(str(caught.value))
    assert not _has_canary(repr(caught.value))


def _direct_payload(request: SessionCreationRequest, **attrs: Any) -> dict[str, Any]:
    from traigent.cloud.client import TraigentCloudClient

    client = object.__new__(TraigentCloudClient)
    client._ensure_owner_metadata = lambda metadata: metadata or {}
    for name, value in attrs.items():
        setattr(client, name, value)
    return TraigentCloudClient._serialize_session_request(client, request)


def test_direct_client_full_mode_sends_declaration():
    payload = _direct_payload(_request())
    assert payload["metadata"]["crew_declaration"]["crew_key"] == CANARY_CREW


def test_direct_client_content_less_strips_before_the_body_is_built(caplog):
    caplog.set_level(logging.DEBUG)
    payload = _direct_payload(
        _request(content_less=cl.ContentLessOptions(enabled=True))
    )
    assert not _has_canary(payload)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1


def test_direct_client_optimization_only_strips(caplog):
    caplog.set_level(logging.DEBUG)
    payload = _direct_payload(_request(), optimization_only=True)
    assert not _has_canary(payload)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    assert not _has_canary(caplog.text)


@pytest.mark.asyncio
async def test_undeclared_request_ignores_a_backend_binding(wire):
    result = await _create(_client(), _request(crew_declaration=None))
    assert "crew_declaration" not in wire[-1]["metadata"]
    assert result.crew_binding.status == "unknown"


async def _direct_create(request: SessionCreationRequest, **attrs: Any):
    from traigent.cloud.client import TraigentCloudClient

    client: Any = object.__new__(TraigentCloudClient)
    client._ensure_owner_metadata = lambda metadata: metadata or {}
    client._raise_if_backend_egress_disabled = lambda *_a, **_k: None
    client._ensure_session = AsyncMock(return_value=None)
    client._get_headers = AsyncMock(return_value={})
    client.api_base_url = URL
    client._aio_session = _Wire()
    client._remember_session_owner = lambda *_a, **_k: None
    client._remember_cost_budget_armed_session = lambda *_a, **_k: None
    for name, value in attrs.items():
        setattr(client, name, value)
    return await TraigentCloudClient.create_optimization_session(client, request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_overrides", "attrs", "sent"),
    [
        ({}, {}, True),
        ({"content_less": cl.ContentLessOptions(enabled=True)}, {}, False),
        ({}, {"optimization_only": True}, False),
        ({"crew_declaration": None}, {}, False),
    ],
    ids=["full", "content_less", "optimization_only", "undeclared"],
)
async def test_direct_client_binding_follows_the_send_decision(
    wire, request_overrides, attrs, sent
):
    result = await _direct_create(_request(**request_overrides), **attrs)
    assert ("crew_declaration" in wire[-1]["metadata"]) is sent
    assert result.crew_binding.status == ("pending_portal" if sent else "unknown")


# ---------------------------------------------------------------------------
# End to end: a REAL ``@traigent.optimize(crew=..., role=...)`` run
# ---------------------------------------------------------------------------


class _E2EResponse(_Response):
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self._payload = payload

    async def json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)


class _E2EWire(_Wire):
    """Records every outgoing request of any verb; only session create is special."""

    requests: list[tuple[str, str, Any]] = []

    def _record(self, method: str, url: Any, kwargs: dict[str, Any]) -> _E2EResponse:
        body = kwargs.get("json", kwargs.get("data"))
        _E2EWire.requests.append((method, str(url), body))
        is_create = (
            method == "POST"
            and str(url).rstrip("/").endswith("/sessions")
            and isinstance(body, dict)
        )
        return _E2EResponse(201, _body()) if is_create else _E2EResponse(200, {})

    def post(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self._record("POST", url, kwargs)

    def put(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self._record("PUT", url, kwargs)

    def patch(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self._record("PATCH", url, kwargs)

    def get(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self._record("GET", url, kwargs)

    def delete(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self._record("DELETE", url, kwargs)


def _e2e_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    _E2EWire.requests = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_API_KEY", KEY)
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", URL)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setattr(aiohttp, "ClientSession", _E2EWire)
    monkeypatch.setattr(
        "traigent.labels.transport.send_session_labels", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", _accept_key)
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: _PUBLIC_ADDRINFO)
    return _E2EWire.requests


def _e2e_function(**extra: Any):
    import traigent
    from traigent.evaluators.base import Dataset, EvaluationExample

    dataset = Dataset(
        [EvaluationExample({"text": f"case-{i}"}, "ok") for i in range(2)],
        name="crew_canary_e2e",
    )

    @traigent.optimize(
        eval_dataset=dataset,
        objectives=["accuracy"],
        configuration_space={"x": ["a", "b"]},
        injection_mode="parameter",
        crew=CANARY_CREW,
        role=CANARY_ROLE,
        **extra,
    )
    async def answer(text: str, config=None) -> str:
        logging.getLogger("traigent.test.crew_e2e").info("evaluating %s", text)
        return "ok"

    return answer


def _span_texts() -> tuple[list[Any], list[Any]]:
    spans = _SPANS["exporter"].get_finished_spans()
    return (
        [
            [s.name, dict(s.attributes or {})]
            + [[e.name, dict(e.attributes or {})] for e in s.events]
            + [[dict(link.attributes or {}) for link in s.links]]
            for s in spans
        ],
        spans,
    )


def _assert_canary_only_in_declaration(
    requests: list[Any], tmp_path: Path, caplog, result: Any
) -> None:
    assert requests, "no outgoing HTTP request was captured"
    carriers = []
    for method, url, body in requests:
        assert not _has_canary(url)
        if isinstance(body, dict) and _has_canary(body):
            carriers.append((method, url, body))
    assert len(carriers) == 1, "the canary must ride exactly one request body"
    body = carriers[0][2]
    assert body["metadata"]["crew_declaration"] == {
        "crew_key": CANARY_CREW,
        "role_key": CANARY_ROLE,
    }
    rest = {k: v for k, v in body.items() if k != "metadata"}
    rest["metadata"] = {
        k: v for k, v in body["metadata"].items() if k != "crew_declaration"
    }
    assert not _has_canary(rest)

    texts, spans = _span_texts()
    assert spans, "the run exported no span: the sweep would prove nothing"
    assert not _has_canary(texts)
    pytest.importorskip("opentelemetry.baggage")
    from opentelemetry import baggage

    assert not _has_canary({k: str(v) for k, v in baggage.get_all().items()})
    _logs_clean(caplog)
    assert not _has_canary([getattr(r, "__dict__", {}) for r in caplog.records])
    _sweep_files(tmp_path)
    assert not _has_canary(repr(result))
    assert not _has_canary(str(result))


@pytest.mark.asyncio
async def test_real_run_keeps_the_canary_on_the_declared_path(
    monkeypatch, tmp_path, caplog
):
    requests = _e2e_env(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    func = _e2e_function()
    result = await func.optimize(algorithm="grid", max_trials=2)
    assert len(result.trials) == 2
    assert func.crew_binding.status == "pending_portal"
    _assert_canary_only_in_declaration(requests, tmp_path, caplog, result)


@pytest.mark.asyncio
async def test_real_content_less_run_sends_the_canary_nowhere(
    monkeypatch, tmp_path, caplog
):
    requests = _e2e_env(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    func = _e2e_function(content_less=cl.ContentLessOptions(enabled=True))
    result = await func.optimize(algorithm="grid", max_trials=2)
    assert requests, "no outgoing HTTP request was captured"
    assert any(
        isinstance(body, dict) and "capabilities" in body for _, _, body in requests
    ), "premise: a content-less session create reached the transport"
    for _, url, body in requests:
        assert not _has_canary(url)
        assert not _has_canary(body)
    assert func.crew_binding.status == "unknown"
    texts, spans = _span_texts()
    assert spans
    assert not _has_canary(texts)
    _logs_clean(caplog)
    _sweep_files(tmp_path)
    assert not _has_canary(repr(result))


@pytest.mark.asyncio
async def test_privacy_run_sends_the_canary_nowhere(monkeypatch, tmp_path, caplog):
    """Real optimize() in privacy mode: no crew declaration on any outgoing body."""
    requests = _e2e_env(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    # The decorator kwarg is dropped before it reaches the config today (see
    # test_backend_session_manager), so set the state the manager sees in privacy
    # mode on the real config object; everything downstream is the real path.
    from traigent.core.backend_session_manager import BackendSessionManager

    real_init = BackendSessionManager.__init__

    def privacy_init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self._traigent_config.privacy_enabled = True

    monkeypatch.setattr(BackendSessionManager, "__init__", privacy_init)
    func = _e2e_function()
    result = await func.optimize(algorithm="grid", max_trials=2)
    creates = [
        body
        for method, url, body in requests
        if method == "POST" and url.rstrip("/").endswith("/sessions")
    ]
    assert creates, "premise: a session create reached the transport"
    for body in creates:
        assert "crew_declaration" not in (body.get("metadata") or {})
    for _, url, body in requests:
        assert not _has_canary(url)
        assert not _has_canary(body)
        assert not _has_canary(json.dumps(body, default=repr).encode().decode())
    assert func.crew_binding.status == "unknown"
    assert NOT_SENT_MESSAGE in caplog.text
    _logs_clean(caplog)
    assert not _has_canary(repr(result))


@pytest.mark.asyncio
async def test_privacy_off_control_does_carry_the_declaration(
    monkeypatch, tmp_path, caplog
):
    """Negative control: the same run without privacy mode DOES send it."""
    requests = _e2e_env(monkeypatch, tmp_path)
    caplog.set_level(logging.DEBUG)
    func = _e2e_function()
    await func.optimize(algorithm="grid", max_trials=2)
    creates = [
        body
        for method, url, body in requests
        if method == "POST" and url.rstrip("/").endswith("/sessions")
    ]
    assert creates
    assert any(
        (b.get("metadata") or {}).get("crew_declaration")
        == {"crew_key": CANARY_CREW, "role_key": CANARY_ROLE}
        for b in creates
    )


# ---------------------------------------------------------------------------
# Redirects: a declaration-bearing POST is never followed (aiohttp keeps the
# body on 307/308); other requests keep the default redirect behaviour.
# ---------------------------------------------------------------------------

REDIRECT_TEXT = "redirect-body-canary-5c2e"


class _RedirectResponse(_Response):
    def __init__(self, status: int) -> None:
        self.status = status

    async def text(self) -> str:
        return REDIRECT_TEXT


class _RedirectWire(_Wire):
    status = 307
    options: list[dict[str, Any]] = []

    def post(self, url: Any, *args: Any, **kwargs: Any) -> _Response:
        _Wire.sent.append(kwargs.get("json"))
        _RedirectWire.options.append(
            {k: v for k, v in kwargs.items() if k == "allow_redirects"}
        )
        if kwargs.get("allow_redirects", True) is False:
            return _RedirectResponse(_RedirectWire.status)
        return _Response()  # redirect followed, final answer is a 201


def _redirect_clean(exc: BaseException, caplog: pytest.LogCaptureFixture) -> None:
    text = str(exc) + repr(exc) + "".join(r.getMessage() for r in caplog.records)
    assert CANARY_CREW not in text
    assert CANARY_ROLE not in text
    assert REDIRECT_TEXT not in text
    assert "redirected; request refused" in str(exc)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("path", ["api_operations", "direct_client"])
async def test_declaration_post_refuses_redirects(
    wire, monkeypatch, caplog, status, path
):
    from traigent.cloud.client import CloudServiceError  # noqa: PLC0415

    _RedirectWire.status = status
    _RedirectWire.options = []
    monkeypatch.setattr(aiohttp, "ClientSession", _RedirectWire)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(CloudServiceError) as info:
        if path == "api_operations":
            await _create(_client(), _request())
        else:
            await _direct_create(_request(), _aio_session=_RedirectWire())
    assert len(wire) == 1
    assert _RedirectWire.options == [{"allow_redirects": False}]
    assert sum(_has_canary(b) for b in wire) == 1  # only the first, refused send
    _redirect_clean(info.value, caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["api_operations", "direct_client"])
async def test_request_without_declaration_keeps_default_redirects(
    wire, monkeypatch, path
):
    _RedirectWire.status = 307
    _RedirectWire.options = []
    monkeypatch.setattr(aiohttp, "ClientSession", _RedirectWire)
    request = _request(crew_declaration=None)
    if path == "api_operations":
        result = await _create(_client(), request)
    else:
        result = await _direct_create(request, _aio_session=_RedirectWire())
    assert _RedirectWire.options == [{}]
    assert result.crew_binding.status == "unknown"


# ---------------------------------------------------------------------------
# Raw metadata value and per-request optimization-only override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["api_operations", "direct_client"])
async def test_raw_metadata_declaration_without_crew_is_not_sent(wire, path):
    request = _request(
        crew_declaration=None,
        metadata={
            "evaluation_set": "evalset-x",
            "crew_declaration": {"crew_key": CANARY_CREW, "role_key": CANARY_ROLE},
        },
    )
    if path == "api_operations":
        result = await _create(_client(), request)
    else:
        result = await _direct_create(request)
    assert not any(_has_canary(b) for b in wire)
    assert result.crew_binding.status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["api_operations", "direct_client"])
async def test_per_request_optimization_only_override_strips_declaration(
    wire, monkeypatch, path
):
    from traigent.utils import optimization_only_egress as oo

    real = oo.policy_of

    def per_request(client: Any, headers: Any = None, url: Any = None) -> Any:
        if headers is not None or url is not None:  # only the send-time check
            return oo.Policy(True, None)
        return real(client)

    monkeypatch.setattr(oo, "policy_of", per_request)
    if path == "api_operations":
        result = await _create(_client(), _request())
    else:
        result = await _direct_create(
            _request(), _get_headers=AsyncMock(return_value={"X": "y"})
        )
    assert wire
    assert not any(_has_canary(b) for b in wire)
    assert result.crew_binding.status == "unknown"


def test_legacy_session_body_drops_a_raw_crew_declaration_in_metadata():
    """A direct caller's raw metadata value never crosses on the legacy path."""
    raw = {"crew_key": CANARY_CREW, "role_key": CANARY_ROLE}
    request = _request(
        metadata={"evaluation_set": "evalset-x", "crew_declaration": raw},
        crew_declaration=None,
        objectives=[{"metric": "accuracy", "orientation": "maximize"}],
    )
    ops = ApiOperations(_client())
    payload = ops._build_legacy_session_payload(request, 3)
    assert "crew_declaration" not in payload["metadata"]
    assert payload["metadata"]["evaluation_set"] == "evalset-x"
    assert not _has_canary(json.dumps(payload, default=repr))
