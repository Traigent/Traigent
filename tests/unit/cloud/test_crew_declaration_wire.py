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
    for path in folder.rglob("*"):
        if path.is_file():
            assert "7f3a91" not in path.read_text(encoding="utf-8", errors="ignore")


def _spans_and_baggage_clean() -> None:
    pytest.importorskip("opentelemetry.baggage")
    from opentelemetry import baggage

    assert not _has_canary({k: str(v) for k, v in baggage.get_all().items()})
    exporter = _SPANS.get("exporter")
    if exporter is not None:
        spans = exporter.get_finished_spans()
        assert not _has_canary([dict(s.attributes or {}) for s in spans])
        assert not _has_canary(
            [[dict(e.attributes or {}) for e in s.events] for s in spans]
        )


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
    return await ApiOperations(client).create_traigent_session_via_api(request)


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
    assert not _has_canary(caplog.text)
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_content_less_final_body_has_no_declaration(wire, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    await _create(_client(), _request(content_less=cl.ContentLessOptions(enabled=True)))
    body = wire[-1]
    assert "capabilities" in body  # premise: this is the content-less shape
    assert "crew_declaration" not in body["metadata"]
    assert not _has_canary(body)
    assert not _has_canary(caplog.text)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_optimization_only_final_body_has_no_declaration(wire, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    await _create(_client(optimization_only=True), _request())
    body = wire[-1]
    assert not _has_canary(body)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    assert not _has_canary(caplog.text)
    _sweep_files(tmp_path)
    _spans_and_baggage_clean()


@pytest.mark.asyncio
async def test_legacy_contract_final_body_has_no_declaration(
    wire, tmp_path, caplog, monkeypatch
):
    monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
    caplog.set_level(logging.DEBUG)
    await _create(_client(), _request())
    body = wire[-1]
    assert "problem_statement" in body  # premise: the legacy shape was sent
    assert not _has_canary(body)
    assert caplog.text.count(NOT_SENT_MESSAGE) == 1
    assert not _has_canary(caplog.text)
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
