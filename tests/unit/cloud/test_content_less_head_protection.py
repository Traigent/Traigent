"""Content-less promotion head protection (phase 0a), SDK side.

A ``content_less_v1`` session is addressed by the opaque ``agent_ref`` of its create 201
(never ``metadata.agent_id``). The run captures the head generation by ref at its start
and a later publish sends ``X-Traigent-Agent-Ref`` plus
``X-Traigent-Expected-Head-Generation``. Every closed refusal maps to a typed exception,
``head_generation_stale`` is never retried or refreshed, a missing capture refuses
locally (no unprotected fallback), and legacy sessions behave exactly as before.

Contract: TraigentSchema ``optimization/best_config_head_endpoints.json``,
``agent_head_by_ref_response_schema.json``, ``best_config_head_refusal_schema.json``,
``execution/best_config_endpoints.json`` (POST decision table).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import requests

import traigent
from traigent.api.types import OptimizationResult
from traigent.cloud import content_less as cl
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.best_config_runtime import (
    CloudBestConfigHeadRefusedError,
    CloudBestConfigHeadStaleError,
    CloudBestConfigStaleHeadError,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
    HeadRefusalCode,
    canonical_json,
    compute_spec_hash,
    sha256_digest,
)
from traigent.core.session_types import SessionCreationResult
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.labels.transport_gate import TransportGateRefusal

pytestmark = pytest.mark.backend_online

AGENT_REF = "agentref_" + "ab" * 16
OTHER_REF = "agentref_" + "cd" * 16
CANARY_AGENT_ID = "cnryAgentId_Rn8F-3hSx1Qj5"
CANARY_TEXT = "cnryBody_Wt2G-6kYd9Vb3"
_SPACE = {"temperature": [0.0, 0.5]}


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"text": "a"}, "ok"),
            EvaluationExample({"text": "b"}, "ok"),
        ],
        name="content_less_head",
    )


# ------------------------------------------------------------ session-create capture


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def json(self) -> dict[str, Any]:
        return self._body


def _created(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": "s1",
        "status": "active",
        "metadata": {"experiment_id": "e", "experiment_run_id": "r"},
        "tenant_id": "tenant_" + "1" * 32,
        "project_id": "project_" + "2" * 32,
        "capabilities": ["content_less_v1"],
        "agent_ref": AGENT_REF,
    }
    body.update(extra)
    return body


def _parse(body: dict[str, Any], *, content_less: bool, tmp_path: Path) -> Any:
    ops = ApiOperations(MagicMock())
    ctx = (cl.RefCache(tmp_path / "refs.json"), "p", {}, {}) if content_less else None
    return asyncio.run(
        ops._parse_session_response(_Response(body), content_less_ctx=ctx)
    )


def test_content_less_create_captures_agent_ref_and_never_reads_metadata_agent_id(
    tmp_path: Path,
) -> None:
    body = _created()
    body["metadata"]["agent_id"] = CANARY_AGENT_ID
    parsed = _parse(body, content_less=True, tmp_path=tmp_path)
    assert parsed.agent_ref == AGENT_REF
    assert parsed.content_less is True
    assert parsed.agent_id is None


def test_content_less_create_without_agent_ref_is_still_content_less(
    tmp_path: Path,
) -> None:
    body = _created()
    del body["agent_ref"]
    parsed = _parse(body, content_less=True, tmp_path=tmp_path)
    assert parsed.content_less is True and parsed.agent_ref is None


def test_legacy_create_still_reads_metadata_agent_id(tmp_path: Path) -> None:
    body = {
        "session_id": "s1",
        "metadata": {
            "experiment_id": "e",
            "experiment_run_id": "r",
            "agent_id": "agent-7",
        },
    }
    parsed = _parse(body, content_less=False, tmp_path=tmp_path)
    assert parsed.agent_id == "agent-7"
    assert parsed.agent_ref is None
    assert parsed.content_less is False


# --------------------------------------------- run -> publish through the real wiring


class _FakeBackend:
    """Tracking client with a ref-addressed compare-and-set head (content-less)."""

    def __init__(
        self,
        *,
        agent_ref: str | None = AGENT_REF,
        head: int = 3,
        by_ref: bool = True,
        content_less: bool = True,
    ) -> None:
        self.agent_ref = agent_ref
        self.head = head
        self.content_less = content_less
        self.ref_reads: list[tuple[str, str | None]] = []
        self.legacy_reads = 0
        self.publishes: list[dict[str, Any]] = []
        self.no_egress = False
        self.cloud_egress_intent = False
        self.enable_fallback = False
        self.local_storage = None
        auth = Mock()
        auth.has_api_key = Mock(return_value=True)
        self.auth_manager = auth
        self.auth = auth
        self.submit_result = Mock()
        self.read_error: Exception | None = None
        self.bindings: dict[str, dict[str, str]] = {}
        self.create_calls = 0
        if not by_ref:
            self.fetch_agent_head_generation_by_ref_sync = None  # type: ignore[assignment]

    def create_session(self, function_name, search_space, optimization_goal, **kw):
        self.create_calls += 1
        if self.content_less:
            # The Backend stores the environment of the create body (optimization_identity
            # ._environment): absent -> "default", a member of SessionEnvironment, else
            # the create is refused. The body is built by the SDK's own payload rewrite.
            body: dict[str, Any] = {}
            cl.apply_to_payload(body, options=kw.get("content_less"), cached_refs={})
            env = body.get("environment")
            if env is None:
                env = "default"
            if env not in cl.ENVIRONMENTS:
                raise CloudServiceError("session create refused")
            self.bindings["bs-1"] = {
                "agent_ref": self.agent_ref or AGENT_REF,
                "environment": env,
            }
            return SessionCreationResult.connected(
                session_id="bs-1", agent_ref=self.agent_ref, content_less=True
            )
        return SessionCreationResult.connected(session_id="bs-1", agent_id="agent-7")

    def get_session_mapping(self, session_id):
        return SimpleNamespace(experiment_id="exp-1", experiment_run_id="run-1")

    def request_trial_slot(self, session_id):
        return SimpleNamespace(trial_id=None, optimization_complete=True, reason="done")

    def _submit_trial_result_via_session(self, **kwargs):
        return True

    def fetch_agent_head_generation_by_ref_sync(self, agent_ref, *, environment):
        self.ref_reads.append((agent_ref, environment))
        if self.read_error is not None:
            raise self.read_error
        return self.head

    def fetch_agent_head_generation_sync(self, agent_id, *, environment=None):
        self.legacy_reads += 1
        return self.head

    def fetch_best_config_sync(self, config_id, **kwargs):
        return None

    def publish_best_config_sync(
        self,
        spec,
        *,
        environment=None,
        if_match=None,
        agent_id=None,
        expected_head_generation=None,
        agent_ref=None,
    ):
        provenance = (spec.get("provenance") or {}) if isinstance(spec, dict) else {}
        self.publishes.append(
            {
                "agent_ref": agent_ref,
                "agent_id": agent_id,
                "expected": expected_head_generation,
                "environment": environment,
                "session_id": provenance.get("source_session_id"),
                "spec_environment": spec.get("environment"),
            }
        )
        if agent_ref is not None:
            # Backend best_config_routes._protected_publish_precondition: the binding is
            # resolved from the stored session named in spec.provenance, never the body.
            sid = provenance.get("source_session_id") or provenance.get("session_id")
            binding = self.bindings.get(sid) if isinstance(sid, str) else None
            if binding is None:  # row 4: no resolvable content-less session binding
                raise CloudBestConfigHeadRefusedError(
                    HeadRefusalCode.HEAD_CAPTURE_INVALID, status_code=400
                )
            if binding["agent_ref"] != agent_ref:
                raise CloudBestConfigHeadRefusedError(
                    HeadRefusalCode.HEAD_AGENT_MISMATCH, status_code=409
                )
            requested = (
                (spec.get("environment") or environment or "default").strip().lower()
            )
            if requested != binding["environment"]:
                raise CloudBestConfigHeadRefusedError(
                    HeadRefusalCode.HEAD_ENVIRONMENT_MISMATCH, status_code=409
                )
        if expected_head_generation is not None:
            if expected_head_generation != self.head:
                if agent_ref is not None:
                    raise CloudBestConfigHeadStaleError()
                raise CloudBestConfigStaleHeadError(
                    current_generation=self.head,
                    expected_generation=expected_head_generation,
                )
            self.head += 1
        return {
            "config_id": spec["config_id"],
            "version": 1,
            "spec": spec,
            "spec_hash": compute_spec_hash(spec),
            "config_hash": sha256_digest(canonical_json(spec["config"])),
        }


@pytest.fixture
def online(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-head")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    for name in (
        "TRAIGENT_ENVIRONMENT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "TRAIGENT_BEST_CONFIG_ENVIRONMENT",
        "TRAIGENT_AGENT_REF",
    ):
        monkeypatch.delenv(name, raising=False)


def _make_wrapper(best_config_environment=None, **decorator_kwargs):
    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space=_SPACE,
        injection_mode="parameter",
        **decorator_kwargs,
    )
    def answer(text: str, config) -> str:
        return "ok"

    if best_config_environment is not None:
        # The decorator does not expose the argument; the state manager is what
        # OptimizedFunction(best_config_environment=...) feeds, so build it the same way.
        from traigent.core.config_state_manager import ConfigStateManager as csm

        answer._csm.best_config_environment = csm._resolve_best_config_environment(
            best_config_environment
        )
        answer._csm.best_config_environment_explicit = (
            csm._best_config_environment_is_explicit(best_config_environment)
        )
    return answer


async def _run(fake: _FakeBackend, best_config_environment=None, **decorator_kwargs):
    answer = _make_wrapper(best_config_environment, **decorator_kwargs)
    with (
        patch.object(
            BackendSessionManager,
            "create_backend_client",
            staticmethod(lambda _c: fake),
        ),
        patch("traigent.optimizers.interactive_optimizer.InteractiveOptimizer"),
        patch("traigent.cloud.client.TraigentCloudClient"),
    ):
        result = await answer.optimize(algorithm="grid")
    assert isinstance(result, OptimizationResult) and result.best_config
    return answer, result


def _publish(answer, fake):
    with patch("traigent.cloud.backend_client.get_backend_client", return_value=fake):
        return answer.publish_best_config()


@pytest.mark.asyncio
async def test_run_captures_ref_environment_and_generation_at_start(online):
    fake = _FakeBackend(head=3)
    _, result = await _run(fake)
    assert result.metadata["promotion_precondition"] == {
        "protected": True,
        "agent_ref": AGENT_REF,
        "environment": "default",
        "generation": 3,
        "session_id": "bs-1",
    }
    assert fake.ref_reads == [(AGENT_REF, "default")]
    assert fake.legacy_reads == 0  # the agent-id head read is never used


@pytest.mark.asyncio
async def test_publish_sends_the_ref_and_the_captured_generation(online):
    fake = _FakeBackend(head=3)
    answer, _ = await _run(fake)
    _publish(answer, fake)
    assert fake.publishes == [
        {
            "agent_ref": AGENT_REF,
            "agent_id": None,
            "expected": 3,
            "environment": "default",
            "session_id": "bs-1",  # the Backend resolves the binding from this
            "spec_environment": "default",
        }
    ]
    assert fake.head == 4
    assert len(fake.ref_reads) == 1  # publishing did not re-read the head


@pytest.mark.asyncio
async def test_stale_publish_surfaces_once_without_refresh_or_retry(online):
    fake = _FakeBackend(head=3)
    answer, result = await _run(fake)
    fake.head = 4  # another writer promoted while the run was in flight

    with pytest.raises(CloudBestConfigHeadStaleError) as excinfo:
        _publish(answer, fake)

    err = excinfo.value
    assert err.code is HeadRefusalCode.HEAD_GENERATION_STALE
    assert err.reason is CloudPublishUnavailableReason.STALE_HEAD_GENERATION
    assert isinstance(err, CloudBestConfigHeadRefusedError)
    assert isinstance(err, CloudPublishUnavailable)
    assert len(fake.publishes) == 1  # not retried
    assert fake.publishes[0]["expected"] == 3  # the captured value, not a refreshed 4
    assert len(fake.ref_reads) == 1  # the head was not re-read to "cure" the refusal
    assert fake.head == 4
    assert err.best_config == result.best_config


@pytest.mark.asyncio
async def test_no_agent_ref_refuses_the_publish_locally(online):
    fake = _FakeBackend(agent_ref=None)
    answer, result = await _run(fake)
    assert result.metadata["promotion_precondition"] == {
        "protected": True,
        "capture_refusal": "capture_missing",
    }
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _publish(answer, fake)
    assert excinfo.value.code is HeadRefusalCode.CAPTURE_MISSING
    assert excinfo.value.status_code is None
    assert excinfo.value.reason is CloudPublishUnavailableReason.HEAD_CAPTURE_MISSING
    assert fake.publishes == []  # nothing was sent, protected or not
    assert excinfo.value.best_config == result.best_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (CloudServiceError("boom"), HeadRefusalCode.CAPTURE_MISSING),
        (
            CloudBestConfigHeadRefusedError(HeadRefusalCode.REFERENCE_NOT_FOUND),
            HeadRefusalCode.REFERENCE_NOT_FOUND,
        ),
        (
            CloudBestConfigHeadRefusedError(HeadRefusalCode.HEAD_CAPTURE_INVALID),
            HeadRefusalCode.HEAD_CAPTURE_INVALID,
        ),
    ],
    ids=["transport", "ref-unknown", "capture-invalid"],
)
async def test_failed_head_read_refuses_the_publish_locally(online, error, code):
    fake = _FakeBackend()
    fake.read_error = error
    answer, _ = await _run(fake)  # the run itself still completes
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _publish(answer, fake)
    assert excinfo.value.code is code
    assert fake.publishes == []


@pytest.mark.asyncio
async def test_client_without_a_by_ref_read_refuses_the_publish_locally(online):
    fake = _FakeBackend(by_ref=False)
    answer, _ = await _run(fake)
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _publish(answer, fake)
    assert excinfo.value.code is HeadRefusalCode.CAPTURE_MISSING
    assert fake.publishes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "canonical"),
    [
        ("staging", "staging"),
        ("production", "production"),
        ("prod", "production"),  # accepted alias
        ("PROD", "production"),
        ("dev", "development"),  # accepted alias
        ("development", "development"),
        ("test", "test"),
        ("testing", "test"),  # accepted alias
        ("stage", "staging"),  # accepted alias
        ("stg", "staging"),  # accepted alias
        ("default", "default"),
    ],
)
async def test_run_environment_is_one_value_from_session_create_to_publish(
    online, monkeypatch, configured, canonical
):
    monkeypatch.setenv("TRAIGENT_BEST_CONFIG_ENVIRONMENT", configured)
    fake = _FakeBackend()
    answer, result = await _run(fake)
    assert fake.bindings["bs-1"]["environment"] == canonical  # what the Backend stored
    assert fake.ref_reads == [(AGENT_REF, canonical)]
    assert result.metadata["promotion_precondition"]["environment"] == canonical

    _publish(answer, fake)

    assert [(p["environment"], p["spec_environment"]) for p in fake.publishes] == [
        (canonical, canonical)
    ]
    assert fake.head == 4  # the fake's binding check passed end to end


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("option", "best_config_arg", "best_config_env", "run_env", "expected"),
    [
        # explicit content-less option beats everything
        ("production", "staging", "development", "test", "production"),
        ("default", "staging", None, "production", "default"),
        # explicitly set best-config environment beats TRAIGENT_ENVIRONMENT,
        # an explicit "default" included
        (None, "default", None, "production", "default"),
        (None, None, "default", "production", "default"),
        (None, "staging", None, "production", "staging"),
        (None, None, "staging", "production", "staging"),
        (None, "default", None, None, "default"),
        # nothing set for the best-config environment: the other sources, then default
        (None, None, None, "production", "production"),
        (None, None, None, None, "default"),
    ],
    ids=[
        "option-beats-all",
        "explicit-default-option-beats-env",
        "explicit-default-arg-beats-env",
        "explicit-default-bc-env-beats-env",
        "arg-beats-env",
        "bc-env-beats-env",
        "explicit-default-no-env",
        "env-when-bc-unset",
        "default-when-nothing-set",
    ],
)
async def test_environment_precedence_is_option_then_best_config_then_env_then_default(
    online, monkeypatch, option, best_config_arg, best_config_env, run_env, expected
):
    if best_config_env is not None:
        monkeypatch.setenv("TRAIGENT_BEST_CONFIG_ENVIRONMENT", best_config_env)
    if run_env is not None:
        monkeypatch.setenv("TRAIGENT_ENVIRONMENT", run_env)
    kwargs: dict[str, Any] = {"best_config_environment": best_config_arg}
    if option is not None:
        kwargs["environment"] = (
            option  # @optimize(environment=...), the content-less option
        )
    fake = _FakeBackend()
    answer, result = await _run(fake, **kwargs)
    assert fake.bindings["bs-1"]["environment"] == expected
    assert result.metadata["promotion_precondition"]["environment"] == expected
    _publish(answer, fake)
    assert [p["environment"] for p in fake.publishes] == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["canary", "prod-eu", "Production Eu", "default2"])
async def test_environment_outside_the_closed_set_refuses_before_session_create(
    online, monkeypatch, bad
):
    monkeypatch.setenv("TRAIGENT_BEST_CONFIG_ENVIRONMENT", bad)
    fake = _FakeBackend()
    with pytest.raises(cl.ContentLessRefusal) as excinfo:
        await _run(fake)
    assert excinfo.value.code == cl.INVALID_ENVIRONMENT
    assert bad not in str(excinfo.value)
    assert fake.create_calls == 0 and fake.ref_reads == [] and fake.publishes == []


@pytest.mark.asyncio
async def test_publish_uses_the_session_bound_environment_not_a_later_change(online):
    fake = _FakeBackend()
    answer, _ = await _run(fake)
    answer._csm.best_config_environment = "production"  # changed after the run
    _publish(answer, fake)
    assert fake.publishes[0]["environment"] == "default"
    assert fake.publishes[0]["spec_environment"] == "default"


@pytest.mark.asyncio
async def test_publish_without_a_session_id_is_refused_locally(online):
    fake = _FakeBackend()
    answer, _ = await _run(fake)
    answer._csm._best_config_precondition.pop("session_id")  # as if never captured
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _publish(answer, fake)
    assert excinfo.value.code is HeadRefusalCode.CAPTURE_MISSING
    assert fake.publishes == []  # refused locally; the SDK never omits the id


def test_fake_enforces_row_4_when_the_spec_names_no_session():
    fake = _FakeBackend()
    spec = {"config_id": "c", "environment": "default", "config": {}, "provenance": {}}
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        fake.publish_best_config_sync(
            spec, environment="default", agent_ref=AGENT_REF, expected_head_generation=3
        )
    assert excinfo.value.code is HeadRefusalCode.HEAD_CAPTURE_INVALID
    assert excinfo.value.status_code == 400
    assert fake.head == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "code", "status"),
    [
        ("agent_ref", HeadRefusalCode.HEAD_AGENT_MISMATCH, 409),
        ("environment", HeadRefusalCode.HEAD_ENVIRONMENT_MISMATCH, 409),
        ("unknown_session", HeadRefusalCode.HEAD_CAPTURE_INVALID, 400),
    ],
)
async def test_session_binding_disagreeing_with_the_headers_is_refused(
    online, change, code, status
):
    fake = _FakeBackend()
    answer, _ = await _run(fake)
    if change == "unknown_session":
        fake.bindings.clear()
    else:
        fake.bindings["bs-1"][change] = (
            OTHER_REF if change == "agent_ref" else "production"
        )
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _publish(answer, fake)
    assert (excinfo.value.code, excinfo.value.status_code) == (code, status)
    assert fake.head == 3  # nothing written


@pytest.mark.asyncio
async def test_legacy_session_is_unchanged(online, monkeypatch):
    monkeypatch.delenv("TRAIGENT_CONTENT_LESS")
    fake = _FakeBackend(content_less=False)
    answer, result = await _run(fake)
    assert result.metadata["promotion_precondition"] == {
        "agent_id": "agent-7",
        "environment": "default",
        "generation": 3,
    }
    assert fake.ref_reads == [] and fake.legacy_reads == 1
    _publish(answer, fake)
    assert fake.publishes == [
        {
            "agent_ref": None,
            "agent_id": "agent-7",
            "expected": 3,
            "environment": "default",
            "session_id": None,
            "spec_environment": "default",
        }
    ]


# ------------------------------------------------------------- HTTP wire (client)

_KEY = "tg_" + "x" * 61  # pragma: allowlist secret
_SPEC = {"config_id": "answerer", "environment": "staging", "config": {"t": 1}}


class _Wire:
    """Replaces the HTTP adapter's send: records requests, replays queued responses."""

    def __init__(self, *responses: tuple[int, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[requests.PreparedRequest] = []

    def send(self, request):
        self.requests.append(request)
        status, body = self.responses.pop(0) if self.responses else (500, {})
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body).encode()
        response.headers["Content-Type"] = "application/json"
        if isinstance(body, dict) and "_location" in body:
            response.headers["Location"] = body["_location"]
        response.url = request.url
        response.request = request
        return response


#: A public (global) address: ``api.test`` must not need real DNS.
_PUBLIC_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def _client() -> BackendIntegratedClient:
    # Root cause of the old canary split: ``online`` unsets the environment markers (so
    # run-environment precedence is deterministic), which removes the autouse
    # ``TRAIGENT_ENVIRONMENT=development`` of tests/conftest.py. Without a development
    # marker url_security fails closed (production assumed), the unresolvable host
    # ``api.test`` raises CloudUrlUnreachableError, the client is marked ``_url_invalid``
    # and every call says "backend egress is disabled (offline)". Resolve the host here
    # so construction never depends on ambient DNS or ambient environment markers.
    with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
        client = BackendIntegratedClient(api_key=_KEY, base_url="https://api.test")
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"Authorization": "Bearer test-token"}
    )
    return client


def _wired(*responses: tuple[int, Any]):
    wire = _Wire(*responses)

    def adapter_send(adapter, request, **kwargs):
        return wire.send(request)

    return wire, patch.object(requests.adapters.HTTPAdapter, "send", adapter_send)


_FIXED_TEXT = {
    "reference_not_found": ("reference_not_found", "Reference not found."),
    "head_capture_invalid": (
        "invalid_request",
        "Head capture is missing or malformed.",
    ),
    "head_environment_mismatch": (
        "head_precondition_refused",
        "Environment does not match the session.",
    ),
    "head_agent_mismatch": (
        "head_precondition_refused",
        "Agent does not match the session.",
    ),
    "head_generation_stale": (
        "head_precondition_refused",
        "Agent head moved since the capture.",
    ),
    "head_write_unauthorized": (
        "head_precondition_refused",
        "Not authorized to promote this agent.",
    ),
    "head_body_invalid": ("invalid_request", "Publish body is invalid."),
    "head_publish_conflict": (
        "head_precondition_refused",
        "Publish conflicts with the current configuration.",
    ),
    "unauthenticated": ("unauthenticated", "Authentication required."),
    "forbidden": ("forbidden", "Forbidden."),
    "internal_error": ("internal_error", "Internal error."),
}
_STATUS = {
    "reference_not_found": 404,
    "head_capture_invalid": 400,
    "head_environment_mismatch": 409,
    "head_agent_mismatch": 409,
    "head_generation_stale": 409,
    "head_write_unauthorized": 403,
    "head_body_invalid": 400,
    "head_publish_conflict": 409,
    "unauthenticated": 401,
    "forbidden": 403,
    "internal_error": 500,
}


def _refusal(code: str) -> dict[str, Any]:
    error, message = _FIXED_TEXT[code]
    return {"success": False, "error": error, "error_code": code, "message": message}


_CONTRACT_DIR = (
    Path(__file__).resolve().parents[2] / "fixtures/contracts/schema_78ae067"
)
#: Byte-for-byte copies of TraigentSchema 78ae067 (traigent_schema/schemas/{optimization,execution}/).
_CONTRACT_SHA256 = {
    "agent_head_by_ref_response_schema.json": "878090557b2944fe13802422242104d7f801484a2e54c86bea2809612b518a61",
    "best_config_endpoints.json": "21684d6da7deeb97bf2301cc8787ada01b7a96d61cb0aa628952716249eaa4bb",
    "best_config_head_endpoints.json": "61c2423fda68c7b4faada5aa4cf0d2d26a6a4443839ebbf43eaa6afa9ad5d92a",
    "best_config_head_refusal_schema.json": "b93dccaeab1cfc056b9715e4723641a80697c649a8bb4675a92d043b0558ea8e",
    "best_config_publish_409_schema.json": "3fd219a7254e7b1897683243c37ab99f4cf01c743fb6411d97aa30ebd7d3b380",
    "best_config_publish_500_schema.json": "45537ee74004ebb2749b66ed1f99950218606b4a9895de6738f67753c7da4cfa",
}


def _contract(name: str) -> Any:
    return json.loads((_CONTRACT_DIR / name).read_text())


def test_vendored_contract_files_are_the_pinned_schema_bytes():
    import hashlib

    for name, digest in _CONTRACT_SHA256.items():
        assert hashlib.sha256((_CONTRACT_DIR / name).read_bytes()).hexdigest() == digest


def test_sdk_code_table_matches_the_vendored_refusal_schema():
    from traigent.cloud.backend_client import _HEAD_REFUSAL_ENVELOPE

    refusal = _contract("best_config_head_refusal_schema.json")
    codes = refusal["properties"]["error_code"]["enum"]
    assert set(_HEAD_REFUSAL_ENVELOPE) == set(codes) == set(_STATUS) == set(_FIXED_TEXT)
    assert set(_HEAD_REFUSAL_ENVELOPE) == {c.value for c in HeadRefusalCode} - {
        "unrecognized",
        "capture_missing",
    }
    for branch in refusal["oneOf"]:
        props = branch["properties"]
        _status, error, message = _HEAD_REFUSAL_ENVELOPE[props["error_code"]["const"]]
        assert (error, message) == (props["error"]["const"], props["message"]["const"])
    assert len(refusal["oneOf"]) == len(codes)
    assert refusal["additionalProperties"] is False
    assert set(refusal["required"]) == {"success", "error", "error_code", "message"}


def test_sdk_status_pairing_matches_the_vendored_endpoint_contract():
    from traigent.cloud.backend_client import _HEAD_REFUSAL_ENVELOPE

    publish = _contract("best_config_endpoints.json")["paths"]["/api/v1/best-configs"][
        "post"
    ]
    responses = publish["responses"]
    s409 = _contract("best_config_publish_409_schema.json")
    s500 = _contract("best_config_publish_500_schema.json")
    by_status: dict[int, set[str]] = {}
    for code, (status, _e, _m) in _HEAD_REFUSAL_ENVELOPE.items():
        by_status.setdefault(status, set()).add(code)
    assert by_status[409] == set(s409["allOf"][1]["properties"]["error_code"]["enum"])
    assert by_status[500] == set(s500["allOf"][1]["properties"]["error_code"]["enum"])
    for status in (400, 403, 404):
        text = responses[str(status)]["description"]
        named = {c for c in _HEAD_REFUSAL_ENVELOPE if c in text}
        assert named and named <= by_status[status], (status, named)
    assert by_status[400] == {"head_capture_invalid", "head_body_invalid"}
    assert by_status[403] == {"head_write_unauthorized", "forbidden"}
    assert by_status[404] == {"reference_not_found"}
    assert by_status[401] == {"unauthenticated"}
    head = _contract("best_config_head_endpoints.json")["paths"][
        "/api/v1/best-configs/agent-heads/by-ref/{agent_ref}"
    ]["get"]["responses"]
    assert set(head) == {"200", "400", "401", "403", "404", "429", "500"}
    ok = _contract("agent_head_by_ref_response_schema.json")
    assert set(ok["required"]) == {"agent_ref", "environment", "generation", "exists"}


def test_head_read_by_ref_uses_the_contract_route_and_validates_the_body():
    ok = {
        "agent_ref": AGENT_REF,
        "environment": "staging",
        "generation": 7,
        "exists": True,
    }
    wire, patcher = _wired((200, ok))
    with patcher:
        assert (
            _client().fetch_agent_head_generation_by_ref_sync(
                AGENT_REF, environment="staging"
            )
            == 7
        )
    request = wire.requests[0]
    assert request.method == "GET"
    assert request.url == (
        f"https://api.test/api/v1/best-configs/agent-heads/by-ref/{AGENT_REF}"
        "?environment=staging"
    )
    assert request.body is None


@pytest.mark.parametrize(
    "body",
    [
        {
            "agent_ref": OTHER_REF,
            "environment": "staging",
            "generation": 1,
            "exists": True,
        },
        {
            "agent_ref": AGENT_REF,
            "environment": "test",
            "generation": 1,
            "exists": True,
        },
        {
            "agent_ref": AGENT_REF,
            "environment": "staging",
            "generation": 1,
            "exists": False,
        },
        {
            "agent_ref": AGENT_REF,
            "environment": "staging",
            "generation": 0,
            "exists": True,
        },
        {
            "agent_ref": AGENT_REF,
            "environment": "staging",
            "generation": True,
            "exists": True,
        },
        {
            "agent_ref": AGENT_REF,
            "environment": "staging",
            "generation": -1,
            "exists": False,
        },
        {"agent_ref": AGENT_REF, "environment": "staging"},
    ],
)
def test_head_read_rejects_a_body_outside_the_contract(body):
    _, patcher = _wired((200, body))
    with patcher, pytest.raises(CloudServiceError):
        _client().fetch_agent_head_generation_by_ref_sync(
            AGENT_REF, environment="staging"
        )


@pytest.mark.parametrize("ref", ["agent-7", "agentref_xyz", "", OTHER_REF.upper()])
def test_head_read_refuses_a_malformed_ref_before_any_request(ref):
    wire, patcher = _wired()
    with patcher, pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _client().fetch_agent_head_generation_by_ref_sync(ref, environment="staging")
    assert excinfo.value.code is HeadRefusalCode.HEAD_CAPTURE_INVALID
    assert wire.requests == []


def test_head_read_refuses_an_environment_outside_the_closed_enum_before_any_request():
    wire, patcher = _wired()
    with patcher, pytest.raises(CloudBestConfigHeadRefusedError):
        _client().fetch_agent_head_generation_by_ref_sync(
            AGENT_REF, environment="prod-eu"
        )
    assert wire.requests == []


def test_protected_publish_sends_both_headers_and_no_agent_id():
    wire, patcher = _wired((201, {"success": True, "data": {"ok": True}}))
    with patcher:
        out = _client().publish_best_config_sync(
            _SPEC,
            environment="staging",
            agent_ref=AGENT_REF,
            expected_head_generation=3,
        )
    assert out == {"ok": True}
    request = wire.requests[0]
    assert request.method == "POST"
    assert request.headers["X-Traigent-Agent-Ref"] == AGENT_REF
    assert request.headers["X-Traigent-Expected-Head-Generation"] == "3"
    assert "X-Traigent-Agent-Id" not in request.headers
    assert request.url == "https://api.test/api/v1/best-configs"


def test_protected_publish_zero_generation_is_sent():
    wire, patcher = _wired((201, {"success": True, "data": {}}))
    with patcher:
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=0
        )
    assert wire.requests[0].headers["X-Traigent-Expected-Head-Generation"] == "0"


@pytest.mark.parametrize("code", sorted(_STATUS))
def test_every_closed_refusal_code_maps_to_a_typed_exception(code):
    wire, patcher = _wired((_STATUS[code], _refusal(code)))
    with patcher, pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=3
        )
    err = excinfo.value
    assert err.code is HeadRefusalCode(code)
    assert err.status_code == _STATUS[code]
    assert isinstance(err, CloudPublishUnavailable)
    assert isinstance(err, CloudBestConfigHeadStaleError) == (
        code == "head_generation_stale"
    )
    assert len(wire.requests) == 1  # one attempt, whatever the code (500 included)


def test_stale_is_never_retried_or_refreshed():
    wire, patcher = _wired(
        (409, _refusal("head_generation_stale")), (201, {"success": True, "data": {}})
    )
    with patcher, pytest.raises(CloudBestConfigHeadStaleError):
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=3
        )
    assert len(wire.requests) == 1  # no resubmission, no head re-read
    assert [r.method for r in wire.requests] == ["POST"]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (409, {"error_code": "BEST_CONFIG_CONFLICT", "message": CANARY_TEXT}),
        (409, ["not", "an", "object"]),
        (400, {"success": False, "error_code": "made_up_code", "message": CANARY_TEXT}),
        (429, {"retry_after": 5, "message": CANARY_TEXT}),
        (502, "<html>" + CANARY_TEXT + "</html>"),
    ],
)
def test_a_body_outside_the_closed_envelope_is_unrecognized_and_not_echoed(
    status, body
):
    _, patcher = _wired((status, body))
    with patcher, pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=3
        )
    assert excinfo.value.code is HeadRefusalCode.UNRECOGNIZED
    assert excinfo.value.status_code == status
    assert CANARY_TEXT not in str(excinfo.value) + repr(excinfo.value)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, _refusal("head_generation_stale")),  # stale claimed on a 500
        (409, {**_refusal("head_generation_stale"), "extra": CANARY_TEXT}),
        (409, {**_refusal("head_generation_stale"), "message": "Agent head moved."}),
        (409, {**_refusal("head_generation_stale"), "error": "invalid_request"}),
        (409, {**_refusal("head_generation_stale"), "success": True}),
        (
            409,
            {
                k: v
                for k, v in _refusal("head_generation_stale").items()
                if k != "message"
            },
        ),
        (400, _refusal("head_publish_conflict")),  # a 409 code on a 400
        (409, _refusal("internal_error")),
        (404, _refusal("head_write_unauthorized")),
    ],
)
def test_malformed_or_mispaired_envelopes_are_unrecognized(status, body):
    _, patcher = _wired((status, body))
    with patcher, pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=3
        )
    err = excinfo.value
    assert err.code is HeadRefusalCode.UNRECOGNIZED
    assert not isinstance(err, CloudBestConfigHeadStaleError)
    assert err.status_code == status
    assert CANARY_TEXT not in str(err) + repr(err)


def test_publish_half_a_protected_precondition_is_refused_before_any_request():
    wire, patcher = _wired()
    with patcher:
        for kwargs in (
            {"agent_ref": AGENT_REF},
            {"agent_ref": AGENT_REF, "agent_id": "a1", "expected_head_generation": 3},
            {"agent_ref": "agent-7", "expected_head_generation": 3},
            {"agent_ref": AGENT_REF, "expected_head_generation": -1},
            {"agent_ref": AGENT_REF, "expected_head_generation": 2**53},
            {"agent_ref": AGENT_REF, "expected_head_generation": True},
        ):
            with pytest.raises(CloudServiceError):
                _client().publish_best_config_sync(_SPEC, **kwargs)
    assert wire.requests == []


def test_legacy_publish_still_uses_the_legacy_headers():
    with patch("requests.post") as mock_post:
        response = Mock(status_code=201, text="{}", headers={})
        response.json.return_value = {"success": True, "data": {"ok": True}}
        mock_post.return_value = response
        _client().publish_best_config_sync(
            _SPEC, agent_id="a1", expected_head_generation=3
        )
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["X-Traigent-Agent-Id"] == "a1"
    assert "X-Traigent-Agent-Ref" not in headers


# ---------------------------------------------------------------- transport gate


def test_protected_publish_refuses_a_redirect_and_sends_nothing_further():
    wire, patcher = _wired(
        (302, {"_location": "https://evil.test/api/v1/best-configs"}),
        (201, {"success": True, "data": {}}),
    )
    with patcher, pytest.raises(TransportGateRefusal):
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=3
        )
    assert [r.url for r in wire.requests] == ["https://api.test/api/v1/best-configs"]


def test_head_read_refuses_a_redirect():
    wire, patcher = _wired((307, {"_location": "https://evil.test/x"}))
    with patcher, pytest.raises(CloudServiceError) as excinfo:
        _client().fetch_agent_head_generation_by_ref_sync(AGENT_REF, environment="test")
    assert isinstance(excinfo.value.__cause__, TransportGateRefusal)
    assert len(wire.requests) == 1  # the redirect target was never contacted


def test_transport_failure_text_does_not_echo_the_url_or_exception():
    def boom(adapter, request, **kwargs):
        raise requests.exceptions.ConnectionError(f"{request.url} {CANARY_TEXT}")

    with (
        patch.object(requests.adapters.HTTPAdapter, "send", boom),
        pytest.raises(CloudServiceError) as excinfo,
    ):
        _client().publish_best_config_sync(
            _SPEC, agent_ref=AGENT_REF, expected_head_generation=3
        )
    assert CANARY_TEXT not in str(excinfo.value)
    assert AGENT_REF not in str(excinfo.value)


# ------------------------------------------------------------------------ canary


@pytest.mark.asyncio
async def test_no_customer_string_reaches_the_head_requests_or_the_logs(online, caplog):
    """Customer-chosen strings in the create 201 and in refusal bodies never leave.

    One test, run end to end under ``online``: the capture refusal through the SDK, then
    the wire level. It passes in any order (see the isolation tests below).
    """
    caplog.set_level(logging.DEBUG)
    fake = _FakeBackend()
    fake.read_error = CloudBestConfigHeadRefusedError(
        HeadRefusalCode.REFERENCE_NOT_FOUND
    )
    answer, _ = await _run(fake)
    with pytest.raises(CloudBestConfigHeadRefusedError):
        _publish(answer, fake)
    assert CANARY_AGENT_ID not in caplog.text and CANARY_TEXT not in caplog.text

    # wire level: only the opaque ref, the closed environment and the integer travel
    wire, patcher = _wired(
        (
            200,
            {
                "agent_ref": AGENT_REF,
                "environment": "staging",
                "generation": 2,
                "exists": True,
            },
        ),
        (
            409,
            {
                **_refusal("head_generation_stale"),
                "message": CANARY_TEXT,
                "agent_id": CANARY_AGENT_ID,
            },
        ),
    )
    with patcher:
        client = _client()
        generation = client.fetch_agent_head_generation_by_ref_sync(
            AGENT_REF, environment="staging"
        )
        with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
            client.publish_best_config_sync(
                _SPEC,
                environment="staging",
                agent_ref=AGENT_REF,
                expected_head_generation=generation,
            )
    assert excinfo.value.code is HeadRefusalCode.UNRECOGNIZED
    sent = "\n".join(
        f"{r.method} {r.url} {dict(r.headers)} {r.body!r}" for r in wire.requests
    )
    for canary in (CANARY_TEXT, CANARY_AGENT_ID):
        assert canary not in sent
        assert canary not in str(excinfo.value) + repr(excinfo.value)
        assert canary not in caplog.text


def test_online_fixture_leaves_the_client_able_to_send_whatever_the_markers(online):
    """The ``online`` fixture removes every environment marker; the client still builds."""
    for name in ("ENVIRONMENT", "TRAIGENT_ENV", "TRAIGENT_ENVIRONMENT"):
        assert name not in os.environ
    client = _client()
    assert getattr(client, "_url_invalid", None) is False
    client._raise_if_backend_egress_disabled("probe")  # does not raise


def test_environment_markers_do_not_leak_between_online_and_plain_tests(monkeypatch):
    """Whatever ran before, a test not using ``online`` sees the conftest dev marker."""
    assert os.environ.get("TRAIGENT_ENVIRONMENT") == "development"
    assert os.environ.get("TRAIGENT_OFFLINE_MODE") == "false"  # backend_online lane
    client = _client()
    assert getattr(client, "_url_invalid", None) is False


@pytest.mark.asyncio
async def test_no_customer_string_reaches_the_logs_when_the_capture_is_refused(
    online, caplog
):
    caplog.set_level(logging.DEBUG)
    fake = _FakeBackend()
    fake.read_error = CloudBestConfigHeadRefusedError(
        HeadRefusalCode.REFERENCE_NOT_FOUND
    )
    answer, _ = await _run(fake)
    with pytest.raises(CloudBestConfigHeadRefusedError):
        _publish(answer, fake)

    assert CANARY_AGENT_ID not in caplog.text and CANARY_TEXT not in caplog.text


@pytest.mark.parametrize("copy_spec", [True])
def test_publish_spec_is_not_mutated(copy_spec):
    spec = copy.deepcopy(_SPEC)
    _, patcher = _wired((201, {"success": True, "data": {}}))
    with patcher:
        _client().publish_best_config_sync(
            spec, agent_ref=AGENT_REF, expected_head_generation=1
        )
    assert spec == _SPEC
