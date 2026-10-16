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
        if not by_ref:
            self.fetch_agent_head_generation_by_ref_sync = None  # type: ignore[assignment]

    def create_session(self, function_name, search_space, optimization_goal, **kw):
        if self.content_less:
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
        self.publishes.append(
            {
                "agent_ref": agent_ref,
                "agent_id": agent_id,
                "expected": expected_head_generation,
                "environment": environment,
            }
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


def _make_wrapper():
    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space=_SPACE,
        injection_mode="parameter",
    )
    def answer(text: str, config) -> str:
        return "ok"

    return answer


async def _run(fake: _FakeBackend):
    answer = _make_wrapper()
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
async def test_capture_for_another_environment_refuses_instead_of_going_unprotected(
    online,
):
    fake = _FakeBackend()
    answer, _ = await _run(fake)
    answer._csm.best_config_environment = "production"
    with pytest.raises(CloudBestConfigHeadRefusedError) as excinfo:
        _publish(answer, fake)
    assert excinfo.value.code is HeadRefusalCode.HEAD_ENVIRONMENT_MISMATCH
    assert fake.publishes == []


@pytest.mark.asyncio
async def test_legacy_session_is_unchanged(online):
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


def _client() -> BackendIntegratedClient:
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


def test_fixture_tables_cover_every_schema_code() -> None:
    schema = Path(
        "/home/nimrod/TraigentProjects/core_project/worktrees/TraigentSchema/content-less"
        "/traigent_schema/schemas/optimization/best_config_head_refusal_schema.json"
    )
    expected = {c.value for c in HeadRefusalCode} - {"unrecognized", "capture_missing"}
    assert set(_FIXED_TEXT) == expected == set(_STATUS)
    if schema.exists():  # the contract itself, when the Schema worktree is present
        enum = json.loads(schema.read_text())["properties"]["error_code"]["enum"]
        assert set(enum) == expected


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
    """Customer-chosen strings in the create 201 and in refusal bodies never leave."""
    caplog.set_level(logging.DEBUG)
    fake = _FakeBackend()
    fake.read_error = CloudBestConfigHeadRefusedError(
        HeadRefusalCode.REFERENCE_NOT_FOUND
    )
    answer, _ = await _run(fake)
    with pytest.raises(CloudBestConfigHeadRefusedError):
        _publish(answer, fake)

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
        with pytest.raises(CloudBestConfigHeadStaleError) as excinfo:
            client.publish_best_config_sync(
                _SPEC,
                environment="staging",
                agent_ref=AGENT_REF,
                expected_head_generation=generation,
            )
    sent = "\n".join(
        f"{r.method} {r.url} {dict(r.headers)} {r.body!r}" for r in wire.requests
    )
    for canary in (CANARY_TEXT, CANARY_AGENT_ID):
        assert canary not in sent
        assert canary not in str(excinfo.value) + repr(excinfo.value)
        assert canary not in caplog.text


@pytest.mark.parametrize("copy_spec", [True])
def test_publish_spec_is_not_mutated(copy_spec):
    spec = copy.deepcopy(_SPEC)
    _, patcher = _wired((201, {"success": True, "data": {}}))
    with patcher:
        _client().publish_best_config_sync(
            spec, agent_ref=AGENT_REF, expected_head_generation=1
        )
    assert spec == _SPEC
