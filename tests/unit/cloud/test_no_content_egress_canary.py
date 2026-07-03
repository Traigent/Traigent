"""No-content-egress canary for default cloud-brain optimization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from traigent import optimize
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.artifact_fingerprints import artifact_fingerprints_to_wire


CANARY_INPUT = "CANARY_INPUT_a1b2"
CANARY_OUTPUT = "CANARY_OUTPUT_c3d4"
CANARY_PROMPT = "CANARY_PROMPT_e5f6"
CANARY_META = "CANARY_META_g7h8"
CANARY_SOURCE = "CANARY_SOURCE_i9j0"
CANARIES = (CANARY_INPUT, CANARY_OUTPUT, CANARY_PROMPT, CANARY_META, CANARY_SOURCE)
FAKE_TRAIGENT_API_KEY = "uk_" + ("a" * 43)
FP_WIRE_PATTERN = r"fp1:[0-9a-f]{64}"


_MISSING = object()


def _json_safe(value: Any) -> Any:
    """Freeze a request argument without mutating the production object."""

    if value is _MISSING:
        return None
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


@dataclass
class _OutboundCapture:
    calls: list[dict[str, Any]] = field(default_factory=list)
    cloud_next_trial_calls: int = 0
    backend_trial_slot_calls: int = 0

    def record(self, method: str, url: Any, kwargs: dict[str, Any]) -> str:
        body_kind = "json" if "json" in kwargs else "data" if "data" in kwargs else None
        body = kwargs.get("json", kwargs.get("data", _MISSING))
        frozen_body = _json_safe(body)
        stage = self._classify(method=method, url=str(url), body=frozen_body)
        self.calls.append(
            {
                "stage": stage,
                "method": method,
                "url": str(url),
                "body_kind": body_kind,
                "body": frozen_body,
            }
        )
        return stage

    def stages(self) -> list[str]:
        return [str(entry["stage"]) for entry in self.calls]

    @staticmethod
    def _classify(method: str, url: str, body: Any) -> str:
        path = urlparse(url).path

        if "upload_example_set" in path or "/datasets" in path:
            return "upload_dataset"
        if path.endswith("/features"):
            return "upload_example_features"
        if method == "POST" and path.endswith("/best-configs"):
            return "best-config-publish"
        if method == "GET" and "/best-configs/" in path:
            return "best-config-fetch"
        if method == "POST" and path.endswith("/hybrid/sessions"):
            return "hybrid-session-create"

        if method == "POST" and path.endswith("/keys/validate"):
            return "key-validation"
        if method == "POST" and path.endswith("/sessions"):
            return "session-create"
        if method == "POST" and path.endswith("/next-trial"):
            if isinstance(body, dict) and "request_metadata" in body:
                return "next-trial"
            return "trial-slot"
        if method == "POST" and path.endswith("/results"):
            return "submit-metrics"
        if method == "POST" and path.endswith("/finalize"):
            return "finalize"
        if method == "PUT" and path.endswith("/status"):
            return "config-run-status"
        if method == "PUT" and path.endswith("/measures"):
            return "config-run-measures"
        if method == "PUT" and "/experiment-runs/runs/" in path:
            return "experiment-run-completion-status"

        return f"{method} {path}"

    def response_for(
        self, method: str, url: Any, kwargs: dict[str, Any]
    ) -> tuple[int, Any]:
        stage = self.record(method, url, kwargs)
        body = kwargs.get("json", kwargs.get("data", _MISSING))

        if stage == "key-validation":
            return 200, {"valid": True}

        if stage == "session-create":
            return (
                201,
                {
                    "session_id": "sess-canary",
                    "status": "active",
                    "metadata": {
                        "experiment_id": "exp-canary",
                        "experiment_run_id": "run-canary",
                    },
                    "optimization_strategy": {"source": "cloud_brain"},
                },
            )

        if stage == "hybrid-session-create":
            return (
                201,
                {
                    "session_id": "hybrid-canary",
                    "token": "optimizer-token",
                    "optimizer_endpoint": "https://backend.example.test/optimizer",
                },
            )

        if stage == "next-trial":
            self.cloud_next_trial_calls += 1
            trial_number = self.cloud_next_trial_calls
            return (
                200,
                {
                    "should_continue": True,
                    "suggestion": {
                        "trial_id": f"cloud-trial-{trial_number}",
                        "session_id": "sess-canary",
                        "trial_number": trial_number,
                        "config": {"temperature": 0.1 if trial_number == 1 else 0.9},
                        "dataset_subset": {
                            "indices": [0, 1],
                            "selection_strategy": "canary_fixture",
                            "confidence_level": 1.0,
                            "estimated_representativeness": 1.0,
                            "metadata": {"dataset_size": 2},
                        },
                        "exploration_type": "exploration",
                        "metadata": {"config_id": f"cfg-{trial_number}"},
                    },
                    "session_status": "active",
                    "metadata": {},
                },
            )

        if stage == "trial-slot":
            self.backend_trial_slot_calls += 1
            slot_number = self.backend_trial_slot_calls
            return (
                200,
                {
                    "should_continue": True,
                    "suggestion": {
                        "trial_id": f"backend-slot-{slot_number}",
                        "session_id": "sess-canary",
                        "trial_number": slot_number,
                        "config": {},
                        "dataset_subset": {
                            "indices": [],
                            "selection_strategy": "slot_only",
                            "confidence_level": 1.0,
                            "estimated_representativeness": 1.0,
                            "metadata": {},
                        },
                        "exploration_type": "slot",
                        "metadata": {},
                    },
                    "session_status": "active",
                },
            )

        if stage == "submit-metrics":
            return 201, {"continue_optimization": True}

        if stage == "config-run-status":
            return 204, {}

        if stage == "config-run-measures":
            return 200, {}

        if stage == "experiment-run-completion-status":
            return 204, {}

        if stage == "upload_example_features":
            return 200, {"ok": True}

        if stage == "best-config-publish":
            spec = body.get("spec", {}) if isinstance(body, dict) else {}
            return 200, {"data": {"config_id": "best-canary", "spec": spec}}

        if stage == "best-config-fetch":
            return (
                200,
                {"data": {"config_id": "best-canary", "spec": {"temperature": 0.1}}},
            )

        if stage == "finalize":
            return (
                200,
                {
                    "session_id": "sess-canary",
                    "best_config": {"temperature": 0.1},
                    "best_metrics": {"accuracy": 1.0},
                    "total_trials": 2,
                    "successful_trials": 2,
                    "total_duration": 0.0,
                    "cost_savings": 0.0,
                    "metadata": {
                        "experiment_id": "exp-canary",
                        "experiment_run_id": "run-canary",
                    },
                },
            )

        return 200, {}


class _FakeAiohttpResponse:
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def __aenter__(self) -> _FakeAiohttpResponse:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        return _json_safe(self._payload)

    async def text(self) -> str:
        return json.dumps(self._payload, default=str)


class _FakeRequestsResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, default=str)

    def json(self) -> Any:
        return _json_safe(self._payload)


def _install_transport_capture(
    monkeypatch: pytest.MonkeyPatch, capture: _OutboundCapture
) -> None:
    from traigent.cloud._aiohttp_compat import aiohttp
    import traigent.cloud.api_operations as api_operations
    import traigent.cloud.auth as auth
    import traigent.cloud.backend_client as backend_client
    import traigent.cloud.client as cloud_client
    import traigent.cloud.session_operations as session_operations
    import traigent.cloud.trial_operations as trial_operations

    class CapturingClientSession:
        __module__ = "aiohttp.client"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.closed = False

        async def __aenter__(self) -> CapturingClientSession:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            await self.close()
            return False

        async def close(self) -> None:
            self.closed = True

        def post(self, url: Any, *args: Any, **kwargs: Any) -> _FakeAiohttpResponse:
            status, payload = capture.response_for("POST", url, kwargs)
            return _FakeAiohttpResponse(status, payload)

        def put(self, url: Any, *args: Any, **kwargs: Any) -> _FakeAiohttpResponse:
            status, payload = capture.response_for("PUT", url, kwargs)
            return _FakeAiohttpResponse(status, payload)

        def get(self, url: Any, *args: Any, **kwargs: Any) -> _FakeAiohttpResponse:
            status, payload = capture.response_for("GET", url, kwargs)
            return _FakeAiohttpResponse(status, payload)

    monkeypatch.setattr(aiohttp, "ClientSession", CapturingClientSession)

    for module in (
        api_operations,
        auth,
        backend_client,
        cloud_client,
        session_operations,
        trial_operations,
    ):
        monkeypatch.setattr(module, "aiohttp", aiohttp)

    try:
        import requests
    except ImportError:  # pragma: no cover - requests is a required dependency
        return

    def fake_requests_post(
        url: Any, *args: Any, **kwargs: Any
    ) -> _FakeRequestsResponse:
        status, payload = capture.response_for("POST", url, kwargs)
        return _FakeRequestsResponse(status, payload)

    def fake_requests_get(url: Any, *args: Any, **kwargs: Any) -> _FakeRequestsResponse:
        status, payload = capture.response_for("GET", url, kwargs)
        return _FakeRequestsResponse(status, payload)

    def fake_requests_put(url: Any, *args: Any, **kwargs: Any) -> _FakeRequestsResponse:
        status, payload = capture.response_for("PUT", url, kwargs)
        return _FakeRequestsResponse(status, payload)

    monkeypatch.setattr(requests, "post", fake_requests_post)
    monkeypatch.setattr(requests, "get", fake_requests_get)
    monkeypatch.setattr(requests, "put", fake_requests_put)


@pytest.fixture
def fail_unmocked_backend_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail offline canaries if code reaches real backend transport."""

    try:
        import requests
    except ImportError:  # pragma: no cover - requests is required in test env
        requests = None  # type: ignore[assignment]

    if requests is not None:

        def fail_requests_request(
            self: Any, method: str, url: Any, *args: Any, **kwargs: Any
        ) -> Any:
            raise AssertionError(f"unmocked outbound backend transport: {method} {url}")

        monkeypatch.setattr(
            requests.sessions.Session,
            "request",
            fail_requests_request,
        )

    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp is present in cloud tests
        return

    async def fail_aiohttp_request(
        self: Any, method: str, url: Any, *args: Any, **kwargs: Any
    ) -> Any:
        raise AssertionError(f"unmocked outbound backend transport: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", fail_aiohttp_request)


def test_fail_unmocked_backend_transport_fixture_blocks_unguarded_request(
    monkeypatch: pytest.MonkeyPatch,
    fail_unmocked_backend_transport: None,
) -> None:
    """Negative control: an unguarded offline backend send must fail the fixture."""

    import requests

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")

    with pytest.raises(AssertionError, match="unmocked outbound backend transport"):
        requests.Session().post(
            "https://backend.example.test/unguarded-probe",
            json={"probe": True},
        )


def _dataset_with_canaries() -> Dataset:
    return Dataset(
        name="canary_dataset",
        description="Dataset with unique content canaries",
        examples=[
            EvaluationExample(
                input_data={
                    "question": f"{CANARY_INPUT} customer question",
                    "prompt": f"{CANARY_PROMPT} prompt template",
                },
                expected_output=f"{CANARY_OUTPUT} expected answer",
                metadata={
                    "example_id": "canary-example-1",
                    "source": f"{CANARY_META} metadata marker",
                },
            ),
            EvaluationExample(
                input_data={
                    "question": f"{CANARY_INPUT} second question",
                    "prompt": f"{CANARY_PROMPT} second prompt",
                },
                expected_output=f"{CANARY_OUTPUT} second expected answer",
                metadata={
                    "example_id": "canary-example-2",
                    "source": f"{CANARY_META} second metadata marker",
                },
            ),
        ],
    )


def _allow_backend_egress_in_test(
    monkeypatch: pytest.MonkeyPatch, local_storage_path: Path
) -> None:
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_TRAIGENT_API_KEY)
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://backend.example.test")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(local_storage_path))
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)


def _backend_client_for_guard_test(*, no_egress: bool = False) -> Any:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    return BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY,
        backend_config=BackendClientConfig(
            backend_base_url="https://backend.example.test"
        ),
        no_egress=no_egress,
    )


async def _call_cloud_client_guard_operation(client: Any, operation: str) -> Any:
    if operation == "create_optimization_session":
        return await client.create_optimization_session(
            "guarded_function",
            {"temperature": [0.1]},
            ["accuracy"],
            dataset_metadata={"size": 0},
            max_trials=1,
        )
    if operation == "get_next_trial":
        return await client.get_next_trial("sess-canary")
    if operation == "submit_trial_result":
        return await client.submit_trial_result(
            session_id="sess-canary",
            trial_id="trial-canary",
            metrics={"accuracy": 1.0},
            duration=0.25,
            status="completed",
        )
    if operation == "finalize_optimization":
        return await client.finalize_optimization("sess-canary")
    raise AssertionError(f"unknown cloud operation: {operation}")


async def _call_api_guard_operation(client: Any, operation: str) -> Any:
    if operation == "config_status":
        return await client._update_config_run_status("cfg-canary", "running")
    if operation == "config_measures":
        return await client._update_config_run_measures(
            "cfg-canary",
            {"accuracy": 1.0},
            execution_time=0.25,
        )
    if operation == "experiment_completion_status":
        return await client._update_experiment_run_status_on_completion(
            "run-canary",
            "completed",
        )
    raise AssertionError(f"unknown api operation: {operation}")


def _call_sync_guard_operation(client: Any, operation: str) -> Any:
    if operation == "upload_example_features":
        return client.upload_example_features(
            "run-canary",
            "simhash_v1",
            [{"example_id": "ex-1", "feature": "abcd"}],
        )
    if operation == "best_config_publish":
        return client.publish_best_config_sync(
            {"temperature": 0.1},
            environment="test",
        )
    if operation == "best_config_fetch":
        return client.fetch_best_config_sync(
            "best-canary",
            environment="test",
            function_ref="guarded_function",
        )
    raise AssertionError(f"unknown sync operation: {operation}")


def _run_canary_optimization(
    *,
    local_storage_path: Path,
    offline: bool = False,
    execution_mode: str | None = None,
):
    kwargs: dict[str, Any] = {
        "eval_dataset": _dataset_with_canaries(),
        "objectives": ["accuracy"],
        "configuration_space": {"temperature": [0.1, 0.9]},
        "algorithm": "auto",
        "max_trials": 2,
        "offline": offline,
        "local_storage_path": str(local_storage_path),
    }
    if execution_mode is not None:
        kwargs["execution_mode"] = execution_mode

    @optimize(**kwargs)
    def canary_app(example: dict[str, Any]) -> str:
        source_marker = "CANARY_SOURCE_i9j0"
        assert source_marker
        assert CANARY_INPUT in example["question"]
        assert CANARY_PROMPT in example["prompt"]
        return "safe-local-output"

    return canary_app.optimize_sync(progress_bar=False)


def _assert_required_production_bodies_were_captured(
    capture: _OutboundCapture,
) -> None:
    stages = capture.stages()
    assert stages.count("session-create") == 1
    assert stages.count("next-trial") == 2
    assert stages.count("submit-metrics") == 2
    assert stages.count("finalize") == 1


def _assert_no_canaries_crossed_wire(capture: _OutboundCapture) -> None:
    assert capture.calls, "canary did not capture any outbound payloads"
    for entry in capture.calls:
        blob = json.dumps(entry["body"], sort_keys=True, default=str)
        hits = [sentinel for sentinel in CANARIES if sentinel in blob]
        assert not hits, (
            f"{entry['stage']} leaked dataset content canaries {hits} in "
            f"{entry['method']} {entry['url']} payload: {blob}"
        )


def _assert_session_create_has_artifact_fingerprints(
    capture: _OutboundCapture,
) -> None:
    session_bodies = [
        entry["body"] for entry in capture.calls if entry["stage"] == "session-create"
    ]
    assert len(session_bodies) == 1
    body = session_bodies[0]
    assert isinstance(body, dict)

    artifact_fingerprints = body["artifact_fingerprints"]
    assert set(artifact_fingerprints) == {
        "dataset",
        "agent",
        "evaluator",
        "config_space",
    }
    assert all(
        value is None
        or (isinstance(value, str) and re.fullmatch(FP_WIRE_PATTERN, value))
        for value in artifact_fingerprints.values()
    )
    assert re.fullmatch(FP_WIRE_PATTERN, artifact_fingerprints["dataset"])
    assert re.fullmatch(FP_WIRE_PATTERN, artifact_fingerprints["agent"])
    assert re.fullmatch(FP_WIRE_PATTERN, artifact_fingerprints["evaluator"])
    assert re.fullmatch(FP_WIRE_PATTERN, artifact_fingerprints["config_space"])
    assert body["fingerprint_meta"] == {
        "algorithm": "fp1",
        "dataset_example_count": 2,
        "source_available": True,
    }


def test_cloud_brain_auto_does_not_egress_dataset_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / "cloud")
    _install_transport_capture(monkeypatch, capture)

    result = _run_canary_optimization(local_storage_path=tmp_path / "cloud")

    stages = capture.stages()
    assert result.source == "cloud_brain"
    _assert_required_production_bodies_were_captured(capture)
    assert "upload_dataset" not in stages
    _assert_session_create_has_artifact_fingerprints(capture)
    _assert_no_canaries_crossed_wire(capture)


def test_artifact_fingerprint_wire_serializer_rejects_raw_content_smuggling() -> None:
    wire = artifact_fingerprints_to_wire(
        {
            "dataset": f"fp1:{CANARY_SOURCE}",
            "agent": "fp1:" + ("a" * 64),
            "evaluator": "fp1:" + ("F" * 64),
            "config_space": "fp1:" + ("0" * 63),
        }
    )

    assert wire == {
        "dataset": None,
        "agent": "fp1:" + ("a" * 64),
        "evaluator": None,
        "config_space": None,
    }


def test_offline_and_legacy_local_modes_construct_no_backend_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_unmocked_backend_transport: None,
) -> None:
    from traigent.core.backend_session_manager import BackendSessionManager

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / "offline")
    _install_transport_capture(monkeypatch, capture)

    def fail_if_backend_client_is_constructed(_config: Any) -> Any:
        raise AssertionError("local-only mode constructed a backend client")

    monkeypatch.setattr(
        BackendSessionManager,
        "create_backend_client",
        staticmethod(fail_if_backend_client_is_constructed),
    )

    offline_result = _run_canary_optimization(
        offline=True,
        local_storage_path=tmp_path / "offline",
    )
    assert offline_result.source == "offline"
    assert capture.calls == []

    legacy_result = _run_canary_optimization(
        execution_mode="local",
        local_storage_path=tmp_path / "legacy-edge",
    )
    assert legacy_result.source == "offline"
    assert capture.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "blocked_message"),
    [
        ("create_optimization_session", "not sending create session"),
        ("get_next_trial", "not sending get next trial"),
        ("submit_trial_result", "not sending submit trial result"),
        ("finalize_optimization", "not sending finalize session"),
    ],
)
@pytest.mark.parametrize(
    ("offline_env", "no_egress"),
    [
        (True, False),
        (False, True),
    ],
)
async def test_cloud_client_low_level_send_guard_blocks_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_unmocked_backend_transport: None,
    operation: str,
    blocked_message: str,
    offline_env: bool,
    no_egress: bool,
) -> None:
    from traigent.cloud.client import CloudEgressBlockedError, TraigentCloudClient

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / "guard")
    if offline_env:
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    _install_transport_capture(monkeypatch, capture)

    client = TraigentCloudClient(
        api_key=FAKE_TRAIGENT_API_KEY,
        base_url="https://backend.example.test",
        no_egress=no_egress,
    )

    with pytest.raises(CloudEgressBlockedError, match=blocked_message):
        await _call_cloud_client_guard_operation(client, operation)

    assert capture.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "blocked_message"),
    [
        ("config_status", "not sending update configuration run status"),
        ("config_measures", "not sending update configuration run measures"),
        ("experiment_completion_status", "not sending update experiment run status"),
    ],
)
@pytest.mark.parametrize(
    ("offline_env", "no_egress"),
    [
        (True, False),
        (False, True),
    ],
)
async def test_api_operations_direct_send_guards_block_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_unmocked_backend_transport: None,
    operation: str,
    blocked_message: str,
    offline_env: bool,
    no_egress: bool,
) -> None:
    from traigent.cloud.client import CloudEgressBlockedError

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / f"api-{operation}")
    if offline_env:
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    _install_transport_capture(monkeypatch, capture)

    client = _backend_client_for_guard_test(no_egress=no_egress)

    with pytest.raises(CloudEgressBlockedError, match=blocked_message):
        await _call_api_guard_operation(client, operation)

    assert capture.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_stage", "expected_result"),
    [
        ("config_status", "config-run-status", True),
        ("config_measures", "config-run-measures", True),
        ("experiment_completion_status", "experiment-run-completion-status", None),
    ],
)
async def test_api_operations_direct_send_helpers_allow_online_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    expected_stage: str,
    expected_result: Any,
) -> None:
    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / f"api-online-{operation}")
    _install_transport_capture(monkeypatch, capture)

    client = _backend_client_for_guard_test()

    result = await _call_api_guard_operation(client, operation)

    assert result == expected_result
    assert expected_stage in capture.stages()


@pytest.mark.parametrize(
    ("operation", "blocked_message"),
    [
        ("upload_example_features", "not sending upload example features"),
        ("best_config_publish", "not sending publish best config"),
        ("best_config_fetch", "not sending fetch best config"),
    ],
)
@pytest.mark.parametrize(
    ("offline_env", "no_egress"),
    [
        (True, False),
        (False, True),
    ],
)
def test_backend_sync_send_guards_block_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_unmocked_backend_transport: None,
    operation: str,
    blocked_message: str,
    offline_env: bool,
    no_egress: bool,
) -> None:
    from traigent.cloud.client import CloudEgressBlockedError

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / f"sync-{operation}")
    if offline_env:
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    _install_transport_capture(monkeypatch, capture)

    client = _backend_client_for_guard_test(no_egress=no_egress)

    with pytest.raises(CloudEgressBlockedError, match=blocked_message):
        _call_sync_guard_operation(client, operation)

    assert capture.calls == []


@pytest.mark.parametrize(
    ("operation", "expected_stage"),
    [
        ("upload_example_features", "upload_example_features"),
        ("best_config_publish", "best-config-publish"),
        ("best_config_fetch", "best-config-fetch"),
    ],
)
def test_backend_sync_send_helpers_allow_online_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    expected_stage: str,
) -> None:
    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / f"sync-online-{operation}")
    _install_transport_capture(monkeypatch, capture)

    client = _backend_client_for_guard_test()

    result = _call_sync_guard_operation(client, operation)

    if operation == "upload_example_features":
        assert result is True
    else:
        assert result["config_id"] == "best-canary"
    assert expected_stage in capture.stages()


@pytest.mark.asyncio
async def test_cloud_client_low_level_send_guard_allows_online_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from traigent.cloud.client import TraigentCloudClient

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / "online")
    _install_transport_capture(monkeypatch, capture)

    client = TraigentCloudClient(
        api_key=FAKE_TRAIGENT_API_KEY,
        base_url="https://backend.example.test",
    )
    try:
        response = await client.create_optimization_session(
            "online_function",
            {"temperature": [0.1]},
            ["accuracy"],
            dataset_metadata={"size": 0},
            max_trials=1,
        )
    finally:
        await client.close()

    assert response.session_id == "sess-canary"
    stages = capture.stages()
    assert stages[-1:] == ["session-create"]
    assert stages in (["session-create"], ["key-validation", "session-create"])


@pytest.mark.asyncio
async def test_backend_integrated_policy_guard_blocks_low_level_sends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_unmocked_backend_transport: None,
) -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )
    from traigent.cloud.client import CloudEgressBlockedError

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / "backend-guard")
    _install_transport_capture(monkeypatch, capture)

    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY,
        backend_config=BackendClientConfig(
            backend_base_url="https://backend.example.test"
        ),
        no_egress=True,
    )

    with pytest.raises(
        CloudEgressBlockedError, match="not sending create hybrid session"
    ):
        await client.create_hybrid_session(
            "guarded_problem",
            {"temperature": [0.1]},
            {"objectives": ["accuracy"], "max_trials": 1},
        )
    with pytest.raises(CloudEgressBlockedError, match="not sending get next trial"):
        await client.request_trial_slot("sess-canary")
    with pytest.raises(
        CloudEgressBlockedError, match="not sending submit trial result"
    ):
        await client._submit_trial_result_via_session(
            session_id="sess-canary",
            trial_id="trial-canary",
            config={"temperature": 0.1},
            metrics={"accuracy": 1.0},
            status="completed",
        )
    with pytest.raises(
        CloudEgressBlockedError, match="not sending finalize hybrid session"
    ):
        await client.finalize_hybrid_session("sess-canary")

    assert capture.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("guard_source", ["env", "policy"])
async def test_offline_legacy_traigent_client_hybrid_zero_transport_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_unmocked_backend_transport: None,
    guard_source: str,
) -> None:
    from traigent.cloud.client import CloudEgressBlockedError
    from traigent.traigent_client import TraigentClient

    class _UnusedAgentBuilder:
        def build_agent(self, _agent_spec: dict[str, Any]) -> Any:
            raise AssertionError("offline guard should fire before local execution")

    capture = _OutboundCapture()
    _allow_backend_egress_in_test(monkeypatch, tmp_path / f"legacy-{guard_source}")
    client_kwargs: dict[str, Any] = {}
    if guard_source == "env":
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    else:
        client_kwargs["no_egress"] = True
    _install_transport_capture(monkeypatch, capture)

    client = TraigentClient(
        execution_mode="hybrid",
        agent_builder=_UnusedAgentBuilder(),
        **client_kwargs,
    )

    with pytest.raises(
        CloudEgressBlockedError, match="not sending create hybrid session"
    ):
        await client.optimize(
            function=lambda _example: "ok",
            dataset={
                "examples": [{"input": {"question": "q"}, "expected_output": "ok"}]
            },
            configuration_space={"model": ["gpt-4o-mini"]},
            objectives=["accuracy"],
            max_trials=1,
        )

    assert capture.calls == []
