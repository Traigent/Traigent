"""BackendG1Transport hardening and whole-flow privacy canaries for the G1 requests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

import traigent
from tests.unit.certification.fakes import FakeG1Backend
from tests.unit.certification.test_bind import _FIXTURE, _evidence, _private_from_seed
from traigent.api.functions import with_usage
from traigent.certification import bind as bind_module
from traigent.certification import client_evidence as ce
from traigent.certification import client_keys as ck
from traigent.certification import pins as pins_module
from traigent.certification import snapshot_binding
from traigent.certification.transport import BackendG1Transport, G1RequestError
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.labels import transport_gate
from traigent.snapshot import (
    EvaluatorInputs,
    build_execution_snapshot,
    publish_snapshot,
)
from traigent.snapshot.backends import LocalDirStore

# --- BackendG1Transport -----------------------------------------------------


class _Client:
    def __init__(self, base: str = "https://api.example.test") -> None:
        self.api_base_url = base
        self.timeout = 12.0
        self.egress_checked: list[str] = []
        self.auth_targets: list[str] = []

    def _raise_if_backend_egress_disabled(self, action: str) -> None:
        self.egress_checked.append(action)

    def _get_sync_auth_headers(self, target: str = "backend") -> dict[str, str]:
        self.auth_targets.append(target)
        return {"Authorization": "x"}


class _Response:
    def __init__(
        self, status: int, body: Any = None, *, bad_json: bool = False
    ) -> None:
        self.status_code = status
        self._body = body
        self._bad_json = bad_json

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("not json")
        return self._body


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"preflight": [], "observe": []}
    monkeypatch.setattr(
        transport_gate,
        "preflight_requests",
        lambda method, url, **kw: calls["preflight"].append((method, url, kw)),
    )
    monkeypatch.setattr(
        transport_gate,
        "observe_requests_response",
        lambda response: calls["observe"].append(response),
    )
    return calls


def _patch_requests(monkeypatch: pytest.MonkeyPatch, response: Any) -> list[Any]:
    sent: list[Any] = []

    def fake(method: str, url: str, **kwargs: Any) -> Any:
        sent.append((method, url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests, "request", fake)
    return sent


def test_success_goes_through_the_transport_gate_without_redirects(
    monkeypatch: pytest.MonkeyPatch, gate: dict[str, list[Any]]
) -> None:
    sent = _patch_requests(monkeypatch, _Response(201, {"ok": True}))
    client = _Client()
    result = BackendG1Transport(client).request("POST", "/api/v1beta/x", {"a": 1})
    assert (result.status, result.body) == (201, {"ok": True})
    assert gate["preflight"] == [
        ("POST", "https://api.example.test/api/v1beta/x", {"json": {"a": 1}})
    ]
    assert len(gate["observe"]) == 1
    assert client.egress_checked and client.auth_targets == ["backend"]
    ((_, url, kwargs),) = sent
    assert url == "https://api.example.test/api/v1beta/x"
    assert kwargs["allow_redirects"] is False
    assert kwargs["json"] == {"a": 1}
    assert kwargs["headers"]["Authorization"] == "x"
    assert kwargs["timeout"] == (5.0, 12.0)


@pytest.mark.parametrize(
    "base",
    [
        "ftp://api.example.test",
        "https://" + "user" + ":" + "word" + "@api.example.test",
        "not a url",
        "",
    ],
)
def test_non_plain_http_origin_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch, gate: dict[str, list[Any]], base: str
) -> None:
    sent = _patch_requests(monkeypatch, _Response(200, {}))
    with pytest.raises(G1RequestError) as caught:
        BackendG1Transport(_Client(base)).request("GET", "/api/v1beta/x")
    assert (caught.value.status, caught.value.code) == (None, "invalid_backend_url")
    assert sent == [] and gate["preflight"] == []


def test_error_status_keeps_only_a_safe_code_token_and_drops_the_body(
    monkeypatch: pytest.MonkeyPatch, gate: dict[str, list[Any]]
) -> None:
    _patch_requests(
        monkeypatch,
        _Response(
            400, {"error_code": "invalid_request", "message": "CANARY-BODY-TEXT"}
        ),
    )
    with pytest.raises(G1RequestError) as caught:
        BackendG1Transport(_Client()).request("POST", "/p", {"a": 1})
    assert (caught.value.status, caught.value.code) == (400, "invalid_request")
    assert "CANARY-BODY-TEXT" not in str(caught.value) + repr(caught.value.__dict__)


@pytest.mark.parametrize(
    "body",
    [{"error_code": "Bad Code!"}, {"error_code": 7}, {"message": "x"}, ["x"], None],
)
def test_unsafe_or_missing_code_becomes_none(
    monkeypatch: pytest.MonkeyPatch, gate: dict[str, list[Any]], body: Any
) -> None:
    _patch_requests(monkeypatch, _Response(403, body))
    with pytest.raises(G1RequestError) as caught:
        BackendG1Transport(_Client()).request("POST", "/p", {})
    assert (caught.value.status, caught.value.code) == (403, None)


def test_request_exception_is_a_closed_transport_error(
    monkeypatch: pytest.MonkeyPatch, gate: dict[str, list[Any]]
) -> None:
    _patch_requests(monkeypatch, requests.ConnectionError("CANARY-EXC-TEXT"))
    with pytest.raises(G1RequestError) as caught:
        BackendG1Transport(_Client()).request("GET", "/p")
    assert (caught.value.status, caught.value.code) == (None, "transport_error")
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "CANARY-EXC-TEXT" not in str(caught.value)


def test_gate_refusal_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = transport_gate.TransportGateRefusal

    def deny(method: str, url: str, **kw: Any) -> None:
        raise refusal("route", {})

    monkeypatch.setattr(transport_gate, "preflight_requests", deny)
    sent = _patch_requests(monkeypatch, _Response(200, {}))
    with pytest.raises(refusal):
        BackendG1Transport(_Client()).request("GET", "/p")
    assert sent == []


# --- whole-flow canaries ----------------------------------------------------

_CANARY_DATASET = "CANARY-DATASET-TEXT"
_CANARY_CONFIG = "CANARY-CONFIG-TEXT"
_CANARY_EVALUATOR = "CANARY-EVALUATOR-SOURCE"


def _wire(fake: FakeG1Backend) -> str:
    return json.dumps(fake.requests, sort_keys=True)


def _assert_clean(
    wire: str,
    evidence: ce.ClientEvidence,
    *,
    private_b64: str | None = None,
    extra: tuple[str, ...] = (),
) -> None:
    for blind in evidence.blinds.values():
        assert blind.hex() not in wire
    for kind in ce.SLOT_KINDS:
        assert evidence.commitments[kind] not in wire
        assert json.dumps(evidence.documents[kind]) not in wire
        assert json.dumps(evidence.documents[kind], sort_keys=True) not in wire
    assert evidence.snapshot_manifest_digest not in wire
    assert evidence.snapshot_manifest_digest[len("sha256:") :] not in wire
    for sentinel in (_CANARY_DATASET, _CANARY_CONFIG, _CANARY_EVALUATOR, *extra):
        assert sentinel not in wire
    if private_b64 is not None:
        assert private_b64 not in wire


def _canary_evaluator_function(prediction: str, expected: str) -> float:
    """CANARY-EVALUATOR-SOURCE"""
    _marker = "CANARY-EVALUATOR-SOURCE"
    return 1.0 if prediction == expected and _marker else 0.0


def _canary_snapshot() -> Any:
    return build_execution_snapshot(
        dataset=[{"input": {"q": _CANARY_DATASET}, "expected": "b"}],
        config={"prompt": _CANARY_CONFIG},
        config_space={"prompt": [_CANARY_CONFIG, _CANARY_CONFIG + "-2"]},
        evaluator=EvaluatorInputs(metric_functions={"a": _canary_evaluator_function}),
    )


def test_bind_flow_never_sends_evidence_or_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeG1Backend()
    fake.fixture = _FIXTURE
    monkeypatch.setattr(
        bind_module.secrets, "token_hex", lambda n=16: _FIXTURE["session"]["nonce"]
    )
    key = ck.ClientKey(tmp_path / "k.json", _private_from_seed())
    project = _FIXTURE["pins"]["project_ref"]
    key.pin(project, key.derive_ref(project))
    source = tmp_path / "pins-source.json"
    source.write_text(json.dumps(_FIXTURE["pins"]))
    pins = pins_module.write_pins(source, tmp_path / "pins" / "pins.json")
    session = _FIXTURE["session"]
    evidence = _evidence()
    bind_module.bind_g1(
        fake,
        evidence,
        key,
        pins,
        run_id="run_1",
        sdk_ref=session["sdk_ref"],
        sdk_version=session["sdk_version"],
        declared_mode=session["declared_mode"],
    )
    assert len(fake.requests) == 5
    private_b64 = json.loads(key.path.read_text())["private_key_pkcs8_b64"]
    _assert_clean(_wire(fake), evidence, private_b64=private_b64)


def test_bind_of_content_bearing_evidence_never_sends_content(tmp_path: Path) -> None:
    store = LocalDirStore(tmp_path / "store")
    snapshot = _canary_snapshot()
    ref = publish_snapshot(store, snapshot)
    # The sentinels really are in the retained snapshot, so the canary has teeth.
    stored = b"".join(
        p.read_bytes() for p in (tmp_path / "store").rglob("*") if p.is_file()
    )
    assert _CANARY_DATASET.encode() in stored and _CANARY_CONFIG.encode() in stored
    evidence = ce.build_client_evidence(
        snapshot, custody=ce.LocalFileCustody(tmp_path / "evidence")
    )
    fake = FakeG1Backend()
    fake.fixture = _FIXTURE
    key = ck.ClientKey(tmp_path / "k.json", _private_from_seed())
    project = _FIXTURE["pins"]["project_ref"]
    key.pin(project, key.derive_ref(project))
    source = tmp_path / "pins-source.json"
    source.write_text(json.dumps(_FIXTURE["pins"]))
    pins = pins_module.write_pins(source, tmp_path / "pins" / "pins.json")
    session = _FIXTURE["session"]
    try:
        bind_module.bind_g1(
            fake,
            evidence,
            key,
            pins,
            run_id="run_1",
            sdk_ref=session["sdk_ref"],
            sdk_version=session["sdk_version"],
            declared_mode=session["declared_mode"],
        )
    except bind_module.BindError:
        pass  # the fixture's prepare response belongs to the fixture evidence
    bodies = [b for _, _, b in fake.requests if b is not None]
    assert any("manifest_root_digest" in b for b in bodies)
    private_b64 = json.loads(key.path.read_text())["private_key_pkcs8_b64"]
    _assert_clean(
        _wire(fake),
        evidence,
        private_b64=private_b64,
        extra=(ref.locator, str(tmp_path / "store")),
    )
    assert ref.manifest_digest not in _wire(fake)


def _scorer(prediction: str, expected: str) -> float:
    return 1.0 if prediction == expected else 0.0


def test_snapshot_capture_path_sends_nothing_but_the_readiness_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.api.types import OptimizationResult
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    for name in ("TRAIGENT_REQUIRE_CLOUD", "TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeG1Backend()
    monkeypatch.setattr(snapshot_binding, "default_transport", lambda: fake)

    async def _cloud(self: Any, *args: Any, **kwargs: Any) -> OptimizationResult:
        return OptimizationResult(
            trials=[],
            best_config={"prompt": _CANARY_CONFIG},
            best_score=1.0,
            optimization_id="cloud",
            duration=0.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy"],
            algorithm="remote",
            timestamp=__import__("datetime").datetime.now(),
        )

    monkeypatch.setattr(OptimizedFunction, "_try_cloud_execution", _cloud)
    store = LocalDirStore(tmp_path / "store")

    @traigent.optimize(
        eval_dataset=Dataset(
            [EvaluationExample({"text": _CANARY_DATASET}, "YES")], name="canary"
        ),
        objectives=["accuracy"],
        configuration_space={"prompt": [_CANARY_CONFIG, _CANARY_CONFIG + "-2"]},
        scoring_function=_canary_evaluator_function,
        execution={"offline": False, "snapshot_store": store},
    )
    def agent(text: str) -> str:
        return with_usage("YES", total_cost=0.0)

    result = agent.optimize_sync(max_trials=1)
    assert result.snapshot_ref is not None
    evidence = ce.ClientEvidence.from_record(
        Path(result.client_evidence_path).read_bytes()
    )
    assert [(m, p, b) for m, p, b in fake.requests] == [
        ("GET", "/api/v1beta/certificate-build-sessions/readiness", None)
    ]
    _assert_clean(
        _wire(fake),
        evidence,
        extra=(result.snapshot_ref.locator, result.snapshot_ref.manifest_digest),
    )
