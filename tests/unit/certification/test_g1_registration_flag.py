"""The closed ``snapshot_g1_required`` flag at run registration (G1 intent).

Without it the Backend issues the plain certificate at run completion and a
later ``certify bind`` can never upgrade it. The flag is a boolean only: no key
other than ``snapshot_g1_required`` may be added to the registration payload.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import traigent
from tests.unit.certification.fakes import FakeG1Backend
from traigent.api.functions import with_usage
from traigent.api.types import OptimizationStatus
from traigent.certification import snapshot_binding
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.models import SessionCreationRequest
from traigent.cloud.session_types import (
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.snapshot.backends import LocalDirStore
from traigent.utils.exceptions import SnapshotBindingError
from traigent.utils.function_identity import resolve_function_descriptor

FLAG = "snapshot_g1_required"


def _request(**overrides: Any) -> SessionCreationRequest:
    base: dict[str, Any] = {
        "function_name": "pkg.mod.fn",
        "configuration_space": {"temperature": [0.1, 0.9]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
        "max_trials": 2,
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _ops() -> ApiOperations:
    client = MagicMock()
    client.backend_config.api_base_url = "https://api.example.test"
    return ApiOperations(client)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "TRAIGENT_CONTENT_LESS",
        "TRAIGENT_SESSION_CONTRACT",
        "TRAIGENT_REQUIRE_CLOUD",
        "TRAIGENT_OFFLINE",
        "TRAIGENT_OFFLINE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))


@pytest.mark.parametrize(
    "builder", ["_build_typed_session_payload", "_build_legacy_session_payload"]
)
def test_payload_adds_exactly_the_closed_boolean_when_set(builder: str) -> None:
    ops = _ops()
    unset = getattr(ops, builder)(_request(), 2)
    on = getattr(ops, builder)(_request(snapshot_g1_required=True), 2)
    assert FLAG not in unset
    assert on[FLAG] is True
    # Privacy canary: nothing else was added or changed.
    assert {k: v for k, v in on.items() if k != FLAG} == unset


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_non_true_values_never_reach_the_wire(value: Any) -> None:
    payload = _ops()._build_typed_session_payload(
        _request(snapshot_g1_required=value), 2
    )
    assert FLAG not in payload


def _manager(backend: Any) -> BackendSessionManager:
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"p": [1, 2]}
    config = TraigentConfig()
    config.execution_mode = "local"
    return BackendSessionManager(
        backend_client=backend,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(objective_names=["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt-g1",
        optimization_status=OptimizationStatus.RUNNING,
    )


def _create(manager: BackendSessionManager) -> Any:
    def func(x: Any) -> Any:
        return x

    dataset = Mock()
    dataset.name = "d"
    dataset.examples = [EvaluationExample({"q": "x"}, "y")]
    dataset.__len__ = Mock(return_value=1)
    return manager.create_session(
        func=func,
        dataset=dataset,
        function_descriptor=resolve_function_descriptor(func),
        max_trials=2,
        start_time=1.0,
    )


def _backend(result: SessionCreationResult) -> Mock:
    client = Mock()
    client.create_session = Mock(return_value=result)
    client.get_session_mapping = Mock(
        return_value=SimpleNamespace(experiment_run_id="run-1")
    )
    client.auth_manager = Mock(has_api_key=Mock(return_value=True))
    return client


def test_manager_passes_flag_only_when_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    plain = _backend(SessionCreationResult.connected(session_id="s1"))
    _create(_manager(plain))
    assert FLAG not in plain.create_session.call_args.kwargs

    declared = _backend(SessionCreationResult.connected(session_id="s2"))
    manager = _manager(declared)
    manager.set_snapshot_g1_required()
    _create(manager)
    assert declared.create_session.call_args.kwargs[FLAG] is True


def test_refused_registration_raises_readiness_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    refused = _backend(
        SessionCreationResult(
            session_id="local-1",
            backend_connected=False,
            failure_reason=SessionCreationFailureReason.SESSION_FAILED,
        )
    )
    manager = _manager(refused)
    manager.set_snapshot_g1_required()
    with pytest.raises(SnapshotBindingError) as caught:
        _create(manager)
    assert caught.value.code == "READINESS_UNAVAILABLE"


def test_egress_off_run_with_g1_intent_is_refused_not_run_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(_backend(SessionCreationResult.connected(session_id="s")))
    manager.set_snapshot_g1_required()
    monkeypatch.setattr(BackendSessionManager, "_egress_disabled", lambda self: True)
    with pytest.raises(SnapshotBindingError) as caught:
        _create(manager)
    assert caught.value.code == "READINESS_UNAVAILABLE"


# ---- end to end through the decorator: only an online, ready run declares intent


def _agent(store: Any, *, offline: bool) -> Any:
    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g1"),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.2]},
        scoring_function=lambda prediction, expected: (
            1.0 if prediction == expected else 0.0
        ),
        execution={"offline": offline, "snapshot_store": store},
    )
    def agent(text: str) -> Any:
        return with_usage("YES", total_cost=0.0)

    return agent


def test_offline_run_never_declares_g1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    seen: list[bool] = []
    original = BackendSessionManager.set_snapshot_g1_required

    def spy(self: BackendSessionManager) -> None:
        seen.append(True)
        original(self)

    monkeypatch.setattr(BackendSessionManager, "set_snapshot_g1_required", spy)
    _agent(LocalDirStore(tmp_path / "store"), offline=True).optimize_sync(max_trials=2)
    assert seen == []


def test_online_ready_run_declares_g1_and_refusal_stops_before_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    # Hermetic: never touch the network, whatever key the shell carries.
    for name in ("TRAIGENT_API_KEY", "OPTIGEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeG1Backend()
    monkeypatch.setattr(snapshot_binding, "default_transport", lambda: fake)
    refused = _backend(
        SessionCreationResult(
            session_id="local-1",
            backend_connected=False,
            failure_reason=SessionCreationFailureReason.SESSION_FAILED,
        )
    )
    monkeypatch.setattr(
        BackendSessionManager, "create_backend_client", staticmethod(lambda _c: refused)
    )
    monkeypatch.setattr(BackendSessionManager, "_egress_disabled", lambda self: False)
    calls: list[str] = []

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g1"),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.2]},
        scoring_function=lambda prediction, expected: (
            1.0 if prediction == expected else 0.0
        ),
        execution={
            "offline": False,
            "snapshot_store": LocalDirStore(tmp_path / "store"),
        },
    )
    def agent(text: str) -> Any:
        calls.append(text)
        return with_usage("YES", total_cost=0.0)

    with pytest.raises(SnapshotBindingError) as caught:
        agent.optimize_sync(algorithm="random", max_trials=2)
    assert caught.value.code == "READINESS_UNAVAILABLE"
    assert caught.value.result is None
    assert calls == []  # refused before any trial ran
    assert refused.create_session.call_args.kwargs[FLAG] is True


@pytest.mark.parametrize("option", [None, False])
def test_g1_intent_requires_a_run_id_independent_of_require_run_id(
    monkeypatch: pytest.MonkeyPatch, option: bool | None
) -> None:
    from traigent.cloud.client import RunIdMissingError

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
    backend = _backend(SessionCreationResult.connected(session_id="s1"))
    backend.get_session_mapping = Mock(
        return_value=SimpleNamespace(experiment_run_id=None)
    )
    manager = _manager(backend)
    manager._require_run_id_option = option
    manager.set_snapshot_g1_required()
    with pytest.raises(RunIdMissingError):
        _create(manager)
    with pytest.raises(RunIdMissingError):
        manager.ensure_run_id_recorded("s1")
