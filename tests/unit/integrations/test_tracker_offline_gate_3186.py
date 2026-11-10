"""MLflow and W&B trackers make no request while offline (#3186)."""

from __future__ import annotations

import types
from typing import Any

import pytest

from traigent.integrations.observability import mlflow as mlflow_module
from traigent.integrations.observability import wandb as wandb_module
from traigent.utils.error_handler import OfflineModeError


class _Recorder:
    """Records every attribute call; stands in for the mlflow / wandb module."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return types.SimpleNamespace(
                id="run-1",
                experiment_id="exp-1",
                info=types.SimpleNamespace(run_id="run-1"),
            )

        return call


@pytest.fixture
def mlflow_stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(mlflow_module, "mlflow", rec)
    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    return rec


@pytest.fixture
def wandb_stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(wandb_module, "wandb", rec)
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    return rec


def test_mlflow_online_control_reaches_server(
    monkeypatch: pytest.MonkeyPatch, mlflow_stub: _Recorder
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    tracker = mlflow_module.TraigentMLflowTracker()
    tracker.start_optimization_run("fn", ["accuracy"], {"p": ["a"]})
    assert "start_run" in mlflow_stub.calls


def test_mlflow_offline_makes_no_request(
    monkeypatch: pytest.MonkeyPatch, mlflow_stub: _Recorder
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    tracker = mlflow_module.TraigentMLflowTracker()
    mlflow_stub.calls.clear()
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    with pytest.raises(OfflineModeError):
        tracker.start_optimization_run("fn", ["accuracy"], {"p": ["a"]})
    with pytest.raises(OfflineModeError):
        tracker.log_trial(types.SimpleNamespace(), 1)
    with pytest.raises(OfflineModeError):
        tracker.end_optimization_run()
    with pytest.raises(OfflineModeError):
        mlflow_module.TraigentMLflowTracker()
    assert mlflow_stub.calls == []


def test_wandb_offline_makes_no_request(
    monkeypatch: pytest.MonkeyPatch, wandb_stub: _Recorder
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    tracker = wandb_module.TraigentWandBTracker()
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    with pytest.raises(OfflineModeError):
        tracker.start_optimization_run("fn", ["accuracy"], {"p": ["a"]})
    with pytest.raises(OfflineModeError):
        tracker.log_trial(types.SimpleNamespace(), 1)
    with pytest.raises(OfflineModeError):
        tracker.end_optimization_run()
    with pytest.raises(OfflineModeError):
        wandb_module.TraigentWandBTracker()
    assert wandb_stub.calls == []


def test_offline_run_scope_gates_trackers(
    monkeypatch: pytest.MonkeyPatch, mlflow_stub: _Recorder
) -> None:
    """``offline=True`` on the run (no env var) gates a tracker built earlier."""
    from traigent.utils.env_config import offline_run_scope

    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    tracker = mlflow_module.TraigentMLflowTracker()
    mlflow_stub.calls.clear()
    with offline_run_scope(True), pytest.raises(OfflineModeError):
        tracker.start_optimization_run("fn", ["accuracy"], {"p": ["a"]})
    assert mlflow_stub.calls == []
