"""Regression coverage for backend session finalization persistence."""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.core.orchestrator import OptimizationOrchestrator


class _HTTPError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _Mapping:
    experiment_id = "exp_finalize"
    experiment_run_id = "run_finalize"


def _result() -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"model": "test"},
        best_score=1.0,
        optimization_id="opt_finalize",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
    )


def _config(monkeypatch: pytest.MonkeyPatch) -> TraigentConfig:
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    config = TraigentConfig()
    config.offline = False
    config.no_egress = False
    return config


def _backend_client(finalize) -> Mock:
    client = Mock()
    client.submit_result = Mock()
    client.get_session_mapping = Mock(return_value=_Mapping())
    client.update_trial_weighted_scores = AsyncMock(return_value=True)
    client.finalize_session_sync = Mock(side_effect=finalize)
    return client


def _manager(config: TraigentConfig, backend_client: Mock) -> BackendSessionManager:
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"model": ["test"]}
    return BackendSessionManager(
        backend_client=backend_client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt_finalize",
        optimization_status=OptimizationStatus.RUNNING,
    )


def _orchestrator(
    config: TraigentConfig, manager: BackendSessionManager
) -> OptimizationOrchestrator:
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orchestrator._trials = []
    orchestrator._stop_reason = None
    orchestrator._optimization_id = "opt_finalize"
    orchestrator._status = OptimizationStatus.COMPLETED
    orchestrator.optimizer = Mock(objectives=["accuracy"])
    orchestrator.objective_schema = None
    orchestrator.backend_session_manager = manager
    orchestrator.traigent_config = config
    orchestrator.cost_enforcer = Mock(
        get_status=Mock(return_value=SimpleNamespace(accumulated_cost_usd=0.0))
    )
    orchestrator._build_certified_selection_report = Mock(return_value=None)
    orchestrator._submit_usage_analytics = AsyncMock()
    orchestrator._submit_workflow_traces = AsyncMock()
    orchestrator.callback_manager = Mock()
    return orchestrator


@pytest.mark.asyncio
async def test_finalize_retries_transient_failures_then_marks_persistence_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def finalize(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _HTTPError(500)
        return {"status": "completed", "metadata": {"finalized_via_api": True}}

    sleeps: list[float] = []
    monkeypatch.setattr(
        "traigent.core.backend_session_manager.time.sleep", sleeps.append
    )

    config = _config(monkeypatch)
    client = _backend_client(finalize)
    manager = _manager(config, client)
    orchestrator = _orchestrator(config, manager)
    result = _result()

    await OptimizationOrchestrator._finalize_optimization(
        orchestrator, result, "session-finalize", None
    )

    assert calls == 3
    assert sleeps == [1.0, 2.0]
    assert result.metadata["persistence_status"] == "succeeded"
    assert result.persistence_failed is False


def test_finalize_does_not_retry_validation_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)
    client = _backend_client(
        lambda *args, **kwargs: (_ for _ in ()).throw(_HTTPError(400))
    )
    manager = _manager(config, client)

    with pytest.raises(_HTTPError):
        manager.finalize_session("session-finalize", OptimizationStatus.COMPLETED)

    assert client.finalize_session_sync.call_count == 1


@pytest.mark.asyncio
async def test_finalize_terminal_failure_is_loud_but_result_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "traigent.core.backend_session_manager.time.sleep", lambda _delay: None
    )
    config = _config(monkeypatch)
    client = _backend_client(
        lambda *args, **kwargs: (_ for _ in ()).throw(_HTTPError(500))
    )
    manager = _manager(config, client)
    orchestrator = _orchestrator(config, manager)
    result = _result()

    caplog.set_level(logging.ERROR, logger="traigent.core.orchestrator")

    await OptimizationOrchestrator._finalize_optimization(
        orchestrator, result, "session-finalize", None
    )

    assert client.finalize_session_sync.call_count == 3
    assert result.metadata["persistence_status"] == "failed"
    assert result.persistence_failed is True
    assert "HTTP 500" in result.metadata["persistence_error"]
    assert "backend session left RUNNING" in caplog.text
    assert "Run `traigent local sync` or check portal" in caplog.text
