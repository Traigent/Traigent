"""Regression tests for #1939: offline optimize() must persist a SYNCABLE session.

Pre-fix, local session persistence lived exclusively behind the backend
client's fallback storage: the offline orchestrator never initializes a
``BackendIntegratedClient``, so an offline run computed a full result yet
wrote nothing under ``$TRAIGENT_RESULTS_FOLDER/sessions/`` — ``traigent local
list`` said "No sessions found" and ``traigent sync`` reported
``total_sessions=0`` (the documented offline → portal workflow was
unreachable).

The fix gives ``BackendSessionManager`` a first-class ``LocalStorageManager``
handle (independent of any backend client): offline runs mint a local session
at create time, persist every trial, and finalize the record to a terminal
"completed" status so ``traigent sync`` classifies it as sync-ELIGIBLE — not
merely listed.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

import traigent
from traigent.api.types import (
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.function_identity import resolve_function_descriptor

_SPACE = {"x": ["a", "b", "c"]}


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"text": "case-0"}, "ok"),
            EvaluationExample({"text": "case-1"}, "ok"),
        ],
        name="offline_sync_1939",
    )


def _offline_config(monkeypatch, tmp_path) -> TraigentConfig:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    return TraigentConfig()


def _grid_optimizer():
    from traigent.optimizers.grid import GridSearchOptimizer

    return GridSearchOptimizer(dict(_SPACE), ["accuracy"])


def _manager(config: TraigentConfig) -> BackendSessionManager:
    return BackendSessionManager(
        backend_client=None,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=None,
        optimizer=_grid_optimizer(),
        optimization_id="opt-1939",
        optimization_status=OptimizationStatus.RUNNING,
    )


def _trial(trial_id: str, x: str, accuracy: float) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"x": x},
        metrics={"accuracy": accuracy},
        status=TrialStatus.COMPLETED,
        duration=0.01,
        timestamp=datetime.now(UTC),
    )


class TestOfflineSessionManagerPersistence:
    """Manager-level: the local store works with NO backend client at all."""

    def test_offline_create_session_mints_local_session(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _offline_config(monkeypatch, tmp_path)
        manager = _manager(config)

        def func(text):
            return text

        ctx = manager.create_session(
            func=func,
            dataset=_dataset(),
            function_descriptor=resolve_function_descriptor(func),
            max_trials=3,
            start_time=time.time(),
        )

        assert ctx.session_id is not None
        storage = LocalStorageManager(str(tmp_path / "results"))
        session = storage.load_session(ctx.session_id)
        assert session is not None
        assert session.status == "pending"
        assert session.metadata is not None
        assert session.metadata.get("offline") is True
        assert session.metadata.get("evaluation_set") == "offline_sync_1939"
        assert (session.optimization_config or {}).get("search_space") == _SPACE

    @pytest.mark.asyncio
    async def test_offline_submit_trial_persists_locally_without_egress(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _offline_config(monkeypatch, tmp_path)
        manager = _manager(config)

        def func(text):
            return text

        ctx = manager.create_session(
            func=func,
            dataset=_dataset(),
            function_descriptor=resolve_function_descriptor(func),
            max_trials=3,
            start_time=time.time(),
        )
        assert ctx.session_id is not None

        outcome = await manager.submit_trial(_trial("t1", "a", 1.0), ctx.session_id)
        # No remote submission happened (egress disabled, no client) …
        assert outcome is False
        # … but the trial IS durably persisted locally.
        storage = LocalStorageManager(str(tmp_path / "results"))
        session = storage.load_session(ctx.session_id)
        assert session is not None
        assert session.completed_trials == 1
        assert session.trials is not None
        assert session.trials[0].config == {"x": "a"}
        assert session.trials[0].score == 1.0

    @pytest.mark.asyncio
    async def test_finalize_local_session_makes_session_sync_eligible(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _offline_config(monkeypatch, tmp_path)
        manager = _manager(config)

        def func(text):
            return text

        ctx = manager.create_session(
            func=func,
            dataset=_dataset(),
            function_descriptor=resolve_function_descriptor(func),
            max_trials=3,
            start_time=time.time(),
        )
        assert ctx.session_id is not None
        for i, x in enumerate(["a", "b", "c"]):
            await manager.submit_trial(
                _trial(f"t{i}", x, 1.0 if x == "a" else 0.0), ctx.session_id
            )

        manager.finalize_local_session(ctx.session_id, OptimizationStatus.COMPLETED)

        storage = LocalStorageManager(str(tmp_path / "results"))
        session = storage.load_session(ctx.session_id)
        assert session is not None
        assert session.status == "completed"
        assert session.best_config == {"x": "a"}

        # The sync layer must classify it ELIGIBLE, not merely list it.
        sync = SyncManager(config)
        status = sync.get_sync_status()
        assert status["total_sessions"] >= 1
        assert status["sync_eligible"] >= 1


class TestOfflineOptimizeEndToEnd:
    """Full offline run -> completed local session -> sync-ELIGIBLE."""

    @pytest.mark.asyncio
    async def test_offline_grid_run_persists_sync_eligible_session(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
        monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
        monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")

        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_SPACE,
            injection_mode="parameter",
        )
        def answer(text: str, config) -> str:
            return "ok"

        result = await answer.optimize(algorithm="grid")

        assert result.status is OptimizationStatus.COMPLETED
        assert len(result.trials) == len(_SPACE["x"])

        # The syncable local session id is surfaced on the result …
        local_session_id = result.metadata.get("local_session_id")
        assert local_session_id

        # … a COMPLETED session with all trials exists on disk …
        storage = LocalStorageManager(str(tmp_path / "results"))
        sessions = storage.list_sessions()
        assert len(sessions) == 1
        session = sessions[0]
        assert session.session_id == local_session_id
        assert session.status == "completed"
        assert session.completed_trials == len(_SPACE["x"])
        assert session.best_config is not None

        # … and `traigent sync` sees exactly one sync-ELIGIBLE run
        # (issue repro asserted total_sessions=0 here pre-fix).
        config = TraigentConfig.from_environment()
        status = SyncManager(config).get_sync_status()
        assert status["total_sessions"] == 1
        assert status["completed_sessions"] == 1
        assert status["sync_eligible"] == 1

    @pytest.mark.asyncio
    async def test_offline_failed_run_finalizes_local_session_as_failed(
        self, monkeypatch, tmp_path
    ) -> None:
        """A failed offline run must not leave the session stuck at pending
        (and must NOT be counted sync-eligible)."""
        config = _offline_config(monkeypatch, tmp_path)
        manager = _manager(config)

        def func(text):
            return text

        ctx = manager.create_session(
            func=func,
            dataset=_dataset(),
            function_descriptor=resolve_function_descriptor(func),
            max_trials=3,
            start_time=time.time(),
        )
        assert ctx.session_id is not None
        manager.finalize_local_session(ctx.session_id, OptimizationStatus.FAILED)

        storage = LocalStorageManager(str(tmp_path / "results"))
        session = storage.load_session(ctx.session_id)
        assert session is not None
        assert session.status == "failed"
        status = SyncManager(config).get_sync_status()
        assert status["sync_eligible"] == 0
