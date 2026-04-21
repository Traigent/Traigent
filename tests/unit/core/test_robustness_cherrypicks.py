"""Tests for the three robustness cherry-picks from the trial-progress patch.

Covers:

1. ``BackendSessionManager._started_trials`` — idempotent
   ``register_trial_start`` so retries of ``submit_trial`` don't re-hit the
   backend registration endpoint.
2. ``SessionOperations.finalize_session`` — session-mapping recovery
   from active-session metadata when the in-memory bridge entry is lost.
3. ``CustomEvaluatorWrapper`` — blocking custom evaluators are executed via
   ``asyncio.to_thread`` so backend progress updates / heartbeats running on
   the main event loop are not starved.

The three fixes were lifted from a larger feature patch; the tests here
intentionally exercise only the robustness surface, not the live-progress
streaming feature that the patch also introduced.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives


@pytest.fixture
def mock_backend_client() -> MagicMock:
    client = MagicMock()
    client.register_trial_start = AsyncMock(return_value=True)
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    client.get_session_mapping = Mock(
        return_value=MagicMock(experiment_run_id="run_test_123")
    )
    auth_manager = Mock()
    auth_manager.has_api_key = Mock(return_value=True)
    client.auth_manager = auth_manager
    return client


@pytest.fixture
def traigent_config() -> TraigentConfig:
    config = TraigentConfig()
    config.execution_mode = "edge_analytics"
    return config


@pytest.fixture
def objective_schema():
    return create_default_objectives(
        objective_names=["accuracy"],
        orientations={"accuracy": "maximize"},
    )


@pytest.fixture
def mock_optimizer() -> MagicMock:
    optimizer = MagicMock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"model": ["gpt-4o-mini"]}
    return optimizer


@pytest.fixture
def manager(
    mock_backend_client: MagicMock,
    traigent_config: TraigentConfig,
    objective_schema,
    mock_optimizer: MagicMock,
) -> BackendSessionManager:
    return BackendSessionManager(
        backend_client=mock_backend_client,
        traigent_config=traigent_config,
        objectives=["accuracy"],
        objective_schema=objective_schema,
        optimizer=mock_optimizer,
        optimization_id="opt-1",
        optimization_status=OptimizationStatus.RUNNING,
    )


def _trial_result(trial_id: str = "trial-1") -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": "gpt-4o-mini"},
        metrics={"accuracy": 0.9},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


class TestStartedTrialsIdempotency:
    """Second submit_trial call for the same (session, trial) pair must not
    re-register the trial start with the backend."""

    @pytest.mark.asyncio
    async def test_register_trial_start_runs_once_across_retries(
        self, manager: BackendSessionManager, mock_backend_client: MagicMock
    ) -> None:
        session_id = "sess-1"
        tr = _trial_result()

        with patch.object(
            manager, "_should_suppress_backend_warnings", return_value=False
        ):
            await manager.submit_trial(tr, session_id)
            await manager.submit_trial(tr, session_id)

        assert mock_backend_client.register_trial_start.await_count == 1
        assert (session_id, tr.trial_id) in manager._started_trials

    @pytest.mark.asyncio
    async def test_failed_registration_not_marked_as_started(
        self, manager: BackendSessionManager, mock_backend_client: MagicMock
    ) -> None:
        mock_backend_client.register_trial_start = AsyncMock(return_value=False)

        session_id = "sess-2"
        tr = _trial_result("trial-2")

        with patch.object(
            manager, "_should_suppress_backend_warnings", return_value=False
        ):
            await manager.submit_trial(tr, session_id)
            await manager.submit_trial(tr, session_id)

        assert mock_backend_client.register_trial_start.await_count == 2
        assert (session_id, tr.trial_id) not in manager._started_trials


class TestSessionMappingRecovery:
    """finalize_session must try to rebuild a session mapping from the
    active-session metadata before giving up on remote finalization."""

    @pytest.mark.asyncio
    async def test_recovers_mapping_from_active_session_metadata(self) -> None:
        from traigent.cloud.backend_client import BackendIntegratedClient

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            client = BackendIntegratedClient(
                api_key="test-key",  # pragma: allowlist secret
                base_url="http://localhost:5000",
            )

        session_id = "sess-recovery"
        active_session = MagicMock()
        active_session.metadata = {
            "experiment_id": "exp-42",
            "experiment_run_id": "run-42",
        }
        active_session.function_name = "my_fn"
        active_session.configuration_space = {"model": ["gpt-4o"]}
        active_session.objectives = ["accuracy"]

        with client._active_sessions_lock:
            client._active_sessions[session_id] = active_session

        client._session_ops._finalize_session_via_api = AsyncMock(return_value=True)

        assert client.session_bridge.get_session_mapping(session_id) is None

        await client._session_ops.finalize_session(session_id)

        mapping = client.session_bridge.get_session_mapping(session_id)
        assert mapping is not None
        assert mapping.experiment_id == "exp-42"
        assert mapping.experiment_run_id == "run-42"
        client._session_ops._finalize_session_via_api.assert_awaited_once_with(
            session_id, "run-42"
        )

    @pytest.mark.asyncio
    async def test_no_recovery_when_active_session_metadata_incomplete(self) -> None:
        from traigent.cloud.backend_client import BackendIntegratedClient

        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            client = BackendIntegratedClient(
                api_key="test-key",  # pragma: allowlist secret
                base_url="http://localhost:5000",
            )

        session_id = "sess-no-meta"
        active_session = MagicMock()
        active_session.metadata = {}

        with client._active_sessions_lock:
            client._active_sessions[session_id] = active_session

        client._session_ops._finalize_session_via_api = AsyncMock(return_value=True)

        await client._session_ops.finalize_session(session_id)

        assert client.session_bridge.get_session_mapping(session_id) is None
        client._session_ops._finalize_session_via_api.assert_not_called()


class TestBlockingEvaluatorOffloading:
    """Blocking custom evaluators must run via asyncio.to_thread so concurrent
    async work (heartbeats, progress callbacks) is not starved."""

    @pytest.mark.asyncio
    async def test_to_thread_does_not_block_event_loop(self) -> None:
        """A 200ms blocking call must overlap a parallel 50ms-cadence
        heartbeat without delaying its ticks. If run inline on the event
        loop, heartbeats would bunch at the end."""
        heartbeats: list[float] = []

        async def heartbeat() -> None:
            for _ in range(5):
                heartbeats.append(time.monotonic())
                await asyncio.sleep(0.05)

        def blocking_work() -> str:
            time.sleep(0.2)
            return "done"

        async def offloaded_call() -> str:
            return await asyncio.to_thread(blocking_work)

        start = time.monotonic()
        _, result = await asyncio.gather(heartbeat(), offloaded_call())
        elapsed = time.monotonic() - start

        assert result == "done"
        assert len(heartbeats) == 5
        # Sequential would be ~0.45s (0.2 blocking + 0.25 heartbeat). With
        # to_thread they overlap and total ~0.25-0.3s.
        assert elapsed < 0.4, f"elapsed={elapsed:.3f}s suggests event-loop starvation"

    def test_evaluator_wrapper_wraps_blocking_evaluator_in_to_thread(self) -> None:
        """Source-level guard against accidental regression: the wrapper's
        sync-evaluator branch must stay on asyncio.to_thread."""
        from traigent.core import evaluator_wrapper

        source = inspect.getsource(evaluator_wrapper.CustomEvaluatorWrapper)
        assert "asyncio.to_thread(" in source, (
            "CustomEvaluatorWrapper must use asyncio.to_thread to run blocking "
            "custom evaluators off the main event loop"
        )
