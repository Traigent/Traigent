"""Regression tests for #1938: connected grid/random must not be truncated by
a backend-side ``optimization_complete``.

Connected ``algorithm="grid"``/``"random"`` resolve to ``LOCAL_ONLY`` intent
with egress enabled: the LOCAL optimizer owns sequencing and stop, and the
backend session is a tracking sink. Pre-fix, the per-trial backend slot
request's ``optimization_complete`` (e.g. ``max_trials_reached``) was turned
into a terminal ``CloudBrainOptimizationComplete`` for ALL algorithms, so the
backend truncated an exhaustive local grid after a handful of trials.

Post-fix, guarded by RESOLVED EXECUTION INTENT (never algorithm-name strings):

* LOCAL_ONLY + online: backend completion is non-terminal — warn once, stop
  all further remote attempts for that session, keep persisting trials
  locally, and mark the result's backend tracking as partial/degraded.
* CLOUD_BRAIN (auto) and cloud-required runs: STRICTLY unchanged — backend
  completion still terminates the run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import traigent.core.backend_session_manager as bsm_module
from traigent.api.types import (
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.config.types import (
    ExecutionMode,
    resolve_execution_policy,
)
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.optimization_pipeline import create_traigent_config
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer

_SPACE = {"x": ["a", "b", "c"], "y": ["1", "2"]}  # 6 grid cells

_EARLY_COMPLETE_MARKER = "continuing the full local enumeration"


class _SlotGatedFakeClient:
    """Backend-client stub whose trial-slot mint closes the session after N."""

    def __init__(self, complete_after: int) -> None:
        self.complete_after = complete_after
        self.slot_calls = 0
        self.submitted: list[dict] = []
        self.no_egress = False
        self.cloud_egress_intent = False
        self.enable_fallback = False
        self.local_storage = None
        auth = Mock()
        auth.has_api_key = Mock(return_value=True)
        self.auth_manager = auth
        self.auth = auth
        self.submit_result = Mock()

    def get_session_mapping(self, session_id):
        return SimpleNamespace(experiment_id="exp-1", experiment_run_id="run-1")

    def request_trial_slot(self, session_id):
        self.slot_calls += 1
        if self.slot_calls > self.complete_after:
            return SimpleNamespace(
                trial_id=None,
                optimization_complete=True,
                reason="max_trials_reached",
            )
        return f"bt-{self.slot_calls}"

    def _submit_trial_result_via_session(self, **kwargs):
        self.submitted.append(kwargs)
        return True


def _config_for(algorithm: str, monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    policy = resolve_execution_policy(algorithm=algorithm, offline=False)
    return create_traigent_config(
        execution_mode=ExecutionMode.LOCAL
        if algorithm in {"grid", "random"}
        else ExecutionMode.HYBRID,
        local_storage_path=None,
        minimal_logging=False,
        privacy_enabled=False,
        execution_policy=policy,
        result_source="explicit_local"
        if algorithm in {"grid", "random"}
        else "cloud_brain",
    )


def _manager(config, fake_client, objectives=None):
    return BackendSessionManager(
        backend_client=fake_client,
        traigent_config=config,
        objectives=objectives or ["accuracy"],
        objective_schema=None,
        optimizer=GridSearchOptimizer(dict(_SPACE), ["accuracy"]),
        optimization_id="opt-1938",
        optimization_status=OptimizationStatus.RUNNING,
    )


def _trial(n: int) -> TrialResult:
    return TrialResult(
        trial_id=f"t{n}",
        config={"x": "a", "y": "1"},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.01,
        timestamp=datetime.now(UTC),
    )


def _count_warnings(monkeypatch) -> list[str]:
    """Record warning messages emitted by the session-manager module logger."""
    messages: list[str] = []
    real_warning = bsm_module.logger.warning

    def _recorder(msg, *args, **kwargs):
        try:
            messages.append(str(msg) % args if args else str(msg))
        except TypeError:
            messages.append(str(msg))
        return real_warning(msg, *args, **kwargs)

    monkeypatch.setattr(bsm_module.logger, "warning", _recorder)
    return messages


class TestLocalSequencingSurvivesBackendCompletion:
    """LOCAL_ONLY + online: backend completion must be non-terminal."""

    @pytest.mark.asyncio
    async def test_backend_complete_is_non_terminal_and_stops_remote_attempts(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("grid", monkeypatch, tmp_path)
        fake = _SlotGatedFakeClient(complete_after=2)
        manager = _manager(config, fake)
        warnings = _count_warnings(monkeypatch)

        # Trials 1-2: tracked normally.
        for n in (1, 2):
            outcome = await manager.submit_trial(_trial(n), "bs-1938")
            assert getattr(outcome, "optimization_complete", False) is False
        assert len(fake.submitted) == 2

        # Trial 3: backend closes the session — the outcome must be
        # NON-terminal so the orchestrator keeps enumerating the grid.
        outcome = await manager.submit_trial(_trial(3), "bs-1938")
        assert getattr(outcome, "optimization_complete", False) is False
        assert manager.backend_remote_early_complete is True
        assert manager.backend_remote_early_complete_reason == "max_trials_reached"

        # Trials 4-5: NO further remote slot/result attempts (do not hammer a
        # server-side-closed session) — slot_calls frozen at 3.
        for n in (4, 5):
            outcome = await manager.submit_trial(_trial(n), "bs-1938")
            assert getattr(outcome, "optimization_complete", False) is False
        assert fake.slot_calls == 3
        assert len(fake.submitted) == 2

        # Warn exactly ONCE across the whole degradation.
        early_warnings = [m for m in warnings if _EARLY_COMPLETE_MARKER in m]
        assert len(early_warnings) == 1

        # Trials kept persisting locally (via the client's fallback storage
        # shim) even after the backend closed the session.
        assert fake.submit_result.call_count == 5

    @pytest.mark.asyncio
    async def test_remote_finalize_and_weighted_updates_skipped_after_complete(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("grid", monkeypatch, tmp_path)
        fake = _SlotGatedFakeClient(complete_after=0)
        # Two objectives so update_weighted_scores would engage if not gated;
        # the fake has NO update/finalize endpoints — an ungated call raises.
        manager = _manager(config, fake, objectives=["accuracy", "cost"])

        await manager.submit_trial(_trial(1), "bs-1938")
        assert manager.backend_remote_early_complete is True

        # Gated: no attribute access on the fake's missing finalize/update
        # endpoints, no exception, and a None/0 result.
        assert manager.finalize_session("bs-1938", OptimizationStatus.COMPLETED) is None
        assert await manager.update_weighted_scores(Mock(), "bs-1938") == 0
        assert manager.build_session_aggregation_payload(Mock(), "bs-1938") is None


class TestCloudBrainCompletionUnchanged:
    """CLOUD_BRAIN (auto) keeps full backend authority over completion."""

    @pytest.mark.asyncio
    async def test_cloud_brain_run_still_terminates_on_backend_complete(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("auto", monkeypatch, tmp_path)
        fake = _SlotGatedFakeClient(complete_after=0)
        manager = _manager(config, fake)

        outcome = await manager.submit_trial(_trial(1), "bs-auto")

        # Terminal outcome, exactly as before the #1938 fix.
        assert getattr(outcome, "optimization_complete", False) is True
        assert outcome.reason == "max_trials_reached"
        assert manager.backend_remote_early_complete is False


class TestPlannedBackendMaxTrials:
    """#1938 wire budget: send the PLANNED LOCAL TRIAL COUNT, never an
    over-/under-stated value."""

    def _stub(self, config, optimizer, max_trials):
        stub = SimpleNamespace(
            traigent_config=config,
            optimizer=optimizer,
            max_trials=max_trials,
        )
        stub._local_optimizer_planned_trials = lambda: (
            OptimizationOrchestrator._local_optimizer_planned_trials(stub)
        )
        return stub

    def test_grid_without_cap_sends_exhaustive_grid_size(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("grid", monkeypatch, tmp_path)
        optimizer = GridSearchOptimizer(dict(_SPACE), ["accuracy"])  # 6 cells
        stub = self._stub(config, optimizer, max_trials=None)
        assert OptimizationOrchestrator._planned_backend_max_trials(stub) == 6

    def test_grid_explicit_binding_cap_respected_verbatim(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("grid", monkeypatch, tmp_path)
        optimizer = GridSearchOptimizer(dict(_SPACE), ["accuracy"])
        stub = self._stub(config, optimizer, max_trials=4)
        assert OptimizationOrchestrator._planned_backend_max_trials(stub) == 4

    def test_grid_oversized_cap_never_overstates_the_plan(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("grid", monkeypatch, tmp_path)
        optimizer = GridSearchOptimizer(dict(_SPACE), ["accuracy"])
        stub = self._stub(config, optimizer, max_trials=30)
        # The local grid exhausts at 6 — 30 would overstate the plan.
        assert OptimizationOrchestrator._planned_backend_max_trials(stub) == 6

    def test_random_without_cap_sends_configured_trial_count(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("random", monkeypatch, tmp_path)
        optimizer = RandomSearchOptimizer(dict(_SPACE), ["accuracy"], max_trials=25)
        stub = self._stub(config, optimizer, max_trials=None)
        assert OptimizationOrchestrator._planned_backend_max_trials(stub) == 25

    def test_cloud_brain_passes_user_value_through_untouched(
        self, monkeypatch, tmp_path
    ) -> None:
        config = _config_for("auto", monkeypatch, tmp_path)
        optimizer = GridSearchOptimizer(dict(_SPACE), ["accuracy"])
        stub = self._stub(config, optimizer, max_trials=30)
        assert OptimizationOrchestrator._planned_backend_max_trials(stub) == 30
        stub_none = self._stub(config, optimizer, max_trials=None)
        assert OptimizationOrchestrator._planned_backend_max_trials(stub_none) is None
