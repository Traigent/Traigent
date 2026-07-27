"""Regression tests for #2020: the result must carry the id `traigent sync` takes.

Pre-fix, a run that was computed locally (no API key, offline, degraded, …)
persisted a perfectly syncable local session but handed the caller only an
``optimization_id`` — which ``traigent sync`` rejects. The documented
"run now, upload later" flow therefore required either ``traigent sync --all``
(whole-store blast radius) or archaeology in ``~/.traigent/sessions/``.

``OptimizationResult.sync_session_id`` closes that: it is the exact argument
``traigent sync <SESSION_ID>`` accepts for THIS run, and it is only populated
when the local store actually holds a record under that id.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import traigent
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.env_config import is_backend_offline
from traigent.utils.exceptions import TraigentStorageError

_SPACE = {"x": ["a", "b"]}


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"text": "case-0"}, "ok"),
            EvaluationExample({"text": "case-1"}, "ok"),
        ],
        name="sync_session_id_2020",
    )


def _isolated_env(monkeypatch, tmp_path) -> None:
    """No key, no network, no LLM spend, and a private results folder.

    Offline mode is explicitly turned OFF here. ``tests/conftest.py``'s autouse
    ``jwt_development_mode`` fixture forces ``TRAIGENT_OFFLINE_MODE=true`` for
    everything outside ``tests/unit/cloud/``, which would silently reroute the
    no-key test through the offline branch instead of the local-fallback branch
    it exists to cover. Both flags ``is_backend_offline()`` reads are cleared
    (``env_config.is_backend_offline``); the offline test below opts back in.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


async def _run_grid(local_storage_path: str | None = None):
    options = {}
    if local_storage_path is not None:
        options["local_storage_path"] = local_storage_path

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space=_SPACE,
        injection_mode="parameter",
        **options,
    )
    def answer(text: str, config) -> str:
        return "ok"

    return await answer.optimize(algorithm="grid")


@pytest.mark.asyncio
async def test_no_key_run_exposes_syncable_session_id(monkeypatch, tmp_path) -> None:
    """The issue repro: no API key, so the local store owns the run."""
    _isolated_env(monkeypatch, tmp_path)

    result = await _run_grid()

    assert result.status is OptimizationStatus.COMPLETED
    # Path guard. tests/conftest.py's autouse fixture forces offline mode on for
    # everything outside tests/unit/cloud/, so without _isolated_env turning it
    # back off this test silently duplicated the offline case below instead of
    # covering the no-key local-fallback branch it is named for.
    assert is_backend_offline() is False
    # A connected grid run is stamped `explicit_local` too, which is exactly why
    # the predicate cannot key on `source` — the proof that this is the no-key
    # FALLBACK branch is on the stored record, asserted below.
    assert result.source == "explicit_local"
    # Pre-fix this attribute did not exist (AttributeError).
    assert result.sync_session_id
    # The whole point: it is NOT the optimization_id, which `sync` rejects.
    assert result.sync_session_id != result.optimization_id

    # The id names a real record in the store `traigent sync` reads …
    store = tmp_path / "results"
    session_files = list((store / "sessions").glob("*.json"))
    assert len(session_files) == 1
    assert session_files[0].stem == result.sync_session_id

    storage = LocalStorageManager(str(store))
    session = storage.load_session(result.sync_session_id)
    assert session is not None
    assert (session.metadata or {}).get("optimization_id") == result.optimization_id
    # … a record written by the no-key local-fallback session path (NOT the
    # offline path): the run tried to create a backend session, had no
    # credential, and fell back to local storage.
    assert (session.metadata or {}).get("execution_mode") == "local_fallback"
    assert (session.metadata or {}).get("backend_fallback") is True

    # … it is mirrored in metadata for the #1939 consumers …
    assert result.metadata["local_session_id"] == result.sync_session_id

    # … and `traigent sync <id>` accepts it (dry-run needs no API key).
    sync = SyncManager(TraigentConfig.from_environment())
    outcome = sync.sync_session_to_cloud(result.sync_session_id, dry_run=True)
    assert outcome["status"] == "success"
    assert outcome["trials_converted"] == len(result.trials)


@pytest.mark.asyncio
async def test_offline_run_exposes_syncable_session_id(monkeypatch, tmp_path) -> None:
    """The one offline test — the #1939 behavior this change must not lose."""
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")

    result = await _run_grid()

    assert is_backend_offline() is True
    assert result.sync_session_id
    assert result.sync_session_id == result.metadata["local_session_id"]


@pytest.mark.asyncio
async def test_custom_storage_path_still_yields_id_but_warns_about_the_store(
    monkeypatch, tmp_path, caplog
) -> None:
    """The id is STORE-RELATIVE, and the SDK has to say so.

    ``local_storage_path`` is a supported public ``@traigent.optimize(...)``
    option, but it is programmatic-only: ``traigent sync`` is a separate process
    whose ``SyncManager`` rebuilds the store from
    ``TraigentConfig.from_environment()``. So a run with a custom root produces
    an id that is correct for its own store and that a default CLI invocation
    rejects — #2020's own failure mode in a narrower case.

    The id is NOT withheld (it is right, and the caller can point the CLI at the
    root); instead the mismatch is announced with the export that fixes it.
    """
    _isolated_env(monkeypatch, tmp_path)
    custom_store = tmp_path / "custom_store"
    # The env var stays pointed somewhere else, so what a fresh `traigent sync`
    # resolves genuinely differs from where this run writes.
    env_store = tmp_path / "results"
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(env_store))

    with caplog.at_level(logging.WARNING):
        result = await _run_grid(local_storage_path=str(custom_store))

    assert result.status is OptimizationStatus.COMPLETED
    # Still populated — a store-relative id is useful, not garbage.
    assert result.sync_session_id

    # It names a record in THIS RUN's store …
    session_files = list((custom_store / "sessions").glob("*.json"))
    assert len(session_files) == 1
    assert session_files[0].stem == result.sync_session_id
    storage = LocalStorageManager(str(custom_store))
    assert storage.load_session(result.sync_session_id) is not None

    # … and the warning names the root the caller must export.
    assert "TRAIGENT_RESULTS_FOLDER" in caplog.text
    assert str(custom_store.resolve()) in caplog.text
    assert result.sync_session_id in caplog.text

    # The reproduction the warning exists for: a default CLI process rejects it.
    default_sync = SyncManager(TraigentConfig.from_environment())
    assert default_sync.storage.storage_path == env_store.resolve()
    with pytest.raises(TraigentStorageError):
        default_sync.sync_session_to_cloud(result.sync_session_id, dry_run=True)

    # … and doing what the warning says makes it work.
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(custom_store))
    pointed_sync = SyncManager(TraigentConfig.from_environment())
    outcome = pointed_sync.sync_session_to_cloud(result.sync_session_id, dry_run=True)
    assert outcome["status"] == "success"


@pytest.mark.asyncio
async def test_default_store_run_emits_no_store_mismatch_warning(
    monkeypatch, tmp_path, caplog
) -> None:
    """The mismatch warning must be silent on a normal run.

    Same shape as ``test_no_key_run_exposes_syncable_session_id`` — no custom
    ``local_storage_path``, so the run's store IS the one a fresh
    ``traigent sync`` resolves — and therefore nothing to warn about.
    """
    _isolated_env(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING):
        result = await _run_grid()

    assert result.sync_session_id
    assert "TRAIGENT_RESULTS_FOLDER" not in caplog.text
    assert "will not find this run as-is" not in caplog.text


def _connected_orchestrator(config, manager):
    """Minimal orchestrator standing in for a connected, tracking-enabled run."""
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orchestrator._trials = []
    orchestrator._stop_reason = None
    orchestrator._optimization_id = "opt-2020-persist-fail"
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
async def test_failed_backend_persistence_withholds_syncable_session_id(
    monkeypatch, tmp_path
) -> None:
    """A deliberate, recorded gap — do not "fix" this back without a repair path.

    The persistence-failure handler logs "Run ``traigent local sync`` ... to
    finalize or repair the session", and a local record for this session does
    exist (#1279) — so it is tempting to hand out ``session_id`` here and spare
    the user a whole-store sync. We deliberately do NOT:

    * An exception out of ``finalize_session`` does not prove the backend
      dropped the run. A timeout, a connection reset, or a response-decode
      failure can all follow a write the backend fully committed, so "the
      backend is not the system of record" is an inference this path cannot
      make. Acting on it would tell the user to re-import a run the backend
      already holds — a duplicate experiment on the portal, not a repair.
    * Even a definitively-unacknowledged finalize would not help: ``traigent
      sync`` imports a whole session, it cannot repair a half-finalized backend
      one.

    So ``sync_session_id`` stays ``None`` for this shape and the existing
    whole-store advice in the log stands. A targeted repair path is follow-up
    work; the gap is recorded in the CHANGELOG's known-gaps note.
    """
    _isolated_env(monkeypatch, tmp_path)

    config = TraigentConfig()
    config.offline = False
    config.no_egress = False
    config.execution_mode = "hybrid"

    session_id = "sess-2020-persist-fail"
    # The connected-run local mirror (#1279) that `traigent local sync` repairs.
    storage = LocalStorageManager(config.get_local_storage_path())
    storage.create_session("answer", session_id=session_id)

    backend_client = Mock()
    backend_client.get_session_mapping = Mock(return_value=None)
    backend_client.update_trial_weighted_scores = AsyncMock(return_value=True)
    backend_client.finalize_session_sync = Mock(
        side_effect=RuntimeError("HTTP 500 finalize exploded")
    )
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"x": ["a"]}
    manager = BackendSessionManager(
        backend_client=backend_client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt-2020-persist-fail",
        optimization_status=OptimizationStatus.RUNNING,
    )
    orchestrator = _connected_orchestrator(config, manager)

    result = OptimizationResult(
        trials=[],
        best_config={"x": "a"},
        best_score=1.0,
        optimization_id="opt-2020-persist-fail",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )

    await OptimizationOrchestrator._finalize_optimization(
        orchestrator, result, session_id, None
    )

    # The run still looks connected and tracking-enabled, so the locality half
    # of the predicate is False and nothing downstream overrides it.
    assert manager.backend_tracking_enabled is True
    assert manager._runtime_degraded is False
    assert result.metadata["persistence_status"] == "failed"
    # The record the tempting "upgrade" would have pointed at really does exist
    # — withholding the id is a decision, not an accident of missing state.
    assert manager.local_session_record_exists(session_id) is True
    assert result.sync_session_id is None
    assert "local_session_id" not in result.metadata


def _trial(
    index: int, *, abandoned: bool = False, constraint_rejected: bool = False
) -> TrialResult:
    """A trial as it lands in ``result.trials``.

    ``abandoned`` mirrors ``_abandon_optuna_trial``'s marker; ``constraint_rejected``
    mirrors ``TrialLifecycleManager._record_pre_constraint_pruned_result``. Both
    append to ``orchestrator._trials`` and are never submitted to the backend —
    note they share no metadata key, which is why the predicate no longer tries
    to recognize the family by metadata shape.
    """
    metadata: dict[str, object] = {}
    if abandoned:
        metadata["abandoned"] = True
    if constraint_rejected:
        metadata["constraint_rejected"] = True
        metadata["stop_reason"] = "trial_rejected_by_constraint"
    never_submitted = abandoned or constraint_rejected
    return TrialResult(
        trial_id=f"trial-{index}",
        config={"x": "a"},
        metrics={"accuracy": 1.0},
        status=TrialStatus.PRUNED if never_submitted else TrialStatus.COMPLETED,
        duration=0.01,
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )


def _partially_tracked_run(config, session_id, *, attempted, acknowledged):
    """A connected, tracking-enabled manager whose backend saw only some trials.

    ``attempted`` is what ``_log_trial_to_backend`` records just before it posts;
    ``acknowledged`` is what came back accepted. The first ``acknowledged``
    attempts are the ones that landed, so ``attempted - acknowledged`` is the
    number of trials the backend is genuinely missing.
    """
    backend_client = Mock()
    backend_client.get_session_mapping = Mock(return_value=None)
    backend_client.update_trial_weighted_scores = AsyncMock(return_value=True)
    backend_client.finalize_session_sync = Mock(
        side_effect=RuntimeError("HTTP 500 finalize exploded")
    )
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"x": ["a"]}
    manager = BackendSessionManager(
        backend_client=backend_client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt-2020-partial",
        optimization_status=OptimizationStatus.RUNNING,
    )
    manager._attempted_trials = {
        (session_id, f"backend-trial-{i}") for i in range(attempted)
    }
    manager._acknowledged_trials = {
        (session_id, f"backend-trial-{i}") for i in range(acknowledged)
    }
    return manager


def _result(trials):
    return OptimizationResult(
        trials=list(trials),
        best_config={"x": "a"},
        best_score=1.0,
        optimization_id="opt-2020-partial",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_partial_acknowledgement_keeps_its_id_when_finalize_also_fails(
    monkeypatch, tmp_path
) -> None:
    """Pins the ASSIGNMENT ORDERING the CHANGELOG's known-gap note describes.

    A finalize failure does not by itself produce an id (see
    ``test_failed_backend_persistence_withholds_syncable_session_id``) — but it
    does not take one away either. The id is assigned *before* the backend
    finalize block and is never cleared, so a run that already qualified under
    another locality clause keeps it. Here both are true at once: the backend
    acknowledged 1 of 3 trials AND finalize blew up.

    Keeping the id is the right call for this combined shape: the partial
    acknowledgement independently proves the backend is missing trials 2 and 3,
    which the finalize failure neither establishes nor refutes.
    """
    _isolated_env(monkeypatch, tmp_path)

    config = TraigentConfig()
    config.offline = False
    config.no_egress = False
    config.execution_mode = "hybrid"

    session_id = "sess-2020-partial-and-finalize-fail"
    storage = LocalStorageManager(config.get_local_storage_path())
    storage.create_session("answer", session_id=session_id)

    manager = _partially_tracked_run(config, session_id, attempted=3, acknowledged=1)
    orchestrator = _connected_orchestrator(config, manager)
    result = _result([_trial(0), _trial(1), _trial(2)])

    await OptimizationOrchestrator._finalize_optimization(
        orchestrator, result, session_id, None
    )

    assert result.metadata["persistence_status"] == "failed"
    assert result.sync_session_id == session_id
    assert result.metadata["local_session_id"] == session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["abandoned", "constraint_rejected"])
async def test_never_submitted_trials_do_not_inflate_the_unacknowledged_denominator(
    monkeypatch, tmp_path, label
) -> None:
    """Trials the SDK never submits must not read as "missing from the backend".

    Two writers append to ``orchestrator._trials`` without ever posting the
    trial, and they share NO metadata key:

    * ``_abandon_optuna_trial`` — ``metadata["abandoned"] = True``. Currently
      unreachable (the abandon paths need ``_optuna_trial_id`` on a trial and no
      shipping optimizer emits one), kept correct for the ask/tell residual.
    * ``TrialLifecycleManager._record_pre_constraint_pruned_result`` —
      ``metadata["constraint_rejected"] = True`` /
      ``stop_reason="trial_rejected_by_constraint"``. LIVE: it fires for every
      config a pre-eval ``constraints=[...]`` predicate rejects, a documented
      public ``@optimize`` option.

    Counting either against the acknowledgements reads a fully-tracked connected
    run as partially acknowledged and hands out a sync id for a run the portal
    already holds — importing it a second time as a duplicate experiment. The
    predicate therefore measures the SDK's own submission ATTEMPTS, which no
    future member of this family can inflate.
    """
    _isolated_env(monkeypatch, tmp_path)

    config = TraigentConfig()
    config.offline = False
    config.no_egress = False
    config.execution_mode = "hybrid"

    session_id = f"sess-2020-{label}"
    storage = LocalStorageManager(config.get_local_storage_path())
    storage.create_session("answer", session_id=session_id)

    # Two real trials submitted and acknowledged; one never-submitted trial in
    # `result.trials`. The backend holds everything it could ever hold.
    manager = _partially_tracked_run(config, session_id, attempted=2, acknowledged=2)
    orchestrator = _connected_orchestrator(config, manager)
    result = _result([_trial(0), _trial(1), _trial(2, **{label: True})])

    await OptimizationOrchestrator._finalize_optimization(
        orchestrator, result, session_id, None
    )

    # The record exists — the None is earned by the predicate, not by a failed
    # durability probe.
    assert manager.local_session_record_exists(session_id) is True
    assert result.sync_session_id is None
    assert "local_session_id" not in result.metadata


@pytest.mark.asyncio
async def test_sync_rejects_optimization_id_with_actionable_message(
    monkeypatch, tmp_path
) -> None:
    """The optimization_id is still rejected — but now it says what to pass."""
    _isolated_env(monkeypatch, tmp_path)

    result = await _run_grid()

    sync = SyncManager(TraigentConfig.from_environment())
    with pytest.raises(TraigentStorageError) as excinfo:
        sync.sync_session_to_cloud(result.optimization_id, dry_run=True)

    message = str(excinfo.value)
    assert "traigent local list" in message
    assert "sync_session_id" in message
