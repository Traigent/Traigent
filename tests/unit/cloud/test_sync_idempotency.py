"""Tests for idempotent cloud sync of local optimization runs (Sync v2).

Covers the W&B-style guarantees added to ``SyncManager`` + the local
``sync_state`` marker: re-syncing an unchanged run is a no-op (no duplicate
cloud experiments), sync state is persisted for status reporting, a changed run
is re-synced, and free-text trial metadata never rides to the backend.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture
def storage(tmp_path) -> LocalStorageManager:
    return LocalStorageManager(str(tmp_path / "store"))


@pytest.fixture
def sync_manager(storage) -> SyncManager:
    sm = SyncManager(TraigentConfig.from_environment(), api_key="test-key")
    sm.storage = storage  # isolate to the temp store
    return sm


def _make_completed_session(storage: LocalStorageManager) -> str:
    session_id = storage.create_session(
        "answer_question",
        optimization_config={"search_space": {"model": ["a", "b"], "temp": [0.0, 1.0]}},
    )
    storage.add_trial_result(
        session_id, config={"model": "a", "temp": 0.0}, score=0.8, cost=0.001
    )
    storage.add_trial_result(
        session_id, config={"model": "b", "temp": 1.0}, score=0.9, cost=0.002
    )
    storage.finalize_session(session_id, "completed")
    return session_id


def _stub_backend_success(sync_manager: SyncManager) -> dict[str, Mock]:
    """Patch the legacy resource uploads so a sync 'succeeds' with no HTTP."""
    mocks = {
        "_sync_agent": Mock(return_value={"success": True, "agent_id": "ag1"}),
        "_sync_benchmark": Mock(return_value={"success": True, "benchmark_id": "bm1"}),
        "_sync_experiment": Mock(
            return_value={"success": True, "experiment_id": "exp1"}
        ),
        "_sync_experiment_run": Mock(
            return_value={"success": True, "experiment_run_id": "run1"}
        ),
    }

    def _runs(_run_id, configuration_runs):
        return {"success": True, "synced": len(configuration_runs), "errors": []}

    mocks["_sync_configuration_runs"] = Mock(side_effect=_runs)
    for name, mock in mocks.items():
        setattr(sync_manager, name, mock)
    return mocks


# --------------------------------------------------------------------------- #
# local_storage.update_sync_state
# --------------------------------------------------------------------------- #


def test_update_sync_state_merges_and_persists(storage):
    sid = _make_completed_session(storage)

    storage.update_sync_state(sid, {"status": "synced", "cloud_experiment_id": "e1"})
    storage.update_sync_state(
        sid, {"cloud_url": "https://x"}, trial_updates={"1": {"status": "uploaded"}}
    )

    reloaded = storage.load_session(sid)
    assert reloaded.sync_state["status"] == "synced"  # preserved across merges
    assert reloaded.sync_state["cloud_experiment_id"] == "e1"
    assert reloaded.sync_state["cloud_url"] == "https://x"
    assert reloaded.sync_state["trials"]["1"]["status"] == "uploaded"


def test_sync_state_survives_round_trip_on_old_sessions(storage):
    """A session written without sync_state loads fine (backward compatible)."""
    sid = _make_completed_session(storage)
    reloaded = storage.load_session(sid)
    assert reloaded.sync_state is None


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def test_first_sync_records_synced_state(sync_manager):
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "success"
    state = sync_manager.storage.load_session(sid).sync_state
    assert state["status"] == "synced"
    assert state["payload_hash"] == result["payload_hash"]
    assert state["cloud_experiment_id"] == "exp1"
    assert state["attempts"] == 1
    mocks["_sync_agent"].assert_called_once()


def test_resync_unchanged_is_noop(sync_manager):
    """Re-syncing an unchanged, already-synced run must NOT create duplicates."""
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)

    sync_manager.sync_session_to_cloud(sid)
    for mock in mocks.values():
        mock.reset_mock()

    second = sync_manager.sync_session_to_cloud(sid)

    assert second["status"] == "already_synced"
    assert second["cloud_experiment_id"] == "exp1"
    # No backend resource was created the second time → no duplicate experiment.
    mocks["_sync_experiment"].assert_not_called()
    mocks["_sync_agent"].assert_not_called()


def test_force_reuploads_even_when_synced(sync_manager):
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    sync_manager.sync_session_to_cloud(sid)
    for mock in mocks.values():
        mock.reset_mock()

    forced = sync_manager.sync_session_to_cloud(sid, force=True)

    assert forced["status"] == "success"
    mocks["_sync_experiment"].assert_called_once()


def test_changed_session_is_resynced(sync_manager):
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    sync_manager.sync_session_to_cloud(sid)
    for mock in mocks.values():
        mock.reset_mock()

    # New trial changes the payload fingerprint → no longer "already synced".
    sync_manager.storage.add_trial_result(
        sid, config={"model": "a", "temp": 1.0}, score=0.95
    )
    sync_manager.storage.finalize_session(sid, "completed")

    result = sync_manager.sync_session_to_cloud(sid)
    assert result["status"] == "success"
    mocks["_sync_experiment"].assert_called_once()


def test_partial_failure_then_resume_reuses_experiment(sync_manager):
    """BLOCKER regression: a retry after a partial failure must reuse the
    cloud experiment, never create a duplicate one."""
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    # First attempt: experiment is created but the run step fails → partial.
    mocks["_sync_experiment_run"].return_value = {"success": False, "error": "boom"}

    first = sync_manager.sync_session_to_cloud(sid)
    assert first["status"] == "partial"
    state = sync_manager.storage.load_session(sid).sync_state
    assert state["status"] == "partial"
    assert state["cloud_experiment_id"] == "exp1"  # experiment was created

    # Second attempt (same content): run now succeeds.
    mocks["_sync_experiment_run"].return_value = {
        "success": True,
        "experiment_run_id": "run1",
    }
    for name in ("_sync_agent", "_sync_benchmark", "_sync_experiment"):
        mocks[name].reset_mock()

    second = sync_manager.sync_session_to_cloud(sid)

    assert second["status"] == "success"
    # The experiment was REUSED, not re-created → no duplicate.
    mocks["_sync_experiment"].assert_not_called()
    assert second["cloud_experiment_id"] == "exp1"


def test_cleanup_skips_delete_when_backup_fails(sync_manager, monkeypatch):
    """`--clean` must never delete a run whose backup failed."""
    sid = _make_completed_session(sync_manager.storage)
    monkeypatch.setattr(sync_manager.storage, "export_session", lambda *a, **k: False)

    result = sync_manager.cleanup_after_sync([sid], keep_backup=True)

    assert result["sessions_deleted"] == 0
    assert any("backup" in e.lower() for e in result["errors"])
    # The run is still on disk.
    assert sync_manager.storage.load_session(sid) is not None


def test_force_all_threads_force(sync_manager):
    captured = {}

    def fake_sync(session_id, dry_run=False, force=False):
        captured["force"] = force
        return {"session_id": session_id, "status": "success"}

    _make_completed_session(sync_manager.storage)
    with patch.object(sync_manager, "sync_session_to_cloud", side_effect=fake_sync):
        sync_manager.sync_all_sessions(force=True)

    assert captured["force"] is True


def test_load_session_tolerates_unknown_future_keys(storage):
    """A session file written by a newer SDK (extra keys) still loads."""
    sid = _make_completed_session(storage)
    session_file = storage.storage_path / "sessions" / f"{sid}.json"
    data = json.loads(session_file.read_text())
    data["some_future_field"] = {"added": "by a newer version"}
    data["trials"][0]["future_trial_field"] = 123
    session_file.write_text(json.dumps(data))

    reloaded = storage.load_session(sid)
    assert reloaded is not None
    assert reloaded.session_id == sid
    assert len(reloaded.trials) == 2


def test_dry_run_uploads_nothing(sync_manager):
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)

    result = sync_manager.sync_session_to_cloud(sid, dry_run=True)

    assert result["status"] == "success"  # legacy dry-run validation status
    assert result["preview"]["already_synced"] is False
    assert result["dry_run"] is True
    for mock in mocks.values():
        mock.assert_not_called()
    # Dry run does not write a synced marker.
    assert sync_manager.storage.load_session(sid).sync_state is None


# --------------------------------------------------------------------------- #
# Status reporting
# --------------------------------------------------------------------------- #


def test_get_sync_status_counts_by_state(sync_manager):
    storage = sync_manager.storage
    synced = _make_completed_session(storage)
    _make_completed_session(storage)  # unsynced
    failed = _make_completed_session(storage)
    storage.update_sync_state(synced, {"status": "synced", "payload_hash": "h"})
    storage.update_sync_state(failed, {"status": "failed"})

    status = sync_manager.get_sync_status()

    assert status["completed_sessions"] == 3
    assert status["synced"] == 1
    assert status["unsynced"] == 1
    assert status["failed"] == 1
    assert status["sync_eligible"] == 2  # unsynced + failed still pending


# --------------------------------------------------------------------------- #
# Privacy: free-text trial metadata must not ride to the backend
# --------------------------------------------------------------------------- #


def test_freetext_metadata_not_in_converted_payload(sync_manager):
    sid = sync_manager.storage.create_session(
        "fn", optimization_config={"search_space": {"model": ["a"]}}
    )
    sentinel = "SENTINEL_secret_prompt_text_should_never_sync"
    sync_manager.storage.add_trial_result(
        sid,
        config={"model": "a"},
        score=0.5,
        metadata={"raw_prompt": sentinel, "tokens": 42},
    )
    sync_manager.storage.finalize_session(sid, "completed")

    session = sync_manager.storage.load_session(sid)
    converted = sync_manager.convert_session_to_traigent_format(session)
    blob = json.dumps(converted, default=str)

    assert sentinel not in blob, "free-text metadata leaked into sync payload"
    # ...but numeric metadata is still forwarded as a measure.
    measures = converted["configuration_runs"][0]["measures"]
    assert measures.get("tokens") == 42


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
