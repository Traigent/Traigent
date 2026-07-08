"""Tests for idempotent cloud sync of local optimization runs (Sync v2).

Covers the W&B-style guarantees added to ``SyncManager`` + the local
``sync_state`` marker: re-syncing an unchanged run is a no-op (no duplicate
cloud experiments), sync state is persisted for status reporting, a changed run
is re-synced, and free-text trial metadata never rides to the backend.
"""

from __future__ import annotations

import itertools
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
    """Patch the content-free session uploads so a sync 'succeeds' with no HTTP."""
    mocks = {
        # Offline sync now imports through the content-free typed-session
        # endpoints: create session -> per-trial results -> finalize. The
        # session binds no benchmark, so an empty server-side dataset never
        # blocks the import (empty-dataset sync fix).
        "_sync_create_session": Mock(
            return_value={
                "success": True,
                "session_id": "sess1",
                "experiment_id": "exp1",
                "experiment_run_id": "run1",
                "project_id": None,
                "tenant_id": None,
            }
        ),
        "_sync_finalize_session": Mock(
            return_value={"success": True, "classification": "completed"}
        ),
    }

    # ``_sync_session_results`` carries the ``already_synced_keys`` / ``on_synced``
    # kwargs and returns a ``skipped`` count (resume idempotency). Mirror the
    # real contract: accept the kwargs and report 0 skipped.
    def _runs(_session_id, configuration_runs, **_kwargs):
        return {
            "success": True,
            "synced": len(configuration_runs),
            "skipped": 0,
            "errors": [],
        }

    mocks["_sync_session_results"] = Mock(side_effect=_runs)
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
    assert state["cloud_session_id"] == "sess1"
    assert state["attempts"] == 1
    mocks["_sync_create_session"].assert_called_once()


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
    # No backend resource was created the second time → no duplicate session.
    mocks["_sync_create_session"].assert_not_called()
    mocks["_sync_session_results"].assert_not_called()


def test_force_reuploads_even_when_synced(sync_manager):
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    sync_manager.sync_session_to_cloud(sid)
    for mock in mocks.values():
        mock.reset_mock()

    forced = sync_manager.sync_session_to_cloud(sid, force=True)

    assert forced["status"] == "success"
    mocks["_sync_create_session"].assert_called_once()


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
    mocks["_sync_create_session"].assert_called_once()


def test_partial_failure_then_resume_reuses_experiment(sync_manager):
    """BLOCKER regression: a retry after a partial failure must reuse the
    cloud session, never create a duplicate one."""
    sid = _make_completed_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    # First attempt: session is created but the result step fails → partial.
    mocks["_sync_session_results"] = Mock(
        return_value={
            "success": False,
            "synced": 0,
            "skipped": 0,
            "errors": ["boom"],
        }
    )
    sync_manager._sync_session_results = mocks["_sync_session_results"]

    first = sync_manager.sync_session_to_cloud(sid)
    assert first["status"] == "partial"
    state = sync_manager.storage.load_session(sid).sync_state
    assert state["status"] == "partial"
    assert state["cloud_experiment_id"] == "exp1"  # session/experiment was created
    assert state["cloud_session_id"] == "sess1"

    # Second attempt (same content): results now succeed.
    def _runs(_session_id, configuration_runs, **_kwargs):
        return {
            "success": True,
            "synced": len(configuration_runs),
            "skipped": 0,
            "errors": [],
        }

    mocks["_sync_session_results"] = Mock(side_effect=_runs)
    sync_manager._sync_session_results = mocks["_sync_session_results"]
    mocks["_sync_create_session"].reset_mock()

    second = sync_manager.sync_session_to_cloud(sid)

    assert second["status"] == "success"
    # The session was REUSED, not re-created → no duplicate.
    mocks["_sync_create_session"].assert_not_called()
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


# --------------------------------------------------------------------------- #
# Regression: empty server-side dataset no longer blocks offline sync
# (content-free typed-session import; issue: HTTP 400 EMPTY_DATASET)
# --------------------------------------------------------------------------- #


def _backend_response(status_code=201, payload=None, text="Created"):
    response = Mock(status_code=status_code, text=text)
    response.json.return_value = (
        payload if payload is not None else {"id": "created-id"}
    )
    return response


def test_empty_dataset_session_syncs_via_content_free_session_endpoints(sync_manager):
    """Regression: a completed run whose server-side dataset would have ZERO
    examples now imports cleanly.

    Before the fix, offline sync created a server-side dataset via
    ``POST /datasets`` with no example rows and bound an experiment_run to it,
    so the backend's ``dataset_run_guard`` rejected the run with HTTP 400
    EMPTY_DATASET. The fix reroutes sync through the content-free typed-session
    endpoints (``POST /sessions`` with tracking_mode=backend_guided and NO
    benchmark), which hit the backend's no-dataset pass-through. This test
    proves the sync (a) succeeds, (b) never creates/binds a benchmark, and
    (c) uses the session endpoints.
    """
    # Two completed trials; the local store retains NO raw example rows, so the
    # server-side dataset for this run would have zero examples.
    sid = _make_completed_session(sync_manager.storage)

    posts: list[str] = []
    slot_counter = itertools.count(1)

    def _route(url, *args, **kwargs):
        posts.append(url)
        if url.endswith("/sessions"):
            return _backend_response(
                201,
                payload={
                    "session_id": "sess-1",
                    "metadata": {
                        "experiment_id": "exp-1",
                        "experiment_run_id": "run-1",
                    },
                },
            )
        if url.endswith("/next-trial"):
            # Each trial gets a UNIQUE backend-minted slot id.
            return _backend_response(
                200,
                payload={"suggestion": {"trial_id": f"bt-{next(slot_counter)}"}},
            )
        if url.endswith("/finalize"):
            return _backend_response(200, payload={"status": "finalized"})
        # per-trial results
        return _backend_response(201, payload={"id": "result-1"})

    sync_manager._session = Mock()
    sync_manager._session.post = Mock(side_effect=_route)

    result = sync_manager.sync_session_to_cloud(sid)

    # (a) The empty-dataset run syncs successfully (was HTTP 400 EMPTY_DATASET).
    assert result["status"] == "success", result.get("errors")

    base = sync_manager.base_url
    results_url = f"{base}/sessions/sess-1/results"

    # (b) NO benchmark is ever created or bound — no dataset POST, no
    # experiment-run POST (the two calls that used to trigger EMPTY_DATASET).
    assert not any("/datasets" in url for url in posts), (
        f"offline sync must not create a benchmark/dataset; posts={posts}"
    )
    assert not any(url.endswith("/runs") for url in posts), (
        f"offline sync must not bind an experiment_run to a dataset; posts={posts}"
    )
    assert not any("/experiment-runs/" in url for url in posts), posts
    assert not any(url.endswith("/agents") for url in posts), posts

    # (c) It DOES use the content-free session endpoints: create, one result
    # per trial (2 trials), and finalize.
    assert posts.count(f"{base}/sessions") == 1
    assert posts.count(results_url) == 2  # one per completed trial
    assert posts.count(f"{base}/sessions/sess-1/finalize") == 1

    # The result payloads are content-free (config + numeric metrics only) and
    # carry the required trial config bound to the backend-minted slot.
    result_calls = [
        call
        for call in sync_manager._session.post.call_args_list
        if call.args[0] == results_url
    ]
    for call in result_calls:
        body = call.kwargs["json"]
        assert body["status"] == "COMPLETED"
        assert "config" in body and isinstance(body["config"], dict)
        assert "metrics" in body
        # trial_id MUST be a string: local trial ids are ints, but the session
        # /results validator rejects a non-string trial_id with HTTP 400
        # ("trial_id must be a string"), which left offline sync stuck at
        # `partial`/exit 1 even though EMPTY_DATASET was already gone. Caught by
        # live api-dev E2E; mocked transport could not (it never type-checks).
        assert isinstance(body["trial_id"], str), body["trial_id"]

    # The create payload is the typed backend_guided contract with no benchmark.
    create_call = next(
        call
        for call in sync_manager._session.post.call_args_list
        if call.args[0] == f"{base}/sessions"
    )
    create_body = create_call.kwargs["json"]
    assert create_body["optimization_strategy"]["tracking_mode"] == "backend_guided"
    assert "benchmark" not in create_body
    assert "benchmark_id" not in create_body
    assert create_body["dataset_metadata"]["privacy_mode"] is True


def test_dataset_metadata_size_is_positive_when_local_size_unknown(sync_manager):
    """The typed path validates dataset_metadata.size as a positive int when the
    key is present, so an unknown/zero local size must be coerced to 1 (like the
    live SDK builder) — otherwise EVERY offline sync would 400 at create."""
    sid = _make_completed_session(sync_manager.storage)  # no dataset_size recorded
    session = sync_manager.storage.load_session(sid)

    converted = sync_manager.convert_session_to_traigent_format(session)

    size = converted["session_create"]["dataset_metadata"]["size"]
    assert isinstance(size, int) and not isinstance(size, bool)
    assert size >= 1


def test_empty_config_trial_degrades_to_partial_with_clear_error(sync_manager):
    """The backend rejects empty trial configs; the SDK must surface a clear
    per-trial error (and a partial sync), not the raw backend message."""
    sid = sync_manager.storage.create_session(
        "fn_empty_cfg", optimization_config={"search_space": {"model": ["a"]}}
    )
    sync_manager.storage.add_trial_result(sid, config={}, score=0.5)
    sync_manager.storage.add_trial_result(sid, config={"model": "a"}, score=0.9)
    sync_manager.storage.finalize_session(sid, "completed")

    slot_counter = itertools.count(1)

    def _route(url, *args, **kwargs):
        if url.endswith("/sessions"):
            return _backend_response(
                201,
                payload={
                    "session_id": "sess-ec",
                    "metadata": {
                        "experiment_id": "exp-ec",
                        "experiment_run_id": "run-ec",
                    },
                },
            )
        if url.endswith("/next-trial"):
            return _backend_response(
                200,
                payload={"suggestion": {"trial_id": f"bt-{next(slot_counter)}"}},
            )
        return _backend_response(201, payload={"id": "ok"})

    sync_manager._session = Mock()
    sync_manager._session.post = Mock(side_effect=_route)

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "partial"
    assert any("empty config" in err for err in result["errors"]), result["errors"]
    # The empty-config trial minted NO slot and was never POSTed; the valid trial
    # was. Exactly one /next-trial (for the valid trial) and one /results.
    next_trial_posts = [
        call
        for call in sync_manager._session.post.call_args_list
        if call.args[0].endswith("/next-trial")
    ]
    assert len(next_trial_posts) == 1
    result_posts = [
        call.kwargs["json"]
        for call in sync_manager._session.post.call_args_list
        if call.args[0].endswith("/results")
    ]
    assert len(result_posts) == 1
    assert result_posts[0]["config"] == {"model": "a"}


def test_failed_trial_submits_real_failed_status(sync_manager):
    """A failed local trial is submitted with its REAL status, not masked to
    COMPLETED (the session results endpoint accepts failed trials)."""
    sid = sync_manager.storage.create_session(
        "fn_failed_trial", optimization_config={"search_space": {"model": ["a", "b"]}}
    )
    sync_manager.storage.add_trial_result(sid, config={"model": "a"}, score=0.9)
    sync_manager.storage.add_trial_result(
        sid, config={"model": "b"}, score=0.0, error="Timeout error"
    )
    sync_manager.storage.finalize_session(sid, "completed")

    slot_counter = itertools.count(1)

    def _route(url, *args, **kwargs):
        if url.endswith("/sessions"):
            return _backend_response(
                201,
                payload={
                    "session_id": "sess-ft",
                    "metadata": {
                        "experiment_id": "exp-ft",
                        "experiment_run_id": "run-ft",
                    },
                },
            )
        if url.endswith("/next-trial"):
            return _backend_response(
                200,
                payload={"suggestion": {"trial_id": f"bt-{next(slot_counter)}"}},
            )
        return _backend_response(201, payload={"id": "ok"})

    sync_manager._session = Mock()
    sync_manager._session.post = Mock(side_effect=_route)

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "success", result.get("errors")
    result_posts = [
        call.kwargs["json"]
        for call in sync_manager._session.post.call_args_list
        if call.args[0].endswith("/results")
    ]
    statuses = {body["config"]["model"]: body["status"] for body in result_posts}
    assert statuses == {"a": "COMPLETED", "b": "FAILED"}


def test_objectives_are_minimal_not_union_of_measures(sync_manager):
    """Regression: the session must NOT declare the union of every per-trial
    measure as objectives.

    The backend requires each declared objective on EVERY completed trial (an
    Optuna objective must be a scalar it can ``tell``). Declaring the union of
    incidental run-level overlays (``run_trials_completed``, ``duration``,
    tokens, ...) made trials that omit one 400 with
    "Completed trial is missing numeric metric". With no objectives on the local
    config the session must fall back to the minimal, universally-present
    ``["score"]``. Caught by live api-dev E2E; mocked transport could not.
    """
    sid = sync_manager.storage.create_session(
        "answer",
        optimization_config={"search_space": {"model": ["a", "b"]}},
    )
    # Two trials with DIFFERENT incidental numeric measures — a union would make
    # both "duration" and "run_trials_completed" required objectives, and each
    # trial is missing the other's.
    sync_manager.storage.add_trial_result(
        sid, config={"model": "a"}, score=0.8, metadata={"duration": 1.2}
    )
    sync_manager.storage.add_trial_result(
        sid, config={"model": "b"}, score=0.9, metadata={"run_trials_completed": 2}
    )
    sync_manager.storage.finalize_session(sid, "completed")

    converted = sync_manager.convert_session_to_traigent_format(
        sync_manager.storage.load_session(sid)
    )

    # Minimal objective set — score only, never the union.
    assert converted["session_create"]["objectives"] == ["score"]
    # Backfill guarantees every declared objective (score) is numeric on every
    # completed trial, so no /results can 400 on a missing objective.
    for run in converted["configuration_runs"]:
        if run["status"] == "COMPLETED":
            assert isinstance(run["measures"]["score"], (int, float))


def test_declared_objective_backfilled_when_missing_on_a_trial(sync_manager):
    """A user-declared objective absent from one completed trial is backfilled
    (from score) so the whole sync can't 400 on that trial."""
    sid = sync_manager.storage.create_session(
        "answer",
        optimization_config={
            "search_space": {"model": ["a", "b"]},
            "objectives": ["accuracy"],
        },
    )
    # Trial 1 reports accuracy; trial 2 does NOT (only score).
    sync_manager.storage.add_trial_result(
        sid,
        config={"model": "a"},
        score=0.8,
        metadata={"all_metrics": {"accuracy": 1.0}},
    )
    sync_manager.storage.add_trial_result(sid, config={"model": "b"}, score=0.5)
    sync_manager.storage.finalize_session(sid, "completed")

    converted = sync_manager.convert_session_to_traigent_format(
        sync_manager.storage.load_session(sid)
    )

    assert converted["session_create"]["objectives"] == ["accuracy"]
    # Both completed trials carry a numeric "accuracy" (trial 2 backfilled).
    for run in converted["configuration_runs"]:
        if run["status"] == "COMPLETED":
            assert isinstance(run["measures"].get("accuracy"), (int, float))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
