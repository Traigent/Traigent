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

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


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


# --------------------------------------------------------------------------- #
# Upgrade idempotency: sync state recorded by a release that ignored offline
# session.metadata (dataset_size / evaluation_set / display name).
# --------------------------------------------------------------------------- #


def _make_offline_session(storage: LocalStorageManager) -> str:
    """An offline-style session: dataset_size/evaluation_set live in metadata."""
    session_id = storage.create_session(
        "answer_question",
        optimization_config={"search_space": {"model": ["a", "b"]}},
        metadata={
            "dataset_size": 18,
            "evaluation_set": "tuning",
            "function_display_name": "run",
            "portal_name": "opt:accuracy · model (abc12345)",
            "offline": True,
        },
    )
    storage.add_trial_result(session_id, config={"model": "a"}, score=0.8)
    storage.add_trial_result(session_id, config={"model": "b"}, score=0.9)
    storage.finalize_session(session_id, "completed")
    return session_id


def _pre_upgrade_hash(sync_manager: SyncManager, session_id: str) -> str:
    """Fingerprint as the pre-PR conversion computed it (origin/develop).

    That conversion read dataset_size / evaluation_set from optimization_config
    only, sent function_name = session.function_name, and added no
    function_display_name / agent_key.
    """
    session = sync_manager.storage.load_session(session_id)
    data = sync_manager.convert_session_to_traigent_format(session)
    create = data["session_create"]
    create["function_name"] = session.function_name
    create["dataset_metadata"]["size"] = 1
    create["metadata"]["evaluation_set"] = "default"
    create["metadata"].pop("function_display_name", None)
    create.pop("agent_key", None)
    return SyncManager._compute_payload_hash(data)


def test_pre_upgrade_hash_differs_from_current_hash(sync_manager):
    sid = _make_offline_session(sync_manager.storage)
    session = sync_manager.storage.load_session(sid)
    current = SyncManager._compute_payload_hash(
        sync_manager.convert_session_to_traigent_format(session)
    )
    assert _pre_upgrade_hash(sync_manager, sid) != current


def test_pre_upgrade_synced_session_is_skipped_not_reuploaded(sync_manager):
    sid = _make_offline_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    sync_manager.storage.update_sync_state(
        sid,
        {
            "status": "synced",
            "payload_hash": _pre_upgrade_hash(sync_manager, sid),
            "cloud_experiment_id": "exp-old",
            "cloud_url": "https://example.test/exp-old",
        },
    )

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "already_synced"
    assert result["cloud_experiment_id"] == "exp-old"
    mocks["_sync_create_session"].assert_not_called()
    mocks["_sync_session_results"].assert_not_called()


def test_pre_upgrade_partial_session_resumes_existing_cloud_session(sync_manager):
    sid = _make_offline_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    sync_manager.storage.update_sync_state(
        sid,
        {
            "status": "partial",
            "payload_hash": _pre_upgrade_hash(sync_manager, sid),
            "cloud_session_id": "sess-old",
            "cloud_experiment_id": "exp-old",
        },
    )

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "success"
    mocks["_sync_create_session"].assert_not_called()
    assert result["cloud_session_id"] == "sess-old"
    assert result["cloud_experiment_id"] == "exp-old"


def test_pre_upgrade_synced_session_with_changed_trials_is_resynced(sync_manager):
    sid = _make_offline_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    sync_manager.storage.update_sync_state(
        sid,
        {
            "status": "synced",
            "payload_hash": _pre_upgrade_hash(sync_manager, sid),
            "cloud_experiment_id": "exp-old",
        },
    )
    sync_manager.storage.add_trial_result(sid, config={"model": "c"}, score=0.95)
    sync_manager.storage.finalize_session(sid, "completed")

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "success"
    mocks["_sync_create_session"].assert_called_once()


# --------------------------------------------------------------------------- #
# Legacy-shape parity with origin/develop + payload_hash_version
# --------------------------------------------------------------------------- #


def _session_with_config(manager: SyncManager, **extra) -> object:
    sid = manager.storage.create_session(
        "fn",
        optimization_config={"search_space": {"model": ["a"]}, **extra},
        metadata={
            "dataset_size": 18,
            "evaluation_set": "tuning",
            "function_display_name": "run",
            "portal_name": "opt:accuracy",
            "agent_key": "k",
        },
    )
    return manager.storage.load_session(sid)


# Expected values are what origin/develop's convert_session_to_traigent_format
# produced (verified once by importing that file): evaluation_set was
# ``opt_config.get("evaluation_set") or "default"`` (any truthy value kept
# verbatim, even non-str); dataset_size accepted only a positive non-bool int.
@pytest.mark.parametrize(
    ("value", "expected"),
    [(7, 7), (None, "default"), ("", "default"), ("tuning", "tuning")],
)
def test_legacy_shape_evaluation_set_matches_origin_develop(
    sync_manager, value, expected
):
    session = _session_with_config(sync_manager, evaluation_set=value)
    create = sync_manager.convert_session_to_traigent_format(
        session, legacy_shape=True
    )["session_create"]
    assert create["metadata"]["evaluation_set"] == expected
    # Offline metadata is ignored in legacy shape.
    assert create["function_name"] == "fn"
    assert "agent_key" not in create
    assert "function_display_name" not in create["metadata"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 1), (0, 1), (18, 18), ("18", 1), (True, 1)],
)
def test_legacy_shape_dataset_size_matches_origin_develop(
    sync_manager, value, expected
):
    session = _session_with_config(sync_manager, dataset_size=value)
    create = sync_manager.convert_session_to_traigent_format(
        session, legacy_shape=True
    )["session_create"]
    assert create["dataset_metadata"]["size"] == expected


def test_current_shape_non_string_evaluation_set_still_defaults(sync_manager):
    session = _session_with_config(sync_manager, evaluation_set=7)
    create = sync_manager.convert_session_to_traigent_format(session)["session_create"]
    assert create["metadata"]["evaluation_set"] == "default"


def test_fresh_sync_writes_payload_hash_version_3(sync_manager):
    sid = _make_offline_session(sync_manager.storage)
    _stub_backend_success(sync_manager)

    sync_manager.sync_session_to_cloud(sid)

    state = sync_manager.storage.load_session(sid).sync_state
    assert state["payload_hash_version"] == 3


def test_versioned_state_with_only_legacy_hash_is_not_treated_as_synced(
    sync_manager,
):
    sid = _make_offline_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    session = sync_manager.storage.load_session(sid)
    legacy = sync_manager._legacy_payload_hash(session)
    assert legacy != SyncManager._compute_payload_hash(
        sync_manager.convert_session_to_traigent_format(session)
    )
    sync_manager.storage.update_sync_state(
        sid,
        {
            "status": "synced",
            "payload_hash": legacy,
            "payload_hash_version": 2,
            "cloud_experiment_id": "exp-new",
        },
    )

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] != "already_synced"
    mocks["_sync_create_session"].assert_called_once()


def test_versioned_partial_state_with_only_legacy_hash_does_not_resume(sync_manager):
    sid = _make_offline_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    session = sync_manager.storage.load_session(sid)
    sync_manager.storage.update_sync_state(
        sid,
        {
            "status": "partial",
            "payload_hash": sync_manager._legacy_payload_hash(session),
            "payload_hash_version": 2,
            "cloud_session_id": "sess-new",
        },
    )

    result = sync_manager.sync_session_to_cloud(sid)

    mocks["_sync_create_session"].assert_called_once()
    assert result["cloud_session_id"] != "sess-new"


# --------------------------------------------------------------------------- #
# Per-example measures must not invalidate fingerprints from older releases
# --------------------------------------------------------------------------- #


def _make_measures_session(
    storage: LocalStorageManager, first_accuracy: float = 1.0
) -> str:
    """Offline session whose trials carry per-example measures."""
    session_id = storage.create_session(
        "answer_question",
        optimization_config={"search_space": {"model": ["a", "b"]}},
        metadata={"dataset_size": 18, "evaluation_set": "tuning", "offline": True},
    )
    for model, score in (("a", 0.8), ("b", 0.9)):
        storage.add_trial_result(
            session_id,
            config={"model": model},
            score=score,
            metadata={
                "measures": [
                    {"example_id": "ex-1", "metrics": {"accuracy": first_accuracy}},
                    {"example_id": "ex-2", "metrics": {"accuracy": 0.5}},
                ]
            },
        )
    storage.finalize_session(session_id, "completed")
    return session_id


def _v2_hash(sync_manager: SyncManager, session) -> str:
    return SyncManager._compute_payload_hash(
        sync_manager.convert_session_to_traigent_format(
            session, include_example_measures=False
        )
    )


def _current_hash(sync_manager: SyncManager, session) -> str:
    return SyncManager._compute_payload_hash(
        sync_manager.convert_session_to_traigent_format(session)
    )


def test_session_with_measures_hashes_differently_per_shape(sync_manager):
    sid = _make_measures_session(sync_manager.storage)
    session = sync_manager.storage.load_session(sid)
    hashes = {
        _current_hash(sync_manager, session),
        _v2_hash(sync_manager, session),
        sync_manager._legacy_payload_hash(session),
    }
    assert len(hashes) == 3


def test_legacy_shape_has_no_example_measures(sync_manager):
    sid = _make_measures_session(sync_manager.storage)
    session = sync_manager.storage.load_session(sid)
    current = sync_manager.convert_session_to_traigent_format(session)
    assert all("example_measures" in r for r in current["configuration_runs"])
    legacy = sync_manager.convert_session_to_traigent_format(session, legacy_shape=True)
    assert all("example_measures" not in r for r in legacy["configuration_runs"])


# Fingerprints of _make_measures_session() as recorded by the releases that
# actually wrote them, computed once by loading `git show <ref>:traigent/cloud/
# sync_manager.py` and hashing convert_session_to_traigent_format(session).
# Pinned literals (not recomputed) so these tests do not share code with the
# fix under test.
# origin/develop (pre-#2489): no payload_hash_version in sync_state.
HASH_WRITTEN_BY_ORIGIN_DEVELOP = "22e1c4c54e621ccf"
# PR #2489 head 29f93034: payload_hash_version 2, no example_measures.
HASH_WRITTEN_BY_PR_2489 = "6fe4f43fb695fe5b"


def _prior_state(version, hash_value, status="synced"):
    state = {
        "status": status,
        "payload_hash": hash_value,
        "cloud_experiment_id": "exp-old",
        "cloud_session_id": "sess-old",
    }
    if version is not None:
        state["payload_hash_version"] = version
    return state


@pytest.mark.parametrize("version", [None, 2])
def test_older_version_state_with_measures_session_is_skipped(sync_manager, version):
    sid = _make_measures_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    old_hash = (
        HASH_WRITTEN_BY_ORIGIN_DEVELOP if version is None else HASH_WRITTEN_BY_PR_2489
    )
    sync_manager.storage.update_sync_state(sid, _prior_state(version, old_hash))

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "already_synced"
    mocks["_sync_create_session"].assert_not_called()


@pytest.mark.parametrize("version", [None, 2])
def test_older_version_partial_state_with_measures_session_resumes(
    sync_manager, version
):
    sid = _make_measures_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    old_hash = (
        HASH_WRITTEN_BY_ORIGIN_DEVELOP if version is None else HASH_WRITTEN_BY_PR_2489
    )
    sync_manager.storage.update_sync_state(
        sid, _prior_state(version, old_hash, status="partial")
    )

    result = sync_manager.sync_session_to_cloud(sid)

    mocks["_sync_create_session"].assert_not_called()
    assert result["cloud_session_id"] == "sess-old"


def test_version_3_state_matches_only_current_hash(sync_manager):
    sid = _make_measures_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    session = sync_manager.storage.load_session(sid)

    sync_manager.storage.update_sync_state(
        sid, _prior_state(3, _current_hash(sync_manager, session))
    )
    assert sync_manager.sync_session_to_cloud(sid)["status"] == "already_synced"
    mocks["_sync_create_session"].assert_not_called()

    for stale in (
        _v2_hash(sync_manager, session),
        sync_manager._legacy_payload_hash(session),
    ):
        sync_manager.storage.update_sync_state(sid, _prior_state(3, stale))
        assert sync_manager.sync_session_to_cloud(sid)["status"] != "already_synced"


@pytest.mark.parametrize("version", [None, 2, 3])
def test_changed_trials_resync_under_every_version(sync_manager, version):
    sid = _make_measures_session(sync_manager.storage)
    mocks = _stub_backend_success(sync_manager)
    session = sync_manager.storage.load_session(sid)
    old_hash = {
        None: HASH_WRITTEN_BY_ORIGIN_DEVELOP,
        2: HASH_WRITTEN_BY_PR_2489,
        3: _current_hash(sync_manager, session),
    }[version]
    sync_manager.storage.update_sync_state(sid, _prior_state(version, old_hash))
    sync_manager.storage.add_trial_result(sid, config={"model": "c"}, score=0.95)
    sync_manager.storage.finalize_session(sid, "completed")

    result = sync_manager.sync_session_to_cloud(sid)

    assert result["status"] == "success"
    mocks["_sync_create_session"].assert_called_once()


def test_pinned_hashes_match_current_shape_reconstruction(sync_manager):
    """The literals are what the current code reproduces (and nothing else)."""
    sid = _make_measures_session(sync_manager.storage)
    session = sync_manager.storage.load_session(sid)
    assert sync_manager._legacy_payload_hash(session) == HASH_WRITTEN_BY_ORIGIN_DEVELOP
    assert _v2_hash(sync_manager, session) == HASH_WRITTEN_BY_PR_2489


# --------------------------------------------------------------------------- #
# _content_free_example_measures: indices, filtering, caps, overflow
# --------------------------------------------------------------------------- #


def _ex(example_id, **metrics):
    return {"example_id": example_id, "metrics": metrics}


def _measures(rows):
    return SyncManager._content_free_example_measures({"measures": rows})


def test_huge_int_metric_is_dropped_and_sync_succeeds(sync_manager):
    sid = sync_manager.storage.create_session(
        "answer_question",
        optimization_config={"search_space": {"model": ["a"]}},
    )
    sync_manager.storage.add_trial_result(
        sid,
        config={"model": "a"},
        score=0.8,
        metadata={"measures": [_ex("example_0", accuracy=1.0, big=10**400)]},
    )
    sync_manager.storage.finalize_session(sid, "completed")
    _stub_backend_success(sync_manager)

    assert _measures([_ex("example_0", big=10**400, ok=1)]) == [
        {"example_id": "example_0", "metrics": {"ok": 1}}
    ]
    assert sync_manager.sync_session_to_cloud(sid)["status"] == "success"


def test_dataset_index_survives_per_trial_compaction():
    trial_a = _measures([_ex("example_0", a=1.0), _ex("example_2", a=0.0)])
    trial_b = _measures([_ex("example_0", a=1.0), _ex("example_1", a=1.0)])
    ids = {m["example_id"] for m in trial_a + trial_b}
    assert ids == {"example_0", "example_1", "example_2"}
    assert [m["example_id"] for m in trial_a] == ["example_0", "example_2"]


def test_hashed_sdk_id_yields_index_only():
    out = _measures([_ex("ex_deadbeef01_7", a=1.0)])
    assert out == [{"example_id": "example_7", "metrics": {"a": 1.0}}]
    assert "deadbeef" not in json.dumps(out)


def test_customer_id_with_trailing_digits_falls_back_to_position():
    rows = [_ex("Alice_has_diabetes_7", a=1.0), _ex("example_x_9", a=1.0)]
    out = _measures(rows)
    assert [m["example_id"] for m in out] == ["example_0", "example_1"]
    assert "Alice" not in json.dumps(out)


def test_empty_metric_rows_are_omitted():
    out = _measures(
        [
            _ex("example_0", a=1.0),
            _ex("example_1", label="text", flag=True, nan=float("nan")),
            {"example_id": "example_2", "metrics": {}},
            _ex("example_3", a=0.5),
        ]
    )
    assert [m["example_id"] for m in out] == ["example_0", "example_3"]


def test_duplicate_index_within_trial_keeps_first():
    out = _measures(
        [_ex("example_4", a=1.0), _ex("ex_abc_4", a=0.0), _ex("example_5", a=2.0)]
    )
    assert out == [
        {"example_id": "example_4", "metrics": {"a": 1.0}},
        {"example_id": "example_5", "metrics": {"a": 2.0}},
    ]


@pytest.mark.parametrize("count,expected", [(1000, 1000), (1001, 1000)])
def test_example_count_cap_boundary(count, expected):
    out = _measures([_ex(f"example_{i}", a=1.0) for i in range(count)])
    assert len(out) == expected
    assert out[-1]["example_id"] == "example_999"


@pytest.mark.parametrize("count,expected", [(50, 50), (51, 50)])
def test_metric_count_cap_boundary(count, expected):
    metrics = {f"m{i}": float(i) for i in range(count)}
    out = _measures([{"example_id": "example_0", "metrics": metrics}])
    assert len(out[0]["metrics"]) == expected


@pytest.mark.parametrize("length,kept", [(100, True), (101, False)])
def test_metric_key_length_boundary(length, kept):
    key = "k" * length
    out = _measures([_ex("example_0", **{key: 1.0, "other": 2.0})])
    assert (key in out[0]["metrics"]) is kept
    assert "other" in out[0]["metrics"]


def test_pre_v3_state_ignores_per_example_score_only_change(sync_manager):
    """Known limitation: pre-v3 fingerprints never covered per-example scores."""
    sid = _make_measures_session(sync_manager.storage, first_accuracy=0.0)
    mocks = _stub_backend_success(sync_manager)
    session = sync_manager.storage.load_session(sid)
    sync_manager.storage.update_sync_state(
        sid, _prior_state(2, HASH_WRITTEN_BY_PR_2489)
    )
    # Persisted state is compared against a recomputed v2 shape, which omits
    # per-example measures entirely.
    assert HASH_WRITTEN_BY_PR_2489 in sync_manager._acceptable_payload_hashes(
        session, 2, "unused-current"
    )
    assert _current_hash(sync_manager, session) != _v2_hash(sync_manager, session)
    assert sync_manager.sync_session_to_cloud(sid)["status"] == "already_synced"
    mocks["_sync_create_session"].assert_not_called()
