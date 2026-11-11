"""#3272: persisting N trials must not reload or re-serialize the whole session."""

import json
from unittest.mock import patch

import pytest

from traigent.storage import local_storage
from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture
def storage(tmp_path):
    return LocalStorageManager(str(tmp_path))


def _add(storage, sid, i):
    storage.add_trial_result(
        sid, {"a": i}, i / 10, metadata={f"m{k}": k * 0.1 for k in range(3)}
    )


def _raw(storage, sid):
    return json.loads((storage.storage_path / "sessions" / f"{sid}.json").read_text())


def test_add_trial_result_does_not_reload_session_per_trial(storage):
    sid = storage.create_session("fn", optimization_config={"space": list(range(50))})
    _add(storage, sid, 0)  # the first append loads once
    with patch.object(
        LocalStorageManager, "load_session", side_effect=AssertionError("reload")
    ):
        for i in range(1, 25):
            _add(storage, sid, i)
    assert len(_raw(storage, sid)["trials"]) == 25


def test_serialization_work_per_trial_is_constant(storage):
    """Counting json.dumps payload sizes: each trial serializes only itself."""
    sid = storage.create_session("fn", optimization_config={"space": list(range(500))})
    _add(storage, sid, 0)
    real_dumps = json.dumps
    sizes: list[int] = []

    def counting_dumps(obj, *a, **kw):
        out = real_dumps(obj, *a, **kw)
        sizes.append(len(out))
        return out

    with (
        patch.object(local_storage.json, "dumps", counting_dumps),
        patch.object(
            local_storage.json, "dump", side_effect=AssertionError("full dump")
        ),
    ):
        for i in range(1, 41):
            _add(storage, sid, i)
    # 2 small payloads per trial (the trial, the per-trial fields); never the
    # 500-value optimization_config nor the growing trial list.
    assert len(sizes) == 80
    assert max(sizes) < 600


def test_all_trials_load_in_order_with_best(storage):
    sid = storage.create_session("fn")
    for i in range(30):
        _add(storage, sid, i)
    loaded = LocalStorageManager(str(storage.storage_path)).load_session(sid)
    assert [t.trial_id for t in loaded.trials] == list(range(1, 31))
    assert [t.config["a"] for t in loaded.trials] == list(range(30))
    assert loaded.completed_trials == loaded.total_trials == 30
    assert loaded.best_score == pytest.approx(2.9)
    assert loaded.best_config == {"a": 29}
    # semantically identical to what a full save would write
    raw = _raw(storage, sid)
    storage._save_session(loaded)
    assert _raw(storage, sid) == raw


def test_old_format_indented_session_loads_and_accepts_trials(storage):
    sid = "20200101_000000_000000_fn_deadbeef"
    old = {
        "session_id": sid,
        "function_name": "fn",
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
        "status": "running",
        "total_trials": 1,
        "completed_trials": 1,
        "best_score": 0.5,
        "best_config": {"a": 0},
        "trials": [{"trial_id": 1, "config": {"a": 0}, "score": 0.5, "timestamp": "t"}],
    }
    (storage.storage_path / "sessions" / f"{sid}.json").write_text(
        json.dumps(old, indent=2)
    )
    assert len(storage.load_session(sid).trials) == 1
    storage.add_trial_result(sid, {"a": 1}, 0.9)
    loaded = storage.load_session(sid)
    assert [t.trial_id for t in loaded.trials] == [1, 2]
    assert loaded.best_score == 0.9
    assert _raw(storage, sid)["total_trials"] == 2


def test_other_writes_between_appends_are_not_lost(storage):
    sid = storage.create_session("fn")
    _add(storage, sid, 0)
    storage.update_session_status(sid, "running")
    _add(storage, sid, 1)
    storage.finalize_session(sid, "completed")
    _add(storage, sid, 2)
    loaded = storage.load_session(sid)
    assert loaded.status == "completed"
    assert [t.trial_id for t in loaded.trials] == [1, 2, 3]


def test_external_change_invalidates_cache(storage):
    sid = storage.create_session("fn")
    _add(storage, sid, 0)
    other = LocalStorageManager(str(storage.storage_path))
    _add(other, sid, 1)
    other.update_session_status(sid, "running")
    _add(storage, sid, 2)  # must see other's trial and status, not stale cache
    loaded = storage.load_session(sid)
    assert [t.trial_id for t in loaded.trials] == [1, 2, 3]
    assert loaded.status == "running"


def test_failed_write_leaves_cache_and_file_consistent(storage):
    sid = storage.create_session("fn")
    _add(storage, sid, 0)
    from traigent.utils.exceptions import TraigentStorageError

    with patch.object(local_storage, "safe_open", side_effect=OSError("disk full")):
        with pytest.raises(TraigentStorageError):
            _add(storage, sid, 1)
    _add(storage, sid, 1)
    assert [t.trial_id for t in storage.load_session(sid).trials] == [1, 2]
    assert not list((storage.storage_path / "sessions").glob("*.tmp.*"))


def test_unserializable_trial_raises_and_does_not_corrupt(storage):
    from traigent.utils.exceptions import TraigentStorageError

    sid = storage.create_session("fn")
    _add(storage, sid, 0)
    with pytest.raises(TraigentStorageError):
        storage.add_trial_result(sid, {"a": object()}, 0.1)
    assert len(storage.load_session(sid).trials) == 1
    _add(storage, sid, 1)
    assert len(storage.load_session(sid).trials) == 2


def test_caller_mutating_config_after_append_does_not_change_stored_best(storage):
    sid = storage.create_session("fn")
    config = {"nested": {"value": 1}}
    storage.add_trial_result(sid, config, 1.0)
    config["nested"]["value"] = 99
    storage.add_trial_result(sid, {"nested": {"value": 2}}, 0.1)
    raw = _raw(storage, sid)
    assert raw["best_config"] == {"nested": {"value": 1}}
    assert raw["trials"][0]["config"] == {"nested": {"value": 1}}


def test_foreign_write_just_after_ours_is_not_overwritten(storage):
    """A writer that lands between our rename and our next append is seen."""
    sid = storage.create_session("fn")
    _add(storage, sid, 0)
    other = LocalStorageManager(str(storage.storage_path))
    _add(other, sid, 1)  # lands right after our last write
    _add(storage, sid, 2)
    assert [t.trial_id for t in storage.load_session(sid).trials] == [1, 2, 3]
