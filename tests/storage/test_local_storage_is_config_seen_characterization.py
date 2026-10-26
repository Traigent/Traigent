"""Characterization tests for ``LocalStorageManager.is_config_seen``.

Pin which sessions, datasets and trials count as a hit, how the optional key
list narrows the comparison, and that internal failures degrade to ``False``.
"""

from __future__ import annotations

import logging

import pytest

from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture
def storage(tmp_path) -> LocalStorageManager:
    return LocalStorageManager(tmp_path)


def _session(storage, function="fn", metadata=None, trials=()):
    session_id = storage.create_session(function, metadata=metadata)
    for config, kwargs in trials:
        storage.add_trial_result(session_id, config, **kwargs)
    return session_id


def test_hit_requires_same_function_dataset_and_config(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"model": "a", "t": 1}, {"score": 0.5})],
    )
    assert storage.is_config_seen("fn", "ds", {"model": "a", "t": 1})
    assert not storage.is_config_seen("other", "ds", {"model": "a", "t": 1})
    assert not storage.is_config_seen("fn", "other-ds", {"model": "a", "t": 1})
    assert not storage.is_config_seen("fn", "ds", {"model": "b", "t": 1})


def test_dataset_name_falls_back_to_dataset_name_metadata(storage):
    _session(
        storage,
        metadata={"dataset_name": "legacy"},
        trials=[({"x": 1}, {"score": 0.1})],
    )
    assert storage.is_config_seen("fn", "legacy", {"x": 1})
    assert not storage.is_config_seen("fn", "ds", {"x": 1})


def test_evaluation_set_wins_over_dataset_name(storage):
    _session(
        storage,
        metadata={"evaluation_set": "new", "dataset_name": "old"},
        trials=[({"x": 1}, {"score": 0.1})],
    )
    assert storage.is_config_seen("fn", "new", {"x": 1})
    assert not storage.is_config_seen("fn", "old", {"x": 1})


def test_session_without_dataset_metadata_never_matches(storage):
    _session(storage, metadata=None, trials=[({"x": 1}, {"score": 0.1})])
    assert not storage.is_config_seen("fn", "ds", {"x": 1})


def test_session_without_trials_is_skipped(storage):
    _session(storage, metadata={"evaluation_set": "ds"})
    assert not storage.is_config_seen("fn", "ds", {"x": 1})


def test_errored_trials_do_not_count(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"x": 1}, {"score": None, "error": "boom"})],
    )
    assert not storage.is_config_seen("fn", "ds", {"x": 1})


def test_any_matching_successful_trial_in_any_session_counts(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"x": 1}, {"score": None, "error": "boom"})],
    )
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"x": 2}, {"score": 0.1}), ({"x": 1}, {"score": 0.2})],
    )
    assert storage.is_config_seen("fn", "ds", {"x": 1})
    assert storage.is_config_seen("fn", "ds", {"x": 2})
    assert not storage.is_config_seen("fn", "ds", {"x": 3})


def test_without_keys_internal_keys_are_ignored_by_the_hash(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"x": 1, "_internal": "a"}, {"score": 0.2})],
    )
    assert storage.is_config_seen("fn", "ds", {"x": 1, "_internal": "b"})
    assert storage.is_config_seen("fn", "ds", {"x": 1})


def test_keys_union_with_realised_user_keys_on_both_sides(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"model": "a", "temperature": 0.2}, {"score": 0.2})],
    )
    # Narrower current spec cannot collapse the broader historical config.
    assert not storage.is_config_seen("fn", "ds", {"model": "a"}, keys=["model"])
    # Same user identity matches even when keys name a subset.
    assert storage.is_config_seen(
        "fn", "ds", {"model": "a", "temperature": 0.2}, keys=["model"]
    )
    # A changed value is new.
    assert not storage.is_config_seen(
        "fn", "ds", {"model": "a", "temperature": 0.9}, keys=["model", "temperature"]
    )


def test_underscore_keys_in_keys_argument_are_ignored(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"model": "a"}, {"score": 0.2})],
    )
    assert storage.is_config_seen(
        "fn", "ds", {"model": "a"}, keys=["model", "_private", 3]
    )


def test_empty_keys_list_behaves_like_no_keys(storage):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"model": "a"}, {"score": 0.2})],
    )
    assert storage.is_config_seen("fn", "ds", {"model": "a"}, keys=[])


def test_missing_loaded_session_is_skipped(storage, monkeypatch):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"x": 1}, {"score": 0.2})],
    )
    monkeypatch.setattr(storage, "load_session", lambda session_id: None)
    assert not storage.is_config_seen("fn", "ds", {"x": 1})


def test_internal_errors_are_swallowed_logged_and_return_false(
    storage, monkeypatch, caplog
):
    _session(
        storage,
        metadata={"evaluation_set": "ds"},
        trials=[({"x": 1}, {"score": 0.2})],
    )

    def boom():
        raise RuntimeError("secret detail")

    monkeypatch.setattr(storage, "list_sessions", boom)
    with caplog.at_level(logging.WARNING):
        assert storage.is_config_seen("fn", "ds", {"x": 1}) is False
    messages = [r.getMessage() for r in caplog.records]
    assert "Error during config deduplication check (RuntimeError)" in messages
    assert not any("secret detail" in m for m in messages)
