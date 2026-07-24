"""Regression tests for user-config-only, spec-aware deduplication."""

from traigent.storage.local_storage import LocalStorageManager


def _seed_session(storage: LocalStorageManager, config: dict[str, object]) -> None:
    session_id = storage.create_session("fn", metadata={"evaluation_set": "ds"})
    storage.add_trial_result(session_id, config, score=0.5)
    storage.update_session_status(session_id, "completed")


def test_dedup_uses_only_user_keys_but_preserves_spec_identity(tmp_path):
    storage = LocalStorageManager(tmp_path)
    _seed_session(
        storage,
        {
            "model": "gpt-4",
            "temperature": 0.2,
            "_optuna_trial_id": 1,
            "_traigent_execution_id": "first",
            "__subset_indices__": [0, 1],
        },
    )

    # Different internal realization metadata is the same user configuration.
    assert storage.is_config_seen(
        "fn",
        "ds",
        {
            "model": "gpt-4",
            "temperature": 0.2,
            "_optuna_trial_id": 99,
            "_traigent_execution_id": "second",
            "__subset_indices__": [2, 3],
        },
        keys=["model", "temperature", "_optuna_trial_id"],
    )
    assert (
        storage.find_cached_result(
            storage.list_sessions()[0].session_id,
            {"model": "gpt-4", "temperature": 0.2, "_optuna_trial_id": 99},
        )
        is not None
    )

    # A changed user choice is genuinely new even when internal data matches.
    assert not storage.is_config_seen(
        "fn",
        "ds",
        {
            "model": "gpt-4",
            "temperature": 0.9,
            "_optuna_trial_id": 1,
        },
        keys=["model", "temperature"],
    )

    # A narrowed current spec cannot collapse a broader historical user config.
    assert not storage.is_config_seen("fn", "ds", {"model": "gpt-4"}, keys=["model"])
