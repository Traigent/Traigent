"""Regression tests for spec-aware config deduplication (#1972).

prefer_new dedup used to hash both the candidate and each historical trial
filtered to the CURRENT optimizer's config_space keys only. When the tuned
variable set narrows between runs, a historical trial that ran under a broader
spec (e.g. it also tuned `temperature`) collapsed onto the surviving keys and
falsely matched a narrower candidate — so the candidate was skipped though it
had never run under the new spec. is_config_seen must now be spec-aware:
compare over the full realized config on both sides.
"""

import shutil
import tempfile

import pytest

from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture()
def storage():
    path = tempfile.mkdtemp()
    yield LocalStorageManager(path)
    shutil.rmtree(path, ignore_errors=True)


def _seed_session(storage, config, dataset="ds"):
    session_id = storage.create_session(
        function_name="fn", metadata={"evaluation_set": dataset}
    )
    storage.add_trial_result(session_id=session_id, config=config, score=0.5)
    storage.update_session_status(session_id, "completed")
    return session_id


def test_historical_broader_spec_does_not_false_match(storage):
    """Run-1 tuned {model, temperature}; run-2 tunes only {model}.

    The run-1 trial must NOT be treated as covering the run-2 candidate, which
    never ran under the narrowed spec.
    """
    _seed_session(storage, {"model": "gpt-4", "temperature": 0.2})

    # Current run tunes only `model`.
    seen = storage.is_config_seen("fn", "ds", {"model": "gpt-4"}, keys=["model"])
    assert seen is False, "narrower candidate must not match a broader historical trial"


def test_exact_full_config_match_is_deduped(storage):
    """Identical realized config on the same keyset is still correctly skipped."""
    _seed_session(storage, {"model": "gpt-4", "temperature": 0.2})

    seen = storage.is_config_seen(
        "fn",
        "ds",
        {"model": "gpt-4", "temperature": 0.2},
        keys=["model", "temperature"],
    )
    assert seen is True


def test_current_broader_spec_does_not_false_match(storage):
    """Run-1 tuned only {model}; run-2 tunes {model, temperature}.

    The run-1 trial (no temperature dimension) must not match a run-2 candidate
    at a specific temperature.
    """
    _seed_session(storage, {"model": "gpt-4"})

    seen = storage.is_config_seen(
        "fn",
        "ds",
        {"model": "gpt-4", "temperature": 0.7},
        keys=["model", "temperature"],
    )
    assert seen is False


def test_same_spec_different_value_not_matched(storage):
    """Same keyset, different tuned value is genuinely new."""
    _seed_session(storage, {"model": "gpt-4", "temperature": 0.2})

    seen = storage.is_config_seen(
        "fn",
        "ds",
        {"model": "gpt-4", "temperature": 0.9},
        keys=["model", "temperature"],
    )
    assert seen is False
