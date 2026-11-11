"""#3013: a ``~`` in TRAIGENT_RESULTS_FOLDER must be expanded by every reader.

``.env.example`` ships ``TRAIGENT_RESULTS_FOLDER=~/.traigent``; dotenv, Compose and
systemd pass the ``~`` through literally. Offline: tmp HOME and cwd, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.cloud.billing import UsageTracker
from traigent.core.best_config_runtime import function_ref_for
from traigent.core.config_state_manager import ConfigStateManager


@pytest.fixture
def tilde_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", "~/.traigent")
    monkeypatch.chdir(work)
    return home, work


def test_usage_tracker_expands_tilde(tilde_env):
    home, work = tilde_env
    tracker = UsageTracker()
    assert tracker.storage_path == home / ".traigent" / "usage.json"
    assert not (work / "~").exists()


def _fn(x):
    return x


def test_best_config_lookup_searches_expanded_results_folder(tilde_env):
    home, work = tilde_env
    artifact = (
        home
        / ".traigent"
        / "optimization_logs"
        / "experiments"
        / _fn.__name__
        / "runs"
        / "run1"
        / "artifacts"
        / "best_config.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps({"function_ref": function_ref_for(_fn), "config": {}}),
        encoding="utf-8",
    )
    manager = ConfigStateManager(
        func=_fn,
        default_config={},
        local_storage_path=None,
        configuration_space={},
        auto_load_best=False,
        load_from=None,
        setup_wrapper_callback=lambda: None,
    )
    found = manager._find_latest_config_path()
    assert found is not None
    assert Path(found) == artifact
