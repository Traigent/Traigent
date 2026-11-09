"""#3195: ``save_to`` is validated before the first trial and a late save failure
never discards the finished run."""

from __future__ import annotations

import os
import warnings

import pytest

import traigent
from traigent.core.config_state_manager import ConfigStateManager
from traigent.utils.exceptions import ConfigurationError

_DATASET = [
    {"input": {"x": 1}, "output": 1},
    {"input": {"x": 2}, "output": 2},
]


def _make(calls: list):
    @traigent.optimize(
        configuration_space={"k": [1, 2]},
        objectives=["accuracy"],
        eval_dataset=_DATASET,
        offline=True,
        algorithm="grid",
        max_trials=2,
    )
    def fn(x):
        calls.append(x)
        return x

    return fn


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


@pytest.mark.parametrize("bad", ["__MISSING__", "__DIR__", 5, b"x.json"])
def test_bad_save_to_fails_before_any_trial(tmp_path, bad):
    if bad == "__MISSING__":
        bad = str(tmp_path / "nope" / "zz" / "out.json")
    elif bad == "__DIR__":
        bad = str(tmp_path)
    calls: list = []
    fn = _make(calls)
    with pytest.raises(ConfigurationError, match="save_to"):
        fn.optimize_sync(max_trials=2, save_to=bad)
    assert calls == []  # zero trials ran


def test_error_names_the_given_path(tmp_path):
    bad = str(tmp_path / "nope" / "out.json")
    with pytest.raises(ConfigurationError) as ei:
        _make([]).optimize_sync(max_trials=1, save_to=bad)
    assert bad in str(ei.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_unwritable_parent_fails_before_any_trial(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    calls: list = []
    try:
        with pytest.raises(ConfigurationError, match="not writable"):
            _make(calls).optimize_sync(max_trials=2, save_to=str(locked / "o.json"))
    finally:
        locked.chmod(0o700)
    assert calls == []


def test_valid_save_to_still_saves(tmp_path):
    out = tmp_path / "out.json"
    result = _make([]).optimize_sync(max_trials=2, save_to=str(out))
    assert out.is_file() and out.stat().st_size > 0
    assert len(result.trials) == 2
    assert "save_to_error" not in result.metadata


def test_late_save_failure_still_returns_result(tmp_path, monkeypatch):
    def _boom(self, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ConfigStateManager, "save_optimization_results", _boom)
    out = tmp_path / "out.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _make([]).optimize_sync(max_trials=2, save_to=str(out))
    assert len(result.trials) == 2
    assert result.best_config is not None
    assert result.metadata["save_to_error"] == {
        "path": str(out),
        "error_type": "OSError",
    }
    assert any(str(out) in str(w.message) for w in caught)
