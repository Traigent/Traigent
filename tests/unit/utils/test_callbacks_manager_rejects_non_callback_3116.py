"""CallbackManager rejects non-callback entries at registration (#3116)."""

from __future__ import annotations

import pytest

from traigent.utils.callbacks import CallbackManager, OptimizationCallback


# --- #3116 -----------------------------------------------------------------
class _Good(OptimizationCallback):
    def on_optimization_start(self, config_space, objectives, algorithm):
        pass

    def on_trial_start(self, trial_number, config):
        pass

    def on_trial_complete(self, trial, progress):
        pass

    def on_optimization_complete(self, result):
        pass


def test_manager_rejects_non_callback_at_registration() -> None:
    with pytest.raises(TypeError, match="NoneType"):
        CallbackManager([None])  # type: ignore[list-item]
    mgr = CallbackManager([_Good()])
    with pytest.raises(TypeError, match="object"):
        mgr.add_callback(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="function"):
        mgr.add_callback(lambda *a: None)  # type: ignore[arg-type]
    assert len(mgr.callbacks) == 1


def test_manager_accepts_valid_callback() -> None:
    mgr = CallbackManager([_Good()])
    mgr.add_callback(_Good())
    mgr.on_optimization_start({}, ["a"], "g")
    assert mgr.get_failure_summary()["total_failures"] == 0
