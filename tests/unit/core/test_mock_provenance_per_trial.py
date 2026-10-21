"""#2516: mock provenance is recorded from what each trial executed under.

A public ``on_trial_complete`` callback may flip mock mode mid-run; the result
must still describe the trials that actually ran, not the state at result build.
"""

from __future__ import annotations

import pytest

import traigent
from traigent.testing import disable_mock_mode
from traigent.utils.callbacks import OptimizationCallback
from traigent.utils.env_config import is_mock_llm
from traigent.utils.results_table import _mock_run_notice


class _FlipAfter(OptimizationCallback):
    """Run ``action`` after the Nth completed trial (1-based)."""

    def __init__(self, after: int, action) -> None:
        self.after = after
        self.action = action
        self.done = 0

    def on_optimization_start(self, config_space, objectives, algorithm) -> None:
        pass

    def on_trial_start(self, trial_number, config) -> None:
        pass

    def on_trial_complete(self, trial, progress) -> None:
        self.done += 1
        if self.done == self.after:
            self.action()

    def on_optimization_complete(self, result) -> None:
        pass


@pytest.fixture(autouse=True)
def _restore_mock_env(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    yield
    disable_mock_mode()


async def _run(callback: _FlipAfter, max_trials: int):
    @traigent.optimize(
        configuration_space={"x": [1, 2, 3, 4]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=max_trials,
    )
    def fn(q: str) -> str:
        return "a"

    return await fn.optimize(
        max_trials=max_trials, algorithm="grid", callbacks=[callback]
    )


def _mock_off(monkeypatch) -> None:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    disable_mock_mode()
    assert not is_mock_llm()


def _mock_on(monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    assert is_mock_llm()


@pytest.mark.asyncio
async def test_mock_trial_then_callback_disables_mock_is_still_labelled(
    monkeypatch,
) -> None:
    result = await _run(_FlipAfter(1, lambda: _mock_off(monkeypatch)), max_trials=1)

    assert result.metadata["mock_llm_run"] is True
    assert "mock_llm_mixed" not in result.metadata
    assert "MOCK RUN" in (_mock_run_notice(result) or "")


@pytest.mark.asyncio
async def test_real_trials_then_callback_enables_mock_is_not_fully_mock(
    monkeypatch,
) -> None:
    _mock_off(monkeypatch)

    result = await _run(
        _FlipAfter(1, lambda: _mock_on(monkeypatch)),
        max_trials=1,
    )

    assert result.metadata["mock_llm_run"] is False
    assert _mock_run_notice(result) is None


@pytest.mark.asyncio
async def test_trials_under_both_states_are_marked_mixed_and_labelled(
    monkeypatch,
) -> None:
    _mock_off(monkeypatch)

    result = await _run(
        _FlipAfter(1, lambda: _mock_on(monkeypatch)),
        max_trials=2,
    )

    assert result.metadata["mock_llm_run"] is True
    assert result.metadata["mock_llm_mixed"] is True
    notice = _mock_run_notice(result) or ""
    assert "mix simulated and real results" in notice
    assert "not real spend" not in notice
    assert "synthetic estimate" not in notice
