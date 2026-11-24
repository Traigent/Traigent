"""Parallel runs report each completed trial under its own number (#2984).

Goes through the real orchestrator: the parallel scheduler fires
``on_trial_start`` with the raw config (internal keys included) while results
carry the filtered config.
"""

from __future__ import annotations

import itertools

import pytest

import traigent
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.callbacks import OptimizationCallback, SimpleProgressCallback


class _Recorder(OptimizationCallback):
    def __init__(self) -> None:
        self.started: list[dict] = []

    def on_optimization_start(self, config_space, objectives, algorithm) -> None:
        pass

    def on_trial_start(self, trial_number, config) -> None:
        self.started.append(dict(config))

    def on_trial_complete(self, trial, progress) -> None:
        pass

    def on_optimization_complete(self, result) -> None:
        pass


@pytest.mark.asyncio
async def test_parallel_completion_lines_use_distinct_trial_numbers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    # Smart optimizers tag configs with internal keys; tag the local grid the same way.
    real_suggest = GridSearchOptimizer.suggest_next_trial
    counter = itertools.count(100)

    def tagged(self, history):
        config = real_suggest(self, history)
        config["_optuna_trial_id"] = next(counter)
        config["_traigent_note"] = "internal"
        return config

    monkeypatch.setattr(GridSearchOptimizer, "suggest_next_trial", tagged)
    lines: list[str] = []
    progress = SimpleProgressCallback(output=lines.append, show_details=True)
    recorder = _Recorder()

    @traigent.optimize(
        configuration_space={"x": [1, 2, 3, 4]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=4,
    )
    def fn(q: str) -> str:
        return "a"

    await fn.optimize(
        max_trials=4,
        algorithm="grid",
        parallel_config={"trial_concurrency": 4},
        callbacks=[recorder, progress],
    )

    assert any(
        key.startswith("_") or key.startswith("__")
        for config in recorder.started
        for key in config
    ), "control: the parallel caller hands the callback configs with internal keys"
    completed = [line.split("]")[0] for line in lines if "Completed" in line]
    assert len(completed) == 4
    assert len(set(completed)) == 4, completed
