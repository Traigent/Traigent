"""MLflow / W&B autolog callbacks complete a real @optimize run (#3187).

Before the fix both callbacks lacked ``on_trial_start`` (the run aborted at
trial 1), received ``(config_space, objectives, algorithm)`` bound to the wrong
parameters (the run name became the config dict) and took ``on_trial_complete``
without ``progress``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import traigent
from traigent.integrations.observability import mlflow as mlflow_module
from traigent.integrations.observability import wandb as wandb_module
from traigent.utils.callbacks import CallbackManager, OptimizationCallback

CONFIG_SPACE = {"a": [1, 2, 3]}


class _RecordingTracker:
    """Stands in for the tracker boundary (no mlflow / wandb needed)."""

    def __init__(self) -> None:
        self.start: dict[str, Any] | None = None
        self.trials: list[tuple[Any, int]] = []
        self.results: list[Any] = []
        self.ended = 0

    def start_optimization_run(self, **kwargs: Any) -> str:
        self.start = kwargs
        return "run-1"

    def log_trial(self, trial: Any, number: int) -> None:
        self.trials.append((trial, number))

    def log_optimization_result(self, result: Any, dataset_path: Any = None) -> None:
        self.results.append(result)

    def end_optimization_run(self) -> None:
        self.ended += 1


CALLBACK_CLASSES = [
    mlflow_module.MLflowOptimizationCallback,
    wandb_module.WandBOptimizationCallback,
]


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


def _run(tmp_path, callback: Any):
    ds = tmp_path / "dataset.jsonl"
    ds.write_text(
        "\n".join(json.dumps({"input": {"x": i}, "output": str(i)}) for i in range(4))
    )

    @traigent.optimize(
        eval_dataset=str(ds),
        objectives=["accuracy"],
        configuration_space=CONFIG_SPACE,
        injection_mode="parameter",
        config_param="config",
        offline=True,
        scoring_function=lambda output, expected, **k: float(output == expected),
    )
    def my_agent_fn(x, config):
        return str(x)

    return asyncio.run(
        my_agent_fn.optimize(algorithm="grid", max_trials=3, callbacks=[callback])
    )


@pytest.mark.parametrize("cls", CALLBACK_CLASSES)
def test_is_an_optimization_callback(cls) -> None:
    assert issubclass(cls, OptimizationCallback)


@pytest.mark.parametrize("cls", CALLBACK_CLASSES)
def test_optimize_run_completes_all_trials_with_correct_tracker_args(
    cls, tmp_path
) -> None:
    tracker = _RecordingTracker()
    result = _run(tmp_path, cls(tracker))

    assert len(result.trials) == 3
    assert [n for _, n in tracker.trials] == [1, 2, 3]
    assert [t.trial_id for t, _ in tracker.trials] == [
        t.trial_id for t in result.trials
    ]
    assert len(tracker.results) == 1 and tracker.ended == 1

    start = tracker.start
    assert start is not None
    assert start["objectives"] == ["accuracy"]
    assert start["configuration_space"] == CONFIG_SPACE
    assert isinstance(start["function_name"], str)
    assert "my_agent_fn" in start["function_name"]


@pytest.mark.parametrize("cls", CALLBACK_CLASSES)
def test_manager_dispatch_without_run_settings_uses_fixed_prefix(cls) -> None:
    tracker = _RecordingTracker()
    manager = CallbackManager([cls(tracker)], timeout=0)
    manager.on_optimization_start({"t": [0, 1]}, ["accuracy"], "grid")
    manager.on_trial_start(1, {"t": 0})

    assert tracker.start is not None
    assert tracker.start["function_name"] == "traigent"
    assert tracker.start["configuration_space"] == {"t": [0, 1]}
    assert tracker.start["objectives"] == ["accuracy"]
    assert manager.get_failure_summary()["total_failures"] == 0


def test_mlflow_run_name_is_not_the_config_dict(monkeypatch, tmp_path) -> None:
    fake = MagicMock()
    monkeypatch.setattr(mlflow_module, "mlflow", fake)
    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    tracker = mlflow_module.TraigentMLflowTracker()
    manager = CallbackManager([mlflow_module.MLflowOptimizationCallback(tracker)], 0)

    manager.on_optimization_start(CONFIG_SPACE, ["accuracy"], "GridSearchOptimizer")

    run_name = fake.start_run.call_args.kwargs["run_name"]
    assert "{" not in run_name and "[" not in run_name
    assert run_name.startswith("traigent_optimization_")
    assert manager.get_failure_summary()["total_failures"] == 0


class _NoTrialStart:
    """A third-party callback missing an optional hook."""

    def __init__(self) -> None:
        self.completed = 0

    def on_optimization_start(self, *a: Any) -> None:  # pragma: no cover
        pass

    def on_trial_complete(self, trial: Any, progress: Any) -> None:
        self.completed += 1


def test_manager_skips_missing_hook_with_one_warning(caplog) -> None:
    cb = _NoTrialStart()
    manager = CallbackManager([cb], timeout=0)
    with caplog.at_level("WARNING"):
        manager.on_trial_start(1, {})
        manager.on_trial_start(2, {})
    warnings = [r for r in caplog.records if "on_trial_start" in r.getMessage()]
    assert len(warnings) == 1
    manager.on_trial_complete(MagicMock(), MagicMock())
    assert cb.completed == 1


def test_manager_still_surfaces_errors_raised_inside_a_hook() -> None:
    class Boom(_NoTrialStart):
        def on_trial_start(self, n: int, c: dict[str, Any]) -> None:
            raise RuntimeError("boom")

    manager = CallbackManager([Boom()], timeout=0)
    manager.on_trial_start(1, {})
    summary = manager.get_failure_summary()
    assert summary["total_failures"] == 1
    assert summary["failures"][0]["method"] == "on_trial_start"


class _ThreadLocalMLflow:
    """MLflow double with a thread-local active-run stack, like the real one."""

    def __init__(self) -> None:
        import threading

        self._local = threading.local()
        self.resumed: list[str] = []
        self.logged_to: list[str | None] = []
        self.ended_run: str | None = None

    def _stack(self) -> list[Any]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def start_run(self, run_name=None, run_id=None):
        run = MagicMock()
        run.info.run_id = run_id or "run-A"
        if run_id:
            self.resumed.append(run_id)
        self._stack().append(run)
        return run

    def active_run(self):
        stack = self._stack()
        return stack[-1] if stack else None

    def end_run(self, status: str | None = None) -> None:
        active = self.active_run()
        if status != "RUNNING":
            self.ended_run = active.info.run_id if active else None
        self._stack().pop()

    def log_param(self, *a, **k) -> None:
        active = self.active_run()
        self.logged_to.append(active.info.run_id if active else None)

    def __getattr__(self, name: str):
        return lambda *a, **k: None


def test_mlflow_hooks_on_different_threads_stay_bound_to_the_run(monkeypatch) -> None:
    import threading

    fake = _ThreadLocalMLflow()
    monkeypatch.setattr(mlflow_module, "mlflow", fake)
    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    tracker = mlflow_module.TraigentMLflowTracker()
    callback = mlflow_module.MLflowOptimizationCallback(tracker)
    callback.on_optimization_start(CONFIG_SPACE, ["accuracy"], "grid")

    errors: list[BaseException] = []

    def other_thread() -> None:
        try:
            tracker.end_optimization_run()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert not errors
    assert fake.resumed == ["run-A"]
    assert fake.ended_run == "run-A"


def test_worker_thread_logging_does_not_leave_run_active_on_that_thread(
    monkeypatch,
) -> None:
    """A pooled worker that logged a trial must not keep the run on its stack."""
    import threading

    fake = _ThreadLocalMLflow()
    monkeypatch.setattr(mlflow_module, "mlflow", fake)
    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    tracker = mlflow_module.TraigentMLflowTracker()
    mlflow_module.MLflowOptimizationCallback(tracker).on_optimization_start(
        CONFIG_SPACE, ["accuracy"], "grid"
    )
    from types import SimpleNamespace

    trial = SimpleNamespace(
        config={"x": 1},
        metrics={},
        duration=0.1,
        trial_id="t1",
        error_message=None,
        timestamp=None,
        status=SimpleNamespace(value="completed"),
        metadata={},
    )
    seen: list[Any] = []

    def worker() -> None:
        tracker.log_trial(trial, 1)
        seen.append(fake.active_run())

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert fake.resumed == ["run-A"]
    assert seen == [None]
    assert set(fake.logged_to) == {"run-A"}
    assert fake.ended_run is None  # still open for the owner to finish
    assert tracker.current_run is not None


@pytest.mark.parametrize("cls", CALLBACK_CLASSES)
def test_legacy_mixed_positional_and_keyword_call(cls) -> None:
    tracker = _RecordingTracker()
    cls(tracker).on_optimization_start(
        "agent", objectives=["accuracy"], configuration_space={"x": [1]}
    )
    assert tracker.start is not None
    assert tracker.start["function_name"] == "agent"
    assert tracker.start["configuration_space"] == {"x": [1]}
