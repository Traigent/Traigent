"""Every callback class shipped in traigent/ registers with CallbackManager (#3116).

The manager requires four hooks at registration. The W&B and MLflow callbacks
lacked ``on_trial_start``, so registering them raised a TypeError.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.integrations.langfuse.callback import LangfuseOptimizationCallback
from traigent.integrations.observability.mlflow import MLflowOptimizationCallback
from traigent.integrations.observability.wandb import WandBOptimizationCallback
from traigent.utils.callbacks import (
    CallbackManager,
    DetailedProgressCallback,
    LoggingCallback,
    ManagedProgressCallback,
    ProgressBarCallback,
    ResultsTableCallback,
    SimpleProgressCallback,
    StatisticsCallback,
)

SHIPPED = [
    DetailedProgressCallback,
    LoggingCallback,
    ManagedProgressCallback,
    ProgressBarCallback,
    ResultsTableCallback,
    SimpleProgressCallback,
    StatisticsCallback,
    LangfuseOptimizationCallback,
    WandBOptimizationCallback,
    MLflowOptimizationCallback,
]


def _bare(cls):
    """Instance without running __init__ (registration only inspects hooks)."""
    return cls.__new__(cls)


@pytest.mark.parametrize("cls", SHIPPED, ids=lambda c: c.__name__)
def test_shipped_callback_registers(cls) -> None:
    manager = CallbackManager(callbacks=[])
    manager.add_callback(_bare(cls))
    assert len(manager.callbacks) == 1


def test_wandb_and_mlflow_callbacks_have_noop_trial_start() -> None:
    for cls in (WandBOptimizationCallback, MLflowOptimizationCallback):
        assert _bare(cls).on_trial_start(1, {"model": "m"}) is None


def test_wandb_callback_logs_trial_through_manager_without_failures() -> None:
    logged = []
    tracker = SimpleNamespace(log_trial=lambda trial, number: logged.append(number))
    callback = WandBOptimizationCallback.__new__(WandBOptimizationCallback)
    callback.tracker = tracker
    callback.trial_count = 0
    manager = CallbackManager(callbacks=[callback])

    manager.on_trial_complete(SimpleNamespace(trial_id="t1"), None)

    assert logged == [1]
    assert manager.callback_failures == []


class _RecordingTracker:
    """Tracker stand-in that records start_optimization_run keywords."""

    def __init__(self) -> None:
        self.started: list[dict] = []

    def start_optimization_run(self, **kwargs) -> str:
        self.started.append(kwargs)
        return "run-1"


@pytest.mark.parametrize("cls", [WandBOptimizationCallback, MLflowOptimizationCallback])
def test_manager_start_maps_config_space_and_names_the_run(cls) -> None:
    """CallbackManager passes (config_space, objectives, algorithm) by position.

    The W&B and MLflow callbacks declared (function_name, objectives,
    configuration_space), so the config space became the run name and the
    algorithm string became the configuration space.
    """
    tracker = _RecordingTracker()
    callback = cls(tracker)
    manager = CallbackManager(callbacks=[callback])
    space = {"temperature": [0.1, 0.9]}

    manager.on_optimization_start(space, ["accuracy"], "random")

    assert manager.callback_failures == []
    assert len(tracker.started) == 1
    started = tracker.started[0]
    assert started["configuration_space"] == space
    assert started["objectives"] == ["accuracy"]
    assert isinstance(started["function_name"], str)


@pytest.mark.parametrize("cls", [WandBOptimizationCallback, MLflowOptimizationCallback])
def test_constructor_function_name_names_the_run(cls) -> None:
    tracker = _RecordingTracker()
    manager = CallbackManager(callbacks=[cls(tracker, function_name="qa_agent")])

    manager.on_optimization_start({"model": ["m"]}, ["accuracy"], "grid")

    assert tracker.started[0]["function_name"] == "qa_agent"


@pytest.mark.parametrize("cls", [WandBOptimizationCallback, MLflowOptimizationCallback])
def test_keyword_start_call_still_works(cls) -> None:
    """The documented keyword form keeps working."""
    tracker = _RecordingTracker()
    cls(tracker).on_optimization_start(
        function_name="f", objectives=["accuracy"], configuration_space={"x": [1]}
    )

    started = tracker.started[0]
    assert started["function_name"] == "f"
    assert started["configuration_space"] == {"x": [1]}


@pytest.mark.parametrize("cls", [WandBOptimizationCallback, MLflowOptimizationCallback])
def test_legacy_positional_start_order_still_works(cls) -> None:
    """The old order (function_name, objectives, configuration_space) still works."""
    tracker = _RecordingTracker()
    cls(tracker).on_optimization_start("my_fn", ["accuracy"], {"x": [1, 2]})

    started = tracker.started[0]
    assert started["function_name"] == "my_fn"
    assert started["configuration_space"] == {"x": [1, 2]}
    assert started["objectives"] == ["accuracy"]
