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
