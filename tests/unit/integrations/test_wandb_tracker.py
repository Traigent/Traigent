"""Tests for the WandB observability helpers."""

from __future__ import annotations

import types
from datetime import datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.integrations.observability import wandb as wandb_module


class DummyRun:
    def __init__(self, run_id: str = "run-123") -> None:
        self.id = run_id


class DummyWandB:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []
        self.saved: list[str] = []
        self.inits: list[dict] = []

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return DummyRun()

    def log(self, data, step=None):
        self.logged.append((data, step))

    def save(self, path):
        self.saved.append(path)

    def finish(self):
        pass


@pytest.fixture
def stubbed_wandb(monkeypatch, tmp_path) -> DummyWandB:
    dummy = DummyWandB()
    monkeypatch.setattr(wandb_module, "wandb", dummy)
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    monkeypatch.chdir(tmp_path)
    return dummy


def test_log_trial_missing_status_raises_not_skipped(
    stubbed_wandb: DummyWandB,
) -> None:
    """A trial missing the contract-guaranteed ``status`` field must fail loud.

    Regression (#1491): the old guard silently skipped logging (asserted
    ``logged == []``) when ``status`` was absent. ``TrialResult.status`` is a
    required field (``traigent/api/types.py:364``), so ``log_trial`` now reads
    ``trial.status.value`` directly — a genuinely missing status is a contract
    break that must raise (AttributeError), not be masked as a silent no-op.
    """
    tracker = wandb_module.TraigentWandBTracker()
    tracker.current_run = DummyRun()

    invalid_trial = types.SimpleNamespace(trial_id="trial-1", duration=2.5)

    with pytest.raises(AttributeError):
        tracker.log_trial(invalid_trial, trial_number=1)

    # Nothing was logged or saved (fail loud, not fake-success).
    assert stubbed_wandb.logged == []
    assert stubbed_wandb.saved == []


def test_log_trial_records_numeric_metrics(stubbed_wandb: DummyWandB) -> None:
    tracker = wandb_module.TraigentWandBTracker()
    tracker.current_run = DummyRun()

    trial = TrialResult(
        trial_id="trial-2",
        config={"model": "gpt-4o", "retries": 2},
        metrics={"accuracy": 0.8, "notes": "skip"},
        status=TrialStatus.COMPLETED,
        duration=12.34,
        timestamp=datetime.now(),
    )

    tracker.log_trial(trial, trial_number=2, step=5)

    assert len(stubbed_wandb.logged) == 1
    logged_payload, logged_step = stubbed_wandb.logged[0]
    assert logged_step == 5
    assert logged_payload["trial_2/status"] == TrialStatus.COMPLETED.value
    assert logged_payload["metrics/accuracy"] == pytest.approx(0.8)
    assert "trial_2/metrics/notes" not in logged_payload
    assert "trial_2.json" in stubbed_wandb.saved


def test_init_wandb_run_accepts_default_arguments(stubbed_wandb: DummyWandB) -> None:
    run_id = wandb_module.init_wandb_run(
        function_name="optimize_me",
        objectives=["accuracy"],
        configuration_space={"lr": [0.01, 0.1]},
        project="demo",
        entity="team",
        tags=["baseline"],
        additional_tags=["auto"],
        config={"env": "test"},
    )

    assert run_id == "run-123"
    assert stubbed_wandb.inits
    init_kwargs = stubbed_wandb.inits[0]
    assert init_kwargs["project"] == "demo"
    assert "optimize_me" in init_kwargs["name"]
