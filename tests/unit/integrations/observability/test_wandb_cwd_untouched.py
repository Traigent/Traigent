"""#3042: the W&B tracker must never read, truncate or delete files in the CWD."""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability FUNC-INTEGRATIONS
# Traceability: REQ-INT-008 SYNC-Observability

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from traigent.api.types import TrialStatus
from traigent.integrations.observability import wandb as wandb_module

USER_FILES = ("configuration_space.json", "trial_1.json", "optimization_result.json")
USER_CONTENT = json.dumps({"user": "data"})


class _RecordingWandb:
    """Stub of the wandb module; reads each saved file when ``save`` is called."""

    def __init__(self) -> None:
        self.saved: dict[str, dict] = {}
        self.save_kwargs: list[dict] = []
        self.saved_paths: list[str] = []

    def init(self, **_kwargs):
        return SimpleNamespace(id="run-1")

    def log(self, *_args, **_kwargs) -> None:
        pass

    def save(self, path, **kwargs) -> None:
        self.saved_paths.append(str(path))
        self.save_kwargs.append(kwargs)
        self.saved[Path(path).name] = json.loads(Path(path).read_text())


@pytest.fixture
def project_dir(monkeypatch, tmp_path):
    fake = _RecordingWandb()
    monkeypatch.setattr(wandb_module, "wandb", fake)
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    work = tmp_path / "project"
    work.mkdir()
    for name in USER_FILES:
        (work / name).write_text(USER_CONTENT)
    monkeypatch.chdir(work)
    return work, fake


def _trial():
    return SimpleNamespace(
        trial_id="t1",
        status=TrialStatus.COMPLETED,
        duration=1.0,
        config={"model": "m"},
        metrics={"acc": 0.5},
        timestamp=datetime(2025, 1, 2, tzinfo=UTC),
    )


def _result():
    return SimpleNamespace(
        best_config={"model": "m"},
        best_metrics={"acc": 0.5},
        trials=[_trial()],
        duration=2.0,
        status=SimpleNamespace(value="completed"),
        metadata={},
    )


def _assert_user_files_intact(work: Path) -> None:
    assert sorted(os.listdir(work)) == sorted(USER_FILES)
    for name in USER_FILES:
        assert (work / name).read_text() == USER_CONTENT


def test_start_run_leaves_user_files_alone(project_dir) -> None:
    work, fake = project_dir
    tracker = wandb_module.TraigentWandBTracker()

    tracker.start_optimization_run("fn", ["accuracy"], {"model": ["a", "b"]})

    assert fake.saved["configuration_space.json"] == {"model": ["a", "b"]}
    _assert_user_files_intact(work)


def test_log_trial_leaves_user_files_alone(project_dir) -> None:
    work, fake = project_dir
    tracker = wandb_module.TraigentWandBTracker()
    tracker.current_run = SimpleNamespace(id="run-1")

    tracker.log_trial(_trial(), 1)

    assert fake.saved["trial_1.json"]["trial_id"] == "t1"
    _assert_user_files_intact(work)


def test_log_result_leaves_user_files_alone(project_dir) -> None:
    work, fake = project_dir
    tracker = wandb_module.TraigentWandBTracker()
    tracker.current_run = SimpleNamespace(id="run-1")

    tracker.log_optimization_result(_result())

    assert fake.saved["optimization_result.json"]["total_trials"] == 1
    _assert_user_files_intact(work)


def test_artifacts_are_staged_outside_cwd_and_cleaned_up(project_dir) -> None:
    work, fake = project_dir
    tracker = wandb_module.TraigentWandBTracker()
    tracker.start_optimization_run("fn", ["accuracy"], {"model": ["a"]})
    tracker.log_trial(_trial(), 1)
    tracker.log_optimization_result(_result())

    assert len(fake.saved_paths) == 3
    for saved in fake.saved_paths:
        assert os.path.isabs(saved)
        assert Path(saved).parent != work
        assert not Path(saved).exists()  # staging dir removed after the call
    # Uploaded immediately, before the staging directory disappears.
    assert all(kw.get("policy") == "now" for kw in fake.save_kwargs)
