"""#3042: W&B artifacts must land in the run's own files dir and actually exist.

Runs against the real ``wandb`` package in offline mode (no network). The
artifacts are staged inside ``wandb.run.dir`` (wandb's own directory), so the
file a run uploads is a real file under its requested name, not a dangling
symlink into a deleted temp directory.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability FUNC-INTEGRATIONS
# Traceability: REQ-INT-008 SYNC-Observability

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

wandb = pytest.importorskip("wandb")

from traigent.api.types import TrialStatus  # noqa: E402
from traigent.integrations.observability import wandb as wandb_module  # noqa: E402


@pytest.fixture
def offline_wandb(monkeypatch, tmp_path):
    """Real wandb run in offline mode, rooted under tmp_path."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    monkeypatch.setenv("WANDB_ANONYMOUS", "allow")
    monkeypatch.setenv("WANDB_DISABLE_GIT", "true")
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb-cache"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb-config"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(wandb_module, "wandb", wandb)
    work = tmp_path / "project"
    work.mkdir()
    monkeypatch.chdir(work)
    yield work
    if wandb.run is not None:
        wandb.finish()


def _trial():
    return SimpleNamespace(
        trial_id="t1",
        status=TrialStatus.COMPLETED,
        duration=1.0,
        config={"model": "m"},
        metrics={"acc": 0.5},
        timestamp=datetime(2025, 1, 2, tzinfo=UTC),
    )


def _assert_real_file(run_dir: Path, name: str, expected: dict) -> None:
    """The uploaded name is a real file (not a symlink into a deleted dir)."""
    target = run_dir / name
    assert target.is_file(), f"{name} missing or dangling in {run_dir}"
    assert json.loads(target.read_text()) == expected


def test_configuration_space_artifact_is_a_real_file_under_its_name(
    offline_wandb,
) -> None:
    tracker = wandb_module.TraigentWandBTracker(project="p3042")
    tracker.start_optimization_run("fn", ["accuracy"], {"model": ["a", "b"]})

    run_dir = Path(tracker.current_run.dir)
    _assert_real_file(run_dir, "configuration_space.json", {"model": ["a", "b"]})
    assert sorted(os.listdir(offline_wandb)) == []  # CWD untouched


def test_trial_artifact_is_a_real_file_under_its_name(offline_wandb) -> None:
    tracker = wandb_module.TraigentWandBTracker(project="p3042")
    tracker.start_optimization_run("fn", ["accuracy"], {"model": ["a"]})

    tracker.log_trial(_trial(), 1)

    run_dir = Path(tracker.current_run.dir)
    _assert_real_file(
        run_dir,
        "trial_1.json",
        {
            "trial_id": "t1",
            "trial_number": 1,
            "config": {"model": "m"},
            "metrics": {"acc": 0.5},
            "duration": 1.0,
            "status": "completed",
            "timestamp": "2025-01-02T00:00:00+00:00",
        },
    )
    assert sorted(os.listdir(offline_wandb)) == []  # CWD untouched
