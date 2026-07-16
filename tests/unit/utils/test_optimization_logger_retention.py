"""Tests for opt-in run-dir retention pruning (Traigent/Traigent#1884).

TRAIGENT_OPTIMIZATION_LOG_MAX_RUNS lets a user cap on-disk run directories
under ``experiments/<name>/runs/`` without changing default behavior: unset
(the default) preserves the historical unbounded-growth behavior exactly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from traigent.utils.optimization_logger import (
    ENV_OPTIMIZATION_LOG_MAX_RUNS,
    OptimizationLogger,
)


def _logger(tmp_path: Path, session_id: str = "sess12345678") -> OptimizationLogger:
    return OptimizationLogger(
        experiment_name="exp",
        session_id=session_id,
        execution_mode="local",
        base_path=tmp_path,
        buffer_size=1,
    )


def _make_old_run_dir(
    runs_dir: Path, name: str, mtime: float, status: str | None = "completed"
) -> Path:
    """Simulate a prior run dir with a controlled mtime.

    ``status`` mirrors what the run writer itself produces: log_session_end()
    -> _update_status() writes ``<run_dir>/status_v2.json`` with a terminal
    ``"completed"``/``"failed"`` status; an active run carries ``"running"``;
    ``None`` = no status file at all (run created but not yet started, or
    crashed before any status write).
    """
    run_dir = runs_dir / name
    (run_dir / "meta").mkdir(parents=True)
    if status is not None:
        (run_dir / "status_v2.json").write_text(
            json.dumps({"status": status, "timestamp": "2026-01-01T00:00:00+00:00"}),
            encoding="utf-8",
        )
    os.utime(run_dir, (mtime, mtime))
    return run_dir


def _runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "experiments" / "exp" / "runs"


# --- default (env unset) => no pruning, byte-for-byte unchanged behavior ---


def test_default_env_unset_prunes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, raising=False)
    runs_dir = tmp_path / "experiments" / "exp" / "runs"
    old_1 = _make_old_run_dir(runs_dir, "20260101_000000_aaaaaaaa", mtime=1000)
    old_2 = _make_old_run_dir(runs_dir, "20260102_000000_bbbbbbbb", mtime=2000)

    log = _logger(tmp_path)

    assert old_1.is_dir()
    assert old_2.is_dir()
    assert log.run_path.is_dir()


# --- opt-in: prune oldest-beyond-N -----------------------------------------


def test_prunes_oldest_beyond_max_runs(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "2")
    runs_dir = _runs_dir(tmp_path)
    oldest = _make_old_run_dir(runs_dir, "20260101_000000_aaaaaaaa", mtime=1000)
    middle = _make_old_run_dir(runs_dir, "20260102_000000_bbbbbbbb", mtime=2000)
    newest_old = _make_old_run_dir(runs_dir, "20260103_000000_cccccccc", mtime=3000)

    log = _logger(tmp_path)

    # max_runs=2 keeps the active run + 1 completed run (the newest completed one).
    assert not oldest.is_dir()
    assert not middle.is_dir()
    assert newest_old.is_dir()
    assert log.run_path.is_dir()
    assert len(list(runs_dir.iterdir())) == 2


def test_no_pruning_when_under_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "10")
    runs_dir = _runs_dir(tmp_path)
    only_old = _make_old_run_dir(runs_dir, "20260101_000000_aaaaaaaa", mtime=1000)

    log = _logger(tmp_path)

    assert only_old.is_dir()
    assert log.run_path.is_dir()


# --- conservative guards -----------------------------------------------------


def test_never_touches_active_run(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "1")

    log = _logger(tmp_path)

    # Even with max_runs=1 and no prior runs, the just-created active run
    # must never be deleted by its own construction.
    assert log.run_path.is_dir()


def test_never_prunes_concurrent_active_run(tmp_path, monkeypatch):
    """A sibling run dir owned by ANOTHER process, still running (status file
    says "running"), must never be pruned — only self's run_path name is
    known to this process, so protection must come from completion evidence."""
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "1")
    runs_dir = _runs_dir(tmp_path)
    concurrent_active = _make_old_run_dir(
        runs_dir, "20260101_000000_aaaaaaaa", mtime=1000, status="running"
    )
    completed = _make_old_run_dir(
        runs_dir, "20260102_000000_bbbbbbbb", mtime=2000, status="completed"
    )

    log = _logger(tmp_path)

    # The concurrently-active run survives despite being the oldest and the
    # cap being 1; the completed run is the one that gets pruned.
    assert concurrent_active.is_dir()
    assert not completed.is_dir()
    assert log.run_path.is_dir()


def test_never_prunes_run_without_status_file(tmp_path, monkeypatch):
    """No completion evidence at all (no status file — e.g. a concurrent run
    between mkdir and its first status write, or a crashed run) -> skip."""
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "1")
    runs_dir = _runs_dir(tmp_path)
    no_evidence = _make_old_run_dir(
        runs_dir, "20260101_000000_aaaaaaaa", mtime=1000, status=None
    )

    log = _logger(tmp_path)

    assert no_evidence.is_dir()
    assert log.run_path.is_dir()


def test_failed_run_counts_as_completed_for_pruning(tmp_path, monkeypatch):
    """A run that finished with status "failed" is terminal — prunable."""
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "1")
    runs_dir = _runs_dir(tmp_path)
    failed_run = _make_old_run_dir(
        runs_dir, "20260101_000000_aaaaaaaa", mtime=1000, status="failed"
    )

    log = _logger(tmp_path)

    assert not failed_run.is_dir()
    assert log.run_path.is_dir()


def test_skips_unparseable_names(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "1")
    runs_dir = _runs_dir(tmp_path)
    weird = runs_dir / "not_a_run_dir_at_all"
    weird.mkdir(parents=True)
    # Give it completion evidence so the naming gate is the deciding factor.
    (weird / "status_v2.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    os.utime(weird, (1, 1))

    log = _logger(tmp_path)

    # Doesn't match the run-dir naming pattern -> left alone, "on any doubt, skip".
    assert weird.is_dir()
    assert log.run_path.is_dir()


def test_never_follows_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "1")
    runs_dir = _runs_dir(tmp_path)
    real_target = tmp_path / "outside_the_store"
    real_target.mkdir()
    (real_target / "marker.txt").write_text("do not delete me")
    # Even with completion evidence inside the target, the symlink guard
    # (checked first) must keep pruning away from it.
    (real_target / "status_v2.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    symlinked_run = runs_dir / "20260101_000000_aaaaaaaa"
    runs_dir.mkdir(parents=True, exist_ok=True)
    symlinked_run.symlink_to(real_target, target_is_directory=True)

    log = _logger(tmp_path)

    # A symlink is never followed/deleted by retention pruning, regardless
    # of whether its name matches the run-dir pattern.
    assert real_target.is_dir()
    assert (real_target / "marker.txt").exists()
    assert log.run_path.is_dir()


def test_invalid_env_value_disables_pruning_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_OPTIMIZATION_LOG_MAX_RUNS, "not-an-int")
    runs_dir = _runs_dir(tmp_path)
    old_run = _make_old_run_dir(runs_dir, "20260101_000000_aaaaaaaa", mtime=1000)

    log = _logger(tmp_path)  # must not raise

    assert old_run.is_dir()
    assert log.run_path.is_dir()
