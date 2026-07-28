"""``traigent local`` must report a truthful ``total_trials`` (#2032).

Before the fix ``OptimizationSession.total_trials`` was stamped 0 at
construction and never written again, so ``local show`` printed "3/0" and any
completion ratio built from the pair divided by zero. ``total_trials`` means
RECORDED trials (``== len(trials)``), never the ``max_trials`` budget.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from traigent.cli.local_commands import local_commands
from traigent.storage.local_storage import LocalStorageManager


def _env(storage_root: Path) -> dict[str, str]:
    return {
        "TRAIGENT_RESULTS_FOLDER": str(storage_root),
        "TRAIGENT_MINIMAL_LOGGING": "true",
        "TRAIGENT_OFFLINE": "true",
        "TRAIGENT_OFFLINE_MODE": "true",
        "TRAIGENT_ENABLE_USAGE_ANALYTICS": "false",
    }


def _session_file(storage_root: Path, session_id: str) -> Path:
    return storage_root / "sessions" / f"{session_id}.json"


def _make_session(storage_root: Path, trial_count: int, *, legacy: bool) -> str:
    """Create a finalized session; optionally rewrite it pre-#2032 style."""
    storage = LocalStorageManager(str(storage_root))
    session_id = storage.create_session(
        "cli_total_trials",
        # A 10-trial budget — the exact value backend_session_manager.py:1387
        # fabricates when the caller supplied none. It must not leak into
        # total_trials.
        optimization_config={"max_trials": 10},
        metadata={"max_trials": 10},
    )
    for index in range(trial_count):
        storage.add_trial_result(session_id, {"param": index}, 0.5 + index * 0.1)
    storage.finalize_session(session_id, "completed")

    if legacy:
        path = _session_file(storage_root, session_id)
        raw = json.loads(path.read_text())
        raw["total_trials"] = 0
        path.write_text(json.dumps(raw, indent=2))

    return session_id


def test_local_show_detailed_reports_recorded_trials(tmp_path):
    """`local show --format detailed` prints 3/3, not the old 3/0."""
    storage_root = tmp_path / "storage"
    session_id = _make_session(storage_root, 3, legacy=False)

    result = CliRunner().invoke(
        local_commands,
        ["show", session_id, "--format", "detailed"],
        env=_env(storage_root),
    )

    assert result.exit_code == 0, result.output
    assert "Trials: 3/3" in result.output
    assert "Trials: 3/0" not in result.output
    # The declared budget of 10 is not the recorded count.
    assert "Trials: 3/10" not in result.output


def test_local_show_json_reports_recorded_trials(tmp_path):
    """`local show --format json` emits the recorded count."""
    storage_root = tmp_path / "storage"
    session_id = _make_session(storage_root, 3, legacy=False)

    result = CliRunner().invoke(
        local_commands,
        ["show", session_id, "--format", "json"],
        env=_env(storage_root),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_trials"] == 3
    assert payload["completed_trials"] == 3
    assert payload["total_trials"] == len(payload["trials"])


def test_local_show_detailed_reconciles_legacy_zero_record(tmp_path):
    """A pre-#2032 record on disk still displays a truthful count."""
    storage_root = tmp_path / "storage"
    session_id = _make_session(storage_root, 2, legacy=True)

    result = CliRunner().invoke(
        local_commands,
        ["show", session_id, "--format", "detailed"],
        env=_env(storage_root),
    )

    assert result.exit_code == 0, result.output
    assert "Trials: 2/2" in result.output
    assert "Trials: 2/0" not in result.output


def test_local_show_json_reconciles_legacy_zero_record(tmp_path):
    """The JSON surface reconciles too, and leaves the file untouched."""
    storage_root = tmp_path / "storage"
    session_id = _make_session(storage_root, 2, legacy=True)
    before = _session_file(storage_root, session_id).read_bytes()

    result = CliRunner().invoke(
        local_commands,
        ["show", session_id, "--format", "json"],
        env=_env(storage_root),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total_trials"] == 2
    # Reading is not a write path.
    assert _session_file(storage_root, session_id).read_bytes() == before


def test_local_list_json_reports_recorded_trials(tmp_path):
    """`local list --format json` renders the session-summary surface."""
    storage_root = tmp_path / "storage"
    _make_session(storage_root, 3, legacy=True)

    result = CliRunner().invoke(
        local_commands,
        ["list", "--format", "json"],
        env=_env(storage_root),
    )

    assert result.exit_code == 0, result.output
    summaries = json.loads(result.output)
    assert len(summaries) == 1
    assert summaries[0]["total_trials"] == 3
    assert summaries[0]["completed_trials"] == 3


def test_local_info_totals_include_legacy_sessions(tmp_path):
    """`local info` aggregates recorded trials across sessions."""
    storage_root = tmp_path / "storage"
    _make_session(storage_root, 2, legacy=True)

    result = CliRunner().invoke(local_commands, ["info"], env=_env(storage_root))

    assert result.exit_code == 0, result.output
    assert "Total Trials: 2" in result.output
