"""Security-focused tests for local CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from traigent.cli.local_commands import edge_analytics_commands
from traigent.storage.local_storage import LocalStorageManager


def _create_session(storage: LocalStorageManager) -> str:
    session_id = storage.create_session("demo_function")
    storage.finalize_session(session_id)
    return session_id


def test_export_session_default_path_within_storage(tmp_path, monkeypatch):
    """Export should write inside the storage exports directory by default."""
    storage_root = tmp_path / "storage"
    env = {
        "TRAIGENT_RESULTS_FOLDER": str(storage_root),
        "TRAIGENT_EDGE_ANALYTICS_MODE": "true",
    }

    storage = LocalStorageManager(str(storage_root))
    session_id = _create_session(storage)

    runner = CliRunner()
    result = runner.invoke(edge_analytics_commands, ["export", session_id], env=env)

    assert result.exit_code == 0
    exported_file = storage_root / "exports" / f"{session_id}.json"
    assert exported_file.exists()


def test_export_session_rejects_path_outside_storage(tmp_path, monkeypatch):
    """Export should refuse to write files outside the storage directory."""
    storage_root = tmp_path / "storage"
    env = {
        "TRAIGENT_RESULTS_FOLDER": str(storage_root),
        "TRAIGENT_EDGE_ANALYTICS_MODE": "true",
    }

    storage = LocalStorageManager(str(storage_root))
    session_id = _create_session(storage)

    outside_path = tmp_path.parent / f"{session_id}.json"
    if outside_path.exists():
        outside_path.unlink()

    runner = CliRunner()
    result = runner.invoke(
        edge_analytics_commands,
        ["export", session_id, "--output", str(outside_path)],
        env=env,
    )

    assert result.exit_code != 0
    assert "Output path must reside within the local storage directory" in result.output
    assert not outside_path.exists()
