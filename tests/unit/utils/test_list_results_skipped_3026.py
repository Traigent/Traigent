"""#3026: list_results must skip a corrupt result loudly (logged and counted).

A corrupt metadata.json was dropped with a bare ``continue``: no log, no count.
Offline: real PersistenceManager over tmp_path.
"""

from __future__ import annotations

import json
import logging

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.utils.persistence import PersistenceManager

_LOGGER = "traigent.utils.persistence"


def _write(base, name, raw: bytes) -> None:
    d = base / name
    d.mkdir()
    (d / "metadata.json").write_bytes(raw)


def _good(base, name, created_at="2026-01-01T00:00:00+00:00") -> None:
    _write(
        base,
        name,
        json.dumps(
            {
                "created_at": created_at,
                "function_name": "f",
                "algorithm": "grid",
                "total_trials": 1,
            }
        ).encode(),
    )


# --- #3026: corrupt JSON is skipped with a warning and counted -------------


def test_corrupt_json_is_logged_and_counted(tmp_path, caplog):
    _good(tmp_path, "good1", "2026-01-01T00:00:00+00:00")
    _good(tmp_path, "good2", "2026-01-02T00:00:00+00:00")
    _write(tmp_path, "trunc", b'{"created_at": ')
    _write(tmp_path, "notjson", b"not json")
    pm = PersistenceManager(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        names = [r["name"] for r in pm.list_results()]
    assert names == ["good2", "good1"]
    assert pm.last_skipped_results == 2
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("trunc" in m for m in messages)
    assert any("notjson" in m for m in messages)
    # The directory is named, never the file contents.
    assert not any("created_at" in m for m in messages)


def test_clean_listing_reports_zero_skipped(tmp_path):
    _good(tmp_path, "good1")
    pm = PersistenceManager(tmp_path)
    assert [r["name"] for r in pm.list_results()] == ["good1"]
    assert pm.last_skipped_results == 0


def test_cli_results_list_prints_unreadable_count(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    _good(store, "good1")
    _write(store, "trunc", b'{"created_at": ')
    result = CliRunner().invoke(cli, ["results", "list", "--storage-dir", str(store)])
    assert result.exit_code == 0, result.output
    assert "1 result(s) unreadable" in result.output
