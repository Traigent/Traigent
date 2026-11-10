"""#3046: one session record with a null or non-ISO created_at must not break
list_sessions() or cleanup_old_sessions() (nor the cleanup --dry-run preview).

The record is skipped, counted and warned about, like an unreadable file.
Offline: real LocalStorageManager over tmp_path.
"""

from __future__ import annotations

import json
import logging

import pytest

from traigent.cli.local_commands import _echo_sessions_older_than
from traigent.storage.local_storage import LocalStorageManager

_OLD = "2020-01-01T00:00:00+00:00"


def _session(storage: LocalStorageManager, created_at) -> str:
    sid = storage.create_session(function_name="f")
    storage.finalize_session(sid, "completed")
    path = storage.storage_path / "sessions" / f"{sid}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["created_at"] = created_at
    path.write_text(json.dumps(data), encoding="utf-8")
    return sid


@pytest.mark.parametrize("bad", [None, "garbage", 17], ids=["null", "non-iso", "int"])
def test_list_sessions_skips_and_counts_bad_created_at(tmp_path, caplog, bad):
    storage = LocalStorageManager(str(tmp_path))
    good = _session(storage, _OLD)
    bad_sid = _session(storage, bad)
    with caplog.at_level(logging.WARNING, logger="traigent.storage.local_storage"):
        sessions = storage.list_sessions()
    assert [s.session_id for s in sessions] == [good]
    assert storage.last_skipped_sessions == 1
    assert any(
        "1 session" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )
    # A direct by-id load is unchanged: the record is still readable.
    assert storage.load_session(bad_sid) is not None


@pytest.mark.parametrize("bad", [None, "garbage"], ids=["null", "non-iso"])
def test_cleanup_old_sessions_completes_past_bad_record(tmp_path, bad):
    storage = LocalStorageManager(str(tmp_path))
    _session(storage, _OLD)
    _session(storage, bad)
    _session(storage, _OLD)
    assert storage.cleanup_old_sessions(days=30) == 2


def test_cleanup_dry_run_preview_survives_bad_record(tmp_path, capsys):
    storage = LocalStorageManager(str(tmp_path))
    _session(storage, _OLD)
    _session(storage, "garbage")
    _echo_sessions_older_than(storage, 30)
    assert "Would delete 1 sessions" in capsys.readouterr().out


def test_mixed_naive_and_aware_sort_newest_first(tmp_path):
    storage = LocalStorageManager(str(tmp_path))
    old = _session(storage, "2020-01-01T00:00:00")  # naive: read as UTC
    new = _session(storage, "2021-01-01T00:00:00+00:00")
    assert [s.session_id for s in storage.list_sessions()] == [new, old]
    assert storage.last_skipped_sessions == 0
