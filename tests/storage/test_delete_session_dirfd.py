"""delete_session must not follow a sessions/ directory swapped for a symlink."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traigent.storage.local_storage import LocalStorageManager

pytestmark = pytest.mark.skipif(
    os.unlink not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"),
    reason="needs dir_fd / O_NOFOLLOW support",
)


def test_parent_swapped_for_symlink_after_validation_deletes_nothing_outside(
    tmp_path: Path, monkeypatch
) -> None:
    store = tmp_path / "store"
    mgr = LocalStorageManager(str(store))
    sid = mgr.create_session("f")
    sessions = store / "sessions"
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / f"{sid}.json"
    victim.write_text("keep me")

    real_path = mgr._session_file_path

    def validate_then_swap(session_id):
        path = real_path(session_id)  # validation passes against the real dir
        moved = tmp_path / "sessions-moved"
        sessions.rename(moved)
        sessions.symlink_to(outside, target_is_directory=True)
        return path

    monkeypatch.setattr(mgr, "_session_file_path", validate_then_swap)

    assert mgr.delete_session(sid) is False
    assert victim.read_text() == "keep me"


def test_delete_still_removes_regular_record(tmp_path: Path) -> None:
    mgr = LocalStorageManager(str(tmp_path / "store"))
    sid = mgr.create_session("f")
    assert mgr.delete_session(sid) is True
    assert mgr.delete_session(sid) is False
