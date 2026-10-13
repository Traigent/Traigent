"""Without dir_fd support (Windows branch) delete must refuse symlinked parents."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traigent.storage.local_storage import LocalStorageManager


def test_symlinked_sessions_dir_refused_without_dir_fd(
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
        path = real_path(session_id)
        sessions.rename(tmp_path / "sessions-moved")
        try:
            sessions.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks unavailable")
        return path

    monkeypatch.setattr(mgr, "_session_file_path", validate_then_swap)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) - {os.unlink})

    assert mgr.delete_session(sid) is False
    assert victim.read_text() == "keep me"


def test_symlinked_leaf_refused_without_dir_fd(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "store"
    mgr = LocalStorageManager(str(store))
    sid = mgr.create_session("f")
    leaf = store / "sessions" / f"{sid}.json"
    target = tmp_path / "target.json"
    target.write_text("keep me")
    leaf.unlink()
    leaf.symlink_to(target)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) - {os.unlink})

    assert mgr.delete_session(sid) is False
    assert target.read_text() == "keep me"
    assert leaf.is_symlink()


def test_regular_record_still_deleted_without_dir_fd(
    tmp_path: Path, monkeypatch
) -> None:
    mgr = LocalStorageManager(str(tmp_path / "store"))
    sid = mgr.create_session("f")
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) - {os.unlink})
    assert mgr.delete_session(sid) is True
