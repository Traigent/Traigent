"""An id embedded in a stored session file must not steer writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.exceptions import TraigentStorageError


def _plant(tmp_path: Path, embedded: str) -> tuple[LocalStorageManager, Path]:
    store = tmp_path / "store"
    mgr = LocalStorageManager(str(store))
    sid = mgr.create_session("f")
    sessions = store / "sessions"
    data = json.loads((sessions / f"{sid}.json").read_text())
    data["session_id"] = embedded
    (sessions / "safe.json").write_text(json.dumps(data))
    return mgr, store


def _snapshot(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("embedded", ["../victim", "..\\victim", "a/b", "C:victim"])
@pytest.mark.parametrize("op", ["update", "finalize"])
def test_embedded_id_cannot_write_outside_sessions(
    tmp_path: Path, embedded: str, op: str
) -> None:
    mgr, store = _plant(tmp_path, embedded)
    before = _snapshot(tmp_path)
    with pytest.raises(TraigentStorageError):
        if op == "update":
            mgr.update_session_status("safe", "failed")
        else:
            mgr.finalize_session("safe")
    after = _snapshot(tmp_path)
    assert after == before
    assert not (store / "victim.json").exists()
    assert not list(tmp_path.glob("**/victim*"))
    with pytest.raises(TraigentStorageError):
        mgr.load_session("safe")
    assert "safe" not in {x.session_id for x in mgr.list_sessions()}


def test_save_session_rejects_unsafe_id_directly(tmp_path: Path) -> None:
    from traigent.utils.exceptions import TraigentStorageError

    mgr, store = _plant(tmp_path, "ok")
    session = mgr.load_session(
        next(p.stem for p in (store / "sessions").glob("*.json") if p.stem != "safe")
    )
    assert session is not None
    session.session_id = "../victim"
    with pytest.raises(TraigentStorageError):
        mgr._save_session(session)
    assert not (store / "victim.json").exists()


def test_matching_embedded_id_still_loads(tmp_path: Path) -> None:
    mgr = LocalStorageManager(str(tmp_path / "s"))
    sid = mgr.create_session("f")
    assert mgr.load_session(sid) is not None
    mgr.update_session_status(sid, "failed")
    assert mgr.load_session(sid).status == "failed"


def test_delete_session_traversal_leaves_victim(tmp_path: Path) -> None:
    mgr = LocalStorageManager(str(tmp_path / "store"))
    victim = tmp_path / "store" / "victim.json"
    victim.write_text("keep")
    assert mgr.delete_session("../victim") is False
    assert victim.read_text() == "keep"


def test_delete_session_symlink_out_of_sessions_not_followed(tmp_path: Path) -> None:
    mgr = LocalStorageManager(str(tmp_path / "store"))
    victim = tmp_path / "store" / "victim.json"
    victim.write_text("keep")
    link = tmp_path / "store" / "sessions" / "safe.json"
    link.symlink_to(victim)
    assert mgr.delete_session("safe") is False
    assert victim.read_text() == "keep"


def test_create_session_does_not_overwrite_rejected_record(tmp_path: Path) -> None:
    mgr, store = _plant(tmp_path, "other")
    safe = store / "sessions" / "safe.json"
    before = safe.read_bytes()
    with pytest.raises(TraigentStorageError):
        mgr.create_session("f", session_id="safe")
    assert safe.read_bytes() == before


def test_create_session_does_not_overwrite_corrupt_record(tmp_path: Path) -> None:
    mgr = LocalStorageManager(str(tmp_path / "store"))
    safe = tmp_path / "store" / "sessions" / "safe.json"
    safe.write_text("{not json")
    with pytest.raises(TraigentStorageError):
        mgr.create_session("f", session_id="safe")
    assert safe.read_text() == "{not json"
