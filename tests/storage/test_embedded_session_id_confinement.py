"""An id embedded in a stored session file must not steer writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.storage.local_storage import LocalStorageManager


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
    if op == "update":
        mgr.update_session_status("safe", "failed")
    else:
        assert mgr.finalize_session("safe") is None
    after = _snapshot(tmp_path)
    assert after == before
    assert not (store / "victim.json").exists()
    assert not list(tmp_path.glob("**/victim*"))
    assert mgr.load_session("safe") is None


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
