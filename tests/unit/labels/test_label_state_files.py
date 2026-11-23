"""State-file regressions for the label transport: no Schema checkout needed.

These tests use only local temp files and the transport's state-file helpers, so they
run in the default PR gate without TRAIGENT_SCHEMA_DIR.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_needs_fcntl = pytest.mark.skipif(
    importlib.util.find_spec("fcntl") is None,
    reason="the state writer needs fcntl advisory locks",
)


_needs_fcntl = pytest.mark.skipif(
    importlib.util.find_spec("fcntl") is None,
    reason="the state writer needs fcntl advisory locks",
)


@_needs_fcntl
def test_normal_state_path_inside_state_dir_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.labels import transport as t

    state_dir = tmp_path / "state"
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(state_dir))
    state = t._state_path()
    assert state == state_dir / "labels_state.json"
    t._retain_sequence(state, "s", 4)
    assert t._retained_sequence(state, "s") == 4
    assert json.loads(state.read_text("utf-8")) == {"s": 4}


def test_state_file_containment_refuses_symlink_nul_and_dotdot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.labels import transport as t

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(state_dir))
    ok = t._state_path()
    assert t._require_local_file(ok) == ok.resolve()
    outside = tmp_path / "outside.lock"
    outside.write_text("")
    link = ok.with_suffix(".lock")
    link.symlink_to(outside)
    with pytest.raises(t.LabelTransportError):
        t._require_local_file(link)
    with pytest.raises(t.LabelTransportError):
        t._require_local_file(state_dir / "x\0y")
    with pytest.raises(t.LabelTransportError):
        t._require_local_file(state_dir / ".." / "outside.json")
    with pytest.raises(t.LabelTransportError):
        t._retain_sequence(ok, "s", 1)


def test_state_files_are_opened_without_following_a_swapped_in_link(
    tmp_path: Path,
) -> None:
    """A link planted after the containment check is refused by the open itself."""
    from traigent.labels import transport as t

    outside = tmp_path / "victim"
    outside.write_text("keep", encoding="utf-8")
    link = tmp_path / "labels_state.lock"
    link.symlink_to(outside)
    for flags in (os.O_RDONLY, os.O_RDWR | os.O_CREAT):
        with pytest.raises(t.LabelTransportError):
            t._open_state_fd(link, flags)
    assert outside.read_text(encoding="utf-8") == "keep"
    with pytest.raises(t.LabelTransportError):
        t._open_state_fd(tmp_path, os.O_RDONLY)  # a directory is not a regular file


@_needs_fcntl
def test_the_retained_state_temp_must_still_be_the_file_that_was_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.labels import transport as t

    monkeypatch.setattr(t, "_state_dir", lambda: tmp_path)
    state = tmp_path / "labels_state.json"
    created: list[Path] = []
    real_open = t._open_state_fd

    def spying(path, flags):
        fd = real_open(path, flags)
        if flags & os.O_EXCL:
            created.append(Path(path))
        return fd

    monkeypatch.setattr(t, "_open_state_fd", spying)
    monkeypatch.setattr(t, "same_entry", lambda fd, path: False)
    with pytest.raises(t.LabelTransportError):
        t._retain_sequence(state, "s", 1)
    assert len(created) == 1  # the temp-creation path really ran
    assert not state.exists()
    # The mismatch path never unlinks the temp name (it is not ours once replaced);
    # the replaced-entry test below asserts that directly.


@_needs_fcntl
@pytest.mark.skipif(
    os.name == "nt",
    reason="os.replace over an open temp file is not allowed on Windows",
)
def test_a_replaced_temp_entry_is_not_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another file replaces the temp name before the identity check: the entry is not ours."""
    from traigent.labels import transport as t

    monkeypatch.setattr(t, "_state_dir", lambda: tmp_path)
    state = tmp_path / "labels_state.json"
    real_same_entry = t.same_entry
    seen: list[Path] = []

    def replacing(fd, path):
        other = tmp_path / "replacement"
        other.write_text("not ours", encoding="utf-8")
        os.replace(other, path)
        seen.append(Path(path))
        return real_same_entry(fd, path)

    monkeypatch.setattr(t, "same_entry", replacing)
    with pytest.raises(t.LabelTransportError):
        t._retain_sequence(state, "s", 1)
    assert len(seen) == 1  # the replacement really happened before the check
    assert seen[0].read_text(encoding="utf-8") == "not ours"  # still there, untouched
    assert not state.exists()


@_needs_fcntl
def test_a_stale_pid_named_temp_does_not_block_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.labels import transport as t

    monkeypatch.setattr(t, "_state_dir", lambda: tmp_path)
    state = tmp_path / "labels_state.json"
    stale = tmp_path / f"labels_state.tmp{os.getpid()}"
    stale.write_text("stale", encoding="utf-8")
    t._retain_sequence(state, "s", 1)
    assert json.loads(state.read_text(encoding="utf-8")) == {"s": 1}
    assert stale.read_text(encoding="utf-8") == "stale"  # never ours to delete


@_needs_fcntl
def test_a_temp_name_collision_retries_with_a_new_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.labels import transport as t

    monkeypatch.setattr(t, "_state_dir", lambda: tmp_path)
    state = tmp_path / "labels_state.json"
    names = iter(["aa" * 8, "bb" * 8])
    monkeypatch.setattr(t.secrets, "token_hex", lambda n: next(names))
    taken = tmp_path / f"labels_state.json.{'aa' * 8}.tmp"
    taken.write_text("not ours", encoding="utf-8")
    t._retain_sequence(state, "s", 1)
    assert json.loads(state.read_text(encoding="utf-8")) == {"s": 1}
    assert taken.read_text(encoding="utf-8") == "not ours"  # the collider is untouched
    assert not (tmp_path / f"labels_state.json.{'bb' * 8}.tmp").exists()


@_needs_fcntl
def test_persistent_temp_collisions_fail_closed_and_leave_them_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.labels import transport as t

    monkeypatch.setattr(t, "_state_dir", lambda: tmp_path)
    state = tmp_path / "labels_state.json"
    taken = tmp_path / f"labels_state.json.{'cc' * 8}.tmp"
    taken.write_text("not ours", encoding="utf-8")
    monkeypatch.setattr(t.secrets, "token_hex", lambda n: "cc" * 8)
    with pytest.raises(t.LabelTransportError):
        t._retain_sequence(state, "s", 1)
    assert taken.read_text(encoding="utf-8") == "not ours"
    assert not state.exists()
