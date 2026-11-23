"""The relay log path is operator-chosen; it must be a plain file, not a redirect."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traigent.egress.log import RelayLog, RelayLogError, iter_lines, operator_log_path


def test_a_plain_path_opens_and_resolves(tmp_path: Path) -> None:
    log = RelayLog(tmp_path / "sub" / "relay.jsonl")
    try:
        assert log.path == (tmp_path / "sub" / "relay.jsonl").resolve()
    finally:
        log.close()


def test_a_dotdot_path_is_normalised_to_the_real_file(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    assert (
        operator_log_path(tmp_path / "a" / ".." / "x.jsonl")
        == (tmp_path / "x.jsonl").resolve()
    )


def test_a_nul_byte_is_refused() -> None:
    with pytest.raises(RelayLogError) as exc:
        operator_log_path("canary\0.jsonl")
    assert "canary" not in str(exc.value)


def test_a_symlinked_log_is_refused_for_write_and_read(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    link = tmp_path / "relay.jsonl"
    os.symlink(target, link)
    with pytest.raises(RelayLogError):
        RelayLog(link)
    assert list(iter_lines(link)) == []


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RelayLogError):
        RelayLog(tmp_path)


def test_a_link_swapped_in_after_the_path_check_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open refuses a symlink itself (O_NOFOLLOW), not only the earlier check."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    victim.chmod(0o644)
    target = tmp_path / "relay.jsonl"
    checked = operator_log_path(target)  # passes: nothing there yet
    os.symlink(victim, target)  # replaced between the check and the open
    monkeypatch.setattr("traigent.egress.log.operator_log_path", lambda p: checked)
    with pytest.raises(RelayLogError):
        RelayLog(target)
    assert victim.read_text() == "keep"
    assert (victim.stat().st_mode & 0o777) == 0o644  # fchmod never reached the target


def test_a_file_swapped_between_the_check_and_the_open_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even without O_NOFOLLOW, the opened file must be the one the path names now."""
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    victim.chmod(0o644)
    target = tmp_path / "relay.jsonl"
    checked = operator_log_path(target)  # passes: nothing there yet
    real_open = os.open

    def swapping_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)  # creates the log file
        os.unlink(target)
        os.symlink(victim, target)  # swapped right after the open
        return fd

    monkeypatch.setattr("traigent.egress.log.operator_log_path", lambda p: checked)
    monkeypatch.setattr("traigent._fsutil.os.open", swapping_open)
    with pytest.raises(RelayLogError):
        RelayLog(target)
    assert victim.read_text() == "keep"
    assert (victim.stat().st_mode & 0o777) == 0o644
