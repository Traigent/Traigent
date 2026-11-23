"""``traigent._fsutil.open_verified``: the descriptor must be the file the path names."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traigent import _fsutil


def _swap_after_open(monkeypatch: pytest.MonkeyPatch, swap) -> None:
    real_open = os.open

    def opening(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        swap()
        return fd

    monkeypatch.setattr(_fsutil.os, "open", opening)


def test_a_plain_file_opens(tmp_path: Path) -> None:
    f = tmp_path / "a"
    f.write_text("x")
    fd = _fsutil.open_verified(f, os.O_RDONLY)
    os.close(fd)


def test_a_missing_file_stays_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _fsutil.open_verified(tmp_path / "nope", os.O_RDONLY)


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        _fsutil.open_verified(tmp_path, os.O_RDONLY)


@pytest.mark.parametrize("nofollow", [True, False])
def test_a_symlink_is_refused_with_or_without_o_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nofollow: bool
) -> None:
    if not nofollow:
        monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    target = tmp_path / "target"
    target.write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        _fsutil.open_verified(link, os.O_RDONLY)
    assert target.read_text() == "keep"


def test_a_swap_between_the_open_and_the_verification_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path is replaced by another file after the open: same_entry must say no."""
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    f = tmp_path / "log"
    f.write_text("original")

    def swap() -> None:
        other = tmp_path / "other"
        other.write_text("other")
        os.replace(other, f)

    _swap_after_open(monkeypatch, swap)
    with pytest.raises(OSError) as err:
        _fsutil.open_verified(f, os.O_RDWR)
    assert str(tmp_path) not in str(err.value)


def test_a_symlink_swapped_in_after_the_open_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "log"
    f.write_text("original")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("x")

    def swap() -> None:
        f.unlink()
        f.symlink_to(elsewhere)

    _swap_after_open(monkeypatch, swap)
    with pytest.raises(OSError):
        _fsutil.open_verified(f, os.O_RDONLY)


def test_no_descriptor_leaks_on_refusal(tmp_path: Path) -> None:
    before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0
    for _ in range(20):
        with pytest.raises(OSError):
            _fsutil.open_verified(tmp_path, os.O_RDONLY)
    if before:
        assert len(os.listdir("/proc/self/fd")) <= before + 1
