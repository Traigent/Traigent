"""``traigent._fsutil.open_verified``: the descriptor must be the file the path names."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traigent import _fsutil


def _symlink_or_skip(link: Path, target: Path) -> None:
    """Create ``link`` -> ``target``, or skip where the platform cannot make symlinks."""
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("this platform or account cannot create symlinks")


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
    _symlink_or_skip(link, target)
    with pytest.raises(OSError):
        _fsutil.open_verified(link, os.O_RDONLY)
    assert target.read_text() == "keep"


@pytest.mark.skipif(
    os.name == "nt", reason="os.replace over an open file is not allowed on Windows"
)
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
    with pytest.raises(_fsutil.UnverifiedFileError) as err:
        _fsutil.open_verified(f, os.O_RDWR)
    assert str(tmp_path) not in str(err.value)


@pytest.mark.skipif(
    os.name == "nt", reason="unlinking or replacing an open file is unsupported"
)
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
    with pytest.raises(_fsutil.UnverifiedFileError):
        _fsutil.open_verified(f, os.O_RDONLY)


def test_no_descriptor_leaks_on_refusal(tmp_path: Path) -> None:
    before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0
    for _ in range(20):
        with pytest.raises(OSError):
            _fsutil.open_verified(tmp_path, os.O_RDONLY)
    if before:
        assert len(os.listdir("/proc/self/fd")) <= before + 1


def _swap_once(monkeypatch: pytest.MonkeyPatch, swap) -> list[int]:
    real_open = os.open
    calls: list[int] = []

    def opening(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        calls.append(fd)
        if len(calls) == 1:
            swap()
        return fd

    monkeypatch.setattr(_fsutil.os, "open", opening)
    return calls


@pytest.mark.skipif(
    os.name == "nt", reason="os.replace over an open file is not allowed on Windows"
)
def test_a_reader_tolerates_one_atomic_publication_between_open_and_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "published"
    f.write_text("old")

    def publish() -> None:
        new = tmp_path / "new"
        new.write_text("new")
        os.replace(new, f)

    calls = _swap_once(monkeypatch, publish)
    fd = _fsutil.open_verified(f, os.O_RDONLY)
    try:
        assert os.read(fd, 10) == b"new"
    finally:
        os.close(fd)
    assert len(calls) == 2


@pytest.mark.skipif(
    os.name == "nt", reason="os.replace over an open file is not allowed on Windows"
)
def test_a_writer_does_not_tolerate_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "published"
    f.write_text("old")

    def publish() -> None:
        new = tmp_path / "new"
        new.write_text("new")
        os.replace(new, f)

    calls = _swap_once(monkeypatch, publish)
    with pytest.raises(OSError):
        _fsutil.open_verified(f, os.O_RDWR)
    assert len(calls) == 1


@pytest.mark.skipif(
    os.name == "nt", reason="os.replace over an open file is not allowed on Windows"
)
def test_a_reader_gives_up_after_bounded_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "published"
    f.write_text("old")
    real_open = os.open
    count = 0

    def opening(path, flags, mode=0o777):
        nonlocal count
        fd = real_open(path, flags, mode)
        count += 1
        new = tmp_path / "new"
        new.write_text("new")
        os.replace(new, f)
        return fd

    monkeypatch.setattr(_fsutil.os, "open", opening)
    with pytest.raises(OSError):
        _fsutil.open_verified(f, os.O_RDONLY)
    assert count == 3


@pytest.mark.skipif(
    os.name == "nt", reason="unlinking or replacing an open file is unsupported"
)
def test_a_symlink_swap_is_not_retried_for_a_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "published"
    f.write_text("old")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("x")

    def swap() -> None:
        f.unlink()
        f.symlink_to(elsewhere)

    calls = _swap_once(monkeypatch, swap)
    with pytest.raises(_fsutil.UnverifiedFileError):
        _fsutil.open_verified(f, os.O_RDONLY)
    assert len(calls) == 1  # a symlink is refused at once, never retried


@pytest.mark.skipif(
    os.name == "nt", reason="os.replace over an open file is not allowed on Windows"
)
def test_missing_after_an_observed_replacement_is_a_verification_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "published"
    f.write_text("old")
    real_open = os.open
    count = 0

    def opening(path, flags, mode=0o777):
        nonlocal count
        count += 1
        if count == 2:
            raise FileNotFoundError(2, "gone")
        fd = real_open(path, flags, mode)
        new = tmp_path / "new"
        new.write_text("new")
        os.replace(new, f)
        return fd

    monkeypatch.setattr(_fsutil.os, "open", opening)
    with pytest.raises(_fsutil.UnverifiedFileError) as err:
        _fsutil.open_verified(f, os.O_RDONLY)
    assert not isinstance(err.value, FileNotFoundError)
    assert str(tmp_path) not in str(err.value)


def test_missing_on_the_first_attempt_keeps_missing_file_semantics(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _fsutil.open_verified(tmp_path / "absent", os.O_RDONLY)


def test_replacement_then_missing_is_an_error_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected filesystem results: no real rename over an open handle is needed."""
    f = tmp_path / "published"
    f.write_text("old")
    other = tmp_path / "other"
    other.write_text("new")
    other_stat = os.lstat(other)
    real_open = os.open
    real_lstat = os.lstat
    opens = 0

    def opening(path, flags, mode=0o777):
        nonlocal opens
        opens += 1
        if opens == 2:
            raise FileNotFoundError(2, "gone")
        return real_open(path, flags, mode)

    def lstat(path, *a, **k):
        # The path now names a different regular file than the opened descriptor.
        return other_stat if os.fspath(path) == os.fspath(f) else real_lstat(path)

    monkeypatch.setattr(_fsutil.os, "open", opening)
    monkeypatch.setattr(_fsutil.os, "lstat", lstat)
    with pytest.raises(_fsutil.UnverifiedFileError) as err:
        _fsutil.open_verified(f, os.O_RDONLY)
    assert not isinstance(err.value, FileNotFoundError)
    assert str(tmp_path) not in str(err.value)
    assert opens == 2
