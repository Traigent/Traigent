"""Characterization of the no-follow scaffold writer, pinned before extraction.

Covers both the descriptor-relative branch and the lstat fallback branch: outcomes,
exact messages, the exact ``os.open`` / ``os.close`` call traces, the order of the
symlink checks and the exception boundaries.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from traigent.mcp import tools

_real_open = os.open
_real_close = os.close
_real_is_symlink = Path.is_symlink

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

needs_posix = pytest.mark.skipif(
    not (_NOFOLLOW and _DIRECTORY and _real_open in os.supports_dir_fd),
    reason="descriptor-relative opens unavailable",
)


class _Trace:
    """Records os.open / os.close calls and delegates to the real functions."""

    def __init__(self) -> None:
        self.opens: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.closes: list[int] = []
        self.fds: list[int] = []

    def open(self, *args: Any, **kwargs: Any) -> int:
        self.opens.append((args, dict(kwargs)))
        fd = _real_open(*args, **kwargs)
        self.fds.append(fd)
        return fd

    def close(self, fd: int) -> None:
        self.closes.append(fd)
        _real_close(fd)


def _install(monkeypatch: pytest.MonkeyPatch, mode: str) -> tuple[_Trace, list[str]]:
    trace = _Trace()
    checked: list[str] = []

    def recording_is_symlink(self: Path) -> bool:
        checked.append(str(self))
        return _real_is_symlink(self)

    monkeypatch.setattr(os, "open", trace.open)
    monkeypatch.setattr(os, "close", trace.close)
    monkeypatch.setattr(Path, "is_symlink", recording_is_symlink)
    if mode == "dirfd":
        monkeypatch.setattr(os, "supports_dir_fd", {trace.open})
    else:
        monkeypatch.setattr(os, "supports_dir_fd", set())
    return trace, checked


@pytest.fixture(params=["dirfd", "fallback"])
def mode(request: pytest.FixtureRequest) -> str:
    if request.param == "dirfd" and not (_NOFOLLOW and _DIRECTORY):
        pytest.skip("descriptor-relative opens unavailable")
    return str(request.param)


def _expected_flags(overwrite: bool) -> int:
    return (
        os.O_WRONLY | os.O_CREAT | _NOFOLLOW | (os.O_TRUNC if overwrite else os.O_EXCL)
    )


def test_writes_new_file_in_nested_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    (root / "a" / "b").mkdir(parents=True)
    _install(monkeypatch, mode)
    tools._write_text_nofollow(
        root / "a" / "b" / "f.txt", "héllo\n", overwrite=False, root=root
    )
    assert (root / "a" / "b" / "f.txt").read_bytes() == "héllo\n".encode()


def test_overwrite_truncates_and_exclusive_refuses_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    target = root / "f.txt"
    target.write_text("long original content", encoding="utf-8")
    _install(monkeypatch, mode)
    with pytest.raises(FileExistsError):
        tools._write_text_nofollow(target, "new", overwrite=False, root=root)
    assert target.read_text(encoding="utf-8") == "long original content"
    tools._write_text_nofollow(target, "new", overwrite=True, root=root)
    assert target.read_text(encoding="utf-8") == "new"


def test_file_permissions_are_0644_before_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    _install(monkeypatch, mode)
    previous = os.umask(0o022)
    try:
        tools._write_text_nofollow(root / "f.txt", "x", overwrite=True, root=root)
    finally:
        os.umask(previous)
    assert stat.S_IMODE((root / "f.txt").stat().st_mode) == 0o644


def test_default_root_is_resolved_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    monkeypatch.chdir(root)
    _install(monkeypatch, mode)
    tools._write_text_nofollow(root / "g.txt", "cwd", overwrite=True)
    assert (root / "g.txt").read_text(encoding="utf-8") == "cwd"


def test_root_argument_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    (root / "sub").mkdir()
    _install(monkeypatch, mode)
    tools._write_text_nofollow(
        root / "f.txt", "x", overwrite=True, root=root / "sub" / ".."
    )
    assert (root / "f.txt").read_text(encoding="utf-8") == "x"


def test_path_outside_root_is_refused_with_value_error_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve() / "root"
    root.mkdir()
    outside = root.parent / "elsewhere.txt"
    trace, _ = _install(monkeypatch, mode)
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(outside, "x", overwrite=True, root=root)
    assert str(info.value) == f"path escapes the workspace root: {outside}"
    assert isinstance(info.value.__cause__, ValueError)
    assert trace.opens == []


def test_workspace_root_itself_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    trace, _ = _install(monkeypatch, mode)
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(root, "x", overwrite=True, root=root)
    assert str(info.value) == "refusing to write to the workspace root itself"
    assert trace.opens == []


def test_missing_directory_propagates_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    _install(monkeypatch, mode)
    with pytest.raises(FileNotFoundError):
        tools._write_text_nofollow(
            root / "nope" / "f.txt", "x", overwrite=True, root=root
        )
    assert not (root / "nope").exists()


def test_symlinked_final_component_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    victim = root / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    link = root / "link.txt"
    link.symlink_to(victim)
    _install(monkeypatch, mode)
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(link, "clobber", overwrite=True, root=root)
    if mode == "fallback":
        assert str(info.value) == f"refusing to write through symlink: {link}"
    assert victim.read_text(encoding="utf-8") == "keep"


def test_symlinked_intermediate_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    (root / "link").symlink_to(real)
    _install(monkeypatch, mode)
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(
            root / "link" / "f.txt", "x", overwrite=True, root=root
        )
    if mode == "fallback":
        assert str(info.value) == f"refusing to write through symlink: {root / 'link'}"
    assert not (real / "f.txt").exists()


def test_fallback_checks_every_component_in_order_then_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "a" / "b").mkdir(parents=True)
    trace, checked = _install(monkeypatch, "fallback")
    target = root / "a" / "b" / "f.txt"
    tools._write_text_nofollow(target, "x", overwrite=True, root=root)
    assert checked == [str(root / "a"), str(root / "a" / "b"), str(target)]
    assert trace.opens == [((target, _expected_flags(True), 0o644), {})]
    assert trace.closes == []


def test_fallback_first_symlink_stops_later_checks_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    real = root / "real"
    (real / "b").mkdir(parents=True)
    (root / "a").symlink_to(real)
    trace, checked = _install(monkeypatch, "fallback")
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(
            root / "a" / "b" / "f.txt", "x", overwrite=True, root=root
        )
    assert str(info.value) == f"refusing to write through symlink: {root / 'a'}"
    assert checked == [str(root / "a")]
    assert trace.opens == []


def test_fallback_intermediate_symlink_wins_over_final_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    (real / "f.txt").symlink_to(real / "elsewhere")
    (root / "a").symlink_to(real)
    trace, checked = _install(monkeypatch, "fallback")
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(root / "a" / "f.txt", "x", overwrite=True, root=root)
    assert str(info.value) == f"refusing to write through symlink: {root / 'a'}"
    assert trace.opens == []


def test_fallback_final_symlink_refused_before_any_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "link.txt").symlink_to(root / "missing.txt")
    trace, checked = _install(monkeypatch, "fallback")
    with pytest.raises(OSError) as info:
        tools._write_text_nofollow(root / "link.txt", "x", overwrite=True, root=root)
    assert str(info.value) == f"refusing to write through symlink: {root / 'link.txt'}"
    assert checked == [str(root / "link.txt")]
    assert trace.opens == []
    assert not (root / "missing.txt").exists()


def test_fallback_without_o_nofollow_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    trace, _ = _install(monkeypatch, "fallback")
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    tools._write_text_nofollow(root / "f.txt", "x", overwrite=False, root=root)
    expected = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    assert trace.opens == [((root / "f.txt", expected, 0o644), {})]


def test_fallback_when_only_directory_flag_is_missing_uses_lstat_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    trace, checked = _install(monkeypatch, "dirfd")
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    tools._write_text_nofollow(root / "f.txt", "x", overwrite=True, root=root)
    assert checked == [str(root / "f.txt")]
    assert trace.opens == [((root / "f.txt", _expected_flags(True), 0o644), {})]


def test_fallback_when_open_is_not_dir_fd_capable_uses_lstat_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    trace, checked = _install(monkeypatch, "fallback")
    tools._write_text_nofollow(root / "f.txt", "x", overwrite=True, root=root)
    assert checked == [str(root / "f.txt")]
    assert [call[1] for call in trace.opens] == [{}]


@needs_posix
def test_dirfd_branch_opens_one_component_at_a_time_and_closes_each_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "a" / "b").mkdir(parents=True)
    trace, checked = _install(monkeypatch, "dirfd")
    tools._write_text_nofollow(
        root / "a" / "b" / "f.txt", "x", overwrite=False, root=root
    )
    dir_flags = os.O_RDONLY | _DIRECTORY | _NOFOLLOW
    root_fd, a_fd, b_fd, _file_fd = trace.fds
    assert trace.opens == [
        ((root, os.O_RDONLY | _DIRECTORY), {}),
        (("a", dir_flags), {"dir_fd": root_fd}),
        (("b", dir_flags), {"dir_fd": a_fd}),
        (("f.txt", _expected_flags(False), 0o644), {"dir_fd": b_fd}),
    ]
    # each directory fd is closed when replaced; the last one in the finally block;
    # the file fd is closed by fdopen (not via os.close of this module)
    assert trace.closes == [root_fd, a_fd, b_fd]
    assert checked == []
    assert len(trace.closes) == 3


@needs_posix
def test_dirfd_branch_closes_directory_fd_when_component_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    (root / "a").symlink_to(real)
    trace, _ = _install(monkeypatch, "dirfd")
    with pytest.raises(OSError):
        tools._write_text_nofollow(root / "a" / "f.txt", "x", overwrite=True, root=root)
    assert len(trace.opens) == 2
    assert trace.closes == [trace.fds[0]]


@needs_posix
def test_dirfd_branch_closes_directory_fd_when_file_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "f.txt").write_text("keep", encoding="utf-8")
    trace, _ = _install(monkeypatch, "dirfd")
    with pytest.raises(FileExistsError):
        tools._write_text_nofollow(root / "f.txt", "x", overwrite=False, root=root)
    assert trace.closes == [trace.fds[0]]
    assert (root / "f.txt").read_text(encoding="utf-8") == "keep"


@needs_posix
def test_dirfd_branch_does_not_call_is_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    _, checked = _install(monkeypatch, "dirfd")
    tools._write_text_nofollow(root / "f.txt", "x", overwrite=True, root=root)
    assert checked == []


def test_write_error_after_open_propagates_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path.resolve()
    _install(monkeypatch, mode)
    real_fdopen: Callable[..., Any] = os.fdopen
    seen: list[int] = []

    def failing_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        seen.append(fd)
        handle = real_fdopen(fd, *args, **kwargs)

        class _Boom:
            def __enter__(self) -> _Boom:
                return self

            def __exit__(self, *exc: object) -> None:
                handle.close()

            def write(self, text: str) -> None:
                raise RuntimeError("disk full")

        return _Boom()

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    with pytest.raises(RuntimeError, match="disk full"):
        tools._write_text_nofollow(root / "f.txt", "x", overwrite=True, root=root)
    assert len(seen) == 1
    assert (root / "f.txt").read_text(encoding="utf-8") == ""
