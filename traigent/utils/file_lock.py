"""Exclusive inter-process file lock for read-modify-write of a shared file."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from traigent.utils.logging import get_logger

try:  # POSIX only; elsewhere the in-process write stays atomic but unlocked
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]

logger = get_logger(__name__)

_lock_warning_emitted = False


@contextmanager
def storage_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock for read-modify-write of ``path``.

    The lock lives on a sibling ``<name>.lock`` file. Each call opens its own
    descriptor, so it excludes other threads of this process as well as other
    processes.
    """
    global _lock_warning_emitted
    if fcntl is None and msvcrt is None:
        if not _lock_warning_emitted:
            _lock_warning_emitted = True
            logger.warning(
                "No file-locking support on this platform; concurrent "
                "writers may lose records"
            )
        yield
        return
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - exercised via monkeypatched test
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


class FileLockTimeoutError(TimeoutError):
    """The lock could not be acquired within the allowed wait."""


class FileLockUnsupportedError(RuntimeError):
    """This platform offers no file-locking primitive."""


def _try_lock_nonblocking(lock_file: Any) -> bool:
    """Take the exclusive lock without waiting; ``False`` when it is held."""
    if fcntl is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    lock_file.seek(0)  # pragma: no cover - Windows
    try:  # pragma: no cover
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:  # pragma: no cover
        return False
    return True  # pragma: no cover


def _lock_directory() -> Path:
    """The per-user directory (mode 0700) under the system temp dir for locks."""
    getuid = getattr(os, "getuid", None)
    uid = getuid() if getuid is not None else None
    directory = (
        Path(tempfile.gettempdir())
        / f"traigent-locks-{uid if uid is not None else 'user'}"
    )
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise FileLockUnsupportedError(
            f"Lock directory {directory} is not a plain directory; refusing to write "
            "without a trustworthy lock"
        )
    if uid is not None:
        if info.st_uid != uid:
            raise FileLockUnsupportedError(
                f"Lock directory {directory} belongs to another user; refusing to "
                "write without a trustworthy lock"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            directory.chmod(0o700)
    return directory


def lock_path_for(path: Path) -> Path:
    """Where :func:`exclusive_lock` keeps the lock for ``path``.

    A file in the system temp directory named after a digest of the data
    file's real absolute path, so nothing is created beside the data file
    (where it could be committed by accident) and two different data files
    never share a lock, however they are spelled.
    """
    key = hashlib.sha256(os.path.realpath(path).encode("utf-8")).hexdigest()[:32]
    return _lock_directory() / f"{key}.lock"


@contextmanager
def exclusive_lock(
    path: Path, *, timeout: float = 30.0, poll_interval: float = 0.05
) -> Iterator[None]:
    """Hold an exclusive cross-process lock for read-modify-write of ``path``.

    Unlike :func:`storage_lock` the wait is bounded and the lock is never
    skipped: if it cannot be taken within ``timeout`` seconds, or the platform
    has no locking primitive, an error is raised and the body does not run.
    The lock lives in the system temp directory (see :func:`lock_path_for`),
    never on ``path`` itself (``path`` may be replaced by rename while the lock
    is held) and never beside it.

    Limits: processes exclude each other only when they share the same temp
    directory and the same OS user. Containers with separate ``/tmp`` that share
    one mounted repository do NOT exclude each other (point ``TMPDIR`` at a
    shared mount to make them), and neither do two OS users, nor hosts sharing
    a network filesystem. ``flock`` semantics on network filesystems vary.

    Raises:
        FileLockTimeoutError: The lock stayed held by another process.
        FileLockUnsupportedError: No locking support on this platform, or no
            trustworthy lock directory.
    """
    if fcntl is None and msvcrt is None:
        raise FileLockUnsupportedError(
            "No file-locking support on this platform; refusing to write without a lock"
        )
    lock_path = lock_path_for(path)
    deadline = time.monotonic() + timeout
    with lock_path.open("a+") as lock_file:
        while not _try_lock_nonblocking(lock_file):
            if time.monotonic() >= deadline:
                raise FileLockTimeoutError(
                    f"Timed out after {timeout:g}s waiting for the lock on "
                    f"{lock_path}; another process is writing {path.name}. "
                    "Nothing was written. Check whether another command is "
                    "still running, then retry. Do not delete the lock file: "
                    "removing a file another process holds locked lets a "
                    "second writer in."
                )
            time.sleep(poll_interval)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
