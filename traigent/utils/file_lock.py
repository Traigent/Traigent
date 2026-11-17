"""Exclusive inter-process file lock for read-modify-write of a shared file."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
