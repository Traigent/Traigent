"""Private helper: open a local file and prove the descriptor is the file at that path.

``O_NOFOLLOW`` is used where the platform has it, but it is never the only defence:
after the open, the descriptor must be a regular file and must be the very file the
path names now (``samestat`` of ``fstat(fd)`` and ``lstat(path)``), and that entry must
not be a symlink.  The second check is portable (it also holds where ``O_NOFOLLOW``
does not exist).  Not part of the public API.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class UnverifiedFileError(OSError):
    """The opened descriptor is not the regular file the path names (no path in message)."""


_BINARY = getattr(os, "O_BINARY", 0)  # Windows: never translate line endings
_READ_ATTEMPTS = 3
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def open_verified(path: str | os.PathLike[str], flags: int, mode: int = 0o600) -> int:
    """Open ``path`` as given and return a descriptor verified against the path.

    ``FileNotFoundError`` from the open itself is passed through unchanged; every other
    failure, including a verification mismatch, is an ``OSError``.  On any failure no
    descriptor is leaked.

    A read-only open tolerates a concurrent atomic replacement: when the path now names
    a different *regular file* (a writer published with ``os.replace`` between the open
    and the check), the open is retried, each attempt fully verified, up to three times.
    A symlink or any non-regular entry is never tolerated, and neither is any mismatch on
    a write, create, append or lock open.
    """
    p = Path(path)
    read_only = not flags & _WRITE_FLAGS
    attempts = _READ_ATTEMPTS if read_only else 1
    replaced = False
    for attempt in range(attempts):
        try:
            return _open_once(p, flags, mode)
        except FileNotFoundError:
            if not replaced:
                raise  # first attempt: genuine missing-file semantics
            # A replacement was already observed, so the file existed: a later
            # absence is a verification failure, not a "missing store".
            raise UnverifiedFileError("path no longer names the opened file") from None
        except _ReplacedError:
            replaced = True
            if attempt == attempts - 1:
                raise UnverifiedFileError(
                    "path no longer names the opened file"
                ) from None
    raise UnverifiedFileError(
        "path no longer names the opened file"
    )  # pragma: no cover


class _ReplacedError(UnverifiedFileError):
    """The path now names another regular file (an atomic replacement)."""


def _open_once(p: Path, flags: int, mode: int) -> int:
    # Without O_NOFOLLOW (no such flag on this platform) the open precedes the entry
    # check, so a symlink planted between the caller's pre-check and the open can be
    # followed, and O_CREAT may create its target, before the mismatch is refused.
    # Trusted-directory model: O_NOFOLLOW protects the final path component where
    # available; elsewhere an open can follow a link before verification rejects
    # it, so the store directory must be writable only by the user.  (Windows
    # Developer Mode lets unprivileged users create symlinks.)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(p, flags | nofollow | _CLOEXEC | _BINARY, mode)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise UnverifiedFileError("not a regular file")
        try:
            named = os.lstat(p)
        except OSError:
            raise UnverifiedFileError("path no longer names the opened file") from None
        if stat.S_ISLNK(named.st_mode):
            raise UnverifiedFileError("path no longer names the opened file")
        if not os.path.samestat(opened, named):
            if stat.S_ISREG(named.st_mode):
                raise _ReplacedError("path now names another file")
            raise UnverifiedFileError("path no longer names the opened file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def same_entry(fd: int, path: str | os.PathLike[str]) -> bool:
    """True when ``path`` is a non-symlink entry that is the file open as ``fd``."""
    try:
        named = os.lstat(path)
        return not stat.S_ISLNK(named.st_mode) and os.path.samestat(os.fstat(fd), named)
    except OSError:
        return False
