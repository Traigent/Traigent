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


def open_verified(path: str | os.PathLike[str], flags: int, mode: int = 0o600) -> int:
    """Open ``path`` as given and return a descriptor verified against the path.

    ``FileNotFoundError`` from the open itself is passed through unchanged; every other
    failure, including a verification mismatch, is an ``OSError``.  On any failure no
    descriptor is leaked.
    """
    p = Path(path)
    # Without O_NOFOLLOW (no such flag on this platform) the open precedes the entry
    # check, so a symlink planted between the caller's pre-check and the open can be
    # followed, and O_CREAT may create its target, before the mismatch is refused.
    # Accepted: those platforms need elevated rights to create symlinks, and an
    # attacker would already need write access to the user's own directory.
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(p, flags | nofollow | _CLOEXEC, mode)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise UnverifiedFileError("not a regular file")
        try:
            named = os.lstat(p)
        except OSError:
            raise UnverifiedFileError("path no longer names the opened file") from None
        if stat.S_ISLNK(named.st_mode) or not os.path.samestat(opened, named):
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
