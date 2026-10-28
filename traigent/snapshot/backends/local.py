"""Local-directory snapshot store (atomic, no-clobber)."""

from __future__ import annotations

import errno
import os
import uuid
from pathlib import Path

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.store import CreateResult, validate_key

__all__ = ["LocalDirStore"]

_BACKEND = "local"
_UNSUPPORTED_LINK_ERRNOS = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV}
)


def _error(code: SnapshotErrorCode) -> SnapshotStoreError:
    return SnapshotStoreError(code, backend=_BACKEND)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LocalDirStore:
    """Stores objects as files under ``root``.

    Each object is written to ``<dir>/.tmp-<uuid>``, fsynced, then hard-linked
    to its final name (``os.link`` never replaces an existing file), the temp
    name is unlinked and the directory fsynced. The final name therefore only
    ever holds complete bytes, and a concurrent writer cannot clobber it.
    """

    max_object_bytes: int | None = None

    def __init__(self, root: str | os.PathLike[str]) -> None:
        try:
            Path(root).mkdir(parents=True, exist_ok=True, mode=0o700)
            self._root = Path(root).resolve()
        except OSError:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None

    def __repr__(self) -> str:
        return self.locator()

    def locator(self) -> str:
        return f"file://{self._root}"

    def _path(self, key: str) -> Path:
        validate_key(key)
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION)
        return path

    def read(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except PermissionError:
            raise _error(SnapshotErrorCode.AUTH) from None
        except OSError:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        dest = self._path(key)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            return self._create(dest, bytes(data))
        except SnapshotStoreError:
            raise
        except PermissionError:
            raise _error(SnapshotErrorCode.AUTH) from None
        except OSError:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None

    def _create(self, dest: Path, data: bytes) -> CreateResult:
        tmp = dest.parent / f".tmp-{uuid.uuid4().hex}"
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, dest)
            except FileExistsError:
                if dest.read_bytes() == data:
                    return CreateResult.EXISTED_IDENTICAL
                raise _error(SnapshotErrorCode.CONFLICT) from None
            except OSError as error:
                if error.errno in _UNSUPPORTED_LINK_ERRNOS:
                    raise _error(SnapshotErrorCode.UNSUPPORTED_BACKEND) from None
                raise
            return CreateResult.CREATED
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            _fsync_dir(dest.parent)
