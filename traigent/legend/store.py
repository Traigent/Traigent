"""``ClientStore``: the customer-side, no-delete byte store a capture is written to.

Keys are scheme-qualified recovery-manifest fingerprints (``rmf0:sha256:...``),
never pillar fingerprints. There is no delete: a store may only add. A ``put``
of an existing key with different bytes is a closed error, never an overwrite.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, Final, Literal, Protocol, runtime_checkable

from traigent.legend import _contract as c
from traigent.legend.errors import LegendError

PutStatus = Literal[
    "stored_unverified", "store_key_conflict", "unreachable", "permission_denied"
]
# ``retrieved`` is local only; the others are ClientStoreStatus values.
GetStatus = Literal[
    "retrieved", "missing", "truncated", "corrupted", "unreachable", "permission_denied"
]
HasStatus = Literal["available", "missing", "unknown"]

BUNDLE_NAME: Final = "bundle.rmf"
#: Layout marker at the root of a :class:`LocalDirectoryStore`. A root holding anything else
#: (e.g. the earlier ``.blob``/``.meta`` pair layout) is refused, never read as "missing".
STORE_FORMAT_FILE: Final = ".legend-store-format"
STORE_FORMAT: Final = "traigent.legend.store/1\n"
LAYOUT_UNSUPPORTED: Final = "local_store_layout_unsupported"
# Manifest fingerprints (rmf0) and the reserved synthetic conformance keys (rmfc0) are
# disjoint namespaces; each gets its own directory so they can never collide.
_KEY_RE: Final = re.compile(
    "(?:"
    + re.escape(c.MANIFEST_FINGERPRINT_PREFIX)
    + "|"
    + re.escape(c.CONFORMANCE_KEY_PREFIX)
    + r")[0-9a-f]{64}"
)


@runtime_checkable
class ClientStore(Protocol):
    """Implemented by the customer's coding agent for their own storage. No delete."""

    def put(
        self, kind: str, manifest_fingerprint: str, stream: BinaryIO
    ) -> PutStatus: ...

    def get(self, kind: str, manifest_fingerprint: str, dest_dir: Path) -> GetStatus:
        """Write the stored bytes to ``dest_dir / BUNDLE_NAME`` and say how it went."""
        ...

    def has(self, kind: str, manifest_fingerprint: str) -> HasStatus: ...


def check_key(kind: str, manifest_fingerprint: str) -> None:
    if kind not in c.ENTITY_KINDS or not _KEY_RE.fullmatch(manifest_fingerprint or ""):
        raise LegendError(
            "invalid_request", "store key must be a kind and rmf0 fingerprint"
        )


class LocalDirectoryStore:
    """Reference store: ONE file per key under ``root`` (header line + bytes).

    The entry is ``"<sha256> <length>\\n" + bytes``, written to a temp file, fsynced and
    published by a hard link (create-if-absent). Digest, length and bytes therefore become
    visible together or not at all, and a concurrent writer can never replace the winner's
    header: there is no second file to race on.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _check_layout(self, create: bool) -> None:
        """Refuse an unknown or legacy layout with a closed code; stamp an empty root."""
        marker = self.root / STORE_FORMAT_FILE
        try:
            if marker.exists():
                if marker.read_text() != STORE_FORMAT:
                    raise LegendError(LAYOUT_UNSUPPORTED, "unsupported store format")
                return
            populated = self.root.is_dir() and any(
                p.is_file() and p.suffix not in (".tmp", ".entry") and p != marker
                for p in self.root.rglob("*")
            )
        except OSError:
            return  # unreachable root: the operation reports that itself
        if populated:
            raise LegendError(LAYOUT_UNSUPPORTED, "unsupported store layout")
        if not create:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(STORE_FORMAT)
            try:
                os.link(tmp, marker)
            except FileExistsError:
                self._check_layout(False)
        finally:
            os.unlink(tmp)

    def _entry(self, kind: str, fp: str) -> Path:
        check_key(kind, fp)
        namespace, _, digest = fp.rpartition(":sha256:")
        return (self.root / kind / namespace / digest).with_suffix(".entry")

    @staticmethod
    def encode(data: bytes) -> bytes:
        return (
            f"{hashlib.sha256(data).hexdigest()} {len(data)}\n".encode("ascii") + data
        )

    @staticmethod
    def _split(raw: bytes) -> tuple[str, int, bytes] | None:
        head, sep, body = raw.partition(b"\n")
        parts = head.split(b" ")
        if not sep or len(parts) != 2 or not parts[1].isdigit():
            return None
        try:
            return parts[0].decode("ascii"), int(parts[1]), body
        except UnicodeDecodeError:
            return None

    def put(self, kind: str, manifest_fingerprint: str, stream: BinaryIO) -> PutStatus:
        path = self._entry(kind, manifest_fingerprint)
        data = stream.read()
        digest = hashlib.sha256(data).hexdigest()
        try:
            self._check_layout(create=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():  # same key: identical bytes are fine, else a conflict
                return self._same(path, digest)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(self.encode(data))
                    handle.flush()
                    os.fsync(handle.fileno())  # durable before it becomes visible
                try:
                    os.link(tmp, path)  # atomic publish; fails if a concurrent put won
                except FileExistsError:
                    return self._same(path, digest)
            finally:
                os.unlink(tmp)
            return "stored_unverified"
        except PermissionError:
            return "permission_denied"
        except OSError:
            return "unreachable"

    def _same(self, path: Path, digest: str) -> PutStatus:
        split = self._split(path.read_bytes())
        return (
            "stored_unverified"
            if split is not None and split[0] == digest
            else "store_key_conflict"
        )

    def get(self, kind: str, manifest_fingerprint: str, dest_dir: Path) -> GetStatus:
        path = self._entry(kind, manifest_fingerprint)
        self._check_layout(create=False)
        try:
            if not path.exists():
                return "missing"
            split = self._split(path.read_bytes())
            if split is None:
                return "corrupted"
            digest, length, data = split
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            (Path(dest_dir) / BUNDLE_NAME).write_bytes(data)
            if len(data) < length:
                return "truncated"
            if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
                return "corrupted"
            return "retrieved"
        except PermissionError:
            return "permission_denied"
        except OSError:
            return "unreachable"

    def has(self, kind: str, manifest_fingerprint: str) -> HasStatus:
        self._check_layout(create=False)
        try:
            return (
                "available"
                if self._entry(kind, manifest_fingerprint).exists()
                else "missing"
            )
        except OSError:
            return "unknown"
