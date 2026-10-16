"""``ClientStore``: the customer-side, no-delete byte store a capture is written to.

Keys are scheme-qualified recovery-manifest fingerprints (``rmf0:sha256:...``),
never pillar fingerprints. There is no delete: a store may only add. A ``put``
of an existing key with different bytes is a closed error, never an overwrite.
"""

from __future__ import annotations

import hashlib
import json
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
    """Reference store: one content-addressed pair of files per key under ``root``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _paths(self, kind: str, fp: str) -> tuple[Path, Path]:
        check_key(kind, fp)
        namespace, _, digest = fp.rpartition(":sha256:")
        base = self.root / kind / namespace / digest
        return base.with_suffix(".blob"), base.with_suffix(".meta")

    def put(self, kind: str, manifest_fingerprint: str, stream: BinaryIO) -> PutStatus:
        blob, meta = self._paths(kind, manifest_fingerprint)
        try:
            blob.parent.mkdir(parents=True, exist_ok=True)
            data = stream.read()
            digest = hashlib.sha256(data).hexdigest()
            if blob.exists():
                if self._existing_digest(meta) != digest:
                    return "store_key_conflict"
                return "stored_unverified"
            # Crash-safe order: the meta is published ATOMICALLY (temp + rename) BEFORE the
            # blob appears, and the blob is published atomically by a hard link. Readers key
            # on the blob, so an orphan meta (crash between the two) is invisible to has/get
            # and is simply replaced by the next put.
            fd, tmp = tempfile.mkstemp(dir=blob.parent, suffix=".tmp")
            mfd, mtmp = tempfile.mkstemp(dir=blob.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                with os.fdopen(mfd, "w") as handle:
                    handle.write(json.dumps({"sha256": digest, "length": len(data)}))
                os.replace(mtmp, meta)
                try:
                    os.link(tmp, blob)  # fails if a concurrent put won the key
                except FileExistsError:
                    if self._existing_digest(meta) != digest:
                        return "store_key_conflict"
            finally:
                for leftover in (tmp, mtmp):
                    try:
                        os.unlink(leftover)
                    except FileNotFoundError:
                        pass
            return "stored_unverified"
        except PermissionError:
            return "permission_denied"
        except OSError:
            return "unreachable"

    @staticmethod
    def _existing_digest(meta: Path) -> str | None:
        try:
            value = json.loads(meta.read_text()).get("sha256")
            return value if isinstance(value, str) else None
        except (OSError, ValueError, AttributeError):
            return None

    def get(self, kind: str, manifest_fingerprint: str, dest_dir: Path) -> GetStatus:
        blob, meta = self._paths(kind, manifest_fingerprint)
        try:
            if not blob.exists():
                return "missing"
            data = blob.read_bytes()
            try:
                info = json.loads(meta.read_text())
                digest, length = info["sha256"], int(info["length"])
            except (OSError, ValueError, KeyError, TypeError):
                return "corrupted"
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
        blob, _ = self._paths(kind, manifest_fingerprint)
        try:
            return "available" if blob.exists() else "missing"
        except OSError:
            return "unknown"
