"""Minimal ``ClientStore`` adapter: one file per key in a directory you control.

Run it (no network, no credentials)::

    python examples/advanced/legend-client-store/directory_store.py

It shows the three rules a real adapter (S3, GCS, a git repo, a NAS) must keep:
atomic publish, never overwrite a different capture, and verify on read.
See docs/guides/legend-client-store.md.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO

from traigent.legend.store import (
    BUNDLE_NAME,
    GetStatus,
    HasStatus,
    PutStatus,
    check_key,
)

_HEADER_END = b"\n"


class DirectoryStore:
    """File layout: ``<root>/<kind>/<digest>.entry`` = ``"<sha256> <length>\\n" + bytes``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _entry(self, kind: str, fingerprint: str) -> Path:
        check_key(
            kind, fingerprint
        )  # rejects anything that is not a kind + rmf0/rmfc0 key
        namespace, _, digest = fingerprint.rpartition(":sha256:")
        return self.root / kind / namespace / f"{digest}.entry"

    @staticmethod
    def _split(raw: bytes) -> tuple[str, int, bytes] | None:
        head, sep, body = raw.partition(_HEADER_END)
        parts = head.split(b" ")
        if not sep or len(parts) != 2 or not parts[1].isdigit():
            return None
        return parts[0].decode("ascii", "replace"), int(parts[1]), body

    def put(self, kind: str, manifest_fingerprint: str, stream: BinaryIO) -> PutStatus:
        path = self._entry(kind, manifest_fingerprint)
        data = stream.read()
        digest = hashlib.sha256(data).hexdigest()
        entry = f"{digest} {len(data)}\n".encode("ascii") + data
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if (
                path.exists()
            ):  # same key: identical bytes are fine, anything else is a conflict
                split = self._split(path.read_bytes())
                return (
                    "stored_unverified"
                    if split and split[0] == digest
                    else "store_key_conflict"
                )
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(entry)
                    handle.flush()
                    os.fsync(handle.fileno())  # durable before it becomes visible
                try:
                    os.link(tmp, path)  # atomic publish; fails if a concurrent put won
                except FileExistsError:
                    split = self._split(path.read_bytes())
                    if not split or split[0] != digest:
                        return "store_key_conflict"
            finally:
                os.unlink(tmp)
            return (
                "stored_unverified"  # an ack is never "verified"; the SDK checks that
            )
        except PermissionError:
            return "permission_denied"
        except OSError:
            return "unreachable"

    def get(self, kind: str, manifest_fingerprint: str, dest_dir: Path) -> GetStatus:
        path = self._entry(kind, manifest_fingerprint)
        try:
            if not path.exists():
                return "missing"
            split = self._split(path.read_bytes())
            if split is None:
                return "corrupted"
            digest, length, body = split
            dest = Path(dest_dir)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / BUNDLE_NAME).write_bytes(body)
            if len(body) < length:
                return "truncated"
            if len(body) != length or hashlib.sha256(body).hexdigest() != digest:
                return "corrupted"
            return "retrieved"
        except PermissionError:
            return "permission_denied"
        except OSError:
            return "unreachable"

    def has(self, kind: str, manifest_fingerprint: str) -> HasStatus:
        try:
            return (
                "available"
                if self._entry(kind, manifest_fingerprint).exists()
                else "missing"
            )
        except OSError:
            return "unknown"


def main() -> int:
    from traigent.legend import (
        CaptureIndex,
        ConfigurationSource,
        activate,
        capture,
        conformance_check,
        restore,
    )

    work = Path(tempfile.mkdtemp(prefix="legend-example-"))
    store, index = DirectoryStore(work / "store"), CaptureIndex()
    outcome = conformance_check(store)
    print("conformance:", outcome.status, outcome.kinds_failed)
    if not outcome.passed:
        return 1
    record = capture(
        "versionref_" + "1" * 32,
        ConfigurationSource({"temperature": 0.2}),
        store=store,
        conformance=outcome,
        index=index,
    )
    print(
        "recovery_state:", record.recovery_state, "coverage:", record.coverage["status"]
    )
    result = restore(record, work / "restored", store=store, index=index)
    print("restore grade:", result.grade)
    if result.grade == "exact":
        print("activated at:", activate(result, work / "live").target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
