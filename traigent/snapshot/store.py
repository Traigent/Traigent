"""The snapshot store contract, publish/verify/load, and the ArtifactStore adapter.

A store is a dumb, atomic, create-if-absent key/value surface. It verifies
nothing about digests: the caller passes bytes and the store only guarantees
"first writer wins; a second writer with different bytes gets CONFLICT".
Layout (relative to the store's own prefix)::

    blobs/sha256/<hex>              one object per distinct part
    snapshots/<hex>/manifest.json   written last, after every blob
    idem/<sha256(idempotency key)>  key -> manifest digest record
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.manifest import (
    DIGEST_PREFIX,
    ExecutionSnapshot,
    LoadedSnapshot,
    digest_of,
    is_valid_digest,
    parse_manifest,
)

__all__ = [
    "CreateResult",
    "SnapshotArtifactStore",
    "SnapshotRef",
    "SnapshotStore",
    "SnapshotVerification",
    "load_snapshot",
    "publish_snapshot",
    "require_secure_endpoint",
    "validate_key",
    "verify_snapshot",
]

_KEY_RE = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*")
_MAX_KEY = 512
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_HTTPS = "https"
_HTTP = "http"


class CreateResult(StrEnum):
    CREATED = "created"
    EXISTED_IDENTICAL = "existed_identical"


@runtime_checkable
class SnapshotStore(Protocol):
    """Atomic create-if-absent byte store. Implementations never log bytes."""

    @property
    def max_object_bytes(self) -> int | None:
        """Largest object the backend accepts, or ``None`` for no policy cap."""

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        """Store ``data`` at ``key`` atomically unless ``key`` exists.

        If the key exists with identical bytes return ``EXISTED_IDENTICAL``;
        with different bytes raise ``SnapshotStoreError(CONFLICT)``.
        """

    def read(self, key: str) -> bytes | None:
        """The bytes at ``key``; ``None`` ONLY for a definite not-found.

        Authentication and transport failures raise ``SnapshotStoreError``.
        """

    def locator(self) -> str:
        """A credential-free, query-free destination string."""


def validate_key(key: object) -> str:
    """Store keys are relative, ``..``-free, slash-separated names."""
    if (
        not isinstance(key, str)
        or len(key) > _MAX_KEY
        or _KEY_RE.fullmatch(key) is None
        or any(segment in (".", "..") for segment in key.split("/"))
    ):
        raise ValueError("invalid snapshot store key")
    return key


def require_secure_endpoint(
    url: str, *, allow_insecure_local: bool, backend: str
) -> None:
    """HTTPS for every remote endpoint; plain HTTP only for explicit loopback."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        raise SnapshotStoreError(
            SnapshotErrorCode.UNSAFE_DESTINATION, backend=backend
        ) from None
    if parts.scheme == _HTTPS and host:
        return
    if parts.scheme == _HTTP and allow_insecure_local and host in _LOCAL_HOSTS:
        return
    raise SnapshotStoreError(SnapshotErrorCode.UNSAFE_DESTINATION, backend=backend)


@dataclass(frozen=True)
class SnapshotRef:
    manifest_digest: str
    locator: str


@dataclass(frozen=True)
class SnapshotVerification:
    """Two independent answers: is it intact, and is it complete.

    ``integrity``: ``ok | manifest_mismatch | part_missing | part_corrupt``.
    ``completeness``: ``complete | partial`` (``partial`` lists every
    unavailable part with its reason). ``problems`` lists ``(path, kind)`` for
    each missing or corrupt part.
    """

    manifest_digest: str
    integrity: str
    completeness: str
    unavailable: tuple[tuple[str, str], ...] = ()
    problems: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return self.integrity == "ok"


def _hex(digest: str) -> str:
    return digest[len(DIGEST_PREFIX) :]


def blob_key(digest: str) -> str:
    return f"blobs/sha256/{_hex(digest)}"


def manifest_key(manifest_digest: str) -> str:
    return f"snapshots/{_hex(manifest_digest)}/manifest.json"


def _require_digest(value: str) -> str:
    if not is_valid_digest(value):
        raise ValueError("manifest digest must be sha256:<64 lowercase hex>")
    return value


def publish_snapshot(store: SnapshotStore, snapshot: ExecutionSnapshot) -> SnapshotRef:
    """Write every part, then the manifest last. Republishing writes nothing new."""
    blobs: dict[str, tuple[str, bytes]] = {}
    for path, entry in snapshot.present_parts():
        data = snapshot.parts.get(path)
        if (
            data is None
            or len(data) != entry["size"]
            or digest_of(data) != entry["sha256"]
        ):
            raise SnapshotStoreError(SnapshotErrorCode.CORRUPT, part=path)
        blobs[entry["sha256"]] = (path, data)
    if len(snapshot.parts) != sum(1 for _ in snapshot.present_parts()):
        raise SnapshotStoreError(SnapshotErrorCode.CORRUPT, part="manifest")
    limit = store.max_object_bytes
    if limit is not None:
        for path, data in [*blobs.values(), ("manifest", snapshot.manifest_bytes)]:
            if len(data) > limit:
                raise SnapshotStoreError(SnapshotErrorCode.TOO_LARGE, part=path)
    for digest, (_, data) in sorted(blobs.items()):
        store.create_if_absent(blob_key(digest), data)
    store.create_if_absent(
        manifest_key(snapshot.manifest_digest), snapshot.manifest_bytes
    )
    return SnapshotRef(snapshot.manifest_digest, store.locator())


def _read_manifest(store: SnapshotStore, manifest_digest: str) -> bytes | None:
    return store.read(manifest_key(_require_digest(manifest_digest)))


def verify_snapshot(store: SnapshotStore, manifest_digest: str) -> SnapshotVerification:
    """Check the stored snapshot against ``manifest_digest``.

    A missing manifest raises ``NOT_FOUND`` and a malformed or unsupported one
    ``CORRUPT``; transport and auth failures raise, they never become
    "missing".
    """
    data = _read_manifest(store, manifest_digest)
    if data is None:
        raise SnapshotStoreError(SnapshotErrorCode.NOT_FOUND, part="manifest")
    if digest_of(data) != manifest_digest:
        return SnapshotVerification(manifest_digest, "manifest_mismatch", "partial")
    manifest = parse_manifest(data)
    unavailable = tuple(
        (path, entry["reason"])
        for path, entry in sorted(manifest["parts"].items())
        if entry.get("status") == "unavailable"
    )
    problems: list[tuple[str, str]] = []
    for path, entry in sorted(manifest["parts"].items()):
        if entry.get("status") == "unavailable":
            continue
        blob = store.read(blob_key(entry["sha256"]))
        if blob is None:
            problems.append((path, "part_missing"))
        elif len(blob) != entry["size"] or digest_of(blob) != entry["sha256"]:
            problems.append((path, "part_corrupt"))
    kinds = {kind for _, kind in problems}
    integrity = (
        "part_missing" if "part_missing" in kinds else "part_corrupt" if kinds else "ok"
    )
    return SnapshotVerification(
        manifest_digest,
        integrity,
        "partial" if unavailable else "complete",
        unavailable,
        tuple(problems),
    )


def load_snapshot(store: SnapshotStore, manifest_digest: str) -> LoadedSnapshot:
    """Read a snapshot, validating every byte string before returning it."""
    data = _read_manifest(store, manifest_digest)
    if data is None:
        raise SnapshotStoreError(SnapshotErrorCode.NOT_FOUND, part="manifest")
    if digest_of(data) != manifest_digest:
        raise SnapshotStoreError(SnapshotErrorCode.CORRUPT, part="manifest")
    manifest = parse_manifest(data)
    parts: dict[str, bytes] = {}
    for path, entry in sorted(manifest["parts"].items()):
        if entry.get("status") == "unavailable":
            continue
        blob = store.read(blob_key(entry["sha256"]))
        if blob is None:
            raise SnapshotStoreError(SnapshotErrorCode.NOT_FOUND, part=path)
        if len(blob) != entry["size"] or digest_of(blob) != entry["sha256"]:
            raise SnapshotStoreError(SnapshotErrorCode.CORRUPT, part=path)
        parts[path] = blob
    return ExecutionSnapshot(data, parts)


class SnapshotArtifactStore:
    """Adapts a :class:`SnapshotStore` to ``connectors.ports.ArtifactStore``.

    ``put(snapshot, idem_key)`` is idempotent per key: the same key with the
    same snapshot returns the same ref; the same key with a different snapshot
    raises ``CONFLICT``. The key -> digest record is itself create-if-absent.
    """

    def __init__(self, store: SnapshotStore) -> None:
        self._store = store

    def __repr__(self) -> str:
        return f"SnapshotArtifactStore({self._store.locator()})"

    @staticmethod
    def _idem_key(idem_key: str) -> str:
        if not isinstance(idem_key, str) or not idem_key:
            raise ValueError("idem_key must be a non-empty string")
        return "idem/" + hashlib.sha256(idem_key.encode("utf-8")).hexdigest()

    def put(self, artifact: ExecutionSnapshot, idem_key: str) -> SnapshotRef:
        record_key = self._idem_key(idem_key)
        record = artifact.manifest_digest.encode("ascii")
        existing = self._store.read(record_key)
        if existing is not None and existing != record:
            raise SnapshotStoreError(SnapshotErrorCode.CONFLICT, part="idem")
        ref = publish_snapshot(self._store, artifact)
        self._store.create_if_absent(record_key, record)
        return ref

    def get(self, version_ref: SnapshotRef) -> LoadedSnapshot:
        return load_snapshot(self._store, version_ref.manifest_digest)
