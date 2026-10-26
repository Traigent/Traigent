"""The ``tes1`` snapshot manifest and the immutable ``ExecutionSnapshot``.

A snapshot is a manifest plus the bytes of every part the manifest declares as
present. The manifest is canonical JSON (UTF-8, sorted keys, compact
separators, no NaN, no floats), so its digest is a function of its content.

v1 snapshots are *unbound client declarations*: they say "these bytes were
captured", not "this is what ran".
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError

__all__ = [
    "FORMAT",
    "ExecutionSnapshot",
    "LoadedSnapshot",
    "PartMediaType",
    "UnavailableReason",
    "canonical_manifest_bytes",
    "digest_of",
    "is_valid_digest",
    "parse_manifest",
    "validate_part_path",
]

FORMAT = "tes1"
DIGEST_PREFIX = "sha256:"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_PART_PATH_RE = re.compile(r"[A-Za-z0-9_.:+=@-]+(/[A-Za-z0-9_.:+=@-]+)*")
_MAX_PART_PATH = 512
_MAX_PARTS = 4096


class UnavailableReason(StrEnum):
    """Why a part is not in the snapshot. Closed list."""

    NOT_PROVIDED = "NOT_PROVIDED"
    NOT_REQUESTED = "NOT_REQUESTED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    EXTERNAL_EVALUATOR = "EXTERNAL_EVALUATOR"
    BUILTIN_EVALUATOR = "BUILTIN_EVALUATOR"
    GIT_UNAVAILABLE = "GIT_UNAVAILABLE"
    DATASET_NOT_MATERIALIZED = "DATASET_NOT_MATERIALIZED"
    NOT_SERIALIZABLE = "NOT_SERIALIZABLE"
    TOO_LARGE = "TOO_LARGE"
    BOUND_STATE_UNAVAILABLE = "BOUND_STATE_UNAVAILABLE"


class PartMediaType(StrEnum):
    JSON = "application/json"
    JSONL = "application/x-ndjson"
    PYTHON = "text/x-python"
    PATCH = "text/x-diff"


def is_valid_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def digest_of(data: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def validate_part_path(path: object) -> str:
    """A part path is a relative, ``..``-free, slash-separated name."""
    if (
        not isinstance(path, str)
        or len(path) > _MAX_PART_PATH
        or _PART_PATH_RE.fullmatch(path) is None
        or any(segment in (".", "..") for segment in path.split("/"))
    ):
        raise ValueError("invalid snapshot part path")
    return path


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Canonical manifest encoding. Raises ``ValueError`` on NaN or floats."""
    _reject_floats(manifest)
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_floats(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float):
            raise ValueError("manifest must not contain floats")
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise ValueError("non-finite number")


def _reject_float(_: str) -> Any:
    raise ValueError("float")


def parse_manifest(manifest_bytes: bytes) -> dict[str, Any]:
    """Parse and structurally validate manifest bytes.

    Raises ``SnapshotStoreError(CORRUPT, part="manifest")`` for anything that
    is not a well-formed, canonical ``tes1`` manifest. The message never
    echoes manifest content.
    """
    try:
        parsed = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
        )
        if not isinstance(parsed, dict) or parsed.get("format") != FORMAT:
            raise ValueError("unsupported format")
        if canonical_manifest_bytes(parsed) != manifest_bytes:
            raise ValueError("not canonical")
        _validate_structure(parsed)
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        raise SnapshotStoreError(SnapshotErrorCode.CORRUPT, part="manifest") from None
    return parsed


def _validate_structure(manifest: Mapping[str, Any]) -> None:
    parts = manifest["parts"]
    required = manifest["required_parts"]
    if not isinstance(parts, dict) or not isinstance(required, list):
        raise ValueError("bad shape")
    if len(parts) > _MAX_PARTS or not isinstance(manifest["binding"], dict):
        raise ValueError("bad shape")
    for path, entry in parts.items():
        validate_part_path(path)
        if not isinstance(entry, dict):
            raise ValueError("bad part")
        if entry.get("status") == "unavailable":
            if entry.get("reason") not in {r.value for r in UnavailableReason}:
                raise ValueError("bad reason")
            continue
        size = entry.get("size")
        if (
            not is_valid_digest(entry.get("sha256"))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(entry.get("media_type"), str)
        ):
            raise ValueError("bad part")
    for path in required:
        if path not in parts:
            raise ValueError("required part not declared")


class ExecutionSnapshot:
    """Immutable manifest bytes plus the bytes of every present part.

    The constructor copies ``parts`` into a private dict and exposes a
    read-only view over that copy only, so mutating the caller's dict after
    construction changes nothing. ``repr`` shows digests, never content.
    """

    __slots__ = ("_manifest", "_manifest_bytes", "_parts")

    def __init__(self, manifest_bytes: bytes, parts: Mapping[str, bytes]) -> None:
        private = {str(path): bytes(data) for path, data in parts.items()}
        self._manifest_bytes = bytes(manifest_bytes)
        self._parts: Mapping[str, bytes] = MappingProxyType(private)
        self._manifest = parse_manifest(self._manifest_bytes)

    @property
    def manifest_bytes(self) -> bytes:
        return self._manifest_bytes

    @property
    def manifest_digest(self) -> str:
        return digest_of(self._manifest_bytes)

    @property
    def parts(self) -> Mapping[str, bytes]:
        return self._parts

    @property
    def manifest(self) -> Mapping[str, Any]:
        """A read-only view of the parsed manifest."""
        return MappingProxyType(self._manifest)

    def unavailable(self) -> tuple[tuple[str, str], ...]:
        """``(path, reason)`` for every part the manifest declares unavailable."""
        return tuple(
            (path, entry["reason"])
            for path, entry in sorted(self._manifest["parts"].items())
            if entry.get("status") == "unavailable"
        )

    def present_parts(self) -> Iterator[tuple[str, dict[str, Any]]]:
        for path, entry in sorted(self._manifest["parts"].items()):
            if entry.get("status") != "unavailable":
                yield path, entry

    def __repr__(self) -> str:
        digests = ",".join(
            f"{path}={entry['sha256'][:19]}" for path, entry in self.present_parts()
        )
        return f"ExecutionSnapshot(manifest={self.manifest_digest}, parts=[{digests}])"


LoadedSnapshot = ExecutionSnapshot
