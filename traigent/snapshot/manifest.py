"""The ``tes1`` snapshot manifest and the immutable ``ExecutionSnapshot``.

A snapshot is a manifest plus the bytes of every part the manifest declares as
present. The manifest is canonical JSON (UTF-8, sorted keys, compact
separators, no NaN, no floats), so its digest is a function of its content.

v1 snapshots are *unbound client declarations*: they say "these bytes were
captured", not "this is what ran".
"""

from __future__ import annotations

import copy
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
    "MANDATORY_PARTS",
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
_CREATED_AT_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")

# Slots every ``tes1`` manifest declares (present or unavailable). The evaluator
# group is the other mandatory piece: at least one ``evaluator/...`` part.
MANDATORY_PARTS: tuple[str, ...] = (
    "dataset/examples.jsonl",
    "config/config.json",
    "config/config_space.json",
    "agent/callable.py",
    "agent/bound_state.json",
    "agent/git.json",
    "agent/assets/prompts.json",
    "agent/assets/tool_definitions.json",
)
_EVALUATOR_PREFIX = "evaluator/"
_FINGERPRINT_KEYS = (
    "dataset_fp1",
    "config_space_fp1",
    "agent_fp1",
    "evaluator_fp1",
    "agent_build_digest",
    "agent_build_coverage",
    "agent_build_unavailable",
)
_CODE_IDENTITY_KEYS = ("origin", "digest", "algorithm")
_COVERAGE_VALUES = frozenset({"complete", "partial"})


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


def _is_str_or_none(value: object) -> bool:
    return value is None or isinstance(value, str)


def _validate_metadata(manifest: Mapping[str, Any]) -> None:
    """Fields every ``tes1`` manifest carries besides its parts."""
    if not isinstance(manifest.get("sdk_version"), str) or not manifest["sdk_version"]:
        raise ValueError("bad sdk_version")
    created = manifest.get("created_at")
    if not isinstance(created, str) or _CREATED_AT_RE.fullmatch(created) is None:
        raise ValueError("bad created_at")
    if manifest.get("binding") != {"kind": "unbound"}:
        raise ValueError("bad binding")
    fingerprints = manifest.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("bad fingerprints")
    if any(
        key not in fingerprints or not _is_str_or_none(fingerprints[key])
        for key in _FINGERPRINT_KEYS
    ):
        raise ValueError("bad fingerprints")
    identity = fingerprints.get("evaluator_code_identity")
    if not isinstance(identity, dict) or any(
        key not in identity or not _is_str_or_none(identity[key])
        for key in _CODE_IDENTITY_KEYS
    ):
        raise ValueError("bad evaluator_code_identity")
    coverage = manifest.get("snapshot_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("agent") not in _COVERAGE_VALUES
        or not isinstance(coverage.get("agent_gaps"), list)
        or not all(isinstance(gap, str) for gap in coverage["agent_gaps"])
    ):
        raise ValueError("bad snapshot_coverage")


def _validate_structure(manifest: Mapping[str, Any]) -> None:
    parts = manifest["parts"]
    required = manifest["required_parts"]
    if not isinstance(parts, dict) or not isinstance(required, list):
        raise ValueError("bad shape")
    if len(parts) > _MAX_PARTS:
        raise ValueError("bad shape")
    _validate_metadata(manifest)
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
    if any(not isinstance(path, str) for path in required) or len(set(required)) != len(
        required
    ):
        raise ValueError("bad required_parts")
    for path in required:
        if path not in parts:
            raise ValueError("required part not declared")
    # Independent of what the manifest says it requires: these slots are always
    # declared, so a manifest that omits any of them is not a ``tes1`` snapshot.
    if any(path not in required for path in MANDATORY_PARTS):
        raise ValueError("mandatory part missing")
    if not any(path.startswith(_EVALUATOR_PREFIX) for path in required):
        raise ValueError("evaluator group missing")


class ExecutionSnapshot:
    """Immutable manifest bytes plus the bytes of every present part.

    The constructor copies ``parts`` into a private dict and exposes a
    read-only view over that copy only, so mutating the caller's dict after
    construction changes nothing. The parsed manifest is likewise private:
    every accessor hands out a deep copy, so no caller can change what is
    displayed or published. ``repr`` shows digests, never content.
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
        """A deep copy of the parsed manifest (changing it changes nothing)."""
        return copy.deepcopy(self._manifest)

    def unavailable(self) -> tuple[tuple[str, str], ...]:
        """``(path, reason)`` for every part the manifest declares unavailable."""
        return tuple(
            (path, entry["reason"])
            for path, entry in sorted(self._manifest["parts"].items())
            if entry.get("status") == "unavailable"
        )

    def present_parts(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """``(path, entry copy)`` for each present part, in path order."""
        for path, entry in sorted(self._manifest["parts"].items()):
            if entry.get("status") != "unavailable":
                yield path, dict(entry)

    def __repr__(self) -> str:
        digests = ",".join(
            f"{path}={entry['sha256'][:19]}" for path, entry in self.present_parts()
        )
        return f"ExecutionSnapshot(manifest={self.manifest_digest}, parts=[{digests}])"


LoadedSnapshot = ExecutionSnapshot
