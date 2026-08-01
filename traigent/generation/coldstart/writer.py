"""Fail-closed local artifact persistence for cold-start evaluation datasets.

The writer deliberately owns the on-disk boundary.  Callers cannot select an
alternative filename, split, or serialization shape: a tuning dataset is either
the complete ``coldstart_tuning.jsonl`` artifact or it is absent.  Audit records
are intentionally *not* Dataset-compatible, so they cannot accidentally be
passed to an optimizer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TUNING_FILENAME = "coldstart_tuning.jsonl"
_AUDIT_FILENAME = "coldstart_audit.jsonl"
_MANIFEST_FILENAME = "coldstart_manifest.json"


class ColdStartArtifactError(ValueError):
    """Raised when cold-start artifacts cannot be safely persisted."""


@dataclass(frozen=True, slots=True)
class ColdStartArtifactPaths:
    """The fixed local artifact names returned by the writer."""

    audit_path: Path
    manifest_path: Path
    tuning_path: Path | None


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes or fail before any artifact is replaced."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ColdStartArtifactError(
            "Cold-start artifact payload must be JSON serializable."
        ) from exc


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Encode JSONL deterministically, including its final newline."""
    encoded_rows = [canonical_json_bytes(dict(row)) for row in rows]
    if not encoded_rows:
        return b""
    return b"\n".join(encoded_rows) + b"\n"


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest used by the manifest integrity check."""
    return hashlib.sha256(payload).hexdigest()


def _has_expected_output(value: Any) -> bool:
    """Reject absent and blank text gold before it can enter a tuning artifact."""
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    """Make an absolute path without resolving through an untrusted symlink."""
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return Path(os.path.abspath(os.fspath(raw)))


def _reject_symlink_components(path: Path) -> None:
    """Reject a target whose existing path components include a symlink."""
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ColdStartArtifactError(
                "Cold-start artifacts refuse symlink output paths."
            )


def _prepare_output_dir(output_dir: str | os.PathLike[str]) -> Path:
    target = _lexical_absolute(output_dir)
    _reject_symlink_components(target)
    target.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target)
    if not target.is_dir():
        raise ColdStartArtifactError("Cold-start output path must be a directory.")
    return target


def _artifact_target(output_dir: Path, filename: str) -> Path:
    target = output_dir / filename
    if target.is_symlink():
        raise ColdStartArtifactError(
            "Cold-start artifacts refuse symlink output targets."
        )
    if target.exists() and not target.is_file():
        raise ColdStartArtifactError(
            "Cold-start artifact target exists but is not a regular file."
        )
    return target


def _atomic_replace(target: Path, payload: bytes) -> None:
    """Durably replace one checked regular-file target with *payload*."""
    if target.is_symlink():
        raise ColdStartArtifactError(
            "Cold-start artifacts refuse symlink output targets."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise ColdStartArtifactError(
            "Could not atomically write cold-start artifact."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _validate_tuning_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Enforce the fixed Dataset-compatible tuning persistence contract."""
    example_ids: set[str] = set()
    for row in rows:
        if "metadata" in row:
            raise ColdStartArtifactError(
                "Cold-start tuning rows must not use a top-level metadata wrapper."
            )
        if not isinstance(row.get("input"), Mapping) or not row["input"]:
            raise ColdStartArtifactError(
                "Cold-start tuning rows require a non-empty mapping input."
            )
        if not _has_expected_output(row.get("expected_output")):
            raise ColdStartArtifactError(
                "Cold-start tuning rows require one non-empty expected_output."
            )
        if isinstance(row.get("expected_output"), Mapping):
            raise ColdStartArtifactError(
                "Cold-start tuning rows do not support mapping expected_output."
            )
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ColdStartArtifactError(
                "Cold-start tuning rows require a non-empty unique example_id."
            )
        if example_id in example_ids:
            raise ColdStartArtifactError(
                "Cold-start tuning example_id values must be unique."
            )
        example_ids.add(example_id)
        provenance = row.get("traigent_coldstart")
        if not isinstance(provenance, Mapping) or provenance.get("split") != "tune":
            raise ColdStartArtifactError(
                "Cold-start tuning rows must carry literal tune provenance."
            )


def _validate_manifest(
    manifest: Mapping[str, Any], tuning_payload: bytes | None
) -> None:
    """Keep manifest claims coupled to the exact payload being persisted."""
    if manifest.get("holdout_prohibited") is not True:
        raise ColdStartArtifactError(
            "Cold-start manifests must prohibit holdout output."
        )
    if tuning_payload is None:
        if manifest.get("outcome") != "discovery_only":
            raise ColdStartArtifactError(
                "Discovery-only manifest must declare discovery_only outcome."
            )
        if (
            manifest.get("dataset_path") is not None
            or manifest.get("dataset_sha256") is not None
        ):
            raise ColdStartArtifactError(
                "Discovery-only manifest must not describe a tuning dataset."
            )
        return

    if manifest.get("outcome") != "eval_set":
        raise ColdStartArtifactError("Eligible manifest must declare eval_set outcome.")
    if manifest.get("dataset_path") != _TUNING_FILENAME:
        raise ColdStartArtifactError("Manifest must use the fixed tuning dataset path.")
    if manifest.get("dataset_sha256") != sha256_bytes(tuning_payload):
        raise ColdStartArtifactError(
            "Manifest dataset SHA-256 must match the exact tuning dataset bytes."
        )


def write_coldstart_artifacts(
    *,
    output_dir: str | os.PathLike[str],
    tuning_rows: Iterable[Mapping[str, Any]] | None,
    audit_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> ColdStartArtifactPaths:
    """Write the fixed cold-start artifact set with per-file atomic replacement.

    ``tuning_rows=None`` is the only discovery-only representation.  In that
    case a pre-existing tuning file is rejected instead of being silently left
    beside a discovery manifest, which would make stale tuning data look newly
    admitted.
    """
    output = _prepare_output_dir(output_dir)
    tuning_target = _artifact_target(output, _TUNING_FILENAME)
    audit_target = _artifact_target(output, _AUDIT_FILENAME)
    manifest_target = _artifact_target(output, _MANIFEST_FILENAME)

    audit_list = [dict(row) for row in audit_rows]
    if any("input" in row or "input_data" in row for row in audit_list):
        raise ColdStartArtifactError(
            "Cold-start audit rows must not be Dataset-compatible."
        )
    audit_payload = jsonl_bytes(audit_list)
    if not audit_payload:
        raise ColdStartArtifactError("Cold-start audit must contain at least one row.")
    manifest_payload = canonical_json_bytes(dict(manifest)) + b"\n"

    tuning_payload: bytes | None = None
    if tuning_rows is not None:
        tuning_list = [dict(row) for row in tuning_rows]
        _validate_tuning_rows(tuning_list)
        tuning_payload = jsonl_bytes(tuning_list)
        if not tuning_payload:
            raise ColdStartArtifactError(
                "Eligible cold-start output must contain at least one tuning row."
            )
    elif tuning_target.exists():
        raise ColdStartArtifactError(
            "Discovery-only output refuses a directory with an existing tuning dataset."
        )

    _validate_manifest(manifest, tuning_payload)

    if tuning_payload is not None:
        _atomic_replace(tuning_target, tuning_payload)
    _atomic_replace(audit_target, audit_payload)
    _atomic_replace(manifest_target, manifest_payload)

    return ColdStartArtifactPaths(
        tuning_path=tuning_target if tuning_payload is not None else None,
        audit_path=audit_target,
        manifest_path=manifest_target,
    )


__all__ = [
    "ColdStartArtifactError",
    "ColdStartArtifactPaths",
    "canonical_json_bytes",
    "jsonl_bytes",
    "sha256_bytes",
    "write_coldstart_artifacts",
]
