"""Client-local evidence for a G1 certificate: four blinded slots and their root.

Everything here is local. A tes1 snapshot (``traigent.snapshot``) is reduced to
four digest-only slot documents; each is committed under its own secret blind
(``sha256_secret_blinded_v1``) and the four commitments form a manifest whose
role-separated digest is the only value that later crosses to the Backend.

The slot documents and the blinds never leave the client. They live in a local
evidence file (mode 0600) behind a pluggable :class:`BlindCustody` hook.

Loss semantics: the blinds are the only way to open a commitment. If the
evidence file is lost, the certificate that was bound to its root stays valid,
but the four slots can no longer be opened or re-derived, and a delayed bind of
that evidence is impossible. There is no escrow; a customer who needs recovery
supplies a custody hook backed by their own secret manager.

Slot documents (producer-defined, digest-only; the only normative shape is
the ``build_process_evidence`` one, the other three are this producer's own
and carry no ids, counts, names, paths, config values or customer text):

- ``agent_revision``: ``{schema, agent_fp1, agent_build_digest,
  selected_config_sha256, config_space_fp1}``
- ``evaluation_dataset``: ``{schema, dataset_fp1, dataset_part_sha256}``
- ``evaluator``: ``{schema, evaluator_fp1, evaluator_code_identity,
  evaluator_part_sha256s}`` (sorted list of part digests)
- ``build_process_evidence``: ``{"snapshot_format": "tes1", "manifest_digest"}``

Each ``schema`` value is ``traigent.client_slot.<kind>.v1``. Documents are
plain str/int/bool/None/list/dict trees; floats and integers beyond 2**53-1
raise ``ClientEvidenceError("UNSUPPORTED_VALUE")``. The Schema known-answer
vectors are exercised in ``tests/unit/certification/test_schema_vectors.py``.

This binds the client's declaration of what was captured. It does not certify
that the agent ran, matches the snapshot, or is complete.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from traigent_schema import fp2

from traigent.snapshot.manifest import ExecutionSnapshot, is_valid_digest
from traigent.snapshot.store import SnapshotStore, load_snapshot, verify_snapshot

_SHA256_VALUE = "sha256:"

__all__ = [
    "BLIND_BYTES",
    "COMMITMENT_SCHEME",
    "ClientEvidence",
    "ClientEvidenceError",
    "BlindCustody",
    "LocalFileCustody",
    "SLOT_KINDS",
    "build_client_evidence",
    "build_slot_documents",
    "manifest_root_digest",
    "reconstruct_and_compare",
    "slot_commitment_digest",
]

COMMITMENT_SCHEME: Final = "sha256_secret_blinded_v1"
BLIND_BYTES: Final = 32
SLOT_KINDS: Final = (
    "agent_revision",
    "evaluation_dataset",
    "evaluator",
    "build_process_evidence",
)
MANIFEST_SCHEMA_VERSION: Final = "traigent.certificate_client_evidence_manifest.v0"
_SLOT_DOMAIN: Final = b"traigent.cert.v0.slot_commitment.v1"
_ROOT_DOMAIN: Final = b"traigent.agent_certificate.client_evidence_manifest_root.v1"
_EVIDENCE_VERSION: Final = 1
_ZERO_ROOT: Final = _SHA256_VALUE + "0" * 64
_DIGEST_HEX_RE = re.compile(r"[0-9a-f]{64}")


class ClientEvidenceError(Exception):
    """Closed, content-free failure. ``code`` is one of the fixed values below."""

    CODES: Final = frozenset(
        {
            "UNSUPPORTED_VALUE",
            "BLIND_INVALID",
            "EVIDENCE_PERSIST_FAILED",
            "EVIDENCE_NOT_FOUND",
            "EVIDENCE_STALE_OR_CORRUPT",
        }
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            raise ValueError("unknown client evidence error code")
        self.code = code
        super().__init__(code)


def _reject_unsupported(value: Any) -> None:
    """Slot documents are plain str/int/bool/None/list/dict trees; floats are refused.

    ``fp2.canonicalize`` accepts finite floats, but the certification profile
    forbids them, so the producer refuses them itself, recursively.
    """
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ClientEvidenceError("UNSUPPORTED_VALUE")
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif item is None or type(item) in (str, bool, int):
            continue
        else:
            raise ClientEvidenceError("UNSUPPORTED_VALUE")


def _jcs(document: Mapping[str, Any]) -> bytes:
    document = dict(document)
    _reject_unsupported(document)
    try:
        return str(fp2.canonicalize(document)).encode("utf-8")
    except Exception:
        # fp2 refuses e.g. integers beyond 2**53-1; surface the producer's closed
        # error, never fp2's (which can describe the offending value).
        raise ClientEvidenceError("UNSUPPORTED_VALUE") from None


def slot_commitment_digest(blind: bytes, kind: str, document: Mapping[str, Any]) -> str:
    """``sha256:`` + HMAC-SHA256(blind, domain || 0x00 || kind || 0x00 || jcs_v1(document))."""
    if kind not in SLOT_KINDS:
        raise ClientEvidenceError("UNSUPPORTED_VALUE")
    if type(blind) is not bytes or len(blind) != BLIND_BYTES:
        raise ClientEvidenceError("BLIND_INVALID")
    message = _SLOT_DOMAIN + b"\x00" + kind.encode("ascii") + b"\x00" + _jcs(document)
    return _SHA256_VALUE + hmac.new(blind, message, hashlib.sha256).hexdigest()


def _manifest_document(commitments: Mapping[str, str]) -> dict[str, Any]:
    if set(commitments) != set(SLOT_KINDS):
        raise ClientEvidenceError("UNSUPPORTED_VALUE")
    document: dict[str, Any] = {"schema_version": MANIFEST_SCHEMA_VERSION}
    for kind in SLOT_KINDS:
        document[kind] = {
            "artifact_kind": kind,
            "commitment_scheme": COMMITMENT_SCHEME,
            "slot_commitment_digest": commitments[kind],
        }
    return document


def manifest_root_digest(commitments: Mapping[str, str]) -> str:
    """``sha256:`` + SHA-256(root domain || 0x00 || jcs_v1(manifest))."""
    message = _ROOT_DOMAIN + b"\x00" + _jcs(_manifest_document(commitments))
    return _SHA256_VALUE + hashlib.sha256(message).hexdigest()


def _part_sha(snapshot: ExecutionSnapshot, path: str) -> str:
    entry = snapshot.manifest["parts"].get(path)
    if not isinstance(entry, dict) or not is_valid_digest(entry.get("sha256")):
        raise ClientEvidenceError("UNSUPPORTED_VALUE")
    return str(entry["sha256"])


def build_slot_documents(snapshot: ExecutionSnapshot) -> dict[str, dict[str, Any]]:
    """The four digest-only slot documents derived from a tes1 snapshot.

    No ids, counts, names, paths, config values or customer text: only digests
    copied from the snapshot manifest. The build slot is exactly
    ``{"snapshot_format": "tes1", "manifest_digest": ...}``.
    """
    manifest = snapshot.manifest
    fingerprints = manifest["fingerprints"]
    evaluator_parts = sorted(
        _part_sha(snapshot, path)
        for path, entry in snapshot.present_parts()
        if path.startswith("evaluator/")
    )
    if not evaluator_parts:
        raise ClientEvidenceError("UNSUPPORTED_VALUE")
    documents: dict[str, dict[str, Any]] = {
        "agent_revision": {
            "schema": "traigent.client_slot.agent_revision.v1",
            "agent_fp1": fingerprints.get("agent_fp1"),
            "agent_build_digest": fingerprints.get("agent_build_digest"),
            "selected_config_sha256": _part_sha(snapshot, "config/config.json"),
            "config_space_fp1": fingerprints.get("config_space_fp1"),
        },
        "evaluation_dataset": {
            "schema": "traigent.client_slot.evaluation_dataset.v1",
            "dataset_fp1": fingerprints.get("dataset_fp1"),
            "dataset_part_sha256": _part_sha(snapshot, "dataset/examples.jsonl"),
        },
        "evaluator": {
            "schema": "traigent.client_slot.evaluator.v1",
            "evaluator_fp1": fingerprints.get("evaluator_fp1"),
            "evaluator_code_identity": fingerprints.get("evaluator_code_identity"),
            "evaluator_part_sha256s": evaluator_parts,
        },
        "build_process_evidence": {
            "snapshot_format": "tes1",
            "manifest_digest": snapshot.manifest_digest,
        },
    }
    for document in documents.values():
        _reject_unsupported(document)
    return documents


@runtime_checkable
class BlindCustody(Protocol):
    """Where the evidence record (documents and blinds) is kept.

    ``save`` must be durable before it returns: the caller binds only after it
    succeeds. ``load`` returns ``None`` when nothing is stored under the root.
    """

    def save(self, root_hex: str, record: bytes) -> None: ...

    def load(self, root_hex: str) -> bytes | None: ...


class LocalFileCustody:
    """Default custody: ``<directory>/<root hex>.json``, file 0600, directory 0700."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        if directory is None:
            directory = Path.home() / ".traigent" / "certification" / "evidence"
        self._directory = Path(directory)

    def path_for(self, root_hex: str) -> str:
        """The evidence file path for ``root_hex`` (local data; never sent anywhere)."""
        return str(self._path(root_hex))

    def _path(self, root_hex: str) -> Path:
        if _DIGEST_HEX_RE.fullmatch(root_hex) is None:
            raise ClientEvidenceError("EVIDENCE_NOT_FOUND")
        return self._directory / f"{root_hex}.json"

    def save(self, root_hex: str, record: bytes) -> None:
        path = self._path(root_hex)
        try:
            self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self._directory, 0o700)
            fd, temp_name = tempfile.mkstemp(dir=self._directory, prefix=".evidence-")
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(record)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, path)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
            directory_fd = os.open(self._directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            raise ClientEvidenceError("EVIDENCE_PERSIST_FAILED") from None

    def load(self, root_hex: str) -> bytes | None:
        try:
            return self._path(root_hex).read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT") from None


@dataclass(frozen=True)
class ClientEvidence:
    """The in-memory result: documents and blinds stay local; only ``root`` is sent."""

    documents: Mapping[str, Mapping[str, Any]]
    blinds: Mapping[str, bytes]
    commitments: Mapping[str, str]
    root: str
    snapshot_manifest_digest: str

    @property
    def root_hex(self) -> str:
        return self.root[len(_SHA256_VALUE) :]

    def __repr__(self) -> str:
        return f"ClientEvidence(root={self.root})"

    def to_record(self) -> bytes:
        record = {
            "version": _EVIDENCE_VERSION,
            "commitment_scheme": COMMITMENT_SCHEME,
            "root": self.root,
            "documents": {kind: dict(self.documents[kind]) for kind in SLOT_KINDS},
            "blinds": {kind: self.blinds[kind].hex() for kind in SLOT_KINDS},
        }
        return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_record(cls, record: bytes) -> ClientEvidence:
        """Parse a stored record, recomputing commitments and the root from its parts."""
        try:
            data = json.loads(record)
            if (
                data["version"] != _EVIDENCE_VERSION
                or data["commitment_scheme"] != COMMITMENT_SCHEME
            ):
                raise ValueError
            documents = {kind: dict(data["documents"][kind]) for kind in SLOT_KINDS}
            blinds = {kind: bytes.fromhex(data["blinds"][kind]) for kind in SLOT_KINDS}
            retained_root = data["root"]
            manifest_digest = documents["build_process_evidence"]["manifest_digest"]
            commitments = {
                kind: slot_commitment_digest(blinds[kind], kind, documents[kind])
                for kind in SLOT_KINDS
            }
            root = manifest_root_digest(commitments)
        except (ValueError, KeyError, TypeError, ClientEvidenceError):
            raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT") from None
        if root != retained_root or not is_valid_digest(manifest_digest):
            raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT")
        return cls(documents, blinds, commitments, root, manifest_digest)


def build_client_evidence(
    snapshot: ExecutionSnapshot,
    *,
    custody: BlindCustody | None = None,
) -> ClientEvidence:
    """Commit a snapshot into the four slots and persist the evidence before returning.

    The record is written durably through ``custody`` (default
    :class:`LocalFileCustody`) before the caller can bind; a persistence failure
    raises ``EVIDENCE_PERSIST_FAILED`` and nothing is bound.
    """
    custody = custody if custody is not None else LocalFileCustody()
    documents = build_slot_documents(snapshot)
    blinds: dict[str, bytes] = {}
    while len(set(blinds.values())) < len(SLOT_KINDS):
        blinds = {kind: secrets.token_bytes(BLIND_BYTES) for kind in SLOT_KINDS}
    commitments = {
        kind: slot_commitment_digest(blinds[kind], kind, documents[kind])
        for kind in SLOT_KINDS
    }
    root = manifest_root_digest(commitments)
    if root == _ZERO_ROOT:
        raise ClientEvidenceError("UNSUPPORTED_VALUE")
    evidence = ClientEvidence(
        documents, blinds, commitments, root, snapshot.manifest_digest
    )
    custody.save(evidence.root_hex, evidence.to_record())
    return evidence


def reconstruct_and_compare(
    root: str,
    *,
    custody: BlindCustody,
    store: SnapshotStore,
) -> ClientEvidence:
    """Local reconstruction required before any delayed bind (SPEC F11).

    Loads the retained record, recomputes all four slot commitments and the root
    from the persisted documents and blinds, checks the root equals ``root``,
    verifies the referenced tes1 snapshot in ``store``, and re-derives all four
    slot documents from it, which must equal the persisted documents. Any mismatch raises
    ``EVIDENCE_STALE_OR_CORRUPT`` (or ``EVIDENCE_NOT_FOUND`` if nothing is stored).
    """
    if not is_valid_digest(root):
        raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT")
    record = custody.load(root[len(_SHA256_VALUE) :])
    if record is None:
        raise ClientEvidenceError("EVIDENCE_NOT_FOUND")
    evidence = ClientEvidence.from_record(record)
    if evidence.root != root:
        raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT")
    try:
        report = verify_snapshot(store, evidence.snapshot_manifest_digest)
        loaded = load_snapshot(store, evidence.snapshot_manifest_digest)
        rederived = build_slot_documents(loaded)
    except Exception:
        raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT") from None
    if (
        not report.ok
        or report.manifest_digest != evidence.snapshot_manifest_digest
        or {kind: dict(evidence.documents[kind]) for kind in SLOT_KINDS} != rederived
    ):
        raise ClientEvidenceError("EVIDENCE_STALE_OR_CORRUPT")
    return evidence
