"""Client-local evidence-manifest model and minimal Backend binding."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from traigent.utils import fp2

from .commitments import (
    ARTIFACT_KINDS,
    COMMITMENT_SCHEME,
    ClientEvidenceOpening,
    _compute_slot_commitment,
    _generate_unique_blinds,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ClientEvidenceBuild",
    "ClientEvidenceManifest",
    "ClientEvidenceSlot",
    "build_client_evidence_manifest",
    "compute_manifest_root",
    "serialize_manifest",
]

MANIFEST_SCHEMA_VERSION = "traigent.certificate_client_evidence_manifest.v0"
_MANIFEST_ROOT_DOMAIN = b"traigent.agent_certificate.client_evidence_manifest_root.v1"
_SLOT_ORDER = ARTIFACT_KINDS
_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_LOCAL_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "agent_revision",
        "evaluation_dataset",
        "evaluator",
        "build_process_evidence",
    }
)
_SLOT_KEYS = frozenset({"artifact_kind", "commitment_scheme", "slot_commitment_digest"})
_BACKEND_BINDING_KEYS = frozenset({"manifest_root_digest", "commitment_scheme"})


class ClientEvidenceSlot:
    """One content-free manifest slot kept inside the client-local model."""

    __slots__ = ("_artifact_kind", "_commitment_scheme", "_slot_commitment_digest")

    def __init__(
        self,
        artifact_kind: str,
        commitment_scheme: str,
        slot_commitment_digest: str,
    ) -> None:
        object.__setattr__(self, "_artifact_kind", artifact_kind)
        object.__setattr__(self, "_commitment_scheme", commitment_scheme)
        object.__setattr__(self, "_slot_commitment_digest", slot_commitment_digest)

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("client-local slot is immutable")
        object.__setattr__(self, name, value)

    @property
    def artifact_kind(self) -> str:
        return self._artifact_kind

    @property
    def commitment_scheme(self) -> str:
        return self._commitment_scheme

    @property
    def slot_commitment_digest(self) -> str:
        return self._slot_commitment_digest

    def __repr__(self) -> str:
        return "ClientEvidenceSlot(<client-local commitment>)"

    @staticmethod
    def _serialization_error() -> TypeError:
        return TypeError("client-local slot cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> object:
        raise self._serialization_error()

    def __reduce__(self) -> object:
        raise self._serialization_error()

    def __copy__(self) -> object:
        raise self._serialization_error()

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise self._serialization_error()

    def __getstate__(self) -> object:
        raise self._serialization_error()

    def _to_local_dict(self) -> dict[str, str]:
        return {
            "artifact_kind": self._artifact_kind,
            "commitment_scheme": self._commitment_scheme,
            "slot_commitment_digest": self._slot_commitment_digest,
        }


class ClientEvidenceManifest:
    """The exactly-four-slot client-local evidence-manifest model.

    The four slot commitments are retained for local root computation.  The
    only plain-dict projection is the minimal Backend binding returned by
    ``to_dict``; generic serializers of this local object fail closed.
    """

    __slots__ = (
        "_agent_revision",
        "_evaluation_dataset",
        "_evaluator",
        "_build_process_evidence",
        "_schema_version",
    )

    def __init__(
        self,
        agent_revision: ClientEvidenceSlot,
        evaluation_dataset: ClientEvidenceSlot,
        evaluator: ClientEvidenceSlot,
        build_process_evidence: ClientEvidenceSlot,
        schema_version: str = MANIFEST_SCHEMA_VERSION,
    ) -> None:
        if type(schema_version) is not str or schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("schema_version is not the v0 manifest version")
        for slot in (
            agent_revision,
            evaluation_dataset,
            evaluator,
            build_process_evidence,
        ):
            if type(slot) is not ClientEvidenceSlot:
                raise TypeError("manifest slots must be ClientEvidenceSlot values")
        object.__setattr__(self, "_agent_revision", agent_revision)
        object.__setattr__(self, "_evaluation_dataset", evaluation_dataset)
        object.__setattr__(self, "_evaluator", evaluator)
        object.__setattr__(self, "_build_process_evidence", build_process_evidence)
        object.__setattr__(self, "_schema_version", schema_version)

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("client-local manifest is immutable")
        object.__setattr__(self, name, value)

    @property
    def agent_revision(self) -> ClientEvidenceSlot:
        return self._agent_revision

    @property
    def evaluation_dataset(self) -> ClientEvidenceSlot:
        return self._evaluation_dataset

    @property
    def evaluator(self) -> ClientEvidenceSlot:
        return self._evaluator

    @property
    def build_process_evidence(self) -> ClientEvidenceSlot:
        return self._build_process_evidence

    @property
    def schema_version(self) -> str:
        return self._schema_version

    def __repr__(self) -> str:
        return "ClientEvidenceManifest(<client-local material>)"

    @staticmethod
    def _serialization_error() -> TypeError:
        return TypeError("client-local manifest cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> object:
        raise self._serialization_error()

    def __reduce__(self) -> object:
        raise self._serialization_error()

    def __copy__(self) -> object:
        raise self._serialization_error()

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise self._serialization_error()

    def __getstate__(self) -> object:
        raise self._serialization_error()

    def to_dict(self) -> dict[str, Any]:
        """Return the only plain-dict projection allowed on the public surface."""

        return {
            "manifest_root_digest": compute_manifest_root(self),
            "commitment_scheme": COMMITMENT_SCHEME,
        }

    @property
    def backend_binding(self) -> dict[str, str]:
        """Return only the root and fixed scheme intended for transport."""

        return self.to_dict()

    def to_backend_binding(self) -> dict[str, str]:
        """Explicit spelling for callers constructing a request body."""

        return self.backend_binding


def _serialization_error() -> TypeError:
    return TypeError("client-local build capability cannot be serialized")


class ClientEvidenceBuild:
    """Opaque result retaining the manifest and private opening capabilities."""

    __slots__ = ("_manifest", "_openings")

    def __init__(
        self,
        manifest: ClientEvidenceManifest,
        openings: tuple[ClientEvidenceOpening, ...],
    ) -> None:
        if type(manifest) is not ClientEvidenceManifest:
            raise TypeError("build manifest must be a ClientEvidenceManifest")
        if type(openings) is not tuple or len(openings) != len(_SLOT_ORDER):
            raise TypeError("build openings are invalid")
        if any(type(opening) is not ClientEvidenceOpening for opening in openings):
            raise TypeError("build openings are invalid")
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_openings", openings)

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("client-local build capability is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return "ClientEvidenceBuild(<client-local>)"

    @property
    def manifest(self) -> ClientEvidenceManifest:
        """Return the content-free manifest model."""

        return self._manifest

    def manifest_dict(self) -> dict[str, Any]:
        """Return the minimal Backend-facing binding, never the local slots."""

        return serialize_manifest(self._manifest)

    @property
    def backend_binding(self) -> dict[str, str]:
        """Return only the root and fixed scheme intended for transport."""

        return serialize_manifest(self._manifest)

    def to_backend_binding(self) -> dict[str, str]:
        """Explicit spelling for callers constructing a request body."""

        return self.backend_binding

    def __reduce_ex__(self, protocol: int) -> object:
        raise _serialization_error()

    def __reduce__(self) -> object:
        raise _serialization_error()

    def __copy__(self) -> object:
        raise _serialization_error()

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise _serialization_error()

    def __getstate__(self) -> object:
        raise _serialization_error()


def _validate_local_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if (
        type(manifest) is not dict
        or any(type(key) is not str for key in manifest)
        or set(manifest) != _LOCAL_MANIFEST_KEYS
    ):
        raise ValueError("manifest must contain exactly the four fixed slots")
    if (
        type(manifest["schema_version"]) is not str
        or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version is invalid")

    result: dict[str, Any] = {"schema_version": MANIFEST_SCHEMA_VERSION}
    for kind in _SLOT_ORDER:
        slot = manifest[kind]
        if (
            type(slot) is not dict
            or any(type(key) is not str for key in slot)
            or set(slot) != _SLOT_KEYS
        ):
            raise ValueError("manifest slot shape is invalid")
        if type(slot["artifact_kind"]) is not str or slot["artifact_kind"] != kind:
            raise ValueError("manifest slot kind is invalid")
        if (
            type(slot["commitment_scheme"]) is not str
            or slot["commitment_scheme"] != COMMITMENT_SCHEME
        ):
            raise ValueError("manifest commitment scheme is invalid")
        digest = slot["slot_commitment_digest"]
        if type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError("manifest slot digest is invalid")
        result[kind] = {
            "artifact_kind": kind,
            "commitment_scheme": COMMITMENT_SCHEME,
            "slot_commitment_digest": digest,
        }
    return result


def _local_manifest_dict(
    manifest: ClientEvidenceManifest | dict[str, Any],
) -> dict[str, Any]:
    """Validate and copy the full manifest for client-local root computation."""

    if type(manifest) is ClientEvidenceManifest:
        candidate = {
            "schema_version": manifest.schema_version,
            "agent_revision": manifest.agent_revision._to_local_dict(),
            "evaluation_dataset": manifest.evaluation_dataset._to_local_dict(),
            "evaluator": manifest.evaluator._to_local_dict(),
            "build_process_evidence": manifest.build_process_evidence._to_local_dict(),
        }
    elif type(manifest) is dict:
        candidate = manifest
    else:
        raise TypeError("manifest must be a ClientEvidenceManifest or plain dict")
    return _validate_local_manifest(candidate)


def _validate_backend_binding(binding: dict[str, Any]) -> dict[str, str]:
    if type(binding) is not dict or set(binding) != _BACKEND_BINDING_KEYS:
        raise TypeError("client-local manifest cannot be serialized")
    digest = binding["manifest_root_digest"]
    scheme = binding["commitment_scheme"]
    if type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("manifest root digest is invalid")
    if type(scheme) is not str or scheme != COMMITMENT_SCHEME:
        raise ValueError("manifest commitment scheme is invalid")
    return {"manifest_root_digest": digest, "commitment_scheme": scheme}


def serialize_manifest(
    manifest: ClientEvidenceManifest | dict[str, Any],
) -> dict[str, str]:
    """Return only the minimal Backend-facing root binding.

    A full four-slot manifest is client-local and cannot be serialized through
    this public/wire helper.  Root computation uses ``_local_manifest_dict``.
    """

    if type(manifest) is ClientEvidenceManifest:
        return _validate_backend_binding(manifest.to_dict())
    if type(manifest) is dict:
        return _validate_backend_binding(manifest)
    raise TypeError("manifest must be a ClientEvidenceManifest or plain dict")


def build_client_evidence_manifest(
    agent_revision: dict[str, Any],
    evaluation_dataset: dict[str, Any],
    evaluator: dict[str, Any],
    build_process_evidence: dict[str, Any],
) -> ClientEvidenceBuild:
    """Build a fixed manifest from exactly four client-local documents."""

    documents = (
        agent_revision,
        evaluation_dataset,
        evaluator,
        build_process_evidence,
    )
    blinds = _generate_unique_blinds(len(documents))
    openings = tuple(
        ClientEvidenceOpening(document, blind)
        for document, blind in zip(documents, blinds, strict=True)
    )
    slots = {
        kind: ClientEvidenceSlot(
            kind,
            COMMITMENT_SCHEME,
            _compute_slot_commitment(
                blind=opening._blind,
                artifact_kind=kind,
                canonical_bytes=opening._canonical_bytes,
            ),
        )
        for kind, opening in zip(_SLOT_ORDER, openings, strict=True)
    }
    manifest = ClientEvidenceManifest(
        agent_revision=slots["agent_revision"],
        evaluation_dataset=slots["evaluation_dataset"],
        evaluator=slots["evaluator"],
        build_process_evidence=slots["build_process_evidence"],
    )
    return ClientEvidenceBuild(manifest, openings)


def compute_manifest_root(manifest: ClientEvidenceManifest | dict[str, Any]) -> str:
    """Compute Schema's role-separated v1 client evidence-manifest root."""

    wire = _local_manifest_dict(manifest)
    canonical = fp2.canonicalize(wire).encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(_MANIFEST_ROOT_DOMAIN + b"\x00" + canonical).hexdigest()
    )
