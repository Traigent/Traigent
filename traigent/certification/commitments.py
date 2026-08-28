"""Client-local secret-blinded commitment primitives for Certificate v0.

The only public operation in this module is the fixed vocabulary used by the
manifest builder. Raw-blind commitment computation and opening capabilities
are deliberately private: callers provide artifact documents, never keys.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

from .canonical import canonicalize_artifact_document

__all__ = ["ARTIFACT_KINDS", "COMMITMENT_SCHEME"]

COMMITMENT_SCHEME = "sha256_secret_blinded_v1"
SLOT_COMMITMENT_DOMAIN = b"traigent.cert.v0.slot_commitment.v1"
ARTIFACT_KINDS = (
    "agent_revision",
    "evaluation_dataset",
    "evaluator",
    "build_process_evidence",
)
_ARTIFACT_KIND_SET = frozenset(ARTIFACT_KINDS)
_MAX_BLIND_GENERATION_ATTEMPTS = 16


def _validate_blind(blind: bytes) -> None:
    if type(blind) is not bytes or len(blind) != 32:
        raise ValueError("blind must be exactly 32 bytes")


def _validate_artifact_kind(artifact_kind: str) -> None:
    if type(artifact_kind) is not str or artifact_kind not in _ARTIFACT_KIND_SET:
        raise ValueError("artifact_kind is not a fixed slot kind")


def _compute_slot_commitment(
    *,
    blind: bytes,
    artifact_kind: str,
    canonical_bytes: bytes,
) -> str:
    """Deterministic private helper for KATs and the local build operation."""

    _validate_blind(blind)
    _validate_artifact_kind(artifact_kind)
    if type(canonical_bytes) is not bytes:
        raise TypeError("canonical artifact bytes must be plain bytes")
    preimage = (
        SLOT_COMMITMENT_DOMAIN
        + b"\x00"
        + artifact_kind.encode("utf-8")
        + b"\x00"
        + canonical_bytes
    )
    return "sha256:" + hmac.new(blind, preimage, hashlib.sha256).hexdigest()


def _generate_unique_blinds(count: int) -> tuple[bytes, ...]:
    """Generate unique blinds for one build, failing closed on a bad source."""

    if type(count) is not int or count != len(ARTIFACT_KINDS):
        raise ValueError("blind generation count is invalid")
    seen: set[bytes] = set()
    blinds: list[bytes] = []
    for _ in range(_MAX_BLIND_GENERATION_ATTEMPTS):
        blind = secrets.token_bytes(32)
        _validate_blind(blind)
        if blind in seen:
            continue
        seen.add(blind)
        blinds.append(blind)
        if len(blinds) == count:
            return tuple(blinds)
    raise RuntimeError("could not generate unique client-local blinds")


class ClientEvidenceOpening:
    """Opaque client-local opening capability.

    The caller's artifact document is canonicalized once and immediately
    discarded. Only private immutable canonical bytes and the private blind
    remain. This class is intentionally not exported from the certification
    package and has no public opening or serialization method.
    """

    __slots__ = ("_canonical_bytes", "_blind")

    def __init__(self, artifact_document: dict[str, Any], blind: bytes) -> None:
        _validate_blind(blind)
        canonical = canonicalize_artifact_document(artifact_document)
        object.__setattr__(self, "_canonical_bytes", canonical.encode("utf-8"))
        object.__setattr__(self, "_blind", blind)

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("client-local opening is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return "ClientEvidenceOpening(<client-local>)"

    def _commitment(self, artifact_kind: str) -> str:
        return _compute_slot_commitment(
            blind=self._blind,
            artifact_kind=artifact_kind,
            canonical_bytes=self._canonical_bytes,
        )

    @staticmethod
    def _serialization_error() -> TypeError:
        return TypeError("client-local opening cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise self._serialization_error()

    def __reduce__(self) -> object:
        raise self._serialization_error()

    def __copy__(self) -> object:
        raise self._serialization_error()

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise self._serialization_error()

    def __getstate__(self) -> object:
        raise self._serialization_error()
