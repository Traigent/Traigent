"""Stable client-local Agent Certificate v0 commitment kernel."""

from .commitments import ARTIFACT_KINDS, COMMITMENT_SCHEME
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    ClientEvidenceBuild,
    ClientEvidenceManifest,
    ClientEvidenceSlot,
    build_client_evidence_manifest,
    compute_manifest_root,
    serialize_manifest,
)
from .attestor import (
    ExpectedCertificateContext,
    create_client_co_attestation,
    verify_certificate_signatures,
    verify_client_co_attestation,
)

__all__ = [
    "ARTIFACT_KINDS",
    "COMMITMENT_SCHEME",
    "MANIFEST_SCHEMA_VERSION",
    "ClientEvidenceBuild",
    "ClientEvidenceManifest",
    "ClientEvidenceSlot",
    "build_client_evidence_manifest",
    "compute_manifest_root",
    "serialize_manifest",
    "create_client_co_attestation",
    "ExpectedCertificateContext",
    "verify_certificate_signatures",
    "verify_client_co_attestation",
]
