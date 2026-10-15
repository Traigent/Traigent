"""Recovery manifest builder: ordered representation digests, coverage and ``rmf0:`` fingerprint."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from traigent.identity.content_identity import canonical_bytes
from traigent.legend import _contract as c
from traigent.legend.errors import LegendError
from traigent.legend.pillars import Pillar


@dataclass(frozen=True)
class RecoveryManifest:
    document: dict[str, Any]
    fingerprint: str


def manifest_fingerprint(document: dict[str, Any]) -> str:
    """``rmf0:sha256:`` + hex SHA-256(domain || 0x00 || jcs_v1(manifest))."""
    framed = c.MANIFEST_DOMAIN.encode("utf-8") + b"\x00" + canonical_bytes(document)
    return c.MANIFEST_FINGERPRINT_PREFIX + hashlib.sha256(framed).hexdigest()


def representation_commitment(digests: Sequence[str]) -> str:
    """``sha256:`` + hex SHA-256(domain || 0x00 || jcs_v1({"representations": digests})).

    ``digests`` are the manifest's representation digests in ordinal order. The same
    construction is used by the Backend's ``identity_verified`` proof and the capture
    request; the manifest itself never leaves the client.
    """
    for d in digests:
        if not c.SHA256_DIGEST_RE.fullmatch(d):
            raise LegendError("invalid_request", "representation digest malformed")
    framed = (
        c.REPRESENTATION_COMMITMENT_DOMAIN.encode("utf-8")
        + b"\x00"
        + canonical_bytes({"representations": list(digests)})
    )
    return "sha256:" + hashlib.sha256(framed).hexdigest()


def pillar_representation_commitment(pillar: Pillar) -> str:
    """The commitment over ``pillar``'s ordered representations (what a proof must bind)."""
    return representation_commitment(
        ["sha256:" + hashlib.sha256(r.data).hexdigest() for r in pillar.representations]
    )


def validate_coverage(kind: str, coverage: dict[str, Any]) -> None:
    """Check a coverage object against ``RequiredComponents(kind)`` before it goes anywhere.

    ``covered`` and ``missing`` must be disjoint and together contain exactly the
    required components (``dependency_lock`` may additionally appear in ``covered``);
    ``complete`` iff ``missing`` is empty. Overlap or inconsistency is
    ``invalid_request``; a ``complete`` status without every required component is
    ``coverage_insufficient``.
    """
    required = set(c.REQUIRED_BY_KIND[kind])
    covered, missing = list(coverage["covered"]), list(coverage["missing"])
    if len(set(covered)) != len(covered) or len(set(missing)) != len(missing):
        raise LegendError("invalid_request", "coverage arrays repeat a component")
    if set(covered) & set(missing):
        raise LegendError("invalid_request", "covered and missing overlap")
    if coverage["status"] not in ("complete", "partial"):
        raise LegendError("invalid_request", "unknown coverage status")
    if (coverage["status"] == "complete") != (not missing):
        raise LegendError("invalid_request", "coverage status disagrees with missing")
    if not set(missing) <= required or not set(covered) <= required | {
        "dependency_lock"
    }:
        raise LegendError(
            "invalid_request", "coverage names a component outside the kind"
        )
    if coverage["status"] == "complete" and not required <= set(covered):
        raise LegendError(
            "coverage_insufficient", "complete needs every required component"
        )
    if not required <= set(covered) | set(missing):
        raise LegendError(
            "invalid_request", "coverage does not partition the required set"
        )


def identity_reproduced(pillar: Pillar, identity_verification_ref: str | None) -> bool:
    """The ``identity_digests_reproduced`` claim the client may make for ``pillar``.

    Locally-reproduced pillars (agent, evaluator files, configuration, dataset) need no
    skipped files. A backend-keyed pillar's keyed identity cannot be reproduced by the
    client alone: the claim is only made when a pre-capture ``identity_verified`` proof
    is cited (a search space has no local reproducer, so the proof is its only route).
    """
    if pillar.skipped != 0:
        return False
    if not pillar.backend_keyed:
        return bool(pillar.reproduced)
    if identity_verification_ref is None:
        return False
    return bool(pillar.reproduced) or pillar.kind == c.KIND_SEARCH_SPACE


def coverage_for(
    pillar: Pillar, identity_verification_ref: str | None = None
) -> dict[str, Any]:
    """Coverage per kind: ``covered`` and ``missing`` partition ``REQUIRED_BY_KIND[kind]``.

    ``complete`` iff ``missing`` is empty. A capture whose identity digests were not
    reproduced (INV-17; for a backend-keyed pillar: no cited proof) cannot be complete:
    the identity-bearing component moves to ``missing``.
    """
    required = c.REQUIRED_BY_KIND[pillar.kind]
    covered = [x for x in required if x in pillar.covered]
    identity = c.IDENTITY_COMPONENT_BY_KIND.get(pillar.kind)
    if identity is not None and not identity_reproduced(
        pillar, identity_verification_ref
    ):
        covered = [x for x in covered if x != identity]
    missing = [x for x in required if x not in covered]
    result = {
        "status": "partial" if missing else "complete",
        "covered": covered,
        "missing": missing,
    }
    validate_coverage(pillar.kind, result)
    return result


def build_recovery_manifest(
    pillar: Pillar,
    *,
    dependencies: Sequence[tuple[str, str]] = (),
    source_entity_ref: str | None = None,
    identity_verification_ref: str | None = None,
) -> RecoveryManifest:
    """Build the manifest for ``pillar`` (``dependencies``: ``(version_ref, kind)`` pairs).

    ``identity_verification_ref`` is the ``eventref_`` of the Backend's pre-capture
    ``identity_verified`` event for this version; a complete backend-keyed manifest
    must cite it (the proof binds :func:`pillar_representation_commitment`).
    """
    if identity_verification_ref is not None and not c.EVENT_REF_RE.fullmatch(
        identity_verification_ref
    ):
        raise LegendError(
            "invalid_request", "identity_verification_ref must be an event ref"
        )
    if not pillar.representations:
        raise LegendError(
            "coverage_insufficient", "nothing was exported for this pillar"
        )
    if source_entity_ref is not None and not c.ENTITY_REF_RE.fullmatch(
        source_entity_ref
    ):
        raise LegendError("invalid_request", "entity must be an opaque entity ref")
    deps = []
    for ref, kind in dependencies:
        if not c.VERSION_REF_RE.fullmatch(ref) or kind not in c.ENTITY_KINDS:
            raise LegendError(
                "invalid_request", "dependency must be an opaque version ref and kind"
            )
        deps.append({"version_ref": ref, "kind": kind})
    document: dict[str, Any] = {
        "schema_version": c.SCHEMA_VERSION,
        "manifest_scheme": c.MANIFEST_SCHEME,
        **pillar.wire_identity(),
        "pillar_fingerprint": pillar.fingerprint,
        "exporter": {
            "exporter_id": c.EXPORTER_ID,
            "exporter_version": c.EXPORTER_VERSION,
        },
        "representations": [
            {
                "ordinal": i,
                "representation_kind": rep.kind,
                "digest": "sha256:" + hashlib.sha256(rep.data).hexdigest(),
                "byte_length": len(rep.data),
            }
            for i, rep in enumerate(pillar.representations)
        ],
        "ordering": pillar.ordering,
        "dependencies": deps,
        "coverage": coverage_for(pillar, identity_verification_ref),
        "identity_digests_reproduced": identity_reproduced(
            pillar, identity_verification_ref
        ),
        "backend_keyed": pillar.backend_keyed,
        "identity_verification_ref": identity_verification_ref,
        "rederivation": {
            "source_entity_ref": source_entity_ref,
            "skipped_file_count": pillar.skipped,
        },
    }
    # wire_identity() carries "fingerprint"; the manifest names it pillar_fingerprint.
    document.pop("fingerprint")
    return RecoveryManifest(document, manifest_fingerprint(document))
