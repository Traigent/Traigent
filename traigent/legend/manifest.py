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


def coverage_for(pillar: Pillar) -> dict[str, Any]:
    """Coverage per kind: ``covered`` and ``missing`` partition ``REQUIRED_BY_KIND[kind]``.

    ``complete`` iff ``missing`` is empty. An agent/evaluator capture that did not
    reproduce the identity digests (INV-17), or skipped files, cannot be complete: the
    identity-bearing component moves to ``missing``.
    """
    required = c.REQUIRED_BY_KIND[pillar.kind]
    covered = [x for x in required if x in pillar.covered]
    identity = c.IDENTITY_COMPONENT_BY_KIND.get(pillar.kind)
    if identity is not None and not (pillar.reproduced and pillar.skipped == 0):
        covered = [x for x in covered if x != identity]
    missing = [x for x in required if x not in covered]
    return {
        "status": "partial" if missing else "complete",
        "covered": covered,
        "missing": missing,
    }


def build_recovery_manifest(
    pillar: Pillar,
    *,
    dependencies: Sequence[tuple[str, str]] = (),
    source_entity_ref: str | None = None,
) -> RecoveryManifest:
    """Build the manifest for ``pillar`` (``dependencies``: ``(version_ref, kind)`` pairs)."""
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
        "coverage": coverage_for(pillar),
        "identity_digests_reproduced": bool(pillar.reproduced and pillar.skipped == 0),
        "rederivation": {
            "source_entity_ref": source_entity_ref,
            "skipped_file_count": pillar.skipped,
        },
    }
    # wire_identity() carries "fingerprint"; the manifest names it pillar_fingerprint.
    document.pop("fingerprint")
    return RecoveryManifest(document, manifest_fingerprint(document))
