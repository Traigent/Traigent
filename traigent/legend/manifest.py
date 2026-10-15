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


def build_recovery_manifest(
    pillar: Pillar,
    *,
    dependencies: Sequence[tuple[str, str]] = (),
) -> RecoveryManifest:
    """Build the manifest for ``pillar`` (``dependencies``: ``(version_ref, kind)`` pairs)."""
    if not pillar.representations:
        raise LegendError(
            "coverage_insufficient", "nothing was exported for this pillar"
        )
    universe = c.COVERED_BY_KIND[pillar.kind]
    covered = [x for x in universe if x in pillar.covered]
    missing = [x for x in universe if x not in pillar.covered]
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
        "coverage": {
            "status": "partial" if missing else "complete",
            "covered": covered,
            "missing": missing,
        },
    }
    # wire_identity() carries "fingerprint"; the manifest names it pillar_fingerprint.
    document.pop("fingerprint")
    return RecoveryManifest(document, manifest_fingerprint(document))
