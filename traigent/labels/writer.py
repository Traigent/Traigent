"""Label writer: verifies the signed binding chain, then seals.

``LabelWriter`` refuses to construct unless the project public key chains to the
PINNED root (see ``trust_anchor``), the binding list is not rolled back, nothing in the
chain is revoked and every ``not_before`` has passed. The writer holds only the project
PUBLIC key: it cannot read what it wrote.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ._codec import LabelCryptoError, unb64u
from .bindings import verify_chain
from .envelope import seal_label
from .trust_anchor import TrustAnchor


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class LabelWriter:
    def __init__(
        self,
        anchor: TrustAnchor,
        *,
        root_public_key: bytes,
        admin_binding: dict[str, Any],
        project_binding: dict[str, Any],
        binding_list: dict[str, Any],
        retained_sequence: int,
        now: str | None = None,
    ) -> None:
        verify_chain(
            anchor.root_fingerprint,
            root_public_key,
            admin_binding,
            project_binding,
            None,
            tenant_ref=anchor.tenant_ref,
            project_ref=anchor.project_ref,
            binding_list=binding_list,
            retained_sequence=retained_sequence,
            now=now or _utc_now(),
        )
        if project_binding["epoch"] is None or project_binding["epoch"] < 1:
            raise LabelCryptoError("project binding has no epoch")
        self._anchor = anchor
        self._pk = unb64u(project_binding["public_key"])
        self._epoch: int = project_binding["epoch"]
        self.list_sequence: int = binding_list["sequence"]

    @property
    def epoch(self) -> int:
        return self._epoch

    def seal(
        self,
        text: str,
        *,
        purpose: str,
        object_ref: str,
        base_revision: int,
        ikm_e: bytes | None = None,
    ) -> dict[str, Any]:
        """Seal ``text`` for a write of ``object_ref`` at ``base_revision + 1``."""
        if base_revision < 0:
            raise LabelCryptoError("base_revision must be at least 0")
        return seal_label(
            self._pk,
            text,
            tenant_ref=self._anchor.tenant_ref,
            project_ref=self._anchor.project_ref,
            purpose=purpose,
            object_ref=object_ref,
            key_epoch=self._epoch,
            revision=base_revision + 1,
            ikm_e=ikm_e,
        )
