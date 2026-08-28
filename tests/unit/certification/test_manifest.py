from __future__ import annotations

import hashlib

import pytest

from traigent.certification import (
    ClientEvidenceManifest,
    ClientEvidenceSlot,
    build_client_evidence_manifest,
    compute_manifest_root,
    serialize_manifest,
)
from traigent.certification.manifest import _local_manifest_dict
from traigent.utils import fp2

_KINDS = (
    "agent_revision",
    "evaluation_dataset",
    "evaluator",
    "build_process_evidence",
)


def _documents() -> tuple[dict[str, str], ...]:
    return tuple({"kind": kind} for kind in _KINDS)


def _fixed_manifest() -> ClientEvidenceManifest:
    digest = "sha256:" + "a" * 64
    slots = {
        kind: ClientEvidenceSlot(kind, "sha256_secret_blinded_v1", digest)
        for kind in _KINDS
    }
    return ClientEvidenceManifest(
        agent_revision=slots["agent_revision"],
        evaluation_dataset=slots["evaluation_dataset"],
        evaluator=slots["evaluator"],
        build_process_evidence=slots["build_process_evidence"],
    )


def test_backend_binding_is_minimal_and_defensive() -> None:
    result = build_client_evidence_manifest(*_documents())
    binding = result.manifest_dict()
    assert set(binding) == {"manifest_root_digest", "commitment_scheme"}
    assert binding["manifest_root_digest"] == compute_manifest_root(result.manifest)
    assert binding["commitment_scheme"] == "sha256_secret_blinded_v1"
    binding["manifest_root_digest"] = "sha256:" + "b" * 64
    assert (
        result.manifest_dict()["manifest_root_digest"]
        != binding["manifest_root_digest"]
    )
    assert result.manifest.backend_binding == result.manifest.to_backend_binding()
    assert result.backend_binding == result.to_backend_binding()


def test_invalid_backend_bindings_and_local_manifest_fail_closed() -> None:
    local = _local_manifest_dict(_fixed_manifest())
    with pytest.raises(TypeError):
        serialize_manifest(local)

    binding = serialize_manifest(_fixed_manifest())
    for bad in (
        {key: value for key, value in binding.items() if key != "commitment_scheme"},
        {**binding, "unknown": "rejected"},
        {**binding, "manifest_root_digest": "not-a-digest"},
    ):
        with pytest.raises((TypeError, ValueError)):
            serialize_manifest(bad)


def test_current_schema_root_domain_known_answer() -> None:
    manifest = _fixed_manifest()
    canonical = fp2.canonicalize(_local_manifest_dict(manifest)).encode("utf-8")
    expected = (
        "sha256:"
        + hashlib.sha256(
            b"traigent.agent_certificate.client_evidence_manifest_root.v1"
            + b"\x00"
            + canonical
        ).hexdigest()
    )
    assert compute_manifest_root(manifest) == expected
    assert compute_manifest_root(manifest) != fp2.digest(manifest.to_dict())
