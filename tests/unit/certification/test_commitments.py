from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle

import pytest

import traigent.certification as certification
from traigent.certification import build_client_evidence_manifest
from traigent.certification.commitments import (
    ClientEvidenceOpening,
    _compute_slot_commitment,
)
from traigent.certification.canonical import canonicalize_artifact_document


def test_private_known_answer() -> None:
    canonical = canonicalize_artifact_document({"alpha": "beta", "n": 1}).encode()
    assert (
        _compute_slot_commitment(
            blind=bytes(range(32)),
            artifact_kind="evaluator",
            canonical_bytes=canonical,
        )
        == "sha256:1229bb640ae4b62da3e6a7ac66bf935e38b3b75fdb0b494b9402d336816f9e14"
    )


def test_opening_is_opaque_and_rejects_state_export() -> None:
    opening = ClientEvidenceOpening({"secret": "client-only"}, bytes(range(32)))
    assert not dataclasses.is_dataclass(opening)
    assert not hasattr(opening, "artifact_document")
    assert not hasattr(opening, "blind")
    assert "client-only" not in repr(opening)
    assert bytes(range(32)).hex() not in repr(opening)
    with pytest.raises(TypeError):
        dataclasses.asdict(opening)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(opening)
    with pytest.raises(TypeError, match="cannot be serialized"):
        copy.copy(opening)
    with pytest.raises(TypeError, match="cannot be serialized"):
        copy.deepcopy(opening)
    with pytest.raises(TypeError, match="cannot be serialized"):
        opening.__getstate__()


def test_public_surface_has_no_raw_blind_or_compatibility_api() -> None:
    for name in (
        "compute_slot_commitment",
        "slot_commitment_digest",
        "generate_blind",
        "new_blind",
        "build_manifest",
        "CommitmentOpening",
    ):
        assert not hasattr(certification, name)
    assert list(inspect.signature(build_client_evidence_manifest).parameters) == [
        "agent_revision",
        "evaluation_dataset",
        "evaluator",
        "build_process_evidence",
    ]
