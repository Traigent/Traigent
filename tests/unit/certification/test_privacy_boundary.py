from __future__ import annotations

import copy
import dataclasses
import json
import pickle

import pytest

from traigent.certification import build_client_evidence_manifest, compute_manifest_root
from traigent.certification import commitments

_KINDS = (
    "agent_revision",
    "evaluation_dataset",
    "evaluator",
    "build_process_evidence",
)


def _documents() -> list[dict[str, object]]:
    return [{"secret": f"client-only-{kind}", "kind": kind} for kind in _KINDS]


def _walk(value: object):
    yield value
    if type(value) is dict:
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif type(value) is list:
        for item in value:
            yield from _walk(item)


def test_manifest_result_and_root_are_content_free(monkeypatch) -> None:
    documents = _documents()
    blinds = tuple(bytes([index]) * 32 for index in range(1, 5))
    generated = iter(blinds)
    monkeypatch.setattr(
        commitments.secrets,
        "token_bytes",
        lambda size: next(generated),
    )
    result = build_client_evidence_manifest(*documents)
    wire = result.manifest_dict()
    root = compute_manifest_root(result.manifest)
    serialized = repr(wire)
    secret_values = [document["secret"] for document in documents]
    assert all(value not in serialized for value in secret_values)
    assert all(
        forbidden not in serialized
        for forbidden in ('"artifact_document"', '"blind"', '"filename"', '"path"')
    )
    assert all(value.hex() not in serialized for value in blinds)
    assert all(value not in root for value in secret_values)
    assert all(value.hex() not in root for value in blinds)
    assert all(value not in repr(result) for value in secret_values)
    assert all(value.hex() not in repr(result) for value in blinds)
    assert set(wire) == {"manifest_root_digest", "commitment_scheme"}
    assert wire["manifest_root_digest"] == root
    assert wire["commitment_scheme"] == "sha256_secret_blinded_v1"
    assert all(
        slot.slot_commitment_digest not in repr(wire)
        for slot in (
            result.manifest.agent_revision,
            result.manifest.evaluation_dataset,
            result.manifest.evaluator,
            result.manifest.build_process_evidence,
        )
    )
    assert all(type(item) in (str, dict) for item in _walk(wire))


def test_caller_mutation_does_not_change_existing_build() -> None:
    documents = _documents()
    result = build_client_evidence_manifest(*documents)
    before = result.manifest_dict()
    before_root = compute_manifest_root(result.manifest)
    documents[0]["secret"] = "mutated-after-build"
    documents[0]["new"] = "also-not-bound"
    assert result.manifest_dict() == before
    assert compute_manifest_root(result.manifest) == before_root


def test_two_builds_over_same_documents_use_independent_blinds() -> None:
    documents = _documents()
    first = build_client_evidence_manifest(*documents).manifest_dict()
    second = build_client_evidence_manifest(*documents).manifest_dict()
    assert first != second
    assert first["manifest_root_digest"] != second["manifest_root_digest"]


def test_client_local_manifest_and_slots_reject_generic_serialization() -> None:
    result = build_client_evidence_manifest(*_documents())
    local_values = (
        result.manifest,
        result.manifest.agent_revision,
    )
    for local in local_values:
        with pytest.raises(TypeError):
            vars(local)
        with pytest.raises(TypeError):
            json.dumps(local)
        with pytest.raises(TypeError):
            pickle.dumps(local)
        with pytest.raises(TypeError):
            copy.copy(local)
        with pytest.raises(TypeError):
            copy.deepcopy(local)
    with pytest.raises(TypeError):
        dataclasses.asdict(result.manifest)


def test_duplicate_blind_source_is_retried_and_eventual_collision_fails(
    monkeypatch,
) -> None:
    values = iter([b"a" * 32, b"a" * 32, b"b" * 32, b"c" * 32, b"d" * 32])
    monkeypatch.setattr(commitments.secrets, "token_bytes", lambda size: next(values))
    result = build_client_evidence_manifest(*_documents())
    assert result.manifest_dict()

    monkeypatch.setattr(commitments.secrets, "token_bytes", lambda size: b"a" * 32)
    with pytest.raises(RuntimeError, match="could not generate unique"):
        build_client_evidence_manifest(*_documents())


def test_build_result_rejects_copy_and_pickle() -> None:
    result = build_client_evidence_manifest(*_documents())
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(result)
    with pytest.raises(TypeError, match="cannot be serialized"):
        copy.copy(result)
    with pytest.raises(TypeError, match="cannot be serialized"):
        copy.deepcopy(result)
