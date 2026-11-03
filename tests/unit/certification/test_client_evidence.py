"""G1 client evidence producer: slots, root, blinds, custody, reconstruction."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from traigent_schema import fp2

from tests.unit.snapshot import helpers
from traigent.certification import client_evidence as ce
from traigent.snapshot import (
    EvaluatorInputs,
    ExecutionSnapshot,
    build_execution_snapshot,
    publish_snapshot,
)
from traigent.snapshot.backends import LocalDirStore


@pytest.fixture
def snapshot() -> ExecutionSnapshot:
    return build_execution_snapshot(
        dataset=[{"input": {"q": "a"}, "expected": "b"}],
        config={"temperature": 0.2},
        config_space={"model": ["x", "y"]},
        evaluator=EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalDirStore:
    return LocalDirStore(tmp_path / "store")


@pytest.fixture
def custody(tmp_path: Path) -> ce.LocalFileCustody:
    return ce.LocalFileCustody(tmp_path / "evidence")


def _expected_slot(blind: bytes, kind: str, document: dict[str, Any]) -> str:
    """Independent re-statement of the Schema formula (not the module under test)."""
    data = (
        b"traigent.cert.v0.slot_commitment.v1"
        + b"\x00"
        + kind.encode()
        + b"\x00"
        + str(fp2.canonicalize(document)).encode()
    )
    return "sha256:" + hmac.new(blind, data, hashlib.sha256).hexdigest()


def test_known_answer_slot_and_root() -> None:
    blind = bytes(range(32))
    document = {"snapshot_format": "tes1", "manifest_digest": "sha256:" + "ab" * 32}
    digest = ce.slot_commitment_digest(blind, "build_process_evidence", document)
    assert digest == _expected_slot(blind, "build_process_evidence", document)
    literal = (
        b"traigent.cert.v0.slot_commitment.v1\x00build_process_evidence\x00"
        b'{"manifest_digest":"sha256:' + b"ab" * 32 + b'","snapshot_format":"tes1"}'
    )
    assert digest == "sha256:" + hmac.new(blind, literal, hashlib.sha256).hexdigest()
    commitments = {
        kind: "sha256:" + f"{i:02x}" * 32 for i, kind in enumerate(ce.SLOT_KINDS)
    }
    manifest = {
        "schema_version": "traigent.certificate_client_evidence_manifest.v0",
        **{
            kind: {
                "artifact_kind": kind,
                "commitment_scheme": "sha256_secret_blinded_v1",
                "slot_commitment_digest": commitments[kind],
            }
            for kind in ce.SLOT_KINDS
        },
    }
    expected_root = (
        "sha256:"
        + hashlib.sha256(
            b"traigent.agent_certificate.client_evidence_manifest_root.v1\x00"
            + str(fp2.canonicalize(manifest)).encode()
        ).hexdigest()
    )
    assert ce.manifest_root_digest(commitments) == expected_root


def test_manifest_validates_against_schema_package(
    snapshot: ExecutionSnapshot, tmp_path: Path
) -> None:
    import importlib.resources as resources

    base = resources.files("traigent_schema") / "schemas" / "certification"
    schema = json.loads(
        (base / "certificate_client_evidence_manifest_v0_schema.json").read_text()
    )
    common = json.loads((base / "certification_common_v0_schema.json").read_text())
    registry_docs = {
        "certification_common_v0_schema.json": common,
        common.get("$id", ""): common,
    }
    evidence = ce.build_client_evidence(
        snapshot, custody=ce.LocalFileCustody(tmp_path / "e")
    )
    manifest = ce._manifest_document(evidence.commitments)
    from referencing import Registry, Resource

    registry = Registry().with_resources(
        [
            (uri, Resource.from_contents(doc))
            for uri, doc in registry_docs.items()
            if uri
        ]
        + [("certification_common_v0_schema.json", Resource.from_contents(common))]
    )
    jsonschema.Draft7Validator(schema, registry=registry).validate(manifest)


def test_documents_digest_only_and_build_slot_exact(
    snapshot: ExecutionSnapshot,
) -> None:
    docs = ce.build_slot_documents(snapshot)
    assert set(docs) == set(ce.SLOT_KINDS)
    assert docs["build_process_evidence"] == {
        "snapshot_format": "tes1",
        "manifest_digest": snapshot.manifest_digest,
    }
    blob = json.dumps(docs)
    for forbidden in ('"q"', "temperature", "0.2", "metric_a", "expected"):
        assert forbidden not in blob
    assert not any(isinstance(v, float) for d in docs.values() for v in d.values())


def test_float_rejected_recursively() -> None:
    cases: list[Any] = [
        {"a": 1.5},
        {"a": [{"b": 0.1}]},
        {"a": float("nan")},
        {"a": b"x"},
        {1: "x"},
    ]
    for bad in cases:
        with pytest.raises(ce.ClientEvidenceError) as caught:
            ce.slot_commitment_digest(bytes(32), "evaluator", bad)
        assert caught.value.code == "UNSUPPORTED_VALUE"


def test_blind_must_be_32_bytes() -> None:
    cases: list[Any] = [b"", bytes(31), bytes(33), "x" * 32]
    for bad in cases:
        with pytest.raises(ce.ClientEvidenceError) as caught:
            ce.slot_commitment_digest(bad, "evaluator", {"a": 1})
        assert caught.value.code == "BLIND_INVALID"


def test_blinds_independent_and_random(
    snapshot: ExecutionSnapshot, tmp_path: Path
) -> None:
    first = ce.build_client_evidence(
        snapshot, custody=ce.LocalFileCustody(tmp_path / "a")
    )
    second = ce.build_client_evidence(
        snapshot, custody=ce.LocalFileCustody(tmp_path / "b")
    )
    assert len(set(first.blinds.values())) == 4
    assert all(len(b) == 32 for b in first.blinds.values())
    assert set(first.blinds.values()).isdisjoint(second.blinds.values())
    assert first.root != second.root  # same snapshot, fresh blinds


def test_evidence_file_mode_and_atomic(
    snapshot: ExecutionSnapshot, custody: ce.LocalFileCustody, tmp_path: Path
) -> None:
    evidence = ce.build_client_evidence(snapshot, custody=custody)
    directory = tmp_path / "evidence"
    path = directory / f"{evidence.root_hex}.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert [p.name for p in directory.iterdir()] == [path.name]  # no temp left behind
    assert ce.ClientEvidence.from_record(path.read_bytes()).root == evidence.root


def test_persist_failure_raises_before_return(
    snapshot: ExecutionSnapshot, tmp_path: Path
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ce.ClientEvidenceError) as caught:
        ce.build_client_evidence(snapshot, custody=ce.LocalFileCustody(blocker / "sub"))
    assert caught.value.code == "EVIDENCE_PERSIST_FAILED"


def test_custom_custody_hook_used(snapshot: ExecutionSnapshot) -> None:
    saved: dict[str, bytes] = {}

    class Memory:
        def save(self, root_hex: str, record: bytes) -> None:
            saved[root_hex] = record

        def load(self, root_hex: str) -> bytes | None:
            return saved.get(root_hex)

    assert isinstance(Memory(), ce.BlindCustody)
    evidence = ce.build_client_evidence(snapshot, custody=Memory())
    assert set(saved) == {evidence.root_hex}


def test_reconstruct_happy_path_and_loss(
    snapshot: ExecutionSnapshot, store: LocalDirStore, custody: ce.LocalFileCustody
) -> None:
    publish_snapshot(store, snapshot)
    evidence = ce.build_client_evidence(snapshot, custody=custody)
    again = ce.reconstruct_and_compare(evidence.root, custody=custody, store=store)
    assert again.root == evidence.root and again.blinds == evidence.blinds
    (
        custody._directory / f"{evidence.root_hex}.json"
    ).unlink()  # loss of the evidence file
    with pytest.raises(ce.ClientEvidenceError) as caught:
        ce.reconstruct_and_compare(evidence.root, custody=custody, store=store)
    assert caught.value.code == "EVIDENCE_NOT_FOUND"


def test_reconstruct_refuses_tampered_record(
    snapshot: ExecutionSnapshot, store: LocalDirStore, custody: ce.LocalFileCustody
) -> None:
    publish_snapshot(store, snapshot)
    evidence = ce.build_client_evidence(snapshot, custody=custody)
    path = custody._directory / f"{evidence.root_hex}.json"
    record = json.loads(path.read_text())
    record["documents"]["evaluation_dataset"]["dataset_part_sha256"] = (
        "sha256:" + "0" * 64
    )
    path.write_text(json.dumps(record))
    with pytest.raises(ce.ClientEvidenceError) as caught:
        ce.reconstruct_and_compare(evidence.root, custody=custody, store=store)
    assert caught.value.code == "EVIDENCE_STALE_OR_CORRUPT"


def test_reconstruct_refuses_when_snapshot_missing_or_changed(
    snapshot: ExecutionSnapshot, store: LocalDirStore, custody: ce.LocalFileCustody
) -> None:
    evidence = ce.build_client_evidence(snapshot, custody=custody)  # never published
    with pytest.raises(ce.ClientEvidenceError) as caught:
        ce.reconstruct_and_compare(evidence.root, custody=custody, store=store)
    assert caught.value.code == "EVIDENCE_STALE_OR_CORRUPT"


def test_record_keeps_no_snapshot_content(
    snapshot: ExecutionSnapshot, custody: ce.LocalFileCustody
) -> None:
    evidence = ce.build_client_evidence(snapshot, custody=custody)
    record = evidence.to_record().decode()
    assert "temperature" not in record and "metric_a" not in record
    assert evidence.blinds["evaluator"].hex() not in repr(evidence)


def test_reconstruct_refuses_documents_not_derived_from_the_referenced_snapshot(
    snapshot: ExecutionSnapshot, store: LocalDirStore, custody: ce.LocalFileCustody
) -> None:
    other = build_execution_snapshot(
        dataset=[{"input": {"q": "zzz"}, "expected": "y"}],
        config={"temperature": 0.9},
        config_space={"model": ["x"]},
        evaluator=EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
    )
    publish_snapshot(store, other)
    # A self-consistent record (valid blinds, matching root) whose slots describe `snapshot`
    # but whose build slot points at the published `other`.
    documents = ce.build_slot_documents(snapshot)
    documents["build_process_evidence"] = ce.build_slot_documents(other)[
        "build_process_evidence"
    ]
    blinds = {kind: bytes([i + 1]) * 32 for i, kind in enumerate(ce.SLOT_KINDS)}
    commitments = {
        k: ce.slot_commitment_digest(blinds[k], k, documents[k]) for k in ce.SLOT_KINDS
    }
    forged = ce.ClientEvidence(
        documents,
        blinds,
        commitments,
        ce.manifest_root_digest(commitments),
        other.manifest_digest,
    )
    custody.save(forged.root_hex, forged.to_record())
    with pytest.raises(ce.ClientEvidenceError) as caught:
        ce.reconstruct_and_compare(forged.root, custody=custody, store=store)
    assert caught.value.code == "EVIDENCE_STALE_OR_CORRUPT"


def test_existing_loose_directory_is_tightened(
    snapshot: ExecutionSnapshot, tmp_path: Path
) -> None:
    directory = tmp_path / "loose"
    directory.mkdir(mode=0o755)
    os.chmod(directory, 0o755)
    ce.build_client_evidence(snapshot, custody=ce.LocalFileCustody(directory))
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
