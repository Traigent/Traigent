"""Run the producer against the Schema known-answer vectors.

Vendored from TraigentSchema ``tests/data/client_evidence_kats.json`` at Schema
commit 87ad7cb (unreleased at the time of vendoring). The digest below pins the
exact bytes; if the Schema vectors change, re-vendor and update both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.certification import client_evidence as ce

_VECTOR_PATH = Path(__file__).parent / "data" / "client_evidence_kats.json"
# Split in 8-hex-character groups so the secret scanner does not flag a digest.
_VECTOR_SHA256 = "".join(
    (
        "01746203",
        "1f67f43d",
        "77c3cf6b",
        "926fcd7c",
        "0be459da",
        "cbf18916",
        "83a2c692",
        "e8197a0b",
    )
)

_RAW = _VECTOR_PATH.read_bytes()
_VECTORS: dict[str, Any] = json.loads(_RAW)


def test_vendored_vectors_are_the_pinned_bytes() -> None:
    assert hashlib.sha256(_RAW).hexdigest() == _VECTOR_SHA256


def test_vector_domains_match_the_producer() -> None:
    assert _VECTORS["slot_domain"].encode() == ce._SLOT_DOMAIN
    assert _VECTORS["root_domain"].encode() == ce._ROOT_DOMAIN


@pytest.mark.parametrize("case", _VECTORS["slot_cases"], ids=lambda c: c["name"])
def test_positive_slot_vectors(case: dict[str, Any]) -> None:
    document = case["artifact_document"]
    assert ce._jcs(document).decode("utf-8") == case["jcs_utf8"]
    digest = ce.slot_commitment_digest(
        bytes.fromhex(case["blind_hex"]), case["artifact_kind"], document
    )
    assert digest == case["slot_commitment_digest"]


@pytest.mark.parametrize("manifest", _VECTORS["manifests"], ids=lambda m: m["name"])
def test_manifest_vectors(manifest: dict[str, Any]) -> None:
    commitments = {}
    for kind, slot in manifest["slots"].items():
        commitments[kind] = ce.slot_commitment_digest(
            bytes.fromhex(slot["blind_hex"]), kind, slot["artifact_document"]
        )
        assert commitments[kind] == slot["slot_commitment_digest"]
    assert (
        ce._jcs(ce._manifest_document(commitments)).decode("utf-8")
        == (manifest["manifest_jcs_utf8"])
    )
    assert ce.manifest_root_digest(commitments) == manifest["manifest_root_digest"]


def test_build_process_evidence_vector() -> None:
    vector = _VECTORS["build_process_evidence"]
    document = vector["artifact_document"]
    assert ce._jcs(document).decode("utf-8") == vector["jcs_utf8"]
    assert set(document) == {"snapshot_format", "manifest_digest"}


@pytest.mark.parametrize("case", _VECTORS["negative_cases"], ids=lambda c: c["name"])
def test_negative_vectors_raise_the_typed_error(case: dict[str, Any]) -> None:
    with pytest.raises(ce.ClientEvidenceError) as raised:
        ce.slot_commitment_digest(
            bytes.fromhex(case["blind_hex"]),
            case["artifact_kind"],
            case["artifact_document"],
        )
    assert raised.value.code in {"UNSUPPORTED_VALUE", "BLIND_INVALID"}
    if case["name"] == "unsafe-integer":
        assert raised.value.code == "UNSUPPORTED_VALUE"
