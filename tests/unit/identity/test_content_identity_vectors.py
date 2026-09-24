"""Content identity v1 conformance: EVERY section of the vendored vector file.

Mirrors TraigentSchema ``tests/test_example_identity.py`` ("Conformance
vectors") against the SDK port :mod:`traigent.identity.content_identity`, and
additionally drives the example vectors through the SDK's production
projection path (:func:`traigent.identity.examples.identify_evaluation_example`
on a real ``EvaluationExample``).

Schema-validation steps of the reference harness (jsonschema checks of the
agent/evaluator manifests) are not repeated here: the SDK does not vendor the
JSON Schemas. The digests and the certifiable/non-certifiable verdicts are.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest

from traigent.evaluators.base import EvaluationExample
from traigent.identity import content_identity as ci
from traigent.identity.examples import identify_evaluation_example
from traigent.identity.keys import ContentIdentityKeys

from .support import key_derivation as kd

FIXTURE = Path(__file__).parent / "fixtures" / "content_identity_v1_vectors.json"
EXPECTED_SHA256 = "7a75e6ab1316a015d1be91211b23e1ac3a0ce169ba29d6913e877856ce34b66e"
VECTORS: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))

#: Every top-level key of the vector file. A new section in a re-vendored file
#: fails test_every_section_is_exercised until a test is wired for it.
KNOWN_SECTIONS = {
    "spec",
    "note",
    "constants",
    "key_derivation",
    "examples",
    "relations",
    "projections",
    "multisets",
    "inclusion_proofs",
    "rejections",
    "json_text_accepted",
    "agent_builds",
    "evaluator_versions",
}
REJECTION_KINDS = {
    "json_text",
    "example_version",
    "multiset",
    "key_derivation",
    "example_input",
    "agent_build",
    "agent_build_certifiable",
    "evaluator_version",
}


def _keys_for(tenant: str) -> ci.TenantIdentityKeys:
    row = next(k for k in VECTORS["key_derivation"] if k["tenant"] == tenant)
    return kd.derive_tenant_keys(
        bytes.fromhex(row["tenant_master_hex"]), row["tenant_id"]
    )


def _grant_for(tenant: str) -> ContentIdentityKeys:
    """The SDK production key object, built the way a Backend grant arrives."""
    row = next(k for k in VECTORS["key_derivation"] if k["tenant"] == tenant)
    return ContentIdentityKeys.from_grant(
        {
            "tenant_id": row["tenant_id"],
            "kid": row["key_id"],
            "example_id_key": row["example_id_key_hex"],
            "example_version_key": row["example_version_key_hex"],
            "encoding": "hex",
        }
    )


KEYS_A = _keys_for("tenant_a")


def _field(case: dict[str, Any], name: str) -> Any:
    return case[name] if name in case else ci.ABSENT


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_master_key_derivation_is_not_shipped() -> None:
    """The SDK never holds a tenant master (ruling D1): no derivation ships."""
    for name in ("derive_tenant_keys", "hkdf_sha256", "hkdf_info"):
        assert not hasattr(ci, name), name


def test_vendored_file_is_unedited() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_vendored_file_matches_installed_schema_when_it_ships_one() -> None:
    try:
        resource = importlib.resources.files("traigent_schema").joinpath(
            "data/content_identity_v1_vectors.json"
        )
    except ModuleNotFoundError:
        pytest.skip("traigent_schema is not installed")
    if not resource.is_file():
        pytest.skip(
            "the installed traigent_schema predates content identity v1 "
            "(schema-pin bump pending)"
        )
    assert resource.read_bytes() == FIXTURE.read_bytes(), (
        "re-vendor tests/unit/identity/fixtures/content_identity_v1_vectors.json "
        "from the pinned TraigentSchema build (see fixtures/SOURCE.md)"
    )


def test_every_section_is_exercised() -> None:
    assert set(VECTORS) == KNOWN_SECTIONS
    assert {case["kind"] for case in VECTORS["rejections"]} == REJECTION_KINDS
    assert VECTORS["spec"] == ci.SCHEME
    assert len(VECTORS["examples"]) >= 20 and len(VECTORS["rejections"]) >= 10


def test_vector_constants_match_the_module() -> None:
    constants = VECTORS["constants"]
    assert constants["hkdf_salt_utf8"].encode() == ci.HKDF_SALT
    assert sorted(constants["reserved_metadata_keys"]) == sorted(
        ci.RESERVED_METADATA_KEYS
    )
    assert constants["domains"] == {
        "agent_build": ci.DOMAIN_AGENT_BUILD,
        "example_id": ci.DOMAIN_EXAMPLE_ID,
        "example_version": ci.DOMAIN_EXAMPLE_VERSION,
        "evaluator_version": ci.DOMAIN_EVALUATOR_VERSION,
        "key_id": ci.DOMAIN_KEY_ID,
        "multiset_leaf": ci.DOMAIN_MULTISET_LEAF,
        "public_input": ci.DOMAIN_PUBLIC_INPUT,
    }
    assert kd.hkdf_info("d", "t_1") == b"d\x00t_1"
    tenant_pattern = constants["tenant_id_pattern"]
    assert re.search(tenant_pattern, "tenant_0a0a0a0a")
    assert not re.search(tenant_pattern, "tenant_0a0a0a0a\n")


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["key_derivation"], ids=lambda c: c["tenant"])
def test_key_derivation_vectors(case: dict[str, Any]) -> None:
    keys = kd.derive_tenant_keys(
        bytes.fromhex(case["tenant_master_hex"]), case["tenant_id"]
    )
    assert keys.key_id == case["key_id"]
    assert keys.example_id_key.hex() == case["example_id_key_hex"]
    assert keys.example_version_key.hex() == case["example_version_key_hex"]
    # The Backend-grant path yields the identical primitive keys.
    grant = _grant_for(case["tenant"]).as_tenant_keys()
    assert grant == keys


def test_rfc5869_hkdf_known_answers() -> None:
    # RFC 5869 A.1 and A.3 (SHA-256).
    okm = kd.hkdf_sha256(
        bytes.fromhex("0b" * 22),
        salt=bytes.fromhex("000102030405060708090a0b0c"),
        info=bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
        length=42,
    )
    assert okm.hex() == (
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    )
    okm = kd.hkdf_sha256(bytes.fromhex("0b" * 22), salt=b"", info=b"", length=42)
    assert okm.hex() == (
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8"
    )


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["examples"], ids=lambda c: c["name"])
def test_example_vectors(case: dict[str, Any]) -> None:
    keys = _keys_for(case["tenant"])
    context = _field(case, "context")
    expected = _field(case, "expected")
    metadata = case.get("metadata")
    expect = case["expect"]
    payload = ci.example_id_payload(case["input"], context=context)
    assert ci.canonical_bytes(payload).decode("utf-8") == expect["id_payload_jcs"]
    example_id = ci.compute_example_id(keys, case["input"], context=context)
    assert example_id == expect["example_id"]
    version_payload = ci.example_version_payload(
        example_id, expected=expected, metadata=metadata
    )
    assert (
        ci.canonical_bytes(version_payload).decode("utf-8")
        == expect["version_payload_jcs"]
    )
    assert (
        ci.compute_example_version(
            keys, example_id, expected=expected, metadata=metadata
        )
        == expect["example_version"]
    )
    assert (
        ci.compute_public_input_digest(case["input"], context=context)
        == expect["public_input_digest"]
    )


@pytest.mark.parametrize("case", VECTORS["examples"], ids=lambda c: c["name"])
def test_example_vectors_through_the_sdk_example_path(case: dict[str, Any]) -> None:
    """The production path: a real EvaluationExample, projected by the SDK."""
    row_metadata: dict[str, Any] = dict(case.get("metadata") or {})
    if "context" in case:
        row_metadata["context"] = case["context"]
    for annotation in ("external_id", "source_ref", "supersedes"):
        if annotation in case:
            row_metadata[annotation] = case[annotation]
    example = EvaluationExample(
        input_data=case["input"],
        expected_output=case.get("expected"),
        metadata=row_metadata,
    )
    identity = identify_evaluation_example(example, _grant_for(case["tenant"]))
    assert identity.example_id == case["expect"]["example_id"]
    assert identity.example_version == case["expect"]["example_version"]
    if isinstance(case.get("external_id"), str):
        assert identity.external_id == case["external_id"]


@pytest.mark.parametrize(
    "case", VECTORS["relations"], ids=lambda c: f"{c['a']}~{c['b']}"
)
def test_relation_vectors(case: dict[str, Any]) -> None:
    by_name = {v["name"]: v["expect"] for v in VECTORS["examples"]}
    same = by_name[case["a"]][case["field"]] == by_name[case["b"]][case["field"]]
    assert same is (case["relation"] == "equal")


@pytest.mark.parametrize("case", VECTORS["projections"], ids=lambda c: c["name"])
def test_projection_vectors(case: dict[str, Any]) -> None:
    given = case["given"]
    projection = ci.project_sdk_example(
        given["input_data"],
        given.get("expected_output", ci.ABSENT),
        given.get("metadata"),
    )
    expect = case["expect"]
    assert projection.input == expect["input"]
    assert projection.context == expect.get("context", ci.ABSENT)
    assert projection.expected == expect.get("expected", ci.ABSENT)
    assert projection.metadata == expect.get("metadata")


@pytest.mark.parametrize("case", VECTORS["json_text_accepted"], ids=lambda c: c["name"])
def test_json_text_accepted_vectors(case: dict[str, Any]) -> None:
    parsed = ci.parse_strict_json(case["json_text"])
    assert (
        ci.compute_example_id(KEYS_A, parsed["input"]) == case["expect"]["example_id"]
    )
    if case["equals_example"]:
        by_name = {v["name"]: v["expect"] for v in VECTORS["examples"]}
        assert (
            case["expect"]["example_id"]
            == by_name[case["equals_example"]]["example_id"]
        )


def test_decimal_range_check_is_independent_of_the_decimal_context() -> None:
    import decimal

    with decimal.localcontext() as context:
        context.prec = 5
        for text in (
            "9007199254740991.0000000000001",
            "-9007199254740991.0000000000001",
        ):
            with pytest.raises(ci.ContentIdentityError):
                ci.parse_strict_json(text)
        assert ci.parse_strict_json("9007199254740990.9999999999999999999") == float(
            2**53 - 1
        )


# ---------------------------------------------------------------------------
# Multisets and inclusion proofs
# ---------------------------------------------------------------------------


def _multiset(case: dict[str, Any]) -> ci.MultisetRoot:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ci.ConflictingExampleVersionsWarning)
        return ci.compute_multiset_root(
            [tuple(item) for item in case["items"]], key_id=case.get("key_id")
        )


@pytest.mark.parametrize("case", VECTORS["multisets"], ids=lambda c: c["name"])
def test_multiset_vectors(case: dict[str, Any]) -> None:
    root = _multiset(case)
    expect = case["expect"]
    assert root.root == expect["root"]
    assert [[m.example_id, m.example_version, m.count] for m in root.members] == expect[
        "members"
    ]
    assert root.distinct_count == expect["distinct_count"]
    assert root.total_count == expect["total_count"]
    assert list(root.conflicting_example_ids) == expect["conflicting_example_ids"]
    assert [ci.multiset_leaf_data(m).hex() for m in root.members] == expect[
        "leaf_data_hex"
    ]


def test_multiset_vector_relations() -> None:
    roots = {m["name"]: m["expect"]["root"] for m in VECTORS["multisets"]}
    assert roots["order_a"] == roots["order_b"]
    assert roots["with_duplicate"] == roots["explicit_count"]
    assert roots["with_duplicate"] != roots["without_duplicate"]


@pytest.mark.parametrize(
    "case",
    VECTORS["inclusion_proofs"],
    ids=lambda c: f"{c['multiset']}-{c['leaf_index']}-{c.get('tamper', 'valid')}",
)
def test_inclusion_proof_vectors(case: dict[str, Any]) -> None:
    proof = ci.InclusionProof(
        member=ci.MultisetMember(*case["member"]),
        leaf_index=case["leaf_index"],
        tree_size=case["tree_size"],
        audit_path=tuple(case["audit_path"]),
        root=case["root"],
    )
    assert ci.verify_inclusion_proof(proof) is case["valid"]
    if case["valid"]:
        # The SDK also BUILDS the identical proof from the named multiset.
        source = next(m for m in VECTORS["multisets"] if m["name"] == case["multiset"])
        built = ci.build_inclusion_proof(_multiset(source), proof.member)
        assert built == proof


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["rejections"], ids=lambda c: c["name"])
def test_rejection_vectors(case: dict[str, Any]) -> None:
    kind = case["kind"]
    with pytest.raises(ci.ContentIdentityError):
        if kind == "json_text":
            ci.compute_example_id(
                KEYS_A, ci.parse_strict_json(case["json_text"])["input"]
            )
        elif kind == "example_version":
            ci.compute_example_version(
                KEYS_A,
                case["example_id"],
                expected=case["expected"],
                metadata=case["metadata"],
            )
        elif kind == "multiset":
            ci.compute_multiset_root([tuple(item) for item in case["items"]])
        elif kind == "key_derivation":
            kd.derive_tenant_keys(
                bytes.fromhex(case["tenant_master_hex"]), case["tenant_id"]
            )
        elif kind == "example_input":
            ci.compute_example_id(KEYS_A, case["input"])
        elif kind == "agent_build":
            ci.compute_agent_build_digest(case["manifest"])
        elif kind == "agent_build_certifiable":
            ci.compute_agent_build_digest(case["manifest"], certifiable=True)
        elif kind == "evaluator_version":
            ci.compute_evaluator_version_digest(case["manifest"])
        else:  # pragma: no cover - test_every_section_is_exercised guards this
            pytest.fail(f"unknown rejection kind {kind}")


# ---------------------------------------------------------------------------
# Agent builds and evaluator versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", VECTORS["agent_builds"], ids=lambda c: c["name"])
def test_agent_build_vectors(case: dict[str, Any]) -> None:
    manifest = case["manifest"]
    assert ci.compute_agent_build_digest(manifest) == case["expect"]["build_digest"]
    if case["expect"]["certifiable"]:
        assert (
            ci.compute_agent_build_digest(manifest, certifiable=True)
            == case["expect"]["build_digest"]
        )
    else:
        with pytest.raises(ci.ContentIdentityError):
            ci.compute_agent_build_digest(manifest, certifiable=True)


def test_every_behaviour_input_changes_the_build_digest() -> None:
    digests = {c["name"]: c["expect"]["build_digest"] for c in VECTORS["agent_builds"]}
    assert digests["full"] == digests["full_reordered"]
    for name in (
        "config_changed",
        "helper_changed",
        "prompt_changed",
        "tool_changed",
        "relabelled",
        "partial_coverage",
    ):
        assert digests[name] != digests["full"], name


@pytest.mark.parametrize("case", VECTORS["evaluator_versions"], ids=lambda c: c["name"])
def test_evaluator_version_vectors(case: dict[str, Any]) -> None:
    assert (
        ci.compute_evaluator_version_digest(case["manifest"])
        == case["expect"]["evaluator_version_digest"]
    )


def test_every_score_input_changes_the_evaluator_digest() -> None:
    digests = {
        c["name"]: c["expect"]["evaluator_version_digest"]
        for c in VECTORS["evaluator_versions"]
    }
    assert digests["base"] == digests["base_reordered_keys"]
    for name in (
        "orientation_changed",
        "weight_changed",
        "judge_model_changed",
        "judge_config_changed",
        "dependency_changed",
        "code_changed",
        "model_free",
        "no_helpers",
    ):
        assert digests[name] != digests["base"], name
    for name in ("model_free_threshold_changed", "model_free_helper_changed"):
        assert digests[name] != digests["model_free"], name
