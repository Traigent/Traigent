"""fp2 conformance gate for the ported `traigent.utils.fp2` module (ALR-1301).

ALR-1301's scope is local computation plus conformance, with no Backend
involvement: every agent-lifecycle Route 1-4 request/response schema in
TraigentSchema has zero `fp2`/`afp2`/`dfp2o`/`efp2`/`cfp2`/`ArtifactVersion`
fields today (only a forbid-list mentions the names), and all four artifact
version slots on Route 5 are unconditional `state: "UNKNOWN", ref: null`. So
this file's whole job is: does the ported algorithm
(`traigent/utils/fp2.py`) agree, byte-for-byte, with the reference
implementation that generated the vendored corpus below?

Corpus provenance: `tests/unit/utils/fixtures/fp2/agent_lifecycle_cases.json` is
vendored byte-for-byte from TraigentSchema `tests/data/fp2/agent_lifecycle_cases.json`
at commit `fbf8d734f865073f9ce52744377516c7a94a8295` (`origin/develop`,
2026-08-10) -- see the sibling `SOURCE.md` for the exact provenance and a
drift-detection recipe. Lives under `fixtures/`, not `data/`: this repo's
`.gitignore` has a blanket `data/` rule for local result dirs, which would
silently untrack anything placed there. The corpus itself, its "shared"-section indirection,
and the runtime-parity story it encodes are TraigentSchema's; this file is a
straight adaptation of TraigentSchema's own
`tests/test_fp2_agent_lifecycle_parity.py` against the ported module instead
of the reference one, so a reader comparing the two files line-by-line
should find them structurally identical.

Of the 18 cases: `dfp2o` and `cfp2` are cross-runtime (must match the JS SDK
byte-for-byte -- this file cannot prove that alone, only that this SDK's
output equals the pinned digest the JS SDK is checked against separately).
`afp2` and `efp2` are within-runtime-only and carry a mandatory `runtime`
discriminator; `test_afp2_efp2_cross_runtime_digests_are_never_equal` below
asserts they are NEVER equal across runtimes -- a cross-runtime match there
is a bug, not success. `fail_closed` cases must raise
`Fp2UnsupportedValue`, never produce a partial digest. The
`preserve_unknown` case demonstrates an unrecognized manifest field survives
byte-for-byte rather than being dropped.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from collections.abc import Callable

import pytest

from traigent.utils.fp2 import Fp2UnsupportedValue, canonicalize, digest

CORPUS = (
    Path(__file__).resolve().parent / "fixtures" / "fp2" / "agent_lifecycle_cases.json"
)

# Same substitution convention as TraigentSchema's own conformance suite:
# markers let JSON express values it cannot natively hold, substituted here
# with Python's native equivalent. Only the two markers this corpus actually
# uses are declared.
_MARKERS: dict[str, Any] = {
    "@@NAN@@": float("nan"),
    "@@LONE_SURROGATE@@": "a\ud800b",
}

_VALID_SCOPES = {
    "cross_runtime",
    "within_runtime_only",
    "preserve_unknown",
    "fail_closed",
}
_WITHIN_RUNTIME_FAMILIES = (
    "agent_basic",
    "agent_bound_state",
    "evaluator_source",
    "evaluator_external",
)


def _resolve_shared_refs(node: Any, shared: dict[str, Any]) -> Any:
    """Expand {"$shared": "<name>"} pointers against the corpus's top-level
    "shared" section, exactly as TraigentSchema's own loader does, so every
    test below sees a plain, fully-inlined manifest and never has to know
    shared fragments exist."""
    if isinstance(node, dict):
        if set(node) == {"$shared"}:
            return _resolve_shared_refs(copy.deepcopy(shared[node["$shared"]]), shared)
        return {key: _resolve_shared_refs(item, shared) for key, item in node.items()}
    if isinstance(node, list):
        return [_resolve_shared_refs(item, shared) for item in node]
    return node


def _cases() -> list[dict[str, Any]]:
    """Load the vendored corpus and resolve its indirection before returning
    cases. Two cases sharing an "expected_ref" are asserting the SAME
    expected output by design (that shared equality IS the convergence claim
    under test), so "expected_ref" is expanded into literal
    "canonical"/"digest" fields here rather than left for every test to
    resolve individually."""
    doc = json.loads(CORPUS.read_text(encoding="utf-8"))
    shared = doc.get("shared", {})

    resolved = []
    for case in doc["cases"]:
        case = dict(case)
        case["value"] = _resolve_shared_refs(case["value"], shared)
        expected_ref = case.pop("expected_ref", None)
        if expected_ref is not None:
            expected = shared[expected_ref]
            case["canonical"] = expected["canonical"]
            case["digest"] = expected["digest"]
        resolved.append(case)
    return resolved


def _substitute(value: Any) -> Any:
    if isinstance(value, str) and value in _MARKERS:
        return _MARKERS[value]
    if isinstance(value, list):
        return [_substitute(item) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item) for key, item in value.items()}
    return value


def _ok_cases() -> list[dict[str, Any]]:
    return [case for case in _cases() if case["expect"] == "ok"]


def _unsupported_cases() -> list[dict[str, Any]]:
    return [case for case in _cases() if case["expect"] == "unsupported"]


def _cases_by_runtime_for_family(family: str) -> dict[str, dict[str, Any]]:
    return {case["runtime"]: case for case in _cases() if case.get("family") == family}


def _case_named(name: str) -> dict[str, Any]:
    return next(c for c in _cases() if c["name"] == name)


def _assert_no_offenders(offenders: list[str], message: str) -> None:
    assert not offenders, f"{message}: {offenders}"


def _assert_fails_closed(value: Any) -> None:
    """Both fp2 entry points must refuse a manifest they cannot represent."""
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(value)
    with pytest.raises(Fp2UnsupportedValue):
        digest(value)


def test_vendored_corpus_file_exists_and_is_non_trivial() -> None:
    cases = _cases()
    assert len(cases) == 18, (
        "the vendored corpus drifted from the 18 cases this story pinned -- "
        "re-vendor from TraigentSchema and update SOURCE.md"
    )


def test_case_names_are_unique() -> None:
    names = [case["name"] for case in _cases()]
    assert len(names) == len(set(names)), "duplicate case names make failures ambiguous"


def test_every_case_declares_a_recognized_scope() -> None:
    offenders = [
        case["name"] for case in _cases() if case.get("scope") not in _VALID_SCOPES
    ]
    _assert_no_offenders(offenders, "cases with an unrecognized scope")


def test_every_fp2_kind_used_by_the_lifecycle_record_is_represented() -> None:
    kinds = {case["kind"] for case in _cases()}
    assert kinds == {"afp2", "dfp2o", "efp2", "cfp2"}


def test_case_kind_matches_the_manifest_kind_property() -> None:
    offenders = [
        case["name"] for case in _cases() if case["value"].get("kind") != case["kind"]
    ]
    _assert_no_offenders(offenders, "case.kind disagrees with value.kind")


@pytest.mark.parametrize("case", _ok_cases(), ids=lambda c: c["name"])
def test_ok_case_canonicalizes_and_digests_as_recorded(case: dict[str, Any]) -> None:
    value = _substitute(case["value"])
    assert canonicalize(value) == case["canonical"]
    assert digest(value) == case["digest"]


@pytest.mark.parametrize("case", _unsupported_cases(), ids=lambda c: c["name"])
def test_unsupported_case_fails_closed_on_both_entry_points(
    case: dict[str, Any],
) -> None:
    _assert_fails_closed(_substitute(case["value"]))


def test_dfp2o_and_cfp2_never_carry_a_runtime_discriminator() -> None:
    """dfp2o/cfp2 are required to match across languages; fp2.md: they
    'carry no such field' -- a runtime token would break exactly the parity
    they exist to provide."""
    offenders = [
        case["name"]
        for case in _cases()
        if case["kind"] in {"dfp2o", "cfp2"} and "runtime" in case["value"]
    ]
    _assert_no_offenders(offenders, "dfp2o/cfp2 cases smuggling a runtime scope")


def test_afp2_and_efp2_always_carry_a_runtime_discriminator() -> None:
    offenders = [
        case["name"]
        for case in _cases()
        if case["kind"] in {"afp2", "efp2"}
        and case["value"].get("runtime") not in ("python", "javascript")
    ]
    _assert_no_offenders(
        offenders, "afp2/efp2 cases with a missing/malformed runtime scope"
    )


# ---------------------------------------------------------------------------
# dfp2o is byte-equal cross-runtime for a canonical dataset object; cfp2 is
# byte-equal cross-runtime for a canonical configuration space.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["dataset_convergence", "config_convergence"])
def test_cross_runtime_scope_kinds_converge_regardless_of_construction_order(
    family: str,
) -> None:
    """Two manifests built with different key insertion order -- standing in
    for two independently-written SDK manifest builders -- must canonicalize
    to byte-identical output. A canonicalizer that leaked construction order
    (e.g. iterating insertion order instead of sorting) would still pass
    every single-value case and only fail here."""
    variants = [case for case in _cases() if case.get("family") == family]
    assert len(variants) >= 2, (
        f"{family} needs at least two construction-order variants"
    )

    canonical_texts = {canonicalize(case["value"]) for case in variants}
    digests = {digest(case["value"]) for case in variants}

    assert len(canonical_texts) == 1, (
        f"{family} variants diverged in canonical text: {canonical_texts}"
    )
    assert len(digests) == 1, f"{family} variants diverged in digest: {digests}"
    assert {case["digest"] for case in variants} == digests, (
        "recorded digest drifted from computed digest"
    )


def test_dfp2o_row_order_is_significant_not_convergent() -> None:
    """The deliberate non-convergent case: row order changes the digest
    (fp2.md: 'Row order is significant... An order-independent digest would
    declare those two runs comparable')."""
    ordered = _case_named("dfp2o_dataset_convergence_input_first")
    reversed_ = _case_named("dfp2o_dataset_row_order_reversed_differs")

    assert digest(ordered["value"]) != digest(reversed_["value"])


# ---------------------------------------------------------------------------
# afp2 and efp2 are deterministic only within each runtime; no cross-runtime
# claim. A cross-runtime match for either is a bug, not success.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", _WITHIN_RUNTIME_FAMILIES)
def test_afp2_efp2_are_deterministic_within_one_runtime(family: str) -> None:
    by_runtime = _cases_by_runtime_for_family(family)
    assert set(by_runtime) == {"python", "javascript"}

    for case in by_runtime.values():
        assert digest(case["value"]) == digest(case["value"]) == case["digest"]


@pytest.mark.parametrize("family", _WITHIN_RUNTIME_FAMILIES)
def test_afp2_efp2_cross_runtime_digests_are_never_equal(family: str) -> None:
    """THE fp2 scope negative. A cross-runtime digest match here is a bug:
    it means the `runtime` discriminator stopped doing its job."""
    by_runtime = _cases_by_runtime_for_family(family)
    python_case = by_runtime["python"]
    javascript_case = by_runtime["javascript"]

    # Guard the test itself: the two sides must share identical non-runtime
    # content, or a digest mismatch would prove nothing about runtime scope.
    python_content = {k: v for k, v in python_case["value"].items() if k != "runtime"}
    javascript_content = {
        k: v for k, v in javascript_case["value"].items() if k != "runtime"
    }
    assert python_content == javascript_content, (
        f"{family} python/javascript variants must share identical non-runtime content "
        "for this to be a real scope test rather than an accidental digest mismatch"
    )

    assert digest(python_case["value"]) != digest(javascript_case["value"])


# ---------------------------------------------------------------------------
# Unknown fp2 fields preserve or fail closed.
# ---------------------------------------------------------------------------


def test_unknown_manifest_field_is_preserved_not_dropped() -> None:
    case = next(c for c in _cases() if c["scope"] == "preserve_unknown")
    canonical = canonicalize(case["value"])

    assert canonical == case["canonical"]
    # Reparse the canonical text: the field none of fp2's four algorithms
    # name must survive byte-for-byte, not merely appear as a substring.
    reparsed = json.loads(canonical)
    assert reparsed["manifest_note"] == case["value"]["manifest_note"]


def test_unsupported_value_fails_the_whole_manifest_closed_across_all_kinds() -> None:
    """The other half of the preserve-or-fail-closed rule: what fp2 cannot
    represent it refuses whole -- never a digest with some fields silently
    dropped -- for every one of the four manifest kinds, not just one."""
    fail_closed_cases = [c for c in _cases() if c["scope"] == "fail_closed"]
    assert {c["kind"] for c in fail_closed_cases} == {"afp2", "dfp2o", "efp2", "cfp2"}

    for case in fail_closed_cases:
        _assert_fails_closed(_substitute(case["value"]))


# ---------------------------------------------------------------------------
# A canonical input mutation changes the digest and turns the corpus red.
# ---------------------------------------------------------------------------


def _mutate_dataset_row(value: dict[str, Any]) -> None:
    value["rows"][0]["expected"] = "5"  # was "4"


def _mutate_configuration_space(value: dict[str, Any]) -> None:
    value["space"]["temperature"] = [0.2, 0.9]  # was [0.1, 0.9]


def _mutate_agent_source_text(value: dict[str, Any]) -> None:
    value["source"] = value["source"].replace("1.0", "2.0")


# (case name, in-place mutator, also assert the canonical text changed too)
# The dataset-row case additionally pins canonical-text sensitivity, not just
# digest sensitivity; the other two only need the digest check.
_DIGEST_MUTATION_CASES: tuple[
    tuple[str, Callable[[dict[str, Any]], None], bool], ...
] = (
    ("dfp2o_dataset_convergence_input_first", _mutate_dataset_row, True),
    ("cfp2_config_convergence_model_first", _mutate_configuration_space, False),
    ("afp2_agent_basic_python", _mutate_agent_source_text, False),
)


@pytest.mark.parametrize(
    "case_name,mutate,also_assert_canonical_changed",
    _DIGEST_MUTATION_CASES,
    ids=[case_name for case_name, _, _ in _DIGEST_MUTATION_CASES],
)
def test_mutating_a_canonical_input_changes_the_digest(
    case_name: str,
    mutate: Callable[[dict[str, Any]], None],
    also_assert_canonical_changed: bool,
) -> None:
    case = _case_named(case_name)
    mutated = copy.deepcopy(case["value"])
    mutate(mutated)

    assert digest(mutated) != case["digest"]
    if also_assert_canonical_changed:
        assert canonicalize(mutated) != case["canonical"]
