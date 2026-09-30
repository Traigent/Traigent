"""SDK attribute contract vs the receiver's shared contract (vendored, hash-locked).

The vendored file is a byte copy of the Schema repo's
``traigent_schema/data/observability/otel_attribute_contract_v1.json``
(TraigentSchema otel-contract branch, commit 700bbb57a).  A hash proves
agreement with that snapshot, not correctness of any consumer: the SDK CONSUMES
the contract (egress set, content-mode vectors, usage rules), and the tests
below drive the SDK from the contract's own vectors so they fail if the SDK
diverges.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from traigent.observability.otel import contract as C

PATH = (
    Path(__file__).parents[3] / "fixtures/observability/otel_attribute_contract_v1.json"
)
PACKAGED = Path(C.__file__).with_name("otel_attribute_contract_v1.json")
LOCKED_SHA256 = "ca27ffe64b71af0a2dfa1f35343a53448ae080beeb93ed17cca2d2e6fab4bc91"
CONTRACT = json.loads(PATH.read_text())

# SDK-only keys the receiver snapshot does not know.  Each must be a reviewed
# integration item (the receiver counts unknown keys as dropped, never fails).
SDK_ONLY_ALLOWED: set[str] = set()


def test_vendored_contract_is_hash_locked():
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == LOCKED_SHA256


def test_content_mode_declaration_matches():
    cm = CONTRACT["content_mode"]
    assert cm["resource_attribute"] == C.CONTENT_MODE_ATTRIBUTE
    assert tuple(cm["values"]) == C.CONTENT_MODES
    assert cm["declaration_grants_permission"] is False


def test_every_sdk_allowlisted_key_is_known_to_the_receiver_or_reviewed():
    known = set(CONTRACT["metadata_allowlist"]["attributes"])
    for attrs in CONTRACT["usage_classes"]["attributes"].values():
        known |= set(attrs)
    known |= set(CONTRACT["usage_classes"]["total_tokens"])
    assert set(C.ATTRIBUTE_ALLOWLIST) - known == SDK_ONLY_ALLOWED


def test_observation_type_values_equal_the_contract_enum():
    rule = CONTRACT["observation_type_mapping"]["explicit_attribute"]
    assert rule["name"] == C.ATTR_OBSERVATION_TYPE
    assert set(rule["allowed_values"]) == set(C.OBSERVATION_TYPES)


def test_sdk_never_allowlists_a_receiver_content_key():
    content = set(CONTRACT["content_keys"]["exact"])
    assert not (set(C.ATTRIBUTE_ALLOWLIST) & content)
    assert not (set(C.RESOURCE_ALLOWLIST) & content)
    for prefix in CONTRACT["content_keys"]["prefixes"]:
        exceptions = set(CONTRACT["content_keys"]["prefix_exceptions"])
        for key in C.ATTRIBUTE_ALLOWLIST:
            assert not key.startswith(prefix) or key in exceptions, key


def test_sdk_content_keys_are_content_to_the_receiver_too():
    content = set(CONTRACT["content_keys"]["exact"])
    sdk_only = {"llm.prompt_template.template", "llm.prompt_template.variables"}
    assert (
        set(C.CONTENT_ATTRIBUTE_KEYS) - content == sdk_only
    )  # dropped by allowlist model


def test_sdk_lengths_never_exceed_receiver_bounds():
    limits = CONTRACT["metadata_allowlist"]["attributes"]
    for key, spec in C.ATTRIBUTE_ALLOWLIST.items():
        rule = limits.get(key)
        if rule and "max_length" in rule and spec.kind in {"str", "enum"}:
            assert spec.max_len <= rule["max_length"], key


def test_lineage_attributes_are_the_ones_the_receiver_reads():
    known = set(CONTRACT["metadata_allowlist"]["attributes"])
    assert {C.ATTR_TRIAL_ID, C.ATTR_OPTIMIZATION_SESSION_ID} <= known


def test_negative_control_a_renamed_lineage_key_would_be_caught():
    assert (
        "traigent.optimization_run_id"
        not in CONTRACT["metadata_allowlist"]["attributes"]
    )


def test_the_contract_the_sdk_loads_is_the_hash_locked_one():
    assert PACKAGED.read_bytes() == PATH.read_bytes()
    assert C.CONTRACT == CONTRACT


def test_span_allowlist_is_exactly_the_contract_egress_set_minus_other_channels():
    egress = set(CONTRACT["metadata_allowlist"]["attributes"])
    assert set(C.ATTRIBUTE_ALLOWLIST) == egress - C.RESOURCE_KEYS - C.EVENT_KEYS
    assert set(C.RESOURCE_ALLOWLIST) <= egress
    assert set(ALLOWED_EVENT_ATTRS) <= egress


ALLOWED_EVENT_ATTRS = [a for attrs in C.ALLOWED_EVENTS.values() for a in attrs]


def test_every_usage_alias_and_marker_is_egress_allowed():
    usage = CONTRACT["usage_classes"]
    aliases = {a for names in usage["attributes"].values() for a in names}
    aliases |= set(usage["total_tokens"])
    aliases.add(usage["semantics_marker"]["attribute"])
    assert aliases <= set(C.ATTRIBUTE_ALLOWLIST)


def test_specs_carry_the_contract_types_and_bounds():
    for key, rule in CONTRACT["metadata_allowlist"]["attributes"].items():
        spec = C.ATTRIBUTE_ALLOWLIST.get(key) or C.RESOURCE_ALLOWLIST.get(key)
        if spec is None:
            assert key in C.EVENT_KEYS, key
            continue
        if rule["type"] == "non_negative_integer":
            assert (spec.kind, spec.lo, spec.hi) == (
                "int",
                rule["minimum"],
                rule["maximum"],
            ), key
        elif rule["type"] == "string":
            assert spec.kind in {"str", "enum"}, key
            assert spec.max_len <= rule["max_length"], key
        elif rule["type"] == "string_array":
            assert (spec.kind, spec.max_items, spec.max_len) == (
                "str_seq",
                rule["max_items"],
                rule["max_length"],
            ), key
        elif rule["type"] == "number":
            assert spec.kind == "float", key


def test_content_mode_resolution_vectors_drive_the_policy():
    from traigent.observability.otel.policy import ContentPolicy

    for vector in CONTRACT["content_mode"]["resolution_vectors"]:
        declared = vector["declared"]
        attrs = (
            {C.CONTENT_MODE_ATTRIBUTE: declared["value"]} if declared["present"] else {}
        )
        got = ContentPolicy(vector["configured"]).effective_mode(attrs)
        assert got == vector["expected"], vector["id"]


def test_negative_control_a_fail_open_resolver_fails_the_vectors():
    """The vectors can fail: a resolver that falls back to the configured mode
    for an invalid declaration (the original defect) disagrees with them."""

    def fail_open(configured, attrs):
        declared = attrs.get(C.CONTENT_MODE_ATTRIBUTE)
        if declared in C.CONTENT_MODES:
            order = CONTRACT["content_mode"]["restrictiveness_order_most_to_least"]
            return min(configured, declared, key=order.index)
        return configured

    wrong = [
        v["id"]
        for v in CONTRACT["content_mode"]["resolution_vectors"]
        if fail_open(
            v["configured"],
            {C.CONTENT_MODE_ATTRIBUTE: v["declared"].get("value")}
            if v["declared"]["present"]
            else {},
        )
        != v["expected"]
    ]
    assert wrong, "the contract vectors no longer catch a fail-open resolver"
