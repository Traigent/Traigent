"""SDK attribute contract vs the receiver's shared contract (vendored, hash-locked).

The vendored file is a snapshot of the Backend receiver branch's contract
(TraigentBackend feat/otlp-ingest, commit f780d4ef2).  Until the Schema repo
publishes the canonical file this test only proves agreement with that
snapshot: it does not prove the receiver's behaviour (a hash proves agreement,
not correctness - see the receiver-side golden tests for that).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from traigent.observability.otel import contract as C

PATH = Path(__file__).parents[3] / "fixtures/observability/otel_attribute_contract_v1.json"
LOCKED_SHA256 = "d4b260cabc3b45b00e0369ca2a70a7139981746d0c26670b04bcda8a2aa5aad0"
CONTRACT = json.loads(PATH.read_text())

# SDK-only keys the receiver snapshot does not know yet.  Each is a reviewed
# integration item (the receiver counts unknown keys as dropped, never fails).
SDK_ONLY_ALLOWED = {"traigent.observation_type"}


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
    assert set(C.CONTENT_ATTRIBUTE_KEYS) - content == sdk_only  # dropped by allowlist model


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
    assert "traigent.optimization_run_id" not in CONTRACT["metadata_allowlist"]["attributes"]
