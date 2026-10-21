"""Privacy config redaction must not leak customer-authored nested dict keys (#2680)."""

import json

import pytest

from traigent.cloud.trial_operations import TrialOperations

pytestmark = pytest.mark.backend_online

SENTINEL = "DEMO_CUSTOMER_TEXT"


def test_contains_string_flags_text_keys_with_non_string_values():
    assert TrialOperations._contains_string({SENTINEL: 1}) is True
    assert TrialOperations._contains_string([{"a": {SENTINEL: 2}}]) is True


def test_contains_string_still_passes_string_free_values():
    assert TrialOperations._contains_string([1, 2.5, None, True]) is False
    assert TrialOperations._contains_string({}) is False


def test_redacted_privacy_config_has_no_nested_key_text():
    config = {
        "temperature": 0.2,
        "settings": {SENTINEL: 1},
        "deep": [{"x": {SENTINEL + "_2": False}}],
    }
    redacted = TrialOperations._redact_privacy_config(config)
    wire = json.dumps(redacted)
    assert SENTINEL not in wire
    assert redacted["temperature"] == 0.2
    assert set(redacted) == {"temperature", "settings", "deep"}
