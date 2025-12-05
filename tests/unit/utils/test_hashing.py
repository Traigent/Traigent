"""Tests for hashing utilities."""

import hashlib
import json

from traigent.utils import hashing


def test_sanitize_set_returns_sorted_list():
    """Sets should be sanitized into deterministically ordered lists."""
    raw = {"gamma", "alpha", "beta"}

    sanitized = hashing._sanitize_json_value(raw)  # pylint: disable=protected-access

    assert sanitized == ["alpha", "beta", "gamma"]


def test_sanitize_set_nested_structure_sorted():
    """Nested structures inside sets should be sorted deterministically."""
    raw = {("b", 2), ("a", 1)}

    sanitized = hashing._sanitize_json_value(raw)  # pylint: disable=protected-access

    assert sanitized == [["a", 1], ["b", 2]]


def test_generate_config_hash_stable_for_sets():
    """Hash generation must remain stable when configs include sets."""
    config_with_set = {"values": {"beta", "alpha"}}

    hash_one = hashing.generate_config_hash(config_with_set)
    hash_two = hashing.generate_config_hash({"values": {"alpha", "beta"}})

    assert hash_one == hash_two

    expected_payload = json.dumps({"values": ["alpha", "beta"]}, sort_keys=True)
    expected_hash = hashlib.sha256(expected_payload.encode()).hexdigest()[:16]
    assert hash_one == expected_hash
