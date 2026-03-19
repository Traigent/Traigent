"""Tests for hashing utilities."""

import hashlib
import json
from datetime import UTC, datetime

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


# --- generate_run_label tests ---


class TestGenerateRunLabel:
    """Tests for generate_run_label()."""

    def test_basic_format(self):
        """Run label follows {name}_{YYYYMMDD}_{HHMMSS}_{hash} format."""
        ts = datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)
        label = hashing.generate_run_label("answer_question", "opt-id-123", ts)
        assert label.startswith("answer_question_20260315_143022_")
        # 6-char hash suffix
        parts = label.split("_")
        assert len(parts[-1]) == 6

    def test_sanitization_special_chars(self):
        """Special characters are replaced with underscores."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        label = hashing.generate_run_label("My Func-Name!", "id", ts)
        assert label.startswith("my_func_name_")

    def test_sanitization_unicode(self):
        """Unicode characters are replaced with underscores."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        label = hashing.generate_run_label("résumé_parser", "id", ts)
        # Non-ASCII replaced, but alnum/underscore preserved
        assert "r" in label
        assert "_parser_" in label

    def test_truncation_long_name(self):
        """Function names longer than 40 chars are truncated."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        long_name = "a" * 60
        label = hashing.generate_run_label(long_name, "id", ts)
        # Name part should be at most 40 chars
        name_part = label.split("_20260101_")[0]
        assert len(name_part) <= 40

    def test_empty_name_fallback(self):
        """Empty function name falls back to 'run'."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        label = hashing.generate_run_label("", "id", ts)
        assert label.startswith("run_")

    def test_symbols_only_name_fallback(self):
        """Name with only special characters falls back to 'run'."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        label = hashing.generate_run_label("@#$%", "id", ts)
        assert label.startswith("run_")

    def test_deterministic(self):
        """Same inputs produce the same label."""
        ts = datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)
        label1 = hashing.generate_run_label("func", "opt-123", ts)
        label2 = hashing.generate_run_label("func", "opt-123", ts)
        assert label1 == label2

    def test_different_opt_ids_differ(self):
        """Different optimization IDs produce different labels."""
        ts = datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)
        label1 = hashing.generate_run_label("func", "opt-123", ts)
        label2 = hashing.generate_run_label("func", "opt-456", ts)
        assert label1 != label2

    def test_hash_is_6_chars(self):
        """Short hash suffix is exactly 6 hex characters."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        label = hashing.generate_run_label("func", "some-uuid", ts)
        short_hash = label.rsplit("_", 1)[-1]
        assert len(short_hash) == 6
        int(short_hash, 16)  # Must be valid hex
