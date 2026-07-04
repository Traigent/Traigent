"""Regression tests for traigent.observability.decorators key-based redaction.

See issue #1649: three SDK sanitizers (this module, dataset_converter, and
agent_spans) used to carry independent keyword lists, so a key redacted by
one path could pass through another. They now share one canonical set via
`traigent.security.redaction.is_sensitive_key_name`.
"""

from __future__ import annotations

import pytest

from traigent.observability.decorators import (
    _is_sensitive_key,
    _sanitize_metadata_value,
)


class TestSensitiveKeyUnion:
    @pytest.mark.parametrize(
        "key",
        [
            # Already covered by this module's old fragment list.
            "api_key",
            "auth_header",
            "private_key",
            "secret",
            "token",
            # Formerly missing here, only present in dataset_converter's regex.
            "credit_card",
            "authorization",
            # Formerly missing here, only present in agent_spans' content list.
            "prompt",
            "response",
            "actual_output",
        ],
    )
    def test_union_keyword_is_redacted(self, key: str) -> None:
        assert _is_sensitive_key(key) is True
        assert _sanitize_metadata_value("super-secret-value", key=key) == "[REDACTED]"

    def test_ordinary_key_passes_through(self) -> None:
        assert _is_sensitive_key("model_name") is False
        assert _sanitize_metadata_value("gpt-4", key="model_name") == "gpt-4"
