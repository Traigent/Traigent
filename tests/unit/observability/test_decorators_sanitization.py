"""Regression tests for traigent.observability.decorators key-based redaction.

See issue #1649: three SDK sanitizers (this module, dataset_converter, and
agent_spans) used to carry independent keyword lists, so a key redacted by
one path could pass through another. They now share the canonical
CREDENTIAL set via `traigent.security.redaction.is_credential_key_name`.

Content-marker fragments (prompt/response/output/...) are deliberately NOT
applied here: this sanitizer processes tuned configuration values
(`traigent_active_config` / optimization context), and config spaces
routinely tune a variable literally named "prompt" — a variant label the
portal must display, not free-form content.
"""

from __future__ import annotations

import pytest

from traigent.config.context import applied_config_context
from traigent.observability.decorators import (
    _ACTIVE_CONFIG_METADATA_KEY,
    _build_observe_enrichment_metadata,
    _is_sensitive_key,
    _sanitize_metadata_value,
)


class TestCredentialKeyUnion:
    @pytest.mark.parametrize(
        "key",
        [
            # Already covered by this module's old fragment list.
            "api_key",
            "auth_header",
            "private_key",
            "secret",
            "token",
            "credential",
            "apikey",
            "password",
            # Formerly missing here, only present in dataset_converter's regex.
            "credit_card",
            "authorization",
        ],
    )
    def test_credential_keyword_is_redacted(self, key: str) -> None:
        assert _is_sensitive_key(key) is True
        assert _sanitize_metadata_value("super-secret-value", key=key) == "[REDACTED]"

    def test_ordinary_key_passes_through(self) -> None:
        assert _is_sensitive_key("model_name") is False
        assert _sanitize_metadata_value("gpt-4", key="model_name") == "gpt-4"


class TestTunedConfigValuesNotContentRedacted:
    """A tuned config variable named "prompt" (e.g. walkthrough demo config
    spaces: {"prompt": ["minimal", "role_based"]}) must SURVIVE decorators
    sanitization un-redacted — the portal shows which variant the optimizer
    picked for the trial."""

    @pytest.mark.parametrize(
        "key",
        ["prompt", "response_format", "output_format", "expected", "completion"],
    )
    def test_content_marker_named_config_key_survives(self, key: str) -> None:
        assert _is_sensitive_key(key) is False
        assert _sanitize_metadata_value("role_based", key=key) == "role_based"

    def test_enrichment_metadata_keeps_tuned_prompt_variant(self) -> None:
        token = applied_config_context.set(
            {
                "prompt": "role_based",
                "temperature": 0.2,
                "api_key": "sk-should-never-ship",  # pragma: allowlist secret
            }
        )
        try:
            enriched = _build_observe_enrichment_metadata()
        finally:
            applied_config_context.reset(token)

        active_config = enriched[_ACTIVE_CONFIG_METADATA_KEY]
        assert active_config["prompt"] == "role_based"
        assert active_config["temperature"] == 0.2
        # Credential keys are still redacted on the very same payload.
        assert active_config["api_key"] == "[REDACTED]"
