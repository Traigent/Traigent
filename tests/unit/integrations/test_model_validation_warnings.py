"""Regression tests for non-blocking model validation warnings."""

from __future__ import annotations

import pytest

from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin
from traigent.integrations.llms.gemini_plugin import GeminiPlugin
from traigent.integrations.llms.mistral_plugin import MistralPlugin
from traigent.integrations.llms.openai_plugin import OpenAIPlugin


class _AlwaysInvalidDiscovery:
    def is_valid_model(self, _: str) -> bool:
        return False


@pytest.mark.parametrize(
    "plugin_cls, warning_prefix",
    [
        (OpenAIPlugin, "Unrecognized OpenAI model"),
        (AnthropicPlugin, "Unrecognized Anthropic model"),
        (GeminiPlugin, "Unrecognized Gemini model"),
        (MistralPlugin, "Unrecognized Mistral model"),
    ],
)
def test_unknown_model_emits_warning_without_validation_error(
    plugin_cls, warning_prefix, monkeypatch, caplog
) -> None:
    """Unknown models should warn and continue (no synthetic validation errors)."""
    from traigent.integrations import model_discovery

    monkeypatch.setattr(
        model_discovery,
        "get_model_discovery",
        lambda _framework: _AlwaysInvalidDiscovery(),
    )
    plugin = plugin_cls()

    with caplog.at_level("WARNING"):
        errors = plugin._validate_model("model", "unknown-model-id")

    assert errors == []
    assert any(warning_prefix in record.message for record in caplog.records)
