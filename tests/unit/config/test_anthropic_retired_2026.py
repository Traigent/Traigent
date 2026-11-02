"""Regression for #2876: retired Claude models must be rejected by validation.

Claude 3.5 Haiku (2026-02-19), Claude 3.5 Sonnet (2025-10-28), Claude Sonnet 4
and Opus 4 (2026-06-15) and Claude Opus 4.1 (2026-08-05) were retired by
Anthropic but missing from the denylist, so ``AnthropicDiscovery`` accepted
them and trials failed at the provider after spend was approved.
"""

from __future__ import annotations

import pytest

from traigent.config.retired_models import is_retired_model
from traigent.integrations.model_discovery.anthropic_discovery import (
    AnthropicDiscovery,
)

RETIRED_CLAUDE_IDS = [
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    "claude-3-5-haiku",
    # Claude 3.5 Sonnet (dashed spellings)
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet",
    # Claude Sonnet 4 / Opus 4 / Opus 4.1
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    # OpenRouter dotted spellings
    "claude-3.5-haiku",
    "claude-3.5-haiku-20241022",
    "claude-3.5-sonnet-20240620",
    "claude-3.5-sonnet-20241022",
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-opus-4.1",
]

# Still served by Anthropic; must stay valid. claude-sonnet-4-5-20250929
# retires 2026-11-30 (not yet retired).
LIVE_CLAUDE_IDS = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4.5",
    "claude-opus-4.5",
]


@pytest.mark.parametrize("model_id", RETIRED_CLAUDE_IDS)
def test_retired_claude_ids_are_retired(model_id: str) -> None:
    assert is_retired_model(model_id)


@pytest.mark.parametrize("model_id", RETIRED_CLAUDE_IDS)
@pytest.mark.parametrize(
    "prefix", ["anthropic/", "openrouter/anthropic/", "anthropic."]
)
def test_retired_claude_ids_with_provider_prefix(model_id: str, prefix: str) -> None:
    assert is_retired_model(prefix + model_id)


@pytest.mark.parametrize("model_id", LIVE_CLAUDE_IDS)
def test_live_claude_ids_not_retired(model_id: str) -> None:
    assert not is_retired_model(model_id)


@pytest.mark.parametrize("model_id", RETIRED_CLAUDE_IDS)
def test_anthropic_discovery_rejects_retired(model_id: str) -> None:
    discovery = AnthropicDiscovery()
    assert not discovery.is_valid_model(model_id)
    assert model_id not in discovery.list_models()


def test_bedrock_alias_of_retired_claude_is_retired() -> None:
    assert is_retired_model("anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert is_retired_model("anthropic.claude-3-5-haiku-20241022-v1:0")
