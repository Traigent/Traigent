"""Tests for reasoning/extended thinking parameter translation.

Tests the translation of generic reasoning_mode/reasoning_budget params
to provider-specific params (OpenAI reasoning_effort, Anthropic thinking,
Gemini thinking_level/thinking_budget).
"""

from __future__ import annotations

import pytest


class TestModelCapabilities:
    """Tests for model capability detection."""

    def test_openai_reasoning_model_detection(self):
        """Test detection of OpenAI reasoning models."""
        from traigent.integrations.utils.model_capabilities import supports_reasoning

        # Reasoning models
        assert supports_reasoning("o1", "openai")
        assert supports_reasoning("o1-mini", "openai")
        assert supports_reasoning("o1-preview", "openai")
        assert supports_reasoning("o3", "openai")
        assert supports_reasoning("o3-mini", "openai")
        assert supports_reasoning("o4-mini", "openai")
        assert supports_reasoning("gpt-5", "openai")
        assert supports_reasoning("gpt-5.1-codex-max", "openai")

        # Non-reasoning models
        assert not supports_reasoning("gpt-4o", "openai")
        assert not supports_reasoning("gpt-4o-mini", "openai")
        assert not supports_reasoning("gpt-4-turbo", "openai")
        assert not supports_reasoning("gpt-3.5-turbo", "openai")

    def test_anthropic_reasoning_model_detection(self):
        """Test detection of Anthropic reasoning models."""
        from traigent.integrations.utils.model_capabilities import supports_reasoning

        # Reasoning models (Claude 4+)
        assert supports_reasoning("claude-sonnet-4-5", "anthropic")
        assert supports_reasoning("claude-opus-4-5", "anthropic")
        assert supports_reasoning("claude-4", "anthropic")

        # Non-reasoning models (Claude 3)
        assert not supports_reasoning("claude-3-opus", "anthropic")
        assert not supports_reasoning("claude-3-sonnet", "anthropic")
        assert not supports_reasoning("claude-3-haiku", "anthropic")

    def test_gemini_reasoning_model_detection(self):
        """Test detection of Gemini reasoning models."""
        from traigent.integrations.utils.model_capabilities import supports_reasoning

        # Reasoning models
        assert supports_reasoning("gemini-2.5-pro", "gemini")
        assert supports_reasoning("gemini-2.5-flash", "gemini")
        assert supports_reasoning("gemini-3-pro", "gemini")
        assert supports_reasoning("gemini-3-flash", "gemini")

        # Non-reasoning models
        assert not supports_reasoning("gemini-1.5-pro", "gemini")
        assert not supports_reasoning("gemini-pro", "gemini")

    def test_gemini_version_detection(self):
        """Test Gemini 3 vs 2.5 detection."""
        from traigent.integrations.utils.model_capabilities import is_gemini_3

        assert is_gemini_3("gemini-3-pro")
        assert is_gemini_3("gemini-3-flash")
        assert not is_gemini_3("gemini-2.5-pro")
        assert not is_gemini_3("gemini-2.5-flash")

    def test_reasoning_effort_levels(self):
        """Test reasoning effort level availability by model."""
        from traigent.integrations.utils.model_capabilities import (
            get_reasoning_effort_levels,
        )

        # o1/o3 models get standard levels
        o3_levels = get_reasoning_effort_levels("o3")
        assert "low" in o3_levels
        assert "medium" in o3_levels
        assert "high" in o3_levels
        assert "minimal" not in o3_levels  # GPT-5+ only
        assert "xhigh" not in o3_levels  # GPT-5.1-codex-max only

        # GPT-5 gets minimal
        gpt5_levels = get_reasoning_effort_levels("gpt-5")
        assert "minimal" in gpt5_levels
        assert "low" in gpt5_levels
        assert "medium" in gpt5_levels
        assert "high" in gpt5_levels

        # GPT-5.1-codex-max gets all levels
        gpt5_codex_levels = get_reasoning_effort_levels("gpt-5.1-codex-max")
        assert "xhigh" in gpt5_codex_levels

    def test_unknown_provider_returns_false(self):
        """Test unknown provider returns False for supports_reasoning."""
        from traigent.integrations.utils.model_capabilities import supports_reasoning

        # Unknown provider should return False
        assert not supports_reasoning("o3", "unknown_provider")
        assert not supports_reasoning("claude-sonnet-4-5", "azure")
        assert not supports_reasoning("gemini-3-pro", "bedrock")

    def test_case_insensitivity_supports_reasoning(self):
        """Test that model name matching is case-insensitive."""
        from traigent.integrations.utils.model_capabilities import supports_reasoning

        # Mixed case should still work
        assert supports_reasoning("O3", "openai")
        assert supports_reasoning("GPT-5", "openai")
        assert supports_reasoning("Claude-Sonnet-4-5", "anthropic")
        assert supports_reasoning("GEMINI-3-PRO", "gemini")

    def test_case_insensitivity_is_gemini_3(self):
        """Test that is_gemini_3 is case-insensitive."""
        from traigent.integrations.utils.model_capabilities import is_gemini_3

        assert is_gemini_3("GEMINI-3-PRO")
        assert is_gemini_3("Gemini-3-Flash")
        assert not is_gemini_3("GEMINI-2.5-PRO")

    def test_unknown_model_effort_levels_default(self):
        """Test unknown models get default effort levels."""
        from traigent.integrations.utils.model_capabilities import (
            get_reasoning_effort_levels,
        )

        # Unknown model returns default levels
        unknown_levels = get_reasoning_effort_levels("unknown-reasoning-model")
        assert unknown_levels == ["low", "medium", "high"]

    def test_get_provider_from_model_comprehensive(self):
        """Test provider detection from various model names."""
        from traigent.integrations.utils.model_capabilities import (
            get_provider_from_model,
        )

        # OpenAI models
        assert get_provider_from_model("gpt-4o") == "openai"
        assert get_provider_from_model("gpt-3.5-turbo") == "openai"
        assert get_provider_from_model("o1-preview") == "openai"
        assert get_provider_from_model("o3-mini") == "openai"
        assert get_provider_from_model("davinci-002") == "openai"
        assert get_provider_from_model("text-embedding-3-small") == "openai"

        # Anthropic models
        assert get_provider_from_model("claude-3-opus") == "anthropic"
        assert get_provider_from_model("claude-sonnet-4-5") == "anthropic"
        assert get_provider_from_model("CLAUDE-3-HAIKU") == "anthropic"

        # Gemini models
        assert get_provider_from_model("gemini-1.5-pro") == "gemini"
        assert get_provider_from_model("gemini-2.5-flash") == "gemini"
        assert get_provider_from_model("GEMINI-3-PRO") == "gemini"

        # Unknown models
        assert get_provider_from_model("llama-3-70b") is None
        assert get_provider_from_model("mistral-large") is None
        assert get_provider_from_model("custom-model") is None

    def test_effort_levels_sorted_order(self):
        """Test that effort levels are returned in logical order."""
        from traigent.integrations.utils.model_capabilities import (
            get_reasoning_effort_levels,
        )

        # GPT-5.1-codex-max should have all levels in order
        levels = get_reasoning_effort_levels("gpt-5.1-codex-max")
        assert levels == ["minimal", "low", "medium", "high", "xhigh"]

        # GPT-5 should have most levels in order
        levels = get_reasoning_effort_levels("gpt-5")
        assert levels == ["minimal", "low", "medium", "high"]


class TestOpenAIReasoningOverrides:
    """Tests for OpenAI reasoning parameter translation."""

    def test_reasoning_mode_to_effort_translation(self):
        """Test generic reasoning_mode -> reasoning_effort translation."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        # Mock config with model
        class MockConfig:
            model = "o3"
            custom_params = {}

        params = {"reasoning_mode": "deep", "temperature": 0.7}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["reasoning_effort"] == "high"
        assert "reasoning_mode" not in result
        assert result["temperature"] == pytest.approx(0.7)

    def test_provider_specific_wins_over_generic(self):
        """Test provider-specific params override generic."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = "o3"
            custom_params = {}

        params = {"reasoning_mode": "deep", "reasoning_effort": "low"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Provider-specific should win
        assert result["reasoning_effort"] == "low"

    def test_non_reasoning_model_skipped(self):
        """Test reasoning params removed for non-reasoning models."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = "gpt-4o"
            custom_params = {}

        params = {
            "reasoning_mode": "deep",
            "reasoning_effort": "high",
            "temperature": 0.7,
        }
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert "reasoning_mode" not in result
        assert "reasoning_effort" not in result
        assert result == {"temperature": 0.7}

    def test_max_completion_tokens_replaces_max_tokens(self):
        """Test max_completion_tokens takes precedence for reasoning models."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = "o3"
            custom_params = {}

        params = {"max_tokens": 1000, "max_completion_tokens": 5000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["max_completion_tokens"] == 5000
        assert "max_tokens" not in result

    def test_no_model_specified_cleans_params(self):
        """Test that params are cleaned when no model is specified."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = None
            custom_params = {}

        params = {
            "reasoning_mode": "deep",
            "reasoning_budget": 8000,
            "temperature": 0.5,
        }
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Generic params should be removed, temperature should remain
        assert "reasoning_mode" not in result
        assert "reasoning_budget" not in result
        assert result["temperature"] == pytest.approx(0.5)

    def test_reasoning_budget_to_max_completion_tokens(self):
        """Test reasoning_budget translates to max_completion_tokens."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = "o3"
            custom_params = {}

        params = {"reasoning_budget": 50000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["max_completion_tokens"] == 50000
        assert "reasoning_budget" not in result

    def test_standard_mode_translation(self):
        """Test reasoning_mode standard translates to medium effort."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = "o3"
            custom_params = {}

        params = {"reasoning_mode": "standard"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["reasoning_effort"] == "medium"

    def test_none_mode_translation(self):
        """Test reasoning_mode none translates to minimal effort (or fallback).

        For o3 models, minimal is not available, so it falls back to low.
        For GPT-5+ models, minimal is available.
        """
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        # o3 doesn't support minimal, so falls back to low
        class MockConfigO3:
            model = "o3"
            custom_params = {}

        params = {"reasoning_mode": "none"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfigO3())
        assert result["reasoning_effort"] == "low"  # Fallback for o3

        # GPT-5 supports minimal
        class MockConfigGPT5:
            model = "gpt-5"
            custom_params = {}

        params = {"reasoning_mode": "none"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfigGPT5())
        assert result["reasoning_effort"] == "minimal"  # Native support

    def test_model_from_params_used(self):
        """Test model from params is used when config model is None."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        class MockConfig:
            model = None
            custom_params = {}

        # Model in params should be detected
        params = {"model": "o3", "reasoning_mode": "deep"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["reasoning_effort"] == "high"
        assert result["model"] == "o3"


class TestAnthropicReasoningOverrides:
    """Tests for Anthropic extended thinking parameter translation."""

    def test_extended_thinking_object_built(self):
        """Test nested thinking object is built correctly."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"extended_thinking": True, "thinking_budget_tokens": 16000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert "extended_thinking" not in result
        assert "thinking_budget_tokens" not in result

    def test_reasoning_mode_to_thinking(self):
        """Test generic reasoning_mode -> thinking translation."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"reasoning_mode": "deep", "reasoning_budget": 32000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 32000

    def test_thinking_disabled_when_none(self):
        """Test thinking disabled when reasoning_mode is none."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"reasoning_mode": "none"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Should not add thinking object when disabled
        assert "thinking" not in result

    def test_minimum_budget_enforced(self):
        """Test minimum budget of 1024 is enforced."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"extended_thinking": True, "thinking_budget_tokens": 500}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking"]["budget_tokens"] >= 1024

    def test_budget_only_enables_thinking(self):
        """Test that providing only a budget enables thinking automatically."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        # Only provide budget, no mode or extended_thinking toggle
        params = {"thinking_budget_tokens": 16000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Should auto-enable thinking since a budget was provided
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 16000

    def test_reasoning_budget_only_enables_thinking(self):
        """Test that generic reasoning_budget alone enables thinking."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        # Only provide generic budget
        params = {"reasoning_budget": 24000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Should auto-enable thinking since a budget was provided
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 24000

    def test_no_model_cleans_anthropic_params(self):
        """Test that params are cleaned when no model is specified."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = None
            custom_params = {}

        params = {
            "reasoning_mode": "deep",
            "reasoning_budget": 8000,
            "extended_thinking": True,
            "thinking_budget_tokens": 16000,
            "temperature": 0.5,
        }
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # All reasoning params should be removed
        assert "reasoning_mode" not in result
        assert "reasoning_budget" not in result
        assert "extended_thinking" not in result
        assert "thinking_budget_tokens" not in result
        assert "thinking" not in result
        assert result["temperature"] == pytest.approx(0.5)

    def test_extended_thinking_false_disables(self):
        """Test extended_thinking=False explicitly disables thinking."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"extended_thinking": False, "thinking_budget_tokens": 16000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Thinking should not be enabled
        assert "thinking" not in result
        assert "extended_thinking" not in result

    def test_zero_budget_uses_minimum(self):
        """Test that zero budget defaults to minimum 1024."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"reasoning_budget": 0}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Budget should be clamped to minimum 1024
        assert result["thinking"]["budget_tokens"] == 1024

    def test_non_reasoning_anthropic_model_cleans_params(self):
        """Test non-reasoning Anthropic models strip all reasoning params."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-3-opus"  # Claude 3 doesn't support extended thinking
            custom_params = {}

        params = {
            "reasoning_mode": "deep",
            "extended_thinking": True,
            "thinking_budget_tokens": 16000,
            "temperature": 0.7,
        }
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # All reasoning params should be removed
        assert "reasoning_mode" not in result
        assert "extended_thinking" not in result
        assert "thinking_budget_tokens" not in result
        assert "thinking" not in result
        assert result["temperature"] == pytest.approx(0.7)

    def test_standard_mode_enables_thinking_with_default_budget(self):
        """Test reasoning_mode standard enables thinking with default budget."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        class MockConfig:
            model = "claude-sonnet-4-5"
            custom_params = {}

        params = {"reasoning_mode": "standard"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 8000  # Default


class TestGeminiReasoningOverrides:
    """Tests for Gemini thinking parameter translation."""

    def test_gemini_3_uses_thinking_level(self):
        """Test Gemini 3 uses thinking_level parameter."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-3-pro"
            custom_params = {}

        params = {"reasoning_mode": "deep"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_level"] == "high"
        assert "thinking_budget" not in result
        assert "reasoning_mode" not in result

    def test_gemini_2_5_uses_thinking_budget(self):
        """Test Gemini 2.5 uses thinking_budget parameter."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-2.5-pro"
            custom_params = {}

        params = {"reasoning_mode": "deep"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_budget"] == 32768
        assert "thinking_level" not in result
        assert "reasoning_mode" not in result

    def test_gemini_2_5_budget_clamped(self):
        """Test Gemini 2.5 thinking_budget is clamped to 32768."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-2.5-pro"
            custom_params = {}

        params = {"reasoning_budget": 100000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_budget"] == 32768

    def test_non_reasoning_gemini_strips_all_thinking_params(self):
        """Test non-reasoning Gemini models strip ALL thinking params.

        Gemini 1.5 doesn't support thinking params - they should all be removed
        to avoid API errors.
        """
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-1.5-pro"
            custom_params = {}

        # Include both generic and provider-specific reasoning params
        params = {
            "reasoning_mode": "deep",
            "reasoning_budget": 8000,
            "thinking_level": "high",
            "thinking_budget": 16000,
            "temperature": 0.7,
        }
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # All reasoning params should be removed for non-reasoning models
        assert "reasoning_mode" not in result
        assert "reasoning_budget" not in result
        assert "thinking_level" not in result
        assert "thinking_budget" not in result
        # Temperature should remain
        assert result["temperature"] == pytest.approx(0.7)

    def test_no_model_cleans_gemini_params(self):
        """Test that params are cleaned when no model is specified."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = None
            custom_params = {}

        params = {
            "reasoning_mode": "deep",
            "reasoning_budget": 8000,
            "temperature": 0.5,
        }
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert "reasoning_mode" not in result
        assert "reasoning_budget" not in result
        assert result["temperature"] == pytest.approx(0.5)

    def test_gemini_3_provider_specific_wins(self):
        """Test provider-specific thinking_level wins over reasoning_mode."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-3-pro"
            custom_params = {}

        params = {"reasoning_mode": "deep", "thinking_level": "low"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Provider-specific should win
        assert result["thinking_level"] == "low"
        assert "reasoning_mode" not in result

    def test_gemini_2_5_provider_specific_wins(self):
        """Test provider-specific thinking_budget wins over reasoning_budget."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-2.5-pro"
            custom_params = {}

        params = {"reasoning_budget": 8000, "thinking_budget": 16000}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        # Provider-specific should win (clamped to max)
        assert result["thinking_budget"] == 16000
        assert "reasoning_budget" not in result

    def test_gemini_3_standard_mode_translation(self):
        """Test standard mode translates to low for Gemini 3."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-3-flash"
            custom_params = {}

        params = {"reasoning_mode": "standard"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_level"] == "low"

    def test_gemini_3_none_mode_translation(self):
        """Test none mode translates to MINIMAL for Gemini 3."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-3-pro"
            custom_params = {}

        params = {"reasoning_mode": "none"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_level"] == "MINIMAL"

    def test_gemini_2_5_standard_mode_translation(self):
        """Test standard mode translates to 8192 budget for Gemini 2.5."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-2.5-flash"
            custom_params = {}

        params = {"reasoning_mode": "standard"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_budget"] == 8192

    def test_gemini_2_5_none_mode_translation(self):
        """Test none mode translates to 0 budget for Gemini 2.5."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = "gemini-2.5-pro"
            custom_params = {}

        params = {"reasoning_mode": "none"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_budget"] == 0

    def test_gemini_model_from_params_used(self):
        """Test model from params is used when config model is None."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        class MockConfig:
            model = None
            custom_params = {}

        # Model in params should be detected
        params = {"model": "gemini-3-pro", "reasoning_mode": "deep"}
        result = plugin._apply_reasoning_overrides(params.copy(), MockConfig())

        assert result["thinking_level"] == "high"
        assert result["model"] == "gemini-3-pro"


class TestFactoryMethods:
    """Tests for reasoning factory methods."""

    def test_reasoning_mode_factory(self):
        """Test Choices.reasoning_mode() factory method."""
        from traigent.api import Choices

        mode = Choices.reasoning_mode()
        assert mode.name == "reasoning_mode"
        assert "none" in mode.values
        assert "standard" in mode.values
        assert "deep" in mode.values
        assert mode.default == "standard"

    def test_reasoning_effort_factory(self):
        """Test Choices.reasoning_effort() factory method."""
        from traigent.api import Choices

        effort = Choices.reasoning_effort()
        assert effort.name == "reasoning_effort"
        assert "minimal" in effort.values
        assert "low" in effort.values
        assert "medium" in effort.values
        assert "high" in effort.values
        assert "xhigh" in effort.values
        assert effort.default == "medium"

    def test_extended_thinking_factory(self):
        """Test Choices.extended_thinking() factory method."""
        from traigent.api import Choices

        thinking = Choices.extended_thinking()
        assert thinking.name == "extended_thinking"
        assert True in thinking.values
        assert False in thinking.values
        assert thinking.default is False

    def test_thinking_level_factory(self):
        """Test Choices.thinking_level() factory method."""
        from traigent.api import Choices

        level = Choices.thinking_level()
        assert level.name == "thinking_level"
        assert "MINIMAL" in level.values
        assert "low" in level.values
        assert "high" in level.values
        assert level.default == "high"

    def test_reasoning_budget_factory(self):
        """Test IntRange.reasoning_budget() factory method."""
        from traigent.api import IntRange

        budget = IntRange.reasoning_budget()
        assert budget.name == "reasoning_budget"
        assert budget.low == 0
        assert budget.high == 128000
        assert budget.default == 8000

    def test_reasoning_tokens_factory(self):
        """Test IntRange.reasoning_tokens() factory method."""
        from traigent.api import IntRange

        tokens = IntRange.reasoning_tokens()
        assert tokens.name == "max_completion_tokens"
        assert tokens.low == 1024
        assert tokens.high == 128000
        assert tokens.default == 32000

    def test_thinking_budget_factory(self):
        """Test IntRange.thinking_budget() factory method."""
        from traigent.api import IntRange

        budget = IntRange.thinking_budget()
        assert budget.name == "thinking_budget_tokens"
        assert budget.low == 1024
        assert budget.high == 128000
        assert budget.default == 8000

    def test_gemini_thinking_budget_factory(self):
        """Test IntRange.gemini_thinking_budget() factory method."""
        from traigent.api import IntRange

        budget = IntRange.gemini_thinking_budget()
        assert budget.name == "thinking_budget"
        assert budget.low == 0
        assert budget.high == 32768
        assert budget.default == 8192

    def test_reasoning_mode_custom_default(self):
        """Test Choices.reasoning_mode() with custom default."""
        from traigent.api import Choices

        mode = Choices.reasoning_mode(default="deep")
        assert mode.default == "deep"

        mode_none = Choices.reasoning_mode(default="none")
        assert mode_none.default == "none"

    def test_reasoning_effort_custom_default(self):
        """Test Choices.reasoning_effort() with custom default."""
        from traigent.api import Choices

        effort = Choices.reasoning_effort(default="high")
        assert effort.default == "high"

        effort_low = Choices.reasoning_effort(default="low")
        assert effort_low.default == "low"

    def test_extended_thinking_custom_default(self):
        """Test Choices.extended_thinking() with custom default."""
        from traigent.api import Choices

        thinking = Choices.extended_thinking(default=True)
        assert thinking.default is True

    def test_thinking_level_custom_default(self):
        """Test Choices.thinking_level() with custom default."""
        from traigent.api import Choices

        level = Choices.thinking_level(default="low")
        assert level.default == "low"

        level_minimal = Choices.thinking_level(default="MINIMAL")
        assert level_minimal.default == "MINIMAL"

    def test_factory_methods_have_unit_attribute(self):
        """Test that IntRange factory methods include unit attribute."""
        from traigent.api import IntRange

        budget = IntRange.reasoning_budget()
        assert budget.unit == "tokens"

        tokens = IntRange.reasoning_tokens()
        assert tokens.unit == "tokens"

        thinking_budget = IntRange.thinking_budget()
        assert thinking_budget.unit == "tokens"

        gemini_budget = IntRange.gemini_thinking_budget()
        assert gemini_budget.unit == "tokens"

    def test_factory_values_are_iterable(self):
        """Test that Choices factory values can be iterated."""
        from traigent.api import Choices

        mode = Choices.reasoning_mode()
        values_list = list(mode.values)
        assert len(values_list) == 3
        assert "none" in values_list
        assert "standard" in values_list
        assert "deep" in values_list

        effort = Choices.reasoning_effort()
        effort_list = list(effort.values)
        assert len(effort_list) == 5


class TestPluginIntegration:
    """Integration tests for plugin override methods with full apply_overrides."""

    def test_openai_full_apply_overrides_preserves_reasoning(self):
        """Test full apply_overrides preserves reasoning params for reasoning models."""
        from traigent.integrations.llms.openai_plugin import OpenAIPlugin

        plugin = OpenAIPlugin()

        # Use dict config which is supported
        config = {"model": "o3", "custom_params": {"temperature": 0.8}}

        kwargs = {
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "test"}],
        }
        result = plugin.apply_overrides(kwargs, config)

        # Reasoning params should be preserved
        assert result.get("reasoning_effort") == "high"
        # Messages should pass through
        assert "messages" in result

    def test_anthropic_full_apply_overrides_builds_thinking(self):
        """Test full apply_overrides builds thinking object correctly."""
        from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()

        # Use dict config which is supported - must include max_tokens (required)
        config = {"model": "claude-sonnet-4-5", "custom_params": {"max_tokens": 4096}}

        kwargs = {
            "extended_thinking": True,
            "thinking_budget_tokens": 20000,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4096,
        }
        result = plugin.apply_overrides(kwargs, config)

        # Thinking object should be built
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 20000
        # Original params should be cleaned
        assert "extended_thinking" not in result
        assert "thinking_budget_tokens" not in result

    def test_gemini_full_apply_overrides_wraps_in_generation_config(self):
        """Test full apply_overrides wraps thinking params in generation_config."""
        from traigent.integrations.llms.gemini_plugin import GeminiPlugin

        plugin = GeminiPlugin()

        # Use dict config which is supported
        config = {"model": "gemini-3-pro", "custom_params": {}}

        kwargs = {"thinking_level": "high", "temperature": 0.7}
        result = plugin.apply_overrides(kwargs, config)

        # Thinking params should be in generation_config
        assert "generation_config" in result
        assert result["generation_config"].get("thinking_level") == "high"
        assert result["generation_config"].get("temperature") == pytest.approx(0.7)
        # Should not be at top level
        assert "thinking_level" not in result
        assert "temperature" not in result
