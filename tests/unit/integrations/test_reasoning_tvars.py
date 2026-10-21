"""Reasoning-effort tuned variables: factories, capability table, translation (#1542).

All tests are offline; no provider is called.
"""

from __future__ import annotations

import pytest

from traigent.api.parameter_ranges import Choices, IntRange
from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin
from traigent.integrations.llms.gemini_plugin import GeminiPlugin
from traigent.integrations.llms.openai_plugin import OpenAIPlugin
from traigent.integrations.utils.model_capabilities import (
    MODEL_REASONING_KNOB,
    get_reasoning_knob,
)
from traigent.providers.validation import _KNOWN_MODELS

_PROVIDER_KEYS = {"openai": "openai", "anthropic": "anthropic", "google": "google"}


class TestFactories:
    def test_reasoning_effort_domain(self) -> None:
        effort = Choices.reasoning_effort()
        assert list(effort.values) == ["minimal", "low", "medium", "high"]
        assert effort.default == "medium"
        assert effort.name == "reasoning_effort"

    def test_reasoning_effort_subset_and_invalid(self) -> None:
        effort = Choices.reasoning_effort(levels=["low", "high"], default="low")
        assert list(effort.values) == ["low", "high"]
        with pytest.raises(ValueError) as exc:
            Choices.reasoning_effort(levels=["low", "turbo"])
        assert "turbo" not in str(exc.value)

    def test_thinking_budget_tokens_domain(self) -> None:
        budget = IntRange.thinking_budget_tokens()
        assert (budget.low, budget.high) == (1024, 32768)
        assert budget.name == "thinking_budget_tokens"
        assert budget.low <= budget.default <= budget.high


class TestCapabilityTable:
    def test_every_id_is_a_known_sdk_model(self) -> None:
        for provider, table in MODEL_REASONING_KNOB.items():
            unknown = set(table) - set(_KNOWN_MODELS[provider])
            assert not unknown, f"{provider}: ids not in _KNOWN_MODELS"

    def test_no_retired_ids(self) -> None:
        for table in MODEL_REASONING_KNOB.values():
            for model_id in table:
                assert not model_id.startswith(("claude-3", "gpt-3.5", "gemini-1"))

    def test_lookup(self) -> None:
        assert get_reasoning_knob("openai", "o3") == "effort"
        assert get_reasoning_knob("openai", "gpt-4o") == "none"
        assert get_reasoning_knob("anthropic", "claude-sonnet-5-5") == "budget_tokens"
        assert get_reasoning_knob("google", "models/gemini-3.8-flash") == "level"
        assert get_reasoning_knob("openai", "not-a-model") is None
        assert get_reasoning_knob("openai", None) is None


class TestOpenAI:
    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

    def test_effort_translated(self) -> None:
        out = self.plugin.apply_overrides(
            {"messages": []}, {"model": "gpt-5", "reasoning_effort": "high"}
        )
        assert out["reasoning_effort"] == "high"

    def test_o_series_minimal_clamped(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "o3", "reasoning_effort": "minimal"}
        )
        assert out["reasoning_effort"] == "low"

    def test_non_reasoning_model_skips(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "gpt-4o", "reasoning_effort": "high"}
        )
        assert "reasoning_effort" not in out

    def test_budget_tokens_never_forwarded(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "gpt-5", "thinking_budget_tokens": 4096}
        )
        assert "thinking_budget_tokens" not in out

    def test_invalid_value_is_value_free(self) -> None:
        with pytest.raises(ValueError) as exc:
            self.plugin.apply_overrides(
                {}, {"model": "gpt-5", "reasoning_effort": "SECRET-VALUE"}
            )
        assert "SECRET-VALUE" not in str(exc.value)

    def test_raw_passthrough_unchanged(self) -> None:
        # kwargs-supplied value is left alone, even outside the factory levels
        out = self.plugin.apply_overrides(
            {"reasoning_effort": "xhigh"},
            {"model": "gpt-5", "reasoning_effort": "low"},
        )
        assert out["reasoning_effort"] == "xhigh"
        # raw SDK-literal value from config also still reaches the provider
        out = self.plugin.apply_overrides(
            {}, {"model": "gpt-5", "reasoning_effort": "xhigh"}
        )
        assert out["reasoning_effort"] == "xhigh"

    def test_unknown_model_forwarded(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "future-model", "reasoning_effort": "low"}
        )
        assert out["reasoning_effort"] == "low"


class TestAnthropic:
    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_effort_to_budget(self) -> None:
        out = self.plugin.apply_overrides(
            {"messages": [{"role": "user", "content": "hi"}]},
            {
                "model": "claude-sonnet-5-5",
                "max_tokens": 16000,
                "reasoning_effort": "medium",
            },
        )
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 8192}
        assert "reasoning_effort" not in out

    def test_explicit_budget_wins(self) -> None:
        out = self.plugin.apply_overrides(
            {},
            {
                "model": "claude-sonnet-5-5",
                "max_tokens": 8000,
                "reasoning_effort": "high",
                "thinking_budget_tokens": 2048,
            },
        )
        assert out["thinking"]["budget_tokens"] == 2048
        assert out["max_tokens"] == 8000

    def test_budget_must_fit_explicit_max_tokens(self) -> None:
        with pytest.raises(ValueError):
            self.plugin.apply_overrides(
                {},
                {
                    "model": "claude-sonnet-5-5",
                    "max_tokens": 2048,
                    "thinking_budget_tokens": 4096,
                },
            )

    def test_budget_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError):
            self.plugin.apply_overrides(
                {},
                {
                    "model": "claude-sonnet-5-5",
                    "max_tokens": 4000,
                    "thinking_budget_tokens": 10,
                },
            )

    def test_raw_thinking_kwarg_untouched(self) -> None:
        raw = {"type": "enabled", "budget_tokens": 3000}
        out = self.plugin.apply_overrides(
            {"thinking": raw},
            {
                "model": "claude-sonnet-5-5",
                "max_tokens": 4000,
                "reasoning_effort": "high",
            },
        )
        assert out["thinking"] == raw

    def test_no_knob_means_no_thinking(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "claude-sonnet-5-5", "max_tokens": 1000}
        )
        assert "thinking" not in out


class TestGemini:
    def setup_method(self) -> None:
        self.plugin = GeminiPlugin()

    def test_gemini3_level(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "gemini-3.8-flash", "reasoning_effort": "low"}
        )
        assert out["generation_config"]["thinking_config"] == {"thinking_level": "LOW"}
        assert "reasoning_effort" not in out

    def test_gemini25_budget_from_effort_and_explicit(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "gemini-2.5-pro", "reasoning_effort": "high"}
        )
        assert out["generation_config"]["thinking_config"] == {"thinking_budget": 24576}
        out = self.plugin.apply_overrides(
            {}, {"model": "gemini-2.5-pro", "thinking_budget_tokens": 2000}
        )
        assert out["generation_config"]["thinking_config"] == {"thinking_budget": 2000}

    def test_non_thinking_model_skips(self) -> None:
        out = self.plugin.apply_overrides(
            {}, {"model": "gemini-2.0-flash", "reasoning_effort": "high"}
        )
        assert "thinking_config" not in out.get("generation_config", {})

    def test_raw_thinking_config_kwarg_untouched(self) -> None:
        raw = {"thinking_budget": 1}
        out = self.plugin.apply_overrides(
            {"thinking_config": raw},
            {"model": "gemini-3.8-flash", "reasoning_effort": "high"},
        )
        assert out["thinking_config"] == raw
        assert "thinking_config" not in out.get("generation_config", {})
