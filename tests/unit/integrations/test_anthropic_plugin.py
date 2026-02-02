"""Tests for the Anthropic integration plugin."""

import pytest

from traigent.integrations.llms.anthropic_plugin import AnthropicPlugin


class TestAnthropicPlugin:
    """Anthropic plugin behaviours."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_apply_overrides_with_dict_config_and_messages(self) -> None:
        """Plugin should normalise dict configs and avoid mutating caller data."""
        original_messages = [
            {"role": "system", "content": "Follow policies."},
            {"role": "user", "content": "Hello!"},
        ]
        kwargs = {"messages": original_messages}
        config_payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 512,
            "stop": ["END"],
            "tools": [
                {
                    "function": {
                        "name": "call_api",
                        "description": "Call an API",
                        "parameters": {"type": "object"},
                    }
                }
            ],
        }

        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # Model pulled from config dict and system promoted from messages
        assert overridden["model"] == "claude-3-haiku-20240307"
        assert overridden["system"] == "Follow policies."
        assert overridden["stop_sequences"] == ["END"]

        # Tools converted to Anthropic format
        assert overridden["tools"][0]["name"] == "call_api"
        assert overridden["tools"][0]["input_schema"] == {"type": "object"}

        # Caller-provided messages remain unchanged while overrides receive trimmed list
        assert len(original_messages) == 2
        assert overridden["messages"][0]["role"] == "user"

    def test_custom_system_param_preserved(self) -> None:
        """Custom system string should take priority over inferred system message."""
        messages = [
            {"role": "system", "content": "Do not override this."},
            {"role": "user", "content": "Ping"},
        ]
        kwargs = {"messages": messages}
        config_payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "custom_params": {"system": "Explicit system"},
        }

        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        assert overridden["system"] == "Explicit system"
        assert len(overridden["messages"]) == 1
        assert len(messages) == 2  # Original list unchanged


class TestAnthropicPluginMetadata:
    """Tests for Anthropic plugin metadata and configuration."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_metadata_name(self) -> None:
        """Plugin metadata should have correct name."""
        assert self.plugin.metadata.name == "anthropic"

    def test_extra_mappings(self) -> None:
        """Plugin should return Anthropic-specific parameter mappings."""
        mappings = self.plugin._get_extra_mappings()
        assert "metadata" in mappings
        assert "anthropic_api_key" in mappings
        assert "anthropic_api_url" in mappings

    def test_provider_specific_rules(self) -> None:
        """Plugin should return validation rules."""
        rules = self.plugin._get_provider_specific_rules()
        assert "model" in rules
        assert "max_tokens" in rules
        assert "temperature" in rules
        # Anthropic temperature is 0-1 (stricter)
        assert rules["temperature"].max_value == pytest.approx(1.0)

    def test_target_classes(self) -> None:
        """Plugin should return list of Anthropic classes to override."""
        classes = self.plugin.get_target_classes()
        assert "anthropic.Anthropic" in classes
        assert "anthropic.AsyncAnthropic" in classes

    def test_target_methods(self) -> None:
        """Plugin should return method mapping for Anthropic classes."""
        methods = self.plugin.get_target_methods()
        assert "anthropic.Anthropic" in methods
        assert "messages.create" in methods["anthropic.Anthropic"]


class TestAnthropicPluginValidation:
    """Tests for Anthropic plugin validation methods."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_validate_model_non_string(self) -> None:
        """Model validation should reject non-string values."""
        errors = self.plugin._validate_model("model", 123)
        assert len(errors) > 0
        assert "string" in errors[0]

    def test_validate_model_empty(self) -> None:
        """Model validation should reject empty strings."""
        errors = self.plugin._validate_model("model", "")
        assert len(errors) > 0
        assert "empty" in errors[0]

    def test_validate_model_valid_string(self) -> None:
        """Model validation should accept valid model names."""
        errors = self.plugin._validate_model("model", "claude-3-opus-20240229")
        # Should not return blocking errors (may warn but not block)
        assert len(errors) == 0


class TestAnthropicPluginMaxTokensDefault:
    """Tests for max_tokens default behavior."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_max_tokens_preserved_from_config(self) -> None:
        """max_tokens from config should be preserved."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {"model": "claude-3-opus-20240229", "max_tokens": 2000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert result["max_tokens"] == 2000

    def test_max_tokens_preserved_from_kwargs(self) -> None:
        """max_tokens from kwargs should be preserved when config doesn't override."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3000}
        config = {"model": "claude-3-opus-20240229", "max_tokens": 2000}
        result = self.plugin.apply_overrides(kwargs, config)

        # kwargs value preserved (3000) but merged with config
        assert result["max_tokens"] == 3000


class TestAnthropicPluginToolsConversion:
    """Tests for tool/function format conversion."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_anthropic_native_tools_preserved(self) -> None:
        """Anthropic-native tool format should be preserved."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }
        config = {"model": "claude-3-opus-20240229", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        # Should preserve Anthropic-native format
        assert result["tools"][0]["name"] == "get_weather"

    def test_tools_with_missing_function_name_skipped(self) -> None:
        """Tools with missing function name should be skipped."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        # Missing 'name'
                        "description": "Get weather info",
                    },
                }
            ],
        }
        config = {"model": "claude-3-opus-20240229", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        # Should have empty tools or tool without name
        tools = result.get("tools", [])
        assert len(tools) == 0 or not any(t.get("name") for t in tools)

    def test_tools_with_openai_format_missing_function(self) -> None:
        """OpenAI-format tools without function key should be converted as-is."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {"name": "direct_tool", "description": "A direct tool"},
            ],
        }
        config = {"model": "claude-3-opus-20240229", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        # Direct Anthropic format should be preserved
        assert result["tools"][0]["name"] == "direct_tool"


class TestAnthropicPluginReasoningOverrides:
    """Tests for extended thinking parameter translation."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_extended_thinking_true(self) -> None:
        """extended_thinking=True should create thinking object with enabled type."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "extended_thinking": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert "budget_tokens" in result["thinking"]  # Default budget
        assert "extended_thinking" not in result  # Should be cleaned up

    def test_extended_thinking_false(self) -> None:
        """extended_thinking=False should not add thinking object."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "extended_thinking": False,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" not in result
        assert "extended_thinking" not in result

    def test_reasoning_mode_deep(self) -> None:
        """reasoning_mode='deep' should enable thinking."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "reasoning_mode": "deep",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert "reasoning_mode" not in result

    def test_reasoning_mode_none(self) -> None:
        """reasoning_mode='none' should disable thinking."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "reasoning_mode": "none",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" not in result
        assert "reasoning_mode" not in result

    def test_thinking_budget_tokens(self) -> None:
        """thinking_budget_tokens should be set in thinking object."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "extended_thinking": True,
            "thinking_budget_tokens": 5000,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert result["thinking"]["budget_tokens"] == 5000
        assert "thinking_budget_tokens" not in result

    def test_thinking_budget_enforces_minimum(self) -> None:
        """Budget should be at least 1024 tokens per Anthropic API."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "extended_thinking": True,
            "thinking_budget_tokens": 500,  # Below minimum
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert result["thinking"]["budget_tokens"] == 1024  # Enforced minimum

    def test_reasoning_budget_to_thinking(self) -> None:
        """reasoning_budget should translate to thinking.budget_tokens."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "reasoning_budget": 10000,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" in result
        assert result["thinking"]["budget_tokens"] == 10000
        assert result["thinking"]["type"] == "enabled"  # Auto-enabled when budget is set
        assert "reasoning_budget" not in result

    def test_reasoning_budget_zero_enforces_minimum(self) -> None:
        """reasoning_budget=0 should still get minimum 1024 budget."""
        kwargs = {
            "model": "claude-sonnet-4-5",
            "reasoning_budget": 0,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-sonnet-4-5", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" in result
        assert result["thinking"]["budget_tokens"] == 1024

    def test_non_reasoning_model_cleans_params(self) -> None:
        """Non-reasoning models should have all reasoning params removed."""
        kwargs = {
            "model": "claude-3-opus-20240229",
            "extended_thinking": True,
            "reasoning_mode": "deep",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        config = {"model": "claude-3-opus-20240229", "max_tokens": 1000}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "thinking" not in result
        assert "extended_thinking" not in result
        assert "reasoning_mode" not in result


class TestAnthropicPluginCustomParamsEdgeCases:
    """Tests for edge cases in custom_params handling."""

    def setup_method(self) -> None:
        self.plugin = AnthropicPlugin()

    def test_custom_params_none(self) -> None:
        """None custom_params should not cause errors."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {"model": "claude-3-opus-20240229", "max_tokens": 1000, "custom_params": None}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "messages" in result

    def test_empty_custom_params(self) -> None:
        """Empty dict custom_params should not cause issues."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "custom_params": {},
        }
        result = self.plugin.apply_overrides(kwargs, config)

        assert "messages" in result

    def test_custom_params_with_system(self) -> None:
        """Custom_params with system should set system parameter."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "custom_params": {"system": "Be helpful"},
        }
        result = self.plugin.apply_overrides(kwargs, config)

        assert result["system"] == "Be helpful"
