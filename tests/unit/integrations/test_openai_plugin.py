"""Tests for the OpenAI integration plugin."""

import pytest

from traigent.integrations.llms.openai_plugin import OpenAIPlugin


class TestOpenAIPlugin:
    """OpenAI plugin behaviours."""

    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

    def test_apply_overrides_returns_copy_when_config_missing(self) -> None:
        """Plugin should bail out safely if no configuration is provided."""
        original_kwargs = {"messages": [{"role": "user", "content": "ping"}]}

        overridden = self.plugin.apply_overrides(original_kwargs, None)

        assert overridden == original_kwargs
        assert overridden is not original_kwargs

    def test_apply_overrides_with_dict_config_uses_custom_params(self) -> None:
        """Plugin should normalise dict configs and honour custom params."""
        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"messages": messages}
        config_payload = {
            "model": "gpt-4",
            "custom_params": {"system": "Follow safety rules."},
        }

        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # Model pulled from config dict
        assert overridden["model"] == "gpt-4"

        # System message injected based on custom params
        assert overridden["messages"][0]["role"] == "system"
        assert overridden["messages"][0]["content"] == "Follow safety rules."

        # Original messages remain untouched
        assert messages == [{"role": "user", "content": "Hello"}]


class TestOpenAIPluginMetadata:
    """Tests for OpenAI plugin metadata and configuration."""

    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

    def test_metadata_name(self) -> None:
        """Plugin metadata should have correct name."""
        assert self.plugin.metadata.name == "openai"

    def test_extra_mappings(self) -> None:
        """Plugin should return OpenAI-specific parameter mappings."""
        mappings = self.plugin._get_extra_mappings()
        assert "functions" in mappings
        assert "function_call" in mappings
        assert "response_format" in mappings

    def test_provider_specific_rules(self) -> None:
        """Plugin should return validation rules."""
        rules = self.plugin._get_provider_specific_rules()
        assert "model" in rules
        assert "max_tokens" in rules
        assert "frequency_penalty" in rules

    def test_target_classes(self) -> None:
        """Plugin should return list of OpenAI classes to override."""
        classes = self.plugin.get_target_classes()
        assert "openai.OpenAI" in classes
        assert "openai.AsyncOpenAI" in classes

    def test_target_methods(self) -> None:
        """Plugin should return method mapping for OpenAI classes."""
        methods = self.plugin.get_target_methods()
        assert "openai.OpenAI" in methods
        assert "chat.completions.create" in methods["openai.OpenAI"]


class TestOpenAIPluginValidation:
    """Tests for OpenAI plugin validation methods."""

    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

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
        errors = self.plugin._validate_model("model", "gpt-4")
        # Should not return blocking errors (may warn but not block)
        assert len(errors) == 0

    def test_validate_seed_valid(self) -> None:
        """Seed validation should accept valid integers."""
        errors = self.plugin._validate_seed("seed", 42)
        assert len(errors) == 0

    def test_validate_seed_non_integer(self) -> None:
        """Seed validation should reject non-integers."""
        errors = self.plugin._validate_seed("seed", "not_int")
        assert len(errors) > 0
        assert "integer" in errors[0]

    def test_validate_seed_too_large(self) -> None:
        """Seed validation should reject values exceeding 2^32 - 1."""
        errors = self.plugin._validate_seed("seed", 2**33)
        assert len(errors) > 0
        assert "exceeds" in errors[0]


class TestOpenAIPluginFunctionCalling:
    """Tests for function/tool calling conversion."""

    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

    def test_functions_to_tools_conversion(self) -> None:
        """Legacy functions should be converted to tools format."""
        kwargs = {
            "model": "gpt-4",
            "functions": [{"name": "test", "parameters": {}}],
            "function_call": "auto",
        }
        result = self.plugin.apply_overrides(kwargs, {"model": "gpt-4"})

        assert "tools" in result
        assert result["tool_choice"] == "auto"
        assert "functions" not in result

    def test_function_call_none(self) -> None:
        """function_call='none' should convert to tool_choice='none'."""
        kwargs = {
            "model": "gpt-4",
            "functions": [{"name": "test", "parameters": {}}],
            "function_call": "none",
        }
        result = self.plugin.apply_overrides(kwargs, {"model": "gpt-4"})

        assert result["tool_choice"] == "none"

    def test_function_call_with_name(self) -> None:
        """function_call with specific name should convert to tool_choice dict."""
        kwargs = {
            "model": "gpt-4",
            "functions": [{"name": "test", "parameters": {}}],
            "function_call": {"name": "test"},
        }
        result = self.plugin.apply_overrides(kwargs, {"model": "gpt-4"})

        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "test"


class TestOpenAIPluginReasoningOverrides:
    """Tests for reasoning parameter translation."""

    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

    def test_reasoning_mode_deep_to_high(self) -> None:
        """reasoning_mode='deep' should translate to reasoning_effort='high'."""
        kwargs = {"model": "o3", "reasoning_mode": "deep"}
        result = self.plugin.apply_overrides(kwargs, {"model": "o3"})

        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "high"
        assert "reasoning_mode" not in result

    def test_reasoning_mode_standard_to_medium(self) -> None:
        """reasoning_mode='standard' should translate to reasoning_effort='medium'."""
        kwargs = {"model": "gpt-5", "reasoning_mode": "standard"}
        result = self.plugin.apply_overrides(kwargs, {"model": "gpt-5"})

        assert result["reasoning_effort"] == "medium"

    def test_reasoning_mode_none_to_minimal(self) -> None:
        """reasoning_mode='none' should translate to reasoning_effort='minimal'."""
        kwargs = {"model": "gpt-5", "reasoning_mode": "none"}
        result = self.plugin.apply_overrides(kwargs, {"model": "gpt-5"})

        assert result["reasoning_effort"] == "minimal"

    def test_reasoning_budget_to_max_completion_tokens(self) -> None:
        """reasoning_budget should translate to max_completion_tokens."""
        kwargs = {"model": "o3", "reasoning_budget": 5000}
        result = self.plugin.apply_overrides(kwargs, {"model": "o3"})

        assert result.get("max_completion_tokens") == 5000
        assert "reasoning_budget" not in result

    def test_non_reasoning_model_cleans_params(self) -> None:
        """Non-reasoning models should have all reasoning params removed."""
        kwargs = {
            "model": "gpt-4o",
            "reasoning_mode": "deep",
            "reasoning_effort": "high",
        }
        result = self.plugin.apply_overrides(kwargs, {"model": "gpt-4o"})

        assert "reasoning_mode" not in result
        assert "reasoning_effort" not in result

    def test_max_tokens_removed_for_reasoning_model(self) -> None:
        """max_tokens should be removed when max_completion_tokens is set."""
        kwargs = {"model": "o3", "max_tokens": 1000, "max_completion_tokens": 2000}
        result = self.plugin.apply_overrides(kwargs, {"model": "o3"})

        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 2000

    def test_reasoning_effort_xhigh_fallback(self) -> None:
        """xhigh should fall back to high for o3 (xhigh not available)."""
        kwargs = {"model": "o3", "reasoning_effort": "xhigh"}
        result = self.plugin.apply_overrides(kwargs, {"model": "o3"})

        assert result["reasoning_effort"] == "high"

    def test_reasoning_effort_minimal_fallback(self) -> None:
        """minimal should fall back to low for o3 (minimal not available)."""
        kwargs = {"model": "o3", "reasoning_effort": "minimal"}
        result = self.plugin.apply_overrides(kwargs, {"model": "o3"})

        assert result["reasoning_effort"] == "low"

    def test_reasoning_effort_unknown_fallback(self) -> None:
        """Unknown effort levels should fall back to medium."""
        kwargs = {"model": "o3", "reasoning_effort": "unknown_level"}
        result = self.plugin.apply_overrides(kwargs, {"model": "o3"})

        assert result["reasoning_effort"] == "medium"


class TestOpenAIPluginSystemMessages:
    """Tests for system message handling."""

    def setup_method(self) -> None:
        self.plugin = OpenAIPlugin()

    def test_system_message_prepended(self) -> None:
        """System message from custom_params should be prepended."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {"model": "gpt-4", "custom_params": {"system": "Be helpful."}}
        result = self.plugin.apply_overrides(kwargs, config)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be helpful."
        assert result["messages"][1]["role"] == "user"

    def test_existing_system_message_not_duplicated(self) -> None:
        """If messages already have system role, don't prepend another."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "Existing"},
                {"role": "user", "content": "Hi"},
            ]
        }
        config = {"model": "gpt-4", "custom_params": {"system": "New system."}}
        result = self.plugin.apply_overrides(kwargs, config)

        # Should keep existing system message
        assert result["messages"][0]["content"] == "Existing"

    def test_none_custom_params(self) -> None:
        """None custom_params should not cause errors."""
        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        config = {"model": "gpt-4", "custom_params": None}
        result = self.plugin.apply_overrides(kwargs, config)

        assert "messages" in result
