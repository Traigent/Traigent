"""Tests for the Mistral AI integration plugin."""

import pytest

from traigent.integrations.llms.mistral_plugin import MistralPlugin
from traigent.integrations.utils import Framework


class TestMistralPlugin:
    """Mistral plugin behaviors."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_apply_overrides_with_dict_config(self) -> None:
        """Plugin should handle raw dict payloads without attribute errors."""
        config_payload = {
            "model": "mistral-large-latest",
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model"] == "mistral-large-latest"
        assert overridden["stream"] is True
        assert overridden["temperature"] == pytest.approx(0.7)
        assert overridden["max_tokens"] == 1000


class TestMistralParameterMappings:
    """Test parameter mapping via ParameterNormalizer (declarative approach)."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_model_preserved(self) -> None:
        """Test that model parameter is preserved."""
        config_payload = {"model": "mistral-small-latest"}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model"] == "mistral-small-latest"

    def test_temperature_preserved(self) -> None:
        """Test that temperature is preserved."""
        config_payload = {"model": "mistral-small-latest", "temperature": 0.5}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["temperature"] == pytest.approx(0.5)

    def test_top_p_preserved(self) -> None:
        """Test that top_p is preserved."""
        config_payload = {"model": "mistral-small-latest", "top_p": 0.9}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["top_p"] == pytest.approx(0.9)

    def test_max_tokens_preserved(self) -> None:
        """Test that max_tokens is preserved."""
        config_payload = {"model": "mistral-small-latest", "max_tokens": 2000}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["max_tokens"] == 2000

    def test_stream_preserved(self) -> None:
        """Test that stream parameter is preserved."""
        config_payload = {"model": "mistral-small-latest", "stream": True}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["stream"] is True

    def test_combined_parameters(self) -> None:
        """Test multiple parameters work together."""
        config_payload = {
            "model": "mistral-large-2411",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "stream": False,
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model"] == "mistral-large-2411"
        assert overridden["temperature"] == pytest.approx(0.7)
        assert overridden["max_tokens"] == 4096
        assert overridden["top_p"] == pytest.approx(0.95)
        assert overridden["stream"] is False

    def test_user_provided_kwarg_not_overwritten(self) -> None:
        """Test that user-provided kwargs take precedence over config mappings."""
        kwargs = {"model": "user-specified-model"}
        config_payload = {"model": "config-model"}

        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # User's value should win
        assert overridden["model"] == "user-specified-model"


class TestMistralSpecificParameters:
    """Test Mistral-specific parameter handling."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_safe_prompt_parameter(self) -> None:
        """Test that safe_prompt parameter is handled."""
        config_payload = {"model": "mistral-small-latest", "safe_prompt": True}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["safe_prompt"] is True

    def test_random_seed_parameter(self) -> None:
        """Test that random_seed parameter is handled."""
        config_payload = {"model": "mistral-small-latest", "random_seed": 42}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["random_seed"] == 42

    def test_seed_to_random_seed_mapping(self) -> None:
        """Test that 'seed' alias maps to 'random_seed'."""
        config_payload = {"model": "mistral-small-latest", "seed": 123}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["random_seed"] == 123

    def test_stop_sequences_to_stop_mapping(self) -> None:
        """Test that stop_sequences maps to stop."""
        config_payload = {
            "model": "mistral-small-latest",
            "stop_sequences": ["END", "STOP"],
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["stop"] == ["END", "STOP"]

    def test_n_parameter(self) -> None:
        """Test that n parameter for multiple completions is handled."""
        config_payload = {"model": "mistral-small-latest", "n": 3}
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["n"] == 3

    def test_frequency_penalty(self) -> None:
        """Test frequency_penalty parameter."""
        config_payload = {
            "model": "mistral-small-latest",
            "frequency_penalty": 0.5,
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["frequency_penalty"] == pytest.approx(0.5)

    def test_presence_penalty(self) -> None:
        """Test presence_penalty parameter."""
        config_payload = {
            "model": "mistral-small-latest",
            "presence_penalty": 0.3,
        }
        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["presence_penalty"] == pytest.approx(0.3)


class TestMistralExtraMappings:
    """Test Mistral-specific extra mappings from _get_extra_mappings()."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_extra_mappings_include_mistral_specific(self) -> None:
        """Verify Mistral-specific params are in extra mappings."""
        mappings = self.plugin._get_extra_mappings()

        assert "random_seed" in mappings
        assert "safe_prompt" in mappings
        assert "parallel_tool_calls" in mappings
        assert "prompt_mode" in mappings
        assert "tool_choice" in mappings
        assert "tools" in mappings

    def test_api_key_mapping(self) -> None:
        """Verify API key alias is in mappings."""
        mappings = self.plugin._get_extra_mappings()

        assert "mistral_api_key" in mappings
        assert mappings["mistral_api_key"] == "api_key"


class TestMistralValidationRules:
    """Test Mistral-specific validation rules."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_temperature_validation_range(self) -> None:
        """Test temperature parameter has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "temperature" in rules
        assert rules["temperature"].min_value == 0.0
        assert rules["temperature"].max_value == 1.0

    def test_top_p_validation_range(self) -> None:
        """Test top_p parameter has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "top_p" in rules
        assert rules["top_p"].min_value == 0.0
        assert rules["top_p"].max_value == 1.0

    def test_max_tokens_validation_range(self) -> None:
        """Test max_tokens parameter has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "max_tokens" in rules
        assert rules["max_tokens"].min_value == 1
        assert rules["max_tokens"].max_value == 128000

    def test_stream_allowed_values(self) -> None:
        """Test stream parameter accepts boolean."""
        rules = self.plugin._get_provider_specific_rules()
        assert "stream" in rules
        assert rules["stream"].allowed_values == [True, False]

    def test_safe_prompt_allowed_values(self) -> None:
        """Test safe_prompt parameter accepts boolean."""
        rules = self.plugin._get_provider_specific_rules()
        assert "safe_prompt" in rules
        assert rules["safe_prompt"].allowed_values == [True, False]

    def test_n_validation_range(self) -> None:
        """Test n parameter has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "n" in rules
        assert rules["n"].min_value == 1
        assert rules["n"].max_value == 16

    def test_frequency_penalty_validation_range(self) -> None:
        """Test frequency_penalty has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "frequency_penalty" in rules
        assert rules["frequency_penalty"].min_value == -2.0
        assert rules["frequency_penalty"].max_value == 2.0

    def test_presence_penalty_validation_range(self) -> None:
        """Test presence_penalty has valid range."""
        rules = self.plugin._get_provider_specific_rules()
        assert "presence_penalty" in rules
        assert rules["presence_penalty"].min_value == -2.0
        assert rules["presence_penalty"].max_value == 2.0


class TestMistralPluginMetadata:
    """Test Mistral plugin metadata and framework."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_framework_is_mistral(self) -> None:
        """Test plugin identifies as Mistral framework."""
        assert self.plugin.FRAMEWORK == Framework.MISTRAL

    def test_metadata_name(self) -> None:
        """Test plugin metadata has correct name."""
        assert self.plugin.metadata.name == "mistral"

    def test_supported_packages(self) -> None:
        """Test plugin lists supported Mistral packages."""
        packages = self.plugin.metadata.supported_packages
        assert "mistralai" in packages

    def test_target_classes_include_mistral_client(self) -> None:
        """Test target classes include core Mistral classes."""
        classes = self.plugin.get_target_classes()
        assert "mistralai.Mistral" in classes

    def test_target_methods_include_chat_complete(self) -> None:
        """Test target methods include chat completion methods."""
        methods = self.plugin.get_target_methods()
        mistral_methods = methods.get("mistralai.Mistral", [])
        assert "chat.complete" in mistral_methods
        assert "chat.complete_async" in mistral_methods
        assert "chat.stream" in mistral_methods


class TestMistralMessageFormatting:
    """Test Mistral message formatting."""

    def setup_method(self) -> None:
        self.plugin = MistralPlugin()

    def test_string_message_wrapped(self) -> None:
        """Test that string messages are wrapped as user messages."""
        kwargs = {"messages": "Hello, Mistral!"}
        config_payload = {"model": "mistral-small-latest"}
        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        assert overridden["messages"] == [
            {"role": "user", "content": "Hello, Mistral!"}
        ]

    def test_dict_messages_preserved(self) -> None:
        """Test that properly formatted messages are preserved."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        kwargs = {"messages": messages}
        config_payload = {"model": "mistral-small-latest"}
        overridden = self.plugin.apply_overrides(kwargs, config_payload)

        # Should be unchanged
        assert overridden["messages"] == messages
