"""Tests for the PydanticAI integration plugin."""

from __future__ import annotations

from traigent.integrations.pydantic_ai.plugin import PydanticAIPlugin
from traigent.integrations.utils import Framework


class TestPydanticAIPlugin:
    """PydanticAI plugin behaviours."""

    def setup_method(self) -> None:
        self.plugin = PydanticAIPlugin()

    def test_framework(self) -> None:
        assert self.plugin.FRAMEWORK == Framework.PYDANTIC_AI

    def test_metadata_name(self) -> None:
        meta = self.plugin.metadata
        assert meta.name == "pydantic_ai"

    def test_metadata_supported_packages(self) -> None:
        meta = self.plugin.metadata
        assert "pydantic_ai" in meta.supported_packages

    def test_metadata_requires_packages(self) -> None:
        meta = self.plugin.metadata
        assert any("pydantic-ai" in p for p in meta.requires_packages)

    def test_supported_canonical_params(self) -> None:
        params = self.plugin._get_supported_canonical_params()
        assert params == {"max_tokens", "temperature", "top_p"}

    def test_extra_mappings(self) -> None:
        extra = self.plugin._get_extra_mappings()
        assert "parallel_tool_calls" in extra

    def test_provider_specific_rules(self) -> None:
        rules = self.plugin._get_provider_specific_rules()
        assert "temperature" in rules
        assert rules["temperature"].min_value == 0.0
        assert rules["temperature"].max_value == 2.0
        assert "max_tokens" in rules
        assert rules["max_tokens"].min_value == 1
        assert rules["max_tokens"].max_value == 128000
        assert "top_p" in rules

    def test_target_classes(self) -> None:
        classes = self.plugin.get_target_classes()
        assert "pydantic_ai.Agent" in classes

    def test_target_methods(self) -> None:
        methods = self.plugin.get_target_methods()
        agent_methods = methods["pydantic_ai.Agent"]
        assert "run" in agent_methods
        assert "run_sync" in agent_methods
        assert "run_stream" in agent_methods
        assert "run_stream_sync" in agent_methods

    def test_apply_overrides_with_none_config(self) -> None:
        kwargs = {"prompt": "Hello"}
        result = self.plugin.apply_overrides(kwargs, None)
        assert result == {"prompt": "Hello"}

    def test_apply_overrides_wraps_into_model_settings(self) -> None:
        """Plugin should inject params INTO model_settings, not as top-level kwargs."""
        config_payload = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.9,
        }
        kwargs = {"prompt": "Hello"}
        result = self.plugin.apply_overrides(kwargs, config_payload)

        assert "model_settings" in result
        ms = result["model_settings"]
        assert ms["temperature"] == 0.7
        assert ms["max_tokens"] == 500
        assert ms["top_p"] == 0.9
        # Original kwarg preserved
        assert result.get("prompt") == "Hello"
        # Params must NOT leak as top-level kwargs
        assert "temperature" not in result
        assert "max_tokens" not in result
        assert "top_p" not in result

    def test_apply_overrides_user_settings_take_precedence(self) -> None:
        """User-provided model_settings should override Traigent config."""
        config_payload = {
            "temperature": 0.7,
            "max_tokens": 500,
        }
        kwargs = {
            "prompt": "Hello",
            "model_settings": {"temperature": 0.1, "custom_key": "keep"},
        }
        result = self.plugin.apply_overrides(kwargs, config_payload)

        ms = result["model_settings"]
        # User value takes precedence
        assert ms["temperature"] == 0.1
        # Traigent config fills in missing params
        assert ms["max_tokens"] == 500
        # User custom key preserved
        assert ms["custom_key"] == "keep"

    def test_apply_overrides_disabled_plugin_is_noop(self) -> None:
        """Disabled plugin should return kwargs unchanged."""
        self.plugin.disable()
        config_payload = {"temperature": 0.7, "max_tokens": 500}
        kwargs = {"prompt": "Hello"}
        result = self.plugin.apply_overrides(kwargs, config_payload)
        assert result == {"prompt": "Hello"}
        assert "model_settings" not in result

    def test_apply_overrides_with_dict_config(self) -> None:
        """Plugin should handle raw dict configs without errors."""
        config_payload = {"temperature": 0.5}
        kwargs = {}
        result = self.plugin.apply_overrides(kwargs, config_payload)
        assert "model_settings" in result
        assert result["model_settings"]["temperature"] == 0.5

    def test_apply_overrides_empty_config_returns_copy(self) -> None:
        """Config with no recognized params should return kwargs unchanged."""
        config_payload = {"unknown_param": 42}
        kwargs = {"prompt": "Hello"}
        result = self.plugin.apply_overrides(kwargs, config_payload)
        assert result == {"prompt": "Hello"}
        assert "model_settings" not in result

    def test_apply_overrides_non_dict_model_settings(self) -> None:
        """Non-dict existing model_settings should be treated as empty."""
        config_payload = {"temperature": 0.7}
        kwargs = {"prompt": "Hello", "model_settings": "not-a-dict"}
        result = self.plugin.apply_overrides(kwargs, config_payload)
        assert isinstance(result["model_settings"], dict)
        assert result["model_settings"]["temperature"] == 0.7
