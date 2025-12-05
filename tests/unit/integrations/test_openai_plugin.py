"""Tests for the OpenAI integration plugin."""

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
