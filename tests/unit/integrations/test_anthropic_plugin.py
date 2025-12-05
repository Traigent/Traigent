"""Tests for the Anthropic integration plugin."""

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
