"""Tests for the LangChain integration plugin."""

from traigent.integrations.llms.langchain_plugin import LangChainPlugin


class TestLangChainPlugin:
    """LangChain plugin behaviours."""

    def setup_method(self) -> None:
        self.plugin = LangChainPlugin()

    def test_apply_overrides_with_dict_config(self) -> None:
        """Plugin should handle raw dict payloads without attribute errors."""
        config_payload = {
            "model": "gpt-4",
            "stream": True,
            "search_type": "mmr",
            "lambda_mult": 0.35,
            "k": 4,
        }

        overridden = self.plugin.apply_overrides({}, config_payload)

        assert overridden["model_name"] == "gpt-4"
        assert "model" not in overridden
        assert overridden["streaming"] is True
        assert overridden["k"] == 4
        assert overridden["search_kwargs"]["lambda_mult"] == 0.35
