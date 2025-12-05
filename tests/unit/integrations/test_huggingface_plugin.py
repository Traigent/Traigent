import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations.llms.huggingface_plugin import HuggingFacePlugin


@pytest.mark.unit
def test_huggingface_plugin_mappings():
    plugin = HuggingFacePlugin()
    mappings = plugin.get_parameter_mappings()
    assert mappings["model"] == "model"
    assert mappings["max_tokens"] == "max_new_tokens"


@pytest.mark.unit
def test_huggingface_plugin_overrides():
    plugin = HuggingFacePlugin()
    config = TraigentConfig(
        model="meta-llama/Llama-2-7b-chat-hf", temperature=0.8, max_tokens=100
    )

    kwargs = {"messages": []}
    overridden = plugin.apply_overrides(kwargs, config)

    assert overridden["model"] == "meta-llama/Llama-2-7b-chat-hf"
    assert overridden["temperature"] == 0.8
    assert overridden["max_new_tokens"] == 100
